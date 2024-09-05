import logging
from dataclasses import dataclass
from functools import reduce
from typing import List, Optional, Tuple, Union
from warnings import warn

import numpy as np
import svox2.csrc as _C
import torch
from torch import autograd, nn
from tqdm import tqdm

from plenoxels.svox2 import utils
from plenoxels.svox2.defs import *

# flake8: noqa


@dataclass
class RenderOptions:
    """
    Rendering options, see comments
    available:
    :param backend: str, renderer backend
    :param background_brightness: float
    :param step_size: float, step size for rendering
    :param sigma_thresh: float
    :param stop_thresh: float
    """

    backend: str = "cuvol"  # One of cuvol, svox1, nvol

    background_brightness: float = 1.0  # [0, 1], the background color black-white

    step_size: float = 0.5  # Step size, in normalized voxels (not used for svox1)
    #  (i.e. 1 = 1 voxel width, different from svox where 1 = grid width!)

    sigma_thresh: float = 1e-10  # Voxels with sigmas < this are ignored, in [0, 1]
    #  make this higher for fast rendering

    stop_thresh: float = (
        1e-7  # Stops rendering if the remaining light intensity/termination, in [0, 1]
    )
    #  probability is <= this much (forward only)
    #  make this higher for fast rendering

    last_sample_opaque: bool = False  # Make the last sample opaque (for forward-facing)

    near_clip: float = 0.0
    use_spheric_clip: bool = False

    random_sigma_std: float = 1.0  # Noise to add to sigma (only if randomize=True)
    random_sigma_std_background: float = 1.0  # Noise to add to sigma
    # (for the BG model; only if randomize=True)

    def _to_cpp(self, randomize: bool = False):
        """
        Generate object to pass to C++
        """
        opt = _C.RenderOptions()
        opt.empty_space_brightness = self.background_brightness
        opt.background_brightness = self.background_brightness
        opt.step_size = self.step_size
        opt.sigma_thresh = self.sigma_thresh
        opt.stop_thresh = self.stop_thresh
        opt.near_clip = self.near_clip
        opt.use_spheric_clip = self.use_spheric_clip
        opt.last_sample_opaque = self.last_sample_opaque

        return opt


@dataclass
class Rays:
    origins: torch.Tensor
    dirs: torch.Tensor

    def _to_cpp(self):
        """
        Generate object to pass to C++
        """
        spec = _C.RaysSpec()
        spec.origins = self.origins
        spec.dirs = self.dirs
        return spec

    def __getitem__(self, key):
        return Rays(self.origins[key], self.dirs[key])

    @property
    def is_cuda(self) -> bool:
        return self.origins.is_cuda and self.dirs.is_cuda


@dataclass
class Camera:
    c2w: torch.Tensor  # OpenCV
    fx: float = 1111.11
    fy: Optional[float] = None
    cx: Optional[float] = None
    cy: Optional[float] = None
    width: int = 800
    height: int = 800

    ndc_coeffs: Union[Tuple[float, float], List[float]] = (-1.0, -1.0)

    @property
    def fx_val(self):
        return self.fx

    @property
    def fy_val(self):
        return self.fx if self.fy is None else self.fy

    @property
    def cx_val(self):
        return self.width * 0.5 if self.cx is None else self.cx

    @property
    def cy_val(self):
        return self.height * 0.5 if self.cy is None else self.cy

    @property
    def using_ndc(self):
        return self.ndc_coeffs[0] > 0.0

    def _to_cpp(self):
        """
        Generate object to pass to C++
        """
        spec = _C.CameraSpec()
        spec.c2w = self.c2w
        spec.fx = self.fx_val
        spec.fy = self.fy_val
        spec.cx = self.cx_val
        spec.cy = self.cy_val
        spec.width = self.width
        spec.height = self.height
        return spec

    @property
    def is_cuda(self) -> bool:
        return self.c2w.is_cuda

    def gen_rays(self) -> Rays:
        """
        Generate the rays for this camera
        :return: (origins (H*W, 3), dirs (H*W, 3))
        """
        origins = (
            self.c2w[None, :3, 3].expand(self.height * self.width, -1).contiguous()
        )
        yy, xx = torch.meshgrid(
            torch.arange(self.height, dtype=torch.float64, device=self.c2w.device)
            + 0.5,
            torch.arange(self.width, dtype=torch.float64, device=self.c2w.device) + 0.5,
        )
        xx = (xx - self.cx_val) / self.fx_val
        yy = (yy - self.cy_val) / self.fy_val
        zz = torch.ones_like(xx)
        dirs = torch.stack((xx, yy, zz), dim=-1)  # OpenCV
        del xx, yy, zz
        dirs /= torch.norm(dirs, dim=-1, keepdim=True)
        dirs = dirs.reshape(-1, 3, 1)
        dirs = (self.c2w[None, :3, :3].double() @ dirs)[..., 0]
        dirs = dirs.reshape(-1, 3).float()

        if self.ndc_coeffs[0] > 0.0:
            origins, dirs = utils.convert_to_ndc(origins, dirs, self.ndc_coeffs)
            dirs /= torch.norm(dirs, dim=-1, keepdim=True)
        return Rays(origins, dirs)


# BEGIN Differentiable CUDA functions with custom gradient
class _SampleGridAutogradFunction(autograd.Function):
    @staticmethod
    def forward(
        ctx,
        data_density: torch.Tensor,
        data_sh: torch.Tensor,
        data_temperature: torch.Tensor,
        grid,
        points: torch.Tensor,
        want_colors: bool,
        want_temperatureerature: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:

        assert not points.requires_grad, "Point gradient not supported"

        out_density, out_sh, out_temperature = _C.sample_grid(
            grid, points, want_colors, want_temperatureerature
        )

        ctx.save_for_backward(points)
        ctx.grid = grid
        ctx.want_colors = want_colors
        ctx.want_temperatureerature = want_temperatureerature

        if want_temperatureerature:
            return out_density, out_sh, out_temperature
        return out_density, out_sh, None

    @staticmethod
    def backward(ctx, grad_out_density, grad_out_sh, grad_out_temperature):
        (points,) = ctx.saved_tensors
        grad_density_grid = torch.zeros_like(ctx.grid.density_data.data)
        grad_sh_grid = torch.zeros_like(ctx.grid.sh_data.data)
        grad_temperature_grid = torch.zeros_like(ctx.grid.temperature_data.data)

        _C.sample_grid_backward(
            ctx.grid,
            points,
            grad_out_density.contiguous(),
            grad_out_sh.contiguous(),
            grad_density_grid,
            grad_sh_grid,
            grad_temperature_grid,
            ctx.want_colors,
        )

        if not ctx.needs_input_grad[0]:
            grad_density_grid = None
        if not ctx.needs_input_grad[1]:
            grad_sh_grid = None

        return grad_density_grid, grad_sh_grid, None, None, None


class _VolumeRenderFunction(autograd.Function):
    @staticmethod
    def forward(ctx, grid, rays, opt, backend: str, include_temperature: bool = False):
        origins = rays.origins
        dirs = rays.dirs / torch.norm(rays.dirs, dim=-1, keepdim=True)
        color = torch.empty(
            (origins.size(0), 3), dtype=torch.float32, device=origins.device
        )

        temperature = torch.empty(
            (origins.size(0),), dtype=torch.float32, device=origins.device
        )
        rays = Rays(origins, dirs)._to_cpp()

        ctx.save_for_backward(color, temperature)
        ctx.dirs = dirs
        ctx.grid = grid
        ctx.rays = rays
        ctx.opt = opt
        ctx.backend = backend

        if include_temperature:
            return color, temperature
        return color, None

    @staticmethod
    def backward(ctx, grad_out):
        (color_cache,) = ctx.saved_tensors
        cu_fn = _C.__dict__[f"volume_render_{ctx.backend}_backward"]

        grad_density_grid = torch.zeros_like(ctx.grid.density_data.data)
        grad_sh_grid = torch.zeros_like(ctx.grid.sh_data.data)
        grad_temperature_grid = torch.zeros_like(ctx.grid.temperature_data.data)

        if ctx.grid.background_data is not None:
            grad_background = torch.zeros_like(ctx.grid.background_data.data)

        grad_holder = _C.GridOutputGrads()

        grad_holder.grad_density_out = grad_density_grid
        grad_holder.grad_sh_out = grad_sh_grid
        grad_holder.grad_temperature_out = grad_temperature_grid

        if ctx.needs_input_grad[2]:
            grad_holder.grad_basis_out = grad_basis
        if ctx.grid.background_data is not None and ctx.needs_input_grad[3]:
            grad_holder.grad_background_out = grad_background
        cu_fn(
            ctx.grid, ctx.rays, ctx.opt, grad_out.contiguous(), color_cache, grad_holder
        )
        ctx.grid = ctx.rays = ctx.opt = None
        if not ctx.needs_input_grad[0]:
            grad_density_grid = None
        if not ctx.needs_input_grad[1]:
            grad_sh_grid = None
        if not ctx.needs_input_grad[2]:
            grad_basis = None
        if not ctx.needs_input_grad[3]:
            grad_background = None
        ctx.basis_data = None

        return (
            grad_density_grid,
            grad_sh_grid,
            grad_basis,
            grad_background,
            grad_temperature_grid,
            None,
            None,
            None,
            None,
            None,
        )


class _TotalVariationFunction(autograd.Function):
    @staticmethod
    def forward(
        ctx,
        data: torch.Tensor,
        links: torch.Tensor,
        start_dim: int,
        end_dim: int,
        use_log_alpha: bool,
        log_alpha_delta: float,
        ignore_edge: bool,
        ndc_coeffs: Tuple[float, float],
    ):
        tv = _C.tv(
            links,
            data,
            start_dim,
            end_dim,
            use_log_alpha,
            log_alpha_delta,
            ignore_edge,
            ndc_coeffs[0],
            ndc_coeffs[1],
        )
        ctx.save_for_backward(links, data)
        ctx.start_dim = start_dim
        ctx.end_dim = end_dim
        ctx.use_log_alpha = use_log_alpha
        ctx.log_alpha_delta = log_alpha_delta
        ctx.ignore_edge = ignore_edge
        ctx.ndc_coeffs = ndc_coeffs
        return tv

    @staticmethod
    def backward(ctx, grad_out):
        links, data = ctx.saved_tensors
        grad_grid = torch.zeros_like(data)
        _C.tv_grad(
            links,
            data,
            ctx.start_dim,
            ctx.end_dim,
            1.0,
            ctx.use_log_alpha,
            ctx.log_alpha_delta,
            ctx.ignore_edge,
            ctx.ndc_coeffs[0],
            ctx.ndc_coeffs[1],
            grad_grid,
        )
        ctx.start_dim = ctx.end_dim = None
        if not ctx.needs_input_grad[0]:
            grad_grid = None
        return grad_grid, None, None, None, None, None, None, None


# END Differentiable CUDA functions with custom gradient


class SparseGrid(nn.Module):
    """
    Main sparse grid data structure.
    initially it will be a dense grid of resolutionlution <resolution>.
    Only float32 is supported.

    :param resolution: int or List[int, int, int], resolutionlution for resampled grid,
    as in the constructor
    :param radius: float or List[float, float, float], the 1/2 side length of the grid,
    optionally in each direction
    :param center: float or List[float, float, float], the center of the grid
    :param basis_type: int, basis type; may use svox2.BASIS_TYPE_* (1 = SH, 4 = learned
    3D texture, 255 = learned MLP)
    :param basis_dim: int, size of basis / number of SH components
                           (must be square number in case of SH)
    :param basis_resolution: int, resolutionlution of grid
    if using BASIS_TYPE_3D_TEXTURE
    :param use_z_order: bool, if true, stores the data initially in a Z-order curve
    if possible
    :param mlp_posenc_size: int, if using BASIS_TYPE_MLP, then enables standard
    axis-aligned positional encoding of
                                 given size on MLP;
                                 if 0 then does not use positional encoding
    :param mlp_width: int, if using BASIS_TYPE_MLP, specifies MLP width
    (hidden dimension)
    :param device: torch.device, device to store the grid
    """

    def __init__(
        self,
        resolution: Union[int, List[int], Tuple[int, int, int]] = 128,
        radius: Union[float, List[float]] = 1.0,
        center: Union[float, List[float]] = [0.0, 0.0, 0.0],
        basis_type: int = BASIS_TYPE_SH,
        basis_dim: int = 9,  # SH/learned basis size; in SH case, square number
        basis_resolution: int = 16,  # Learned basis resolutionlution
        # (x^3 embedding grid)
        use_z_order: bool = False,
        use_sphere_bound: bool = False,
        mlp_posenc_size: int = 0,
        mlp_width: int = 16,
        background_nlayers: int = 0,  # BG MSI layers
        background_resolution: int = 256,  # BG MSI cubemap face size
        device: Union[torch.device, str] = "cpu",
        include_temperature: bool = False,
    ):
        super().__init__()
        self.basis_type = basis_type
        if basis_type == BASIS_TYPE_SH:
            assert (
                utils.isqrt(basis_dim) is not None
            ), "basis_dim (SH) must be a square number"
        assert (
            basis_dim >= 1 and basis_dim <= utils.MAX_SH_BASIS
        ), f"basis_dim 1-{utils.MAX_SH_BASIS} supported"
        self.basis_dim = basis_dim
        self.include_temperature = include_temperature

        self.mlp_posenc_size = mlp_posenc_size
        self.mlp_width = mlp_width

        self.background_nlayers = background_nlayers
        assert (
            background_nlayers == 0 or background_nlayers > 1
        ), "Please use at least 2 MSI layers (trilerp limitation)"
        self.background_resolution = background_resolution

        if isinstance(resolution, int):
            resolution = [resolution] * 3
        else:
            assert (
                len(resolution) == 3
            ), "resolution must be an integer or indexable object of 3 ints"

        if use_z_order and not (
            resolution[0] == resolution[1]
            and resolution[0] == resolution[2]
            and utils.is_pow2(resolution[0])
        ):
            logging.info(
                "Morton code requires a cube grid of power-of-2 size, ignoring..."
            )
            use_z_order = False

        if isinstance(radius, float) or isinstance(radius, int):
            radius = [radius] * 3
        if isinstance(radius, torch.Tensor):
            radius = radius.to(device="cpu", dtype=torch.float32)
        else:
            radius = torch.tensor(radius, dtype=torch.float32, device="cpu")
        if isinstance(center, torch.Tensor):
            center = center.to(device="cpu", dtype=torch.float32)
        else:
            center = torch.tensor(center, dtype=torch.float32, device="cpu")

        self.radius: torch.Tensor = radius  # CPU
        self.center: torch.Tensor = center  # CPU
        self._offset = 0.5 * (1.0 - self.center / self.radius)
        self._scaling = 0.5 / self.radius

        n3: int = reduce(lambda x, y: x * y, resolution)
        if use_z_order:
            init_links = utils.gen_morton(
                resolution[0], device=device, dtype=torch.int32
            ).flatten()
        else:
            init_links = torch.arange(n3, device=device, dtype=torch.int32)

        if use_sphere_bound:
            X = torch.arange(resolution[0], dtype=torch.float32, device=device) - 0.5
            Y = torch.arange(resolution[1], dtype=torch.float32, device=device) - 0.5
            Z = torch.arange(resolution[2], dtype=torch.float32, device=device) - 0.5
            X, Y, Z = torch.meshgrid(X, Y, Z)
            points = torch.stack((X, Y, Z), dim=-1).view(-1, 3)
            gsz = torch.tensor(resolution)
            roffset = 1.0 / gsz - 1.0
            rscaling = 2.0 / gsz
            points = torch.addcmul(
                roffset.to(device=points.device),
                points,
                rscaling.to(device=points.device),
            )

            norms = points.norm(dim=-1)
            mask = norms <= 1.0 + (3**0.5) / gsz.max()
            self.capacity: int = mask.sum()

            data_mask = torch.zeros(n3, dtype=torch.int32, device=device)
            idxs = init_links[mask].long()
            data_mask[idxs] = 1
            data_mask = torch.cumsum(data_mask, dim=0) - 1

            init_links[mask] = data_mask[idxs].int()
            init_links[~mask] = -1
        else:
            self.capacity = n3

        self.density_data = nn.Parameter(
            torch.zeros(self.capacity, 1, dtype=torch.float32, device=device)
        )
        # Called sh for legacy reasons, but it's just the coeffients for whatever
        # spherical basis functions
        self.sh_data = nn.Parameter(
            torch.zeros(
                self.capacity, self.basis_dim * 3, dtype=torch.float32, device=device
            )
        )
        self.temperature_data = nn.Parameter(
            0.5 * torch.ones(self.capacity, 1, dtype=torch.float32, device=device)
        )

        self.basis_data = nn.Parameter(
            torch.empty(0, 0, 0, 0, dtype=torch.float32, device=device),
            requires_grad=False,
        )

        self.background_links: Optional[torch.Tensor]
        self.background_data: Optional[torch.Tensor]
        if self.use_background:
            background_capacity = (self.background_resolution**2) * 2
            background_links = torch.arange(
                background_capacity, dtype=torch.int32, device=device
            ).reshape(self.background_resolution * 2, self.background_resolution)
            self.register_buffer("background_links", background_links)
            self.background_data = nn.Parameter(
                torch.zeros(
                    background_capacity,
                    self.background_nlayers,
                    4,
                    dtype=torch.float32,
                    device=device,
                )
            )
        else:
            self.background_data = nn.Parameter(
                torch.empty(0, 0, 0, dtype=torch.float32, device=device),
                requires_grad=False,
            )

        self.register_buffer("links", init_links.view(resolution))
        self.links: torch.Tensor
        self.opt = RenderOptions()
        self.sparse_grad_indexer: Optional[torch.Tensor] = None
        self.sparse_sh_grad_indexer: Optional[torch.Tensor] = None
        self.sparse_temperature_grad_indexer: Optional[torch.Tensor] = None
        self.sparse_background_indexer: Optional[torch.Tensor] = None
        self.density_rms: Optional[torch.Tensor] = None
        self.temperature_rms: Optional[torch.Tensor] = None
        self.sh_rms: Optional[torch.Tensor] = None
        self.background_rms: Optional[torch.Tensor] = None
        self.basis_rms: Optional[torch.Tensor] = None

        if self.links.is_cuda and use_sphere_bound:
            self.accelerate()

    @property
    def data_dim(self):
        """
        Get the number of channels in the data, including color + density
        (similar to svox 1)
        """
        return self.sh_data.size(1) + 1

    @property
    def basis_resolution(self):
        """
        Return the resolutionlution of the learned spherical function data if using
        3D learned texture, or 0 if only using SH
        """
        return self.basis_data.size(0) if self.BASIS_TYPE_3D_TEXTURE else 0

    @property
    def use_background(self):
        return self.background_nlayers > 0

    @property
    def shape(self):
        return list(self.links.shape) + [self.data_dim]

    def _fetch_links(self, links):
        results_sigma = torch.zeros(
            (links.size(0), 1), device=links.device, dtype=torch.float32
        )
        results_sh = torch.zeros(
            (links.size(0), self.sh_data.size(1)),
            device=links.device,
            dtype=torch.float32,
        )
        results_temperature = torch.zeros(
            (links.size(0), 1),
            device=links.device,
            dtype=torch.float32,
        )

        mask = links >= 0
        idxs = links[mask].long()

        results_sigma[mask] = self.density_data[idxs]
        results_sh[mask] = self.sh_data[idxs]
        results_temperature[mask] = self.temperature_data[idxs].to(device=links.device)
        return results_sigma, results_sh, results_temperature

    def sample(
        self,
        points: torch.Tensor,
        use_kernel: bool = True,
        grid_coords: bool = False,
        want_colors: bool = True,
        want_temperature: bool = True,
    ):
        """
        Grid sampling with trilinear interpolation.
        Behaves like torch.nn.functional.grid_sample
        with padding mode border and align_corners=False
        (better for multi-resolutionlution).

        Any voxel with link < 0 (empty) is considered to have 0 values in all channels
        prior to interpolating.

        :param points: torch.Tensor, (N, 3)
        :param use_kernel: bool, if false uses pure PyTorch version even if on CUDA.
        :param grid_coords: bool, if true then uses grid coordinates
        ([-0.5, resolution[i]-0.5 ] in each dimension);
                                  more numerically exact for resampling
        :param want_colors: bool, if true (default) returns density and colors,
                            else returns density and a dummy tensor to be ignored
                            (much faster)

        :return: (density, color)
        """
        if (
            use_kernel
            and self.links.is_cuda
            and _C is not None
            and not want_temperature
        ):
            assert points.is_cuda
            return _SampleGridAutogradFunction.apply(
                self.density_data,
                self.sh_data,
                self.temperature_data,
                self._to_cpp(grid_coords=grid_coords),
                points,
                want_colors,
                want_temperature,
            )
        else:
            if not grid_coords:
                points = self.world2grid(points)
            points.clamp_min_(0.0)
            for i in range(3):
                points[:, i].clamp_max_(self.links.size(i) - 1)
            l = points.to(torch.long)
            for i in range(3):
                l[:, i].clamp_max_(self.links.size(i) - 2)
            wb = points - l
            wa = 1.0 - wb

            lx, ly, lz = l.unbind(-1)
            links000 = self.links[lx, ly, lz]
            links001 = self.links[lx, ly, lz + 1]
            links010 = self.links[lx, ly + 1, lz]
            links011 = self.links[lx, ly + 1, lz + 1]
            links100 = self.links[lx + 1, ly, lz]
            links101 = self.links[lx + 1, ly, lz + 1]
            links110 = self.links[lx + 1, ly + 1, lz]
            links111 = self.links[lx + 1, ly + 1, lz + 1]

            sigma000, rgb000, temperature000 = self._fetch_links(links000)
            sigma001, rgb001, temperature001 = self._fetch_links(links001)
            sigma010, rgb010, temperature010 = self._fetch_links(links010)
            sigma011, rgb011, temperature011 = self._fetch_links(links011)
            sigma100, rgb100, temperature100 = self._fetch_links(links100)
            sigma101, rgb101, temperature101 = self._fetch_links(links101)
            sigma110, rgb110, temperature110 = self._fetch_links(links110)
            sigma111, rgb111, temperature111 = self._fetch_links(links111)

            c00 = sigma000 * wa[:, 2:] + sigma001 * wb[:, 2:]
            c01 = sigma010 * wa[:, 2:] + sigma011 * wb[:, 2:]
            c10 = sigma100 * wa[:, 2:] + sigma101 * wb[:, 2:]
            c11 = sigma110 * wa[:, 2:] + sigma111 * wb[:, 2:]
            c0 = c00 * wa[:, 1:2] + c01 * wb[:, 1:2]
            c1 = c10 * wa[:, 1:2] + c11 * wb[:, 1:2]
            samples_sigma = c0 * wa[:, :1] + c1 * wb[:, :1]

            if want_colors:
                c00 = rgb000 * wa[:, 2:] + rgb001 * wb[:, 2:]
                c01 = rgb010 * wa[:, 2:] + rgb011 * wb[:, 2:]
                c10 = rgb100 * wa[:, 2:] + rgb101 * wb[:, 2:]
                c11 = rgb110 * wa[:, 2:] + rgb111 * wb[:, 2:]
                c0 = c00 * wa[:, 1:2] + c01 * wb[:, 1:2]
                c1 = c10 * wa[:, 1:2] + c11 * wb[:, 1:2]
                samples_rgb = c0 * wa[:, :1] + c1 * wb[:, :1]
            else:
                samples_rgb = torch.empty_like(self.sh_data[:0])

            t00 = sigma000 * wa[:, 2:] + sigma001 * wb[:, 2:]
            t01 = sigma010 * wa[:, 2:] + sigma011 * wb[:, 2:]
            t10 = sigma100 * wa[:, 2:] + sigma101 * wb[:, 2:]
            t11 = sigma110 * wa[:, 2:] + sigma111 * wb[:, 2:]
            t0 = t00 * wa[:, 1:2] + t01 * wb[:, 1:2]
            t1 = t10 * wa[:, 1:2] + t11 * wb[:, 1:2]

            if want_temperature:
                t00 = temperature000 * wa[:, 2:] + temperature001 * wb[:, 2:]
                t01 = temperature010 * wa[:, 2:] + temperature011 * wb[:, 2:]
                t10 = temperature100 * wa[:, 2:] + temperature101 * wb[:, 2:]
                t11 = temperature110 * wa[:, 2:] + temperature111 * wb[:, 2:]
                t0 = t00 * wa[:, 1:2] + t01 * wb[:, 1:2]
                t1 = t10 * wa[:, 1:2] + t11 * wb[:, 1:2]
                samples_temperature = t0 * wa[:, :1] + t1 * wb[:, :1]
            else:
                samples_temperature = torch.empty_like(self.temperature_data[:0])

            return samples_sigma, samples_rgb, samples_temperature

    def forward(self, points: torch.Tensor, use_kernel: bool = True):
        return self.sample(points, use_kernel=use_kernel)

    def _volume_render_gradcheck_lerp(self, rays: Rays, return_raylen: bool = False):
        """
        trilerp gradcheck version
        """
        origins = self.world2grid(rays.origins)
        dirs = rays.dirs / torch.norm(rays.dirs, dim=-1, keepdim=True)
        viewdirs = dirs
        B = dirs.size(0)
        assert origins.size(0) == B
        gsz = self._grid_size()
        dirs = dirs * (self._scaling * gsz).to(device=dirs.device)
        delta_scale = 1.0 / dirs.norm(dim=1)
        dirs *= delta_scale.unsqueeze(-1)

        sh_mult = utils.eval_sh_bases(self.basis_dim, viewdirs)
        invdirs = 1.0 / dirs

        gsz = self._grid_size()
        gsz_cu = gsz.to(device=dirs.device)
        t1 = (-0.5 - origins) * invdirs
        t2 = (gsz_cu - 0.5 - origins) * invdirs

        t = torch.min(t1, t2)
        t[dirs == 0] = -1e9
        t = torch.max(t, dim=-1).values.clamp_min_(self.opt.near_clip)

        tmax = torch.max(t1, t2)
        tmax[dirs == 0] = 1e9
        tmax = torch.min(tmax, dim=-1).values
        if return_raylen:
            return tmax - t

        log_light_intensity = torch.zeros(B, device=origins.device)
        out_rgb = torch.zeros((B, 3), device=origins.device)
        out_temperature = torch.zeros((B,), device=origins.device)
        # channel or only one
        good_indices = torch.arange(B, device=origins.device)

        origins_ini = origins
        dirs_ini = dirs

        mask = t <= tmax
        good_indices = good_indices[mask]
        origins = origins[mask]
        dirs = dirs[mask]

        #  invdirs = invdirs[mask]
        del invdirs
        t = t[mask]
        sh_mult = sh_mult[mask]
        tmax = tmax[mask]

        while good_indices.numel() > 0:
            pos = origins + t[:, None] * dirs
            pos = pos.clamp_min_(0.0)
            pos = pos.to(device=gsz.device)
            pos[:, 0] = torch.clamp_max(pos[:, 0], gsz[0] - 1)
            pos[:, 1] = torch.clamp_max(pos[:, 1], gsz[1] - 1)
            pos[:, 2] = torch.clamp_max(pos[:, 2], gsz[2] - 1)

            l = pos.to(torch.long)
            l.clamp_min_(0)
            # Perform the operation and ensure the result is long
            l[:, 0] = torch.clamp_max(l[:, 0].float(), gsz[0] - 2).long()
            l[:, 1] = torch.clamp_max(l[:, 1].float(), gsz[1] - 2).long()
            l[:, 2] = torch.clamp_max(l[:, 2].float(), gsz[2] - 2).long()
            pos -= l

            # BEGIN CRAZY TRILERP
            lx, ly, lz = l.unbind(-1)
            links000 = self.links[lx, ly, lz]
            links001 = self.links[lx, ly, lz + 1]
            links010 = self.links[lx, ly + 1, lz]
            links011 = self.links[lx, ly + 1, lz + 1]
            links100 = self.links[lx + 1, ly, lz]
            links101 = self.links[lx + 1, ly, lz + 1]
            links110 = self.links[lx + 1, ly + 1, lz]
            links111 = self.links[lx + 1, ly + 1, lz + 1]

            sigma000, rgb000, temperature000 = self._fetch_links(links000)
            sigma001, rgb001, temperature001 = self._fetch_links(links001)
            sigma010, rgb010, temperature010 = self._fetch_links(links010)
            sigma011, rgb011, temperature011 = self._fetch_links(links011)
            sigma100, rgb100, temperature100 = self._fetch_links(links100)
            sigma101, rgb101, temperature101 = self._fetch_links(links101)
            sigma110, rgb110, temperature110 = self._fetch_links(links110)
            sigma111, rgb111, temperature111 = self._fetch_links(links111)

            pos = pos.to(device=sigma000.device)
            wa, wb = 1.0 - pos, pos
            c00 = sigma000 * wa[:, 2:] + sigma001 * wb[:, 2:]
            c01 = sigma010 * wa[:, 2:] + sigma011 * wb[:, 2:]
            c10 = sigma100 * wa[:, 2:] + sigma101 * wb[:, 2:]
            c11 = sigma110 * wa[:, 2:] + sigma111 * wb[:, 2:]
            c0 = c00 * wa[:, 1:2] + c01 * wb[:, 1:2]
            c1 = c10 * wa[:, 1:2] + c11 * wb[:, 1:2]
            sigma = c0 * wa[:, :1] + c1 * wb[:, :1]

            c00 = rgb000 * wa[:, 2:] + rgb001 * wb[:, 2:]
            c01 = rgb010 * wa[:, 2:] + rgb011 * wb[:, 2:]
            c10 = rgb100 * wa[:, 2:] + rgb101 * wb[:, 2:]
            c11 = rgb110 * wa[:, 2:] + rgb111 * wb[:, 2:]
            c0 = c00 * wa[:, 1:2] + c01 * wb[:, 1:2]
            c1 = c10 * wa[:, 1:2] + c11 * wb[:, 1:2]
            rgb = c0 * wa[:, :1] + c1 * wb[:, :1]

            t00 = temperature000 * wa[:, 2:] + temperature001 * wb[:, 2:]
            t01 = temperature010 * wa[:, 2:] + temperature011 * wb[:, 2:]
            t10 = temperature100 * wa[:, 2:] + temperature101 * wb[:, 2:]
            t11 = temperature110 * wa[:, 2:] + temperature111 * wb[:, 2:]
            t0 = t00 * wa[:, 1:2] + t01 * wb[:, 1:2]
            t1 = t10 * wa[:, 1:2] + t11 * wb[:, 1:2]
            temperature = t0 * wa[:, :1] + t1 * wb[:, :1]

            # END CRAZY TRILERP

            log_att = (
                -self.opt.step_size
                * torch.relu(sigma[..., 0])
                * delta_scale[good_indices]
            )
            weight = torch.exp(log_light_intensity[good_indices]) * (
                1.0 - torch.exp(log_att)
            )
            temperature = temperature.squeeze()

            # <[B', 3, n_sh_coeffs]>
            rgb_sh = rgb.reshape(-1, 3, self.basis_dim)
            rgb = torch.clamp_min(
                torch.sum(sh_mult.unsqueeze(-2) * rgb_sh, dim=-1) + 0.5,
                0.0,
            )  # < [B', 3]>
            rgb = weight[:, None] * rgb[:, :3]

            out_rgb[good_indices] = out_rgb[good_indices] + rgb

            temperature = weight * temperature

            out_temperature[good_indices] += temperature

            log_light_intensity[good_indices] += log_att
            t += self.opt.step_size

            mask = t <= tmax

            good_indices = good_indices[mask]
            origins = origins[mask]
            dirs = dirs[mask]
            #  invdirs = invdirs[mask]
            t = t[mask]
            sh_mult = sh_mult[mask]
            tmax = tmax[mask]

        if self.use_background:
            # Render the MSI background model
            csi = utils.ConcentricSpheresIntersector(
                gsz_cu, origins_ini, dirs_ini, delta_scale
            )
            inner_radius = (
                torch.cross(csi.origins, csi.dirs, dim=-1).norm(dim=-1) + 1e-3
            )
            inner_radius = inner_radius.clamp_min(1.0)
            _, t_last = csi.intersect(inner_radius)
            n_steps = int(self.background_nlayers / self.opt.step_size) + 2
            layer_scale = (self.background_nlayers - 1) / (n_steps + 1)

            def fetch_bg_link(lx, ly, lz):
                results = torch.zeros(
                    [lx.shape[0], self.background_data.size(-1)], device=lx.device
                )
                lnk = self.background_links[lx, ly]
                mask = lnk >= 0
                results[mask] = self.background_data[lnk[mask].long(), lz[mask]]
                return results

            for i in range(n_steps):
                r: float = n_steps / (n_steps - i - 0.5)
                normalized_inv_radius = min(
                    (i + 1) * layer_scale, self.background_nlayers - 1
                )
                layerid = min(int(normalized_inv_radius), self.background_nlayers - 2)
                interp_wt = normalized_inv_radius - layerid

                active_mask, t = csi.intersect(r)
                active_mask = active_mask & (r >= inner_radius)
                if active_mask.count_nonzero() == 0:
                    continue
                t_sub = t[active_mask]
                t_mid_sub = (t_sub + t_last[active_mask]) * 0.5
                sphpos = (
                    csi.origins[active_mask]
                    + t_mid_sub.unsqueeze(-1) * csi.dirs[active_mask]
                )
                invr_mid = 1.0 / torch.norm(sphpos, dim=-1)
                sphpos *= invr_mid.unsqueeze(-1)

                xy = utils.xyz2equirect(sphpos, self.background_links.size(1))
                z = torch.clamp(
                    (1.0 - invr_mid) * self.background_nlayers - 0.5,
                    0.0,
                    self.background_nlayers - 1,
                )
                points = torch.cat([xy, z.unsqueeze(-1)], dim=-1)
                l = points.to(torch.long)
                l[..., 0].clamp_max_(self.background_links.size(0) - 1)
                l[..., 1].clamp_max_(self.background_links.size(1) - 1)
                l[..., 2].clamp_max_(self.background_nlayers - 2)

                wb = points - l
                wa = 1.0 - wb
                lx, ly, lz = l.unbind(-1)
                lnx = (lx + 1) % self.background_links.size(0)
                lny = (ly + 1) % self.background_links.size(1)
                lnz = lz + 1

                v000 = fetch_bg_link(lx, ly, lz)
                v001 = fetch_bg_link(lx, ly, lnz)
                v010 = fetch_bg_link(lx, lny, lz)
                v011 = fetch_bg_link(lx, lny, lnz)
                v100 = fetch_bg_link(lnx, ly, lz)
                v101 = fetch_bg_link(lnx, ly, lnz)
                v110 = fetch_bg_link(lnx, lny, lz)
                v111 = fetch_bg_link(lnx, lny, lnz)

                c00 = v000 * wa[:, 2:] + v001 * wb[:, 2:]
                c01 = v010 * wa[:, 2:] + v011 * wb[:, 2:]
                c10 = v100 * wa[:, 2:] + v101 * wb[:, 2:]
                c11 = v110 * wa[:, 2:] + v111 * wb[:, 2:]
                c0 = c00 * wa[:, 1:2] + c01 * wb[:, 1:2]
                c1 = c10 * wa[:, 1:2] + c11 * wb[:, 1:2]
                rgba = c0 * wa[:, :1] + c1 * wb[:, :1]

                log_att = (
                    -csi.world_step_scale[active_mask]
                    * torch.relu(rgba[:, -1])
                    * (t_sub - t_last[active_mask])
                )
                weight = torch.exp(log_light_intensity[active_mask]) * (
                    1.0 - torch.exp(log_att)
                )
                rgb = torch.clamp_min(rgba[:, :3] * utils.SH_C0 + 0.5, 0.0)
                out_rgb[active_mask] += rgb * weight[:, None]
                log_light_intensity[active_mask] += log_att
                t_last[active_mask] = t[active_mask]

        # Add background color
        if self.opt.background_brightness:
            out_rgb += (
                torch.exp(log_light_intensity).unsqueeze(-1)
                * self.opt.background_brightness
            )

        if self.include_temperature:
            return out_rgb, out_temperature
        return out_rgb

    def _volume_render_gradcheck_nvol_lerp(
        self, rays: Rays, return_raylen: bool = False
    ):
        """
        trilerp gradcheck version
        """
        origins = self.world2grid(rays.origins)
        dirs = rays.dirs / torch.norm(rays.dirs, dim=-1, keepdim=True)
        viewdirs = dirs
        B = dirs.size(0)
        assert origins.size(0) == B
        gsz = self._grid_size()
        dirs = dirs * (self._scaling * gsz).to(device=dirs.device)
        delta_scale = 1.0 / dirs.norm(dim=1)
        dirs *= delta_scale.unsqueeze(-1)

        sh_mult = utils.eval_sh_bases(self.basis_dim, viewdirs)
        invdirs = 1.0 / dirs

        gsz = self._grid_size()
        gsz_cu = gsz.to(device=dirs.device)
        t1 = (-0.5 - origins) * invdirs
        t2 = (gsz_cu - 0.5 - origins) * invdirs

        t = torch.min(t1, t2)
        t[dirs == 0] = -1e9
        t = torch.max(t, dim=-1).values.clamp_min_(self.opt.near_clip)

        tmax = torch.max(t1, t2)
        tmax[dirs == 0] = 1e9
        tmax = torch.min(tmax, dim=-1).values
        if return_raylen:
            return tmax - t

        total_alpha = torch.zeros(B, device=origins.device)
        out_rgb = torch.zeros((B, 3), device=origins.device)
        out_temperature = torch.zeros((B,), device=origins.device)
        good_indices = torch.arange(B, device=origins.device)

        origins_ini = origins
        dirs_ini = dirs

        mask = t <= tmax
        good_indices = good_indices[mask]
        origins = origins[mask]
        dirs = dirs[mask]

        #  invdirs = invdirs[mask]
        del invdirs
        t = t[mask]
        sh_mult = sh_mult[mask]
        tmax = tmax[mask]

        while good_indices.numel() > 0:
            pos = origins + t[:, None] * dirs
            pos = pos.clamp_min_(0.0)
            pos[:, 0] = torch.clamp_max(pos[:, 0], gsz[0] - 1)
            pos[:, 1] = torch.clamp_max(pos[:, 1], gsz[1] - 1)
            pos[:, 2] = torch.clamp_max(pos[:, 2], gsz[2] - 1)

            l = pos.to(torch.long)
            l.clamp_min_(0)
            l[:, 0] = torch.clamp_max(l[:, 0], gsz[0] - 2)
            l[:, 1] = torch.clamp_max(l[:, 1], gsz[1] - 2)
            l[:, 2] = torch.clamp_max(l[:, 2], gsz[2] - 2)
            pos -= l

            # BEGIN CRAZY TRILERP
            lx, ly, lz = l.unbind(-1)
            links000 = self.links[lx, ly, lz]
            links001 = self.links[lx, ly, lz + 1]
            links010 = self.links[lx, ly + 1, lz]
            links011 = self.links[lx, ly + 1, lz + 1]
            links100 = self.links[lx + 1, ly, lz]
            links101 = self.links[lx + 1, ly, lz + 1]
            links110 = self.links[lx + 1, ly + 1, lz]
            links111 = self.links[lx + 1, ly + 1, lz + 1]

            sigma000, rgb000, temperature000 = self._fetch_links(links000)
            sigma001, rgb001, temperature001 = self._fetch_links(links001)
            sigma010, rgb010, temperature010 = self._fetch_links(links010)
            sigma011, rgb011, temperature011 = self._fetch_links(links011)
            sigma100, rgb100, temperature100 = self._fetch_links(links100)
            sigma101, rgb101, temperature101 = self._fetch_links(links101)
            sigma110, rgb110, temperature110 = self._fetch_links(links110)
            sigma111, rgb111, temperature111 = self._fetch_links(links111)

            wa, wb = 1.0 - pos, pos
            c00 = sigma000 * wa[:, 2:] + sigma001 * wb[:, 2:]
            c01 = sigma010 * wa[:, 2:] + sigma011 * wb[:, 2:]
            c10 = sigma100 * wa[:, 2:] + sigma101 * wb[:, 2:]
            c11 = sigma110 * wa[:, 2:] + sigma111 * wb[:, 2:]
            c0 = c00 * wa[:, 1:2] + c01 * wb[:, 1:2]
            c1 = c10 * wa[:, 1:2] + c11 * wb[:, 1:2]
            sigma = c0 * wa[:, :1] + c1 * wb[:, :1]

            c00 = rgb000 * wa[:, 2:] + rgb001 * wb[:, 2:]
            c01 = rgb010 * wa[:, 2:] + rgb011 * wb[:, 2:]
            c10 = rgb100 * wa[:, 2:] + rgb101 * wb[:, 2:]
            c11 = rgb110 * wa[:, 2:] + rgb111 * wb[:, 2:]
            c0 = c00 * wa[:, 1:2] + c01 * wb[:, 1:2]
            c1 = c10 * wa[:, 1:2] + c11 * wb[:, 1:2]
            rgb = c0 * wa[:, :1] + c1 * wb[:, :1]

            t00 = temperature000 * wa[:, 2:] + temperature001 * wb[:, 2:]
            t01 = temperature010 * wa[:, 2:] + temperature011 * wb[:, 2:]
            t10 = temperature100 * wa[:, 2:] + temperature101 * wb[:, 2:]
            t11 = temperature110 * wa[:, 2:] + temperature111 * wb[:, 2:]
            t0 = t00 * wa[:, 1:2] + t01 * wb[:, 1:2]
            t1 = t10 * wa[:, 1:2] + t11 * wb[:, 1:2]
            temperature = t0 * wa[:, :1] + t1 * wb[:, :1]

            # END CRAZY TRILERP

            log_att = (
                -self.opt.step_size
                * torch.relu(sigma[..., 0])
                * delta_scale[good_indices]
            )
            weight = torch.exp(log_light_intensity[good_indices]) * (
                1.0 - torch.exp(log_att)
            )
            delta_alpha = 1.0 - torch.exp(log_att)
            total_alpha_sub = total_alpha[good_indices]
            new_total_alpha = torch.clamp_max(total_alpha_sub + delta_alpha, 1.0)
            weight = new_total_alpha - total_alpha_sub
            total_alpha[good_indices] = new_total_alpha

            # [B', 3, n_sh_coeffs]
            rgb_sh = rgb.reshape(-1, 3, self.basis_dim)
            rgb = torch.clamp_min(
                torch.sum(sh_mult.unsqueeze(-2) * rgb_sh, dim=-1) + 0.5,
                0.0,
            )  # [B', 3]
            rgb = weight[:, None] * rgb[:, :3]

            out_rgb[good_indices] += rgb

            temperature = weight[:, None] * temperature[:, :]

            out_temperature[good_indices] += temperature

            t += self.opt.step_size

            mask = t <= tmax
            good_indices = good_indices[mask]
            origins = origins[mask]
            dirs = dirs[mask]
            t = t[mask]
            sh_mult = sh_mult[mask]
            tmax = tmax[mask]

        # Add background color
        if self.opt.background_brightness:
            out_rgb += (1.0 - total_alpha).unsqueeze(
                -1
            ) * self.opt.background_brightness

        if self.include_temperature:
            return out_rgb, out_temperature
        return out_rgb, None

    def volume_render(
        self,
        rays: Rays,
        use_kernel: bool = True,
        randomize: bool = False,
        return_raylen: bool = False,
    ):
        """
        Standard volume rendering. See grid.opt.* (RenderOptions) for configs.

        :param rays: Rays, (origins (N, 3), dirs (N, 3))
        :param use_kernel: bool, if false uses pure PyTorch version even if on CUDA.
        :param randomize: bool, whether to enable randomness
        :param return_raylen: bool, if true then only returns the length of the
                                    ray-cube intersection and quits
        :param include_temperature: bool,
        if true, it returns temperature image + rgb image
        :return: (N, 3), predicted RGB
        """
        if (
            use_kernel
            and self.links.is_cuda
            and _C is not None
            and not return_raylen
            # and not self.include_temperature
        ):
            assert rays.is_cuda
            basis_data = (
                self._eval_basis_mlp(rays.dirs)
                if self.basis_type == BASIS_TYPE_MLP
                else None
            )
            return _VolumeRenderFunction.apply(
                self._to_cpp(replace_basis_data=basis_data),
                rays._to_cpp(),
                self.opt._to_cpp(randomize=randomize),
                self.opt.backend,
                self.include_temperature,
            )
        else:
            warn("Using slow volume rendering, should only be used for debugging")
            if self.opt.backend == "nvol":
                return self._volume_render_gradcheck_nvol_lerp(
                    rays,
                    return_raylen=return_raylen,
                )
            else:
                return self._volume_render_gradcheck_lerp(
                    rays,
                    return_raylen=return_raylen,
                )

    def volume_render_fused_rgbt(
        self,
        rays: Rays,
        rgb_gt: torch.Tensor,
        temperature_gt: torch.Tensor | None = None,
        randomize: bool = False,
        beta_loss: float = 0.0,
        sparsity_loss: float = 0.0,
        t_loss: float = 0.0,
        density_threshold: float = 0.0,
        t_surface_loss: float = 0.0,
        l1_loss: float = 0.0,
        two_steps: int = False,
    ):
        """
        Standard volume rendering with fused MSE gradient generation,
            given a ground truth color for each pixel.
        Will update the *.grad tensors for each parameter
        You can then subtract the grad manually or use the optim_*_step methods

        See grid.opt.* (RenderOptions) for configs.

        :param rays: Rays, (origins (N, 3), dirs (N, 3))
        :param rgb_gt: (N, 3), GT pixel colors, each channel in [0, 1]
        :param temperature_gt: (N, ), GT temperature , each channel in [0, 1]
        :param randomize: bool, whether to enable randomness
        :param beta_loss: float, weighting for beta loss to add to the gradient.
                                 (fused into the backward pass).
                                 This is average over the rays in the batch.
                                 Beta loss also from neural volumes:
                                 [Lombardi et al., ToG 2019]
        :param t_loss: Magnitude of density backpropagation from temperature loss
        :param l1_loss: Magnitude of the L1 reconstruction loss on thermal images
        :param density_threshold: density threshold used to render surface
        temperature used in the surface loss
        :param t_surface_loss: Magnitude of surface temperature loss
        :param two_steps: Turn true if you removed some data from the training set

        :return: (N, 3), predicted RGB
        """
        assert (
            _C is not None and self.sh_data.is_cuda
        ), "CUDA extension is currently required for fused"
        assert rays.is_cuda
        grad_density, grad_sh, grad_basis, grad_bg, grad_temperature = (
            self._get_data_grads()
        )

        rgb_out = torch.zeros_like(rgb_gt).to(device=self.density_data.device)

        if temperature_gt is None:
            temperature_out = None
        else:
            temperature_out = torch.zeros_like(temperature_gt).to(
                device=self.density_data.device
            )
            temperature_gt = temperature_gt.mean(axis=1).unsqueeze(-1)
            temperature_out = temperature_out.mean(axis=1).unsqueeze(-1)

        basis_data: Optional[torch.Tensor] = None

        self.sparse_grad_indexer = torch.zeros(
            (self.density_data.size(0),),
            dtype=torch.bool,
            device=self.density_data.device,
        )

        grad_holder = _C.GridOutputGrads()
        grad_holder.grad_density_out = grad_density
        grad_holder.grad_sh_out = grad_sh
        grad_holder.mask_out = self.sparse_grad_indexer

        if self.use_background:
            grad_holder.grad_background_out = grad_bg
            self.sparse_background_indexer = torch.zeros(
                list(self.background_data.shape[:-1]),
                dtype=torch.bool,
                device=self.background_data.device,
            )
            grad_holder.mask_background_out = self.sparse_background_indexer

        if self.include_temperature:
            grad_holder.grad_temperature_out = grad_temperature

            custom_gradient = torch.zeros_like(temperature_gt).to(
                device=self.density_data.device
            )

            if two_steps:  # case where we dropped images
                origins = self.world2grid(rays.origins)
                options = {
                    "dtype": origins.dtype,
                    "device": origins.device,
                    "requires_grad": False,
                }

                log_transmit = torch.empty((origins.size(0),), **options)

                _C.volume_render_cuvol_rgbt_forward(
                    self._to_cpp(),
                    rays._to_cpp(),
                    self.opt._to_cpp(),
                    rgb_out.float(),
                    temperature_out.float(),
                    log_transmit.float(),
                )

                # Create a mask where temperature_gt is equal to 0 to not optimize them
                mask_temperature = temperature_gt == 0
                mask_rgb = rgb_gt == 0

                # Replace values in temperature_gt with corresponding values
                # from temperature_out using the mask
                temperature_gt[mask_temperature] = temperature_out[mask_temperature]
                rgb_gt[mask_rgb] = rgb_out[mask_rgb]

                _C.volume_render_cuvol_rgbt_backward(
                    self._to_cpp(),
                    rays._to_cpp(),
                    self.opt._to_cpp(),
                    rgb_gt.float(),
                    temperature_gt.float(),
                    custom_gradient.float(),
                    1.0,  # l2_loss
                    beta_loss,
                    sparsity_loss,
                    t_loss,
                    t_surface_loss,
                    density_threshold,
                    l1_loss,
                    rgb_out.float(),
                    temperature_out.float(),
                    grad_holder,
                    log_transmit.float(),
                )
            else:
                _C.volume_render_cuvol_rgbt_fused(
                    self._to_cpp(),
                    rays._to_cpp(),
                    self.opt._to_cpp(),
                    rgb_gt.float(),
                    temperature_gt.float(),
                    custom_gradient.float(),
                    1.0,  # l2_loss
                    beta_loss,
                    sparsity_loss,
                    t_loss,
                    t_surface_loss,
                    density_threshold,
                    l1_loss,
                    rgb_out.float(),
                    temperature_out.float(),
                    grad_holder,
                )

            self.sparse_sh_grad_indexer = self.sparse_grad_indexer.clone()
            self.sparse_temperature_grad_indexer = self.sparse_grad_indexer.clone()

            return rgb_out.clamp_max_(1.0), temperature_out.clamp_max_(1.0)

        cu_fn = _C.__dict__[f"volume_render_{self.opt.backend}_fused"]
        #  with utils.Timing("actual_render"):
        cu_fn(
            self._to_cpp(replace_basis_data=basis_data),
            rays._to_cpp(),
            self.opt._to_cpp(randomize=randomize),
            rgb_gt,
            beta_loss,
            sparsity_loss,
            rgb_out,
            grad_holder,
        )

        self.sparse_sh_grad_indexer = self.sparse_grad_indexer.clone()
        return rgb_out, temperature_out

    def volume_render_image(
        self,
        camera: Camera,
        use_kernel: bool = True,
        randomize: bool = False,
        batch_size: int = 5000,
        return_raylen: bool = False,
    ):
        """
        Standard volume rendering (entire image version).
        See grid.opt.* (RenderOptions) for configs.

        :param camera: Camera
        :param use_kernel: bool, if false uses pure PyTorch version even if on CUDA.
        :param randomize: bool, whether to enable randomness
        :return: (H, W, 3), predicted RGB image
        """
        if (
            not torch.is_grad_enabled()
            and not return_raylen
            and not self.include_temperature
        ):
            # Use the fast image render kernel if available
            cu_fn = _C.__dict__["volume_render_cuvol_image"]

            rgb_out = cu_fn(self._to_cpp(), camera._to_cpp(), self.opt._to_cpp())
            return rgb_out, torch.zeros((rgb_out.shape[0], rgb_out.shape[1], 1))
        else:
            # Manually generate rays for now
            rays = camera.gen_rays()
            all_rgb_out = []
            all_temperature_out = [] if self.include_temperature else None

            for batch_start in range(0, camera.height * camera.width, batch_size):
                rgb_out_part, temperature_out_part = self.volume_render(
                    rays[batch_start : batch_start + batch_size],
                    use_kernel=use_kernel,
                    randomize=randomize,
                    return_raylen=return_raylen,
                )
                if self.include_temperature:
                    all_temperature_out.append(temperature_out_part)
                all_rgb_out.append(rgb_out_part)

            all_rgb_out = torch.cat(all_rgb_out, dim=0)
            rgb_image = all_rgb_out.view(camera.height, camera.width, -1)

            if self.include_temperature:
                all_temperature_out = torch.cat(all_temperature_out, dim=0)
                temperature_image = all_temperature_out.view(
                    camera.height, camera.width, -1
                )
                return rgb_image, temperature_image

            return rgb_image, None

    def volume_render_depth(self, rays: Rays, sigma_thresh: Optional[float] = None):
        """
          Volumetric depth rendering for rays

          :param rays: Rays, (origins (N, 3), dirs (N, 3))
          :param sigma_thresh: Optional[float].
          If None then finds the standard expected termination
        (NOTE: this is the absolute length along the ray,
              not the z-depth as usually expected);
        else then finds the first point where sigma strictly exceeds sigma_thresh

          :return: (N,)
        """
        if sigma_thresh is None:
            return _C.volume_render_expected_term(
                self._to_cpp(), rays._to_cpp(), self.opt._to_cpp()
            )
        return _C.volume_render_sigma_thresh(
            self._to_cpp(), rays._to_cpp(), self.opt._to_cpp(), sigma_thresh
        )

    def volume_render_surface_temperature(
        self, rays: Rays, sigma_thresh: float | None = None
    ):
        """
        Rendering of surface temeprature

        :param rays: Rays, (origins (N, 3), dirs (N, 3))
        :param sigma_thresh: Optional[float].
        If None then finds the standard expected termination

        :return: surface temperature (N,)
        """
        if sigma_thresh is None:
            return _C.volume_render_surface_temperature(
                self._to_cpp(), rays._to_cpp(), self.opt._to_cpp()
            )
        return _C.volume_render_surface_temperature_sigma_thresh(
            self._to_cpp(), rays._to_cpp(), self.opt._to_cpp(), sigma_thresh
        )

    def volume_render_depth_image(
        self,
        camera: Camera,
        sigma_thresh: Optional[float] = None,
        batch_size: int = 5000,
    ):
        """
          Volumetric depth rendering for full image

          :param camera: Camera, a single camera
          :param sigma_thresh: Optional[float].
          If None then finds the standard expected termination
        (NOTE: this is the absolute length along the ray,
        not the z-depth as usually expected);
        else then finds the first point where sigma strictly exceeds sigma_thresh

          :return: depth (H, W)
        """
        rays = camera.gen_rays()
        all_depths = []
        for batch_start in range(0, camera.height * camera.width, batch_size):
            depths = self.volume_render_depth(
                rays[batch_start : batch_start + batch_size], sigma_thresh
            )
            all_depths.append(depths)
        all_depth_out = torch.cat(all_depths, dim=0)
        return all_depth_out.view(camera.height, camera.width)

    def volume_render_surface_temperature_image(
        self,
        camera: Camera,
        sigma_thresh: float | None = None,
        batch_size: int = 5000,
    ):
        """
        Rendering of surface temperature

        :param camera: Camera, a single camera

        :return: surface_temperature (H, W)
        """
        rays = camera.gen_rays()
        all_surface_temperature = []
        for batch_start in range(0, camera.height * camera.width, batch_size):
            surface_temperature = self.volume_render_surface_temperature(
                rays[batch_start : batch_start + batch_size], sigma_thresh
            )
            all_surface_temperature.append(surface_temperature)
        all_surface_temperature_out = torch.cat(all_surface_temperature, dim=0)

        return all_surface_temperature_out.view(camera.height, camera.width).clamp(
            0.0, 1.0
        )

    def resample(
        self,
        resolution: Union[int, List[int]],
        sigma_thresh: float = 5.0,
        weight_thresh: float = 0.01,
        dilate: int = 2,
        cameras: Optional[List[Camera]] = None,
        use_z_order: bool = False,
        accelerate: bool = True,
        weight_render_stop_thresh: float = 1e-7,
        max_elements: int = 0,
    ):
        """
        Resample and sparsify the grid; used to increase the resolutionlution
        :param resolution: int or List[int, int, int], resolutionlution
        for resampled grid, as in the constructor
        :param sigma_thresh: float,
        threshold to apply on the sigma (if using sigma thresh i.e. cameras NOT given)
        :param weight_thresh: float,
        threshold to apply on the weights (if using weight thresh i.e. cameras given)
        :param dilate: int, if true applies dilation of size <dilate>
        to the 3D mask for nodes to keep in the grid
        (keep neighbors in all 28 directions, including diagonals, of the desired nodes)
        :param cameras: Optional[List[Camera]],
        optional list of cameras in OpenCV convention (if given, uses weight thresholding)
        :param use_z_order: bool, if true, stores the data initially in a Z-order curve
        if possible
        :param accelerate: bool, if true (default), calls grid.accelerate()
        after resampling
                           to build distance transform table (only if on CUDA)
        :param weight_render_stop_thresh: float, stopping threshold
        for grid weight render in [0, 1];
         0.0 = no thresholding, 1.0 = hides everything.
         Useful for force-cutting off
         junk that contributes very little at the end of a ray
        :param max_elements: int, if nonzero, an upper bound on the number of elements
        in the upsampled grid; we will adjust the threshold to match it
        """
        with torch.no_grad():
            device = self.links.device
            if isinstance(resolution, int):
                resolution = [resolution] * 3
            else:
                assert (
                    len(resolution) == 3
                ), "resolution must be an integer or indexable object of 3 ints"

            if use_z_order and not (
                resolution[0] == resolution[1]
                and resolution[0] == resolution[2]
                and utils.is_pow2(resolution[0])
            ):
                logging.info(
                    "Morton code requires a cube grid of power-of-2 size, ignoring..."
                )
                use_z_order = False

            self.capacity: int = reduce(lambda x, y: x * y, resolution)
            curr_resolution = self.links.shape
            dtype = torch.float32
            resolution_facts = [
                0.5 * curr_resolution[i] / resolution[i] for i in range(3)
            ]
            X = torch.linspace(
                resolution_facts[0] - 0.5,
                curr_resolution[0] - resolution_facts[0] - 0.5,
                resolution[0],
                dtype=dtype,
            )
            Y = torch.linspace(
                resolution_facts[1] - 0.5,
                curr_resolution[1] - resolution_facts[1] - 0.5,
                resolution[1],
                dtype=dtype,
            )
            Z = torch.linspace(
                resolution_facts[2] - 0.5,
                curr_resolution[2] - resolution_facts[2] - 0.5,
                resolution[2],
                dtype=dtype,
            )
            X, Y, Z = torch.meshgrid(X, Y, Z)
            points = torch.stack((X, Y, Z), dim=-1).view(-1, 3)

            if use_z_order:
                morton = utils.gen_morton(resolution[0], dtype=torch.long).view(-1)
                points[morton] = points.clone()
            points = points.to(device=device)

            use_weight_thresh = cameras is not None

            batch_size = 720720
            all_sample_vals_density = []

            for i in tqdm(range(0, len(points), batch_size)):
                sample_vals_density, _, _ = self.sample(
                    points[i : i + batch_size],
                    grid_coords=True,
                    want_colors=False,
                )
                sample_vals_density = sample_vals_density
                all_sample_vals_density.append(sample_vals_density)

            self.density_data.grad = None
            self.temperature_data.grad = None
            self.sh_data.grad = None
            self.sparse_grad_indexer = None
            self.sparse_temperature_grad_indexer = None
            self.sparse_sh_grad_indexer = None
            self.density_rms = None
            self.temperature_rms = None
            self.sh_rms = None

            sample_vals_density = torch.cat(all_sample_vals_density, dim=0).view(
                resolution
            )
            del all_sample_vals_density
            if use_weight_thresh:
                gsz = torch.tensor(resolution)
                offset = (self._offset * gsz - 0.5).to(device=device)
                scaling = (self._scaling * gsz).to(device=device)
                max_wt_grid = torch.zeros(
                    resolution, dtype=torch.float32, device=device
                )

                for i, cam in enumerate(cameras):
                    _C.grid_weight_render(
                        sample_vals_density,
                        cam._to_cpp(),
                        0.5,
                        weight_render_stop_thresh,
                        #  self.opt.last_sample_opaque,
                        False,
                        offset,
                        scaling,
                        max_wt_grid,
                    )

                sample_vals_mask = max_wt_grid >= weight_thresh
                if (
                    max_elements > 0
                    and max_elements < max_wt_grid.numel()
                    and max_elements < torch.count_nonzero(sample_vals_mask)
                ):
                    # To bound the memory usage
                    weight_thresh_bounded = (
                        torch.topk(max_wt_grid.view(-1), k=max_elements, sorted=False)
                        .values.min()
                        .item()
                    )
                    weight_thresh = max(weight_thresh, weight_thresh_bounded)
                    logging.info(
                        " Readjusted weight thresh to fit to memory:", weight_thresh
                    )
                    sample_vals_mask = max_wt_grid >= weight_thresh
                del max_wt_grid
            else:
                sample_vals_mask = sample_vals_density >= sigma_thresh
                if (
                    max_elements > 0
                    and max_elements < sample_vals_density.numel()
                    and max_elements < torch.count_nonzero(sample_vals_mask)
                ):
                    # To bound the memory usage
                    sigma_thresh_bounded = (
                        torch.topk(
                            sample_vals_density.view(-1), k=max_elements, sorted=False
                        )
                        .values.min()
                        .item()
                    )
                    sigma_thresh = max(sigma_thresh, sigma_thresh_bounded)
                    logging.info(
                        " Readjusted sigma thresh to fit to memory:", sigma_thresh
                    )
                    sample_vals_mask = sample_vals_density >= sigma_thresh

                if self.opt.last_sample_opaque:
                    # Don't delete the last z layer
                    sample_vals_mask[:, :, -1] = 1

            if dilate:
                for i in range(int(dilate)):
                    sample_vals_mask = _C.dilate(sample_vals_mask)
            sample_vals_mask = sample_vals_mask.view(-1)
            sample_vals_density = sample_vals_density.view(-1)
            sample_vals_density = sample_vals_density[sample_vals_mask]
            cnz = torch.count_nonzero(sample_vals_mask).item()

            # Now we can get the colors for the sparse points
            points = points[sample_vals_mask]
            logging.info("Pass 2/2 (color), eval", cnz, "sparse pts")
            all_sample_vals_sh = []
            all_sample_vals_temperature = []

            for i in tqdm(range(0, len(points), batch_size)):
                _, sample_vals_sh, sample_vals_temperature = self.sample(
                    points[i : i + batch_size],
                    grid_coords=True,
                    want_colors=True,
                    want_temperature=self.include_temperature,
                )

                all_sample_vals_sh.append(sample_vals_sh)
                if self.include_temperature:
                    all_sample_vals_temperature.append(sample_vals_temperature)

            sample_vals_sh = (
                torch.cat(all_sample_vals_sh, dim=0)
                if len(all_sample_vals_sh)
                else torch.empty_like(self.sh_data[:0])
            )

            if self.include_temperature:
                sample_vals_temperature = (
                    torch.cat(all_sample_vals_temperature, dim=0)
                    if len(all_sample_vals_temperature)
                    else torch.empty_like(sample_vals_density)
                )
                del self.temperature_data
                del all_sample_vals_temperature

            del self.density_data
            del self.sh_data
            del all_sample_vals_sh

            if use_z_order:
                inv_morton = torch.empty_like(morton)
                inv_morton[morton] = torch.arange(morton.size(0), dtype=morton.dtype)
                inv_idx = inv_morton[sample_vals_mask]
                init_links = torch.full(
                    (sample_vals_mask.size(0),), fill_value=-1, dtype=torch.int32
                )
                init_links[inv_idx] = torch.arange(inv_idx.size(0), dtype=torch.int32)
            else:
                init_links = (
                    torch.cumsum(sample_vals_mask.to(torch.int32), dim=-1).int() - 1
                )
                init_links[~sample_vals_mask] = -1

            self.capacity = cnz
            logging.info(" New cap:", self.capacity)
            del sample_vals_mask

            self.density_data = nn.Parameter(
                sample_vals_density.view(-1, 1).to(device=device)
            )
            self.sh_data = nn.Parameter(sample_vals_sh.to(device=device))
            if self.include_temperature:
                self.temperature_data = nn.Parameter(
                    sample_vals_temperature.view(-1, 1).to(device=device)
                )
            self.links = init_links.view(resolution).to(device=device)

            if accelerate and self.links.is_cuda:
                self.accelerate()

    def sparsify_background(
        self,
        sigma_thresh: float = 1.0,
        dilate: int = 1,
    ):
        device = self.background_links.device
        sigma_mask = torch.zeros(
            list(self.background_links.shape) + [self.background_nlayers],
            dtype=torch.bool,
            device=device,
        ).view(-1, self.background_nlayers)
        nonempty_mask = self.background_links.view(-1) >= 0
        data_mask = self.background_data[..., -1] >= sigma_thresh
        sigma_mask[nonempty_mask] = data_mask
        sigma_mask = sigma_mask.view(
            list(self.background_links.shape) + [self.background_nlayers]
        )
        for _ in range(int(dilate)):
            sigma_mask = _C.dilate(sigma_mask)

        sigma_mask = sigma_mask.any(-1) & nonempty_mask.view(
            self.background_links.shape
        )
        self.background_links[~sigma_mask] = -1
        retain_vals = self.background_links[sigma_mask]
        self.background_links[sigma_mask] = torch.arange(
            retain_vals.size(0), dtype=torch.int32, device=device
        )
        self.background_data = nn.Parameter(
            self.background_data.data[retain_vals.long()]
        )

    def accelerate(self):
        """
        Accelerate using cuda function.
        """
        assert (
            _C is not None and self.links.is_cuda
        ), "CUDA extension is currently required for accelerate"
        _C.accel_dist_prop(self.links)

    def world2grid(self, points):
        """
        World coordinates to grid coordinates. Grid coordinates are
        normalized to [0, n_voxels] in each side

        :param points: (N, 3)
        :return: (N, 3)
        """
        gsz = self._grid_size()
        offset = self._offset * gsz - 0.5
        scaling = self._scaling * gsz
        return torch.addcmul(
            offset.to(device=points.device), points, scaling.to(device=points.device)
        )

    def grid2world(self, points):
        """
        Grid coordinates to world coordinates. Grid coordinates are
        normalized to [0, n_voxels] in each side

        :param points: (N, 3)
        :return: (N, 3)
        """
        gsz = self._grid_size()
        roffset = self.radius * (1.0 / gsz - 1.0) + self.center
        rscaling = 2.0 * self.radius / gsz
        return torch.addcmul(
            roffset.to(device=points.device), points, rscaling.to(device=points.device)
        )

    def save(self, path: str, compress: bool = False):
        """
        Save to a path
        """
        save_fn = np.savez_compressed if compress else np.savez
        data = {
            "radius": self.radius.numpy(),
            "center": self.center.numpy(),
            "links": self.links.cpu().numpy(),
            "density_data": self.density_data.data.cpu().numpy(),
            "sh_data": self.sh_data.data.cpu().numpy().astype(np.float16),
            "temperature_data": self.temperature_data.data.cpu()
            .numpy()
            .astype(np.float16),
        }

        if self.use_background:
            data["background_links"] = self.background_links.cpu().numpy()
            data["background_data"] = self.background_data.data.cpu().numpy()
        data["basis_type"] = self.basis_type

        save_fn(path, **data)

    @classmethod
    def load(
        cls,
        path: str,
        device: Union[torch.device, str] = "cpu",
        include_temperature: bool = False,
    ):
        """
        Load from path
        """
        z = np.load(path)
        if "data" in z.keys():
            # Compatibility
            all_data = z.f.data
            sh_data = all_data[..., 1:]
            density_data = all_data[..., :1]
        else:
            sh_data = z.f.sh_data
            density_data = z.f.density_data
            temperature_data = z.f.temperature_data

        if "background_data" in z:
            background_data = z["background_data"]
            background_links = z["background_links"]
        else:
            background_data = None

        links = z.f.links
        basis_dim = (sh_data.shape[1]) // 3
        radius = z.f.radius.tolist() if "radius" in z.files else [1.0, 1.0, 1.0]
        center = z.f.center.tolist() if "center" in z.files else [0.0, 0.0, 0.0]
        grid = cls(
            1,
            radius=radius,
            center=center,
            basis_dim=basis_dim,
            use_z_order=False,
            device="cpu",
            basis_type=z["basis_type"].item() if "basis_type" in z else BASIS_TYPE_SH,
            mlp_posenc_size=(
                z["mlp_posenc_size"].item() if "mlp_posenc_size" in z else 0
            ),
            mlp_width=z["mlp_width"].item() if "mlp_width" in z else 16,
            background_nlayers=0,
            include_temperature=include_temperature,
        )
        if sh_data.dtype != np.float32:
            sh_data = sh_data.astype(np.float32)
        if density_data.dtype != np.float32:
            density_data = density_data.astype(np.float32)
        if temperature_data.dtype != np.float32:
            temperature_data = temperature_data.astype(np.float32)

        sh_data = torch.from_numpy(sh_data).to(device=device)
        density_data = torch.from_numpy(density_data).to(device=device)
        grid.sh_data = nn.Parameter(sh_data)
        grid.density_data = nn.Parameter(density_data)
        temperature_data_tensor = torch.from_numpy(temperature_data)
        grid.temperature_data = nn.Parameter(temperature_data_tensor)

        grid.links = torch.from_numpy(links).to(device=device)
        grid.capacity = grid.sh_data.size(0)

        grid.basis_data = nn.Parameter(grid.basis_data.data.to(device=device))

        if background_data is not None:
            background_data = torch.from_numpy(background_data).to(device=device)
            grid.background_nlayers = background_data.shape[1]
            grid.background_resolution = background_links.shape[1]
            grid.background_data = nn.Parameter(background_data)
            grid.background_links = torch.from_numpy(background_links).to(device=device)
        else:
            grid.background_data.data = grid.background_data.data.to(device=device)

        if grid.links.is_cuda:
            grid.accelerate()
        return grid

    def tv(
        self,
        log_alpha: bool = False,
        log_alpha_delta: float = 2.0,
        ndc_coeffs: Tuple[float, float] = (-1.0, -1.0),
    ):
        """
        Compute total variation over sigma,
        similar to Neural Volumes [Lombardi et al., ToG 2019]

        :return: torch.Tensor, size scalar, the TV value (sum over channels,
                 mean over voxels)
        """
        assert (
            _C is not None and self.sh_data.is_cuda
        ), "CUDA extension is currently required for total variation"
        assert not log_alpha, "No longer supported"
        return _TotalVariationFunction.apply(
            self.density_data,
            self.links,
            0,
            1,
            log_alpha,
            log_alpha_delta,
            False,
            ndc_coeffs,
        )

    def tv_color(
        self,
        start_dim: int = 0,
        end_dim: Optional[int] = None,
        log_alpha: bool = False,
        log_alpha_delta: float = 2.0,
        ndc_coeffs: Tuple[float, float] = (-1.0, -1.0),
    ):
        """
        Compute total variation on color

        :param start_dim: int, first color channel dimension to compute TV over
        (inclusive).
                          Default 0.
        :param end_dim: int, last color channel dimension to compute TV over
        (exclusive).
                          Default None = all dimensions until the end.

        :return: torch.Tensor, size scalar, the TV value (sum over channels,
                 mean over voxels)
        """
        assert (
            _C is not None and self.sh_data.is_cuda
        ), "CUDA extension is currently required for total variation"
        assert not log_alpha, "No longer supported"
        if end_dim is None:
            end_dim = self.sh_data.size(1)
        end_dim = end_dim + self.sh_data.size(1) if end_dim < 0 else end_dim
        start_dim = start_dim + self.sh_data.size(1) if start_dim < 0 else start_dim
        return _TotalVariationFunction.apply(
            self.sh_data,
            self.links,
            start_dim,
            end_dim,
            log_alpha,
            log_alpha_delta,
            True,
            ndc_coeffs,
        )

    def inplace_tv_grad(
        self,
        grad: torch.Tensor,
        scaling: float = 1.0,
        sparse_frac: float = 0.01,
        log_alpha: bool = False,
        log_alpha_delta: float = 2.0,
        ndc_coeffs: Tuple[float, float] = (-1.0, -1.0),
        contiguous: bool = True,
    ):
        """
        Add gradient of total variation for sigma as in Neural Volumes
        [Lombardi et al., ToG 2019]
        directly into the gradient tensor, multiplied by 'scaling'
        """
        assert (
            _C is not None and self.density_data.is_cuda and grad.is_cuda
        ), "CUDA extension is currently required for total variation"

        assert not log_alpha, "No longer supported"
        rand_cells = self._get_rand_cells(sparse_frac, contiguous=contiguous)
        if rand_cells is not None:
            if rand_cells.size(0) > 0:
                _C.tv_grad_sparse(
                    self.links,
                    self.density_data,
                    rand_cells,
                    self._get_sparse_grad_indexer(),
                    0,
                    1,
                    scaling,
                    log_alpha,
                    log_alpha_delta,
                    False,
                    self.opt.last_sample_opaque,
                    ndc_coeffs[0],
                    ndc_coeffs[1],
                    grad,
                )
        else:
            _C.tv_grad(
                self.links,
                self.density_data,
                0,
                1,
                scaling,
                log_alpha,
                log_alpha_delta,
                False,
                ndc_coeffs[0],
                ndc_coeffs[1],
                grad,
            )
            self.sparse_grad_indexer: Optional[torch.Tensor] = None

    def inplace_tv_temperature_grad(
        self,
        grad: torch.Tensor,
        scaling: float = 1.0,
        sparse_frac: float = 0.01,
        log_alpha: bool = False,
        log_alpha_delta: float = 2.0,
        ndc_coeffs: Tuple[float, float] = (-1.0, -1.0),
        contiguous: bool = True,
    ):
        """
        Add gradient of total variation for temperature
        directly into the gradient tensor, multiplied by 'scaling'
        """
        assert (
            _C is not None and self.temperature_data.is_cuda and grad.is_cuda
        ), "CUDA extension is currently required for total variation"

        assert not log_alpha, "No longer supported"
        rand_cells = self._get_rand_cells(sparse_frac, contiguous=contiguous)

        if rand_cells is not None:
            if rand_cells.size(0) > 0:
                indexer = self._get_sparse_temperature_grad_indexer()
                _C.tv_grad_sparse(
                    self.links,
                    self.temperature_data,
                    rand_cells,
                    indexer,
                    0,
                    1,
                    scaling,
                    log_alpha,
                    log_alpha_delta,
                    False,
                    self.opt.last_sample_opaque,
                    ndc_coeffs[0],
                    ndc_coeffs[1],
                    grad,
                )
        else:
            _C.tv_grad(
                self.links,
                self.temperature_data,
                0,
                1,
                scaling,
                log_alpha,
                log_alpha_delta,
                False,
                ndc_coeffs[0],
                ndc_coeffs[1],
                grad,
            )
            self.sparse_temperature_grad_indexer = None

    def inplace_tv_color_grad(
        self,
        grad: torch.Tensor,
        start_dim: int = 0,
        end_dim: Optional[int] = None,
        scaling: float = 1.0,
        sparse_frac: float = 0.01,
        log_alpha: bool = False,
        log_alpha_delta: float = 2.0,
        ndc_coeffs: Tuple[float, float] = (-1.0, -1.0),
        contiguous: bool = True,
    ):
        """
        Add gradient of total variation for color
        directly into the gradient tensor, multiplied by 'scaling'

        :param start_dim: int, first color channel dimension to compute TV over
         (inclusive).
                          Default 0.
        :param end_dim: int, last color channel dimension to compute TV over
        (exclusive).
                          Default None = all dimensions until the end.
        """
        assert (
            _C is not None and self.sh_data.is_cuda and grad.is_cuda
        ), "CUDA extension is currently required for total variation"
        assert not log_alpha, "No longer supported"
        if end_dim is None:
            end_dim = self.sh_data.size(1)
        end_dim = end_dim + self.sh_data.size(1) if end_dim < 0 else end_dim
        start_dim = start_dim + self.sh_data.size(1) if start_dim < 0 else start_dim

        rand_cells = self._get_rand_cells(sparse_frac, contiguous=contiguous)
        if rand_cells is not None:
            if rand_cells.size(0) > 0:
                indexer = self._get_sparse_sh_grad_indexer()

                _C.tv_grad_sparse(
                    self.links,
                    self.sh_data,
                    rand_cells,
                    indexer,
                    start_dim,
                    end_dim,
                    scaling,
                    log_alpha,
                    log_alpha_delta,
                    True,
                    False,
                    ndc_coeffs[0],
                    ndc_coeffs[1],
                    grad,
                )
        else:
            _C.tv_grad(
                self.links,
                self.sh_data,
                start_dim,
                end_dim,
                scaling,
                log_alpha,
                log_alpha_delta,
                True,
                ndc_coeffs[0],
                ndc_coeffs[1],
                grad,
            )

            self.sparse_sh_grad_indexer = None

    def inplace_l2_color_grad(
        self,
        grad: torch.Tensor,
        start_dim: int = 0,
        end_dim: Optional[int] = None,
        scaling: float = 1.0,
    ):
        """
        Add gradient of L2 regularization for color
        directly into the gradient tensor, multiplied by 'scaling'
        (no CUDA extension used)

        :param start_dim: int, first color channel dimension to compute TV over
        (inclusive).
                          Default 0.
        :param end_dim: int, last color channel dimension to compute TV over
         (exclusive).
                          Default None = all dimensions until the end.
        """
        with torch.no_grad():
            if end_dim is None:
                end_dim = self.sh_data.size(1)
            end_dim = end_dim + self.sh_data.size(1) if end_dim < 0 else end_dim
            start_dim = start_dim + self.sh_data.size(1) if start_dim < 0 else start_dim

            if self.sparse_sh_grad_indexer is None:
                scale = scaling / self.sh_data.size(0)
                grad[:, start_dim:end_dim] += scale * self.sh_data[:, start_dim:end_dim]
            else:
                indexer = self._maybe_convert_sparse_grad_indexer(sh=True)
                nz: int = (
                    torch.count_nonzero(indexer).item()
                    if indexer.dtype == torch.bool
                    else indexer.size(0)
                )
                scale = scaling / nz
                grad[indexer, start_dim:end_dim] += (
                    scale * self.sh_data[indexer, start_dim:end_dim]
                )

    def inplace_l2_regul_temperature_grad(
        self,
        scaling: float = 1.0,
    ):
        """
        Add gradient of L2 regularization for temperature
        directly into the gradient tensor, multiplied by 'scaling'
        (no CUDA extension used)

        :param start_dim: int, first color channel dimension to compute TV over
        (inclusive).
                          Default 0.
        :param end_dim: int, last color channel dimension to compute TV over
        (exclusive).
                          Default None = all dimensions until the end.
        """
        _, _, _, _, grad = self._get_data_grads()
        with torch.no_grad():

            if self.sparse_temperature_grad_indexer is None:
                scale = scaling / self.temperature_data.size(0)
                grad[:] += scale * self.temperature_data[:]
            else:
                indexer = self._maybe_convert_sparse_grad_indexer(temperature=True)
                nz: int = (
                    torch.count_nonzero(indexer).item()
                    if indexer.dtype == torch.bool
                    else indexer.size(0)
                )
                scale = scaling / nz
                grad[indexer] += scale * self.temperature_data[indexer]

    def inplace_tv_background_grad(
        self,
        grad: torch.Tensor,
        scaling: float = 1.0,
        scaling_density: Optional[float] = None,
        sparse_frac: float = 0.01,
        contiguous: bool = False,
    ):
        """
        Add gradient of total variation for color
        directly into the gradient tensor, multiplied by 'scaling'
        """
        assert (
            _C is not None and self.sh_data.is_cuda and grad.is_cuda
        ), "CUDA extension is currently required for total variation"

        rand_cells_bg = self._get_rand_cells_background(sparse_frac, contiguous)
        indexer = self._get_sparse_background_grad_indexer()
        if scaling_density is None:
            scaling_density = scaling

        _C.msi_tv_grad_sparse(
            self.background_links,
            self.background_data,
            rand_cells_bg,
            indexer,
            scaling,
            scaling_density,
            grad,
        )

    def inplace_tv_basis_grad(self, grad: torch.Tensor, scaling: float = 1.0):
        bd = self.basis_data
        tv_val = torch.mean(
            torch.sqrt(
                1e-5
                + (bd[:-1, :-1, 1:] - bd[:-1, :-1, :-1]) ** 2
                + (bd[:-1, 1:, :-1] - bd[:-1, :-1, :-1]) ** 2
                + (bd[1:, :-1, :-1] - bd[:-1, :-1, :-1]) ** 2
            ).sum(dim=-1)
        )
        tv_val_scaled = tv_val * scaling
        tv_val_scaled.backward()

    def optim_density_step(
        self,
        lr: float,
        beta: float = 0.9,
        epsilon: float = 1e-8,
        optim: str = "rmsprop",
    ):
        """
        Execute RMSprop or sgd step on density
        """
        assert (
            _C is not None and self.sh_data.is_cuda
        ), "CUDA extension is currently required for optimizers"

        indexer = self._maybe_convert_sparse_grad_indexer()
        if optim == "rmsprop":
            if (
                self.density_rms is None
                or self.density_rms.shape != self.density_data.shape
            ):
                del self.density_rms
                self.density_rms = torch.zeros_like(self.density_data.data)
            _C.rmsprop_step(
                self.density_data.data,
                self.density_rms,
                self.density_data.grad,
                indexer,
                beta,
                lr,
                epsilon,
                -1e9,
                lr,
            )
        elif optim == "sgd":
            _C.sgd_step(self.density_data.data, self.density_data.grad, indexer, lr, lr)
        else:
            raise NotImplementedError(f"Unsupported optimizer {optim}")

    def optim_temperature_step(
        self,
        lr: float,
        beta: float = 0.9,
        epsilon: float = 1e-8,
        optim: str = "rmsprop",
    ):
        """
        Execute RMSprop or sgd step on temperature
        """
        assert (
            _C is not None and self.temperature_data.is_cuda
        ), "CUDA extension is currently required for optimizers"

        indexer = self._maybe_convert_sparse_grad_indexer(temperature=True)
        if optim == "rmsprop":
            if (
                self.temperature_rms is None
                or self.temperature_rms.shape != self.temperature_data.shape
            ):
                del self.temperature_rms
                self.temperature_rms = torch.zeros_like(self.temperature_data.data)

            _C.rmsprop_step(
                self.temperature_data.data,
                self.temperature_rms,
                self.temperature_data.grad,
                indexer,
                beta,
                lr,
                epsilon,
                -1e9,
                lr,
            )
        elif optim == "sgd":
            _C.sgd_step(
                self.temperature_data.data, self.temperature_data.grad, indexer, lr, lr
            )
        else:
            raise NotImplementedError(f"Unsupported optimizer {optim}")

    def optim_sh_step(
        self,
        lr: float,
        beta: float = 0.9,
        epsilon: float = 1e-8,
        optim: str = "rmsprop",
    ):
        """
        Execute RMSprop/SGD step on SH
        """
        assert (
            _C is not None and self.sh_data.is_cuda
        ), "CUDA extension is currently required for optimizers"

        indexer = self._maybe_convert_sparse_grad_indexer(sh=True)
        if optim == "rmsprop":
            if self.sh_rms is None or self.sh_rms.shape != self.sh_data.shape:
                del self.sh_rms
                self.sh_rms = torch.zeros_like(self.sh_data.data)  # FIXME init?
            _C.rmsprop_step(
                self.sh_data.data,
                self.sh_rms,
                self.sh_data.grad,
                indexer,
                beta,
                lr,
                epsilon,
                -1e9,
                lr,
            )
        elif optim == "sgd":
            _C.sgd_step(self.sh_data.data, self.sh_data.grad, indexer, lr, lr)
        else:
            raise NotImplementedError(f"Unsupported optimizer {optim}")

    def optim_background_step(
        self,
        lr_sigma: float,
        lr_color: float,
        beta: float = 0.9,
        epsilon: float = 1e-8,
        optim: str = "rmsprop",
    ):
        """
        Execute RMSprop or sgd step on density
        """
        assert (
            _C is not None and self.sh_data.is_cuda
        ), "CUDA extension is currently required for optimizers"

        indexer = self._maybe_convert_sparse_grad_indexer(bg=True)
        n_chnl = self.background_data.size(-1)
        if optim == "rmsprop":
            if (
                self.background_rms is None
                or self.background_rms.shape != self.background_data.shape
            ):
                del self.background_rms
                self.background_rms = torch.zeros_like(self.background_data.data)
            _C.rmsprop_step(
                self.background_data.data.view(-1, n_chnl),
                self.background_rms.view(-1, n_chnl),
                self.background_data.grad.view(-1, n_chnl),
                indexer,
                beta,
                lr_color,
                epsilon,
                -1e9,
                lr_sigma,
            )
        elif optim == "sgd":
            _C.sgd_step(
                self.background_data.data.view(-1, n_chnl),
                self.background_data.grad.view(-1, n_chnl),
                indexer,
                lr_color,
                lr_sigma,
            )
        else:
            raise NotImplementedError(f"Unsupported optimizer {optim}")

    def is_cubic_pow2(self):
        """
        Check if the current grid is cubic (same in all dims) with power-of-2 size.
        This allows for conversion to svox 1 and Z-order curve (Morton code)
        """
        resolution = self.links.shape
        return (
            resolution[0] == resolution[1]
            and resolution[0] == resolution[2]
            and utils.is_pow2(resolution[0])
        )

    def _to_cpp(
        self,
        grid_coords: bool = False,
        replace_basis_data: Optional[torch.Tensor] = None,
    ):
        """
        Generate object to pass to C++
        """

        gspec = _C.SparseGridSpec()
        gspec.density_data = self.density_data
        gspec.temperature_data = self.temperature_data.cuda()
        gspec.sh_data = self.sh_data
        gspec.links = self.links

        if grid_coords:
            gspec._offset = torch.zeros_like(self._offset)
            gspec._scaling = torch.zeros_like(self._offset)
        else:
            gsz = self._grid_size()
            gspec._offset = self._offset * gsz - 0.5
            gspec._scaling = self._scaling * gsz

        gspec.basis_dim = self.basis_dim

        if self.use_background:
            gspec.background_links = self.background_links
            gspec.background_data = self.background_data
        return gspec

    def _grid_size(self):
        return torch.tensor(self.links.shape, device="cpu", dtype=torch.float32)

    def _get_data_grads(self):
        ret = []
        for subitem in [
            "density_data",
            "sh_data",
            "basis_data",
            "background_data",
            "temperature_data",
        ]:
            param = self.__getattr__(subitem)
            if not param.requires_grad:
                ret.append(torch.zeros_like(param.data))
            else:
                if (
                    not hasattr(param, "grad")
                    or param.grad is None
                    or param.grad.shape != param.data.shape
                ):
                    if hasattr(param, "grad"):
                        del param.grad
                    param.grad = torch.zeros_like(param.data)
                ret.append(param.grad)
        return ret

    def _get_sparse_grad_indexer(self):
        indexer = self.sparse_grad_indexer
        if indexer is None:
            indexer = torch.empty(
                (0,), dtype=torch.bool, device=self.density_data.device
            )
        return indexer

    def _get_sparse_temperature_grad_indexer(self):
        indexer = self.sparse_temperature_grad_indexer
        if indexer is None:
            indexer = torch.empty(
                (0,), dtype=torch.bool, device=self.density_data.device
            )
        return indexer

    def _get_sparse_sh_grad_indexer(self):
        indexer = self.sparse_sh_grad_indexer
        if indexer is None:
            indexer = torch.empty(
                (0,), dtype=torch.bool, device=self.density_data.device
            )
        return indexer

    def _get_sparse_background_grad_indexer(self):
        indexer = self.sparse_background_indexer
        if indexer is None:
            indexer = torch.empty(
                (0, 0), dtype=torch.bool, device=self.density_data.device
            )
        return indexer

    def _maybe_convert_sparse_grad_indexer(self, sh=False, bg=False, temperature=False):
        """
        Automatically convert sparse grad indexer from mask to
        indices, if it is efficient
        """
        if sh:
            indexer = self.sparse_sh_grad_indexer
        if temperature:
            indexer = self.sparse_temperature_grad_indexer
        else:
            indexer = self.sparse_grad_indexer

        if bg:
            indexer = self.sparse_background_indexer
            if indexer is not None:
                indexer = indexer.view(-1)
        if indexer is None:
            return torch.empty((), device=self.density_data.device)
        if (
            indexer.dtype == torch.bool
            and torch.count_nonzero(indexer).item() < indexer.size(0) // 8
        ):
            # Highly sparse (use index)
            indexer = torch.nonzero(indexer.flatten(), as_tuple=False).flatten()
        return indexer

    def _get_rand_cells(
        self, sparse_frac: float, force: bool = False, contiguous: bool = True
    ):
        if sparse_frac < 1.0 or force:
            assert (
                self.sparse_grad_indexer is None
                or self.sparse_grad_indexer.dtype == torch.bool
            ), "please call sparse loss after rendering and before gradient updates"
            grid_size = self.links.size(0) * self.links.size(1) * self.links.size(2)
            sparse_num = max(int(sparse_frac * grid_size), 1)

            if contiguous:
                start = np.random.randint(0, grid_size)
                arr = torch.arange(
                    start,
                    start + sparse_num,
                    dtype=torch.int32,
                    device=self.links.device,
                )

                if start > grid_size - sparse_num:
                    arr[grid_size - sparse_num - start :] -= grid_size
                return arr
            else:
                return torch.randint(
                    0,
                    grid_size,
                    (sparse_num,),
                    dtype=torch.int32,
                    device=self.links.device,
                )
        return None

    def _get_rand_cells_background(self, sparse_frac: float, contiguous: bool = True):
        assert (
            self.use_background
        ), "Can only use sparse background loss if using background"
        assert (
            self.sparse_background_indexer is None
            or self.sparse_background_indexer.dtype == torch.bool
        ), "please call sparse loss after rendering and before gradient updates"
        grid_size = (
            self.background_links.size(0)
            * self.background_links.size(1)
            * self.background_data.size(1)
        )
        sparse_num = max(int(sparse_frac * grid_size), 1)
        if contiguous:
            start = np.random.randint(0, grid_size)  # - sparse_num + 1)
            arr = torch.arange(
                start, start + sparse_num, dtype=torch.int32, device=self.links.device
            )
            if start > grid_size - sparse_num:
                arr[grid_size - sparse_num - start :] -= grid_size
            return arr
        else:
            return torch.randint(
                0, grid_size, (sparse_num,), dtype=torch.int32, device=self.links.device
            )
