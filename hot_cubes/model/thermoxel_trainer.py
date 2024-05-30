import gc
import json
import logging
import math
import sys
from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.cuda
import torch.optim
from tqdm import tqdm

import hot_cubes.svox2_tmp as svox2
from hot_cubes.model.training_param import Param
from hot_cubes.renderer_evaluator.model_evaluator import Evaluator
from hot_cubes.renderer_evaluator.render_param import RenderParam
from plenoxels.opt.util import config_util
from plenoxels.opt.util.dataset import datasets
from plenoxels.opt.util.dataset_base import DatasetBase
from plenoxels.opt.util.util import get_expon_lr_func, viridis_cmap

sys.path.append(".")
sys.path.append("./hot_cubes")
device = "cuda" if torch.cuda.is_available() else "cpu"


class ThermoxelTrainer:
    def __init__(
        self, dataset: DatasetBase, dataset_val: DatasetBase, param: Param
    ) -> None:
        self._param = param
        self.dataset = dataset
        self.dataset_val = dataset_val

        self.resolution_list = json.loads(self._param.reso)
        resolution_id = 0

        self.grid = svox2.SparseGrid(
            reso=self.resolution_list[resolution_id],
            center=dataset.scene_center,
            radius=dataset.scene_radius,
            use_sphere_bound=dataset.use_sphere_bound and not self._param.nosphereinit,
            basis_dim=self._param.sh_dim,
            use_z_order=True,
            device=device,
            basis_reso=self._param.basis_reso,
            basis_type=svox2.__dict__["BASIS_TYPE_" + self._param.basis_type.upper()],
            mlp_posenc_size=self._param.mlp_posenc_size,
            mlp_width=self._param.mlp_width,
            background_nlayers=self._param.background_nlayers,
            background_reso=self._param.background_reso,
        )

        self._lr_sigma_func = get_expon_lr_func(
            self._param.lr_sigma,
            self._param.lr_sigma_final,
            self._param.lr_sigma_delay_steps,
            self._param.lr_sigma_delay_mult,
            self._param.lr_sigma_decay_steps,
        )
        self._lr_sh_func = get_expon_lr_func(
            self._param.lr_sh,
            self._param.lr_sh_final,
            self._param.lr_sh_delay_steps,
            self._param.lr_sh_delay_mult,
            self._param.lr_sh_decay_steps,
        )

        self._lr_sigma_bg_func = get_expon_lr_func(
            self._param.lr_sigma_bg,
            self._param.lr_sigma_bg_final,
            self._param.lr_sigma_bg_delay_steps,
            self._param.lr_sigma_bg_delay_mult,
            self._param.lr_sigma_bg_decay_steps,
        )
        self._lr_color_bg_func = get_expon_lr_func(
            self._param.lr_color_bg,
            self._param.lr_color_bg_final,
            self._param.lr_color_bg_delay_steps,
            self._param.lr_color_bg_delay_mult,
            self._param.lr_color_bg_decay_steps,
        )

        self.ckpt_path = Path(self._param.train_dir) / Path("ckpt.npz")

    def optimize(
        self,
        param: Param,
        dataset: datasets,
        factor: float,
        dataset_val: datasets,
        use_sparsify: bool = True,
    ) -> None:

        # DC -> gray; mind the SH scaling!
        self.grid.sh_data.data[:] = 0.0
        self.grid.density_data.data[:] = (
            0.0 if param.lr_fg_begin_step > 0 else param.init_sigma
        )

        if self.grid.use_background:
            self.grid.background_data.data[..., -1] = param.init_sigma_bg

        self.grid.requires_grad_(True)
        config_util.setup_render_opts(self.grid.opt, param)
        logging.info("Render options", self.grid.opt)

        resolution_id = 0

        resample_cameras = [
            svox2.Camera(
                c2w.to(device=device),
                dataset.intrins.get("fx", i),
                dataset.intrins.get("fy", i),
                dataset.intrins.get("cx", i),
                dataset.intrins.get("cy", i),
                width=dataset.get_image_size(i)[1],
                height=dataset.get_image_size(i)[0],
                ndc_coeffs=dataset.ndc_coeffs,
            )
            for i, c2w in enumerate(dataset.c2w)
        ]

        last_upsamp_step = param.init_iters

        if param.enable_random:
            logging.warn(
                "Randomness is enabled for training "
                "(normal for LLFF & scenes with background)"
            )

        for gstep_id_base in range(param.n_epoch):
            dataset.shuffle_rays()
            # epoch_size = dataset.rays.origins.size(0)
            # batches_per_epoch = (epoch_size - 1) // param.batch_size + 1

            self.train_step(
                gstep_id_base=gstep_id_base,
            )
            gc.collect()
            # gstep_id_base += batches_per_epoch

            # Overwrite prev checkpoints since they are very huge
            if (
                param.save_every > 0
                and (gstep_id_base + 1) % max(factor, param.save_every) == 0
                and not param.tune_mode
            ):
                logging.info("Saving", self.ckpt_path)
                self.grid.save(self.ckpt_path)

            if (gstep_id_base - last_upsamp_step) < param.upsamp_every:
                continue

            last_upsamp_step = gstep_id_base

            if resolution_id >= len(self.resolution_list) - 1:
                continue
            resolution_id += 1

            logging.info(
                "* Upsampling from",
                self.resolution_list[resolution_id],
                "to",
                self.resolution_list[resolution_id + 1],
            )
            if param.tv_early_only > 0:
                logging.info("turning off TV regularization")
                param.lambda_tv = 0.0
                param.lambda_tv_sh = 0.0
            elif param.tv_decay != 1.0:
                param.lambda_tv *= param.tv_decay
                param.lambda_tv_sh *= param.tv_decay

            z_reso = (
                self.resolution_list[resolution_id]
                if isinstance(self.resolution_list[resolution_id], int)
                else self.resolution_list[resolution_id][2]
            )

            self.grid.resample(
                reso=self.resolution_list[resolution_id],
                sigma_thresh=param.density_thresh,
                weight_thresh=(param.weight_thresh / z_reso if use_sparsify else 0.0),
                dilate=2,  # use_sparsify,
                cameras=(resample_cameras if param.thresh_type == "weight" else None),
                max_elements=param.max_self.grid_elements,
            )

            if self.grid.use_background and resolution_id <= 1:
                self.grid.sparsify_background(param.background_density_thresh)

            if param.upsample_density_add:
                self.grid.density_data.data[:] += param.upsample_density_add

            if factor > 1:
                logging.info(
                    "* Using higher resolution images due to large grid; new factor",
                    factor,
                )
                factor //= 2
                dataset.gen_rays(factor=factor)
                dataset.shuffle_rays()

        logging.info("* Final eval and save")

        self.eval_step()
        if not param.tune_nosave:
            self.grid.save(self.ckpt_path)
            self.test_step()

    def train_step(
        self,
        gstep_id_base: int,
    ) -> None:
        epoch_size = self.dataset.rays.origins.size(0)
        batches_per_epoch = (epoch_size - 1) // self._param.batch_size + 1

        lr_sigma_factor, lr_sh_factor, lr_basis_factor = 1.0, 1.0, 1.0

        logging.info("Train step")

        pbar = tqdm(
            enumerate(range(0, epoch_size, self._param.batch_size)),
            total=batches_per_epoch,
        )

        train_stats = {"Train_mse": 0.0, "Train_psnr": 0.0, "Train_invsqr_mse": 0.0}

        for iter_id, batch_begin in pbar:
            gstep_id = iter_id + gstep_id_base * batches_per_epoch
            if (
                self._param.lr_fg_begin_step > 0
                and gstep_id == self._param.lr_fg_begin_step
            ):
                self.grid.density_data.data[:] = self._param.init_sigma
            lr_sigma = self._lr_sigma_func(gstep_id) * lr_sigma_factor
            lr_sh = self._lr_sh_func(gstep_id) * lr_sh_factor
            lr_sigma_bg = (
                self._lr_sigma_bg_func(gstep_id - self._param.lr_basis_begin_step)
                * lr_basis_factor
            )
            lr_color_bg = (
                self._lr_color_bg_func(gstep_id - self._param.lr_basis_begin_step)
                * lr_basis_factor
            )
            if not self._param.lr_decay:
                lr_sigma = self._param.lr_sigma * lr_sigma_factor
                lr_sh = self._param.lr_sh * lr_sh_factor

            batch_end = min(batch_begin + self._param.batch_size, epoch_size)
            batch_origins = self.dataset.rays.origins[batch_begin:batch_end]
            batch_dirs = self.dataset.rays.dirs[batch_begin:batch_end]

            # Get ground truths
            rgb_gt = self.dataset.rays.gt[batch_begin:batch_end]
            thermal_gt = self.dataset.rays.gt_thermal[batch_begin:batch_end]

            #  ThermalRays is used
            rays = svox2.Rays(batch_origins, batch_dirs)

            rgb_pred, temp_pred = self.grid.volume_render_fused_rgbt(
                rays, rgb_gt, thermal_gt, t_loss=self._param.t_loss
            )

            _, rgb_mse, rgb_psnr = ThermoxelTrainer.compute_mse_psnr(rgb_gt, rgb_pred)
            ThermoxelTrainer.update_stats_with_mse_psnr(rgb_mse, rgb_psnr, train_stats)

            _, thermal_mse, thermal_psnr = ThermoxelTrainer.compute_mse_psnr(
                thermal_gt, temp_pred
            )

            # Stats
            log_every = self._param.epoch_size // self._param.log_per_epoch
            if (iter_id + 1) % log_every == 0:
                # Print averaged stats
                pbar.set_description(f"epoch {gstep_id_base} psnr={rgb_psnr:.2f}")

                for stat_name in train_stats:
                    stat_val = train_stats[stat_name] / log_every
                    mlflow.log_metric("train" + stat_name, stat_val)
                    train_stats[stat_name] = 0.0

                if self._param.weight_decay_sh < 1.0:
                    self.grid.sh_data.data *= self._param.weight_decay_sigma
                if self._param.weight_decay_sigma < 1.0:
                    self.grid.density_data.data *= self._param.weight_decay_sh

            self.add_regularizers(
                gstep_id,
                lr_sigma,
                lr_sh,
                lr_sigma_bg,
                lr_color_bg,
            )

    def add_regularizers(
        self,
        gstep_id: int,
        lr_sigma: float,
        lr_sh: float,
        lr_sigma_bg: float,
        lr_color_bg: float,
    ) -> None:
        # Apply TV/Sparsity regularizers
        if self._param.lambda_tv > 0.0:
            #  with Timing("tv_inpl"):
            self.grid.inplace_tv_grad(
                self.grid.density_data.grad.to(torch.float32),
                scaling=self._param.lambda_tv,
                sparse_frac=self._param.tv_sparsity,
                logalpha=self._param.tv_logalpha,
                ndc_coeffs=self.dataset.ndc_coeffs,
                contiguous=self._param.tv_contiguous,
            )
        if self._param.lambda_tv_sh > 0.0:
            #  with Timing("tv_color_inpl"):
            self.grid.inplace_tv_color_grad(
                self.grid.sh_data.grad,
                scaling=self._param.lambda_tv_sh,
                sparse_frac=self._param.tv_sh_sparsity,
                ndc_coeffs=self.dataset.ndc_coeffs,
                contiguous=self._param.tv_contiguous,
            )

        # Manual SGD/rmsprop step
        if gstep_id >= self._param.lr_fg_begin_step:
            self.grid.optim_density_step(
                lr_sigma, beta=self._param.rms_beta, optim=self._param.sigma_optim
            )
            self.grid.optim_sh_step(
                lr_sh, beta=self._param.rms_beta, optim=self._param.sh_optim
            )

        if self.grid.use_background:
            self.grid.optim_background_step(
                lr_sigma_bg,
                lr_color_bg,
                beta=self._param.rms_beta,
                optim=self._param.bg_optim,
            )

    def eval_step(self, to_log: bool = True) -> None:
        """
        Evaluate the model on the validation set
        """
        # Put in a function to avoid memory leak
        logging.info("Eval step")
        with torch.no_grad():
            stats_val = {"Eval_psnr": 0.0, "Eval_mse": 0.0}
            img_ids = range(0, self.dataset_val.n_images, 1)
            for i, img_id in tqdm(enumerate(img_ids), total=len(img_ids)):
                # rgb_pred_val = rgb_gt_val = None

                c2w = self.dataset_val.c2w[img_id].to(device=device)
                cam = svox2.Camera(
                    c2w,
                    self.dataset_val.intrins.get("fx", img_id),
                    self.dataset_val.intrins.get("fy", img_id),
                    self.dataset_val.intrins.get("cx", img_id),
                    self.dataset_val.intrins.get("cy", img_id),
                    width=self.dataset_val.get_image_size(img_id)[1],
                    height=self.dataset_val.get_image_size(img_id)[0],
                    ndc_coeffs=self.dataset_val.ndc_coeffs,
                )

                rgb_pred_val, thermal_pred_val = self.grid.volume_render_image(
                    cam,
                    use_kernel=True,
                    include_temperature=self._param.include_temperature,
                )

                rgb_gt_val = self.dataset_val.gt[img_id].cpu()
                thermal_gt_val = self.dataset_val.gt_thermal[img_id].cpu().mean(axis=2)

                rgb_pred_val, thermal_pred_val = (
                    ThermoxelTrainer.process_rendered_images(
                        rgb_pred_val, thermal_pred_val
                    )
                )

                all_mses_rgb, mse_rgb_num, rgb_psnr = ThermoxelTrainer.compute_mse_psnr(
                    rgb_gt_val, rgb_pred_val
                )
                all_mses_thermal, mse_thermal_num, thermal_psnr = (
                    ThermoxelTrainer.compute_mse_psnr(rgb_gt_val, rgb_pred_val)
                )

                ThermoxelTrainer.update_stats_with_mse_psnr(
                    mse_rgb_num, rgb_psnr, stats_val
                )

                if not to_log:
                    continue

                ThermoxelTrainer._log_concat_image(
                    rgb_gt_val.numpy(),
                    rgb_pred_val,
                    f"outputs/val_image_{img_id:04d}.png",
                )

                ThermoxelTrainer._log_concat_image(
                    thermal_gt_val.cpu(),
                    thermal_pred_val,
                    f"outputs/val_thermal_image_{img_id:04d}.png",
                )

                if self._param.log_mse_image:
                    mse_rgb_img = all_mses_rgb / all_mses_rgb.max()
                    mlflow.log_image(
                        np.array(mse_rgb_img),
                        f"outputs/val_mse_image" f"_{img_id:04d}.png",
                    )

                    mse_thermal = all_mses_thermal / all_mses_thermal.max()

                    mlflow.log_image(
                        np.array(mse_thermal),
                        f"outputs/val_mse_thermal_image" f"" f"_{img_id:04d}.png",
                    )
                if self._param.log_depth_map:
                    depth_img = self.grid.volume_render_depth_image(
                        cam,
                        (
                            self._param.log_depth_map_use_thresh
                            if self._param.log_depth_map_use_thresh
                            else None
                        ),
                    )
                    depth_img = viridis_cmap(depth_img.cpu())

                    ThermoxelTrainer._log_concat_image(
                        rgb_gt_val.cpu().numpy(),
                        depth_img,
                        f"outputs/val_depth_map_{img_id:04d}.png",
                    )

            stats_val["Eval_mse"] /= self.dataset_val.n_images
            stats_val["Eval_psnr"] /= self.dataset_val.n_images
            for stat_name in stats_val:
                mlflow.log_metric(stat_name, stats_val[stat_name])

            logging.info("eval stats:", stats_val)

    def test_step(self) -> None:
        render_param = RenderParam(
            ckpt=self.ckpt_path,
            data_dir=self._param.data_dir,
            render_dir="./",
            nobg=False,
            dataset_type="auto",
            train=True,
        )
        dataset_test = datasets[self._param.dataset_type](
            self._param.data_dir,
            split="test",
            **{
                "dataset_type": self._param.dataset_type,
                "seq_id": self._param.seq_id,
                "epoch_size": self._param.epoch_size,  # batch_size
                "white_bkgd": self._param.white_bkgd,
                "hold_every": 8,
                "normalize_by_bbox": self._param.normalize_by_bbox,
                "data_bbox_scale": self._param.data_bbox_scale,
                "cam_scale_factor": self._param.cam_scale_factor,
                "normalize_by_camera": self._param.normalize_by_camera,
                "permutation": False,
            },
        )

        evaluator = Evaluator(param=render_param, dataset=dataset_test)
        evaluator.save_metric(log_only=True)

    @staticmethod
    def process_rendered_images(
        rgb_pred: torch.tensor, thermal_pred: torch.tensor
    ) -> tuple[torch.tensor, torch.tensor]:
        rgb_pred.clamp_max_(1.0)
        thermal_pred = torch.squeeze(thermal_pred, dim=2)

        return rgb_pred.cpu(), thermal_pred.cpu()

    @staticmethod
    def _log_concat_image(im_1: np.ndarray, im_2: np.ndarray, name) -> None:
        concat_im = np.concatenate([im_1, im_2], axis=1)
        mlflow.log_image(np.array(concat_im), name)

    @staticmethod
    def compute_mse_psnr(im: torch.tensor, im_gt: torch.tensor) -> None:
        mse_map = (im.cpu() - im_gt.cpu()) ** 2
        mse_num = mse_map.mean().item()
        psnr = -10.0 * math.log10(mse_num)
        return mse_map, mse_num, psnr

    @staticmethod
    def update_stats_with_mse_psnr(
        mse_num: float, psnr: float, stats: dict[float]
    ) -> None:

        if math.isnan(psnr):
            raise RuntimeError("NAN PSNR")

        for stat_name in stats:
            if stat_name.endswith("_mse"):
                stats[stat_name] += mse_num
            if stat_name.endswith("_psnr"):
                stats[stat_name] += psnr
            if stat_name.endswith("_invsqr_mse"):
                stats[stat_name] += 1.0 / mse_num**2
