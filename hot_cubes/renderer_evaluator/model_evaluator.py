import json
import logging
import math
import statistics
import sys
from pathlib import Path

import hot_cubes.svox2_tmp as svox2
import imageio
import mlflow
import numpy as np
import torch
from skimage.metrics import structural_similarity

from hot_cubes.renderer_evaluator.render_param import RenderParam
from plenoxels.opt.util.dataset_base import DatasetBase

sys.path.append("../../plenoxels/opt")
sys.path.append("./hot_cubes")


class Evaluator:
    def __init__(self, dataset: DatasetBase, param: RenderParam) -> None:
        self._param = param
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self.psnr_list: list[float] = []
        self.ssim_list: list[float] = []

        if self._param.lpips:
            import lpips

            self._lpips_vgg = (
                lpips.LPIPS(net="vgg").eval().to(self._device)
            )  # Slow to load
            self.lpips_list: list[float] = []

        self._grid = svox2.SparseGrid.load(self._param.ckpt, device=self._device)

        if self._grid.use_background:
            if self._param.nobg:
                self._grid.background_data.data[..., -1] = 0.0
                self._param.render_dir += "_nobg"
            if self._param.nofg:
                self._grid.density_data.data[:] = 0.0
                self._param.render_dir += "_nofg"

        Evaluator._setup_render_opts_no_args(
            opt=self._grid.opt,
            step_size=self._param.step_size,
            sigma_thresh=self._param.sigma_thresh,
            stop_thresh=self._param.stop_thresh,
            background_brightness=self._param.background_brightness,
            renderer_backend=self._param.renderer_backend,
            random_sigma_std=self._param.random_sigma_std,
            random_sigma_std_background=self._param.random_sigma_std_background,
            last_sample_opaque=self._param.last_sample_opaque,
            near_clip=self._param.near_clip,
            use_spheric_clip=self._param.use_spheric_clip,
        )

        if self._param.blackbg:
            self._param.render_dir += "_blackbg"
            self._grid.opt.background_brightness = 0.0

        Path(self._param.render_dir).mkdir(parents=True, exist_ok=True)
        self._metrics = self._compute_metrics(dataset)

    @staticmethod
    def compute_mse_psnr_ssim(
        im: torch.Tensor, im_gt: torch.Tensor
    ) -> tuple[float, float, float]:

        mse = (im - im_gt) ** 2
        mse_num: float = mse.mean().item()
        psnr = -10.0 * math.log10(mse_num)

        ssim = structural_similarity(
            im_gt.cpu().numpy(), im.cpu().numpy(), channel_axis=2, data_range=1
        )

        return mse_num, psnr, ssim

    def compute_lpips(self, im: torch.Tensor, im_gt: torch.Tensor) -> float:
        lpips = self._lpips_vgg(
            im_gt.permute([2, 0, 1]).contiguous(),
            im.permute([2, 0, 1]).contiguous(),
            normalize=True,
        ).item()
        return lpips

    def _compute_metrics(self, dataset: DatasetBase) -> dict[str, float]:
        # NOTE: no_grad enables the fast image-level rendering kernel
        # for cuvol backend only
        # other backends will manually generate rays per frame (slow)
        with torch.no_grad():

            img_eval_interval = max(dataset.n_images // self._param.n_eval, 1)
            c2ws = dataset.c2w.to(device=self._device)
            video_frames = []

            for img_id in range(0, dataset.n_images, img_eval_interval):

                dset_h, dset_w = dataset.get_image_size(img_id)
                w = (
                    dset_w
                    if self._param.crop == 1.0
                    else int(dset_w * self._param.crop)
                )
                h = (
                    dset_h
                    if self._param.crop == 1.0
                    else int(dset_h * self._param.crop)
                )

                cam = svox2.Camera(
                    c2ws[img_id],
                    dataset.intrins.get("fx", img_id),
                    dataset.intrins.get("fy", img_id),
                    dataset.intrins.get("cx", img_id) + (w - dset_w) * 0.5,
                    dataset.intrins.get("cy", img_id) + (h - dset_h) * 0.5,
                    w,
                    h,
                    ndc_coeffs=dataset.ndc_coeffs,
                )
                im, im_thermal = self._grid.volume_render_image(
                    cam,
                    use_kernel=True,
                    return_raylen=self._param.ray_len,
                    include_temperature=True,
                )

                im.clamp(0.0, 1.0)
                im_gt = dataset.gt[img_id].to(device=self._device)

                im_thermal.clamp(0.0, 1.0)
                im_gt_thermal = dataset.gt_thermal[img_id].to(device=self._device)

                _, psnr_rgb, ssim_rgb = Evaluator.compute_mse_psnr_ssim(im, im_gt)
                self.psnr_list.append(psnr_rgb)
                self.ssim_list.append(ssim_rgb)

                mlflow.log_metric("PSNR on test", psnr_rgb)
                mlflow.log_metric("SSIM on test", ssim_rgb)
                logging.info(img_id, "PSNR", psnr_rgb, "SSIM", ssim_rgb)

                if self._param.lpips:
                    lpips_i = self.compute_lpips(im, im_gt)
                    self.lpips_list.append(lpips_i)

                    mlflow.log_metric("LPIPS on test", lpips_i)
                    logging.info(img_id, "LPIPS", lpips_i)

                im = im.cpu().numpy()
                concat_rgb_im = np.concatenate([im_gt.cpu().numpy(), im], axis=1)
                concat_rgb_im = (concat_rgb_im * 255).astype(np.uint8)

                im_thermal = im_thermal.cpu().numpy()

                concat_im_thermal = np.concatenate(
                    [im_gt_thermal.cpu().numpy().mean(axis=2), im_thermal.squeeze(-1)],
                    axis=1,
                )
                concat_im_thermal = (concat_im_thermal * 255).astype(np.uint8)

                mlflow.log_image(concat_rgb_im, f"outputs/test_image_{img_id:04d}.png")
                mlflow.log_image(
                    concat_im_thermal, f"outputs/test_thermal_{img_id:04d}.png"
                )

                if self._param.imsave:
                    img_path = Path(self._param.render_dir) / Path(f"{img_id:04d}.png")
                    imageio.imwrite(img_path, concat_rgb_im)
                if self._param.vidsave:
                    video_frames.append(concat_rgb_im)

            avg_psnr, std_psnr = statistics.fmean(self.psnr_list), statistics.stdev(
                self.psnr_list
            )
            mlflow.log_metric("AVERAGE PSNR on test", avg_psnr)
            mlflow.log_metric("STD PSNR on test", std_psnr)

            avg_ssim, std_ssim = statistics.fmean(self.ssim_list), statistics.stdev(
                self.ssim_list
            )
            mlflow.log_metric("AVERAGE SSIM on test", avg_ssim)
            mlflow.log_metric("STD SSIM on test", std_ssim)

            if self._param.lpips:
                avg_lpips, std_lpips = statistics.fmean(
                    self.lpips_list
                ), statistics.stdev(self.lpips_list)
                mlflow.log_metric("AVERAGE LPIPS on test", avg_lpips)
                mlflow.log_metric("STD LPIPS on test", std_lpips)

            if not self._param.vidsave and len(video_frames):
                vid_path = self._param.render_dir + ".mp4"
                imageio.mimwrite(
                    vid_path, video_frames, fps=self._param.fps, macro_block_size=8
                )
                mlflow.log_artifact(vid_path, artifact_path="videos")

    def save_metric(self, log_only: bool = False) -> None:
        all_metrics = {
            "psnr": self.psnr_list,
            "ssim": self.ssim_list,
            "lpips": self.lpips_list if self._param.lpips else [],
        }
        mlflow.log_dict(all_metrics, "metrics")
        if log_only:
            return
        with open(self._param.metric_path + "test_metric.json", "w") as file:
            json.dump(all_metrics, file)

    @staticmethod
    def _setup_render_opts_no_args(
        opt,
        step_size: float,
        sigma_thresh: float,
        stop_thresh: float,
        background_brightness: float,
        renderer_backend: str,
        random_sigma_std: float,
        random_sigma_std_background: float,
        last_sample_opaque: bool,
        near_clip: float,
        use_spheric_clip: bool,
    ) -> None:
        """
        Pass render arguments to the SparseGrid renderer options
        """
        opt.step_size = step_size
        opt.sigma_thresh = sigma_thresh
        opt.stop_thresh = stop_thresh
        opt.background_brightness = background_brightness
        opt.backend = renderer_backend
        opt.random_sigma_std = random_sigma_std
        opt.random_sigma_std_background = random_sigma_std_background
        opt.last_sample_opaque = last_sample_opaque
        opt.near_clip = near_clip
        opt.use_spheric_clip = use_spheric_clip
