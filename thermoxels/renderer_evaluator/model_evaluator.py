import json
import logging
import math
import statistics
from pathlib import Path

import imageio
import mlflow
import numpy as np
import torch
from skimage.metrics import structural_similarity

import thermoxels.svox2_temperature as svox2
from thermoxels.datasets.thermo_scene_dataset import ThermoSceneDataset
from thermoxels.model.thermoxel_trainer import ThermoxelTrainer
from thermoxels.renderer_evaluator.render_param import RenderParam
from thermoxels.renderer_evaluator.thermal_evaluation_metrics import (
    compute_thermal_metrics,
)


class Evaluator:
    def __init__(self, dataset: ThermoSceneDataset, param: RenderParam) -> None:
        self._param = param
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self.psnr_list: list[float] = []
        self.ssim_list: list[float] = []

        self.thermal_psnr_list: list[float] = []
        self.hssim_list: list[float] = []
        self.thermal_ssim_list: list[float] = []
        self.thermal_mae_list: list[float] = []
        self.mae_roi_threshold = dataset.roi_threshold
        self.t_max = dataset.t_max
        self.t_min = dataset.t_min
        self.thermal_mae_roi_list: list[float] = []

        if self._param.lpips:
            import lpips

            self._lpips_vgg = (
                lpips.LPIPS(net="vgg").eval().to(self._device)
            )  # Slow to load
            self.lpips_list: list[float] = []

        self._grid = svox2.SparseGrid.load(
            self._param.model_path,
            device=self._device,
            is_thermoxels=self._param.is_thermoxels,
        )

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

        return float(mse_num), float(psnr), float(ssim)

    def compute_lpips(self, im: torch.Tensor, im_gt: torch.Tensor) -> float:
        lpips = self._lpips_vgg(
            im_gt.permute([2, 0, 1]).contiguous(),
            im.permute([2, 0, 1]).contiguous(),
            normalize=True,
        ).item()
        return lpips

    def _compute_metrics(self, dataset: ThermoSceneDataset) -> None:
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
                )

                if not self._param.is_thermoxels:
                    # This swaps outputs to train Plenoxels on thermal images :
                    im_thermal = im.mean(axis=2).unsqueeze(-1)

                im, im_thermal = ThermoxelTrainer.process_rendered_images(
                    im, im_thermal
                )
                im_gt = dataset.gt[img_id].cpu()

                im_gt_thermal = dataset.gt_thermal[img_id].cpu().mean(axis=2)

                self._compare_and_log_metrics(
                    img_id, im, im_gt, im_thermal, im_gt_thermal
                )

                Evaluator.log_concat_image(
                    im_gt.cpu().numpy(),
                    im.cpu().numpy(),
                    f"{self._param.render_dir}/test_image_{img_id:04d}.png",
                )

                Evaluator.log_concat_image(
                    im_gt_thermal.numpy(),
                    im_thermal.cpu().numpy(),
                    f"{self._param.render_dir}/test_thermal_{img_id:04d}.png",
                )

                if self._param.imsave or self._param.vidsave:
                    concat_rgb_im = np.concatenate(
                        [im_gt.cpu().numpy(), im.cpu().numpy()], axis=1
                    )
                if self._param.imsave:
                    img_path = Path(self._param.render_dir) / Path(f"{img_id:04d}.png")
                    imageio.imwrite(img_path, concat_rgb_im)
                if self._param.vidsave:
                    video_frames.append(concat_rgb_im)

            self._compute_and_log_average_metrics()

            if not self._param.vidsave and len(video_frames):
                vid_path = self._param.render_dir + ".mp4"
                imageio.mimwrite(
                    vid_path, video_frames, fps=self._param.fps, macro_block_size=8
                )
                mlflow.log_artifact(vid_path, artifact_path="videos")

    def save_metric(self, prefix: str = "") -> None:
        all_metrics = {
            "RGB psnr": self.psnr_list,
            "RGB ssim": self.ssim_list,
            "RGB lpips": self.lpips_list if self._param.lpips else [],
            "Thermal psnr": self.thermal_psnr_list,
            "Thermal ssim": self.thermal_ssim_list,
            "Thermal mae": self.thermal_mae_list,
            "Thermal mae_roi": self.thermal_mae_roi_list,
            "Thermal hssim": self.hssim_list,
            "RGB psnr mean": statistics.fmean(self.psnr_list),
            "RGB ssim mean": statistics.fmean(self.ssim_list),
            "RGB lpips mean": statistics.fmean(self.lpips_list)
            if self._param.lpips
            else -1,
            "Thermal psnr mean": statistics.fmean(self.thermal_psnr_list),
            "Thermal ssim mean": statistics.fmean(self.thermal_ssim_list),
            "Thermal mae mean": statistics.fmean(self.thermal_mae_list),
            "Thermal mae_roi mean": statistics.fmean(self.thermal_mae_roi_list),
            "Thermal hssim mean": statistics.fmean(self.hssim_list),
        }
        mlflow.log_dict(all_metrics, str(self._param.render_dir / "metrics.json"))
        with open(self._param.metric_path / "metric.json", "w") as file:
            json.dump(all_metrics, file)

    def _compare_and_log_metrics(
        self,
        img_id: int,
        im: torch.Tensor,
        im_gt: torch.Tensor,
        im_thermal: torch.Tensor,
        im_gt_thermal: torch.Tensor,
    ) -> None:
        _, psnr_rgb, ssim_rgb = Evaluator.compute_mse_psnr_ssim(im, im_gt)
        self.psnr_list.append(psnr_rgb)
        self.ssim_list.append(ssim_rgb)

        mlflow.log_metric("RGB PSNR on test", psnr_rgb)
        mlflow.log_metric("RGB SSIM on test", ssim_rgb)
        logging.info(img_id, "RGB PSNR", psnr_rgb, "RGB SSIM", ssim_rgb)

        _, psnr_thermal, mae_thermal, mae_roi_thermal, hssim, ssim_thermal = (
            compute_thermal_metrics(
                self.t_min,
                self.t_max,
                self.mae_roi_threshold,
                im_thermal,
                im_gt_thermal,
            )
        )

        self.thermal_psnr_list.append(psnr_thermal)
        self.hssim_list.append(hssim)
        self.thermal_ssim_list.append(ssim_thermal)
        self.thermal_mae_list.append(mae_thermal)
        self.thermal_mae_roi_list.append(mae_roi_thermal)
        mlflow.log_metric("Thermal PSNR on test", psnr_thermal)
        mlflow.log_metric("Thermal HSSIM on test", hssim)
        mlflow.log_metric("tSSIM on test", ssim_thermal)
        mlflow.log_metric("Thermal MAE on test", mae_thermal)
        mlflow.log_metric("Thermal MAE_roi on test", mae_roi_thermal)

        logging.info(img_id, "Thermal PSNR", psnr_thermal, "Thermal SSIM", hssim)

        if self._param.lpips:
            lpips_i = self.compute_lpips(im.to(self._device), im_gt.to(self._device))
            self.lpips_list.append(lpips_i)

            mlflow.log_metric("RGB LPIPS on test", lpips_i)
            logging.info(img_id, "RGB LPIPS", lpips_i)

    def _compute_and_log_average_metrics(self) -> None:
        avg_psnr, std_psnr = (
            statistics.fmean(self.psnr_list),
            statistics.stdev(self.psnr_list),
        )
        mlflow.log_metric("Mean RGB PSNR on test", avg_psnr)
        mlflow.log_metric("STD RGB PSNR on test", std_psnr)

        avg_ssim, std_ssim = (
            statistics.fmean(self.ssim_list),
            statistics.stdev(self.ssim_list),
        )
        mlflow.log_metric("Mean RGB SSIM on test", avg_ssim)
        mlflow.log_metric("STD RGB SSIM on test", std_ssim)

        avg_thermal_psnr, std_thermal_psnr = (
            statistics.fmean(self.thermal_psnr_list),
            statistics.stdev(self.thermal_psnr_list),
        )

        mlflow.log_metric("Mean Thermal PSNR on test", avg_thermal_psnr)
        mlflow.log_metric("STD Thermal PSNR on test", std_thermal_psnr)

        avg_hssim, std_hssim = (
            statistics.fmean(self.hssim_list),
            statistics.stdev(self.hssim_list),
        )

        mlflow.log_metric("Mean Thermal HSSIM on test", avg_hssim)
        mlflow.log_metric("STD Thermal HSSIM on test", std_hssim)

        avg_ssim_thermal, std_ssim_thermal = (
            statistics.fmean(self.thermal_ssim_list),
            statistics.stdev(self.thermal_ssim_list),
        )

        mlflow.log_metric("Mean tSSIM on test", avg_ssim_thermal)
        mlflow.log_metric("STD tSSIM on test", std_ssim_thermal)

        avg_thermal_mae, std_thermal_mae = (
            statistics.fmean(self.thermal_mae_list),
            statistics.stdev(self.thermal_mae_list),
        )

        mlflow.log_metric("Mean Thermal MAE on test", avg_thermal_mae)
        mlflow.log_metric("STD Thermal MAE on test", std_thermal_mae)

        avg_thermal_mae_roi, std_thermal_mae_roi = (
            statistics.fmean(self.thermal_mae_roi_list),
            statistics.stdev(self.thermal_mae_roi_list),
        )

        mlflow.log_metric("Mean Thermal MAE roi on test", avg_thermal_mae_roi)
        mlflow.log_metric("STD Thermal MAE roi on test", std_thermal_mae_roi)

        if self._param.lpips:
            avg_lpips, std_lpips = (
                statistics.fmean(self.lpips_list),
                statistics.stdev(self.lpips_list),
            )
            mlflow.log_metric("Mean RGB LPIPS on test", avg_lpips)
            mlflow.log_metric("STD RGB LPIPS on test", std_lpips)

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

    @staticmethod
    def log_concat_image(im_1: np.ndarray, im_2: np.ndarray, name: str) -> None:
        concat_im = np.concatenate([im_1, im_2], axis=1)
        mlflow.log_image(np.array(concat_im), name)
