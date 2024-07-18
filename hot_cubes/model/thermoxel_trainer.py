import gc
import json
import logging
import sys
from pathlib import Path

import hot_cubes.svox2_temperature as svox2
import mlflow
import numpy as np
import torch
import torch.cuda
import torch.optim
from tqdm import tqdm

from hot_cubes.model.training_param import Param
from hot_cubes.renderer_evaluator.model_evaluator import Evaluator
from hot_cubes.renderer_evaluator.render_param import RenderParam
from hot_cubes.renderer_evaluator.thermal_evaluation_metrics import (
    compute_thermal_metric_maps,
    compute_psnr,
)
from plenoxels.opt.util import config_util
from plenoxels.opt.util.dataset import datasets
from plenoxels.opt.util.dataset_base import DatasetBase
from plenoxels.opt.util.util import get_expon_lr_func, viridis_cmap

sys.path.append(".")
sys.path.append("./hot_cubes")
device = "cuda" if torch.cuda.is_available() else "cpu"
# logging.basicConfig(level=logging.INFO)


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
            radius=self._param.scene_radius,
            basis_dim=self._param.sh_dim,
            use_z_order=True,
            device=device,
            basis_reso=self._param.basis_reso,
            basis_type=svox2.__dict__["BASIS_TYPE_" + self._param.basis_type.upper()],
            mlp_posenc_size=self._param.mlp_posenc_size,
            mlp_width=self._param.mlp_width,
            background_nlayers=self._param.background_nlayers,
            background_reso=self._param.background_reso,
            include_temperature=self._param.include_temperature,
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
        self._lr_temperature_func = get_expon_lr_func(
            self._param.lr_temperature,
            self._param.lr_temperature_final,
            self._param.lr_temperature_delay_steps,
            self._param.lr_temperature_delay_mult,
            self._param.lr_temperature_decay_steps,
        )

        self.ckpt_path = Path(self._param.train_dir) / Path("ckpt.npz")

    def optimize(
        self,
        factor: float,
        use_sparsify: bool = True,
    ) -> None:

        # DC -> gray; mind the SH scaling!
        self.grid.sh_data.data[:] = 0.0
        self.grid.density_data.data[:] = (
            0.0 if self._param.lr_fg_begin_step > 0 else self._param.init_sigma
        )

        if self.grid.use_background:
            self.grid.background_data.data[..., -1] = self._param.init_sigma_bg

        self.grid.requires_grad_(True)
        config_util.setup_render_opts(self.grid.opt, self._param)
        logging.info("Render options", self.grid.opt)

        resolution_id = 0

        resample_cameras = [
            svox2.Camera(
                c2w.to(device=device),
                self.dataset.intrins.get("fx", i),
                self.dataset.intrins.get("fy", i),
                self.dataset.intrins.get("cx", i),
                self.dataset.intrins.get("cy", i),
                width=self.dataset.get_image_size(i)[1],
                height=self.dataset.get_image_size(i)[0],
                ndc_coeffs=self.dataset.ndc_coeffs,
            )
            for i, c2w in enumerate(self.dataset.c2w)
        ]

        last_upsamp_step = self._param.init_iters

        if self._param.enable_random:
            logging.warn(
                "Randomness is enabled for training "
                "(normal for LLFF & scenes with background)"
            )

        for global_step_id_base in range(self._param.n_epoch):
            logging.info("reso_id", resolution_id)
            self.dataset.shuffle_rays()

            self.train_step(
                global_step_id_base=global_step_id_base,
            )
            gc.collect()

            # Overwrite prev checkpoints since they are very huge
            if (
                self._param.save_every > 0
                and (global_step_id_base + 1) % max(factor, self._param.save_every) == 0
                and not self._param.tune_mode
            ):
                logging.info("Saving", self.ckpt_path)
                self.grid.save(self.ckpt_path)

            if (global_step_id_base - last_upsamp_step) < self._param.upsamp_every:
                continue

            last_upsamp_step = global_step_id_base

            if resolution_id >= len(self.resolution_list) - 1:
                continue

            logging.info(
                "* Upsampling from",
                self.resolution_list[resolution_id],
                "to",
                self.resolution_list[resolution_id + 1],
            )
            resolution_id += 1
            if self._param.tv_early_only > 0:
                logging.info("turning off TV regularization")
                self._param.lambda_tv = 0.0
                self._param.lambda_tv_sh = 0.0
            elif self._param.tv_decay != 1.0:
                self._param.lambda_tv *= self._param.tv_decay
                self._param.lambda_tv_sh *= self._param.tv_decay

            z_reso = (
                self.resolution_list[resolution_id]
                if isinstance(self.resolution_list[resolution_id], int)
                else self.resolution_list[resolution_id][2]
            )

            self.grid.resample(
                reso=self.resolution_list[resolution_id],
                sigma_thresh=self._param.density_thresh,
                weight_thresh=(
                    self._param.weight_thresh / z_reso if use_sparsify else 0.0
                ),
                dilate=2,  # use_sparsify,
                cameras=(
                    resample_cameras if self._param.thresh_type == "weight" else None
                ),
                max_elements=self._param.max_grid_elements,
            )

            if self.grid.use_background and resolution_id <= 1:
                self.grid.sparsify_background(self._param.background_density_thresh)

            if self._param.upsample_density_add:
                self.grid.density_data.data[:] += self._param.upsample_density_add

            if factor > 1:
                logging.info(
                    "* Using higher resolution images due to large grid; new factor",
                    factor,
                )
                factor //= 2
                self.dataset.gen_rays(factor=factor)
                self.dataset.shuffle_rays()

        logging.info("* Final eval and save ")

        self.eval_step()
        if not self._param.tune_nosave:
            self.grid.save(self.ckpt_path)
            mlflow.log_artifact(self.ckpt_path)
            self.test_step()

    def train_step(
        self,
        global_step_id_base: int,
    ) -> None:
        epoch_size = self.dataset.rays.origins.size(0)
        batches_per_epoch = (epoch_size - 1) // self._param.batch_size + 1

        lr_sigma_factor = 1.0
        lr_sh_factor = 1.0
        lr_basis_factor = 1.0
        lr_temperature_factor = 1.0

        logging.info("Train step")

        pbar = tqdm(
            enumerate(range(0, epoch_size, self._param.batch_size)),
            total=batches_per_epoch,
        )

        train_stats = {
            "Train_rgb_mse": 0.0,
            "Train_rgb_psnr": 0.0,
            "Train_invsqr_rgb_mse": 0.0,
            "Train_thermal_mse": 0.0,
            "Train_thermal_mae": 0.0,
            "Train_thermal_psnr": 0.0,
            "Train_invsqr_thermal_mse": 0.0,
        }

        for iter_id, batch_begin in pbar:
            global_step_id = iter_id + global_step_id_base * batches_per_epoch
            if (
                self._param.lr_fg_begin_step > 0
                and global_step_id == self._param.lr_fg_begin_step
            ):
                self.grid.density_data.data[:] = self._param.init_sigma
            lr_sigma = self._lr_sigma_func(global_step_id) * lr_sigma_factor
            lr_sh = self._lr_sh_func(global_step_id) * lr_sh_factor
            lr_temperature = (
                self._lr_temperature_func(global_step_id) * lr_temperature_factor
            )
            lr_sigma_bg = (
                self._lr_sigma_bg_func(global_step_id - self._param.lr_basis_begin_step)
                * lr_basis_factor
            )
            lr_color_bg = (
                self._lr_color_bg_func(global_step_id - self._param.lr_basis_begin_step)
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
                rays=rays,
                rgb_gt=rgb_gt,
                temp_gt=thermal_gt,
                t_loss=self._param.t_loss,
                beta_loss=self._param.lambda_beta,
                sparsity_loss=self._param.lambda_sparsity,
                density_threshold=self._param.density_thresh,
                t_surface_loss=self._param.t_surface_loss,
                l1_loss=self._param.l1_loss,
            )

            _, rgb_mse, rgb_psnr = ThermoxelTrainer.compute_mse_psnr(rgb_gt, rgb_pred)
            ThermoxelTrainer._update_rgb_stats(rgb_mse, rgb_psnr, train_stats)

            _, thermal_mse, thermal_psnr = ThermoxelTrainer.compute_mse_psnr(
                thermal_gt, temp_pred
            )

            thermal_mae = torch.abs(thermal_gt - temp_pred).mean().item()

            ThermoxelTrainer._update_thermal_stats(
                thermal_mse, thermal_psnr, None, None, thermal_mae, None, train_stats
            )

            # Stats
            log_every = self._param.epoch_size // self._param.log_per_epoch
            if (iter_id + 1) % log_every == 0:
                # Print averaged stats
                pbar.set_description(
                    f"epoch {global_step_id_base} RGB-psnr={rgb_psnr:.2f} "
                    f"Thermal-psnr={thermal_psnr:.2f}"
                )

                for stat_name in train_stats:
                    stat_val = train_stats[stat_name] / log_every
                    mlflow.log_metric(stat_name, stat_val)
                    train_stats[stat_name] = 0.0
            if (iter_id + 1) % self._param.print_every == 0:
                if self._param.weight_decay_sh < 1.0:
                    self.grid.sh_data.data *= self._param.weight_decay_sigma
                if self._param.weight_decay_sigma < 1.0:
                    self.grid.density_data.data *= self._param.weight_decay_sh

            if global_step_id_base <= self._param.freeze_rgb_after:
                self._add_rgb_regularizers(
                    global_step_id, lr_sigma, lr_sh, lr_sigma_bg, lr_color_bg
                )

            self._add_thermal_regularizers(global_step_id, lr_temperature)

    def _add_thermal_regularizers(
        self,
        global_step_id: int,
        lr_temperature: float,
    ) -> None:

        if self._param.lambda_tv_temp > 0.0:
            self.grid.inplace_tv_temperature_grad(
                self.grid.temperature_data.grad,
                scaling=self._param.lambda_tv_temp,
                sparse_frac=self._param.tv_temp_sparsity,
                ndc_coeffs=self.dataset.ndc_coeffs,
                contiguous=self._param.tv_contiguous,
            )
        # Manual SGD/rmsprop step
        if global_step_id >= self._param.lr_fg_begin_step:
            self.grid.optim_temperature_step(
                lr_temperature,
                beta=self._param.rms_beta,
                optim=self._param.temp_optim,
            )

    def _add_rgb_regularizers(
        self,
        global_step_id: int,
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
            self.grid.inplace_tv_color_grad(
                self.grid.sh_data.grad,
                scaling=self._param.lambda_tv_sh,
                sparse_frac=self._param.tv_sh_sparsity,
                ndc_coeffs=self.dataset.ndc_coeffs,
                contiguous=self._param.tv_contiguous,
            )

        if (
            self._param.lambda_tv_background_sigma > 0.0
            or self._param.lambda_tv_background_color > 0.0
        ):
            self.grid.inplace_tv_background_grad(
                self.grid.background_data.grad,
                scaling=self._param.lambda_tv_background_color,
                scaling_density=self._param.lambda_tv_background_sigma,
                sparse_frac=self._param.tv_background_sparsity,
                contiguous=self._param.tv_contiguous,
            )

        # Manual SGD/rmsprop step
        if global_step_id >= self._param.lr_fg_begin_step:
            self.grid.optim_density_step(
                lr_sigma,
                beta=self._param.rms_beta,
                optim=self._param.sigma_optim,
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
            stats_val = {
                "Eval_mean_rgb_psnr": 0.0,
                "Eval_mean_rgb_mse": 0.0,
                "Eval_mean_thermal_psnr": 0.0,
                "Eval_mean_thermal_mse": 0.0,
                "Eval_mean_thermal_mae": 0.0,
                "Eval_mean_thermal_mae_roi": 0.0,
                "Eval_mean_hssim": 0.0,
                "Eval_mean_tssim": 0.0,
            }
            img_ids = range(0, self.dataset_val.n_images, 1)
            for i, img_id in tqdm(enumerate(img_ids), total=len(img_ids)):

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

                (
                    thermal_mse_map,
                    thermal_mae_map,
                    thermal_mae_roi_map,
                    hssim_map,
                    tssim_map,
                ) = compute_thermal_metric_maps(
                    t_min=self.dataset_val.t_min,
                    t_max=self.dataset_val.t_max,
                    mae_roi_threshold=self.dataset_val.roi_threshold,
                    im_thermal=thermal_gt_val,
                    im_gt_thermal=thermal_pred_val,
                )

                mse_thermal_num = thermal_mse_map.mean().item()
                thermal_psnr = compute_psnr(mse_thermal_num)

                ThermoxelTrainer._update_rgb_stats(mse_rgb_num, rgb_psnr, stats_val)
                ThermoxelTrainer._update_thermal_stats(
                    mse_thermal_num,
                    thermal_psnr,
                    hssim_map.mean().item(),
                    tssim_map.mean().item(),
                    thermal_mae_map.mean().item(),
                    thermal_mae_roi_map.mean().item(),
                    stats_val,
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

                    mse_thermal = thermal_mse_map / thermal_mse_map.max()

                    mlflow.log_image(
                        np.array(mse_thermal),
                        f"outputs/val_mse_thermal_image" f"" f"_{img_id:04d}.png",
                    )
                if self._param.log_mae_image:
                    mae_map = thermal_mae_map / thermal_mae_map.max()

                    mlflow.log_image(
                        np.array(mae_map.unsqueeze(-1)),
                        f"outputs/val_mae_thermal_image_{img_id:04d}.png",
                    )
                if self._param.log_depth_map:
                    depth_img = self.grid.volume_render_depth_image(
                        camera=cam,
                        sigma_thresh=None,
                    )
                    depth_img = viridis_cmap(depth_img.cpu())

                    ThermoxelTrainer._log_concat_image(
                        rgb_gt_val.cpu().numpy(),
                        depth_img,
                        f"outputs/val_depth_map_{img_id:04d}.png",
                    )
                if self._param.log_surface_temperature:
                    surface_temp = self.grid.volume_render_surface_temperature_image(
                        camera=cam,
                        sigma_thresh=self._param.density_thresh,
                    )

                    ThermoxelTrainer._log_concat_image(
                        thermal_gt_val.cpu(),
                        surface_temp.cpu(),
                        f"outputs/val_surface_temp_image_{img_id:04d}.png",
                    )

            for stat_name in stats_val:
                stats_val[stat_name] /= self.dataset_val.n_images
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
            include_temperature=self._param.include_temperature,
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
        rgb_pred = rgb_pred.clamp(0.0, 1.0)
        thermal_pred = thermal_pred.clamp(0.0, 1.0)
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
        psnr = compute_psnr(mse_num)
        return mse_map, mse_num, psnr

    @staticmethod
    def _update_rgb_stats(rgb_mse: float, rgb_psnr: float, stats: dict[float]) -> None:
        for stat_name in stats:
            if stat_name.endswith("rgb_mse"):
                stats[stat_name] += rgb_mse
            if stat_name.endswith("rgb_psnr"):
                stats[stat_name] += rgb_psnr
            if stat_name.endswith("rgb_invsqr_mse"):
                stats[stat_name] += 1.0 / rgb_mse**2

    @staticmethod
    def _update_thermal_stats(
        thermal_mse: float,
        thermal_psnr: float,
        hssim: float | None,
        tssim: float | None,
        mae: float,
        mae_roi: float | None,
        stats: dict[float],
    ) -> None:
        for stat_name in stats:
            if stat_name.endswith("thermal_mse"):
                stats[stat_name] += thermal_mse
            if stat_name.endswith("thermal_psnr"):
                stats[stat_name] += thermal_psnr
            if stat_name.endswith("thermal_invsqr_mse"):
                stats[stat_name] += 1.0 / thermal_mse**2
            if stat_name.endswith("hssim") and hssim is not None:
                stats[stat_name] += hssim
            if stat_name.endswith("tssim") and tssim is not None:
                stats[stat_name] += tssim
            if stat_name.endswith("thermal_mae"):
                stats[stat_name] += mae
            if stat_name.endswith("thermal_mae_roi") and mae_roi is not None:
                stats[stat_name] += mae_roi
