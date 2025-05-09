import math
import os

import imageio
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity


# Define the convolution function with the uniform filter
def uniform_filter_fn(
    z: torch.tensor, window: torch.tensor, stride: int = 4
) -> torch.tensor:
    """
    Apply a uniform filter to an image tensor z with a window tensor.
    """
    width, height, num_channels = z.shape
    z = z.view(-1, num_channels, width, height)
    window = window.view(-1, 1, window.shape[0], window.shape[1])
    # z is a tensor of size [B, H, W, C]
    return F.conv2d(
        z,
        weight=window,
        stride=stride,
        padding=0,
        groups=1,
    )


def compute_hssim(
    im: torch.tensor,
    im_gt: torch.tensor,
    kernel_size: int = 4,
    stride: int = 4,
    C2: float = 9e-4,
) -> float:
    """
    Compute the Heat-based Structural Similarity Index Measure (HSSIM) between two
    thermal images. Based on : https://arxiv.org/abs/2403.10340

    :return: HSSIM value as float
    """
    device = im_gt.device

    # Take into account case where image are (w,h) instead of (w, h, 1)
    if len(im.shape) == 2:
        im = im.unsqueeze(-1)
    if len(im_gt.shape) == 2:
        im_gt = im_gt.unsqueeze(-1)

    window = torch.ones((kernel_size, kernel_size, 1), device=device) / (
        kernel_size * kernel_size
    )

    # Mean filters
    mu1 = uniform_filter_fn(im, window, stride=stride)
    mu2 = uniform_filter_fn(im_gt, window, stride=stride)

    # Variance and covariance
    sigma1_sq = uniform_filter_fn(im**2, window, stride=stride) - mu1**2
    sigma2_sq = uniform_filter_fn(im_gt**2, window, stride=stride) - mu2**2
    sigma12 = uniform_filter_fn(im * im_gt, window, stride=stride) - mu1 * mu2

    # Compute HSSIM
    numerator = 2 * sigma12 + C2
    denominator = sigma1_sq + sigma2_sq + C2
    hssim_map = numerator / denominator

    return hssim_map


def mae_thermal(
    gt: torch.Tensor,
    pred: torch.Tensor,
    threshold: float,
    cold_flag: bool,
    max_temperature: float,
    min_temperature: float,
) -> torch.Tensor:
    if cold_flag:
        indices_foreground = torch.where(gt < threshold)
    else:
        indices_foreground = torch.where(gt > threshold)

    # Save GT tensor as binarize image given threshold
    gt_save = torch.zeros_like(gt)
    gt_save[indices_foreground] = 1.0
    git_save_as_numpy = gt_save.cpu().numpy()
    gt_save_as_numpy = (git_save_as_numpy * 255).astype("uint8")

    i = 0
    while os.path.exists("outputs/sample%s.png" % i):
        i += 1
    imageio.imwrite("outputs/sample%s.png" % i, gt_save_as_numpy)

    gt = gt[indices_foreground]
    gt = gt * (max_temperature - min_temperature) + min_temperature

    pred = pred[indices_foreground]
    pred = pred * (max_temperature - min_temperature) + min_temperature
    mae = torch.abs(gt - pred)

    return mae


def compute_psnr(mse_num: float) -> float:
    """
    This funciton computes the psnr from the mean squared error assuming a scale of 1

    returns: psnr as float
    """
    return -10.0 * math.log10(mse_num)


def compute_thermal_metric_maps(
    t_min: float,
    t_max: float,
    mae_roi_threshold: float,
    im_thermal: torch.Tensor,
    im_gt_thermal: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, torch.Tensor]:
    mse_map = (im_thermal.cpu() - im_gt_thermal.cpu()) ** 2

    mae_map = mae_thermal(
        im_gt_thermal.cpu(),
        im_thermal.cpu(),
        0,
        False,
        t_max,
        t_min,
    )

    mae_roi_map = mae_thermal(
        im_gt_thermal.cpu(),
        im_thermal.cpu(),
        mae_roi_threshold,
        False,
        t_max,
        t_min,
    )

    hssim_map = compute_hssim(im_thermal.cpu(), im_gt_thermal.cpu())

    _, tssim_map = structural_similarity(
        im_thermal.cpu().numpy(),
        im_gt_thermal.cpu().numpy(),
        channel_axis=None,
        data_range=1,
        full=True,
    )

    return mse_map, mae_map, mae_roi_map, hssim_map, tssim_map


def compute_thermal_metrics(
    t_min: float,
    t_max: float,
    mae_roi_threshold: float,
    im_thermal: torch.Tensor,
    im_gt_thermal: torch.Tensor,
) -> tuple[float, float, float, float, float, float]:
    mse_map, mae_map, mae_roi_map, hssim_map, tssim_map = compute_thermal_metric_maps(
        t_min=t_min,
        t_max=t_max,
        mae_roi_threshold=mae_roi_threshold,
        im_thermal=im_thermal,
        im_gt_thermal=im_gt_thermal,
    )

    mse_num = mse_map.mean().item()
    psnr_thermal = compute_psnr(mse_num)
    mae = mae_map.mean().item()
    mae_roi = mae_roi_map.mean().item()
    hssim = hssim_map.mean().item()
    tssim = tssim_map.mean().item()

    return mse_num, psnr_thermal, mae, mae_roi, hssim, tssim
