import torch
import torch.nn.functional as F


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

    # Average HSSIM over sliding windows
    hssim = hssim_map.mean().item()

    return hssim


def mae_thermal(gt, pred, threshold, cold_flag, max_temperature, min_temperature):
    if cold_flag:
        indices_foreground = torch.where(gt < threshold)
    else:
        indices_foreground = torch.where(gt > threshold)

    gt = gt[indices_foreground]
    gt = gt * (max_temperature - min_temperature) + min_temperature

    pred = pred[indices_foreground]
    pred = pred * (max_temperature - min_temperature) + min_temperature
    mae = torch.abs(gt - pred)

    return mae
