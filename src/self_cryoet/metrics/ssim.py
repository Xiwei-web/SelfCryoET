import torch
import torch.nn.functional as F


def compute_ssim_3d(pred: torch.Tensor, target: torch.Tensor, window_size: int = 3, data_range: float = 1.0) -> torch.Tensor:
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    padding = window_size // 2
    kernel = torch.ones((1, 1, window_size, window_size, window_size), device=pred.device, dtype=pred.dtype)
    kernel = kernel / kernel.numel()

    mu_x = F.conv3d(pred, kernel, padding=padding)
    mu_y = F.conv3d(target, kernel, padding=padding)
    sigma_x = F.conv3d(pred * pred, kernel, padding=padding) - mu_x**2
    sigma_y = F.conv3d(target * target, kernel, padding=padding) - mu_y**2
    sigma_xy = F.conv3d(pred * target, kernel, padding=padding) - mu_x * mu_y

    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)
    return (numerator / (denominator + 1e-8)).mean()

