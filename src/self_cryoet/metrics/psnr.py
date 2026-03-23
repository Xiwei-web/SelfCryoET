import math

import torch


def compute_psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    mse = torch.mean((pred - target) ** 2)
    if mse <= 0:
        return torch.tensor(float("inf"), device=pred.device)
    return 20 * torch.log10(torch.tensor(data_range, device=pred.device)) - 10 * torch.log10(mse)

