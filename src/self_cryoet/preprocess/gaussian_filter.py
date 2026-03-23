from typing import Sequence

import torch
import torch.nn.functional as F
import numpy as np


def _gaussian_kernel1d(kernel_size: int, sigma: float) -> torch.Tensor:
    coords = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2
    kernel = torch.exp(-(coords**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    return kernel


def gaussian_filter_tensor(volume: torch.Tensor, sigma: float = 1.0, kernel_size: int | None = None) -> torch.Tensor:
    if kernel_size is None:
        kernel_size = int(2 * round(3 * sigma) + 1)

    kernel_1d = _gaussian_kernel1d(kernel_size, sigma).to(volume.device, volume.dtype)
    kernel_3d = torch.einsum("i,j,k->ijk", kernel_1d, kernel_1d, kernel_1d)
    kernel_3d = kernel_3d / kernel_3d.sum()
    kernel_3d = kernel_3d.view(1, 1, kernel_size, kernel_size, kernel_size)

    padding = kernel_size // 2
    return F.conv3d(volume, kernel_3d, padding=padding)


def gaussian_filter_3d(volume: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    tensor = torch.from_numpy(volume).float().unsqueeze(0).unsqueeze(0)
    filtered = gaussian_filter_tensor(tensor, sigma=sigma)
    return filtered.squeeze(0).squeeze(0).cpu().numpy()

