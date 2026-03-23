from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from ..preprocess.bilateral_filter import bilateral_filter_3d
from ..preprocess.edge_enhancer import compute_edge_map
from ..preprocess.gaussian_filter import gaussian_filter_3d
from ..preprocess.normalize import normalize_volume
from ..utils.io import load_volume, to_tensor
from .patch_sampler import PatchSampler


@dataclass
class VolumeDatasetConfig:
    volume_path: str
    patch_size: Sequence[int]
    stride: Sequence[int]
    normalize: bool = True
    gaussian_sigma: float = 1.0
    bilateral_kernel_size: int = 5
    bilateral_sigma_spatial: float = 2.0
    bilateral_sigma_intensity: float = 0.1


class CryoETVolumeDataset(Dataset):
    def __init__(self, config: VolumeDatasetConfig, transform=None):
        self.config = config
        self.transform = transform

        volume = load_volume(config.volume_path)
        if config.normalize:
            volume = normalize_volume(volume)

        self.raw_volume = volume.astype(np.float32)
        self.guidance_volume = gaussian_filter_3d(self.raw_volume, sigma=config.gaussian_sigma)
        self.bilateral_volume = bilateral_filter_3d(
            self.raw_volume,
            kernel_size=config.bilateral_kernel_size,
            sigma_spatial=config.bilateral_sigma_spatial,
            sigma_intensity=config.bilateral_sigma_intensity,
        )
        self.edge_volume = compute_edge_map(self.bilateral_volume)
        self.indices = PatchSampler(
            volume_shape=self.raw_volume.shape,
            patch_size=config.patch_size,
            stride=config.stride,
        ).generate()

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        z, y, x = self.indices[idx]
        pd, ph, pw = self.config.patch_size

        sample = {
            "noisy": to_tensor(self.raw_volume[z : z + pd, y : y + ph, x : x + pw]),
            "guide": to_tensor(self.guidance_volume[z : z + pd, y : y + ph, x : x + pw]),
            "bilateral": to_tensor(self.bilateral_volume[z : z + pd, y : y + ph, x : x + pw]),
            "edge": to_tensor(self.edge_volume[z : z + pd, y : y + ph, x : x + pw]),
            "index": torch.tensor([z, y, x], dtype=torch.long),
        }

        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def get_full_volume(self) -> Dict[str, np.ndarray]:
        return {
            "noisy": self.raw_volume,
            "guide": self.guidance_volume,
            "bilateral": self.bilateral_volume,
            "edge": self.edge_volume,
        }

