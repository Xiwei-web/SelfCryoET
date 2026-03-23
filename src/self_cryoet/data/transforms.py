from dataclasses import dataclass
from typing import Dict

import torch


@dataclass
class RandomFlip3D:
    p: float = 0.5

    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        for dim in (1, 2, 3):
            if torch.rand(1).item() < self.p:
                sample = {k: torch.flip(v, dims=(dim,)) for k, v in sample.items()}
        return sample


@dataclass
class RandomRotate90:
    p: float = 0.5

    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if torch.rand(1).item() >= self.p:
            return sample
        k = int(torch.randint(0, 4, (1,)).item())
        return {k_: torch.rot90(v, k=k, dims=(2, 3)) for k_, v in sample.items()}


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for transform in self.transforms:
            sample = transform(sample)
        return sample

