import torch
import torch.nn as nn


def volume_unshuffle(x: torch.Tensor, factor: int) -> torch.Tensor:
    b, c, d, h, w = x.shape
    if d % factor != 0 or h % factor != 0 or w % factor != 0:
        raise ValueError("Volume dimensions must be divisible by factor.")
    x = x.view(b, c, d // factor, factor, h // factor, factor, w // factor, factor)
    x = x.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()
    return x.view(b, c * factor**3, d // factor, h // factor, w // factor)


def volume_shuffle(x: torch.Tensor, factor: int) -> torch.Tensor:
    b, c, d, h, w = x.shape
    if c % (factor**3) != 0:
        raise ValueError("Channel dimension must be divisible by factor^3.")
    out_c = c // (factor**3)
    x = x.view(b, out_c, factor, factor, factor, d, h, w)
    x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
    return x.view(b, out_c, d * factor, h * factor, w * factor)


class VolumeUnshuffle(nn.Module):
    def __init__(self, factor: int = 3):
        super().__init__()
        self.factor = factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return volume_unshuffle(x, self.factor)


class VolumeShuffle(nn.Module):
    def __init__(self, factor: int = 3):
        super().__init__()
        self.factor = factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return volume_shuffle(x, self.factor)

