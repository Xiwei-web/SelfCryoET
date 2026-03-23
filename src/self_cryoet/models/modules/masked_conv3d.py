import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseMaskedConv3d(nn.Conv3d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5, **kwargs):
        padding = kwargs.pop("padding", kernel_size // 2)
        super().__init__(in_channels, out_channels, kernel_size, padding=padding, bias=kwargs.pop("bias", True), **kwargs)
        self.register_buffer("mask", self._build_mask())

    def _build_mask(self) -> torch.Tensor:
        mask = torch.ones_like(self.weight.data)
        kz, ky, kx = self.weight.shape[-3:]
        cz, cy, cx = kz // 2, ky // 2, kx // 2

        z0, z1 = max(cz - 1, 0), min(cz + 2, kz)
        y0, y1 = max(cy - 1, 0), min(cy + 2, ky)
        x0, x1 = max(cx - 1, 0), min(cx + 2, kx)
        mask[:, :, z0:z1, y0:y1, x0:x1] = 0
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight * self.mask
        return F.conv3d(
            x,
            weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

