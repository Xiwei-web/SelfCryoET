import torch
import torch.nn as nn


class TotalVariationLoss(nn.Module):
    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        dz = torch.abs(volume[:, :, 1:, :, :] - volume[:, :, :-1, :, :]).mean()
        dy = torch.abs(volume[:, :, :, 1:, :] - volume[:, :, :, :-1, :]).mean()
        dx = torch.abs(volume[:, :, :, :, 1:] - volume[:, :, :, :, :-1]).mean()
        return dx + dy + dz

