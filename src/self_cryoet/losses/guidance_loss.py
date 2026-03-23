import torch
import torch.nn as nn


class GuidanceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.MSELoss()

    def forward(self, pred: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        return self.criterion(pred, guide)

