from typing import Dict

import torch
import torch.nn as nn

from .unet_bsn import UShapeBSN


class SelfCryoETNet(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.backbone = UShapeBSN(**kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        pred = self.forward(batch["noisy"])
        return {"pred": pred}

