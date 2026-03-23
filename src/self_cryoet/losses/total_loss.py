from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn

from .edge_loss import EdgeEnhancementLoss
from .guidance_loss import GuidanceLoss
from .reconstruction_loss import ReconstructionLoss
from .tv_loss import TotalVariationLoss


@dataclass
class LossWeights:
    reconstruction: float = 0.8
    guidance: float = 0.5
    edge: float = 0.05
    tv: float = 0.01


class TotalLoss(nn.Module):
    def __init__(self, weights: LossWeights | None = None):
        super().__init__()
        self.weights = weights or LossWeights()
        self.reconstruction = ReconstructionLoss()
        self.guidance = GuidanceLoss()
        self.edge = EdgeEnhancementLoss()
        self.tv = TotalVariationLoss()

    def forward(self, pred: torch.Tensor, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        loss_rec = self.reconstruction(pred, batch["noisy"])
        loss_guide = self.guidance(pred, batch["guide"])
        loss_edge = self.edge(pred, batch["bilateral"])
        loss_tv = self.tv(pred)
        total = (
            self.weights.reconstruction * loss_rec
            + self.weights.guidance * loss_guide
            + self.weights.edge * loss_edge
            + self.weights.tv * loss_tv
        )
        return {
            "loss": total,
            "loss_rec": loss_rec,
            "loss_guide": loss_guide,
            "loss_edge": loss_edge,
            "loss_tv": loss_tv,
        }

