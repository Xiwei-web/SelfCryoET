import torch
import torch.nn as nn

from ..preprocess.edge_enhancer import edge_map_tensor


class EdgeEnhancementLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.MSELoss()

    def forward(self, pred: torch.Tensor, bilateral: torch.Tensor) -> torch.Tensor:
        pred_edge = edge_map_tensor(pred)
        ref_edge = edge_map_tensor(bilateral)
        return self.criterion(pred_edge, ref_edge)

