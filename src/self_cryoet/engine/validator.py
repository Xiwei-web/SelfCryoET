from typing import Dict

import torch

from ..metrics.psnr import compute_psnr
from ..metrics.ssim import compute_ssim_3d


@torch.no_grad()
def validate(model, dataloader, criterion, device: torch.device) -> Dict[str, float]:
    model.eval()
    loss_meter = 0.0
    psnr_meter = 0.0
    ssim_meter = 0.0
    count = 0

    for batch in dataloader:
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        pred = model(batch["noisy"])
        losses = criterion(pred, batch)

        loss_meter += losses["loss"].item()
        psnr_meter += compute_psnr(pred, batch["guide"], data_range=1.0).item()
        ssim_meter += compute_ssim_3d(pred, batch["guide"], data_range=1.0).item()
        count += 1

    return {
        "val_loss": loss_meter / max(count, 1),
        "val_psnr": psnr_meter / max(count, 1),
        "val_ssim": ssim_meter / max(count, 1),
    }

