from typing import Dict

import torch
from torch.utils.data import DataLoader

from ..data.collate import volume_collate_fn
from ..data.volume_dataset import CryoETVolumeDataset, VolumeDatasetConfig
from ..engine.checkpoint import load_checkpoint
from ..engine.validator import validate
from ..losses.total_loss import TotalLoss
from ..models.network import SelfCryoETNet


def run_eval_pipeline(config: Dict) -> Dict[str, float]:
    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    dataset = CryoETVolumeDataset(VolumeDatasetConfig(**config["dataset"]))
    loader = DataLoader(
        dataset,
        batch_size=config.get("batch_size", 2),
        shuffle=False,
        num_workers=config.get("num_workers", 0),
        collate_fn=volume_collate_fn,
    )

    model = SelfCryoETNet(**config.get("model", {})).to(device)
    load_checkpoint(config["checkpoint_path"], model=model, map_location=device)
    criterion = TotalLoss()
    return validate(model, loader, criterion, device)

