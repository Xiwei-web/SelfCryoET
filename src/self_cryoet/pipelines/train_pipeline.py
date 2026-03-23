from typing import Dict

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split

from ..data.collate import volume_collate_fn
from ..data.transforms import Compose, RandomFlip3D, RandomRotate90
from ..data.volume_dataset import CryoETVolumeDataset, VolumeDatasetConfig
from ..engine.trainer import Trainer, TrainerConfig
from ..losses.total_loss import LossWeights, TotalLoss
from ..models.network import SelfCryoETNet
from ..utils.logger import setup_logger
from ..utils.seed import seed_everything


def run_train_pipeline(config: Dict) -> Dict:
    seed_everything(config.get("seed", 42))
    logger = setup_logger("self_cryoet.train", config.get("log_file"))
    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    transform = Compose([RandomFlip3D(), RandomRotate90()]) if config.get("augment", True) else None
    dataset = CryoETVolumeDataset(VolumeDatasetConfig(**config["dataset"]), transform=transform)

    val_ratio = config.get("val_ratio", 0.2)
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_set,
        batch_size=config.get("batch_size", 2),
        shuffle=True,
        num_workers=config.get("num_workers", 0),
        collate_fn=volume_collate_fn,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=config.get("batch_size", 2),
        shuffle=False,
        num_workers=config.get("num_workers", 0),
        collate_fn=volume_collate_fn,
    )

    model = SelfCryoETNet(**config.get("model", {}))
    optimizer = Adam(
        model.parameters(),
        lr=config.get("lr", 2e-4),
        betas=tuple(config.get("betas", (0.5, 0.999))),
    )
    criterion = TotalLoss(LossWeights(**config.get("loss_weights", {})))
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        config=TrainerConfig(**config.get("trainer", {})),
        logger=logger,
    )
    return trainer.fit(train_loader, val_loader)

