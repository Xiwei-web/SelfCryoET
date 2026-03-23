from dataclasses import dataclass
from typing import Dict, Optional

import torch

from .checkpoint import save_checkpoint
from .validator import validate


@dataclass
class TrainerConfig:
    epochs: int = 15
    grad_clip: Optional[float] = None
    log_interval: int = 10
    checkpoint_dir: str = "checkpoints"


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        criterion,
        device: torch.device,
        config: TrainerConfig,
        logger=None,
        scheduler=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.config = config
        self.logger = logger
        self.scheduler = scheduler

    def fit(self, train_loader, val_loader=None) -> Dict[str, float]:
        best_metric = float("-inf")
        history = {}

        self.model.to(self.device)
        for epoch in range(1, self.config.epochs + 1):
            train_stats = self._train_one_epoch(train_loader, epoch)
            history[f"epoch_{epoch}"] = train_stats

            if val_loader is not None:
                val_stats = validate(self.model, val_loader, self.criterion, self.device)
                history[f"epoch_{epoch}"].update(val_stats)
                metric = val_stats["val_psnr"]
                if metric > best_metric:
                    best_metric = metric
                    save_checkpoint(
                        {
                            "epoch": epoch,
                            "model": self.model.state_dict(),
                            "optimizer": self.optimizer.state_dict(),
                            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
                            "history": history,
                        },
                        f"{self.config.checkpoint_dir}/best.pt",
                    )
            if self.scheduler is not None:
                self.scheduler.step()

        return history

    def _train_one_epoch(self, train_loader, epoch: int) -> Dict[str, float]:
        self.model.train()
        running_loss = 0.0

        for step, batch in enumerate(train_loader, start=1):
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
            pred = self.model(batch["noisy"])
            losses = self.criterion(pred, batch)

            self.optimizer.zero_grad(set_to_none=True)
            losses["loss"].backward()

            if self.config.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)

            self.optimizer.step()
            running_loss += losses["loss"].item()

            if self.logger and step % self.config.log_interval == 0:
                self.logger.info(
                    "epoch=%d step=%d loss=%.6f rec=%.6f guide=%.6f edge=%.6f tv=%.6f",
                    epoch,
                    step,
                    losses["loss"].item(),
                    losses["loss_rec"].item(),
                    losses["loss_guide"].item(),
                    losses["loss_edge"].item(),
                    losses["loss_tv"].item(),
                )

        return {"train_loss": running_loss / max(len(train_loader), 1)}

