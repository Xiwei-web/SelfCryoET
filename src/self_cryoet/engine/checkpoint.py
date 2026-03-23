from pathlib import Path
from typing import Any, Dict, Optional

import torch

from ..utils.io import ensure_dir


def save_checkpoint(state: Dict[str, Any], path: str) -> None:
    path_obj = Path(path)
    ensure_dir(path_obj.parent)
    torch.save(state, path_obj)


def load_checkpoint(path: str, model=None, optimizer=None, scheduler=None, map_location: str = "cpu") -> Dict[str, Any]:
    checkpoint = torch.load(path, map_location=map_location)
    if model is not None and "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
    return checkpoint

