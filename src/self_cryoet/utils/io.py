from pathlib import Path
from typing import Any, Dict, Union

import numpy as np
import torch


PathLike = Union[str, Path]


def ensure_dir(path: PathLike) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_volume(path: PathLike) -> np.ndarray:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path).astype(np.float32)
    if suffix in {".pt", ".pth"}:
        tensor = torch.load(path, map_location="cpu")
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy().astype(np.float32)
    raise ValueError(f"Unsupported volume format: {path}")


def save_volume(volume: np.ndarray, path: PathLike) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        np.save(path, volume.astype(np.float32))
        return
    if suffix in {".pt", ".pth"}:
        torch.save(torch.from_numpy(volume.astype(np.float32)), path)
        return
    raise ValueError(f"Unsupported volume format: {path}")


def to_tensor(array: np.ndarray, add_channel_dim: bool = True) -> torch.Tensor:
    tensor = torch.from_numpy(array).float()
    if add_channel_dim and tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    return tensor


def detach_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def save_checkpoint_payload(payload: Dict[str, Any], path: PathLike) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    torch.save(payload, path)

