import numpy as np


def normalize_volume(volume: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    volume = volume.astype(np.float32)
    mean = volume.mean()
    std = volume.std()
    return (volume - mean) / (std + eps)


def minmax_volume(volume: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    volume = volume.astype(np.float32)
    min_val = volume.min()
    max_val = volume.max()
    return (volume - min_val) / (max_val - min_val + eps)

