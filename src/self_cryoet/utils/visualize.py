from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def save_slice_grid(
    volume: np.ndarray,
    output_path: str,
    axis: int = 0,
    indices: Optional[list[int]] = None,
    title: Optional[str] = None,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if indices is None:
        size = volume.shape[axis]
        indices = [size // 4, size // 2, (3 * size) // 4]

    slices = [np.take(volume, idx, axis=axis) for idx in indices]
    fig, axes = plt.subplots(1, len(slices), figsize=(4 * len(slices), 4))
    if len(slices) == 1:
        axes = [axes]

    for ax, image, idx in zip(axes, slices, indices):
        ax.imshow(image, cmap="gray")
        ax.set_title(f"slice={idx}")
        ax.axis("off")

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)

