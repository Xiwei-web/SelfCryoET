import numpy as np


def bilateral_filter_3d(
    volume: np.ndarray,
    kernel_size: int = 5,
    sigma_spatial: float = 2.0,
    sigma_intensity: float = 0.1,
) -> np.ndarray:
    pad = kernel_size // 2
    padded = np.pad(volume, pad_width=pad, mode="reflect")
    result = np.zeros_like(volume, dtype=np.float32)

    grid = np.arange(-pad, pad + 1, dtype=np.float32)
    zz, yy, xx = np.meshgrid(grid, grid, grid, indexing="ij")
    spatial = np.exp(-(xx**2 + yy**2 + zz**2) / (2 * sigma_spatial**2))

    depth, height, width = volume.shape
    for z in range(depth):
        for y in range(height):
            for x in range(width):
                patch = padded[z : z + kernel_size, y : y + kernel_size, x : x + kernel_size]
                center = padded[z + pad, y + pad, x + pad]
                intensity = np.exp(-((patch - center) ** 2) / (2 * sigma_intensity**2 + 1e-8))
                weights = spatial * intensity
                weights /= weights.sum() + 1e-8
                result[z, y, x] = np.sum(weights * patch)
    return result

