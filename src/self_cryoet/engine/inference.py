from typing import Dict, Tuple

import numpy as np
import torch


@torch.no_grad()
def sliding_window_inference(model, volume: torch.Tensor, patch_size: Tuple[int, int, int], stride: Tuple[int, int, int]) -> torch.Tensor:
    model.eval()
    _, _, depth, height, width = volume.shape
    pd, ph, pw = patch_size
    sd, sh, sw = stride

    output = torch.zeros_like(volume)
    counter = torch.zeros_like(volume)

    z_positions = list(range(0, max(depth - pd + 1, 1), sd))
    y_positions = list(range(0, max(height - ph + 1, 1), sh))
    x_positions = list(range(0, max(width - pw + 1, 1), sw))

    if z_positions[-1] != depth - pd:
        z_positions.append(max(depth - pd, 0))
    if y_positions[-1] != height - ph:
        y_positions.append(max(height - ph, 0))
    if x_positions[-1] != width - pw:
        x_positions.append(max(width - pw, 0))

    for z in z_positions:
        for y in y_positions:
            for x in x_positions:
                patch = volume[:, :, z : z + pd, y : y + ph, x : x + pw]
                pred = model(patch)
                output[:, :, z : z + pd, y : y + ph, x : x + pw] += pred
                counter[:, :, z : z + pd, y : y + ph, x : x + pw] += 1

    return output / counter.clamp_min(1.0)


@torch.no_grad()
def infer_single_volume(model, volume: np.ndarray, device: torch.device, patch_size, stride) -> np.ndarray:
    tensor = torch.from_numpy(volume).float().unsqueeze(0).unsqueeze(0).to(device)
    pred = sliding_window_inference(model, tensor, patch_size=patch_size, stride=stride)
    return pred.squeeze(0).squeeze(0).cpu().numpy()

