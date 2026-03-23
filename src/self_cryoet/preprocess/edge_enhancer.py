import numpy as np
import torch
import torch.nn.functional as F


def _kirsch_kernels_2d() -> torch.Tensor:
    kernels = [
        [[5, 5, 5], [-3, 0, -3], [-3, -3, -3]],
        [[5, 5, -3], [5, 0, -3], [-3, -3, -3]],
        [[5, -3, -3], [5, 0, -3], [5, -3, -3]],
        [[-3, -3, -3], [5, 0, -3], [5, 5, -3]],
        [[-3, -3, -3], [-3, 0, -3], [5, 5, 5]],
        [[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]],
        [[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]],
        [[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]],
    ]
    return torch.tensor(kernels, dtype=torch.float32).unsqueeze(1)


def compute_edge_map(volume: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    tensor = torch.from_numpy(volume).float().unsqueeze(0).unsqueeze(0)
    kernels = _kirsch_kernels_2d().to(tensor.device)

    responses = []
    for axis in range(3):
        if axis == 0:
            slices = tensor.squeeze(0).permute(1, 0, 2, 3)
        elif axis == 1:
            slices = tensor.squeeze(0).permute(2, 0, 1, 3)
        else:
            slices = tensor.squeeze(0).permute(3, 0, 1, 2)

        response = F.conv2d(slices, kernels, padding=1)
        response = response.max(dim=1).values

        if axis == 0:
            response = response.permute(1, 0, 2)
        elif axis == 1:
            response = response.permute(1, 0, 2)
        else:
            response = response.permute(1, 2, 0)
        responses.append(response)

    edge = torch.stack(responses, dim=0).max(dim=0).values
    edge = torch.clamp(edge, min=threshold)
    return edge.cpu().numpy().astype(np.float32)


def edge_map_tensor(volume: torch.Tensor, threshold: float = 0.0) -> torch.Tensor:
    batch = []
    for item in volume:
        batch.append(torch.from_numpy(compute_edge_map(item.squeeze(0).detach().cpu().numpy(), threshold)).unsqueeze(0))
    return torch.stack(batch, dim=0).to(volume.device, volume.dtype)

