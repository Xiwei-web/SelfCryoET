import torch


def compute_fsc(volume_a: torch.Tensor, volume_b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    fa = torch.fft.fftn(volume_a, dim=(-3, -2, -1))
    fb = torch.fft.fftn(volume_b, dim=(-3, -2, -1))
    numerator = (fa * torch.conj(fb)).real.sum()
    denominator = torch.sqrt((fa.abs() ** 2).sum() * (fb.abs() ** 2).sum()) + eps
    return numerator / denominator

