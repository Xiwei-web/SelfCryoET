import torch

from self_cryoet.losses.edge_loss import EdgeEnhancementLoss
from self_cryoet.losses.guidance_loss import GuidanceLoss
from self_cryoet.losses.reconstruction_loss import ReconstructionLoss
from self_cryoet.losses.total_loss import LossWeights, TotalLoss
from self_cryoet.losses.tv_loss import TotalVariationLoss


def test_reconstruction_loss_is_zero_for_identical_inputs():
    loss_fn = ReconstructionLoss()
    x = torch.ones(2, 1, 3, 3, 3)

    loss = loss_fn(x, x)

    assert torch.isclose(loss, torch.tensor(0.0))


def test_guidance_loss_is_positive_for_different_inputs():
    loss_fn = GuidanceLoss()
    pred = torch.zeros(1, 1, 3, 3, 3)
    guide = torch.ones(1, 1, 3, 3, 3)

    loss = loss_fn(pred, guide)

    assert loss.item() > 0


def test_tv_loss_is_zero_for_constant_volume():
    loss_fn = TotalVariationLoss()
    volume = torch.ones(1, 1, 4, 4, 4)

    loss = loss_fn(volume)

    assert torch.isclose(loss, torch.tensor(0.0))


def test_edge_loss_is_zero_for_identical_inputs():
    loss_fn = EdgeEnhancementLoss()
    volume = torch.randn(1, 1, 5, 5, 5)

    loss = loss_fn(volume, volume)

    assert torch.isclose(loss, torch.tensor(0.0))


def test_total_loss_returns_full_loss_dictionary():
    loss_fn = TotalLoss(LossWeights(reconstruction=1.0, guidance=1.0, edge=1.0, tv=1.0))
    pred = torch.randn(2, 1, 5, 5, 5)
    batch = {
        "noisy": torch.randn(2, 1, 5, 5, 5),
        "guide": torch.randn(2, 1, 5, 5, 5),
        "bilateral": torch.randn(2, 1, 5, 5, 5),
    }

    losses = loss_fn(pred, batch)

    assert set(losses.keys()) == {"loss", "loss_rec", "loss_guide", "loss_edge", "loss_tv"}
    for value in losses.values():
        assert value.ndim == 0
        assert torch.isfinite(value)

