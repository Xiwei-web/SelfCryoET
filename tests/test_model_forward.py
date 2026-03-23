import torch

from self_cryoet.models.network import SelfCryoETNet
from self_cryoet.models.unet_bsn import UShapeBSN


def test_ushape_bsn_forward_preserves_input_shape():
    model = UShapeBSN(
        in_channels=1,
        out_channels=1,
        base_channels=8,
        num_levels=2,
        blocks_per_level=1,
        shuffle_factor=3,
    )
    x = torch.randn(2, 1, 27, 27, 27)

    y = model(x)

    assert y.shape == x.shape


def test_network_forward_batch_returns_prediction_dict():
    model = SelfCryoETNet(
        in_channels=1,
        out_channels=1,
        base_channels=8,
        num_levels=2,
        blocks_per_level=1,
        shuffle_factor=3,
    )
    batch = {"noisy": torch.randn(1, 1, 27, 27, 27)}

    outputs = model.forward_batch(batch)

    assert set(outputs.keys()) == {"pred"}
    assert outputs["pred"].shape == batch["noisy"].shape


def test_network_backward_pass_produces_gradients():
    model = SelfCryoETNet(
        in_channels=1,
        out_channels=1,
        base_channels=8,
        num_levels=2,
        blocks_per_level=1,
        shuffle_factor=3,
    )
    x = torch.randn(1, 1, 27, 27, 27)
    target = torch.randn(1, 1, 27, 27, 27)

    pred = model(x)
    loss = torch.mean((pred - target) ** 2)
    loss.backward()

    grads = [param.grad for param in model.parameters() if param.requires_grad]
    assert any(grad is not None for grad in grads)

