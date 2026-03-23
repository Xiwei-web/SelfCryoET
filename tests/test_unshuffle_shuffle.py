import pytest
import torch

from self_cryoet.models.modules.volume_shuffle import (
    VolumeShuffle,
    VolumeUnshuffle,
    volume_shuffle,
    volume_unshuffle,
)


def test_volume_unshuffle_and_shuffle_are_inverse_operations():
    x = torch.arange(1 * 2 * 6 * 6 * 6, dtype=torch.float32).view(1, 2, 6, 6, 6)

    y = volume_unshuffle(x, factor=3)
    z = volume_shuffle(y, factor=3)

    assert y.shape == (1, 54, 2, 2, 2)
    assert torch.equal(z, x)


def test_volume_unshuffle_shuffle_modules_match_functional_api():
    x = torch.randn(2, 1, 9, 9, 9)
    unshuffle = VolumeUnshuffle(factor=3)
    shuffle = VolumeShuffle(factor=3)

    y1 = unshuffle(x)
    y2 = volume_unshuffle(x, factor=3)
    z1 = shuffle(y1)
    z2 = volume_shuffle(y2, factor=3)

    assert torch.allclose(y1, y2)
    assert torch.allclose(z1, z2)
    assert torch.allclose(z1, x)


def test_volume_unshuffle_raises_for_invalid_spatial_shape():
    x = torch.randn(1, 1, 5, 6, 6)

    with pytest.raises(ValueError, match="divisible by factor"):
        volume_unshuffle(x, factor=3)


def test_volume_shuffle_raises_for_invalid_channel_shape():
    x = torch.randn(1, 10, 2, 2, 2)

    with pytest.raises(ValueError, match="divisible by factor\\^3"):
        volume_shuffle(x, factor=3)

