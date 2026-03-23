import torch

from self_cryoet.models.modules.masked_conv3d import SparseMaskedConv3d


def test_sparse_masked_conv_masks_center_3x3x3_region():
    layer = SparseMaskedConv3d(1, 1, kernel_size=5, bias=False)
    mask = layer.mask[0, 0]

    assert mask.shape == (5, 5, 5)
    assert torch.count_nonzero(mask[1:4, 1:4, 1:4]) == 0
    assert torch.count_nonzero(mask) == 125 - 27


def test_sparse_masked_conv_ignores_center_voxel_when_only_center_is_active():
    layer = SparseMaskedConv3d(1, 1, kernel_size=5, bias=False)
    with torch.no_grad():
        layer.weight.fill_(1.0)

    x = torch.zeros(1, 1, 7, 7, 7)
    x[:, :, 3, 3, 3] = 1.0
    y = layer(x)

    assert torch.isclose(y[0, 0, 3, 3, 3], torch.tensor(0.0))


def test_sparse_masked_conv_produces_expected_shape():
    layer = SparseMaskedConv3d(2, 4, kernel_size=5)
    x = torch.randn(3, 2, 9, 9, 9)
    y = layer(x)

    assert y.shape == (3, 4, 9, 9, 9)

