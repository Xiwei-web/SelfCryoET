import numpy as np
import torch

from self_cryoet.data.volume_dataset import CryoETVolumeDataset, VolumeDatasetConfig


def test_volume_dataset_returns_expected_keys_and_shapes(tmp_path):
    volume = np.random.randn(9, 9, 9).astype(np.float32)
    volume_path = tmp_path / "volume.npy"
    np.save(volume_path, volume)

    config = VolumeDatasetConfig(
        volume_path=str(volume_path),
        patch_size=(3, 3, 3),
        stride=(3, 3, 3),
        normalize=True,
        gaussian_sigma=1.0,
        bilateral_kernel_size=3,
        bilateral_sigma_spatial=1.0,
        bilateral_sigma_intensity=0.5,
    )
    dataset = CryoETVolumeDataset(config)

    sample = dataset[0]

    assert len(dataset) == 27
    assert set(sample.keys()) == {"noisy", "guide", "bilateral", "edge", "index"}
    assert sample["noisy"].shape == (1, 3, 3, 3)
    assert sample["guide"].shape == (1, 3, 3, 3)
    assert sample["bilateral"].shape == (1, 3, 3, 3)
    assert sample["edge"].shape == (1, 3, 3, 3)
    assert sample["index"].shape == (3,)
    assert sample["noisy"].dtype == torch.float32


def test_volume_dataset_exposes_full_preprocessed_volumes(tmp_path):
    volume = np.random.randn(6, 6, 6).astype(np.float32)
    volume_path = tmp_path / "volume.npy"
    np.save(volume_path, volume)

    dataset = CryoETVolumeDataset(
        VolumeDatasetConfig(
            volume_path=str(volume_path),
            patch_size=(3, 3, 3),
            stride=(3, 3, 3),
            bilateral_kernel_size=3,
            bilateral_sigma_spatial=1.0,
            bilateral_sigma_intensity=0.5,
        )
    )

    full_volume = dataset.get_full_volume()

    assert set(full_volume.keys()) == {"noisy", "guide", "bilateral", "edge"}
    for value in full_volume.values():
        assert value.shape == (6, 6, 6)
        assert value.dtype == np.float32

