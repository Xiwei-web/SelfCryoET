#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from self_cryoet.data.volume_dataset import CryoETVolumeDataset, VolumeDatasetConfig
from self_cryoet.utils.io import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export preprocessed Cryo-ET patches to disk.")
    parser.add_argument("--input", type=str, required=True, help="Input volume path (.npy/.pt/.pth).")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save patch files.")
    parser.add_argument("--patch-size", type=int, nargs=3, required=True, metavar=("D", "H", "W"))
    parser.add_argument("--stride", type=int, nargs=3, required=True, metavar=("D", "H", "W"))
    parser.add_argument("--normalize", action="store_true", help="Apply z-score normalization.")
    parser.add_argument("--gaussian-sigma", type=float, default=1.0)
    parser.add_argument("--bilateral-kernel-size", type=int, default=5)
    parser.add_argument("--bilateral-sigma-spatial", type=float, default=2.0)
    parser.add_argument("--bilateral-sigma-intensity", type=float, default=0.1)
    parser.add_argument(
        "--save-arrays",
        nargs="*",
        default=["noisy", "guide", "bilateral", "edge"],
        choices=["noisy", "guide", "bilateral", "edge"],
        help="Patch tensors to export.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)

    dataset = CryoETVolumeDataset(
        VolumeDatasetConfig(
            volume_path=args.input,
            patch_size=tuple(args.patch_size),
            stride=tuple(args.stride),
            normalize=args.normalize,
            gaussian_sigma=args.gaussian_sigma,
            bilateral_kernel_size=args.bilateral_kernel_size,
            bilateral_sigma_spatial=args.bilateral_sigma_spatial,
            bilateral_sigma_intensity=args.bilateral_sigma_intensity,
        )
    )

    for i in range(len(dataset)):
        sample = dataset[i]
        z, y, x = sample["index"].tolist()
        stem = f"patch_{i:05d}_z{z}_y{y}_x{x}"
        patch_dir = ensure_dir(Path(output_dir) / stem)

        for key in args.save_arrays:
            array = sample[key].squeeze(0).cpu().numpy().astype(np.float32)
            np.save(patch_dir / f"{key}.npy", array)

        np.save(patch_dir / "index.npy", sample["index"].cpu().numpy().astype(np.int64))


if __name__ == "__main__":
    main()

