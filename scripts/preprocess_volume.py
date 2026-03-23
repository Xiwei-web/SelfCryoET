#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from self_cryoet.preprocess.bilateral_filter import bilateral_filter_3d
from self_cryoet.preprocess.edge_enhancer import compute_edge_map
from self_cryoet.preprocess.gaussian_filter import gaussian_filter_3d
from self_cryoet.preprocess.normalize import normalize_volume
from self_cryoet.utils.io import load_volume, save_volume


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess a Cryo-ET volume.")
    parser.add_argument("--input", type=str, required=True, help="Input volume path (.npy/.pt/.pth).")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save processed volumes.")
    parser.add_argument("--normalize", action="store_true", help="Apply z-score normalization before filtering.")
    parser.add_argument("--gaussian-sigma", type=float, default=1.0, help="Gaussian smoothing sigma.")
    parser.add_argument("--bilateral-kernel-size", type=int, default=5, help="Bilateral filter kernel size.")
    parser.add_argument("--bilateral-sigma-spatial", type=float, default=2.0, help="Bilateral spatial sigma.")
    parser.add_argument("--bilateral-sigma-intensity", type=float, default=0.1, help="Bilateral intensity sigma.")
    parser.add_argument(
        "--prefix",
        type=str,
        default="volume",
        help="Prefix used for saved artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    volume = load_volume(args.input)
    if args.normalize:
        volume = normalize_volume(volume)

    gaussian = gaussian_filter_3d(volume, sigma=args.gaussian_sigma)
    bilateral = bilateral_filter_3d(
        volume,
        kernel_size=args.bilateral_kernel_size,
        sigma_spatial=args.bilateral_sigma_spatial,
        sigma_intensity=args.bilateral_sigma_intensity,
    )
    edge = compute_edge_map(bilateral)

    save_volume(volume, output_dir / f"{args.prefix}_normalized.npy")
    save_volume(gaussian, output_dir / f"{args.prefix}_gaussian.npy")
    save_volume(bilateral, output_dir / f"{args.prefix}_bilateral.npy")
    save_volume(edge, output_dir / f"{args.prefix}_edge.npy")


if __name__ == "__main__":
    main()

