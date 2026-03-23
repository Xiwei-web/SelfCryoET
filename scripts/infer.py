#!/usr/bin/env python3
from __future__ import annotations

import argparse

from self_cryoet.pipelines.infer_pipeline import run_infer_pipeline
from self_cryoet.utils.config import load_yaml_config, override_from_dotlist


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference for a single Cryo-ET volume.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument(
        "--override",
        nargs="*",
        default=None,
        help="Override config values with dotlist syntax.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    config = override_from_dotlist(config, args.override)
    run_infer_pipeline(config)


if __name__ == "__main__":
    main()

