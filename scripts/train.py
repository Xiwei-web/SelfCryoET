#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from self_cryoet.pipelines.train_pipeline import run_train_pipeline
from self_cryoet.utils.config import load_yaml_config, override_from_dotlist


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Self-CryoET on a single noisy volume.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument(
        "--override",
        nargs="*",
        default=None,
        help="Override config values with dotlist syntax, e.g. trainer.epochs=20 batch_size=1",
    )
    parser.add_argument(
        "--save-history",
        type=str,
        default=None,
        help="Optional JSON path for training history.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    config = override_from_dotlist(config, args.override)
    history = run_train_pipeline(config)

    if args.save_history is not None:
        output_path = Path(args.save_history)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()

