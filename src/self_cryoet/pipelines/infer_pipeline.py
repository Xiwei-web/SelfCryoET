from typing import Dict

import torch

from ..engine.checkpoint import load_checkpoint
from ..engine.inference import infer_single_volume
from ..models.network import SelfCryoETNet
from ..preprocess.normalize import normalize_volume
from ..utils.io import load_volume, save_volume


def run_infer_pipeline(config: Dict) -> str:
    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    volume = load_volume(config["input_path"])
    if config.get("normalize", True):
        volume = normalize_volume(volume)

    model = SelfCryoETNet(**config.get("model", {})).to(device)
    load_checkpoint(config["checkpoint_path"], model=model, map_location=device)
    output = infer_single_volume(
        model,
        volume,
        device=device,
        patch_size=tuple(config.get("patch_size", (108, 108, 108))),
        stride=tuple(config.get("stride", (54, 54, 54))),
    )
    save_volume(output, config["output_path"])
    return config["output_path"]

