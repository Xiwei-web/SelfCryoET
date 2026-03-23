from typing import Dict, List

import torch


def volume_collate_fn(batch: List[Dict]) -> Dict:
    output = {}
    keys = batch[0].keys()
    for key in keys:
        value = batch[0][key]
        if torch.is_tensor(value):
            output[key] = torch.stack([item[key] for item in batch], dim=0)
        else:
            output[key] = [item[key] for item in batch]
    return output

