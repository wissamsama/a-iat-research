from pathlib import Path

import torch

from models.registry import build_model, is_enriched_checkpoint


def load_checkpoint(path, map_location="cpu"):
    checkpoint = torch.load(Path(path), map_location=map_location)
    if not is_enriched_checkpoint(checkpoint):
        raise ValueError(
            "This .pth file is a legacy state_dict, not an enriched checkpoint. "
            "Use a migrated run checkpoint or provide the architecture manually."
        )
    return checkpoint


def load_trained_model(path, map_location="cpu", eval_mode=True):
    checkpoint = load_checkpoint(path, map_location=map_location)
    input_shape = checkpoint.get("input_shape") or [3, 32, 32]
    input_size = int(input_shape[1])
    model = build_model(checkpoint["model_name"], checkpoint["num_classes"], input_size=input_size)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(map_location if isinstance(map_location, torch.device) else torch.device(map_location))
    if eval_mode:
        model.eval()
    return model, checkpoint
