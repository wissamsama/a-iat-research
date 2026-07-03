from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import yaml

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from datasets.floodcastbench_fno_plus_official_dataset import FloodCastBenchFNOPlusOfficialDataset
from models.fno_plus_official import FNOPlusOfficial3d
from tools.recompute_fno_plus_official_metrics import MetricAccumulator


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def build_dataset(config: dict, split: str):
    dataset_config = config["dataset"]
    return FloodCastBenchFNOPlusOfficialDataset(
        root=config["paths"]["dataset_root"],
        event=dataset_config.get("event", "australia"),
        fidelity=dataset_config.get("fidelity", "high"),
        resolution=dataset_config.get("resolution", "60m"),
        split=split,
        sample_length=int(dataset_config.get("sample_length", 20)),
        stride=int(dataset_config.get("stride", 20)),
        split_counts=dataset_config.get("split_counts"),
    )


def build_model(config: dict) -> FNOPlusOfficial3d:
    model_config = config["model"]
    return FNOPlusOfficial3d(
        input_channels=int(model_config.get("input_channels", 6)),
        output_steps=int(model_config.get("output_steps", 19)),
        modes=int(model_config.get("modes", 12)),
        width=int(model_config.get("width", 20)),
        fourier_layers=int(model_config.get("fourier_layers", 4)),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate official FNO+ reproduction attempt v0.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    config = load_yaml(args.run_dir / "config.yaml")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    dataset = build_dataset(config, args.split)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    model = build_model(config).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    accumulator = MetricAccumulator()
    with torch.no_grad():
        for x, target, _ in loader:
            pred = model(x.to(device))
            accumulator.update(pred, target.to(device))
    result = {
        "run_dir": str(args.run_dir),
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "split": args.split,
        "samples": len(dataset),
        "metrics": accumulator.compute(),
    }
    text = json.dumps(result, indent=2)
    output = args.output or (args.run_dir / f"{args.split}_metrics_checkpoint_best_official_v0.json")
    output.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
