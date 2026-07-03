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

from datasets.floodcastbench_fno_plus_official_v1_dataset import build_fno_plus_official_v1_dataset  # noqa: E402
from models.fno_plus_official import FNOPlusOfficial3d  # noqa: E402
from tools.train_floodcastbench_fno_plus_official_v1 import PhysicalMetricAccumulator  # noqa: E402


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


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
    parser = argparse.ArgumentParser(description="Evaluate official FNO+ v1 normalized experiment.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    config = load_yaml(args.run_dir / "config.yaml")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    stats = checkpoint.get("normalization_stats")
    if stats is None:
        stats_path = args.run_dir / "normalization_stats.json"
        stats = json.loads(stats_path.read_text(encoding="utf-8"))

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    dataset = build_fno_plus_official_v1_dataset(
        config["paths"]["dataset_root"],
        config,
        split=args.split,
        normalization_stats=stats,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    accumulator = PhysicalMetricAccumulator((0.001, 0.01))
    with torch.no_grad():
        for x, target_norm, _ in loader:
            x = x.to(device)
            target_norm = target_norm.to(device)
            pred_norm = model(x)
            pred_physical = dataset.inverse_transform_target(pred_norm)
            target_physical = dataset.inverse_transform_target(target_norm)
            accumulator.update(pred_physical, target_physical)

    result = {
        "run_dir": str(args.run_dir),
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "split": args.split,
        "samples": len(dataset),
        "metrics": accumulator.compute(),
        "metric_units": "physical water-depth units after inverse transform",
    }
    text = json.dumps(result, indent=2)
    output = args.output or (args.run_dir / f"{args.split}_metrics_checkpoint_best_official_v1.json")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    output.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
