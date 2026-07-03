from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from datasets.floodcastbench_fno_dataset import FloodCastBenchFNODataset
from evaluation.floodcastbench_official_metrics import OfficialFloodMetricAccumulator
from models.fno_plus import FNOPlus2d


def paper_formula_rmse(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Formula as written in the prompt: mean((y - p)^2 / y^2), no square root."""

    pred = pred.float()
    target = target.float()
    return torch.mean((pred - target) ** 2 / (target**2 + eps))


def classical_rmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean((pred.float() - target.float()) ** 2))


class MetricAccumulator:
    def __init__(self) -> None:
        self.paper_sum = 0.0
        self.classical_sse = 0.0
        self.count = 0
        self.current = OfficialFloodMetricAccumulator((0.001, 0.01))

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        pred_cpu = pred.detach().float().cpu()
        target_cpu = target.detach().float().cpu()
        self.paper_sum += float(torch.sum((pred_cpu - target_cpu) ** 2 / (target_cpu**2 + 1e-12)).item())
        self.classical_sse += float(torch.sum((pred_cpu - target_cpu) ** 2).item())
        self.count += int(target_cpu.numel())
        self.current.update(pred_cpu, target_cpu)

    def compute(self) -> dict[str, float]:
        base = self.current.compute()
        return {
            "paper_formula_rmse": self.paper_sum / self.count if self.count else math.nan,
            "current_relative_rmse": base["relative_rmse"],
            "classical_rmse": math.sqrt(self.classical_sse / self.count) if self.count else math.nan,
            "nse": base["nse"],
            "pearson_r": base["pearson_r"],
            "csi_gamma_0_001": base["csi_gamma_0_001"],
            "csi_gamma_0_01": base["csi_gamma_0_01"],
            "pixels": self.count,
        }


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def main() -> int:
    parser = argparse.ArgumentParser(description="Recompute FNO+ metrics with the paper RMSE formula.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    config = load_config(args.run_dir / "config.yaml")
    dataset_config = config["dataset"]
    dataset = FloodCastBenchFNODataset(
        root=config["paths"]["dataset_root"],
        event=dataset_config.get("event", "australia"),
        fidelity=dataset_config.get("fidelity", "high"),
        resolution=dataset_config.get("resolution", "60m"),
        split=args.split,
        sample_length=int(dataset_config.get("sample_length", 20)),
        stride=int(dataset_config.get("stride", 20)),
        include_dem=bool(dataset_config.get("include_dem", True)),
        include_rainfall=bool(dataset_config.get("include_rainfall", True)),
        include_time=bool(dataset_config.get("include_time", True)),
        split_counts=dataset_config.get("split_counts"),
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    model_config = config["model"]
    model = FNOPlus2d(
        input_channels=dataset.input_channels,
        output_steps=int(model_config.get("output_steps", 19)),
        modes=int(model_config.get("modes", 12)),
        width=int(model_config.get("width", 20)),
        fourier_layers=int(model_config.get("fourier_layers", 4)),
    ).to(device)
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
        "paper_formula_note": "Formula is mean((y-p)^2/(y^2+eps)); no square root; dry cells are epsilon-stabilized.",
    }
    text = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
