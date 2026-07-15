"""WPB3 (reports/fno_plus_beat_paper_plan.md): diagnose why the naive latent-
Mamba FNO+ variant underperforms vanilla, by stratifying test-split relRMSE
into wet (target > gamma) and dry (target <= gamma) pixels for both models.

No retraining -- runs each already-trained checkpoint once on the test split.
Distinguishes two hypotheses: (a) the gap is dominated by dry-region noise
(fixable by a small output-magnitude fix), or (b) the gap is spread across
wet pixels too (the temporal-mixing placement itself hurts real predictions,
not just baseline noise).
"""

from __future__ import annotations

import argparse
import json
import math
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
from models.fno_plus_official_mamba import FNOPlusOfficial3dMamba  # noqa: E402


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


class StratifiedAccumulator:
    """Like PhysicalMetricAccumulator, but keeps separate SSE/count/negative
    tallies for wet (target > gamma) and dry (target <= gamma) pixels, plus
    the pooled total -- so a single eval pass yields both the usual pooled
    relRMSE and the wet/dry breakdown in one go."""

    def __init__(self, gamma: float = 0.001) -> None:
        self.gamma = float(gamma)
        self.strata = {
            "pooled": {"sse": 0.0, "count": 0, "target_sq_sum": 0.0, "negative": 0},
            "wet": {"sse": 0.0, "count": 0, "target_sq_sum": 0.0, "negative": 0},
            "dry": {"sse": 0.0, "count": 0, "target_sq_sum": 0.0, "negative": 0},
        }

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        pred = pred.detach().float().cpu()
        target = target.detach().float().cpu()
        error = pred - target
        wet_mask = target > self.gamma

        for name, mask in (("pooled", None), ("wet", wet_mask), ("dry", ~wet_mask)):
            s = self.strata[name]
            sel_error = error if mask is None else error[mask]
            sel_target = target if mask is None else target[mask]
            sel_pred = pred if mask is None else pred[mask]
            s["sse"] += float((sel_error**2).sum().item())
            s["count"] += int(sel_target.numel())
            s["target_sq_sum"] += float((sel_target**2).sum().item())
            s["negative"] += int((sel_pred < 0).sum().item())

    def compute(self) -> dict[str, dict[str, float]]:
        eps = 1e-12
        out: dict[str, dict[str, float]] = {}
        for name, s in self.strata.items():
            count = max(s["count"], 1)
            out[name] = {
                "pixels": s["count"],
                "current_relative_rmse": math.sqrt(s["sse"] / (s["target_sq_sum"] + eps)),
                "classical_rmse": math.sqrt(s["sse"] / count),
                "negative_prediction_ratio": s["negative"] / count,
            }
        return out


def build_model(config: dict) -> torch.nn.Module:
    model_config = config["model"]
    model_name = str(model_config.get("name", "")).lower()
    common = {
        "input_channels": int(model_config.get("input_channels", 6)),
        "output_steps": int(model_config.get("output_steps", 19)),
        "modes": int(model_config.get("modes", 12)),
        "width": int(model_config.get("width", 20)),
        "fourier_layers": int(model_config.get("fourier_layers", 4)),
    }
    if "mamba" not in model_name:
        return FNOPlusOfficial3d(**common, output_offset=int(model_config.get("output_offset", 1)))
    mamba_config = model_config.get("mamba", {})
    return FNOPlusOfficial3dMamba(
        **common,
        mamba_layers=int(mamba_config.get("layers", 1)),
        d_state=int(mamba_config.get("d_state", 16)),
        d_conv=int(mamba_config.get("d_conv", 4)),
        expand=int(mamba_config.get("expand", 2)),
        residual=bool(mamba_config.get("residual", True)),
        layer_norm=bool(mamba_config.get("layer_norm", True)),
    )


def evaluate_run(run_dir: Path, checkpoint_path: Path, device: torch.device, gamma: float) -> dict:
    config = load_yaml(run_dir / "config.yaml")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    stats = checkpoint.get("normalization_stats")
    if stats is None:
        stats = json.loads((run_dir / "normalization_stats.json").read_text(encoding="utf-8"))

    dataset = build_fno_plus_official_v1_dataset(
        config["paths"]["dataset_root"], config, split="test", normalization_stats=stats
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    accumulator = StratifiedAccumulator(gamma=gamma)
    with torch.no_grad():
        for x, target_norm, _ in loader:
            x = x.to(device)
            target_norm = target_norm.to(device)
            pred_norm = model(x)
            pred_physical = dataset.inverse_transform_target(pred_norm)
            target_physical = dataset.inverse_transform_target(target_norm)
            accumulator.update(pred_physical, target_physical)
    return {
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "gamma": gamma,
        "strata": accumulator.compute(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vanilla-run-dir", type=Path, required=True)
    parser.add_argument("--vanilla-checkpoint", type=Path, required=True)
    parser.add_argument("--mamba-run-dir", type=Path, required=True)
    parser.add_argument("--mamba-checkpoint", type=Path, required=True)
    parser.add_argument("--gamma", type=float, default=0.001)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)

    vanilla = evaluate_run(args.vanilla_run_dir, args.vanilla_checkpoint, device, args.gamma)
    mamba = evaluate_run(args.mamba_run_dir, args.mamba_checkpoint, device, args.gamma)

    result = {"vanilla": vanilla, "mamba": mamba}
    text = json.dumps(result, indent=2)
    print(text)
    if args.output:
        args.output.write_text(text, encoding="utf-8")

    print("\n=== Summary (wet/dry relRMSE breakdown) ===")
    for name, run in (("vanilla", vanilla), ("mamba", mamba)):
        strata = run["strata"]
        print(
            f"{name}: pooled={strata['pooled']['current_relative_rmse']:.5f} "
            f"wet={strata['wet']['current_relative_rmse']:.5f} "
            f"(n={strata['wet']['pixels']}) "
            f"dry={strata['dry']['current_relative_rmse']:.5f} "
            f"(n={strata['dry']['pixels']}) "
            f"neg_ratio_dry={strata['dry']['negative_prediction_ratio']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
