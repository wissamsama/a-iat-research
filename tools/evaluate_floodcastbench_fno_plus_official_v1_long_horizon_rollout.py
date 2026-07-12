from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from datasets.floodcastbench_fno_dataset import _read_raster  # noqa: E402
from datasets.floodcastbench_fno_plus_official_dataset import (  # noqa: E402
    FloodCastBenchFNOPlusOfficialDataset,
)
from datasets.floodcastbench_fno_plus_official_v1_dataset import (  # noqa: E402
    PHYSICAL_INPUT_CHANNELS,
    TARGET_KEY,
    build_fno_plus_official_v1_dataset,
)
from models.fno_plus_official import FNOPlusOfficial3d  # noqa: E402
from models.fno_plus_official_mamba import FNOPlusOfficial3dMamba  # noqa: E402


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(data: dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_model(config: dict[str, Any]) -> torch.nn.Module:
    model_config = config["model"]
    model_name = str(model_config.get("name", "fno_plus_official_v1_normalized")).lower()
    common_kwargs = {
        "input_channels": int(model_config.get("input_channels", 6)),
        "output_steps": int(model_config.get("output_steps", 19)),
        "modes": int(model_config.get("modes", 12)),
        "width": int(model_config.get("width", 20)),
        "fourier_layers": int(model_config.get("fourier_layers", 4)),
    }
    if "mamba" not in model_name:
        return FNOPlusOfficial3d(**common_kwargs, output_offset=int(model_config.get("output_offset", 1)))

    mamba_config = model_config.get("mamba", {})
    return FNOPlusOfficial3dMamba(
        **common_kwargs,
        mamba_layers=int(mamba_config.get("layers", 1)),
        d_state=int(mamba_config.get("d_state", 16)),
        d_conv=int(mamba_config.get("d_conv", 4)),
        expand=int(mamba_config.get("expand", 2)),
        residual=bool(mamba_config.get("residual", True)),
        layer_norm=bool(mamba_config.get("layer_norm", True)),
    )


class MetricAccumulator:
    def __init__(self, gammas: list[float]) -> None:
        self.gammas = tuple(float(gamma) for gamma in gammas)
        self.sse = 0.0
        self.ae = 0.0
        self.count = 0
        self.target_sum = 0.0
        self.target_sq_sum = 0.0
        self.pred_sum = 0.0
        self.pred_sq_sum = 0.0
        self.cross_sum = 0.0
        self.pred_min = math.inf
        self.pred_max = -math.inf
        self.target_min = math.inf
        self.target_max = -math.inf
        self.negative_count = 0
        self.gamma_counts = {
            gamma: {"tp": 0.0, "fp": 0.0, "fn": 0.0, "pred": 0.0, "target": 0.0}
            for gamma in self.gammas
        }

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        pred = pred.detach().float().cpu()
        target = target.detach().float().cpu()
        error = pred - target
        self.sse += float((error * error).sum().item())
        self.ae += float(error.abs().sum().item())
        self.count += int(target.numel())
        self.target_sum += float(target.sum().item())
        self.target_sq_sum += float((target * target).sum().item())
        self.pred_sum += float(pred.sum().item())
        self.pred_sq_sum += float((pred * pred).sum().item())
        self.cross_sum += float((pred * target).sum().item())
        self.pred_min = min(self.pred_min, float(pred.min().item()))
        self.pred_max = max(self.pred_max, float(pred.max().item()))
        self.target_min = min(self.target_min, float(target.min().item()))
        self.target_max = max(self.target_max, float(target.max().item()))
        self.negative_count += int((pred < 0).sum().item())

        for gamma in self.gammas:
            pred_mask = pred > gamma
            target_mask = target > gamma
            counts = self.gamma_counts[gamma]
            counts["tp"] += float((pred_mask & target_mask).sum().item())
            counts["fp"] += float((pred_mask & ~target_mask).sum().item())
            counts["fn"] += float((~pred_mask & target_mask).sum().item())
            counts["pred"] += float(pred_mask.sum().item())
            counts["target"] += float(target_mask.sum().item())

    def compute(self) -> dict[str, Any]:
        eps = 1e-12
        count = max(self.count, 1)
        target_mean = self.target_sum / count
        pred_mean = self.pred_sum / count
        target_var_sum = self.target_sq_sum - self.count * target_mean * target_mean
        pred_var_sum = self.pred_sq_sum - self.count * pred_mean * pred_mean
        covariance_sum = self.cross_sum - self.count * pred_mean * target_mean
        metrics: dict[str, Any] = {
            "samples_or_maps": "",
            "pixels": self.count,
            "classical_rmse": math.sqrt(self.sse / count),
            "current_relative_rmse": math.sqrt(self.sse / max(self.target_sq_sum, eps)),
            "nse": 1.0 - self.sse / max(target_var_sum, eps),
            "pearson_r": covariance_sum / math.sqrt(max(target_var_sum * pred_var_sum, eps)),
            "bias": pred_mean - target_mean,
            "mae": self.ae / count,
            "negative_prediction_ratio": self.negative_count / count,
            "pred_min": self.pred_min,
            "pred_max": self.pred_max,
            "pred_mean": pred_mean,
            "target_min": self.target_min,
            "target_max": self.target_max,
            "target_mean": target_mean,
        }
        for gamma, counts in self.gamma_counts.items():
            key = gamma_key(gamma)
            tp = counts["tp"]
            fp = counts["fp"]
            fn = counts["fn"]
            predicted_area = counts["pred"]
            true_area = counts["target"]
            precision = tp / max(tp + fp, eps)
            recall = tp / max(tp + fn, eps)
            metrics.update(
                {
                    f"csi_gamma_{key}": tp / max(tp + fp + fn, eps),
                    f"precision_gamma_{key}": precision,
                    f"recall_gamma_{key}": recall,
                    f"f1_gamma_{key}": 2.0 * precision * recall / max(precision + recall, eps),
                    f"tp_gamma_{key}": int(tp),
                    f"fp_gamma_{key}": int(fp),
                    f"fn_gamma_{key}": int(fn),
                    f"predicted_area_gamma_{key}": int(predicted_area),
                    f"true_area_gamma_{key}": int(true_area),
                    f"flooded_area_ratio_gamma_{key}": predicted_area / max(true_area, eps),
                }
            )
        return metrics


def gamma_key(gamma: float) -> str:
    return str(float(gamma)).replace(".", "_")


def build_raw_dataset(config: dict[str, Any], root: Path, split: str, stride: int, split_counts: dict[str, int]):
    dataset_config = config.get("dataset", {})
    return FloodCastBenchFNOPlusOfficialDataset(
        root=root,
        event=dataset_config.get("event", "australia"),
        fidelity=dataset_config.get("fidelity", "high"),
        resolution=dataset_config.get("resolution", "60m"),
        split=split,
        sample_length=int(dataset_config.get("sample_length", 20)),
        stride=stride,
        split_counts=split_counts,
        context_length=int(dataset_config.get("context_length", 0)),
    )


def load_water_frame(dataset: FloodCastBenchFNOPlusOfficialDataset, frame_index: int) -> torch.Tensor:
    return torch.from_numpy(_read_raster(dataset.frames[frame_index].path)).float()


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = Path(args.run_dir)
    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_files = {
        "summary": output_dir / "long_horizon_summary.json",
        "by_horizon": output_dir / "long_horizon_metrics_by_horizon.csv",
        "per_step": output_dir / "long_horizon_metrics_per_step.csv",
        "path": output_dir / "long_horizon_path_metrics.csv",
        "protocol": output_dir / "long_horizon_rollout_protocol.txt",
    }
    existing = [str(path) for path in output_files.values() if path.exists()]
    if existing and not args.force:
        raise FileExistsError("Refusing to overwrite existing output files: " + ", ".join(existing))

    config = load_yaml(run_dir / "config.yaml")
    normalization_stats = load_json(run_dir / "normalization_stats.json")
    dataset_root = Path(config.get("paths", {}).get("dataset_root", "/home/wissam/utem-workspace/data/FloodCastBench"))
    stride = int(config.get("dataset", {}).get("stride", 20))

    # Compatibility check: instantiate the official-v1 test dataset using the exact run stats.
    _ = build_fno_plus_official_v1_dataset(dataset_root, config, "test", normalization_stats)

    context_length = int(config.get("dataset", {}).get("context_length", 0))
    sample_length = int(config.get("dataset", {}).get("sample_length", 20))
    window_length = context_length + sample_length

    test_raw = build_raw_dataset(
        config,
        dataset_root,
        split="test",
        stride=stride,
        split_counts=config.get("dataset", {}).get("split_counts"),
    )
    water_frame_count = len(test_raw.frames)
    full_raw = build_raw_dataset(
        config,
        dataset_root,
        split="train",
        stride=1,
        split_counts={"train": water_frame_count - window_length + 1, "val": 0, "test": 0},
    )

    test_starts = [
        int(test_raw[index][2]["global_sample_index"]) * stride + context_length for index in range(len(test_raw))
    ]
    horizons = [int(horizon) for horizon in args.horizons]
    horizon_labels = {
        horizon: "T+20_paper_t20_direct" if horizon == 19 else f"T+{horizon}"
        for horizon in horizons
    }
    output_steps = int(config.get("model", {}).get("output_steps", 19))

    def is_valid_rollout_start(start: int, horizon: int) -> bool:
        target_exists = start + horizon < water_frame_count
        last_chunk_start = start + ((horizon - 1) // output_steps) * output_steps
        last_input_window_exists = last_chunk_start + sample_length <= water_frame_count
        has_history = start >= context_length
        return target_exists and last_input_window_exists and has_history

    valid_starts = {
        horizon_labels[horizon]: [start for start in test_starts if is_valid_rollout_start(start, horizon)]
        for horizon in horizons
    }

    stats = normalization_stats["channels"]

    def normalize_channel(name: str, tensor: torch.Tensor) -> torch.Tensor:
        channel_stats = stats[name]
        return (tensor - float(channel_stats["mean"])) / float(channel_stats["std"])

    def inverse_target(tensor: torch.Tensor) -> torch.Tensor:
        target_stats = stats[TARGET_KEY]
        return tensor * float(target_stats["std"]) + float(target_stats["mean"])

    def make_rollout_input(
        window_start: int, history_frames: list[torch.Tensor], current_depth: torch.Tensor
    ) -> torch.Tensor:
        # window_start (= current_start - context_length) is only used to pull
        # DEM/rainfall/X/Y/T -- channels that are always known regardless of
        # rollout progress. The depth channel is built explicitly from
        # `history_frames` (a rolling buffer of the last context_length
        # frames, real at first and increasingly the model's OWN past
        # predictions as the rollout advances past the original start -- see
        # the caller) plus `current_depth` broadcast across the current+target
        # positions, exactly mirroring the training-time convention (see
        # FloodCastBenchFNOPlusOfficialDataset.__getitem__). Never re-reads
        # ground truth for positions beyond the rollout's original start --
        # that would leak future information a real deployment wouldn't have.
        x, _, _ = full_raw[window_start]
        x = x.clone()
        depth_channel = PHYSICAL_INPUT_CHANNELS["initial_depth"]
        if context_length > 0:
            x[depth_channel, :, :, :context_length] = torch.stack(history_frames, dim=-1)
        x[depth_channel, :, :, context_length:] = current_depth.unsqueeze(-1).expand(
            -1, -1, x.shape[-1] - context_length
        )
        for name, channel_index in PHYSICAL_INPUT_CHANNELS.items():
            x[channel_index] = normalize_channel(name, x[channel_index])
        return x

    def target_sequence(start_index: int, steps: int) -> torch.Tensor:
        return torch.stack(
            [load_water_frame(full_raw, start_index + offset) for offset in range(1, steps + 1)],
            dim=0,
        )

    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_model(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = (
        checkpoint.get("model_state_dict")
        or checkpoint.get("model_state")
        or checkpoint.get("state_dict")
        or checkpoint
    )
    model.load_state_dict(state)
    model.eval()
    checkpoint_epoch = checkpoint.get("epoch") if isinstance(checkpoint, dict) else None

    needed_by_start = {
        start: max(horizon for horizon in horizons if start in valid_starts[horizon_labels[horizon]])
        for start in set(start for starts in valid_starts.values() for start in starts)
    }

    rollouts: dict[int, dict[str, torch.Tensor]] = {}
    with torch.no_grad():
        for start in sorted(needed_by_start):
            steps = min(needed_by_start[start], water_frame_count - start - 1)
            current_depth = load_water_frame(full_raw, start)
            history_frames = [
                load_water_frame(full_raw, start - context_length + offset) for offset in range(context_length)
            ]
            current_start = start
            produced = 0
            pred_chunks: list[torch.Tensor] = []
            while produced < steps:
                window_start = current_start - context_length
                x = make_rollout_input(window_start, history_frames, current_depth).unsqueeze(0).to(device)
                pred_norm = model(x).squeeze(0).squeeze(0)
                pred_physical = inverse_target(pred_norm).detach().float().cpu().permute(2, 0, 1)
                take = min(pred_physical.shape[0], steps - produced)
                pred_chunks.append(pred_physical[:take])
                produced += take
                if context_length > 0:
                    # Roll the history buffer forward: the old current_depth
                    # plus all but the last of this chunk's predictions become
                    # the new (self-predicted, once past the original start)
                    # history, keeping only the most recent context_length.
                    combined = history_frames + [current_depth] + [pred_physical[j] for j in range(take - 1)]
                    history_frames = combined[-context_length:]
                current_depth = pred_physical[take - 1]
                current_start += take
            rollouts[start] = {
                "initial": load_water_frame(full_raw, start),
                "pred": torch.cat(pred_chunks, dim=0),
                "target": target_sequence(start, steps),
            }

    by_horizon_rows: list[dict[str, Any]] = []
    path_rows: list[dict[str, Any]] = []
    for horizon in horizons:
        label = horizon_labels[horizon]
        starts = valid_starts[label]
        acc = MetricAccumulator(args.gammas)
        for start in starts:
            acc.update(rollouts[start]["pred"][horizon - 1], rollouts[start]["target"][horizon - 1])
        row = {
            "checkpoint_name": args.checkpoint_name,
            "horizon_label": label,
            "horizon_steps": horizon,
            "rollout_samples": len(starts),
            "start_indices": ";".join(str(start) for start in starts),
        }
        row.update(acc.compute())
        by_horizon_rows.append(row)

        for gamma in args.gammas:
            path_rows.append(
                compute_path_metrics(
                    checkpoint_name=args.checkpoint_name,
                    horizon_label=label,
                    horizon_steps=horizon,
                    starts=starts,
                    rollouts=rollouts,
                    gamma=float(gamma),
                )
            )

    per_step_rows: list[dict[str, Any]] = []
    max_horizon = max(horizons)
    for step in range(1, max_horizon + 1):
        starts = [start for start in test_starts if start in rollouts and step <= rollouts[start]["pred"].shape[0]]
        if not starts:
            continue
        acc = MetricAccumulator(args.gammas)
        for start in starts:
            acc.update(rollouts[start]["pred"][step - 1], rollouts[start]["target"][step - 1])
        row = {"checkpoint_name": args.checkpoint_name, "step": step, "rollout_samples": len(starts)}
        row.update(acc.compute())
        per_step_rows.append(row)

    protocol = (
        "Official-v1 normalized FNO+ long-horizon autoregressive extension. "
        "H=19 is reported as T+20/paper t20 direct because the model outputs t=2..20. "
        "True initial depth at t0 is used. The model predicts 19 future physical maps after inverse-transform; "
        "the last predicted physical map becomes the next initial depth and is broadcast over the 20-step input time axis. "
        "Rainfall/DEM/coordinate inputs are rebuilt from the advanced dataset frame index. "
        "Target maps are read directly from the physical water-depth frame list, so no complete future dataset window is required. "
        "Metrics use raw inverse-transformed predictions with no clipping, threshold cleanup, or post-processing."
    )

    summary = {
        "script": str(Path(__file__).resolve()),
        "checkpoint_name": args.checkpoint_name,
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint_epoch,
        "run_dir": str(run_dir),
        "config_path": str(run_dir / "config.yaml"),
        "normalization_stats_path": str(run_dir / "normalization_stats.json"),
        "output_dir": str(output_dir),
        "device": str(device),
        "horizons": horizons,
        "horizon_labels": horizon_labels,
        "gammas": args.gammas,
        "test_start_indices": test_starts,
        "valid_starts_by_horizon": valid_starts,
        "water_frame_count": water_frame_count,
        "rainfall_frame_count": len(full_raw.rainfall_frames),
        "no_clipping": True,
        "target_reading": "direct_from_full_raw.frames_with__read_raster",
        "initial_depth_rollout_broadcast": True,
        "produced_step_counter": True,
        "protocol": protocol,
        "metrics_by_horizon": by_horizon_rows,
        "path_metrics": path_rows,
    }

    save_json(summary, output_files["summary"])
    write_csv(output_files["by_horizon"], by_horizon_rows)
    write_csv(output_files["per_step"], per_step_rows)
    write_csv(output_files["path"], path_rows)
    output_files["protocol"].write_text(protocol + "\n", encoding="utf-8")
    return summary


def compute_path_metrics(
    checkpoint_name: str,
    horizon_label: str,
    horizon_steps: int,
    starts: list[int],
    rollouts: dict[int, dict[str, torch.Tensor]],
    gamma: float,
) -> dict[str, Any]:
    tp = fp = fn = 0.0
    prop_tp = prop_fp = prop_fn = 0.0
    valid_steps = 0
    empty_true_steps = 0
    empty_pred_steps = 0

    for start in starts:
        initial_mask = rollouts[start]["initial"] > gamma
        pred_final = rollouts[start]["pred"][horizon_steps - 1] > gamma
        target_final = rollouts[start]["target"][horizon_steps - 1] > gamma
        pred_path = pred_final & ~initial_mask
        target_path = target_final & ~initial_mask
        tp += float((pred_path & target_path).sum().item())
        fp += float((pred_path & ~target_path).sum().item())
        fn += float((~pred_path & target_path).sum().item())

        prev_pred = initial_mask
        prev_target = initial_mask
        for step in range(horizon_steps):
            pred_mask = rollouts[start]["pred"][step] > gamma
            target_mask = rollouts[start]["target"][step] > gamma
            pred_new = pred_mask & ~prev_pred
            target_new = target_mask & ~prev_target
            if int(pred_new.sum().item()) == 0:
                empty_pred_steps += 1
            if int(target_new.sum().item()) == 0:
                empty_true_steps += 1
            if int(pred_new.sum().item()) > 0 or int(target_new.sum().item()) > 0:
                valid_steps += 1
            prop_tp += float((pred_new & target_new).sum().item())
            prop_fp += float((pred_new & ~target_new).sum().item())
            prop_fn += float((~pred_new & target_new).sum().item())
            prev_pred = pred_mask
            prev_target = target_mask

    eps = 1e-12
    precision = tp / max(tp + fp, eps)
    recall = tp / max(tp + fn, eps)
    prop_precision = prop_tp / max(prop_tp + prop_fp, eps)
    prop_recall = prop_tp / max(prop_tp + prop_fn, eps)
    return {
        "checkpoint_name": checkpoint_name,
        "horizon_label": horizon_label,
        "horizon_steps": horizon_steps,
        "gamma": gamma,
        "samples": len(starts),
        "path_iou": tp / max(tp + fp + fn, eps),
        "path_precision": precision,
        "path_recall": recall,
        "path_f1": 2.0 * precision * recall / max(precision + recall, eps),
        "path_tp": int(tp),
        "path_fp": int(fp),
        "path_fn": int(fn),
        "path_predicted_area": int(tp + fp),
        "path_true_area": int(tp + fn),
        "path_area_ratio": (tp + fp) / max(tp + fn, eps),
        "propagation_path_iou": prop_tp / max(prop_tp + prop_fp + prop_fn, eps),
        "propagation_path_precision": prop_precision,
        "propagation_path_recall": prop_recall,
        "valid_propagation_steps": valid_steps,
        "empty_true_propagation_steps": empty_true_steps,
        "empty_predicted_propagation_steps": empty_pred_steps,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-name", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--horizons", type=int, nargs="+", default=[19, 72, 144])
    parser.add_argument("--gammas", type=float, nargs="+", default=[0.001, 0.01])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--force", action="store_true", help="Allow overwriting output files.")
    return parser.parse_args()


def main() -> None:
    summary = evaluate(parse_args())
    print(
        json.dumps(
            {
                "checkpoint_name": summary["checkpoint_name"],
                "checkpoint_epoch": summary["checkpoint_epoch"],
                "output_dir": summary["output_dir"],
                "valid_starts_by_horizon": summary["valid_starts_by_horizon"],
                "metrics_by_horizon": [
                    {
                        key: row[key]
                        for key in (
                            "horizon_label",
                            "horizon_steps",
                            "rollout_samples",
                            "classical_rmse",
                            "current_relative_rmse",
                            "nse",
                            "pearson_r",
                            "csi_gamma_0_001",
                            "csi_gamma_0_01",
                            "negative_prediction_ratio",
                        )
                    }
                    for row in summary["metrics_by_horizon"]
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
