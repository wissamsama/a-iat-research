from __future__ import annotations

"""WP9 (master plan): deep ensemble of independently-trained deterministic
twins as the missing standard uncertainty-quantification baseline.

The calibration pivot's central claim -- "a deterministic twin cannot
provide uncertainty" -- is false as stated: an ensemble of N independently
trained deterministic models (Lakshminarayanan et al. 2017) is the standard
UQ baseline and is a valid comparison point for V2's M-scenario diffusion
ensemble. The N twin checkpoints (seeds 42/7/123) are already trained; this
script only evaluates them jointly.

Design: for a given window, every twin runs one deterministic forward
rollout on the SAME sample (same sensor mask, drawn once from the dataset's
fixed eval mask bank, reused across all members -- mirrors V2's own
convention of one mask shared across its M diffusion scenarios). The N
single-scenario outputs are stacked into an [N, l, H, W] ensemble and fed
into the same MetricAccumulator / CalibrationAccumulator machinery used for
V2, so the two families are compared on identical metrics and identical
finite-ensemble-bias corrections.
"""

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from datasets.floodcastbench_diff_sparse_v2_dataset import build_diff_sparse_v2_dataset  # noqa: E402
from models.deterministic_twin import build_v2_family_model  # noqa: E402
from tools.evaluate_floodcastbench_diff_sparse_v2 import (  # noqa: E402
    CalibrationAccumulator,
    MultiHorizonPathAccumulator,
    load_checkpoint,
    rollout_window,
)
from tools.evaluate_floodcastbench_fno_plus_official_v1_long_horizon_rollout import (  # noqa: E402
    MetricAccumulator as OfficialMetricAccumulator,
    write_csv as write_dynamic_csv,
)
from tools.evaluate_floodcastbench_diff_sparse_v1 import (  # noqa: E402
    MetricAccumulator,
    persistence_forecast,
    to_physical,
    unique_dir,
)
from tools.train_floodcastbench_diff_sparse_v1 import (  # noqa: E402
    cli_args_for_summary,
    command_reconstruction,
    git_status_short,
    load_config,
    path_from_config,
    resolve_device,
    resolve_path,
    save_json,
)
from tools.train_floodcastbench_diff_sparse_v2 import DeltaSpec, load_delta_stats  # noqa: E402
from training.utils import set_seed  # noqa: E402

EVAL_STATUS = "det_twin_deep_ensemble_wp9_floodcastbench_rollout_eval"
DOES_NOT_CLAIM = [
    "official FloodCastBench benchmark performance",
    "official DIFF-SPARSE TideWatch reproduction",
    "physical aleatoric uncertainty (see calibration caveat)",
]
OFFICIAL_GAMMAS = [0.001, 0.01]
# Fixed independently of any member's training seed: guarantees every twin
# in the ensemble sees the identical eval mask bank draw per window, which is
# what makes the ensemble comparable to V2's M-scenarios-share-one-mask setup.
EVAL_MASK_SEED = 42

STEP_FIELDS = [
    "split",
    "step",
    "horizon_label",
    "nrmse",
    "rmse_normalized",
    "mae_normalized",
    "rmse_physical_m",
    "mae_physical_m",
    "nacrps",
    "persistence_nrmse",
    "persistence_rmse_normalized",
    "persistence_mae_normalized",
    "persistence_rmse_physical_m",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="WP9: deep-ensemble-of-twins calibration evaluation.")
    parser.add_argument("--config", type=Path, required=True, help="Base config shared by every member (masking/eval params must match across seeds).")
    parser.add_argument("--checkpoint", type=Path, nargs="+", required=True, help=">=2 independently-trained twin checkpoints (different seeds).")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--max-windows", type=int)
    parser.add_argument("--tile-chunk", type=int, default=64)
    parser.add_argument("--tile-stride", type=int)
    parser.add_argument("--persistence-mode", choices=["oracle", "sparse"], default="oracle")
    parser.add_argument("--missing-rate", type=float, help="Override eval sparsity (must match what the checkpoints were trained/selected for).")
    parser.add_argument("--mask-structure", choices=["random", "gauge", "cluster"])
    parser.add_argument("--no-clamp-physical", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    if len(args.checkpoint) < 2:
        raise ValueError("WP9 requires >= 2 independently-trained checkpoints to form a deep ensemble.")

    config = load_config(args.config)
    if args.missing_rate is not None:
        config.setdefault("masking", {})["missing_rate"] = float(args.missing_rate)
        print(f"NOTE: eval missing_rate overridden to {args.missing_rate}")
    if args.mask_structure is not None:
        config.setdefault("masking", {})["eval_mask_structure"] = args.mask_structure
        print(f"NOTE: eval mask structure overridden to {args.mask_structure}")
    set_seed(EVAL_MASK_SEED)

    device = resolve_device(args.device)

    checkpoints = [load_checkpoint(path) for path in args.checkpoint]
    water_stats = checkpoints[0]["normalization_stats"]["channels"]["water"]
    for index, checkpoint in enumerate(checkpoints[1:], start=1):
        other = checkpoint["normalization_stats"]["channels"]["water"]
        if other != water_stats:
            raise ValueError(
                f"normalization stats mismatch between checkpoint 0 ({args.checkpoint[0]}) and "
                f"{index} ({args.checkpoint[index]}) -- not a fair ensemble (different train splits/stats)."
            )

    dataset = build_diff_sparse_v2_dataset(
        path_from_config(config, "dataset_root"),
        config,
        split=args.split,
        normalization_stats=checkpoints[0]["normalization_stats"],
        patch_mode="full",
    )

    models = []
    for checkpoint_path, checkpoint in zip(args.checkpoint, checkpoints):
        model = build_v2_family_model(config).to(device)
        use_ema = bool(config.get("evaluation", {}).get("use_ema", True)) and checkpoint.get("ema_state_dict")
        if use_ema:
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    parameter.copy_(checkpoint["ema_state_dict"][name])
        else:
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.eval()
        print(f"member: {checkpoint_path.parent.name} epoch={checkpoint.get('epoch')} ema={use_ema}")
        models.append(model)
    num_members = len(models)

    prediction_config = config.get("prediction", {})
    delta_stats = checkpoints[0].get("delta_stats") or load_delta_stats(
        Path(prediction_config["delta_stats_json"]) if prediction_config.get("delta_stats_json") else None
    )
    delta = DeltaSpec(str(prediction_config.get("target", "delta")), water_stats, delta_stats)

    evaluation_config = config.get("evaluation", {})
    remask = bool(evaluation_config.get("rollout_remask", False))
    clamp_physical = bool(evaluation_config.get("clip_x0_physical", True)) and not args.no_clamp_physical

    if args.output_dir is not None:
        output_base = resolve_path(args.output_dir, PROJECT_DIR)
    else:
        experiment_root = path_from_config(config, "experiment_root")
        timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        output_base = experiment_root / "wp9_det_twin_ensemble" / f"eval_{args.split}_m{num_members}_{timestamp}"
    output_dir = unique_dir(output_base)

    total_windows = len(dataset)
    windows = min(total_windows, args.max_windows) if args.max_windows else total_windows
    patch_size = dataset.patch_size
    tile_stride = int(args.tile_stride or max(1, patch_size // 2))
    prediction_length = dataset.prediction_length

    print(f"code_root: {PROJECT_DIR}")
    print(f"members: {[str(p) for p in args.checkpoint]}")
    print(f"output_dir: {output_dir}")
    print(f"split: {args.split} windows: {windows}/{total_windows}")
    print(f"num_members: {num_members} missing_rate: {dataset.missing_rate} mask_mode: {dataset.mask_mode}")
    print(
        f"patch_size: {patch_size} tile_stride: {tile_stride} "
        f"persistence_mode: {args.persistence_mode} rollout_remask: {remask} "
        f"target_space: {delta.mode} (scale={delta.scale:.6g}) clamp_physical: {clamp_physical}"
    )

    model_metrics = MetricAccumulator(prediction_length)
    persistence_metrics = MetricAccumulator(prediction_length)
    official_overall_accumulator = OfficialMetricAccumulator(gammas=OFFICIAL_GAMMAS)
    official_step_accumulators = {
        step: OfficialMetricAccumulator(gammas=OFFICIAL_GAMMAS) for step in range(prediction_length)
    }
    path_accumulator = MultiHorizonPathAccumulator(prediction_length, gammas=OFFICIAL_GAMMAS)
    calibration_accumulator = CalibrationAccumulator(prediction_length, gammas=OFFICIAL_GAMMAS, num_members=num_members)

    for window_index in range(windows):
        sample = dataset[window_index]
        target = sample["target"].to(device)

        member_outputs = []
        for model in models:
            generator = torch.Generator(device=device).manual_seed(EVAL_MASK_SEED * 1_000_003 + window_index)
            prediction = rollout_window(
                model,
                sample,
                num_scenarios=1,
                patch_size=patch_size,
                tile_stride=tile_stride,
                tile_chunk=args.tile_chunk,
                remask=remask,
                mask_mode=dataset.mask_mode,
                delta=delta,
                clamp=clamp_physical,
                generator=generator,
                device=device,
            )
            member_outputs.append(prediction[0])
        predictions = torch.stack(member_outputs, dim=0)  # [M, l, H, W]
        if not torch.isfinite(predictions).all():
            raise FloatingPointError(f"Non-finite ensemble predictions in window {window_index}")

        mean_forecast = predictions.mean(dim=0)
        median_forecast = predictions.median(dim=0).values
        if clamp_physical:
            floor = delta.floor_absolute
            mean_forecast = mean_forecast.clamp(min=floor)
            median_forecast = median_forecast.clamp(min=floor)
            predictions = predictions.clamp(min=floor)
        persistence = persistence_forecast(sample, prediction_length, device, args.persistence_mode)

        model_metrics.update(mean_forecast, predictions, target)
        persistence_metrics.update(persistence, None, target)
        mean_forecast_physical = to_physical(mean_forecast, water_stats)
        if clamp_physical:
            mean_forecast_physical = mean_forecast_physical.clamp(min=0.0)
        target_physical = to_physical(target, water_stats)
        official_overall_accumulator.update(mean_forecast_physical, target_physical)
        for step in range(prediction_length):
            official_step_accumulators[step].update(mean_forecast_physical[step], target_physical[step])
        initial_physical = to_physical(sample["context_water_true"][-1].to(device), water_stats)
        path_accumulator.update(mean_forecast_physical, target_physical, initial_physical)

        scenarios_physical = to_physical(predictions, water_stats)
        if clamp_physical:
            scenarios_physical = scenarios_physical.clamp(min=0.0)
        calibration_accumulator.update(scenarios_physical, target_physical, window_index)

        print(f"window {window_index + 1}/{windows} done", flush=True)

    water_std = float(water_stats["std"])
    model_result = model_metrics.finalize(water_std)
    persistence_result = persistence_metrics.finalize(water_std)

    rows = []
    for step in range(prediction_length):
        model_step = model_result["per_step"][step]
        persistence_step = persistence_result["per_step"][step]
        rows.append(
            {
                "split": args.split,
                "step": step + 1,
                "horizon_label": f"h{dataset.context_length + step + 1}",
                "nrmse": model_step["nrmse"],
                "rmse_normalized": model_step["rmse_normalized"],
                "mae_normalized": model_step["mae_normalized"],
                "rmse_physical_m": model_step["rmse_physical_m"],
                "mae_physical_m": model_step["mae_physical_m"],
                "nacrps": model_step["nacrps"],
                "persistence_nrmse": persistence_step["nrmse"],
                "persistence_rmse_normalized": persistence_step["rmse_normalized"],
                "persistence_mae_normalized": persistence_step["mae_normalized"],
                "persistence_rmse_physical_m": persistence_step["rmse_physical_m"],
            }
        )
    metrics_path = output_dir / "eval_metrics_per_step.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=STEP_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    path_rows = path_accumulator.per_step_metrics()
    official_rows = []
    for step in range(prediction_length):
        horizon_label = f"h{dataset.context_length + step + 1}"
        official_step = official_step_accumulators[step].compute()
        official_step.update(
            {
                "checkpoint_names": [p.parent.name for p in args.checkpoint],
                "step": dataset.context_length + step + 1,
                "rollout_step": step + 1,
                "horizon_label": horizon_label,
                "rollout_samples": windows,
            }
        )
        official_step.update(
            {key: value for key, value in path_rows[step].items() if key not in ("rollout_step", "samples")}
        )
        official_rows.append(official_step)
    official_metrics_path = output_dir / "eval_metrics_official_per_step.csv"
    write_dynamic_csv(official_metrics_path, official_rows)

    official_overall = official_overall_accumulator.compute()
    pooled_propagation = path_accumulator.pooled_propagation()

    model_overall = model_result["overall"]
    persistence_overall = persistence_result["overall"]
    improvement = None
    if persistence_overall["rmse_normalized"] > 0:
        improvement = 100.0 * (
            persistence_overall["rmse_normalized"] - model_overall["rmse_normalized"]
        ) / persistence_overall["rmse_normalized"]

    calibration = calibration_accumulator.summary()
    save_json(calibration, output_dir / "eval_calibration.json")

    summary: dict[str, Any] = {
        "config_path": str(args.config),
        "checkpoint_paths": [str(p) for p in args.checkpoint],
        "checkpoint_epochs": [c.get("epoch") for c in checkpoints],
        "ensemble_type": "deep_ensemble_independent_seeds",
        "num_members": num_members,
        "split": args.split,
        "windows_evaluated": windows,
        "windows_total": total_windows,
        "missing_rate": dataset.missing_rate,
        "mask_mode": dataset.mask_mode,
        "eval_mask_bank_size": dataset.eval_mask_bank_size,
        "eval_mask_seed": EVAL_MASK_SEED,
        "context_length": dataset.context_length,
        "prediction_length": prediction_length,
        "patch_size": patch_size,
        "tile_stride": tile_stride,
        "persistence_mode": args.persistence_mode,
        "rollout_remask": remask,
        "target_space": delta.mode,
        "clamp_physical": clamp_physical,
        "device": str(device),
        "model": model_result,
        "persistence": persistence_result,
        "official_metrics_physical": {
            "units": "meters",
            "gammas_m": OFFICIAL_GAMMAS,
            "source": "mean deep-ensemble-of-twins forecast and target inverse-transformed with shared train water stats",
            "overall": official_overall,
            "per_step": official_rows,
            "per_step_csv": str(official_metrics_path),
            "pooled_propagation": pooled_propagation,
        },
        "rmse_improvement_percent_vs_persistence": improvement,
        "eval_metrics_per_step_csv": str(metrics_path),
        "eval_metrics_official_per_step_csv": str(official_metrics_path),
        "output_directory": str(output_dir),
        "calibration": {
            "file": str(output_dir / "eval_calibration.json"),
            "num_members": calibration["num_members"],
            "coverage_pooled": {name: entry["pooled"] for name, entry in calibration["coverage"].items()},
            "caveat": calibration["caveat"],
        },
        "command_reconstruction": command_reconstruction(),
        "git_status_short": git_status_short(),
        "cli_args": {
            **cli_args_for_summary(args),
            "checkpoint": [str(p) for p in args.checkpoint],
        },
        "scientific_status": EVAL_STATUS,
        "does_not_claim": DOES_NOT_CLAIM,
    }
    save_json(summary, output_dir / "eval_summary.json")
    print("=== WP9 DEEP-ENSEMBLE-OF-TWINS EVAL ===")
    print(f"num_members={num_members} missing_rate={dataset.missing_rate}")
    print(f"coverage_pooled: {summary['calibration']['coverage_pooled']}")
    nominal_finite_ensemble = {name: entry["nominal_finite_ensemble"] for name, entry in calibration["coverage"].items()}
    print(f"nominal_finite_ensemble: {nominal_finite_ensemble}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
