from __future__ import annotations

import argparse
import csv
import json
import math
import shlex
import statistics
import subprocess
import sys
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from datasets.floodcastbench_fno_dataset import (  # noqa: E402
    EVENTS,
    RAINFALL_FOLDERS,
    _event_key,
    _load_frames,
    _read_raster,
)


DEFAULT_EXPERIMENT_ROOT = Path("/home/wissam/utem-workspace/experiments/FloodCastBench")
PROTOCOL_HORIZONS = {"h50": 50, "h100": 100}

BY_HORIZON_FIELDS = [
    "protocol",
    "split",
    "horizon_index",
    "horizon_label",
    "eligible_sample_count",
    "physical_mse_m",
    "physical_rmse_m",
    "physical_mae_m",
    "normalized_mse_global",
    "normalized_rmse_global",
    "normalized_mae_global",
    "direct_normalized_mse_if_direct_horizon",
    "direct_normalized_rmse_if_direct_horizon",
    "direct_normalized_mae_if_direct_horizon",
    "pixel_count",
]

PER_SAMPLE_FIELDS = [
    "protocol",
    "split",
    "sample_index",
    "global_sample_index",
    "start_index",
    "input_timestamp",
    "horizon_index",
    "horizon_label",
    "physical_mse_m",
    "physical_rmse_m",
    "physical_mae_m",
    "normalized_mse_global",
    "normalized_rmse_global",
    "normalized_mae_global",
    "direct_normalized_mse_if_direct_horizon",
    "direct_normalized_rmse_if_direct_horizon",
    "direct_normalized_mae_if_direct_horizon",
    "pixel_count",
]

DOES_NOT_CLAIM = [
    "DIFF-SPARSE evaluation",
    "official FloodCastBench benchmark performance",
    "physical-unit forecast skill",
    "sparse-sensor robustness",
    "long-horizon model performance",
]


class RasterCache:
    def __init__(self, max_items: int = 512) -> None:
        self.max_items = int(max_items)
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()

    def get(self, path: Path) -> np.ndarray:
        key = str(path)
        if key in self._cache:
            value = self._cache.pop(key)
            self._cache[key] = value
            return value
        value = _read_raster(path)
        self._cache[key] = value
        if len(self._cache) > self.max_items:
            self._cache.popitem(last=False)
        return value


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def save_json(data: dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def command_reconstruction() -> str:
    return " ".join(shlex.quote(part) for part in [sys.executable, *sys.argv])


def git_status_short() -> list[str]:
    try:
        completed = subprocess.run(
            ["git", "status", "--short"],
            cwd=PROJECT_DIR,
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception as exc:
        return [f"git status --short failed: {exc!r}"]
    if completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip() or f"returncode={completed.returncode}"
        return [f"git status --short failed: {message}"]
    return [line for line in completed.stdout.splitlines() if line.strip()]


def safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in str(value)).strip("_")


def experiment_suffix(config: dict[str, Any]) -> str:
    return safe_name(config.get("experiment", {}).get("name", "fcb_diff_sparse_dense_missing0_highfid_60m"))


def latest_experiment_dir(config: dict[str, Any]) -> Path:
    root = Path(config.get("paths", {}).get("experiment_root", DEFAULT_EXPERIMENT_ROOT))
    suffix = experiment_suffix(config)
    candidates = sorted(
        [path for path in root.glob(f"*_{suffix}") if path.is_dir()],
        key=lambda path: (path.stat().st_mtime, path.name),
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No matching experiment directory found under {root} for suffix {suffix!r}")
    return candidates[0]


def unique_dir(path: Path) -> Path:
    candidate = path
    attempt = 1
    while candidate.exists():
        attempt += 1
        candidate = path.with_name(f"{path.name}_r{attempt}")
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def cli_args_for_summary(args: argparse.Namespace) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            result[key] = str(value)
        else:
            result[key] = value
    return result


def water_dir(data_root: Path, config: dict[str, Any]) -> Path:
    dataset_config = config.get("dataset", {})
    fidelity = str(dataset_config.get("fidelity", "high")).lower()
    if fidelity == "high":
        family = "High-fidelity flood forecasting"
    elif fidelity == "low":
        family = "Low-fidelity flood forecasting"
    else:
        raise ValueError(f"Unsupported fidelity {fidelity!r}")
    event_key = _event_key(dataset_config.get("event", "australia"))
    event = EVENTS[event_key]
    resolution = str(dataset_config.get("resolution", "60m")).lower()
    path = data_root / family / resolution / event
    if not path.exists():
        raise FileNotFoundError(f"Water-depth folder not found: {path}")
    return path


def rainfall_dir(data_root: Path, config: dict[str, Any]) -> Path:
    event_key = _event_key(config.get("dataset", {}).get("event", "australia"))
    path = data_root / "Relevant data" / "Rainfall" / RAINFALL_FOLDERS[event_key]
    if not path.exists():
        raise FileNotFoundError(f"Rainfall folder not found: {path}")
    return path


def split_ranges(split_counts: dict[str, int]) -> dict[str, tuple[int, int]]:
    train = int(split_counts.get("train", 0))
    val = int(split_counts.get("val", 0))
    test = int(split_counts.get("test", 0))
    return {
        "train": (0, train),
        "val": (train, train + val),
        "test": (train + val, train + val + test),
    }


def horizon_label(horizon: int) -> str:
    return f"h{horizon:02d}"


def empty_accumulator() -> dict[str, float]:
    return {"sum": 0.0, "sum_sq": 0.0, "abs_sum": 0.0, "count": 0.0, "min": math.inf, "max": -math.inf}


def update_value_stats(accumulator: dict[str, float], array: np.ndarray) -> None:
    values = array.astype(np.float64, copy=False)
    accumulator["sum"] += float(values.sum())
    accumulator["sum_sq"] += float(np.square(values).sum())
    accumulator["count"] += float(values.size)
    accumulator["min"] = min(accumulator["min"], float(values.min()))
    accumulator["max"] = max(accumulator["max"], float(values.max()))


def update_error_stats(accumulator: dict[str, float], prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    diff = prediction.astype(np.float64, copy=False) - target.astype(np.float64, copy=False)
    sq_sum = float(np.square(diff).sum())
    abs_sum = float(np.abs(diff).sum())
    count = float(diff.size)
    accumulator["sum_sq"] += sq_sum
    accumulator["abs_sum"] += abs_sum
    accumulator["count"] += count
    mse = sq_sum / count
    return {"mse": mse, "rmse": math.sqrt(max(mse, 0.0)), "mae": abs_sum / count, "count": int(count)}


def finalize_value_stats(accumulator: dict[str, float], label: str, horizon_range: str, train_sample_count: int) -> dict[str, Any]:
    count = accumulator["count"]
    if count <= 0:
        raise RuntimeError(f"Cannot finalize empty stats for {label}")
    mean = accumulator["sum"] / count
    variance = max(accumulator["sum_sq"] / count - mean * mean, 0.0)
    return {
        "name": label,
        "mean": float(mean),
        "std": float(math.sqrt(variance)),
        "min": float(accumulator["min"]),
        "max": float(accumulator["max"]),
        "pixel_count": int(count),
        "train_sample_count": int(train_sample_count),
        "horizon_range": horizon_range,
        "fit_split": "train",
        "train_only": True,
    }


def finalize_error_stats(accumulator: dict[str, float]) -> dict[str, float]:
    count = accumulator["count"]
    if count <= 0:
        raise RuntimeError("Cannot finalize empty error accumulator")
    mse = accumulator["sum_sq"] / count
    return {
        "mse": float(mse),
        "rmse": float(math.sqrt(max(mse, 0.0))),
        "mae": float(accumulator["abs_sum"] / count),
        "count": int(count),
    }


def normalized_from_physical(physical: dict[str, float], std: float) -> dict[str, float]:
    if std <= 0:
        raise ValueError(f"Cannot normalize with std={std}")
    return {
        "mse": float(physical["mse"] / (std * std)),
        "rmse": float(physical["rmse"] / std),
        "mae": float(physical["mae"] / std),
    }


def compute_starts(config: dict[str, Any], frame_count: int) -> dict[str, list[int]]:
    dataset_config = config.get("dataset", {})
    sample_length = int(dataset_config.get("sample_length", 20))
    stride = int(dataset_config.get("stride", 20))
    split_counts = dataset_config.get("split_counts")
    if not split_counts:
        raise ValueError("dataset.split_counts is required")
    starts = [start for start in range(0, frame_count, stride) if start + sample_length <= frame_count]
    ranges = split_ranges(split_counts)
    return {split: starts[start:end] for split, (start, end) in ranges.items()}


def eligible_starts(starts: list[int], frame_count: int, max_horizon: int) -> list[int]:
    return [start for start in starts if start + max_horizon - 1 < frame_count]


def excluded_samples(starts: list[int], frame_count: int, max_horizon: int, stride: int) -> list[dict[str, int]]:
    excluded = []
    for local_index, start in enumerate(starts):
        available = frame_count - start
        if available < max_horizon:
            excluded.append(
                {
                    "sample_index": local_index,
                    "global_sample_index": int(start // stride),
                    "start_index": start,
                    "available_max_horizon": available,
                }
            )
    return excluded


def get_frame(cache: RasterCache, frames: list[Any], index: int) -> np.ndarray:
    return cache.get(frames[index].path)


def compute_normalization_stats(
    protocol: str,
    max_horizon: int,
    train_starts: list[int],
    frames: list[Any],
    cache: RasterCache,
) -> dict[str, dict[str, Any]]:
    global_acc = empty_accumulator()
    direct_acc = empty_accumulator()
    for start in train_starts:
        for horizon in range(2, max_horizon + 1):
            target = get_frame(cache, frames, start + horizon - 1)
            update_value_stats(global_acc, target)
            if horizon == max_horizon:
                update_value_stats(direct_acc, target)
    return {
        f"target_depth_h2_{protocol}_global": finalize_value_stats(
            global_acc,
            f"target_depth_h2_{protocol}_global",
            f"h02:{protocol}",
            len(train_starts),
        ),
        f"target_depth_{protocol}_direct": finalize_value_stats(
            direct_acc,
            f"target_depth_{protocol}_direct",
            protocol,
            len(train_starts),
        ),
    }


def save_map(path: Path, array: np.ndarray, title: str, cmap: str = "viridis", vmin: float | None = None, vmax: float | None = None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(6, 5))
    artist = axis.imshow(array.astype(np.float32), cmap=cmap, vmin=vmin, vmax=vmax)
    axis.set_title(title)
    axis.axis("off")
    fig.colorbar(artist, ax=axis, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def maybe_save_maps(
    maps_dir: Path,
    protocol: str,
    split: str,
    local_sample_index: int,
    horizon: int,
    prediction: np.ndarray,
    target: np.ndarray,
) -> list[str]:
    wanted = {2, 20, 50}
    if protocol == "h100":
        wanted.add(100)
    if local_sample_index != 0 or horizon not in wanted:
        return []
    label = horizon_label(horizon)
    prefix = f"{protocol}_{split}_batch000_{label}"
    error = np.abs(prediction - target)
    scale = np.concatenate([prediction.reshape(-1), target.reshape(-1)])
    vmin = float(scale.min())
    vmax = float(scale.max())
    files = [
        (maps_dir / f"{prefix}_target.png", target, f"{protocol} {split} {label} target", "viridis", vmin, vmax),
        (
            maps_dir / f"{prefix}_persistence.png",
            prediction,
            f"{protocol} {split} {label} persistence",
            "viridis",
            vmin,
            vmax,
        ),
        (maps_dir / f"{prefix}_abs_error.png", error, f"{protocol} {split} {label} abs error", "magma", None, None),
    ]
    saved = []
    for path, array, title, cmap, map_vmin, map_vmax in files:
        save_map(path, array, title, cmap=cmap, vmin=map_vmin, vmax=map_vmax)
        saved.append(str(path))
    return saved


def evaluate_protocol_split(
    protocol: str,
    max_horizon: int,
    split: str,
    starts: list[int],
    frames: list[Any],
    cache: RasterCache,
    global_stats: dict[str, Any],
    direct_stats: dict[str, Any],
    stride: int,
    maps_dir: Path | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], list[str]]:
    horizon_accumulators = {horizon: empty_accumulator() for horizon in range(2, max_horizon + 1)}
    per_sample_rows: list[dict[str, Any]] = []
    saved_maps: list[str] = []
    for local_sample_index, start in enumerate(starts):
        prediction = get_frame(cache, frames, start)
        for horizon in range(2, max_horizon + 1):
            target = get_frame(cache, frames, start + horizon - 1)
            physical = update_error_stats(horizon_accumulators[horizon], prediction, target)
            global_norm = normalized_from_physical(physical, float(global_stats["std"]))
            direct_norm = normalized_from_physical(physical, float(direct_stats["std"])) if horizon == max_horizon else {}
            per_sample_rows.append(
                {
                    "protocol": protocol,
                    "split": split,
                    "sample_index": local_sample_index,
                    "global_sample_index": int(start // stride),
                    "start_index": start,
                    "input_timestamp": int(frames[start].timestamp),
                    "horizon_index": horizon - 1,
                    "horizon_label": horizon_label(horizon),
                    "physical_mse_m": physical["mse"],
                    "physical_rmse_m": physical["rmse"],
                    "physical_mae_m": physical["mae"],
                    "normalized_mse_global": global_norm["mse"],
                    "normalized_rmse_global": global_norm["rmse"],
                    "normalized_mae_global": global_norm["mae"],
                    "direct_normalized_mse_if_direct_horizon": direct_norm.get("mse", ""),
                    "direct_normalized_rmse_if_direct_horizon": direct_norm.get("rmse", ""),
                    "direct_normalized_mae_if_direct_horizon": direct_norm.get("mae", ""),
                    "pixel_count": physical["count"],
                }
            )
            if maps_dir is not None:
                saved_maps.extend(maybe_save_maps(maps_dir, protocol, split, local_sample_index, horizon, prediction, target))

    by_horizon_rows: list[dict[str, Any]] = []
    for horizon in range(2, max_horizon + 1):
        physical = finalize_error_stats(horizon_accumulators[horizon])
        global_norm = normalized_from_physical(physical, float(global_stats["std"]))
        direct_norm = normalized_from_physical(physical, float(direct_stats["std"])) if horizon == max_horizon else {}
        by_horizon_rows.append(
            {
                "protocol": protocol,
                "split": split,
                "horizon_index": horizon - 1,
                "horizon_label": horizon_label(horizon),
                "eligible_sample_count": len(starts),
                "physical_mse_m": physical["mse"],
                "physical_rmse_m": physical["rmse"],
                "physical_mae_m": physical["mae"],
                "normalized_mse_global": global_norm["mse"],
                "normalized_rmse_global": global_norm["rmse"],
                "normalized_mae_global": global_norm["mae"],
                "direct_normalized_mse_if_direct_horizon": direct_norm.get("mse", ""),
                "direct_normalized_rmse_if_direct_horizon": direct_norm.get("rmse", ""),
                "direct_normalized_mae_if_direct_horizon": direct_norm.get("mae", ""),
                "pixel_count": physical["count"],
            }
        )
    best = min(by_horizon_rows, key=lambda row: float(row["normalized_rmse_global"]))
    worst = max(by_horizon_rows, key=lambda row: float(row["normalized_rmse_global"]))
    average_rmse = sum(float(row["normalized_rmse_global"]) for row in by_horizon_rows) / len(by_horizon_rows)
    selected = {}
    for horizon in (2, 10, 20, 50, 100):
        if horizon <= max_horizon:
            row = by_horizon_rows[horizon - 2]
            selected[horizon_label(horizon)] = {
                "physical_rmse_m": row["physical_rmse_m"],
                "physical_mae_m": row["physical_mae_m"],
                "normalized_rmse_global": row["normalized_rmse_global"],
                "normalized_mae_global": row["normalized_mae_global"],
                "direct_normalized_rmse_if_direct_horizon": row["direct_normalized_rmse_if_direct_horizon"],
                "direct_normalized_mae_if_direct_horizon": row["direct_normalized_mae_if_direct_horizon"],
            }
    split_summary = {
        "eligible_sample_count": len(starts),
        "best_horizon": {"horizon_label": best["horizon_label"], "normalized_rmse_global": best["normalized_rmse_global"]},
        "worst_horizon": {
            "horizon_label": worst["horizon_label"],
            "normalized_rmse_global": worst["normalized_rmse_global"],
        },
        "average_normalized_rmse_global": average_rmse,
        "selected_horizons": selected,
    }
    return by_horizon_rows, per_sample_rows, split_summary, saved_maps


def protocol_arg(values: list[str]) -> list[str]:
    protocols = []
    for value in values:
        key = value.lower()
        if key not in PROTOCOL_HORIZONS:
            raise argparse.ArgumentTypeError(f"Unsupported protocol {value!r}; expected h50 or h100")
        protocols.append(key)
    return protocols


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit high-horizon persistence and normalization for FloodCastBench.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--protocols", nargs="+", default=["h50", "h100"])
    parser.add_argument("--save-maps", action="store_true")
    args = parser.parse_args()

    protocols = protocol_arg(args.protocols)
    config = yaml.safe_load(args.config.read_text())
    data_root = args.data_root
    dataset_config = config.get("dataset", {})
    split_counts = dataset_config.get("split_counts")
    if not split_counts:
        raise ValueError("dataset.split_counts is required")
    stride = int(dataset_config.get("stride", 20))
    sample_length = int(dataset_config.get("sample_length", 20))

    selected_water_dir = water_dir(data_root, config)
    selected_rainfall_dir = rainfall_dir(data_root, config)
    frames = _load_frames(selected_water_dir)
    rainfall_files = sorted(selected_rainfall_dir.glob("*.tif"))
    if not rainfall_files:
        raise FileNotFoundError(f"No rainfall files found in {selected_rainfall_dir}")

    split_starts_all = compute_starts(config, len(frames))
    experiment_dir = latest_experiment_dir(config)
    output_dir = unique_dir(
        experiment_dir / f"high_horizon_persistence_audit_{datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}"
    )
    maps_dir = output_dir / "maps"
    if args.save_maps:
        maps_dir.mkdir(parents=True, exist_ok=False)

    cache = RasterCache(max_items=512)
    normalization_stats: dict[str, Any] = {}
    eligibility: dict[str, Any] = {}
    all_by_horizon_rows: list[dict[str, Any]] = []
    all_per_sample_rows: list[dict[str, Any]] = []
    protocol_summaries: dict[str, Any] = {}
    map_files: list[str] = []

    for protocol in protocols:
        max_horizon = PROTOCOL_HORIZONS[protocol]
        eligible_by_split = {
            split: eligible_starts(starts, len(frames), max_horizon)
            for split, starts in split_starts_all.items()
        }
        excluded_by_split = {
            split: excluded_samples(starts, len(frames), max_horizon, stride)
            for split, starts in split_starts_all.items()
        }
        eligibility[protocol] = {
            split: {
                "total_config_samples": len(split_starts_all[split]),
                "eligible_count": len(eligible_by_split[split]),
                "excluded_count": len(excluded_by_split[split]),
                "excluded_samples": excluded_by_split[split],
            }
            for split in ("train", "val", "test")
        }
        if not eligible_by_split["train"]:
            raise RuntimeError(f"No eligible train samples for protocol {protocol}")

        stats = compute_normalization_stats(protocol, max_horizon, eligible_by_split["train"], frames, cache)
        normalization_stats.update(stats)
        global_stats = stats[f"target_depth_h2_{protocol}_global"]
        direct_stats = stats[f"target_depth_{protocol}_direct"]

        protocol_summaries[protocol] = {
            "max_horizon": max_horizon,
            "eligibility": eligibility[protocol],
            "normalization_stats_keys": {
                "global": f"target_depth_h2_{protocol}_global",
                "direct": f"target_depth_{protocol}_direct",
            },
            "splits": {},
        }
        for split in ("train", "val", "test"):
            rows_by_horizon, rows_per_sample, split_summary, saved_maps = evaluate_protocol_split(
                protocol,
                max_horizon,
                split,
                eligible_by_split[split],
                frames,
                cache,
                global_stats,
                direct_stats,
                stride,
                maps_dir if args.save_maps else None,
            )
            all_by_horizon_rows.extend(rows_by_horizon)
            all_per_sample_rows.extend(rows_per_sample)
            map_files.extend(saved_maps)
            protocol_summaries[protocol]["splits"][split] = split_summary

    by_horizon_path = output_dir / "high_horizon_persistence_by_horizon.csv"
    per_sample_path = output_dir / "high_horizon_persistence_per_sample.csv"
    normalization_path = output_dir / "high_horizon_normalization_stats.json"
    summary_path = output_dir / "high_horizon_persistence_summary.json"
    write_csv(by_horizon_path, all_by_horizon_rows, BY_HORIZON_FIELDS)
    write_csv(per_sample_path, all_per_sample_rows, PER_SAMPLE_FIELDS)
    save_json(
        {
            "fit_split": "train",
            "train_only": True,
            "protocols": protocols,
            "stats": normalization_stats,
        },
        normalization_path,
    )

    rainfall_max_index_needed = max(frame.timestamp // 1800 for frame in frames)
    summary = {
        "config_path": str(args.config),
        "data_root": str(data_root),
        "water_dir": str(selected_water_dir),
        "rainfall_dir": str(selected_rainfall_dir),
        "raw_water_frames": {
            "count": len(frames),
            "first_timestamp": frames[0].timestamp,
            "last_timestamp": frames[-1].timestamp,
            "frame_step_seconds_unique": sorted(set(b.timestamp - a.timestamp for a, b in zip(frames, frames[1:]))),
        },
        "rainfall_coverage": {
            "count": len(rainfall_files),
            "max_index_needed": rainfall_max_index_needed,
            "covers_all_water_timestamps_without_clamping": rainfall_max_index_needed < len(rainfall_files),
            "existing_mapping": "rainfall_index = min(water_timestamp // 1800, len(rainfall_frames) - 1)",
        },
        "dataset_windowing": {
            "sample_length_existing_config": sample_length,
            "stride": stride,
            "split_counts": split_counts,
            "note": "Metrics use raw physical water frames beyond the official-v1 h2:h20 truncation.",
        },
        "protocols": protocol_summaries,
        "normalization_stats_file": str(normalization_path),
        "by_horizon_csv": str(by_horizon_path),
        "per_sample_csv": str(per_sample_path),
        "map_files": map_files,
        "metric_units": {
            "physical": "meters_diagnostic_not_official",
            "normalized_global": "train_only_protocol_target_standardization",
            "direct_normalized": "train_only_direct_horizon_target_standardization",
        },
        "scientific_status": "high_horizon_persistence_and_normalization_audit",
        "does_not_claim": DOES_NOT_CLAIM,
        "output_dir": str(output_dir),
        "command_reconstruction": command_reconstruction(),
        "git_status_short": git_status_short(),
        "cli_args": cli_args_for_summary(args),
    }
    save_json(summary, summary_path)
    print("=== FLOODCASTBENCH HIGH-HORIZON PERSISTENCE AND NORMALIZATION AUDIT ===")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
