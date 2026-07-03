from __future__ import annotations

import argparse
import json
import shlex
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from datasets.floodcastbench_fno_dataset import (  # noqa: E402
    EVENTS,
    RAINFALL_FOLDERS,
    _event_key,
    _load_frames,
)


DEFAULT_EXPERIMENT_ROOT = Path("/home/wissam/utem-workspace/experiments/FloodCastBench")


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def save_json(data: dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


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


def summarize_horizons(values: list[int]) -> dict[str, Any]:
    if not values:
        return {
            "sample_count": 0,
            "min_available_horizon_label": None,
            "median_available_horizon_label": None,
            "max_available_horizon_label": None,
            "h20_count": 0,
            "h50_count": 0,
            "h100_count": 0,
            "h20_all": False,
            "h50_all": False,
            "h100_all": False,
        }
    sorted_values = sorted(values)
    return {
        "sample_count": len(values),
        "min_available_horizon_label": min(values),
        "median_available_horizon_label": float(statistics.median(sorted_values)),
        "max_available_horizon_label": max(values),
        "h20_count": sum(value >= 20 for value in values),
        "h50_count": sum(value >= 50 for value in values),
        "h100_count": sum(value >= 100 for value in values),
        "h20_all": all(value >= 20 for value in values),
        "h50_all": all(value >= 50 for value in values),
        "h100_all": all(value >= 100 for value in values),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit maximum available FloodCastBench future horizon.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    dataset_config = config.get("dataset", {})
    split_counts = dataset_config.get("split_counts")
    if not split_counts:
        raise ValueError("Config must contain dataset.split_counts for split-wise horizon audit")

    data_root = args.data_root
    selected_water_dir = water_dir(data_root, config)
    selected_rainfall_dir = rainfall_dir(data_root, config)
    water_frames = _load_frames(selected_water_dir)
    rainfall_files = sorted(selected_rainfall_dir.glob("*.tif"))
    if not rainfall_files:
        raise FileNotFoundError(f"No rainfall TIFF files found in {selected_rainfall_dir}")

    sample_length = int(dataset_config.get("sample_length", 20))
    stride = int(dataset_config.get("stride", 20))
    starts = [start for start in range(0, len(water_frames), stride) if start + sample_length <= len(water_frames)]
    ranges = split_ranges(split_counts)
    total_requested = sum(int(split_counts.get(split, 0)) for split in ("train", "val", "test"))
    if total_requested > len(starts):
        raise ValueError(f"Requested {total_requested} split samples, but only {len(starts)} starts exist")

    frame_step_seconds = sorted(set(b.timestamp - a.timestamp for a, b in zip(water_frames, water_frames[1:])))
    water_max_timestamp = water_frames[-1].timestamp
    rainfall_max_index_needed_for_water = max(frame.timestamp // 1800 for frame in water_frames)
    rainfall_index_available = rainfall_max_index_needed_for_water < len(rainfall_files)

    split_summaries: dict[str, Any] = {}
    all_selected_horizons: list[int] = []
    for split, (start_index, end_index) in ranges.items():
        split_starts = starts[start_index:end_index]
        horizons = [len(water_frames) - start for start in split_starts]
        all_selected_horizons.extend(horizons)
        split_summaries[split] = {
            **summarize_horizons(horizons),
            "start_indices": split_starts,
            "input_timestamp_min": water_frames[min(split_starts)].timestamp if split_starts else None,
            "input_timestamp_max": water_frames[max(split_starts)].timestamp if split_starts else None,
        }

    common_selected = summarize_horizons(all_selected_horizons)
    all_possible_horizons = [len(water_frames) - start for start in starts]

    experiment_dir = latest_experiment_dir(config)
    output_dir = unique_dir(experiment_dir / f"horizon_availability_audit_{datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}")
    summary = {
        "config_path": str(args.config),
        "data_root": str(data_root),
        "water_dir": str(selected_water_dir),
        "rainfall_dir": str(selected_rainfall_dir),
        "dataset": {
            "fidelity": dataset_config.get("fidelity", "high"),
            "event": dataset_config.get("event", "australia"),
            "resolution": dataset_config.get("resolution", "60m"),
            "sample_length": sample_length,
            "stride": stride,
            "split_counts": split_counts,
        },
        "raw_water_frames": {
            "count": len(water_frames),
            "first_timestamp": water_frames[0].timestamp,
            "last_timestamp": water_max_timestamp,
            "frame_step_seconds_unique": frame_step_seconds,
            "first_path": str(water_frames[0].path),
            "last_path": str(water_frames[-1].path),
        },
        "rainfall_frames": {
            "count": len(rainfall_files),
            "first_file": rainfall_files[0].name,
            "last_file": rainfall_files[-1].name,
            "existing_loader_mapping": "rainfall_index = min(water_timestamp // 1800, len(rainfall_frames) - 1)",
            "max_index_needed_for_all_water_frames": rainfall_max_index_needed_for_water,
            "covers_all_water_frame_timestamps_without_clamping": rainfall_index_available,
        },
        "official_v1_current_limit": {
            "sample_length": sample_length,
            "returned_target_sequence": "h2:h20",
            "target_count": sample_length - 1,
            "dataset_enforces_sample_length_20": True,
            "would_truncate_beyond_h20": True,
        },
        "window_starts": {
            "total_possible_starts_with_current_sample_length": len(starts),
            "first_start_index": starts[0],
            "last_start_index": starts[-1],
            "all_possible_starts_availability": summarize_horizons(all_possible_horizons),
        },
        "split_availability": split_summaries,
        "common_selected_split_availability": common_selected,
        "horizon_answers": {
            "beyond_h20_exists_in_raw_data": max(all_possible_horizons) > 20,
            "h50_available_from_raw_data": max(all_possible_horizons) >= 50,
            "h100_available_from_raw_data": max(all_possible_horizons) >= 100,
            "h50_available_for_all_current_train_val_test_samples": common_selected["h50_all"],
            "h100_available_for_all_current_train_val_test_samples": common_selected["h100_all"],
            "maximum_horizon_available_for_all_current_train_val_test_samples": common_selected[
                "min_available_horizon_label"
            ],
            "maximum_horizon_available_for_all_current_train_samples": split_summaries["train"][
                "min_available_horizon_label"
            ],
            "maximum_horizon_available_for_all_current_val_samples": split_summaries["val"][
                "min_available_horizon_label"
            ],
            "maximum_horizon_available_for_all_current_test_samples": split_summaries["test"][
                "min_available_horizon_label"
            ],
        },
        "feasibility": {
            "supporting_h50_or_h100_requires_new_dataset_wrapper": True,
            "reason_new_wrapper_needed": (
                "FloodCastBenchFNOPlusOfficialDataset enforces sample_length=20 and returns only h2:h20."
            ),
            "supporting_h50_or_h100_should_recompute_or_audit_normalization_stats": True,
            "normalization_note": (
                "Existing target_depth stats were fit on train h2:h20 targets. Higher-horizon targets may have a "
                "different distribution, so a higher-horizon target normalization audit/recompute is needed for a "
                "strict comparison."
            ),
            "affects_existing_h2_training_behavior": False,
            "existing_h2_baseline_can_remain_unchanged_if_new wrapper/config is separate": True,
        },
        "scientific_status": "horizon_availability_audit_only",
        "does_not_claim": [
            "model evaluation",
            "official FloodCastBench benchmark performance",
            "physical-unit forecast skill",
            "sparse-sensor robustness",
            "long-horizon model validation",
        ],
        "output_dir": str(output_dir),
        "command_reconstruction": command_reconstruction(),
        "git_status_short": git_status_short(),
    }
    save_json(summary, output_dir / "horizon_availability_summary.json")
    print("=== FLOODCASTBENCH HORIZON AVAILABILITY AUDIT ===")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
