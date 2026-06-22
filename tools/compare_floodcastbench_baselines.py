from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = PROJECT_DIR / "outputs" / "floodcastbench_baselines"

COLUMNS = [
    ("method", 14),
    ("split", 6),
    ("horizon", 7),
    ("mae", 11),
    ("rmse", 11),
    ("mse", 11),
    ("nse", 11),
    ("csi_0.001", 11),
    ("csi_0.01", 11),
    ("path_iou_0.001", 15),
    ("path_iou_0.01", 14),
    ("rmse_new_0.001", 14),
    ("rmse_new_0.01", 13),
]


def format_value(value) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.6g}"
    return str(value)


def load_result(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def row_from_baseline_result(result: dict) -> dict:
    metadata = result["metadata"]
    return {
        "method": metadata.get("baseline", "persistence"),
        "split": metadata["split"],
        "horizon": metadata["horizon"],
        "mae": result["water_depth"].get("mae"),
        "rmse": result["water_depth"].get("rmse"),
        "mse": result["water_depth"].get("mse"),
        "nse": result["water_depth"].get("nse"),
        "csi_0.001": result["mask"].get("0.001", {}).get("csi"),
        "csi_0.01": result["mask"].get("0.01", {}).get("csi"),
        "path_iou_0.001": result["propagation_path"].get("0.001", {}).get("iou"),
        "path_iou_0.01": result["propagation_path"].get("0.01", {}).get("iou"),
        "rmse_new_0.001": None,
        "rmse_new_0.01": None,
    }


def row_from_cnn_summary(summary: dict, split: str) -> dict:
    if split == "val":
        metrics = summary.get("best_val_metrics") or {}
    elif split == "test":
        metrics = summary.get("test_metrics") or {}
    else:
        raise ValueError("CNN summary split must be val or test")
    return {
        "method": summary.get("model", "cnn_baseline"),
        "split": split,
        "horizon": summary.get("horizon"),
        "mae": metrics.get("mae"),
        "rmse": metrics.get("rmse"),
        "mse": metrics.get("mse"),
        "nse": metrics.get("nse"),
        "csi_0.001": metrics.get("csi_gamma_0_001"),
        "csi_0.01": metrics.get("csi_gamma_0_01"),
        "path_iou_0.001": metrics.get("path_iou_gamma_0_001"),
        "path_iou_0.01": metrics.get("path_iou_gamma_0_01"),
        "rmse_new_0.001": metrics.get("rmse_newly_flooded_gamma_0_001"),
        "rmse_new_0.01": metrics.get("rmse_newly_flooded_gamma_0_01"),
    }


def sort_key(row: dict):
    method_order = {"persistence": 0, "linear_delta": 1, "flood_cnn_baseline": 2, "cnn_baseline": 2}
    split_order = {"val": 0, "test": 1, "train": 2}
    return (
        int(row["horizon"] or 0),
        split_order.get(row["split"], 99),
        method_order.get(row["method"], 99),
        row["method"],
    )


def print_table(rows: list[dict]) -> None:
    header = " | ".join(name.ljust(width) for name, width in COLUMNS)
    separator = "-+-".join("-" * width for _, width in COLUMNS)
    print(header)
    print(separator)
    for row in sorted(rows, key=sort_key):
        print(" | ".join(format_value(row.get(name)).ljust(width) for name, width in COLUMNS))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare saved FloodCastBench baseline JSON results.")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--split", choices=("val", "test"), default=None)
    parser.add_argument("--cnn-summary", type=Path, default=None)
    args = parser.parse_args()

    results_dir = args.results_dir if args.results_dir.is_absolute() else PROJECT_DIR / args.results_dir
    paths = sorted(results_dir.glob("*.json"))
    if not paths:
        raise SystemExit(f"No saved baseline JSON files found in: {results_dir}")

    rows = []
    for path in paths:
        result = load_result(path)
        row = row_from_baseline_result(result)
        if args.horizon is not None and int(row["horizon"]) != int(args.horizon):
            continue
        if args.split is not None and row["split"] != args.split:
            continue
        rows.append(row)

    if args.cnn_summary is not None:
        summary_path = args.cnn_summary if args.cnn_summary.is_absolute() else PROJECT_DIR / args.cnn_summary
        summary = load_result(summary_path)
        cnn_split = args.split or "val"
        rows.append(row_from_cnn_summary(summary, cnn_split))

    if not rows:
        raise SystemExit("No rows matched the requested filters.")
    print_table(rows)


if __name__ == "__main__":
    main()
