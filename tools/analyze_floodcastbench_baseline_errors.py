from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
import warnings
from pathlib import Path

import torch
import yaml
from rasterio.errors import NotGeoreferencedWarning

PROJECT_DIR = Path(__file__).resolve().parents[1]

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from baselines.floodcastbench import available_baselines, predict_floodcastbench_baseline, prediction_rule_for_baseline
from datasets import FloodCastBenchWaterDepthDataset
from metrics import binary_mask_metrics, region_error_metrics, water_depth_metrics

DEFAULT_GAMMAS = (0.001, 0.01)
REGIONS = (
    "target_wet",
    "target_dry",
    "newly_flooded",
    "already_flooded",
    "stable_dry",
    "receded",
)


def gamma_key(gamma: float) -> str:
    return str(gamma).replace(".", "_")


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def split_ratios_from_config(config: dict) -> tuple[float, float, float]:
    ratios = config.get("split_ratios", {})
    return (
        float(ratios.get("train", 0.70)),
        float(ratios.get("val", 0.15)),
        float(ratios.get("test", 0.15)),
    )


def build_dataset(config: dict, split: str, horizon: int) -> FloodCastBenchWaterDepthDataset:
    dataset_config = config["dataset"]
    return FloodCastBenchWaterDepthDataset(
        root=PROJECT_DIR / dataset_config.get("root", "data/FloodCastBench"),
        event=dataset_config.get("event", "Australia flood"),
        fidelity=dataset_config.get("fidelity", "high"),
        resolution=dataset_config.get("resolution", "30m"),
        input_window=int(dataset_config.get("input_window", 5)),
        horizon=int(horizon),
        split=split,
        split_ratios=split_ratios_from_config(dataset_config),
        normalization=dataset_config.get("normalization", "none"),
    )


def finite_count(tensor: torch.Tensor) -> int:
    return int(torch.isfinite(tensor.detach().reshape(-1)).sum().item())


def ratio(mask: torch.Tensor, denominator: int) -> float:
    if denominator == 0:
        return math.nan
    return float(mask.detach().bool().sum().item()) / denominator


def sample_row(
    sample_index: int,
    baseline: str,
    dataset: FloodCastBenchWaterDepthDataset,
    x: torch.Tensor,
    y: torch.Tensor,
    meta: dict,
    gammas: tuple[float, ...],
) -> dict:
    current = x[-1]
    prediction = predict_floodcastbench_baseline(x, horizon=dataset.horizon, baseline=baseline)
    target = y
    global_metrics = water_depth_metrics(prediction, target)
    valid_pixels = finite_count(target)

    row = {
        "sample_index": sample_index,
        "baseline": baseline,
        "split": dataset.split,
        "horizon": dataset.horizon,
        "event": dataset.event,
        "resolution": dataset.resolution,
        "target_timestamp": meta["target_timestamp"],
        "last_input_timestamp": meta["input_timestamps"][-1],
        "input_timestamps": ";".join(str(value) for value in meta["input_timestamps"]),
        "target_path": meta["target_path"],
        "prediction_rule": prediction_rule_for_baseline(baseline),
        "mae_global": global_metrics["mae"],
        "mse_global": global_metrics["mse"],
        "rmse_global": global_metrics["rmse"],
    }

    for gamma in gammas:
        key = gamma_key(gamma)
        target_wet = target > gamma
        target_dry = target <= gamma
        current_wet = current > gamma
        current_dry = current <= gamma
        pred_wet = prediction > gamma
        newly_flooded = target_wet & current_dry
        already_flooded = target_wet & current_wet
        stable_dry = target_dry & current_dry
        receded = target_dry & current_wet
        pred_newly_flooded = pred_wet & current_dry

        mask_metrics = binary_mask_metrics(pred_wet, target_wet)
        path_metrics = binary_mask_metrics(pred_newly_flooded, newly_flooded)

        row[f"flooded_pixel_ratio_target_gamma_{key}"] = ratio(target_wet, valid_pixels)
        row[f"flooded_pixel_ratio_pred_gamma_{key}"] = ratio(pred_wet, valid_pixels)
        row[f"newly_flooded_pixel_ratio_target_gamma_{key}"] = ratio(newly_flooded, valid_pixels)
        row[f"newly_flooded_pixel_ratio_pred_gamma_{key}"] = ratio(pred_newly_flooded, valid_pixels)

        for metric_name, value in mask_metrics.items():
            row[f"{metric_name}_gamma_{key}"] = value
        row[f"path_iou_gamma_{key}"] = path_metrics["iou"]
        row[f"path_f1_gamma_{key}"] = path_metrics["f1"]
        row[f"path_precision_gamma_{key}"] = path_metrics["precision"]
        row[f"path_recall_gamma_{key}"] = path_metrics["recall"]

        region_masks = {
            "target_wet": target_wet,
            "target_dry": target_dry,
            "newly_flooded": newly_flooded,
            "already_flooded": already_flooded,
            "stable_dry": stable_dry,
            "receded": receded,
        }
        for region_name, region_mask in region_masks.items():
            metrics = region_error_metrics(prediction, target, region_mask)
            row[f"count_{region_name}_gamma_{key}"] = metrics["count"]
            row[f"mae_{region_name}_gamma_{key}"] = metrics["mae"]
            row[f"rmse_{region_name}_gamma_{key}"] = metrics["rmse"]
    return row


def nanmean(values: list[float]) -> tuple[float, int]:
    finite = [float(value) for value in values if value is not None and not math.isnan(float(value))]
    if not finite:
        return math.nan, 0
    return sum(finite) / len(finite), len(finite)


def nanmedian(values: list[float]) -> tuple[float, int]:
    finite = sorted(float(value) for value in values if value is not None and not math.isnan(float(value)))
    if not finite:
        return math.nan, 0
    middle = len(finite) // 2
    if len(finite) % 2:
        return finite[middle], len(finite)
    return (finite[middle - 1] + finite[middle]) / 2.0, len(finite)


def nanmax(values: list[float]) -> tuple[float, int]:
    finite = [float(value) for value in values if value is not None and not math.isnan(float(value))]
    if not finite:
        return math.nan, 0
    return max(finite), len(finite)


def add_aggregate(summary: dict, name: str, values: list[float], reducer) -> None:
    value, count = reducer(values)
    summary[name] = value
    summary[f"{name}_valid_samples"] = count


def build_summary(rows: list[dict], dataset: FloodCastBenchWaterDepthDataset, baseline: str, raw_unchanged: bool, elapsed: float, gammas: tuple[float, ...]) -> dict:
    summary = {
        "baseline": baseline,
        "split": dataset.split,
        "horizon": dataset.horizon,
        "event": dataset.event,
        "resolution": dataset.resolution,
        "num_samples": len(rows),
        "prediction_rule": prediction_rule_for_baseline(baseline),
        "elapsed_seconds": elapsed,
        "raw_root_mtime_unchanged": raw_unchanged,
    }
    add_aggregate(summary, "mean_rmse_global", [row["rmse_global"] for row in rows], nanmean)
    add_aggregate(summary, "median_rmse_global", [row["rmse_global"] for row in rows], nanmedian)
    add_aggregate(summary, "max_rmse_global", [row["rmse_global"] for row in rows], nanmax)
    for gamma in gammas:
        key = gamma_key(gamma)
        add_aggregate(summary, f"mean_path_iou_gamma_{key}", [row[f"path_iou_gamma_{key}"] for row in rows], nanmean)
        add_aggregate(summary, f"mean_rmse_newly_flooded_gamma_{key}", [row[f"rmse_newly_flooded_gamma_{key}"] for row in rows], nanmean)
        add_aggregate(summary, f"mean_newly_flooded_pixel_ratio_gamma_{key}", [row[f"newly_flooded_pixel_ratio_target_gamma_{key}"] for row in rows], nanmean)
    return summary


def sorted_valid(rows: list[dict], column: str, reverse: bool) -> list[dict]:
    valid = [row for row in rows if column in row and row[column] is not None and not math.isnan(float(row[column]))]
    return sorted(valid, key=lambda row: float(row[column]), reverse=reverse)


def ranking_specs_for_rows(rows: list[dict]) -> list[tuple[str, bool]]:
    preferred = [
        ("rmse_global", True),
        ("rmse_newly_flooded_gamma_0_001", True),
        ("rmse_newly_flooded_gamma_0_01", True),
        ("path_iou_gamma_0_001", False),
        ("newly_flooded_pixel_ratio_target_gamma_0_001", True),
    ]
    available = set(rows[0]) if rows else set()
    return [(column, reverse) for column, reverse in preferred if column in available]


def build_worst_rows(rows: list[dict], gammas: tuple[float, ...], top_k: int) -> list[dict]:
    output = []
    seen = set()
    for column, reverse in ranking_specs_for_rows(rows):
        for rank, row in enumerate(sorted_valid(rows, column, reverse=reverse)[:top_k], start=1):
            marker = (column, row["sample_index"])
            if marker in seen:
                continue
            seen.add(marker)
            output.append({
                "ranking_reason": column,
                "rank": rank,
                "sample_index": row["sample_index"],
                "target_timestamp": row["target_timestamp"],
                "metric_value": row[column],
                "rmse_global": row["rmse_global"],
                "rmse_newly_flooded_gamma_0_001": row.get("rmse_newly_flooded_gamma_0_001", math.nan),
                "rmse_newly_flooded_gamma_0_01": row.get("rmse_newly_flooded_gamma_0_01", math.nan),
                "path_iou_gamma_0_001": row.get("path_iou_gamma_0_001", math.nan),
                "path_iou_gamma_0_01": row.get("path_iou_gamma_0_01", math.nan),
                "newly_flooded_pixel_ratio_target_gamma_0_001": row.get("newly_flooded_pixel_ratio_target_gamma_0_001", math.nan),
                "target_path": row["target_path"],
                "figure_path": "",
            })
    return output


def event_slug(event: str) -> str:
    return event.lower().replace(" flood", "").replace(" ", "_")


def output_base_name(dataset: FloodCastBenchWaterDepthDataset, baseline: str) -> str:
    return f"{baseline}_{event_slug(dataset.event)}_{dataset.resolution}_h{dataset.horizon}_{dataset.split}"


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_plots(rows: list[dict], output_dir: Path, base_name: str) -> None:
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    x = [row["target_timestamp"] for row in rows]
    plots = [
        ("rmse_global", "RMSE global"),
        ("path_iou_gamma_0_001", "Path IoU gamma=0.001"),
        ("newly_flooded_pixel_ratio_target_gamma_0_001", "Newly flooded ratio gamma=0.001"),
        ("rmse_newly_flooded_gamma_0_001", "RMSE newly flooded gamma=0.001"),
    ]
    for column, title in plots:
        y = [row.get(column, math.nan) for row in rows]
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(x, y, linewidth=1.0)
        ax.set_title(title)
        ax.set_xlabel("target timestamp")
        ax.set_ylabel(column)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / f"{base_name}_{column}.png", dpi=140)
        plt.close(fig)


def to_numpy_2d(tensor: torch.Tensor):
    return tensor.detach().cpu().squeeze().numpy()


def finite_max(*tensors: torch.Tensor) -> float:
    values = []
    for tensor in tensors:
        flat = tensor.detach().float().reshape(-1)
        finite = flat[torch.isfinite(flat)]
        if finite.numel():
            values.append(float(finite.max().item()))
    return max(values) if values else 1.0


def safe_metric(row: dict, key: str) -> float:
    value = row.get(key, math.nan)
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def figure_filename(base_name: str, row: dict, ranking_reason: str) -> str:
    metric = ranking_reason.replace("rmse_newly_flooded", "worst_rmse_newly_flooded")
    return (
        f"{base_name}_sample_{int(row['sample_index']):04d}_"
        f"t_{int(row['target_timestamp'])}_{metric}.png"
    )


def selected_worst_samples(worst_rows: list[dict]) -> dict[int, list[str]]:
    selected = {}
    for row in worst_rows:
        selected.setdefault(int(row["sample_index"]), [])
        reason = str(row["ranking_reason"])
        if reason not in selected[int(row["sample_index"] )]:
            selected[int(row["sample_index"] )].append(reason)
    return selected


def save_worst_sample_figure(
    dataset: FloodCastBenchWaterDepthDataset,
    baseline: str,
    row: dict,
    ranking_reasons: list[str],
    output_dir: Path,
    base_name: str,
    gamma: float,
) -> Path:
    import matplotlib.pyplot as plt

    sample_index = int(row["sample_index"])
    x, y, meta = dataset[sample_index]
    current = x[-1]
    target = y
    prediction = predict_floodcastbench_baseline(x, horizon=dataset.horizon, baseline=baseline)
    error = torch.abs(prediction - target)

    target_mask = target > gamma
    pred_mask = prediction > gamma
    current_mask = current > gamma
    target_new = target_mask & (~current_mask)
    pred_new = pred_mask & (~current_mask)

    depth_vmax = finite_max(current, target, prediction)
    error_vmax = finite_max(error)
    panels = [
        (to_numpy_2d(current), "current D_t", "viridis", 0, depth_vmax, True),
        (to_numpy_2d(target), "target D_t+h", "viridis", 0, depth_vmax, True),
        (to_numpy_2d(prediction), "prediction", "viridis", 0, depth_vmax, True),
        (to_numpy_2d(error), "absolute error", "magma", 0, error_vmax, True),
        (to_numpy_2d(target_mask), f"target mask g={gamma}", "gray", 0, 1, False),
        (to_numpy_2d(pred_mask), f"pred mask g={gamma}", "gray", 0, 1, False),
        (to_numpy_2d(target_new), "target newly flooded", "gray", 0, 1, False),
        (to_numpy_2d(pred_new), "pred newly flooded", "gray", 0, 1, False),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for ax, (image, title, cmap, vmin, vmax, add_colorbar) in zip(axes.flat, panels):
        im = ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
        if add_colorbar:
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    key = gamma_key(gamma)
    title = (
        f"{baseline} | {dataset.split} | h={dataset.horizon} | sample={sample_index} | "
        f"last_t={meta['input_timestamps'][-1]} | target_t={meta['target_timestamp']}\n"
        f"RMSE={safe_metric(row, 'rmse_global'):.4g} | "
        f"RMSE_new={safe_metric(row, f'rmse_newly_flooded_gamma_{key}'):.4g} | "
        f"PathIoU={safe_metric(row, f'path_iou_gamma_{key}'):.4g} | "
        f"new_ratio={safe_metric(row, f'newly_flooded_pixel_ratio_target_gamma_{key}'):.4g} | "
        f"reasons={', '.join(ranking_reasons)}"
    )
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.91))

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / figure_filename(base_name, row, ranking_reasons[0])
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def save_worst_figures(
    dataset: FloodCastBenchWaterDepthDataset,
    baseline: str,
    rows: list[dict],
    worst_rows: list[dict],
    output_dir: Path,
    base_name: str,
    gamma: float,
) -> int:
    row_by_index = {int(row["sample_index"]): row for row in rows}
    reasons_by_sample = selected_worst_samples(worst_rows)
    figure_dir = output_dir / "figures" / base_name
    figure_paths = {}
    for sample_index, reasons in reasons_by_sample.items():
        row = row_by_index[sample_index]
        figure_paths[sample_index] = save_worst_sample_figure(
            dataset=dataset,
            baseline=baseline,
            row=row,
            ranking_reasons=reasons,
            output_dir=figure_dir,
            base_name=base_name,
            gamma=gamma,
        )
    for worst_row in worst_rows:
        path = figure_paths.get(int(worst_row["sample_index"]))
        if path is not None:
            worst_row["figure_path"] = str(path)
    return len(figure_paths)


def analyze(args) -> tuple[list[dict], dict, list[dict], Path, Path, Path, int]:
    config_path = args.config if args.config.is_absolute() else PROJECT_DIR / args.config
    config = load_config(config_path)
    dataset = build_dataset(config, split=args.split, horizon=args.horizon)
    baseline = args.baseline.lower()
    if baseline not in available_baselines():
        available = ", ".join(available_baselines())
        raise ValueError(f"Unsupported baseline '{baseline}'. Available baselines: {available}")

    sample_count = len(dataset) if args.max_samples is None else min(len(dataset), args.max_samples)
    rows = []
    root_mtime_before = dataset.root.stat().st_mtime
    start = time.perf_counter()
    with torch.no_grad():
        for index in range(sample_count):
            x, y, meta = dataset[index]
            rows.append(sample_row(index, baseline, dataset, x, y, meta, tuple(args.gamma)))
            if args.progress_every and (index + 1) % args.progress_every == 0:
                print(f"progress: {index + 1}/{sample_count}")
    elapsed = time.perf_counter() - start
    root_mtime_after = dataset.root.stat().st_mtime

    summary = build_summary(rows, dataset, baseline, root_mtime_before == root_mtime_after, elapsed, tuple(args.gamma))
    worst_rows = build_worst_rows(rows, tuple(args.gamma), args.top_k)

    output_dir = args.output_dir if args.output_dir.is_absolute() else PROJECT_DIR / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    base_name = output_base_name(dataset, baseline)
    generated_figures = 0
    if args.save_worst_figures:
        generated_figures = save_worst_figures(dataset, baseline, rows, worst_rows, output_dir, base_name, args.figure_gamma)
        summary["worst_figures_generated"] = generated_figures
        summary["worst_figures_gamma"] = args.figure_gamma
        summary["worst_figures_dir"] = str(output_dir / "figures" / base_name)

    per_sample_path = output_dir / f"{base_name}_per_sample.csv"
    summary_path = output_dir / f"{base_name}_summary.json"
    worst_path = output_dir / f"{base_name}_worst_samples.csv"

    if args.save:
        write_csv(per_sample_path, rows)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        write_csv(worst_path, worst_rows)
    if args.plot:
        save_plots(rows, output_dir / "figures", base_name)

    return rows, summary, worst_rows, per_sample_path, summary_path, worst_path, generated_figures


def print_summary(summary: dict, worst_rows: list[dict], top_k: int) -> None:
    print("FloodCastBench baseline error analysis")
    print(f"baseline: {summary['baseline']}")
    print(f"split: {summary['split']}")
    print(f"horizon: {summary['horizon']}")
    print(f"samples: {summary['num_samples']}")
    print(f"mean_rmse_global: {summary['mean_rmse_global']:.6g}")
    print(f"median_rmse_global: {summary['median_rmse_global']:.6g}")
    print(f"max_rmse_global: {summary['max_rmse_global']:.6g}")
    for suffix in ("0_001", "0_01"):
        if f"mean_path_iou_gamma_{suffix}" not in summary:
            continue
        print(f"mean_path_iou_gamma_{suffix}: {summary[f'mean_path_iou_gamma_{suffix}']:.6g}")
        print(f"mean_rmse_newly_flooded_gamma_{suffix}: {summary[f'mean_rmse_newly_flooded_gamma_{suffix}']:.6g}")
        print(f"mean_newly_flooded_pixel_ratio_gamma_{suffix}: {summary[f'mean_newly_flooded_pixel_ratio_gamma_{suffix}']:.6g}")
    print(f"raw_root_mtime_unchanged: {summary['raw_root_mtime_unchanged']}")
    if "worst_figures_generated" in summary:
        print(f"worst_figures_generated: {summary['worst_figures_generated']}")
        print(f"worst_figures_dir: {summary['worst_figures_dir']}")
    print(f"\nWorst-sample entries shown: {min(len(worst_rows), top_k)}")
    for row in worst_rows[:top_k]:
        print(f"{row['ranking_reason']} rank={row['rank']} sample={row['sample_index']} value={row['metric_value']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-sample FloodCastBench deterministic baseline error analysis.")
    parser.add_argument("--config", type=Path, default=PROJECT_DIR / "configs" / "floodcastbench_water_depth.yaml")
    parser.add_argument("--baseline", choices=available_baselines(), default="persistence")
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--gamma", type=float, nargs="+", default=list(DEFAULT_GAMMAS))
    parser.add_argument("--figure-gamma", type=float, default=0.001, help="Gamma threshold used in worst-sample diagnostic figures.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--save-worst-figures", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "outputs" / "floodcastbench_error_analysis")
    args = parser.parse_args()

    warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
    rows, summary, worst_rows, per_sample_path, summary_path, worst_path, generated_figures = analyze(args)
    print_summary(summary, worst_rows, args.top_k)
    if args.save:
        print(f"Saved per-sample CSV: {per_sample_path}")
        print(f"Saved summary JSON: {summary_path}")
        print(f"Saved worst-samples CSV: {worst_path}")
    if args.plot:
        print(f"Saved figures under: {args.output_dir / 'figures'}")
    if args.save_worst_figures:
        print(f"Saved worst-sample figures: {generated_figures}")


if __name__ == "__main__":
    main()
