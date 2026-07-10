from __future__ import annotations

"""Aggregate V2-family evaluation results into the paper's main tables
(master plan T2/T3): one row per eval dir in a long CSV, plus mean +/- std
grouped by (model label x missing_rate) rendered as a markdown table.

Works on any eval output directory produced by
tools/evaluate_floodcastbench_diff_sparse_v2.py (diffusion V2, deterministic
twin, ablations, V2.1 Manning, UK event) -- the model label is derived from
the run config name recorded in the summary, or overridden per glob via
LABEL=GLOB arguments.

Usage:
  python tools/aggregate_v2_family_results.py \
      --eval-globs \
        "V2=experiments/FloodCastBench/*fcb_diff_sparse_v2_highfid*/eval_rollout_test_*" \
        "twin=experiments/FloodCastBench/*fcb_det_twin*/eval_rollout_test_*" \
      --output-csv reports/v2_family_eval_aggregate.csv \
      --output-md reports/v2_family_eval_table.md
"""

import argparse
import csv
import glob
import json
import math
import statistics
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

ROW_FIELDS = [
    "label",
    "eval_dir",
    "checkpoint_epoch",
    "seed",
    "missing_rate",
    "split",
    "windows_evaluated",
    "num_scenarios",
    "context_length",
    "mask_structure",
    "relrmse_overall",
    "rmse_m_overall",
    "mae_m_overall",
    "nse_overall",
    "csi_0_001_overall",
    "csi_0_01_overall",
    "model_nrmse",
    "model_nacrps",
    "persistence_nrmse",
    "path_iou_0_001_pooled",
    "path_iou_0_001_pooled_median",
    "coverage50_pooled",
    "coverage90_pooled",
]

AGG_METRICS = [
    "relrmse_overall",
    "rmse_m_overall",
    "csi_0_001_overall",
    "model_nrmse",
    "model_nacrps",
    "path_iou_0_001_pooled_median",
    "coverage90_pooled",
]


def extract_seed(summary: dict) -> int | None:
    checkpoint_path = str(summary.get("checkpoint_path", ""))
    config_path = str(summary.get("config_path", ""))
    for text in (checkpoint_path, config_path):
        for token in ("seed7", "seed123", "seed42"):
            if token in text:
                return int(token.removeprefix("seed"))
    # base configs carry seed 42 without a name suffix
    return 42 if checkpoint_path else None


def read_row(label: str, eval_dir: Path) -> dict | None:
    summary_path = eval_dir / "eval_summary.json"
    if not summary_path.exists():
        return None
    with summary_path.open("r", encoding="utf-8") as file:
        summary = json.load(file)
    official = summary.get("official_metrics_physical", {})
    overall = official.get("overall", {})
    pooled_path = official.get("pooled_propagation", {})
    pooled_path_median = official.get("pooled_propagation_median", {})
    model_block = summary.get("model", {})
    persistence_block = summary.get("persistence", {})
    calibration = summary.get("calibration", {}) or {}
    coverage = calibration.get("coverage_pooled", {})

    def pooled(block: dict, key: str) -> float | None:
        overall = block.get("overall")
        return overall.get(key) if isinstance(overall, dict) else None

    return {
        "label": label,
        "eval_dir": str(eval_dir),
        "checkpoint_epoch": summary.get("checkpoint_epoch"),
        "seed": extract_seed(summary),
        "missing_rate": summary.get("missing_rate"),
        "split": summary.get("split"),
        "windows_evaluated": summary.get("windows_evaluated"),
        "num_scenarios": summary.get("num_scenarios"),
        "context_length": summary.get("context_length"),
        "mask_structure": summary.get("cli_args", {}).get("mask_structure") or "random",
        "relrmse_overall": overall.get("current_relative_rmse"),
        "rmse_m_overall": overall.get("classical_rmse"),
        "mae_m_overall": overall.get("mae"),
        "nse_overall": overall.get("nse"),
        "csi_0_001_overall": overall.get("csi_gamma_0_001"),
        "csi_0_01_overall": overall.get("csi_gamma_0_01"),
        "model_nrmse": pooled(model_block, "nrmse"),
        "model_nacrps": pooled(model_block, "nacrps"),
        "persistence_nrmse": pooled(persistence_block, "nrmse"),
        "path_iou_0_001_pooled": pooled_path.get("final_path_iou_gamma_0_001"),
        "path_iou_0_001_pooled_median": pooled_path_median.get("final_path_iou_gamma_0_001"),
        "coverage50_pooled": coverage.get("50"),
        "coverage90_pooled": coverage.get("90"),
    }


def format_cell(values: list[float]) -> str:
    clean = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not clean:
        return "--"
    mean = statistics.fmean(clean)
    if len(clean) == 1:
        return f"{mean:.4g} (N=1)"
    return f"{mean:.4g} ± {statistics.stdev(clean):.2g} (N={len(clean)})"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval-globs", nargs="+", required=True,
        help="LABEL=GLOB entries; each glob matches eval output directories",
    )
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path)
    args = parser.parse_args()

    rows: list[dict] = []
    for entry in args.eval_globs:
        if "=" not in entry:
            raise SystemExit(f"--eval-globs entries must be LABEL=GLOB, got {entry!r}")
        label, pattern = entry.split("=", 1)
        matches = sorted(glob.glob(pattern))
        if not matches:
            print(f"WARNING: no eval dirs matched {pattern!r}")
        for match in matches:
            row = read_row(label, Path(match))
            if row is not None:
                rows.append(row)
    if not rows:
        raise SystemExit("no eval summaries found")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=ROW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"long CSV written: {args.output_csv} ({len(rows)} rows)")

    if args.output_md:
        groups: dict[tuple, list[dict]] = {}
        for row in rows:
            key = (row["label"], row["missing_rate"], row["mask_structure"], row["split"])
            groups.setdefault(key, []).append(row)
        lines = [
            "<!-- generated by tools/aggregate_v2_family_results.py; do not edit by hand -->",
            "",
            "| model | missing_rate | masks | split | " + " | ".join(AGG_METRICS) + " |",
            "|" + "---|" * (4 + len(AGG_METRICS)),
        ]
        for key in sorted(groups, key=lambda k: (str(k[0]), float(k[1] or 0), str(k[2]))):
            group = groups[key]
            label, missing_rate, mask_structure, split = key
            cells = [format_cell([row[m] for row in group]) for m in AGG_METRICS]
            lines.append(
                f"| {label} | {missing_rate} | {mask_structure} | {split} | " + " | ".join(cells) + " |"
            )
        seeds_note = (
            "\nSeeds per group are inferred from run names; single-seed groups are "
            "flagged (N=1) and carry no claim (master plan rule R1).\n"
        )
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text("\n".join(lines) + seeds_note, encoding="utf-8")
        print(f"markdown table written: {args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
