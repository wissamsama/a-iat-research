from __future__ import annotations

"""Build the FNO+ scientific metric dashboard from run artifacts.

Reads the dense per-step long-horizon rollout CSVs, the sparse path-metric
CSVs, and the official-protocol pooled test metrics of the official-v1
normalized FNO+ run, then emits a standalone interactive HTML dashboard.

Fixes applied versus the hand-written 2026-06-29 dashboard:
- The official Table 4 FNO+ "RMSE" (0.003941) is placed on the relative-RMSE
  metric, not on classical physical-unit RMSE. The Table 4 definition is
  provably pooled sqrt(SSE / sum(y^2)): the dataset ratio
  TotVar/sum(y^2) = 0.731329 reproduces the published NSE of U-Net, FNO, and
  FNO+ from their published RMSE to ~2e-7 (see
  reports/fno_plus_dashboard_audit.md).
- Official Table 4 values are pooled t=2..20 aggregates, so they are drawn as
  reference lines spanning steps 1..19 instead of being pinned to a single
  per-step point, and the repo's own same-protocol pooled values are shown
  next to them.
- The x axis is numeric in rollout steps with correct wall-clock time labels
  (1 step = 300 s), fixing the previous "T+216 = 216 h" mislabel (real: 18 h).
- Step 19 is annotated as the paper's t=20 (the 19th predicted frame).
"""

import argparse
import csv
import html
import json
import math
from pathlib import Path
from typing import Any


DEFAULT_RUN_DIR = Path(
    "/home/wissam/utem-workspace/experiments/FloodCastBench/"
    "28-06-2026_15-59-18_fcb_fno_plus_official_v1_normalized_100epoch_highfid_60m"
)
DEFAULT_OUTPUT = Path(
    "/home/wissam/utem-workspace/experiments/FloodCastBench/fno_plus_metric_dashboard_scientific.html"
)
SECONDS_PER_STEP = 300
CANONICAL_STEPS = [1, 5, 10, 19, 37, 54, 72, 96, 120, 144, 168, 192, 216]

OFFICIAL_TABLE4 = {
    "FNO+ (official Table 4)": {
        "current_relative_rmse": 0.003941,
        "nse": 0.999979,
        "pearson_r": 0.999990,
        "csi_gamma_0_001": 0.939638,
        "csi_gamma_0_01": 0.984588,
    },
    "FNO (official Table 4)": {
        "current_relative_rmse": 0.004258,
        "nse": 0.999975,
        "pearson_r": 0.999987,
        "csi_gamma_0_001": 0.895553,
        "csi_gamma_0_01": 0.980748,
    },
}

def frac(num: str, den: str) -> str:
    """Real stacked fraction (numerator over denominator), not an inline a/b."""
    return f'<span class="frac"><span class="num">{num}</span><span class="den">{den}</span></span>'


def sqrt(content: str) -> str:
    """Square root with an overline spanning its full argument."""
    return f'&radic;<span class="overline">{content}</span>'


METRIC_GROUPS = [
    {
        "name": "Continuous depth metrics",
        "metrics": ["current_relative_rmse", "classical_rmse", "mae", "bias", "nse", "pearson_r"],
    },
    {
        "name": "Flood-mask metrics",
        "metrics": [
            "csi_gamma_0_001",
            "csi_gamma_0_01",
            "precision_gamma_0_001",
            "recall_gamma_0_001",
            "f1_gamma_0_001",
            "precision_gamma_0_01",
            "recall_gamma_0_01",
            "f1_gamma_0_01",
            "negative_prediction_ratio",
        ],
    },
    {
        "name": "Propagation metrics (12 audit horizons only)",
        "metrics": ["path_iou_0_001", "path_iou_0_01", "propagation_path_iou_0_001", "propagation_path_iou_0_01"],
    },
]

METRIC_INFO = {
    "current_relative_rmse": {
        "label": "Relative RMSE (Table 4 definition)",
        "better": "lower",
        "desc": (
            "Pooled relative L2 error: sqrt(sum((pred-target)^2) / sum(target^2)). This is provably the "
            "official Table 4 'RMSE' definition: the dataset ratio TotVar/sum(y^2)=0.731329 reproduces the "
            "published NSE of U-Net, FNO and FNO+ from their published RMSE to ~2e-7."
        ),
        "formula": "relRMSE = " + sqrt(frac(
            "&Sigma;<sub>i</sub> (&ycirc;<sub>i</sub> - y<sub>i</sub>)<sup>2</sup>",
            "&Sigma;<sub>i</sub> y<sub>i</sub><sup>2</sup>",
        )),
    },
    "classical_rmse": {
        "label": "Classical RMSE (m)",
        "better": "lower",
        "desc": (
            "Root mean squared water-depth error in physical meters. The official Table 4 value is NOT "
            "comparable here (it is a relative metric); no official reference line is drawn on purpose."
        ),
        "formula": "RMSE = " + sqrt(frac("&Sigma;<sub>i</sub> (&ycirc;<sub>i</sub> - y<sub>i</sub>)<sup>2</sup>", "N")),
    },
    "mae": {
        "label": "MAE (m)",
        "better": "lower",
        "desc": "Average absolute water-depth error per pixel, in meters.",
        "formula": "MAE = " + frac("&Sigma;<sub>i</sub> |&ycirc;<sub>i</sub> - y<sub>i</sub>|", "N"),
    },
    "bias": {
        "label": "Bias (m)",
        "better": "closer to zero",
        "desc": "Mean signed error: positive = systematic overprediction of water depth.",
        "formula": "Bias = " + frac("&Sigma;<sub>i</sub> (&ycirc;<sub>i</sub> - y<sub>i</sub>)", "N"),
    },
    "nse": {
        "label": "NSE",
        "better": "higher",
        "desc": "Nash-Sutcliffe efficiency over pooled pixels of the step (paper eq. 3).",
        "formula": "NSE = 1 - " + frac(
            "&Sigma;<sub>i</sub> (&ycirc;<sub>i</sub> - y<sub>i</sub>)<sup>2</sup>",
            "&Sigma;<sub>i</sub> (y<sub>i</sub> - <span class=\"bar\">y</span>)<sup>2</sup>",
        ),
    },
    "pearson_r": {
        "label": "Pearson r",
        "better": "higher",
        "desc": "Linear correlation between predicted and true water depth (paper eq. 4).",
        "formula": "r = " + frac("Cov(&ycirc;, y)", "&sigma;<sub>&ycirc;</sub> &sigma;<sub>y</sub>"),
    },
    "negative_prediction_ratio": {
        "label": "Negative prediction ratio",
        "better": "lower",
        "desc": "Fraction of predicted pixels with physically impossible negative depth (plausibility diagnostic).",
        "formula": "NegRatio = " + frac("|{i : &ycirc;<sub>i</sub> &lt; 0}|", "N"),
    },
}
for gamma_key, gamma_txt in (("0_001", "0.001"), ("0_01", "0.01")):
    METRIC_INFO[f"csi_gamma_{gamma_key}"] = {
        "label": f"CSI @ {gamma_txt} m",
        "better": "higher",
        "desc": f"Critical success index of the flooded mask at threshold {gamma_txt} m (paper eq. 5).",
        "formula": "CSI = " + frac("TP", "TP + FP + FN") + f'<span class="gamma-line">&gamma; = {gamma_txt} m</span>',
    }
    METRIC_INFO[f"precision_gamma_{gamma_key}"] = {
        "label": f"Precision @ {gamma_txt} m",
        "better": "higher",
        "desc": f"Fraction of predicted flooded pixels (depth &gt; {gamma_txt} m) that are truly flooded.",
        "formula": "Precision = " + frac("TP", "TP + FP") + f'<span class="gamma-line">&gamma; = {gamma_txt} m</span>',
    }
    METRIC_INFO[f"recall_gamma_{gamma_key}"] = {
        "label": f"Recall @ {gamma_txt} m",
        "better": "higher",
        "desc": f"Fraction of truly flooded pixels (depth &gt; {gamma_txt} m) recovered by the prediction.",
        "formula": "Recall = " + frac("TP", "TP + FN") + f'<span class="gamma-line">&gamma; = {gamma_txt} m</span>',
    }
    METRIC_INFO[f"f1_gamma_{gamma_key}"] = {
        "label": f"F1 @ {gamma_txt} m",
        "better": "higher",
        "desc": f"Harmonic mean of flood-mask precision and recall at {gamma_txt} m.",
        "formula": "F1 = " + frac("2 &middot; Precision &middot; Recall", "Precision + Recall") + f'<span class="gamma-line">&gamma; = {gamma_txt} m</span>',
    }
    METRIC_INFO[f"path_iou_{gamma_key}"] = {
        "label": f"PathIoU @ {gamma_txt} m",
        "better": "higher",
        "desc": f"IoU of newly flooded area (vs the initial frame) at the horizon, threshold {gamma_txt} m.",
        "formula": "PathIoU = " + frac("|P&#770; &cap; P|", "|P&#770; &cup; P|") + f'<span class="gamma-line">&gamma; = {gamma_txt} m</span>',
    }
    METRIC_INFO[f"propagation_path_iou_{gamma_key}"] = {
        "label": f"Propagation PathIoU @ {gamma_txt} m",
        "better": "higher",
        "desc": f"Stepwise IoU of newly flooded pixels during rollout, threshold {gamma_txt} m.",
        "formula": "PropPathIoU = " + frac("|&Delta;P&#770; &cap; &Delta;P|", "|&Delta;P&#770; &cup; &Delta;P|") + f'<span class="gamma-line">&gamma; = {gamma_txt} m</span>',
    }


def read_per_step(path: Path) -> dict[int, dict[str, float]]:
    rows: dict[int, dict[str, float]] = {}
    with path.open("r", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            step = int(row["step"])
            rows[step] = {
                key: float(value)
                for key, value in row.items()
                if key not in ("checkpoint_name", "samples_or_maps") and value not in ("", None)
            }
    return rows


def read_path_metrics(path: Path) -> dict[int, dict[str, float]]:
    result: dict[int, dict[str, float]] = {}
    with path.open("r", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            step = int(row["horizon_steps"])
            gamma_key = str(float(row["gamma"])).replace(".", "_")
            entry = result.setdefault(step, {})
            entry[f"path_iou_{gamma_key}"] = float(row["path_iou"])
            entry[f"propagation_path_iou_{gamma_key}"] = float(row["propagation_path_iou"])
    return result


def read_diff_sparse_official_per_step(path: Path) -> dict[int, dict[str, float]]:
    rows: dict[int, dict[str, float]] = {}
    with path.open("r", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            step = int(row["step"])
            values: dict[str, float] = {}
            for key, value in row.items():
                if key in ("checkpoint_name", "samples_or_maps", "horizon_label") or value in ("", None):
                    continue
                try:
                    values[key] = float(value)
                except ValueError:
                    continue
            rows[step] = values
    return rows


def read_diff_sparse_path_metrics(summary_path: Path) -> dict[int, dict[str, float]]:
    with summary_path.open("r", encoding="utf-8") as file:
        summary = json.load(file)
    official = summary.get("official_metrics_physical", {})
    result: dict[int, dict[str, float]] = {}
    for key, metrics in official.items():
        if not key.startswith("path_h") or not isinstance(metrics, dict):
            continue
        horizon_label = str(metrics.get("horizon_label", ""))
        if not horizon_label.startswith("h"):
            continue
        step = int(horizon_label[1:])
        entry: dict[str, float] = {}
        for gamma_key_text in ("0_001", "0_01"):
            path_value = metrics.get(f"path_iou_gamma_{gamma_key_text}")
            prop_value = metrics.get(f"propagation_path_iou_gamma_{gamma_key_text}")
            if path_value is not None:
                entry[f"path_iou_{gamma_key_text}"] = float(path_value)
            if prop_value is not None:
                entry[f"propagation_path_iou_{gamma_key_text}"] = float(prop_value)
        result[step] = entry
    return result


def curve_series(
    name: str,
    status: str,
    color: str,
    width: float,
    per_step: dict[int, dict[str, float]],
    path_metrics: dict[int, dict[str, float]],
    steps: list[int],
) -> dict[str, Any]:
    metric_names = [m for group in METRIC_GROUPS for m in group["metrics"]]
    values: dict[str, dict[str, float | None]] = {m: {} for m in metric_names}
    samples: dict[str, float | None] = {}
    for step in steps:
        row = per_step.get(step, {})
        path_row = path_metrics.get(step, {})
        samples[str(step)] = row.get("rollout_samples")
        for metric in metric_names:
            value = row.get(metric)
            if value is None:
                value = path_row.get(metric)
            values[metric][str(step)] = value
    return {
        "name": name,
        "status": status,
        "kind": "curve",
        "style": {"color": color, "width": width},
        "values": values,
        "samples": samples,
    }


def reference_series(name: str, status: str, color: str, dash: str, refs: dict[str, float]) -> dict[str, Any]:
    return {
        "name": name,
        "status": status,
        "kind": "reference",
        "style": {"color": color, "dash": dash, "width": 2.0},
        "span": [1, 19],
        "refs": refs,
    }


# Categorical slots from the validated dataviz palette (references/palette.md),
# non-adjacent so pairwise CVD separation stays well above the safety floor.
# Each slot carries its light/dark step so the chart follows prefers-color-scheme.
COLOR_CURVE = {"light": "#2a78d6", "dark": "#3987e5"}  # slot 1 blue — the actual result
COLOR_TARGET = {"light": "#e34948", "dark": "#e66767"}  # slot 6 red — the benchmark to beat
COLOR_PROTOCOL = {"light": "#eb6834", "dark": "#d95926"}  # slot 8 orange — same-protocol fairness check
COLOR_DIFFSPARSE = {"light": "#4a3aa7", "dark": "#9085e9"}  # slot 5 violet — DIFF-SPARSE comparison


def build_data(run_dir: Path, diff_sparse_eval_dir: Path | None = None) -> dict[str, Any]:
    rollout_dir = run_dir / "long_horizon_rollout_eval_dense_v2"
    best_steps = read_per_step(rollout_dir / "checkpoint_best" / "long_horizon_metrics_per_step.csv")
    best_path = read_path_metrics(rollout_dir / "checkpoint_best" / "long_horizon_path_metrics.csv")
    steps = sorted(best_steps)

    with (run_dir / "test_metrics_checkpoint_best_official_v1_normalized.json").open() as file:
        pooled_best = json.load(file)

    series: list[dict[str, Any]] = [
        curve_series(
            "FNO+ (this repo) — rollout, step by step",
            "validation-selected checkpoint, autoregressive rollout, physical units, no clipping",
            COLOR_CURVE,
            2.5,
            best_steps,
            best_path,
            steps,
        ),
        reference_series(
            "FNO+ — published Table 4 result",
            "published reference; pooled aggregate over the 19 output steps, drawn as a line over steps 1..19",
            COLOR_TARGET,
            "7 5",
            OFFICIAL_TABLE4["FNO+ (official Table 4)"],
        ),
        reference_series(
            "FNO+ (this repo) — same protocol as Table 4",
            "this run evaluated with the official pooled t2..t20 protocol — the fair apples-to-apples comparison point",
            COLOR_PROTOCOL,
            "2 3",
            {
                "current_relative_rmse": pooled_best["metrics"]["current_relative_rmse"],
                "classical_rmse": pooled_best["metrics"]["classical_rmse"],
                "nse": pooled_best["metrics"]["nse"],
                "pearson_r": pooled_best["metrics"]["pearson_r"],
                "csi_gamma_0_001": pooled_best["metrics"]["csi_gamma_0_001"],
                "csi_gamma_0_01": pooled_best["metrics"]["csi_gamma_0_01"],
            },
        ),
    ]
    if diff_sparse_eval_dir is not None:
        diff_sparse_per_step = read_diff_sparse_official_per_step(
            diff_sparse_eval_dir / "eval_metrics_official_per_step.csv"
        )
        diff_sparse_path = read_diff_sparse_path_metrics(diff_sparse_eval_dir / "eval_summary.json")
        series.append(
            curve_series(
                "DIFF-SPARSE v1 — rollout, step by step (dense, no missing sensors)",
                "dense DIFF-SPARSE v1 rollout, shared physical metrics, native h13..h20 horizon only",
                COLOR_DIFFSPARSE,
                2.4,
                diff_sparse_per_step,
                diff_sparse_path,
                sorted(diff_sparse_per_step),
            )
        )

    source_files = {
        "run_dir": str(run_dir),
        "per_step_best": str(rollout_dir / "checkpoint_best" / "long_horizon_metrics_per_step.csv"),
        "per_step_last": str(rollout_dir / "checkpoint_last" / "long_horizon_metrics_per_step.csv"),
        "path_best": str(rollout_dir / "checkpoint_best" / "long_horizon_path_metrics.csv"),
        "path_last": str(rollout_dir / "checkpoint_last" / "long_horizon_path_metrics.csv"),
        "pooled_best": str(run_dir / "test_metrics_checkpoint_best_official_v1_normalized.json"),
        "generator": str(Path(__file__).resolve()),
    }
    if diff_sparse_eval_dir is not None:
        source_files["diff_sparse_eval_dir"] = str(diff_sparse_eval_dir)
        source_files["diff_sparse_per_step"] = str(diff_sparse_eval_dir / "eval_metrics_official_per_step.csv")
        source_files["diff_sparse_summary"] = str(diff_sparse_eval_dir / "eval_summary.json")

    return {
        "title": "FloodCastBench FNO+ baseline metrics",
        "seconds_per_step": SECONDS_PER_STEP,
        "steps": steps,
        "canonical_steps": CANONICAL_STEPS,
        "paper_t20_step": 19,
        "metric_groups": METRIC_GROUPS,
        "metric_info": METRIC_INFO,
        "series": series,
        "notes": {
            "protocol": (
                "Per-step curve is a project-side autoregressive rollout diagnostic on the Australia 60m test "
                "region (physical units, no clipping). The official Table 4 value is a pooled t=2..20 aggregate, "
                "drawn as a reference line over steps 1..19 next to this run's own pooled same-protocol result, "
                "so like is compared with like."
            ),
            "metric_proof": (
                "Table 4 “RMSE” is pooled sqrt(SSE/sum(y^2)): the dataset ratio TotVar/sum(y^2)=0.731329 "
                "reproduces the published NSE of U-Net, FNO and FNO+ from their published RMSE to ~2e-7."
            ),
            "periodicity": (
                "The sawtooth with period 19 steps is architectural, not noise: FNO+ predicts a fixed 19-step "
                "chunk from one initial frame (output_steps=19); every 19 steps the rollout restarts from its "
                "own most-drifted prediction, so error is lowest right after a restart and grows toward each "
                "chunk's far edge before resetting."
            ),
            "samples": (
                "Later rollout steps keep fewer valid test starts (14 at step 1 down to 3 at step 216); the "
                "per-step sample count is shown in tooltips and in the table."
            ),
            "time": "1 step = 300 s; step 19 is the paper's t=20 target; step 216 = 18 h.",
            "diff_sparse": (
                "When provided, DIFF-SPARSE v1 is shown only on its native dense rollout window h13..h20; "
                "no values are fabricated outside that horizon range."
            ),
        },
        "source_files": source_files,
    }


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>__TITLE__</title>
<style>
:root {
  --surface-1: #fcfcfb;
  --page: #f9f9f7;
  --text-primary: #0b0b0b;
  --text-secondary: #52514e;
  --text-muted: #898781;
  --gridline: #e1e0d9;
  --baseline: #c3c2b7;
  --border: rgba(11,11,11,0.10);
  --note-bg: #fff7df;
  --note-line: #e6ca72;
  --hover-wash: rgba(11,11,11,0.045);
}
@media (prefers-color-scheme: dark) {
  :root {
    --surface-1: #1a1a19;
    --page: #0d0d0d;
    --text-primary: #ffffff;
    --text-secondary: #c3c2b7;
    --text-muted: #898781;
    --gridline: #2c2c2a;
    --baseline: #383835;
    --border: rgba(255,255,255,0.10);
    --note-bg: #2a2410;
    --note-line: #a3801f;
    --hover-wash: rgba(255,255,255,0.06);
  }
}
* { box-sizing:border-box; }
body { margin:0; background:var(--page); color:var(--text-primary); font-family:system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width:1240px; margin:0 auto; padding:28px 24px 48px; }
h1 { margin:0 0 8px; font-size:clamp(26px,3.4vw,38px); line-height:1.15; letter-spacing:-0.01em; }
.subtitle { color:var(--text-secondary); line-height:1.55; max-width:980px; font-size:14.5px; margin-bottom:12px; }
.scinote { background:var(--note-bg); border:1px solid var(--note-line); border-radius:10px; padding:12px 14px; font-size:13px; line-height:1.6; max-width:1080px; margin-bottom:16px; }
.scinote b { color:var(--text-primary); }
.panel { background:var(--surface-1); border:1px solid var(--border); border-radius:12px; margin-bottom:16px; }
.controls { padding:16px; display:grid; gap:14px; }
.group-title { font-size:11.5px; font-weight:700; color:var(--text-muted); text-transform:uppercase; letter-spacing:.05em; margin-bottom:7px; }
.buttons { display:flex; flex-wrap:wrap; gap:7px; }
button { border:1px solid var(--border); background:transparent; color:var(--text-secondary); border-radius:8px; padding:7px 11px; font:inherit; font-size:12.5px; cursor:pointer; transition:background .12s ease, color .12s ease, border-color .12s ease; }
button:hover { background:var(--hover-wash); color:var(--text-primary); }
button.active { background:var(--text-primary); color:var(--surface-1); border-color:var(--text-primary); }
.explain { display:grid; grid-template-columns:1.3fr 1fr .32fr; gap:22px; padding:14px 16px; border-top:1px solid var(--gridline); font-size:13px; line-height:1.55; }
.explain b { display:block; margin-bottom:5px; font-size:11px; text-transform:uppercase; letter-spacing:.05em; color:var(--text-muted); font-weight:700; }
.explain > div:nth-child(1), .explain > div:nth-child(2), .explain > div:nth-child(3) { color:var(--text-secondary); }
#metricFormula { font-size:15px; color:var(--text-primary); line-height:1.5; }
.bar { text-decoration:overline; }
#metricFormula .overline { border-top:1.5px solid currentColor; padding-top:2px; display:inline-block; }
#metricFormula .frac { display:inline-grid; grid-template-rows:auto auto; align-items:center; justify-items:center; vertical-align:middle; margin:0 .2em; line-height:1.15; font-size:.92em; }
#metricFormula .frac .num { display:block; border-bottom:1.6px solid currentColor; padding:0 .3em .14em; }
#metricFormula .frac .den { display:block; padding:.14em .3em 0; }
#metricFormula .gamma-line { display:block; margin-top:8px; font-size:.82em; color:var(--text-muted); }
.chart-panel { padding:16px 18px 12px; }
.legend { display:flex; flex-wrap:wrap; gap:8px 6px; font-size:13px; font-weight:600; margin-bottom:10px; }
.legend-item { display:flex; align-items:center; gap:8px; border-radius:8px; padding:5px 9px; cursor:pointer; user-select:none; transition:background .12s ease, opacity .12s ease; color:var(--text-primary); }
.legend-item:hover { background:var(--hover-wash); }
.legend-item.disabled { opacity:.38; }
.legend-key { width:22px; height:2px; border-radius:2px; display:inline-block; flex:none; }
.legend-key.dashed { background:none !important; border-top:2px dashed currentColor; height:0; }
.chart-metric-title { margin:2px 0 8px; text-align:center; font-size:14px; font-weight:700; color:var(--text-primary); }
#chart { width:100%; min-height:520px; display:block; cursor:crosshair; touch-action:none; }
.selection-rect { fill:rgba(42,120,214,.12); stroke:#2a78d6; stroke-width:1.5; stroke-dasharray:5 4; pointer-events:none; }
.caption { color:var(--text-muted); font-size:12px; margin:8px 4px 4px; line-height:1.55; }
.table-panel { padding:16px 18px; overflow-x:auto; }
table { width:100%; border-collapse:collapse; font-size:12.5px; }
th,td { border-bottom:1px solid var(--gridline); padding:8px 8px; text-align:right; white-space:nowrap; font-variant-numeric:tabular-nums; }
th:first-child,td:first-child { text-align:left; white-space:normal; font-variant-numeric:normal; }
th { color:var(--text-muted); font-size:10.5px; text-transform:uppercase; letter-spacing:.04em; font-weight:700; }
td b { color:var(--text-primary); font-weight:600; }
.status { color:var(--text-muted); font-size:11.5px; display:block; font-weight:400; }
.tooltip { position:fixed; pointer-events:none; background:var(--text-primary); color:var(--surface-1); border-radius:8px; padding:9px 11px; font-size:12.5px; line-height:1.45; opacity:0; transform:translate(14px,14px); z-index:20; box-shadow:0 8px 22px rgba(0,0,0,.22); max-width:260px; }
.tooltip .t-value { font-weight:700; font-size:14px; }
.tooltip .t-series { color:var(--text-muted); display:flex; align-items:center; gap:6px; margin-top:2px; }
.tooltip .t-key { width:14px; height:2px; display:inline-block; flex:none; }
.zoom-label { cursor:pointer; user-select:none; }
</style>
</head>
<body>
<main>
<h1>__TITLE__</h1>
<div class="subtitle">Dense per-step autoregressive rollout diagnostics for the official-v1 normalized FNO+ run, with the official Table 4 reference drawn on the metric it actually corresponds to. Drag to zoom, double-click anywhere in the chart to reset, click legend entries to toggle series.</div>
<div class="scinote" id="notes"></div>
<div class="panel controls" id="controls"></div>
<div class="panel explain">
  <div><b>Explanation</b><span id="metricDesc"></span></div>
  <div><b>Formula</b><span id="metricFormula"></span></div>
  <div><b>Direction</b><span id="metricBetter"></span></div>
</div>
<div class="panel chart-panel">
  <div class="legend" id="legend"></div>
  <div class="chart-metric-title" id="chartMetricTitle"></div>
  <svg id="chart" viewBox="0 0 1180 520" preserveAspectRatio="xMidYMid meet"></svg>
  <div class="caption" id="caption"></div>
</div>
<div class="panel table-panel">
  <div class="group-title" id="tableTitle"></div>
  <div id="valueTable"></div>
</div>
</main>
<div class="tooltip" id="tooltip"></div>
<script>
const DATA = __DATA__;
let currentMetric = DATA.metric_groups[0].metrics[0];
const visibleSeries = Object.fromEntries(DATA.series.map(s => [s.name, true]));
let zoomX = null, zoomY = null, dragStart = null, dragCurrent = null, lastScale = null;
const darkQuery = window.matchMedia ? window.matchMedia('(prefers-color-scheme: dark)') : null;
const isDark = () => !!(darkQuery && darkQuery.matches);
const seriesColor = s => isDark() ? s.style.color.dark : s.style.color.light;
const fmt = v => v == null || Number.isNaN(v) ? '—' : Number(v).toPrecision(6);
const stepTime = s => { const min = s * DATA.seconds_per_step / 60; return min < 60 ? `${min} min` : `${(min/60).toFixed(min % 60 ? 1 : 0)} h`; };
const el = (n,a={},t='') => { const e=document.createElementNS('http://www.w3.org/2000/svg',n); for(const[k,v]of Object.entries(a))e.setAttribute(k,v); if(t!=='')e.textContent=t; return e; };
const htmlEl = (n,c,t) => { const e=document.createElement(n); if(c)e.className=c; if(t!=null)e.appendChild(document.createTextNode(t)); return e; };
function cssVar(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }

function xRange() { return zoomX || [DATA.steps[0], DATA.steps[DATA.steps.length-1]]; }
function activeSeries() { return DATA.series.filter(s => visibleSeries[s.name]); }

function renderNotes() {
  const n = document.getElementById('notes');
  n.innerHTML = '';
  Object.values(DATA.notes).forEach((t, i) => {
    if (i > 0) n.appendChild(document.createElement('br'));
    n.appendChild(document.createTextNode('• ' + t));
  });
  document.getElementById('caption').textContent = DATA.notes.samples + ' ' + DATA.notes.time;
}

function renderControls() {
  const c = document.getElementById('controls');
  c.innerHTML = '';
  DATA.metric_groups.forEach(group => {
    const g = htmlEl('div');
    g.appendChild(htmlEl('div','group-title',group.name));
    const bs = htmlEl('div','buttons');
    group.metrics.forEach(metric => {
      const info = DATA.metric_info[metric] || {label: metric};
      const b = document.createElement('button');
      b.textContent = info.label;
      b.className = metric === currentMetric ? 'active' : '';
      b.onclick = () => { currentMetric = metric; zoomY = null; render(); };
      bs.appendChild(b);
    });
    g.appendChild(bs);
    c.appendChild(g);
  });
}

function renderInfo() {
  const info = DATA.metric_info[currentMetric] || {};
  document.getElementById('metricDesc').textContent = info.desc || '';
  document.getElementById('metricFormula').innerHTML = info.formula || '';
  document.getElementById('metricBetter').textContent = info.better || '';
}

function renderLegend() {
  const l = document.getElementById('legend');
  l.innerHTML = '';
  DATA.series.forEach(s => {
    const item = htmlEl('div','legend-item' + (visibleSeries[s.name] ? '' : ' disabled'));
    const key = htmlEl('span','legend-key' + (s.kind === 'reference' ? ' dashed' : ''));
    key.style.background = s.kind === 'reference' ? 'none' : seriesColor(s);
    key.style.color = seriesColor(s);
    item.appendChild(key);
    item.appendChild(document.createTextNode(s.name));
    item.title = s.status || '';
    item.onclick = () => { visibleSeries[s.name] = !visibleSeries[s.name]; zoomY = null; render(); };
    l.appendChild(item);
  });
}

function seriesPoints(s) {
  if (s.kind === 'reference') return [];
  const [x0,x1] = xRange();
  return DATA.steps.filter(st => st >= x0 && st <= x1).map(st => {
    const v = s.values[currentMetric] ? s.values[currentMetric][String(st)] : null;
    return v == null ? null : {step: st, v: Number(v), n: s.samples[String(st)]};
  }).filter(Boolean);
}

function getYDomain(series) {
  let vals = [];
  series.forEach(s => {
    if (s.kind === 'reference') { const r = s.refs[currentMetric]; if (r != null) vals.push(Number(r)); }
    else seriesPoints(s).forEach(p => vals.push(p.v));
  });
  if (!vals.length) vals = [0,1];
  let min = Math.min(...vals), max = Math.max(...vals);
  if (zoomY) { min = zoomY[0]; max = zoomY[1]; }
  if (min === max) { min -= 1; max += 1; }
  const pad = zoomY ? 0 : (max-min)*0.08;
  return [min-pad, max+pad];
}

function showTooltip(ev, valueText, seriesLabel, color, extraLine) {
  const tooltip = document.getElementById('tooltip');
  tooltip.style.opacity = 1;
  tooltip.style.left = ev.clientX + 'px';
  tooltip.style.top = ev.clientY + 'px';
  tooltip.innerHTML = '';
  const value = htmlEl('div','t-value', valueText);
  tooltip.appendChild(value);
  const row = htmlEl('div','t-series');
  const key = htmlEl('span','t-key');
  key.style.background = color;
  row.appendChild(key);
  row.appendChild(document.createTextNode(seriesLabel));
  tooltip.appendChild(row);
  if (extraLine) {
    const extra = htmlEl('div','t-series', extraLine);
    extra.style.marginTop = '2px';
    tooltip.appendChild(extra);
  }
}
function hideTooltip() { document.getElementById('tooltip').style.opacity = 0; }

function renderChart() {
  const svg = document.getElementById('chart');
  svg.innerHTML = '';
  const surface = cssVar('--surface-1') || '#fcfcfb';
  const gridline = cssVar('--gridline') || '#e1e0d9';
  const baseline = cssVar('--baseline') || '#c3c2b7';
  const textMuted = cssVar('--text-muted') || '#898781';
  const textSecondary = cssVar('--text-secondary') || '#52514e';
  const textPrimary = cssVar('--text-primary') || '#0b0b0b';
  const W=1180, H=520, m={left:92,right:30,top:20,bottom:70};
  const plotW=W-m.left-m.right, plotH=H-m.top-m.bottom;
  const series = activeSeries();
  const [x0,x1] = xRange();
  const [yMin,yMax] = getYDomain(series);
  const xFor = st => m.left + (st-x0)/Math.max(x1-x0,1e-9)*plotW;
  const yFor = v => m.top + (1-(v-yMin)/(yMax-yMin))*plotH;
  const stepForX = x => x0 + Math.max(0,Math.min(1,(x-m.left)/plotW))*(x1-x0);
  const valueForY = y => yMax - (Math.max(m.top,Math.min(m.top+plotH,y))-m.top)/plotH*(yMax-yMin);
  lastScale = {m,plotW,plotH,xFor,yFor,stepForX,valueForY};

  svg.appendChild(el('rect',{x:0,y:0,width:W,height:H,fill:surface}));
  svg.appendChild(el('rect',{x:m.left,y:m.top,width:plotW,height:plotH,fill:'none',stroke:baseline,'stroke-width':1}));
  const defs = el('defs'); const clip = el('clipPath',{id:'pc'});
  clip.appendChild(el('rect',{x:m.left,y:m.top,width:plotW,height:plotH}));
  defs.appendChild(clip); svg.appendChild(defs);
  const layer = el('g',{'clip-path':'url(#pc)'});

  for (let i=0;i<=5;i++) {
    const y=m.top+i/5*plotH, val=yMax-i/5*(yMax-yMin);
    svg.appendChild(el('line',{x1:m.left,x2:m.left+plotW,y1:y,y2:y,stroke:gridline,'stroke-width':1}));
    svg.appendChild(el('text',{x:m.left-10,y:y+4,'text-anchor':'end',fill:textMuted,'font-size':'11.5'},fmt(val)));
  }
  // Canonical steps cluster tightly at the low end (1,5,10,19) and spread out
  // at the high end (37..216) on a linear axis. Draw a gridline for every
  // canonical step, but only draw its text label when there is enough pixel
  // space since the last labeled tick, so labels never overlap. The first
  // tick always gets a label for orientation.
  const ticks = DATA.canonical_steps.filter(st => st>=x0 && st<=x1);
  const tickList = ticks.length ? ticks : [Math.round(x0), Math.round(x1)];
  const minLabelGap = 46;
  let lastLabelX = -Infinity;
  tickList.forEach((st, idx) => {
    const x = xFor(st);
    svg.appendChild(el('line',{x1:x,x2:x,y1:m.top,y2:m.top+plotH,stroke:gridline,'stroke-width':1}));
    if (idx === 0 || (x - lastLabelX) >= minLabelGap) {
      lastLabelX = x;
      const lbl = st===DATA.paper_t20_step ? `${st}*` : String(st);
      svg.appendChild(el('text',{x,y:H-32,'text-anchor':'middle',fill:textSecondary,'font-size':'12','font-weight':'600'},lbl));
      svg.appendChild(el('text',{x,y:H-17,'text-anchor':'middle',fill:textMuted,'font-size':'10'},stepTime(st)));
    }
  });

  series.forEach(s => {
    const color = seriesColor(s);
    if (s.kind === 'reference') {
      const r = s.refs[currentMetric];
      if (r == null) return;
      const [sx0,sx1] = s.span;
      const xa = xFor(Math.max(sx0,x0)), xb = xFor(Math.min(sx1,x1));
      if (xb <= xa) return;
      const y = yFor(Number(r));
      const hit = el('line',{x1:xa,x2:xb,y1:y,y2:y,stroke:'transparent','stroke-width':16});
      const line = el('line',{x1:xa,x2:xb,y1:y,y2:y,stroke:color,'stroke-width':s.style.width,'stroke-dasharray':s.style.dash||'','stroke-linecap':'round'});
      hit.addEventListener('mousemove',ev=>showTooltip(ev, fmt(r), s.name, color, 'pooled t=2..20'));
      hit.addEventListener('mouseleave',hideTooltip);
      layer.appendChild(line);
      layer.appendChild(hit);
      return;
    }
    const pts = seriesPoints(s);
    if (!pts.length) return;
    if (pts.length>1) layer.appendChild(el('path',{d:pts.map((p,i)=>(i?'L':'M')+xFor(p.step)+','+yFor(p.v)).join(' '),fill:'none',stroke:color,'stroke-width':s.style.width,'stroke-linecap':'round','stroke-linejoin':'round'}));
    const sparse = pts.length <= 20;
    pts.forEach(p => {
      const emphasized = sparse || DATA.canonical_steps.includes(p.step);
      const r = sparse ? 4.5 : (emphasized ? 4 : 1.8);
      const c = el('circle',{cx:xFor(p.step),cy:yFor(p.v),r,fill:color,stroke:emphasized?surface:'none','stroke-width':emphasized?2:0});
      const hitR = el('circle',{cx:xFor(p.step),cy:yFor(p.v),r:Math.max(r,12),fill:'transparent'});
      const label = (DATA.metric_info[currentMetric]||{label:currentMetric}).label;
      const extra = `step ${p.step} · ${stepTime(p.step)}${p.step===DATA.paper_t20_step?' · paper t=20':''} · ${p.n??'?'} test starts`;
      hitR.addEventListener('mousemove',ev=>showTooltip(ev, `${label}: ${fmt(p.v)}`, s.name, color, extra));
      hitR.addEventListener('mouseleave',hideTooltip);
      layer.appendChild(c);
      layer.appendChild(hitR);
    });
  });
  svg.appendChild(layer);

  document.getElementById('chartMetricTitle').textContent = (DATA.metric_info[currentMetric]||{label:currentMetric}).label;
  svg.appendChild(el('text',{x:m.left+plotW/2,y:H-3,'text-anchor':'middle',fill:textSecondary,'font-size':'12','font-weight':'600'},'Rollout step (1 step = 5 min · * = paper t=20)'));
  if (zoomX||zoomY) {
    const z = el('text',{class:'zoom-label',x:W-36,y:20,'text-anchor':'end',fill:seriesColor(DATA.series[0]),'font-size':'11.5','font-weight':'700'},'ZOOM — double-click to reset');
    z.addEventListener('click',resetZoom);
    svg.appendChild(z);
  }
  if (dragStart&&dragCurrent) {
    svg.appendChild(el('rect',{class:'selection-rect',x:Math.min(dragStart.x,dragCurrent.x),y:Math.min(dragStart.y,dragCurrent.y),width:Math.abs(dragCurrent.x-dragStart.x),height:Math.abs(dragCurrent.y-dragStart.y)}));
  }
}

function renderTable() {
  const info = DATA.metric_info[currentMetric]||{label:currentMetric};
  const [x0,x1] = xRange();
  const cols = DATA.canonical_steps.filter(st=>st>=x0&&st<=x1);
  document.getElementById('tableTitle').textContent = `${info.label} at audit steps${zoomX?' (zoomed)':''}`;
  let h = '<table><thead><tr><th>Series</th>'+cols.map(st=>`<th>step ${st}${st===DATA.paper_t20_step?'*':''}<br><span style="font-weight:400">${stepTime(st)}</span></th>`).join('')+'</tr></thead><tbody>';
  activeSeries().forEach(s => {
    h += `<tr><td><b>${s.name}</b><span class="status">${s.status||''}</span></td>`;
    cols.forEach(st => {
      let v = null;
      if (s.kind==='reference') { v = (st>=s.span[0]&&st<=s.span[1]) ? s.refs[currentMetric] : null; }
      else { v = s.values[currentMetric] ? s.values[currentMetric][String(st)] : null; }
      h += `<td>${fmt(v)}</td>`;
    });
    h += '</tr>';
  });
  const curve = DATA.series.find(s=>s.kind==='curve');
  h += `<tr><td><b>valid test starts</b><span class="status">rollout samples entering each step</span></td>`+cols.map(st=>`<td>${curve.samples[String(st)]??'/'}</td>`).join('')+'</tr>';
  h += '</tbody></table>';
  document.getElementById('valueTable').innerHTML = h;
}

function resetZoom(){ zoomX=null; zoomY=null; dragStart=null; dragCurrent=null; render(); }
function svgPoint(evt){ const svg=document.getElementById('chart'); const b=svg.getBoundingClientRect(); const v=svg.viewBox.baseVal; return {x:v.x+(evt.clientX-b.left)/b.width*v.width, y:v.y+(evt.clientY-b.top)/b.height*v.height}; }
function setupInteraction(){
  const svg=document.getElementById('chart');
  // Double-click ANYWHERE in the chart (axes, margins, plot area) resets to the
  // standard scale — not just the zoom-active label. Bound on the whole <svg>,
  // and it also clears any half-finished drag selection.
  svg.addEventListener('dblclick', e => { e.preventDefault(); resetZoom(); });
  svg.addEventListener('mousedown',e=>{ if(!lastScale)return; const p=svgPoint(e); const m=lastScale.m; if(p.x<m.left||p.x>m.left+lastScale.plotW||p.y<m.top||p.y>m.top+lastScale.plotH)return; dragStart={x:p.x,y:p.y}; dragCurrent={x:p.x,y:p.y}; renderChart(); });
  window.addEventListener('mousemove',e=>{ if(!dragStart||!lastScale)return; const p=svgPoint(e); const m=lastScale.m; dragCurrent={x:Math.max(m.left,Math.min(m.left+lastScale.plotW,p.x)),y:Math.max(m.top,Math.min(m.top+lastScale.plotH,p.y))}; renderChart(); });
  window.addEventListener('keydown',e=>{ if(e.key==='Escape')resetZoom(); });
  window.addEventListener('mouseup',()=>{
    if(!dragStart||!dragCurrent||!lastScale)return;
    const dx=Math.abs(dragCurrent.x-dragStart.x), dy=Math.abs(dragCurrent.y-dragStart.y);
    if(dx>20&&dy>20){
      const s0=lastScale.stepForX(Math.min(dragStart.x,dragCurrent.x));
      const s1=lastScale.stepForX(Math.max(dragStart.x,dragCurrent.x));
      zoomX=[Math.max(DATA.steps[0],Math.floor(s0)),Math.min(DATA.steps[DATA.steps.length-1],Math.ceil(s1))];
      const y0=lastScale.valueForY(Math.max(dragStart.y,dragCurrent.y));
      const y1=lastScale.valueForY(Math.min(dragStart.y,dragCurrent.y));
      if(Number.isFinite(y0)&&Number.isFinite(y1)&&y0!==y1)zoomY=[Math.min(y0,y1),Math.max(y0,y1)];
    }
    dragStart=null; dragCurrent=null; render();
  });
  if (darkQuery) darkQuery.addEventListener('change', render);
}
function render(){ renderNotes(); renderControls(); renderInfo(); renderLegend(); renderChart(); renderTable(); }
setupInteraction();
render();
</script>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the FNO+ scientific metric dashboard from run artifacts.")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--diff-sparse-eval-dir", type=Path)
    args = parser.parse_args()

    data = build_data(args.run_dir, args.diff_sparse_eval_dir)
    page = HTML_TEMPLATE.replace("__TITLE__", html.escape(data["title"])).replace(
        "__DATA__", json.dumps(data, allow_nan=False)
    )
    args.output.write_text(page, encoding="utf-8")
    print(f"dashboard written: {args.output} ({len(page)} bytes)")
    print(f"steps: {len(data['steps'])} series: {[s['name'] for s in data['series']]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
