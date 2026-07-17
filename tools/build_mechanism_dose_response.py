from __future__ import annotations

"""WP12 Phase 1 (free, CPU): the dose-response ratio curve.

For each candidate time step dt (multiples of the native 300 s cadence),
recompute sigma_delta(dt) = std of per-step changes between frames i and
i+stride, by subsampling the frame sequence already on disk. sigma_field
(wet-pixel std) does not depend on dt. The deliverable is the curve
ratio(dt) = sigma_field / sigma_delta(dt) for each event, which (i) shows
how much of the ratio axis the benchmark can span by subsampling alone and
(ii) picks the 3-4 training dt values for WP12 Phase 2 (the crossed
{dt x target-representation} experiment).

No model is involved -- pure data measurement. Writes a JSON with all
numbers and a small publication-style figure.

Usage:
  python tools/build_mechanism_dose_response.py \
      --configs configs/floodcastbench_diff_sparse_v2_highfid_60m.yaml \
                configs/floodcastbench_diff_sparse_v2_uk_highfid_60m.yaml \
      --strides 1 2 3 6 12 24 \
      --output-json experiments/FloodCastBench/wp12_dose_response/ratio_curve.json \
      --output-figure paper/figures/f2b_dose_response_ratio.pdf
"""

import argparse
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from datasets.floodcastbench_fno_dataset import EVENTS, _load_frames, _read_raster  # noqa: E402
from datasets.floodcastbench_diff_sparse_v1_dataset import split_frame_ranges  # noqa: E402
from tools.train_floodcastbench_diff_sparse_v1 import load_config, path_from_config  # noqa: E402

NATIVE_DT_SECONDS = 300
PAIR_SUBSAMPLE = 20  # sample a (i, i+stride) pair every this many frames
WET_THRESHOLD_M = 0.001


def collect_event(config_path: Path, strides: list[int]) -> dict:
    config = load_config(config_path)
    dataset_config = config.get("dataset", {})
    root = path_from_config(config, "dataset_root")
    event = EVENTS[str(dataset_config.get("event", "australia")).lower()]
    family = (
        "High-fidelity flood forecasting"
        if str(dataset_config.get("fidelity", "high")).lower() == "high"
        else "Low-fidelity flood forecasting"
    )
    frames = _load_frames(root / family / str(dataset_config.get("resolution", "60m")) / event)
    train_start, train_end = split_frame_ranges(len(frames), dataset_config.get("split_counts"))["train"]

    # sigma_field from wet pixels of the sampled anchor frames (dt-independent)
    wet_values: list[np.ndarray] = []
    per_stride: dict[int, dict] = {}
    cache: dict[int, np.ndarray] = {}

    def frame(i: int) -> np.ndarray:
        if i not in cache:
            cache[i] = _read_raster(frames[i].path)
            # bound memory: keep the cache small, reads are cheap
            if len(cache) > 4 * max(strides) // min(strides) + 64:
                cache.pop(next(iter(cache)))
        return cache[i]

    for stride in strides:
        sq_sum = 0.0
        mean_sum = 0.0
        count = 0
        pairs = 0
        for i in range(train_start, train_end - stride, PAIR_SUBSAMPLE):
            depth = frame(i)
            depth_next = _read_raster(frames[i + stride].path)
            delta = depth_next - depth
            sq_sum += float(np.square(delta, dtype=np.float64).sum())
            mean_sum += float(delta.sum(dtype=np.float64))
            count += delta.size
            pairs += 1
            if stride == strides[0]:
                wet = depth[depth >= WET_THRESHOLD_M]
                if wet.size:
                    wet_values.append(wet.astype(np.float64))
        mean = mean_sum / max(count, 1)
        variance = sq_sum / max(count, 1) - mean * mean
        per_stride[stride] = {
            "dt_seconds": stride * NATIVE_DT_SECONDS,
            "pairs_sampled": pairs,
            "delta_std_m": float(np.sqrt(max(variance, 0.0))),
            "delta_rms_m": float(np.sqrt(sq_sum / max(count, 1))),
        }

    all_wet = np.concatenate(wet_values) if wet_values else np.zeros(1)
    field_std = float(np.std(all_wet))
    for stride in strides:
        entry = per_stride[stride]
        entry["ratio_field_over_delta"] = field_std / max(entry["delta_std_m"], 1e-12)

    return {
        "event": event,
        "config": str(config_path),
        "train_frame_range": [int(train_start), int(train_end)],
        "field_std_wet_m": field_std,
        "per_stride": per_stride,
    }


def main() -> int:
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", type=Path, nargs="+", required=True)
    parser.add_argument("--strides", type=int, nargs="+", default=[1, 2, 3, 6, 12, 24])
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-figure", type=Path, required=True)
    args = parser.parse_args()

    strides = sorted(set(args.strides))
    events = [collect_event(path, strides) for path in args.configs]

    payload = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "native_dt_seconds": NATIVE_DT_SECONDS,
        "pair_subsample": PAIR_SUBSAMPLE,
        "wet_threshold_m": WET_THRESHOLD_M,
        "strides": strides,
        "events": events,
        "purpose": (
            "WP12 phase 1: dose-response ratio curve sigma_field/sigma_delta(dt). "
            "Training dt values for phase 2 should span >=1 order of magnitude of ratio."
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    fig, ax = plt.subplots(figsize=(4.2, 2.9), dpi=200)
    for entry, color, marker in zip(events, ("#0072B2", "#D55E00"), ("o", "s")):
        dts = [entry["per_stride"][s]["dt_seconds"] / 60 for s in strides]
        ratios = [entry["per_stride"][s]["ratio_field_over_delta"] for s in strides]
        ax.loglog(dts, ratios, marker=marker, ms=4, lw=1.2, color=color, label=entry["event"])
        for x, y in zip(dts, ratios):
            ax.annotate(f"{y:.0f}", (x, y), textcoords="offset points", xytext=(0, 5),
                        ha="center", fontsize=6.5, color=color)
    ax.set_xlabel("time step $\\Delta t$ (minutes, log)")
    ax.set_ylabel("$\\sigma_{\\mathrm{field}}/\\sigma_\\Delta$ (log)")
    ax.legend(frameon=False)
    ax.grid(lw=0.4, alpha=0.35, which="both")
    ax.set_axisbelow(True)
    fig.tight_layout()
    args.output_figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_figure, bbox_inches="tight")
    plt.close(fig)

    for entry in events:
        print(f"== {entry['event']} (field_std={entry['field_std_wet_m']:.4f} m) ==")
        for s in strides:
            e = entry["per_stride"][s]
            print(f"  dt={e['dt_seconds']:>5d}s  sigma_delta={e['delta_std_m']:.6f} m  "
                  f"ratio={e['ratio_field_over_delta']:.0f}")
    print(f"json:   {args.output_json}")
    print(f"figure: {args.output_figure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
