from __future__ import annotations

"""Paper figure F2 (master plan §6): the mechanism behind the absolute-field
diffusion failure. Two panels from REAL train data only (no model runs):

  1. Distribution of per-step physical water-depth changes |x_{t+1}-x_t| vs
     the distribution of the wet-pixel field values themselves -- the
     signal-vs-field scale gap (~400x on Australia 60m).
  2. The scale bar: delta std vs field std per event, with the persistence
     floor annotation -- any generative model whose sampling noise exceeds
     the delta scale must lose to persistence at this timestep.

Usage:
  python tools/build_mechanism_figure.py \
      --configs configs/floodcastbench_diff_sparse_v2_highfid_60m.yaml \
                configs/floodcastbench_diff_sparse_v2_uk_highfid_60m.yaml \
      --output experiments/FloodCastBench/paper_figures/f2_mechanism.png
"""

import argparse
import sys
import warnings
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

PAIR_STRIDE = 20  # subsample consecutive train pairs for the histograms
WET_THRESHOLD_M = 0.001


def collect_event(config_path: Path) -> dict:
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

    deltas: list[np.ndarray] = []
    wet_values: list[np.ndarray] = []
    # Subsample PAIRS of ADJACENT frames (i, i+1): the mechanism claim is
    # about the 300 s per-step delta -- striding the sequence and diffing
    # neighbors of the strided list would silently measure 20-step deltas.
    for i in range(train_start, train_end - 1, PAIR_STRIDE):
        depth = _read_raster(frames[i].path)
        depth_next = _read_raster(frames[i + 1].path)
        deltas.append(np.abs(depth_next - depth).flatten())
        wet_values.append(depth[depth >= WET_THRESHOLD_M].flatten())
    all_deltas = np.concatenate(deltas)
    all_wet = np.concatenate(wet_values)
    return {
        "event": event,
        "deltas": all_deltas,
        "wet": all_wet,
        "delta_std": float(np.std(np.concatenate([d for d in deltas]))),
        "field_std": float(np.std(all_wet)),
    }


def main() -> int:
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    events = [collect_event(path) for path in args.configs]

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.titlesize": 9.5,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), dpi=200,
                             gridspec_kw={"width_ratios": [1.5, 1.0]})

    ax = axes[0]
    bins = np.logspace(-6, 1.2, 80)
    for entry, (c_delta, c_wet) in zip(events, (("#0072B2", "#56B4E9"), ("#D55E00", "#E69F00"))):
        ax.hist(np.clip(entry["deltas"], 1e-6, None), bins=bins, histtype="step",
                density=True, color=c_delta, lw=1.1,
                label=f"{entry['event']}: per-step change $|\\Delta|$")
        ax.hist(entry["wet"], bins=bins, histtype="step", density=True,
                color=c_wet, linestyle="--", lw=1.1,
                label=f"{entry['event']}: depth (wet pixels)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("meters (log scale)")
    ax.set_ylabel("density (log)")
    ax.set_title("(a) Per-step change vs. field amplitude", loc="left")
    ax.legend(fontsize=6.8, frameon=False)

    ax = axes[1]
    labels = [e["event"] for e in events]
    x = np.arange(len(events))
    ax.bar(x - 0.18, [e["delta_std"] for e in events], width=0.36, color="#0072B2",
           label="std of per-step change")
    ax.bar(x + 0.18, [e["field_std"] for e in events], width=0.36, color="#E69F00",
           label="std of field (wet pixels)")
    ax.set_yscale("log")
    ax.set_xticks(x, labels)
    ax.set_ylabel("meters (log)")
    top = max(e["field_std"] for e in events)
    ax.set_ylim(top=top * 40)
    for i, e in enumerate(events):
        ratio = e["field_std"] / max(e["delta_std"], 1e-12)
        ax.text(i + 0.18, e["field_std"] * 1.6, f"$\\times${ratio:.0f}", ha="center",
                fontsize=9, fontweight="700")
    ax.set_title("(b) Scale gap per event", loc="left")
    ax.legend(fontsize=6.4, frameon=False, loc="upper left")

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    plt.close(fig)
    for e in events:
        print(f"{e['event']}: delta_std={e['delta_std']:.6f} m, field_std(wet)={e['field_std']:.4f} m, "
              f"ratio x{e['field_std']/max(e['delta_std'],1e-12):.0f}")
    print(f"figure written: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
