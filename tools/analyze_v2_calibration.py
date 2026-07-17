from __future__ import annotations

"""WP3 calibration figures (paper F5) from eval_calibration.json files.

Reads one or more evaluation output directories (each containing the
eval_calibration.json written by tools/evaluate_floodcastbench_diff_sparse_v2)
and renders, per directory: the reliability diagram (both flood thresholds),
the rank histogram (all vs active pixels), and the spread-skill curve.
Multiple directories (e.g. the 3 sparsity levels) are drawn side by side.

Usage:
  python tools/analyze_v2_calibration.py \
      --eval-dirs DIR [DIR...] --labels "m0.0" "m0.5" "m0.95" \
      --output experiments/FloodCastBench/v2_calibration_figure.png
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


def load_calibration(eval_dir: Path) -> dict:
    path = eval_dir / "eval_calibration.json"
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", help="One label per eval dir (default: dir names)")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    labels = args.labels or [d.name for d in args.eval_dirs]
    if len(labels) != len(args.eval_dirs):
        raise SystemExit("--labels must match --eval-dirs")
    columns = len(args.eval_dirs)

    plt.rcParams.update({
        "font.family": "serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    fig, axes = plt.subplots(3, columns, figsize=(3.4 * columns, 8.2), dpi=200, squeeze=False)

    for column, (eval_dir, label) in enumerate(zip(args.eval_dirs, labels)):
        calibration = load_calibration(eval_dir)
        members = calibration["num_members"]

        ax = axes[0][column]
        for gamma_key, marker in zip(sorted(calibration["reliability"]), ("o", "s")):
            entry = calibration["reliability"][gamma_key]
            probs = entry["forecast_probability"]
            observed = entry["pooled_observed_frequency"]
            counts = entry["pooled_count"]
            kept = [i for i in range(len(probs)) if counts[i] > 0]
            ax.plot([probs[i] for i in kept], [observed[i] for i in kept],
                    marker=marker, label=gamma_key.replace("gamma_", "$\\gamma$="))
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="perfect")
        ax.set_xlabel("forecast probability (k/M)")
        ax.set_ylabel("observed frequency")
        ax.set_title(label, fontsize=9)
        ax.legend(fontsize=8)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)

        ax = axes[1][column]
        hist = calibration["rank_histogram"]
        ranks = list(range(members + 1))
        width = 0.4
        ax.bar([r - width / 2 for r in ranks], hist["frequency"], width=width, label="all pixels")
        ax.bar([r + width / 2 for r in ranks], hist["active_frequency"], width=width, label="active pixels")
        ax.axhline(hist["uniform_reference"], color="k", linestyle="--", linewidth=0.8, label="uniform")
        ax.set_xlabel("rank of truth in ensemble")
        ax.set_ylabel("frequency")
        pass  # column title on top row only
        ax.legend(fontsize=8)

        ax = axes[2][column]
        bins = [b for b in calibration["spread_skill"] if b["count"] > 0]
        ax.loglog([b["mean_spread_m"] for b in bins], [b["rmse_m"] for b in bins], "o-", label="RMSE (ensemble mean)")
        spreads = [b["mean_spread_m"] for b in bins if b["mean_spread_m"] > 0]
        if spreads:
            ax.loglog(spreads, spreads, "k--", linewidth=0.8, label="RMSE = spread")
        ax.set_xlabel("ensemble spread (m)")
        ax.set_ylabel("error (m)")
        pass  # column title on top row only
        ax.legend(fontsize=8)

        coverage = calibration["coverage"]
        text = "  ".join(
            f"CI{name}: {entry['pooled']:.2f} (nominal {entry['nominal_finite_ensemble']:.2f})"
            for name, entry in sorted(coverage.items())
        )
        axes[0][column].text(0.02, 0.97, text, transform=axes[0][column].transAxes,
                             fontsize=7.5, va="top",
                             bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"))

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    plt.close(fig)
    print(f"figure written: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
