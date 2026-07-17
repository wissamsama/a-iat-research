"""Publication figures for paper 1 (DIFF-SPARSE controlled comparison).

Generates vector PDFs into paper/figures/. Data provenance:
  - F3 (performance vs sparsity): WP1 table (3 seeds, master plan §4-WP1,
    2026-07-12) + WP2 ctx12 column (seed42 quick 4/13-window read,
    2026-07-15). The sparse WP1 points are PRELIMINARY pending the WP6
    extended-budget re-evaluation -- the caption must say so.
  - F6 (calibration comparison): read live from eval_calibration.json files
    (WP9 twin ensemble m50; WP7 V2 gauge/cluster m50/m95; V2 dense).

Style: Okabe-Ito colorblind-safe palette, serif fonts to match newtx body
text, no chartjunk, direct labels where possible.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
WORKSPACE = Path("/home/wissam/utem-workspace")

# Okabe-Ito
C_V1 = "#E69F00"      # orange
C_V2_12 = "#56B4E9"   # sky blue
C_V2 = "#0072B2"      # blue
C_TWIN = "#D55E00"    # vermillion
C_ENS = "#CC79A7"     # purple-pink
C_NOM = "#666666"

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
    "figure.dpi": 200,
    "savefig.bbox": "tight",
})


def fig_performance_vs_sparsity() -> None:
    """F3: normalized rollout RMSE vs sparsity for V1 / V2@12 / V2 / twin.

    V1+V2@12: seed42 quick read (WP2). V2/twin: mean over 3 seeds with
    min-max seed range as error bars (WP1 table). Log y -- the V1 gap is
    3 orders of magnitude in dense.
    """
    labels = ["dense", "50% missing", "95% missing"]
    x = np.arange(3)

    v1 = np.array([0.862, 0.898, 1.007])           # WP2, seed42
    v2_12 = np.array([0.000889, 0.230, 0.526])      # WP2, seed42
    # WP1 3-seed table (per-seed values -> mean + min/max range)
    v2_seeds = np.array([
        [0.001311, 0.000923, 0.002490],
        [0.395202, 0.388081, 0.251160],
        [0.567427, 0.560316, 0.453630],
    ])
    twin_seeds = np.array([
        [0.000420, 0.000499, 0.000410],
        [0.381563, 0.292901, 0.375410],
        [0.342173, 0.331412, 0.335170],
    ])

    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    w = 0.19

    def bars(offset, mean, color, label, seeds=None, hatch=None):
        err = None
        if seeds is not None:
            lo = mean - seeds.min(axis=1)
            hi = seeds.max(axis=1) - mean
            err = np.vstack([lo, hi])
        ax.bar(x + offset, mean, w, color=color, label=label,
               yerr=err, error_kw={"lw": 0.8, "capsize": 2}, hatch=hatch,
               edgecolor="white", linewidth=0.4)

    bars(-1.5 * w, v1, C_V1, "V1 (absolute-space diffusion)")
    bars(-0.5 * w, v2_12, C_V2_12, "V2, context 12 (=V1)")
    bars(+0.5 * w, v2_seeds.mean(axis=1), C_V2, "V2 (delta-space diffusion)", v2_seeds)
    bars(+1.5 * w, twin_seeds.mean(axis=1), C_TWIN, "Deterministic twin", twin_seeds)

    ax.set_yscale("log")
    ax.set_ylim(2e-4, 3)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Rollout RMSE (normalized, test)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.22), ncols=2,
              frameon=False, handlelength=1.4, columnspacing=1.2)
    ax.grid(axis="y", which="major", lw=0.4, alpha=0.35)
    ax.set_axisbelow(True)
    fig.savefig(HERE / "f3_performance_sparsity.pdf")
    plt.close(fig)
    print("wrote f3_performance_sparsity.pdf")


def load_calibration(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def fig_calibration_comparison() -> None:
    """F6: (a) coverage of central intervals vs the finite-ensemble nominal;
    (b) rank histograms (active pixels) -- V2 (M=8) vs twin ensemble (M=3)."""

    e = WORKSPACE / "experiments/FloodCastBench"
    sources = [
        ("Twin ens.\n(M=3) m50", C_ENS,
         e / "wp9_det_twin_ensemble/eval_test_m3_17-07-2026_00-11-29/eval_calibration.json"),
        ("V2 gauge\nm50", C_V2,
         e / "wp7_structured_masks_eval/seed42_m0.5_gauge/eval_calibration.json"),
        ("V2 clust.\nm50", C_V2,
         e / "wp7_structured_masks_eval/seed42_m0.5_cluster/eval_calibration.json"),
        ("V2 gauge\nm95", C_V2,
         e / "wp7_structured_masks_eval/seed42_m0.95_gauge/eval_calibration.json"),
        ("V2 clust.\nm95", C_V2,
         e / "wp7_structured_masks_eval/seed42_m0.95_cluster/eval_calibration.json"),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 2.7),
                                   gridspec_kw={"width_ratios": [1.35, 1.0]})

    # (a) ratio observed / finite-ensemble-nominal for the 90% interval
    names, ratios50, ratios90, colors = [], [], [], []
    for name, color, path in sources:
        c = load_calibration(path)
        cov = c["coverage"]
        names.append(name)
        colors.append(color)
        ratios50.append(cov["50"]["pooled"] / cov["50"]["nominal_finite_ensemble"])
        ratios90.append(cov["90"]["pooled"] / cov["90"]["nominal_finite_ensemble"])

    xx = np.arange(len(names))
    w = 0.36
    ax1.bar(xx - w / 2, ratios50, w, color=colors, alpha=0.50, label="50% interval",
            edgecolor="white", linewidth=0.4)
    ax1.bar(xx + w / 2, ratios90, w, color=colors, label="90% interval",
            edgecolor="white", linewidth=0.4)
    ax1.axhline(1.0, color=C_NOM, lw=1.0, ls="--")
    ax1.text(len(names) - 0.45, 1.03, "perfect", color=C_NOM, fontsize=7.5, ha="right")
    ax1.set_xticks(xx, names, fontsize=7.6)
    ax1.set_ylabel("Observed / nominal coverage")
    ax1.set_ylim(0, 1.15)
    from matplotlib.patches import Patch
    ax1.legend(handles=[Patch(facecolor="#888888", alpha=0.50, label="50% interval"),
                        Patch(facecolor="#888888", label="90% interval")],
               frameon=False, loc="upper left")
    ax1.grid(axis="y", lw=0.4, alpha=0.35)
    ax1.set_axisbelow(True)
    ax1.set_title("(a) Central-interval coverage (finite-ensemble corrected)", loc="left")

    # (b) rank histograms, active pixels, frequency vs uniform
    twin_c = load_calibration(sources[0][2])
    v2_c = load_calibration(sources[1][2])
    for c, color, label in ((v2_c, C_V2, "V2 (M=8, m50 gauge)"),
                            (twin_c, C_ENS, "Twin ensemble (M=3, m50)")):
        freq = np.array(c["rank_histogram"]["active_frequency"])
        ranks = np.arange(len(freq)) / (len(freq) - 1)
        ax2.plot(ranks, freq * (len(freq)), marker="o", ms=3, lw=1.2,
                 color=color, label=label)
    ax2.axhline(1.0, color=C_NOM, lw=1.0, ls="--")
    ax2.set_xlabel("Normalized rank of truth in ensemble")
    ax2.set_ylabel("Relative frequency")
    ax2.legend(frameon=False, fontsize=7.2)
    ax2.grid(lw=0.4, alpha=0.35)
    ax2.set_axisbelow(True)
    ax2.set_title("(b) Rank histogram (active pixels)", loc="left")

    fig.tight_layout()
    fig.savefig(HERE / "f6_calibration_comparison.pdf")
    plt.close(fig)
    print("wrote f6_calibration_comparison.pdf")


if __name__ == "__main__":
    fig_performance_vs_sparsity()
    fig_calibration_comparison()
