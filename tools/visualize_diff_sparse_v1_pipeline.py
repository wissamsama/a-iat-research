from __future__ import annotations

"""One-shot explanatory figure for the DIFF-SPARSE v1 pipeline: real inputs,
real masking, a real reverse-diffusion trajectory (captured mid-sampling, not
illustrative), and real multi-scenario predictions vs the real ground truth --
all from one trained checkpoint and one real dataset window. No synthetic or
mocked data anywhere; every panel is either a raw model input or an actual
tensor produced by running the trained model.

Usage:
    python tools/visualize_diff_sparse_v1_pipeline.py \
        --checkpoint <path/to/checkpoint_best.pth> \
        --split test --window-index 0 --num-scenarios 4 \
        --output <path/to/figure.png>
"""

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from datasets.floodcastbench_diff_sparse_v1_dataset import FloodCastBenchDiffSparseV1Dataset  # noqa: E402
from models.diff_sparse_v1 import DiffSparseV1Model  # noqa: E402
from tools.evaluate_floodcastbench_diff_sparse_v1 import load_checkpoint, to_physical  # noqa: E402
from tools.train_floodcastbench_diff_sparse_v1 import path_from_config, resolve_device  # noqa: E402


@torch.no_grad()
def sample_with_trajectory(
    model: DiffSparseV1Model,
    context_embedding: torch.Tensor,
    shape: tuple[int, ...],
    generator: torch.Generator,
    capture_steps: set[int],
) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
    """Real DDPM reverse sampling (Algorithm 2), capturing x_t at requested step
    indices. Identical math to DiffSparseV1Model.sample(), just with snapshots
    kept for visualization instead of discarded."""

    device = context_embedding.device
    x_t = torch.randn(shape, device=device, generator=generator, dtype=context_embedding.dtype)
    trajectory: dict[int, torch.Tensor] = {}
    if model.diffusion_steps in capture_steps:
        trajectory[model.diffusion_steps] = x_t[0, 0].clone().cpu()
    for step in reversed(range(model.diffusion_steps)):
        timesteps = torch.full((shape[0],), step, device=device, dtype=torch.long)
        x0_hat = model.denoise(x_t, timesteps, context_embedding)
        mean = model.posterior_coef_x0[step] * x0_hat + model.posterior_coef_xt[step] * x_t
        if step > 0:
            noise = torch.randn(shape, device=device, generator=generator, dtype=x_t.dtype)
            x_t = mean + torch.sqrt(model.posterior_variance[step].clamp(min=0.0)) * noise
        else:
            x_t = mean
        if step in capture_steps:
            trajectory[step] = x_t[0, 0].clone().cpu()
    return x_t, trajectory


def add_heatmap(ax, array: np.ndarray, title: str, cmap: str, vmin: float, vmax: float, norm=None) -> None:
    if norm is not None:
        artist = ax.imshow(array, cmap=cmap, norm=norm)
    else:
        artist = ax.imshow(array, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=10, fontweight="600", color="#1a1a19", pad=6)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("#e1e0d9")
    colorbar = plt.colorbar(artist, ax=ax, fraction=0.046, pad=0.03)
    colorbar.ax.tick_params(labelsize=7.5)
    # Force plain fixed-point notation: a near-constant array (e.g. a fully dry
    # rainfall panel) otherwise triggers matplotlib's automatic scientific
    # scale-factor notation, which silently rescales the displayed tick values
    # -- hiding only the scale-factor label (as an earlier version of this
    # script did) leaves the misleading rescaled numbers visible with no
    # indication they aren't the true values.
    colorbar.formatter.set_useOffset(False)
    colorbar.formatter.set_scientific(False)
    colorbar.update_ticks()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--window-index", type=int, default=0)
    parser.add_argument("--patch-seed", type=int, default=123, help="RNG seed for the illustrative random crop/mask")
    parser.add_argument("--num-scenarios", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    device = resolve_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint)
    config = checkpoint["config"]
    stats = checkpoint["normalization_stats"]
    water_stats = stats["channels"]["water"]
    water_mean, water_std = float(water_stats["mean"]), float(water_stats["std"])

    model = DiffSparseV1Model(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    root = path_from_config(config, "dataset_root")
    torch.manual_seed(args.patch_seed)
    dataset = FloodCastBenchDiffSparseV1Dataset(root, config, split=args.split, normalization_stats=stats, patch_mode="random")
    sample = dataset[args.window_index]
    context_length = dataset.context_length
    missing_rate = dataset.missing_rate

    batch = {key: value.unsqueeze(0).to(device) for key, value in sample.items() if key != "meta"}
    model_batch = {
        "context_water_masked": batch["context_water_masked"],
        "sensor_mask": batch["sensor_mask"],
        "dem": batch["dem"],
        "rainfall_context": batch["rainfall"][:, :context_length],
        "timestamps_context": batch["timestamps"][:, :context_length],
    }
    target_normalized = batch["target"][:, 0:1]  # first predicted step, real ground truth

    context_embedding = model.encode_context(model_batch)
    shape = target_normalized.shape

    capture_steps = {model.diffusion_steps, 15, 10, 5, 1, 0}
    generator = torch.Generator(device=device).manual_seed(args.patch_seed * 1000 + 1)
    scenario0, trajectory = sample_with_trajectory(model, context_embedding, shape, generator, capture_steps)

    scenarios = [scenario0[0, 0].clone().cpu()]
    for scenario_index in range(1, args.num_scenarios):
        generator = torch.Generator(device=device).manual_seed(args.patch_seed * 1000 + 1 + scenario_index)
        prediction = model.sample(context_embedding, shape, generator=generator)
        scenarios.append(prediction[0, 0].clone().cpu())
    mean_forecast = torch.stack(scenarios).mean(dim=0)

    # ---- physical-unit conversions for real-quantity panels ----
    def water_to_physical(normalized: torch.Tensor) -> np.ndarray:
        return (normalized * water_std + water_mean).numpy()

    context_last_masked = water_to_physical(batch["context_water_masked"][0, -1].cpu())
    sensor_mask_map = batch["sensor_mask"][0, 0].cpu().numpy()
    dem_physical = (batch["dem"][0, 0].cpu() * float(stats["channels"]["dem"]["std"]) + float(stats["channels"]["dem"]["mean"])).numpy()
    rainfall_last_context = (model_batch["rainfall_context"][0, -1].cpu() * float(stats["channels"]["rainfall"]["std"]) + float(stats["channels"]["rainfall"]["mean"])).numpy()
    target_physical = water_to_physical(target_normalized[0, 0].cpu())
    mean_forecast_physical = water_to_physical(mean_forecast)
    scenario_physical = [water_to_physical(s) for s in scenarios]

    water_vmin = float(min(context_last_masked.min(), target_physical.min(), mean_forecast_physical.min(), *(s.min() for s in scenario_physical)))
    water_vmax = float(max(context_last_masked.max(), target_physical.max(), mean_forecast_physical.max(), *(s.max() for s in scenario_physical)))

    # ---- figure layout ----
    plt.rcParams["figure.facecolor"] = "#fcfcfb"
    plt.rcParams["axes.facecolor"] = "#fcfcfb"
    n_traj = len(capture_steps)
    n_scenario_cols = max(len(scenarios), 2)
    ncols = max(4, n_traj, n_scenario_cols + 1)
    fig = plt.figure(figsize=(2.75 * ncols, 2.6 * 3 + 0.6), dpi=150, constrained_layout=True)
    gridspec = fig.add_gridspec(3, ncols)

    fig.suptitle(
        f"DIFF-SPARSE v1 — pipeline réel, aucune donnée illustrative (checkpoint "
        f"{args.checkpoint.parent.name}, missing_rate={missing_rate:.2f}, split={args.split}, "
        f"fenêtre {args.window_index})\n"
        "Rangée 1 : entrées réelles du modèle  —  Rangée 2 : vraie trajectoire de diffusion inverse "
        "(un scénario, échelle indépendante par étape)  —  Rangée 3 : vérité terrain vs scénarios réels (échelle physique partagée)",
        fontsize=12.5, fontweight="700", color="#0b0b0b",
    )

    # Row 1: real inputs
    row1 = [
        ("Contexte eau masqué\n(dernière frame, réel)", context_last_masked, "viridis", water_vmin, water_vmax, None),
        ("Masque capteurs\n(1=observé, 0=manquant)", sensor_mask_map, "gray", 0.0, 1.0, None),
        ("DEM (élévation)", dem_physical, "terrain", None, None, None),
        ("Pluie\n(dernière frame contexte)", rainfall_last_context, "Blues", None, None, None),
    ]
    for col, (title, array, cmap, vmin, vmax, norm) in enumerate(row1):
        ax = fig.add_subplot(gridspec[0, col])
        add_heatmap(ax, array, title, cmap, vmin, vmax, norm)
    for col in range(len(row1), ncols):
        fig.add_subplot(gridspec[0, col]).axis("off")

    # Row 2: real reverse-diffusion trajectory (latent/noisy space, diverging colormap).
    # Each panel is normalized to ITS OWN +/- max (noted in the title) rather than a
    # shared scale: x_T's unit-variance noise is ~30x smaller in amplitude than the
    # final x_0 signal, so a shared scale would render every early step as blank white.
    ordered_steps = sorted(trajectory.keys(), reverse=True)
    for col, step in enumerate(ordered_steps):
        ax = fig.add_subplot(gridspec[1, col])
        label = "x_T (bruit pur)" if step == model.diffusion_steps else ("x_0 (débruitage final)" if step == 0 else f"x_{step}")
        panel = trajectory[step].numpy()
        abs_max = float(np.abs(panel).max()) or 1e-6
        norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0.0, vmax=abs_max)
        add_heatmap(ax, panel, f"{label}\néchelle indiv. ±{abs_max:.2f}", "RdBu_r", None, None, norm)
    for col in range(len(ordered_steps), ncols):
        fig.add_subplot(gridspec[1, col]).axis("off")

    # Row 3: real ground truth vs real scenarios (physical water depth, shared scale)
    ax = fig.add_subplot(gridspec[2, 0])
    add_heatmap(ax, target_physical, "Vérité terrain réelle\n(frame future observée)", "viridis", water_vmin, water_vmax)
    ax = fig.add_subplot(gridspec[2, 1])
    add_heatmap(ax, mean_forecast_physical, f"Prévision moyenne\n({len(scenarios)} scénarios réels)", "viridis", water_vmin, water_vmax)
    for index, scenario in enumerate(scenario_physical):
        col = 2 + index
        if col >= ncols:
            break
        ax = fig.add_subplot(gridspec[2, col])
        add_heatmap(ax, scenario, f"Scénario réel #{index + 1}", "viridis", water_vmin, water_vmax)
    for col in range(2 + len(scenario_physical), ncols):
        fig.add_subplot(gridspec[2, col]).axis("off")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"figure written: {args.output}")
    print(f"missing_rate={missing_rate} context_length={context_length} diffusion_steps={model.diffusion_steps}")
    print(f"water range shown (physical m): [{water_vmin:.4f}, {water_vmax:.4f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
