from __future__ import annotations

"""Same explanatory figure as visualize_diff_sparse_v1_pipeline.py, adapted for
DIFF-SPARSE v2 (temporal tokens + spatial encoder, delta-space prediction).
Real inputs, real reverse-diffusion trajectory, real multi-scenario
predictions vs real ground truth, from one trained pilot checkpoint.
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from datasets.floodcastbench_diff_sparse_v2_dataset import build_diff_sparse_v2_dataset  # noqa: E402
from models.diff_sparse_v2 import DiffSparseV2Model  # noqa: E402
from tools.evaluate_floodcastbench_diff_sparse_v2 import load_checkpoint  # noqa: E402
from tools.train_floodcastbench_diff_sparse_v1 import path_from_config, resolve_device  # noqa: E402
from tools.train_floodcastbench_diff_sparse_v2 import DeltaSpec, prepare_model_batch  # noqa: E402
from tools.visualize_diff_sparse_v1_pipeline import add_heatmap  # noqa: E402


@torch.no_grad()
def sample_with_trajectory(model, tokens, spatial, shape, generator, capture_steps, clip_x0):
    device = spatial.device
    x_t = torch.randn(shape, device=device, generator=generator, dtype=spatial.dtype)
    trajectory: dict[int, torch.Tensor] = {}
    if model.diffusion_steps in capture_steps:
        trajectory[model.diffusion_steps] = x_t[0, 0].clone().cpu()
    for step in reversed(range(model.diffusion_steps)):
        timesteps = torch.full((shape[0],), step, device=device, dtype=torch.long)
        x0_hat = model.denoise(x_t, timesteps, tokens, spatial)
        if clip_x0 is not None:
            floor, ceiling = clip_x0
            if floor is not None:
                x0_hat = torch.maximum(x0_hat, floor.to(x0_hat.dtype)) if torch.is_tensor(floor) else x0_hat.clamp(min=float(floor))
        mean = model.posterior_coef_x0[step] * x0_hat + model.posterior_coef_xt[step] * x_t
        if step > 0:
            noise = torch.randn(shape, device=device, generator=generator, dtype=x_t.dtype)
            x_t = mean + torch.sqrt(model.posterior_variance[step].clamp(min=0.0)) * noise
        else:
            x_t = mean
        if step in capture_steps:
            trajectory[step] = x_t[0, 0].clone().cpu()
    return x_t, trajectory


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--window-index", type=int, default=0)
    parser.add_argument("--patch-seed", type=int, default=123)
    parser.add_argument("--num-scenarios", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    device = resolve_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint)
    config = checkpoint["config"]
    stats = checkpoint["normalization_stats"]
    delta_stats = checkpoint.get("delta_stats")
    water_stats = stats["channels"]["water"]
    water_mean, water_std = float(water_stats["mean"]), float(water_stats["std"])

    prediction_config = config.get("prediction", {})
    delta = DeltaSpec(str(prediction_config.get("target", "delta")), water_stats, delta_stats)
    gamma_wet = float(prediction_config.get("wet_threshold_m", 0.001))
    wet_threshold_normalized = delta.floor_absolute + gamma_wet / water_std

    model = DiffSparseV2Model(config).to(device)
    use_ema = checkpoint.get("ema_state_dict") is not None
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if use_ema:
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                parameter.copy_(checkpoint["ema_state_dict"][name])
    model.eval()

    root = path_from_config(config, "dataset_root")
    torch.manual_seed(args.patch_seed)
    dataset = build_diff_sparse_v2_dataset(root, config, split=args.split, normalization_stats=stats, patch_mode="random")
    dataset.augmentation = False  # keep the illustrative crop un-augmented
    context_length = dataset.context_length
    missing_rate = dataset.missing_rate
    sample = dataset[args.window_index]

    raw_batch = {key: value.unsqueeze(0).to(device) for key, value in sample.items() if key != "meta"}
    model_batch = prepare_model_batch(raw_batch, context_length, delta, wet_threshold_normalized, change_weight=0.0)
    target_absolute = model_batch["target_absolute"]
    base = model_batch["base"]

    tokens, spatial = model.encode_context(model_batch)
    shape = model_batch["target"].shape
    clip_x0 = delta.clip_for_sampler(base, enabled=True)

    capture_steps = {model.diffusion_steps, 30, 20, 10, 3, 0}
    generator = torch.Generator(device=device).manual_seed(args.patch_seed * 1000 + 1)
    scenario0_pred, trajectory = sample_with_trajectory(model, tokens, spatial, shape, generator, capture_steps, clip_x0)

    scenario_preds = [scenario0_pred]
    for scenario_index in range(1, args.num_scenarios):
        generator = torch.Generator(device=device).manual_seed(args.patch_seed * 1000 + 1 + scenario_index)
        prediction = model.sample(tokens, spatial, shape, generator=generator, clip_x0=clip_x0)
        scenario_preds.append(prediction)
    scenarios_absolute = [delta.to_absolute(p, base, clamp=True)[0, 0].clone().cpu() for p in scenario_preds]
    mean_forecast_absolute = torch.stack(scenarios_absolute).mean(dim=0)

    def water_to_physical(normalized: torch.Tensor) -> np.ndarray:
        return (normalized * water_std + water_mean).numpy()

    context_last_masked = water_to_physical(raw_batch["context_water_masked"][0, -1].cpu())
    sensor_mask_map = raw_batch["sensor_mask"][0, 0].cpu().numpy()
    dem_physical = (raw_batch["dem"][0, 0].cpu() * float(stats["channels"]["dem"]["std"]) + float(stats["channels"]["dem"]["mean"])).numpy()
    rainfall_last_context = (model_batch["rainfall_context"][0, -1].cpu() * float(stats["channels"]["rainfall"]["std"]) + float(stats["channels"]["rainfall"]["mean"])).numpy()
    rainfall_target = (model_batch["rainfall_target"][0, 0].cpu() * float(stats["channels"]["rainfall"]["std"]) + float(stats["channels"]["rainfall"]["mean"])).numpy()
    target_physical = water_to_physical(target_absolute[0, 0].cpu())
    mean_forecast_physical = water_to_physical(mean_forecast_absolute)
    scenario_physical = [water_to_physical(s) for s in scenarios_absolute]

    water_vmin = float(min(context_last_masked.min(), target_physical.min(), mean_forecast_physical.min(), *(s.min() for s in scenario_physical)))
    water_vmax = float(max(context_last_masked.max(), target_physical.max(), mean_forecast_physical.max(), *(s.max() for s in scenario_physical)))

    plt.rcParams["figure.facecolor"] = "#fcfcfb"
    plt.rcParams["axes.facecolor"] = "#fcfcfb"
    n_traj = len(capture_steps)
    n_scenario_cols = max(len(scenario_physical), 2)
    ncols = max(5, n_traj, n_scenario_cols + 1)
    fig = plt.figure(figsize=(2.75 * ncols, 2.6 * 3 + 0.6), dpi=150, constrained_layout=True)
    gridspec = fig.add_gridspec(3, ncols)
    fig.suptitle(
        f"DIFF-SPARSE v2 — pipeline réel, aucune donnée illustrative (checkpoint pilote "
        f"{args.checkpoint.parent.name}, ema={use_ema}, missing_rate={missing_rate:.2f}, "
        f"split={args.split}, fenêtre {args.window_index}, espace cible={delta.mode})\n"
        "Rangée 1 : entrées réelles (dont pluie du pas prédit, nouveau en v2)  —  "
        "Rangée 2 : vraie trajectoire de diffusion inverse en espace delta (échelle indép. par étape)  —  "
        "Rangée 3 : vérité terrain vs scénarios réels reconstruits en absolu (échelle physique partagée)",
        fontsize=12, fontweight="700", color="#0b0b0b",
    )

    row1 = [
        ("Contexte eau masqué\n(dernière frame, réel)", context_last_masked, "viridis", water_vmin, water_vmax, None),
        ("Masque capteurs\n(1=observé, 0=manquant)", sensor_mask_map, "gray", 0.0, 1.0, None),
        ("DEM (élévation)", dem_physical, "terrain", None, None, None),
        ("Pluie\n(dernière frame contexte)", rainfall_last_context, "Blues", None, None, None),
        ("Pluie du pas PRÉDIT\n(nouveau en v2)", rainfall_target, "Blues", None, None, None),
    ]
    for col, (title, array, cmap, vmin, vmax, norm) in enumerate(row1):
        ax = fig.add_subplot(gridspec[0, col])
        add_heatmap(ax, array, title, cmap, vmin, vmax, norm)
    for col in range(len(row1), ncols):
        fig.add_subplot(gridspec[0, col]).axis("off")

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

    ax = fig.add_subplot(gridspec[2, 0])
    add_heatmap(ax, target_physical, "Vérité terrain réelle\n(frame future observée)", "viridis", water_vmin, water_vmax)
    ax = fig.add_subplot(gridspec[2, 1])
    add_heatmap(ax, mean_forecast_physical, f"Prévision moyenne\n({len(scenario_physical)} scénarios réels)", "viridis", water_vmin, water_vmax)
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
    print(f"missing_rate={missing_rate} context_length={context_length} diffusion_steps={model.diffusion_steps} target_space={delta.mode}")
    print(f"water range shown (physical m): [{water_vmin:.4f}, {water_vmax:.4f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
