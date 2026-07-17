from __future__ import annotations

"""Qualitative prediction maps for the paper (ground truth vs models).

For ONE test window and ONE fixed evaluation mask, rolls out the diffusion
forecaster (mean of M scenarios) and its deterministic twin, then composes
a publication figure: rows = lead times, columns = ground truth, each
model's forecast and absolute error. Shared color scales (depth: viridis,
error: magma) so panels are comparable at a glance. Raw arrays are also
saved to .npz for reproducibility.

Both models must share the same dataset/config family (the V2-family
pipeline); the sensor mask and window are identical across models by
construction, so differences are attributable to the models alone.

Usage:
  python tools/build_qualitative_maps_figure.py \
      --ours-config configs/floodcastbench_diff_sparse_v2_highfid_60m.yaml \
      --ours-checkpoint <ckpt_best.pth> \
      --twin-config configs/floodcastbench_det_twin_highfid_60m.yaml \
      --twin-checkpoint <ckpt_best.pth> \
      --missing-rate 0.95 --window 0 --horizons 1 6 12 \
      --output paper/figures/f7_qualitative_maps.pdf
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from datasets.floodcastbench_diff_sparse_v2_dataset import build_diff_sparse_v2_dataset  # noqa: E402
from models.deterministic_twin import build_v2_family_model  # noqa: E402
from tools.evaluate_floodcastbench_diff_sparse_v2 import load_checkpoint, rollout_window  # noqa: E402
from tools.evaluate_floodcastbench_diff_sparse_v1 import to_physical  # noqa: E402
from tools.train_floodcastbench_diff_sparse_v1 import load_config, path_from_config, resolve_device  # noqa: E402
from tools.train_floodcastbench_diff_sparse_v2 import DeltaSpec, load_delta_stats  # noqa: E402
from training.utils import set_seed  # noqa: E402

EVAL_MASK_SEED = 42


@torch.no_grad()
def run_model(config_path: Path, checkpoint_path: Path, missing_rate: float,
              window: int, num_scenarios: int, device: torch.device):
    config = load_config(config_path)
    config.setdefault("masking", {})["missing_rate"] = float(missing_rate)
    set_seed(EVAL_MASK_SEED)

    checkpoint = load_checkpoint(checkpoint_path)
    stats = checkpoint["normalization_stats"]
    water_stats = stats["channels"]["water"]

    dataset = build_diff_sparse_v2_dataset(
        path_from_config(config, "dataset_root"), config,
        split="test", normalization_stats=stats, patch_mode="full",
    )
    model = build_v2_family_model(config).to(device)
    use_ema = bool(config.get("evaluation", {}).get("use_ema", True)) and checkpoint.get("ema_state_dict")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if use_ema:
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                parameter.copy_(checkpoint["ema_state_dict"][name])
    model.eval()

    prediction_config = config.get("prediction", {})
    delta_stats = checkpoint.get("delta_stats") or load_delta_stats(
        Path(prediction_config["delta_stats_json"]) if prediction_config.get("delta_stats_json") else None
    )
    delta = DeltaSpec(str(prediction_config.get("target", "delta")), water_stats, delta_stats)

    sample = dataset[window]
    generator = torch.Generator(device=device).manual_seed(EVAL_MASK_SEED * 1_000_003 + window)
    predictions = rollout_window(
        model, sample,
        num_scenarios=num_scenarios,
        patch_size=dataset.patch_size,
        tile_stride=max(1, dataset.patch_size // 2),
        tile_chunk=64,
        remask=bool(config.get("evaluation", {}).get("rollout_remask", False)),
        mask_mode=dataset.mask_mode,
        delta=delta, clamp=True, generator=generator, device=device,
    )
    mean_forecast = predictions.mean(dim=0).clamp(min=delta.floor_absolute)
    forecast_physical = to_physical(mean_forecast, water_stats).clamp(min=0.0).cpu().numpy()
    target_physical = to_physical(sample["target"].to(device), water_stats).clamp(min=0.0).cpu().numpy()
    mask = sample["sensor_mask"][0].cpu().numpy()
    return forecast_physical, target_physical, mask


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ours-config", type=Path)
    parser.add_argument("--ours-checkpoint", type=Path)
    parser.add_argument("--twin-config", type=Path)
    parser.add_argument("--twin-checkpoint", type=Path)
    parser.add_argument("--missing-rate", type=float, default=0.95)
    parser.add_argument("--window", type=int, default=0)
    parser.add_argument("--num-scenarios", type=int, default=8)
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 6, 12])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--from-npz", type=Path,
                        help="Re-plot from a previously saved .npz (skips GPU inference entirely).")
    args = parser.parse_args()

    npz_path = args.output.with_suffix(".npz")
    if args.from_npz is not None:
        data = np.load(args.from_npz)
        target, ours, twin = data["target"], data["ours"], data["twin"]
        args.horizons = list(data["horizons"])
    else:
        for name in ("ours_config", "ours_checkpoint", "twin_config", "twin_checkpoint"):
            if getattr(args, name) is None:
                raise SystemExit(f"--{name.replace('_', '-')} is required unless --from-npz is given")
        device = resolve_device(args.device)
        ours, target, mask = run_model(args.ours_config, args.ours_checkpoint,
                                       args.missing_rate, args.window, args.num_scenarios, device)
        twin, target2, mask2 = run_model(args.twin_config, args.twin_checkpoint,
                                         args.missing_rate, args.window, 1, device)
        if not np.allclose(target, target2):
            raise RuntimeError("targets differ between the two runs -- window/mask mismatch")
        if not np.allclose(mask, mask2):
            raise RuntimeError("sensor masks differ between the two runs -- not a controlled comparison")

        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(npz_path, target=target, ours=ours, twin=twin, mask=mask,
                            horizons=np.array(args.horizons), missing_rate=args.missing_rate,
                            window=args.window)

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 8,
        "axes.titlesize": 8.5,
    })
    steps = [h - 1 for h in args.horizons]
    # Water depth is heavy-tailed (a few deep-channel pixels dwarf the flood
    # extent); a colormap capped at the true max renders as near-black.
    # Cap at a high percentile of the GROUND TRUTH (not the models, which
    # can spike higher on outlier pixels) so all panels share one physically
    # meaningful scale and the actual flood texture stays visible.
    depth_max = float(np.percentile(target[steps], 99.0))
    err_ours = np.abs(ours - target)
    err_twin = np.abs(twin - target)
    err_max = float(np.percentile(np.concatenate([err_ours[steps], err_twin[steps]]), 99.0))

    ncols, nrows = 5, len(steps)
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.1 * ncols, 2.15 * nrows), squeeze=False)
    col_titles = ["Ground truth", "$\\Delta$-Diff (mean of 8)", "$\\Delta$-Diff $|$error$|$",
                  "Twin", "Twin $|$error$|$"]
    ims = []
    for r, step in enumerate(steps):
        panels = [
            (target[step], "viridis", depth_max),
            (ours[step], "viridis", depth_max),
            (err_ours[step], "magma", err_max),
            (twin[step], "viridis", depth_max),
            (err_twin[step], "magma", err_max),
        ]
        for c, (array, cmap, vmax) in enumerate(panels):
            ax = axes[r][c]
            im = ax.imshow(array, cmap=cmap, vmin=0.0, vmax=vmax, interpolation="nearest")
            ims.append((im, cmap))
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(col_titles[c])
            if c == 0:
                ax.set_ylabel(f"$t+{args.horizons[r]}$ ({args.horizons[r] * 5} min)", fontsize=8.5)

    depth_im = next(im for im, cmap in ims if cmap == "viridis")
    err_im = next(im for im, cmap in ims if cmap == "magma")
    fig.subplots_adjust(right=0.90, wspace=0.04, hspace=0.06)
    cax1 = fig.add_axes([0.915, 0.55, 0.013, 0.33])
    cax2 = fig.add_axes([0.915, 0.12, 0.013, 0.33])
    fig.colorbar(depth_im, cax=cax1).set_label("water depth (m)", fontsize=7.5)
    fig.colorbar(err_im, cax=cax2).set_label("$|$error$|$ (m)", fontsize=7.5)

    fig.savefig(args.output, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"figure written: {args.output}")
    print(f"arrays saved:   {npz_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
