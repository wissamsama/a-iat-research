"""WP-ablation cost table (paper Sec 6.8): measured latency/memory/params for
Delta-Diff (40 denoising steps x 8 scenarios) vs Twin (1 forward pass) on one
real 64x64 tile from the Australia dense eval split, both loaded from their
trained checkpoints so the comparison uses real (not architecture-only)
models -- though timing/memory/param-count do not depend on training state.

Usage: python tools/benchmark_floodcastbench_cost.py --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from datasets.floodcastbench_diff_sparse_v2_dataset import build_diff_sparse_v2_dataset  # noqa: E402
from models.deterministic_twin import build_v2_family_model  # noqa: E402
from tools.evaluate_floodcastbench_diff_sparse_v2 import load_checkpoint  # noqa: E402
from tools.train_floodcastbench_diff_sparse_v1 import load_config, path_from_config, resolve_device  # noqa: E402


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def load_model_and_batch(config_path: str, checkpoint_path: str, device: torch.device, patch_size: int):
    config = load_config(Path(config_path))
    checkpoint = load_checkpoint(Path(checkpoint_path))
    stats = checkpoint["normalization_stats"]

    dataset = build_diff_sparse_v2_dataset(
        path_from_config(config, "dataset_root"), config, split="test", normalization_stats=stats, patch_mode="full"
    )
    model = build_v2_family_model(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if checkpoint.get("ema_state_dict"):
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                parameter.copy_(checkpoint["ema_state_dict"][name])
    model.eval()

    sample = dataset[0]
    ctx_len = config["dataset"]["context_length"]
    context = sample["context_water_masked"][:ctx_len, :patch_size, :patch_size].unsqueeze(0).to(device)
    mask = sample["sensor_mask"][:, :patch_size, :patch_size].unsqueeze(0).to(device)
    dem = sample["dem"][:, :patch_size, :patch_size].unsqueeze(0).to(device)
    rainfall = sample["rainfall"][: ctx_len + 1, :patch_size, :patch_size].unsqueeze(0).to(device)
    timestamps = sample["timestamps"][:ctx_len].unsqueeze(0).to(device)
    model_batch = {
        "context_water_masked": context,
        "sensor_mask": mask,
        "dem": dem,
        "rainfall_context": rainfall[:, :ctx_len],
        "rainfall_target": rainfall[:, ctx_len : ctx_len + 1],
        "timestamps_context": timestamps,
    }
    return model, model_batch, config


def time_sample(model: torch.nn.Module, model_batch: dict, n_scenarios: int, patch_size: int, repeats: int, device: torch.device) -> tuple[float, float]:
    with torch.no_grad():
        tokens, spatial = model.encode_context(model_batch)
        tokens_b = tokens.repeat_interleave(n_scenarios, dim=0)
        spatial_b = spatial.repeat_interleave(n_scenarios, dim=0)
        shape = (n_scenarios, 1, patch_size, patch_size)
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(device)
        times = []
        for _ in range(repeats):
            start = time.perf_counter()
            model.sample(tokens_b, spatial_b, shape)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
        peak_mem_mb = torch.cuda.max_memory_allocated(device) / 1e6 if device.type == "cuda" else float("nan")
        return sum(times) / len(times), peak_mem_mb


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diff-config", default="local_spark_configs/floodcastbench_diff_sparse_v2_highfid_60m.yaml")
    parser.add_argument("--diff-checkpoint", required=True)
    parser.add_argument("--twin-config", default="local_spark_configs/floodcastbench_det_twin_highfid_60m.yaml")
    parser.add_argument("--twin-checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--num-scenarios", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", default="reports/floodcastbench_cost_table.json")
    args = parser.parse_args()

    device = resolve_device(args.device)

    diff_model, diff_batch, diff_cfg = load_model_and_batch(args.diff_config, args.diff_checkpoint, device, args.patch_size)
    twin_model, twin_batch, twin_cfg = load_model_and_batch(args.twin_config, args.twin_checkpoint, device, args.patch_size)

    diff_params = count_params(diff_model)
    twin_params = count_params(twin_model)

    diff_latency_s, diff_mem_mb = time_sample(diff_model, diff_batch, args.num_scenarios, args.patch_size, args.repeats, device)
    twin_latency_s, twin_mem_mb = time_sample(twin_model, twin_batch, 1, args.patch_size, args.repeats, device)

    result = {
        "patch_size": args.patch_size,
        "num_scenarios_diff_diff": args.num_scenarios,
        "diffusion_steps": diff_cfg["diffusion"]["steps"],
        "network_evals_per_prediction": {
            "diff_diff": diff_cfg["diffusion"]["steps"] * args.num_scenarios,
            "twin": 1,
        },
        "params": {"diff_diff": diff_params, "twin": twin_params, "match": diff_params == twin_params},
        "latency_seconds_per_tile_prediction": {"diff_diff": diff_latency_s, "twin": twin_latency_s},
        "speedup_twin_over_diff": diff_latency_s / twin_latency_s if twin_latency_s > 0 else None,
        "peak_gpu_memory_mb": {"diff_diff": diff_mem_mb, "twin": twin_mem_mb},
        "device": str(device),
    }
    print(json.dumps(result, indent=2))
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nwritten to {out_path}")


if __name__ == "__main__":
    main()
