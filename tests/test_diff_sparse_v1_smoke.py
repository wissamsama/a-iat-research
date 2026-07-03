from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
import yaml

from datasets.floodcastbench_diff_sparse_v1_dataset import (
    FloodCastBenchDiffSparseV1Dataset,
    apply_observation_masking,
    generate_sensor_mask,
    split_frame_ranges,
    window_starts_for_split,
)
from models.diff_sparse_v1 import DiffSparseV1Model
from tools.evaluate_floodcastbench_diff_sparse_v1 import (
    OfficialMetricAccumulator,
    persistence_forecast,
    tile_blend_window,
    tile_positions,
    to_physical,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_DIR / "configs" / "floodcastbench_diff_sparse_v1_highfid_60m.yaml"


def _small_config(missing_rate: float = 0.5) -> dict:
    return {
        "dataset": {
            "context_length": 4,
            "prediction_length": 2,
            "patch_size": 32,
            "include_dem": True,
            "include_rainfall": True,
            "include_covariates": True,
        },
        "masking": {"missing_rate": missing_rate, "mask_mode": "noise"},
        "model": {
            "context_conv_channels": [8, 16],
            "context_embedding_dim": 16,
            "covariate_dim": 8,
            "unet_channels": [8, 16, 16, 32],
            "resnet_layers_per_block": 1,
            "cross_attention_blocks": 2,
            "attention_heads": 2,
            "groupnorm_groups": 4,
            "time_embedding_dim": 32,
            "conditioning": "cross_attention_concat",
        },
        "diffusion": {
            "steps": 20,
            "beta_schedule": "linear",
            "beta_start": 0.0001,
            "beta_end": 1.0,
            "prediction_type": "x0",
        },
    }


def _fake_model_batch(config: dict, batch: int = 2, size: int = 32) -> dict:
    c = config["dataset"]["context_length"]
    return {
        "context_water_masked": torch.randn(batch, c, size, size),
        "sensor_mask": torch.ones(batch, 1, size, size),
        "dem": torch.randn(batch, 1, size, size),
        "rainfall_context": torch.randn(batch, c, size, size),
        "timestamps_context": torch.arange(c, dtype=torch.float32).repeat(batch, 1) * 300.0,
        "target": torch.randn(batch, 1, size, size),
    }


def test_terminal_snr_is_exactly_zero() -> None:
    """Regression test for the truncated-schedule bug: with beta_end=1.0 the
    forward process must end at pure noise so sampling from N(0, I) is in-distribution."""

    model = DiffSparseV1Model(_small_config())
    assert float(model.sqrt_alpha_cumprod[-1]) == 0.0
    assert float(model.sqrt_one_minus_alpha_cumprod[-1]) == 1.0
    # First reverse step must fully discard the initial noise.
    assert float(model.posterior_coef_xt[-1]) == 0.0


def test_q_sample_terminal_step_is_pure_noise() -> None:
    model = DiffSparseV1Model(_small_config())
    x0 = torch.randn(2, 1, 8, 8)
    noise = torch.randn_like(x0)
    t = torch.full((2,), model.diffusion_steps - 1, dtype=torch.long)
    assert torch.allclose(model.q_sample(x0, t, noise=noise), noise)


def test_oracle_denoiser_recovers_x0_exactly() -> None:
    """With a perfect x0 predictor the sampler must return x0 regardless of noise."""

    model = DiffSparseV1Model(_small_config())
    x0 = torch.randn(2, 1, 8, 8)
    context = torch.zeros(2, 16, 8, 8)
    result = model.sample(context, x0.shape, denoiser=lambda x_t, t: x0)
    assert torch.allclose(result, x0, atol=1e-5)


def test_sensor_mask_fraction_and_noise_masking() -> None:
    generator = torch.Generator().manual_seed(0)
    mask = generate_sensor_mask(32, 32, missing_rate=0.95, generator=generator)
    assert tuple(mask.shape) == (1, 32, 32)
    assert int(mask.sum().item()) == round(0.05 * 32 * 32)

    water = torch.randn(4, 32, 32)
    masked = apply_observation_masking(water, mask, "noise", generator=generator)
    sensor_cells = mask.bool().expand_as(water)
    assert torch.equal(masked[sensor_cells], water[sensor_cells])
    assert not torch.equal(masked[~sensor_cells], water[~sensor_cells])

    zero_masked = apply_observation_masking(water, mask, "zeros")
    assert torch.all(zero_masked[~sensor_cells] == 0)

    dense = generate_sensor_mask(16, 16, missing_rate=0.0)
    assert torch.all(dense == 1)
    assert torch.equal(apply_observation_masking(water[:, :16, :16], dense, "noise"), water[:, :16, :16])


def test_split_frame_ranges_match_canonical_windows() -> None:
    ranges = split_frame_ranges(2881)
    assert ranges == {"train": (0, 2320), "val": (2320, 2600), "test": (2600, 2881)}
    assert len(window_starts_for_split(ranges["train"], 20, 1)) == 2301
    assert len(window_starts_for_split(ranges["val"], 20, 20)) == 14
    assert len(window_starts_for_split(ranges["test"], 20, 20)) == 14


def test_model_forward_and_training_step() -> None:
    config = _small_config()
    model = DiffSparseV1Model(config)
    batch = _fake_model_batch(config)
    loss, diagnostics = model.training_step_loss(batch)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert diagnostics["pred_finite"] == 1.0

    embedding = model.encode_context(batch)
    assert tuple(embedding.shape) == (2, 16, 32, 32)
    sample = model.sample(embedding, (2, 1, 32, 32))
    assert tuple(sample.shape) == (2, 1, 32, 32)
    assert torch.isfinite(sample).all()


def test_paper_faithful_attention_only_conditioning() -> None:
    config = _small_config()
    config["model"]["conditioning"] = "cross_attention"
    model = DiffSparseV1Model(config)
    batch = _fake_model_batch(config)
    loss, _ = model.training_step_loss(batch)
    assert torch.isfinite(loss)


def _real_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _require_data(config: dict) -> Path:
    root = Path(config["paths"]["dataset_root"])
    water_dir = root / "High-fidelity flood forecasting" / "60m" / "Australia"
    if not water_dir.exists():
        pytest.skip(f"FloodCastBench data unavailable: {water_dir}")
    return root


def test_dataset_window_counts_and_shapes() -> None:
    config = _real_config()
    root = _require_data(config)
    train = FloodCastBenchDiffSparseV1Dataset(root, config, split="train")
    val = FloodCastBenchDiffSparseV1Dataset(root, config, split="val")
    test = FloodCastBenchDiffSparseV1Dataset(root, config, split="test")
    assert len(train) == 2301
    assert len(val) == 14
    assert len(test) == 14

    sample = train[0]
    c, l, p = train.context_length, train.prediction_length, train.patch_size
    assert tuple(sample["context_water_masked"].shape) == (c, p, p)
    assert tuple(sample["sensor_mask"].shape) == (1, p, p)
    assert tuple(sample["dem"].shape) == (1, p, p)
    assert tuple(sample["rainfall"].shape) == (c + l, p, p)
    assert tuple(sample["timestamps"].shape) == (c + l,)
    assert tuple(sample["target"].shape) == (l, p, p)
    assert torch.isfinite(sample["target"]).all()

    full = val[0]
    assert tuple(full["target"].shape) == (l, 536, 536)
    assert tuple(full["sensor_mask"].shape) == (1, 536, 536)


def test_eval_masks_are_deterministic_round_robin() -> None:
    config = _real_config()
    config["masking"]["missing_rate"] = 0.95
    root = _require_data(config)
    val_a = FloodCastBenchDiffSparseV1Dataset(root, config, split="val")
    val_b = FloodCastBenchDiffSparseV1Dataset(root, config, split="val")
    mask_a = val_a[0]["sensor_mask"]
    mask_b = val_b[0]["sensor_mask"]
    assert torch.equal(mask_a, mask_b)
    bank = val_a.eval_mask_bank_size
    assert torch.equal(val_a[0]["sensor_mask"], val_a[bank % len(val_a)]["sensor_mask"]) or bank >= len(val_a)


def test_rollout_tiles_use_overlap_and_cover_field_end() -> None:
    starts = tile_positions(size=536, patch=64, stride=48)
    assert starts[0] == 0
    assert starts[-1] == 536 - 64
    assert max(b - a for a, b in zip(starts, starts[1:])) <= 48
    assert len(starts) > len(tile_positions(size=536, patch=64, stride=64))


def test_tile_blend_window_is_positive_and_center_weighted() -> None:
    window = tile_blend_window(64, device=torch.device("cpu"), dtype=torch.float32)
    assert tuple(window.shape) == (64, 64)
    assert torch.isfinite(window).all()
    assert float(window.min().item()) > 0.0
    assert float(window.max().item()) == pytest.approx(1.0)
    assert window[32, 32] > window[0, 0]


def test_sparse_persistence_uses_sensor_cells_and_train_mean_fill() -> None:
    sample = {
        "context_water_true": torch.tensor(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]],
            ]
        ),
        "sensor_mask": torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]),
    }
    sparse = persistence_forecast(sample, prediction_length=3, device=torch.device("cpu"), mode="sparse")
    oracle = persistence_forecast(sample, prediction_length=3, device=torch.device("cpu"), mode="oracle")
    expected_sparse = torch.tensor([[5.0, 0.0], [0.0, 8.0]])
    expected_oracle = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    assert tuple(sparse.shape) == (3, 2, 2)
    assert torch.equal(sparse[0], expected_sparse)
    assert torch.equal(sparse[1], expected_sparse)
    assert torch.equal(oracle[0], expected_oracle)


def test_to_physical_round_trip_from_standardized_water_stats() -> None:
    water_stats = {"mean": 2.5, "std": 0.25}
    physical = torch.tensor([[2.25, 2.5], [2.75, 3.0]])
    normalized = (physical - float(water_stats["mean"])) / float(water_stats["std"])
    assert torch.allclose(to_physical(normalized, water_stats), physical)


def test_official_metric_accumulator_finite_on_tiny_physical_pair() -> None:
    accumulator = OfficialMetricAccumulator(gammas=[0.001])
    pred = torch.tensor([[0.0, 0.002], [0.004, -0.001]])
    target = torch.tensor([[0.0, 0.003], [0.001, 0.003]])
    accumulator.update(pred, target)
    metrics = accumulator.compute()
    for key in ("classical_rmse", "current_relative_rmse", "nse", "pearson_r", "mae", "csi_gamma_0_001"):
        assert math.isfinite(float(metrics[key]))
    assert metrics["tp_gamma_0_001"] == 1
    assert metrics["fp_gamma_0_001"] == 1
    assert metrics["fn_gamma_0_001"] == 1
