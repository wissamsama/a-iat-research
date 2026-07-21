from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
import yaml

from datasets.floodcastbench_diff_sparse_v2_dataset import apply_dihedral, _StridedFrameView
from models.diff_sparse_v2 import (
    ConsistencyLoss,
    DiffSparseV2Model,
    _temporal_down_block_output_size,
)
from tools.evaluate_floodcastbench_diff_sparse_v2 import MultiHorizonPathAccumulator
from tools.train_floodcastbench_diff_sparse_v2 import (
    DeltaSpec,
    ExponentialMovingAverage,
    change_weight_map,
    normalized_zero_depth,
    prepare_model_batch,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_DIR / "configs" / "floodcastbench_diff_sparse_v2_highfid_60m.yaml"


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
            "context_groupnorm_groups": 4,
            "spatial_context_channels": 8,
            "spatial_context_hidden_channels": 16,
            "include_target_rainfall": True,
            "unet_channels": [8, 16, 16, 32],
            "resnet_layers_per_block": 1,
            "cross_attention_blocks": 2,
            "groupnorm_groups": 4,
            "dropout": 0.0,
        },
        "diffusion": {
            "steps": 20,
            "beta_schedule": "linear",
            "beta_start": 0.0001,
            "beta_end": 1.0,
            "prediction_type": "x0",
        },
        "loss": {"snr_gamma": None},
    }


def _fake_model_batch(config: dict, batch: int = 2, size: int = 32) -> dict:
    c = config["dataset"]["context_length"]
    return {
        "context_water_masked": torch.randn(batch, c, size, size),
        "sensor_mask": torch.ones(batch, 1, size, size),
        "dem": torch.randn(batch, 1, size, size),
        "rainfall_context": torch.randn(batch, c, size, size),
        "rainfall_target": torch.randn(batch, 1, size, size),
        "timestamps_context": torch.arange(c, dtype=torch.float32).repeat(batch, 1) * 300.0,
        "target": torch.randn(batch, 1, size, size),
    }


def test_terminal_snr_is_exactly_zero() -> None:
    model = DiffSparseV2Model(_small_config())
    assert float(model.sqrt_alpha_cumprod[-1]) == 0.0
    assert float(model.sqrt_one_minus_alpha_cumprod[-1]) == 1.0
    assert float(model.posterior_coef_xt[-1]) == 0.0


def test_reverse_diffusion_uses_raw_beta_variance() -> None:
    model = DiffSparseV2Model(_small_config())
    assert torch.equal(model.posterior_variance, model.betas)


def test_all_parameters_registered_before_first_forward() -> None:
    """Every trainable layer (temporal token_linear, spatial encoder, UNet) must
    be registered from construction -- see V1's lazily-built-layer regression."""

    model = DiffSparseV2Model(_small_config())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer_param_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
    for name, parameter in model.named_parameters():
        assert id(parameter) in optimizer_param_ids, f"{name} missing from optimizer"


def test_model_forward_training_step_and_sample() -> None:
    config = _small_config()
    model = DiffSparseV2Model(config)
    batch = _fake_model_batch(config)
    loss, diagnostics = model.training_step_loss(batch)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert diagnostics["pred_finite"] == 1.0

    tokens, spatial = model.encode_context(batch)
    assert tuple(tokens.shape) == (2, config["dataset"]["context_length"], 16)
    assert tuple(spatial.shape) == (2, config["model"]["spatial_context_channels"], 32, 32)
    sample = model.sample(tokens, spatial, (2, 1, 32, 32))
    assert tuple(sample.shape) == (2, 1, 32, 32)
    assert torch.isfinite(sample).all()


def test_oracle_denoiser_recovers_x0_exactly() -> None:
    model = DiffSparseV2Model(_small_config())
    x0 = torch.randn(2, 1, 8, 8)
    tokens = torch.zeros(2, 4, 16)
    spatial = torch.zeros(2, 8, 8, 8)
    result = model.sample(tokens, spatial, x0.shape, denoiser=lambda x_t, t: x0)
    assert torch.allclose(result, x0, atol=1e-5)


def test_sampling_clip_floor_is_respected() -> None:
    """With a clip floor, no sampled value may fall below it (the V2 physical
    non-negativity guarantee driving the propagation-path IoU fix)."""

    model = DiffSparseV2Model(_small_config())
    floor = -0.25
    x0 = torch.full((2, 1, 8, 8), -3.0)  # oracle predicting far below the floor
    tokens = torch.zeros(2, 4, 16)
    spatial = torch.zeros(2, 8, 8, 8)
    result = model.sample(tokens, spatial, x0.shape, denoiser=lambda x_t, t: x0, clip_x0=(floor, None))
    assert float(result.min().item()) >= floor - 1e-6


def test_min_snr_loss_weights_shape_and_floor() -> None:
    config = _small_config()
    config["loss"] = {"snr_gamma": 5.0, "snr_weight_floor": 0.05}
    model = DiffSparseV2Model(config)
    assert model.loss_weights.shape == (20,)
    assert float(model.loss_weights.min().item()) > 0.0
    assert abs(float(model.loss_weights.mean().item()) - 1.0) < 1e-5
    batch = _fake_model_batch(config)
    loss, _ = model.training_step_loss(batch)
    assert torch.isfinite(loss)


def test_consistency_loss_penalizes_hydraulic_discontinuity() -> None:
    """The reference penalty relu(water_level_diff * elevation_diff) fires when a
    low cell's water surface (elevation + depth) rises ABOVE an adjacent higher
    cell's surface while that higher cell stays dry -- water that should have
    spilled uphill but did not. A low pool whose surface stays below the
    neighboring high ground is physically fine and incurs zero penalty."""

    consistency = ConsistencyLoss()
    elevation = torch.zeros(1, 1, 5, 5)
    elevation[0, 0, :, 3:] = 2.0  # a step up on the right

    ok_water = torch.zeros(1, 1, 5, 5)
    ok_water[0, 0, :, :3] = 1.0  # pool surface at level 1 < high ground level 2
    violating_water = torch.zeros(1, 1, 5, 5)
    violating_water[0, 0, :, :3] = 3.0  # pool surface at level 3 > dry high ground level 2

    ok_penalty = float(consistency(ok_water, elevation).item())
    violating_penalty = float(consistency(violating_water, elevation).item())
    assert ok_penalty == pytest.approx(0.0)
    assert violating_penalty > 0.0


def _delta_spec(mode: str = "delta") -> DeltaSpec:
    water_stats = {"mean": 0.1, "std": 0.25, "max": 10.0}
    delta_stats = {"delta_std_physical": 0.001}
    return DeltaSpec(mode, water_stats, delta_stats if mode == "delta" else None)


def test_prepare_model_batch_slices_target_rainfall_and_delta_target() -> None:
    c, l, size = 4, 2, 16
    batch = {
        "context_water_masked": torch.randn(2, c, size, size),
        "context_water_true": torch.randn(2, c, size, size),
        "sensor_mask": torch.ones(2, 1, size, size),
        "dem": torch.randn(2, 1, size, size),
        "rainfall": torch.arange(float(2 * (c + l) * size * size)).reshape(2, c + l, size, size),
        "timestamps": torch.arange(c + l, dtype=torch.float32).repeat(2, 1) * 300.0,
        "target": torch.randn(2, l, size, size),
    }
    delta = _delta_spec()
    model_batch = prepare_model_batch(batch, c, delta, wet_threshold_normalized=-0.396, change_weight=3.0)
    assert tuple(model_batch["rainfall_context"].shape) == (2, c, size, size)
    assert tuple(model_batch["rainfall_target"].shape) == (2, 1, size, size)
    assert torch.equal(model_batch["rainfall_target"][:, 0], batch["rainfall"][:, c])
    assert tuple(model_batch["target"].shape) == (2, 1, size, size)
    # Delta target round trip: base + scale*target == absolute target (dense mask).
    reconstructed = model_batch["base"] + delta.scale * model_batch["target"]
    assert torch.allclose(reconstructed, batch["target"][:, 0:1], atol=1e-5)
    # Dense mask -> base is exactly the true last context frame.
    assert torch.allclose(model_batch["base"][:, 0], batch["context_water_true"][:, -1])
    assert "pixel_weights" in model_batch


def test_delta_spec_round_trip_and_sampler_floor() -> None:
    delta = _delta_spec()
    assert delta.floor_absolute == pytest.approx(-0.4)
    assert delta.scale == pytest.approx(0.001 / 0.25)
    base = torch.full((2, 1, 4, 4), -0.35)
    absolute = torch.full((2, 1, 4, 4), -0.3)
    target = delta.to_target_space(absolute, base)
    assert torch.allclose(delta.to_absolute(target, base, clamp=False), absolute, atol=1e-6)
    # Clamped reconstruction never goes below the physical floor.
    deep_negative = torch.full((2, 1, 4, 4), -100.0)
    clamped = delta.to_absolute(deep_negative, base, clamp=True)
    assert float(clamped.min().item()) >= delta.floor_absolute - 1e-6
    # Per-pixel sampler floor maps the absolute floor into delta space.
    floor, ceiling = delta.clip_for_sampler(base, enabled=True)
    assert ceiling is None
    expected = (delta.floor_absolute - base) / delta.scale
    assert torch.allclose(floor, expected)
    assert delta.clip_for_sampler(base, enabled=False) is None
    absolute_spec = _delta_spec("absolute")
    scalar_floor, _ = absolute_spec.clip_for_sampler(base, enabled=True)
    assert scalar_floor == pytest.approx(absolute_spec.floor_absolute)


def test_delta_spec_pushforward_ceiling_bounds_runaway_extrapolation() -> None:
    """Regression test for the 2026-07-06 pilot instability: an unbounded
    pushforward reconstruction produced an extreme value that poisoned
    training and permanently destabilized the run (epoch 52). to_absolute
    with bound_ceiling=True must clamp such extrapolations."""

    delta = _delta_spec()
    base = torch.zeros(2, 1, 4, 4)
    runaway_prediction = torch.full((2, 1, 4, 4), 1e6)  # a wild single-shot guess
    unbounded = delta.to_absolute(runaway_prediction, base, clamp=True, bound_ceiling=False)
    bounded = delta.to_absolute(runaway_prediction, base, clamp=True, bound_ceiling=True)
    assert float(unbounded.max().item()) > delta.ceiling_absolute
    assert float(bounded.max().item()) == pytest.approx(delta.ceiling_absolute)
    assert delta.ceiling_absolute > delta.floor_absolute


def test_scale_for_observed_base_regimes() -> None:
    """Regime-aware target scale: observed pixels carry the temporal-delta
    scale, unobserved pixels the marginal field scale (1.0 normalized).
    Dense input reduces exactly to the scalar delta scale."""

    delta = _delta_spec()
    dense = torch.ones(2, 1, 4, 4)
    scale_dense = delta.scale_for_observed_base(dense)
    assert torch.allclose(scale_dense, torch.full_like(dense, delta.scale))
    empty = torch.zeros(2, 1, 4, 4)
    scale_empty = delta.scale_for_observed_base(empty)
    assert torch.allclose(scale_empty, torch.ones_like(empty))
    mixed = torch.zeros(1, 1, 2, 2)
    mixed[0, 0, 0, 0] = 1.0
    scale_mixed = delta.scale_for_observed_base(mixed)
    assert scale_mixed[0, 0, 0, 0] == pytest.approx(delta.scale)
    assert scale_mixed[0, 0, 1, 1] == pytest.approx(1.0)
    assert _delta_spec("absolute").scale_for_observed_base(dense) is None


def test_round_trip_with_tensor_scale() -> None:
    delta = _delta_spec()
    mask = (torch.rand(2, 1, 8, 8) > 0.5).float()
    scale = delta.scale_for_observed_base(mask)
    base = torch.randn(2, 1, 8, 8) * mask
    absolute = torch.randn(2, 1, 8, 8)
    target = delta.to_target_space(absolute, base, scale=scale)
    reconstructed = delta.to_absolute(target, base, clamp=False, scale=scale)
    assert torch.allclose(reconstructed, absolute, atol=1e-5)
    floor, _ = delta.clip_for_sampler(base, enabled=True, scale=scale)
    assert torch.allclose(floor, (delta.floor_absolute - base) / scale)


def test_sparse_target_scale_regression_m95() -> None:
    """Regression test for the 2026-07-09 sparse divergence: at masked pixels
    the base is the train-mean fill, so dividing the O(1) reconstruction
    residual by the tiny dense-calibrated delta std produced O(400) targets
    that exploded training at m50/m95 across seeds. With the per-pixel
    regime scale, sparse targets must stay O(1)."""

    c, l, size = 4, 2, 16
    torch.manual_seed(0)
    delta = _delta_spec()
    mask = (torch.rand(2, 1, size, size) > 0.95).float()  # ~m95
    # Realistic temporal structure: the next frame differs from the last
    # context frame by a delta of the calibrated std (like the real data),
    # while the field itself has unit marginal variance.
    context_true = torch.randn(2, c, size, size)
    target = context_true[:, -1:].repeat(1, l, 1, 1) + delta.scale * torch.randn(2, l, size, size)
    batch = {
        "context_water_masked": context_true * mask,
        "context_water_true": context_true,
        "sensor_mask": mask,
        "dem": torch.randn(2, 1, size, size),
        "rainfall": torch.randn(2, c + l, size, size),
        "timestamps": torch.arange(c + l, dtype=torch.float32).repeat(2, 1) * 300.0,
        "target": target,
    }
    model_batch = prepare_model_batch(batch, c, delta, wet_threshold_normalized=-0.396, change_weight=3.0)
    target = model_batch["target"]
    assert float(target.abs().max().item()) < 50.0  # old code: O(1/delta.scale) = 250+
    assert float(target.std().item()) < 10.0
    # Round trip through the stored per-pixel scale is exact.
    reconstructed = delta.to_absolute(
        target, model_batch["base"], clamp=False, scale=model_batch["target_scale"]
    )
    assert torch.allclose(reconstructed, batch["target"][:, 0:1], atol=1e-5)


def test_pushforward_loss_is_restricted_to_observed_pixels() -> None:
    """Pushforward step-2 residuals at MASKED pixels are the model's own
    reconstruction error -- not representable at the delta scale (they
    saturated the bound and spiked train_loss ~10^3 at m50/m95, 2026-07-09
    pilot 1). The pushforward batch must zero the loss weights at masked
    pixels and keep dense (mask == 1) behavior unchanged."""

    from tools.train_floodcastbench_diff_sparse_v2 import pushforward_batch

    config = _small_config()
    model = DiffSparseV2Model(config)
    c, l, size = config["dataset"]["context_length"], 2, 32
    torch.manual_seed(0)
    mask = (torch.rand(2, 1, size, size) > 0.5).float()
    batch = {
        "context_water_masked": torch.randn(2, c, size, size),
        "context_water_true": torch.randn(2, c, size, size),
        "sensor_mask": mask,
        "dem": torch.randn(2, 1, size, size),
        "rainfall": torch.randn(2, c + l, size, size),
        "timestamps": torch.arange(c + l, dtype=torch.float32).repeat(2, 1) * 300.0,
        "target": torch.randn(2, l, size, size),
    }
    delta = _delta_spec()
    model_batch = prepare_model_batch(batch, c, delta, wet_threshold_normalized=-0.396, change_weight=3.0)
    pushed = pushforward_batch(model, batch, model_batch, c, delta, -0.396, 3.0, clamp=True)
    assert pushed is not None
    weights = pushed["pixel_weights"]
    assert float(weights[mask.bool()].min().item()) >= 1.0  # observed pixels keep full weight
    assert float(weights[~mask.bool()].abs().max().item()) == 0.0  # masked pixels excluded
    assert pushed["target_scale"] is None  # step-2 regime: scalar delta scale


def test_pushforward_target_bound_clamps_synthetic_targets() -> None:
    """The pushforward step-2 target is synthetic (residual vs the model's own
    imperfect base); it must never exceed 1.5x the largest physically
    observed frame-to-frame delta in scaled units."""

    water_stats = {"mean": 0.1, "std": 0.25, "max": 10.0}
    delta_stats = {"delta_std_physical": 0.001, "delta_abs_max_physical": 0.1}
    delta = DeltaSpec("delta", water_stats, delta_stats)
    assert delta.pushforward_target_bound == pytest.approx(1.5 * 0.1 / 0.001)
    # Fixture without abs max (older stats files) -> bound disabled, no crash.
    assert _delta_spec().pushforward_target_bound is None


def test_change_weight_map_marks_state_changes_only() -> None:
    threshold = 0.0
    base = torch.tensor([[[[-1.0, 1.0], [1.0, -1.0]]]])
    target = torch.tensor([[[[1.0, 1.0], [-1.0, -1.0]]]])  # flips at (0,0) and (1,0)
    weights = change_weight_map(target, base, threshold, change_weight=3.0)
    assert weights is not None
    assert weights[0, 0, 0, 0] == pytest.approx(4.0)
    assert weights[0, 0, 0, 1] == pytest.approx(1.0)
    assert weights[0, 0, 1, 0] == pytest.approx(4.0)
    assert weights[0, 0, 1, 1] == pytest.approx(1.0)
    assert change_weight_map(target, base, threshold, change_weight=0.0) is None


def test_ema_update_math_and_swap() -> None:
    model = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    ema = ExponentialMovingAverage(model, decay=0.9)
    with torch.no_grad():
        model.weight.fill_(2.0)
    ema.update(model)
    assert torch.allclose(ema.shadow["weight"], torch.full((2, 2), 0.9 * 1.0 + 0.1 * 2.0))
    ema.swap_in(model)
    assert torch.allclose(model.weight, torch.full((2, 2), 1.1))
    ema.swap_out(model)
    assert torch.allclose(model.weight, torch.full((2, 2), 2.0))


def test_dihedral_transforms_are_consistent_and_shape_safe() -> None:
    tensor = torch.arange(2 * 3 * 4 * 4, dtype=torch.float32).reshape(2, 3, 4, 4)
    seen = set()
    for index in range(8):
        transformed = apply_dihedral(tensor, index)
        assert transformed.shape == tensor.shape
        assert float(transformed.sum().item()) == pytest.approx(float(tensor.sum().item()))
        seen.add(tuple(transformed.flatten().tolist()))
    assert len(seen) == 8  # all 8 dihedral transforms are distinct
    assert torch.equal(apply_dihedral(tensor, 0), tensor)


def test_normalized_zero_depth() -> None:
    water_stats = {"mean": 0.1, "std": 0.25}
    assert normalized_zero_depth(water_stats) == pytest.approx(-0.4)


def test_multi_horizon_path_accumulator_per_step_and_pooled() -> None:
    accumulator = MultiHorizonPathAccumulator(prediction_length=2, gammas=[0.5])
    initial = torch.zeros(4, 4)
    initial[0, 0] = 1.0  # initially flooded corner
    target = torch.zeros(2, 4, 4)
    target[0, 0, 1] = 1.0                # step 1: one new pixel
    target[1, 0, 1] = 1.0
    target[1, 1, 1] = 1.0                # step 2: another new pixel
    pred = torch.zeros(2, 4, 4)
    pred[0, 0, 1] = 1.0                  # step 1: correct
    pred[1, 0, 1] = 1.0
    pred[1, 2, 2] = 1.0                  # step 2: wrong pixel
    accumulator.update(pred, target, initial)

    rows = accumulator.per_step_metrics()
    assert len(rows) == 2
    key = "path_iou_gamma_0_5"
    prop_key = "propagation_path_iou_gamma_0_5"
    assert rows[0][key] == pytest.approx(1.0)       # cumulative: {0,1} vs {0,1}
    assert rows[0][prop_key] == pytest.approx(1.0)  # new at step 1: exact match
    assert rows[1][key] == pytest.approx(1.0 / 3.0) # cumulative: {(0,1),(2,2)} vs {(0,1),(1,1)}
    assert rows[1][prop_key] == pytest.approx(0.0)  # new at step 2: disjoint

    pooled = accumulator.pooled_propagation()
    assert pooled["propagation_path_iou_gamma_0_5"] == pytest.approx(1.0 / 3.0)
    assert pooled["final_path_iou_gamma_0_5"] == pytest.approx(1.0 / 3.0)


def test_multi_horizon_path_accumulator_rejects_bad_shapes() -> None:
    accumulator = MultiHorizonPathAccumulator(prediction_length=2, gammas=[0.5])
    with pytest.raises(ValueError):
        accumulator.update(torch.zeros(3, 4, 4), torch.zeros(3, 4, 4), torch.zeros(4, 4))
    with pytest.raises(ValueError):
        accumulator.update(torch.zeros(2, 4, 4), torch.zeros(2, 4, 4), torch.zeros(3, 3))


def test_temporal_down_block_output_size_matches_reference_arithmetic() -> None:
    # patch 64, 3 blocks: 64->60->30, 30->26->13, 13->9->4 (reference 16-dim token at 4x4)
    assert _temporal_down_block_output_size(64, 3) == 4
    # patch 32, 2 blocks: 32->28->14, 14->10->5
    assert _temporal_down_block_output_size(32, 2) == 5
    with pytest.raises(ValueError):
        _temporal_down_block_output_size(8, 3)


def _real_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def test_real_config_builds_model_and_runs_forward() -> None:
    config = _real_config()
    model = DiffSparseV2Model(config)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    assert parameters > 1_000_000
    c = config["dataset"]["context_length"]
    size = config["dataset"]["patch_size"]
    batch = {
        "context_water_masked": torch.randn(1, c, size, size),
        "sensor_mask": torch.ones(1, 1, size, size),
        "dem": torch.randn(1, 1, size, size),
        "rainfall_context": torch.randn(1, c, size, size),
        "rainfall_target": torch.randn(1, 1, size, size),
        "timestamps_context": torch.arange(c, dtype=torch.float32).repeat(1, 1) * 300.0,
        "target": torch.randn(1, 1, size, size),
    }
    loss, diagnostics = model.training_step_loss(batch)
    assert torch.isfinite(loss)
    assert diagnostics["pred_finite"] == 1.0


def test_strided_frame_view_stride_1_is_identity() -> None:
    # WP12 dose-response (paper master plan): frame_stride=1 must be
    # byte-identical to V1's original contiguous slicing -- this is the
    # default for every existing config, so a regression here would be
    # silent everywhere.
    frames = list(range(20))
    view = _StridedFrameView(frames, stride=1)
    assert view[3:8] == frames[3:8]
    assert view[0:20] == frames[0:20]


def test_strided_frame_view_spaces_frames_by_stride() -> None:
    # WP12's crossed {Delta t x target} design (Delta t in {300,900,1800,7200}s
    # at native 300s cadence -> stride in {1,3,6,24}): window_length frames
    # spaced `stride` apart, starting at the requested window start.
    frames = list(range(100))
    view = _StridedFrameView(frames, stride=3)
    # window_length=6 (e.g. context 4 + prediction 2) starting at raw index 10:
    # expect frames[10], frames[13], ..., frames[25] (6 frames, spaced by 3).
    assert view[10:16] == [10, 13, 16, 19, 22, 25]
    assert len(view[10:16]) == 6


def test_strided_frame_view_rejects_non_contiguous_slices() -> None:
    # V1's frozen __getitem__ only ever requests a step=1 slice
    # (frames[start:start+window_length]) -- any other access pattern
    # would mean V1's indexing contract changed underneath this proxy,
    # and silently returning wrong frames would be far worse than failing
    # loudly here.
    view = _StridedFrameView(list(range(20)), stride=3)
    with pytest.raises(TypeError):
        view[0:10:2]
    with pytest.raises(TypeError):
        view[5]
