from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from datasets.floodcastbench_diff_sparse_v2_dataset import (  # noqa: E402
    DEFAULT_MANNING_LOOKUP,
    generate_cluster_mask,
    generate_gauge_mask,
)
from models.diff_sparse_v2 import DiffSparseV2Model, SpatialContextEncoder, TemporalContextEncoder  # noqa: E402
from tests.test_diff_sparse_v2_smoke import _fake_model_batch, _small_config  # noqa: E402
from tools.evaluate_floodcastbench_diff_sparse_v2 import CalibrationAccumulator  # noqa: E402


# ------------------------------------------------------------------ WP3 -----


def test_calibration_requires_two_members() -> None:
    with pytest.raises(ValueError):
        CalibrationAccumulator(2, gammas=[0.001], num_members=1)


def test_calibration_reliability_hand_checked() -> None:
    """M=2 members, 1 step, 2x2 grid with hand-computable wet counts."""

    acc = CalibrationAccumulator(1, gammas=[0.5], num_members=2)
    scenarios = torch.tensor([
        [[[1.0, 0.0], [1.0, 0.0]]],   # member 1
        [[[1.0, 0.0], [0.0, 0.0]]],   # member 2
    ])  # wet counts at gamma=0.5: [[2, 0], [1, 0]]
    target = torch.tensor([[[1.0, 0.0], [1.0, 1.0]]])  # wet: TL, BL, BR
    acc.update(scenarios, target, window_index=0)
    entry = acc.summary()["reliability"]["gamma_0.5"]
    # k=0: two pixels (TR dry-obs, BR wet-obs) -> observed freq 0.5
    # k=1: one pixel (BL), observed wet -> 1.0 ; k=2: one pixel (TL), wet -> 1.0
    assert entry["pooled_count"] == [2.0, 1.0, 1.0]
    assert entry["pooled_observed_frequency"] == pytest.approx([0.5, 1.0, 1.0])


def test_calibration_rank_histogram_and_coverage_shapes() -> None:
    torch.manual_seed(0)
    members, length = 8, 3
    acc = CalibrationAccumulator(length, gammas=[0.001, 0.01], num_members=members)
    # Target drawn from the same distribution as the members -> ranks roughly
    # uniform, coverage near nominal (loose bounds; statistical smoke).
    scenarios = torch.rand(members, length, 24, 24)
    target = torch.rand(length, 24, 24)
    acc.update(scenarios, target, window_index=0)
    summary = acc.summary()
    freq = summary["rank_histogram"]["frequency"]
    assert len(freq) == members + 1
    assert sum(freq) == pytest.approx(1.0)
    assert max(freq) < 3.0 / (members + 1)  # no catastrophic spike
    for name in ("50", "90"):
        entry = summary["coverage"][name]
        # A perfectly calibrated ensemble matches the FINITE-ENSEMBLE nominal
        # ((hi-lo)*(M-1)/(M+1)), not the naive nominal -- the small-M bias the
        # accumulator documents. abs tolerance: statistical smoke on 24x24x3.
        assert entry["pooled"] == pytest.approx(entry["nominal_finite_ensemble"], abs=0.06)
        assert entry["nominal_finite_ensemble"] < entry["nominal"]
    spread_n = sum(b["count"] for b in summary["spread_skill"])
    assert spread_n == pytest.approx(length * 24 * 24)


# ------------------------------------------------------------------ WP7 -----


def test_gauge_mask_budget_and_occupancy_bias() -> None:
    torch.manual_seed(0)
    height = width = 40
    occupancy = torch.zeros(height, width)
    occupancy[:, :10] = 1.0  # "river" on the left quarter
    generator = torch.Generator().manual_seed(7)
    mask = generate_gauge_mask(height, width, missing_rate=0.9, occupancy=occupancy, generator=generator)
    sensor_count = int(mask.sum())
    assert sensor_count == round(0.1 * height * width)
    in_river = float(mask[0, :, :10].sum())
    assert in_river / sensor_count > 0.8  # strongly biased toward the wet area


def test_cluster_mask_budget_and_compactness() -> None:
    height = width = 40
    generator = torch.Generator().manual_seed(7)
    mask = generate_cluster_mask(height, width, missing_rate=0.9, generator=generator)
    sensor_count = int(mask.sum())
    assert sensor_count == round(0.1 * height * width)
    # Compactness: mean nearest-neighbor distance among sensors is far below
    # the i.i.d. expectation for the same budget.
    points = mask[0].nonzero(as_tuple=False).float()
    distance = torch.cdist(points, points)
    distance.fill_diagonal_(float("inf"))
    mean_nearest = distance.min(dim=1).values.mean()
    assert float(mean_nearest) < 1.8  # clustered pixels are mostly adjacent


def test_mask_generators_dense_and_empty_edges() -> None:
    occupancy = torch.rand(8, 8)
    assert generate_gauge_mask(8, 8, 0.0, occupancy).sum() == 64
    assert generate_cluster_mask(8, 8, 0.0).sum() == 64
    assert generate_cluster_mask(8, 8, 1.0).sum() == 0


# ------------------------------------------------------------------ WP5 -----


def _manning_config() -> dict:
    config = _small_config()
    config["dataset"]["include_manning"] = True
    return config


def test_manning_channel_grows_encoder_inputs_and_forward_works() -> None:
    base = _small_config()
    with_manning = _manning_config()
    t_base = TemporalContextEncoder(base)
    t_manning = TemporalContextEncoder(with_manning)
    assert t_manning.blocks[0].conv1.in_channels == t_base.blocks[0].conv1.in_channels + 1
    s_base = SpatialContextEncoder(base)
    s_manning = SpatialContextEncoder(with_manning)
    assert s_manning.encoder[0].in_channels == s_base.encoder[0].in_channels + 1

    model = DiffSparseV2Model(with_manning)
    batch = _fake_model_batch(with_manning)
    batch["manning"] = torch.randn(2, 1, 32, 32)
    loss, diagnostics = model.training_step_loss(batch)
    assert torch.isfinite(loss) and diagnostics["pred_finite"] == 1.0


def test_manning_missing_raises_when_enabled() -> None:
    model = DiffSparseV2Model(_manning_config())
    batch = _fake_model_batch(_manning_config())
    with pytest.raises(ValueError, match="manning"):
        model.encode_context(batch)


def test_default_manning_lookup_values_are_physical() -> None:
    for code, value in DEFAULT_MANNING_LOOKUP.items():
        assert 0.01 <= value <= 0.2, (code, value)
