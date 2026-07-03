# DIFF-SPARSE v1 Evaluation Notes

This note tracks the FloodCastBench high-fidelity 60m DIFF-SPARSE v1 adaptation.
It is an engineering evaluation record, not an official benchmark report.

## Scientific Scope

- Status: DIFF-SPARSE v1 FloodCastBench adaptation sanity evaluation.
- Does not claim official FloodCastBench benchmark performance.
- Does not claim official DIFF-SPARSE TideWatch reproduction.
- Does not claim superiority over persistence or FNO+ unless the recorded numbers show it.
- Does not claim uncertainty calibration.

## Baseline 40-Epoch Checkpoint

- Run: `/home/wissam/utem-workspace/experiments/FloodCastBench/03-07-2026_15-51-43_fcb_diff_sparse_v1_highfid_60m`
- Checkpoint: `/home/wissam/utem-workspace/checkpoints/FloodCastBench/03-07-2026_15-51-43_fcb_diff_sparse_v1_highfid_60m/checkpoint_best.pth`
- Split protocol: context h1..h12, autoregressive prediction h13..h20.
- Missing rate: 0.0.
- Existing evaluation tiling: historical non-overlap/minimal-overlap path before seam blending.

## Overall Metrics Already Confirmed

| Split | Model RMSE norm | Persistence RMSE norm | Model MAE norm | Persistence MAE norm | Model NRMSE | Persistence NRMSE | Model NACRPS | Persistence NACRPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| val | 0.089557 | 0.004677 | 0.050236 | 0.001439 | 0.001746 | 0.000091 | 0.041067 | 0.001177 |
| test | 0.093462 | 0.002534 | 0.051656 | 0.000788 | 0.001826 | 0.000049 | 0.042030 | 0.000642 |

## Per-Step RMSE, Physical Meters, Existing h13..h20 Eval

| Horizon | Val model | Val persistence | Test model | Test persistence |
|---|---:|---:|---:|---:|
| h13 | 0.019131 | 0.000272 | 0.020162 | 0.000147 |
| h14 | 0.020602 | 0.000543 | 0.021511 | 0.000293 |
| h15 | 0.022332 | 0.000813 | 0.023185 | 0.000439 |
| h16 | 0.024067 | 0.001082 | 0.024898 | 0.000585 |
| h17 | 0.025836 | 0.001351 | 0.026711 | 0.000731 |
| h18 | 0.028380 | 0.001619 | 0.029445 | 0.000877 |
| h19 | 0.031121 | 0.001886 | 0.032548 | 0.001022 |
| h20 | 0.033596 | 0.002152 | 0.035448 | 0.001167 |

## Current Interpretation

Persistence remains much stronger through h20, but the ratio shrinks with
horizon. The current evidence says the model learned spatial structure, yet the
short-lead persistence baseline is unusually strong at 5-minute frame spacing.

## Pending Seam-Blended Eval

The evaluator now supports:

- overlapping 64x64 tiles with default stride 48;
- cell-centered Hann distance-to-center blending;
- persisted `tile_stride`, `tile_blending`, and `persistence_mode`.

Acceptance check after the background eval completes:

| Split | Eval directory | Tile stride | Overall model RMSE norm | Overall persistence RMSE norm | Visual seam status |
|---|---|---:|---:|---:|---|
| val | `/home/wissam/utem-workspace/experiments/FloodCastBench/03-07-2026_15-51-43_fcb_diff_sparse_v1_highfid_60m/eval_rollout_val_03-07-2026_16-44-30` | 48 | 0.084263 | 0.004677 | much fainter grid pattern in step08 abs-error map |

Before/after against the previous val rollout:

| Eval | Tile stride | Tile blending | Overall model RMSE norm | Overall model MAE norm |
|---|---:|---|---:|---:|
| `eval_rollout_val_03-07-2026_16-06-02` | not recorded | uniform/minimal overlap | 0.089557 | 0.050236 |
| `eval_rollout_val_03-07-2026_16-44-30` | 48 | cell-centered Hann distance-to-center | 0.084263 | 0.047464 |

Visual check: the old step08 abs-error PNG showed visible 64x64 tile-boundary
grid lines, especially in the lower-right quadrant. The new overlapping/Hann
eval makes that grid much fainter while preserving flood-boundary error
structure. This is an eval-time artifact fix, not evidence of improved
training quality.

## Pending Extended-Horizon Crossover Check

Eval-only config:
`configs/floodcastbench_diff_sparse_v1_highfid_60m_pred20.yaml`

This keeps context_length=12 and extends prediction_length=20, giving h13..h32
when enough frames fit within the fixed split boundaries. No retraining is
implied by this config.

| Split | Eval directory | Windows evaluated | First horizon | Last horizon | Crossover |
|---|---|---:|---|---|---|
| val | pending | pending | h13 | h32 | pending |
| test | pending | pending | h13 | h32 | pending |
