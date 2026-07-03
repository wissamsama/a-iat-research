# DIFF-SPARSE v1 for FloodCastBench — Design Notes

Adaptation of DIFF-SPARSE (Islam et al., "Towards High Resolution Probabilistic
Coastal Inundation Forecasting from Sparse Observations", AAAI 2026,
arXiv:2505.05381) to FloodCastBench high-fidelity 60m Australia data.

## Files

| Component | Path |
|---|---|
| Dataset | `datasets/floodcastbench_diff_sparse_v1_dataset.py` |
| Model | `models/diff_sparse_v1.py` |
| Training | `tools/train_floodcastbench_diff_sparse_v1.py` |
| Evaluation | `tools/evaluate_floodcastbench_diff_sparse_v1.py` |
| Config | `configs/floodcastbench_diff_sparse_v1_highfid_60m.yaml` |
| Tests | `tests/test_diff_sparse_v1_smoke.py` |
| Normalization stats | `outputs/floodcastbench_normalization/diff_sparse_v1_water_dem_rainfall_train_stats.json` |

## Paper → FloodCastBench mapping

| Paper | This adaptation |
|---|---|
| TideWatch hourly inundation, 30m, patches D×D | FloodCastBench Australia 60m, 300s frames, 536×536 field, random 64×64 patches for training, tiled full field for evaluation |
| Sensor mask M_k, sparsity 0/50/95% | `masking.missing_rate` (config or `--missing-rate`), static per-sample mask, exact sensor count |
| Noise masking (Algorithm 1, lines 3-5) | `masking.mask_mode: noise` (or `zeros` = paper's ablation baseline) |
| Elevation s_k | DEM raster, standardized separately |
| Temporal covariates z_t (hour-of-day, day-of-month, sinusoidal) | hour-of-day sinusoids (3 frequencies) + event-time fraction from frame timestamps |
| — (tidal domain has no rain) | **rainfall forcing added as context channels** (dense, exogenous) |
| Context 12, train prediction 1, test prediction 12 | context 12, prediction 8 (12+8 = 20-frame windows keep the canonical 116/14/14 split frame ranges) |
| Diffusion N=20, β ∈ [1e-4, **1.0**], x0-prediction (Table 2) | identical |
| Conv context blocks [16,32,64], embedding 32, UNet [16,32,32,64], 2 ResNet layers, 2 cross-attn blocks, GN 8 (Table 3, D=64) | identical |
| LR 1e-3, ReduceLROnPlateau (patience 3, factor 0.5), 40 epochs, batch 32 | identical |
| 10 fixed eval masks, round-robin | `eval_mask_bank_size: 10`, seeded |
| Scenarios: 2 val / 8 test | identical |
| NRMSE (eq. 15), NACRPS (eq. 16) | identical, plus normalized/physical RMSE/MAE and dense-persistence comparison |

## The β_end = 1.0 fix

The earlier dense sanity baseline (`models/diff_sparse.py`) used β ∈ [1e-4, 0.02]
over 20 steps — the standard **1000-step** image schedule truncated to 20 steps.
That leaves ᾱ_T ≈ 0.82 (terminal SNR ≈ 4.5): the noisiest training input still
contained ~82% of the target's variance, so (a) training leaked the target
through `x_noisy` and rewarded a light-denoise shortcut instead of conditional
generation, and (b) reverse sampling initialized from N(0, I) was far outside
the training distribution. That is the mechanism behind the h100 reverse-sampling
failure (teacher-forced RMSE 0.16 vs reverse-sampled 0.85).

The paper's Table 2 uses β_max = 1.0: ᾱ_T = 0 exactly, the terminal forward step
is pure noise, and the first reverse step has posterior coefficient 0 on x_t —
the initial noise is fully discarded. `tests/test_diff_sparse_v1_smoke.py`
contains regression tests for both properties, plus an oracle-denoiser test
proving the sampler returns x0 exactly when the network is perfect.

## Deviations from the paper (all flagged in the config)

1. **Rainfall context channels.** FloodCastBench floods are rain-driven; rainfall
   frames (dense, exogenous, resampled to the water grid) are concatenated to
   the context conv input, and future rainfall is fed to the sliding context
   during rollout. Disable with `dataset.include_rainfall: false`.
2. **`conditioning: cross_attention_concat` (default).** The paper conditions the
   UNet through cross-attention only. Pure cross-attention has no intrinsic
   spatial alignment between context pixels and output pixels, so this
   implementation adds fixed 2D sine-cosine positional encodings on both sides
   of the attention *and*, by default, concatenates the context embedding to
   the UNet input. `conditioning: cross_attention` gives the paper-faithful
   attention-only variant.
3. **Prediction length 8 (not 12).** Keeps windows at 20 frames so split frame
   ranges match the canonical FNO+/persistence splits exactly
   (train frames [0, 2320), val [2320, 2600), test [2600, 2881)).
4. **Rollout re-masking (`evaluation.rollout_remask: true`).** The paper is
   ambiguous about whether sampled frames appended to the context are re-masked.
   We re-mask with the same static sensor mask so the inference-time context
   distribution matches training. Set false to append dense samples.
5. **Single shared water normalization** for context and targets (train-only),
   removing the initial/target statistic mismatch of the official-v1 pipeline
   that previously forced "retargeted persistence".
6. **Deterministic validation.** `training.val_seed` fixes masks/noise/timesteps
   during validation (RNG state saved/restored), so `val_loss` — the checkpoint
   selection metric — is comparable across epochs.
7. **Training windows slide with stride 1** inside the train frame range
   (2301 windows/epoch) instead of the 116 non-overlapping benchmark windows;
   val/test keep the canonical non-overlapping windows.

## Evaluation protocol

Per window: the full 536×536 field is tiled into 64×64 patches (overlap on the
last row/column, overlapping predictions averaged), each tile is rolled out
autoregressively for `prediction_length` steps × M scenarios (Algorithm 2:
every frame sampled from pure noise conditioned on the current masked context).
Metrics: NRMSE (eq. 15, denominator = max−min of observations over the evaluated
set), NACRPS (eq. 16, empirical CRPS), normalized and physical (meters)
RMSE/MAE, per-step and overall, against dense persistence (last true context
frame — an oracle baseline under sparsity, deliberately conservative).

## Commands

```bash
# stats are precomputed once; omit --stats-json to recompute (~25 s)
python tools/train_floodcastbench_diff_sparse_v1.py \
  --config configs/floodcastbench_diff_sparse_v1_highfid_60m.yaml \
  --stats-json outputs/floodcastbench_normalization/diff_sparse_v1_water_dem_rainfall_train_stats.json

# sparsity is one flag away (paper levels: 0.0 / 0.5 / 0.95)
python tools/train_floodcastbench_diff_sparse_v1.py --config ... --missing-rate 0.95

python tools/evaluate_floodcastbench_diff_sparse_v1.py \
  --config configs/floodcastbench_diff_sparse_v1_highfid_60m.yaml \
  --checkpoint <run>/checkpoint_best.pth \
  --split test --save-maps            # cross-sparsity eval: add --missing-rate

pytest tests/test_diff_sparse_v1_smoke.py -q
```

## Status and non-claims

First functional version: pipeline verified end-to-end (9/9 smoke tests, dry
run, bounded 2-epoch pilot, 2-window rollout eval). Results do not claim
official FloodCastBench benchmark performance, official DIFF-SPARSE TideWatch
reproduction, superiority over persistence/FNO+ until a full trained evaluation
says so, or uncertainty calibration.
