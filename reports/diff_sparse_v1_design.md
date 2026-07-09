# DIFF-SPARSE v1 for FloodCastBench — Design Notes

Adaptation of DIFF-SPARSE (Islam et al., "Towards High Resolution Probabilistic
Coastal Inundation Forecasting from Sparse Observations", AAAI 2026,
arXiv:2505.05381) to FloodCastBench high-fidelity 60m Australia data.

## 2026-07-05: verified against the official reference implementation

Everything below was cross-checked against the paper's own released code
(`github.com/KAI10/Diff-Sparse`, cloned and read directly — `diffusion.py`,
`hidden_state_net.py`, `patch_embedding.py`, `lightning_diffusion.py`,
`lightning_training.py`, `training_config.py`, `TidewatchDataset.py`,
`TidewatchPatchDataset.py`, `datastore.py`, `consistency_loss.py`, `utils.py`,
`generate_test_masks.py`), not just the paper prose. Three architectural facts
were previously wrong or unverified in this adaptation and are now fixed
exactly (see "Deviations" below, items 2/3/4). This reading also resolved one
previously-ambiguous design choice (item 6, rollout re-masking) with direct
evidence from the reference's rollout code.

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

| Paper / reference implementation | This adaptation |
|---|---|
| TideWatch hourly inundation, 30m, patches D×D | FloodCastBench Australia 60m, 300s frames, 536×536 field, random 64×64 patches for training, tiled full field for evaluation |
| Sensor mask M_k, sparsity 0/50/95% | `masking.missing_rate` (config or `--missing-rate`), static per-sample mask, exact sensor count (reference uses per-cell Bernoulli(p), giving the same expected sparsity with nonzero count variance — immaterial, not changed) |
| Noise masking (Algorithm 1, lines 3-5) | `masking.mask_mode: noise` (or `zeros` = paper's ablation baseline) |
| Elevation s_k, static per-timestep channel | DEM raster, standardized separately, broadcast across context_length exactly like the reference |
| Temporal covariates z_t: reference uses raw (no MLP) day-of-month-sin/cos + hour-sin/cos (4 dims), concatenated per-timestep before the shared token projection | hour-of-day sinusoids (3 frequencies) + event-time fraction (7 dims raw, same "concatenate per-timestep, no separate MLP" structure); day-of-month has no meaningful cycle over FloodCastBench's ~10-day event, so event-time fraction substitutes for it — the one still-open, low-priority difference |
| — (tidal domain has no rain) | **rainfall added as a 4th per-timestep channel** (varies with t, like water) |
| Context 12, train prediction 1, test/val prediction 12 | identical (24-frame windows; 2297 train / 13 val / 13 test eligible windows inside the unchanged canonical split frame ranges) |
| Diffusion N=20, β ∈ [1e-4, **1.0**], x0-prediction (Table 2) | identical |
| **Reverse-step noise: reference uses raw β_t ("Option 1"), not β̃_t ("Option 2", commented out)** | identical (fixed 2026-07-05, was β̃_t before) |
| **Conditioning: temporal token sequence** (one heavily-pooled per-timestep spatial summary, `encoder_hidden_states`) via `hidden_state_net.py`'s unpadded Conv3d+AvgPool3d stack | identical — `TemporalContextEncoder` in `models/diff_sparse_v1.py` |
| **UNet: `diffusers.UNet2DConditionModel`**, `block_out_channels=(16,32,32,64)`, `layers_per_block=2`, `cross_attention_dim=32`, `norm_num_groups=16`, cross-attention at the **2 middle levels** (`DownBlock2D, CrossAttnDownBlock2D, CrossAttnDownBlock2D, DownBlock2D`), `only_cross_attention=False` (self- + cross-attention combined) | identical — same library, same block config, same placement |
| LR 1e-3, ReduceLROnPlateau (patience 3, factor 0.5), 40 epochs, batch 32 | LR/factor/batch identical; epochs and patience deviate, see items 8/9 below |
| 10 fixed eval masks, round-robin | `eval_mask_bank_size: 10`, seeded |
| Scenarios: 2 val / 8 test | identical |
| Rollout: model's own dense prediction re-enters context **unmasked** (`generate_multistep_scenarios` never re-masks) | `rollout_remask: false` (fixed 2026-07-05, was `true`) |
| NRMSE (eq. 15), NACRPS (eq. 16) | identical, plus normalized/physical RMSE/MAE and dense-persistence comparison |
| Validation metric driving the LR scheduler: reference uses full ancestral-sampling `val_masked_nrmse` (expensive — real DDPM generation every val epoch) | cheaper one-step denoising-MSE proxy (`val_loss`); see item 9 |

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

## The 2026-07-05 architecture rewrite

A first "paper-faithful" attempt (context/prediction=12, hand-rolled pixel-
aligned cross-attention with fixed 2D positional encodings, no concat pathway)
plateaued completely — `train_loss` stuck at ~0.4 for all 161 epochs instead of
dropping to ~0.0005 like the earlier (non-paper) concat variant. Two
hypotheses were tested (attention placement: deep vs shallow UNet levels — no
effect; `lr_patience`: raising 3→15 did fix the plateau in isolation and on a
60-epoch real-data pilot). Before committing that fix to a full retrain, the
official reference code was cloned and read line by line, which surfaced a
much larger problem: **the hand-rolled architecture wasn't a minor variant of
the paper's mechanism, it was a different mechanism.**

Confirmed by reading `hidden_state_net.py` and `lightning_training.py`
directly:

1. **Conditioning is temporal, not spatial.** The reference's `HiddenStateNet`
   reduces each context frame to a small spatial summary (3 unpadded
   Conv3d+AvgPool3d blocks — kernel `(1,3,3)`, no padding, `AvgPool3d(1,2,2)`),
   producing one token per context timestep (12 tokens total), each a flattened
   `c*h'*w'`-dim vector (exactly 16-dim at patch_size=64, per-timestep raw
   covariates concatenated before a single shared linear projection to
   `context_embedding_dim`). This sequence is fed as `encoder_hidden_states` to
   a standard diffusers cross-attention UNet — mechanically identical to how
   Stable Diffusion conditions image generation on a sequence of text tokens.
   It is **not** a pixel-aligned spatial map; there is no per-pixel alignment
   between context and output pixels anywhere in the reference's attention
   mechanism. `TemporalContextEncoder` in `models/diff_sparse_v1.py` now
   replicates this exactly (including the unpadded shrinkage arithmetic,
   verified to produce the same 16-dim token at patch_size=64).
2. **The UNet is `diffusers.UNet2DConditionModel`**, not a hand-rolled UNet:
   `block_out_channels=(16,32,32,64)`, `layers_per_block=2`,
   `cross_attention_dim=32`, `norm_num_groups=16`,
   `down_block_types=(DownBlock2D, CrossAttnDownBlock2D, CrossAttnDownBlock2D,
   DownBlock2D)` (cross-attention at the **2 middle** levels — not the 2
   deepest or 2 shallowest, both of which were tried and both plateaued
   identically under the old hand-rolled design), `only_cross_attention=False`
   (diffusers' default — each cross-attention block also includes self-
   attention over the UNet's own spatial features, which the old
   cross-attention-only design never had).
3. **Reverse-step noise uses raw β_t** (`diffusion.py`'s active "Option 1"),
   not the tighter β̃_t posterior variance (its "Option 2", present only as a
   commented-out alternative in the reference). This adaptation had
   implemented β̃_t; now uses raw β_t, matching the reference exactly.

`models/diff_sparse_v1.py` was rewritten around `diffusers.UNet2DConditionModel`
(new dependency) and a new `TemporalContextEncoder`, replacing the previous
`ContextEncoder`/`CrossAttentionBlock`/hand-rolled `ResBlock` UNet entirely —
no ablation path kept, since the previous mechanism was not a documented paper
variant, it was simply not what the paper's authors built. `lr_patience`
reverts to the paper's own **3** under this architecture (to be re-verified
empirically — see the training protocol report for pilot results); the
earlier 15 was a compensating fix for the old, structurally different
attention mechanism and may no longer be necessary.

## Deviations from the paper (all flagged in the config)

1. **Rainfall context channel.** FloodCastBench floods are rain-driven; a
   rainfall frame (dense, exogenous, resampled to the water grid) is added as
   a 4th per-timestep channel alongside water/DEM/sensor-mask, and future
   rainfall is fed to the sliding context during rollout. Disable with
   `dataset.include_rainfall: false`.
2. **(Resolved 2026-07-05.)** Cross-attention conditioning is now a temporal
   token sequence via `TemporalContextEncoder`, matching
   `hidden_state_net.py` exactly (see above). Previously this adaptation used
   a hand-rolled pixel-aligned spatial cross-attention with fixed 2D
   positional encodings — a reasonable-looking but materially different
   mechanism, not a paper variant.
3. **(Resolved 2026-07-05.)** The UNet is now `diffusers.UNet2DConditionModel`
   configured exactly like the reference (see above), replacing a hand-rolled
   UNet that had the right channel/layer counts (Table 3) but the wrong
   attention placement, no self-attention, and a different conditioning
   mechanism.
4. **(Resolved 2026-07-05.)** Reverse-diffusion noise variance is now raw β_t,
   matching the reference's active code path (previously β̃_t).
5. **Prediction length 12** (paper Table 2, "test prediction 12"). Matches the
   reference exactly; only eligible-window counts differ from the earlier
   (pre-2026-07-05) prediction-8 setup (2297/13/13 vs 2301/14/14) since split
   frame ranges are unchanged.
6. **(Resolved 2026-07-05, was ambiguous.)** `evaluation.rollout_remask: false`
   — the reference's `generate_multistep_scenarios` never re-masks: the
   model's own dense prediction re-enters the sliding context directly, with
   the (unchanged) elevation/sensor-mask channels re-attached. This repo
   previously defaulted to re-masking (re-injecting noise at non-sensor
   positions of the model's own prediction) as an interpretation under paper
   ambiguity; direct reading of the rollout code resolves this definitively.
7. **Single shared water normalization** for context and targets (train-only),
   removing the initial/target statistic mismatch of the official-v1 pipeline
   that previously forced "retargeted persistence". The reference computes an
   equivalent train-only mean/std but does not have this specific
   initial/target mismatch problem to solve (it has no separate official-v1
   pipeline).
8. **Deterministic validation.** `training.val_seed` fixes masks/noise/timesteps
   during validation (RNG state saved/restored), so `val_loss` — the checkpoint
   selection metric — is comparable across epochs.
9. **Validation metric is a cheap proxy, not the reference's real generative
   metric.** The reference's `ReduceLROnPlateau` watches `val_masked_nrmse`,
   computed via full ancestral DDPM sampling (`generate_multistep_scenarios`)
   every validation epoch — expensive, but a direct measurement of generative
   quality. This adaptation uses a one-step denoising-MSE proxy (`val_loss`)
   instead, to keep the already-inflated 161-epoch budget (see item 11)
   affordable. This proxy is plausibly noisier/less monotone than the
   reference's real metric, which may be why `lr_patience=3` (the paper's own
   value) needed raising to 15 under the old (structurally different)
   attention mechanism — to be re-tested under the current, corrected
   architecture before assuming this deviation is still necessary.
10. **Training windows slide with stride 1** inside the train frame range
    (2297 windows/epoch at the 24-frame window length) instead of the 116
    non-overlapping benchmark windows; val/test keep the canonical stride-20
    start positions (13 eligible windows each at window length 24).
11. **`training.epochs: 161` (paper: 40)** — the paper's 40 epochs is tied to
    TideWatch's dataset size; 161 gives gradient-step parity with FNO+
    official-v1 (see `reports/diff_sparse_v1_fno_plus_training_protocol.md`).
    `checkpoint_best.pth` is selected by deterministic val_loss, so epochs past
    convergence do not change the selected model.
12. **`training.grad_clip_norm: 1.0`** — not in the paper (Algorithm 1 takes a
    plain gradient step); kept as a conservative stability guard.
13. **Temporal covariate features**: hour-of-day sinusoids at 3 frequencies +
    event-time fraction (7 raw dims), vs. the reference's day-of-month-sin/cos
    + hour-sin/cos (4 raw dims). FloodCastBench's Australia event spans ~10
    days, too short for a meaningful day-of-month cycle; event-time fraction
    is used as a domain-appropriate substitute. Both schemes concatenate raw
    (non-MLP) per-timestep features before the shared token projection —
    structurally identical, only the specific feature set differs. Lowest-
    priority remaining difference; not changed.

## Evaluation protocol

Per window: the full 536×536 field is tiled into 64×64 patches (overlap on the
last row/column, overlapping predictions averaged), each tile is rolled out
autoregressively for `prediction_length` steps × M scenarios (Algorithm 2:
every frame sampled from pure noise conditioned on the current masked context;
the model's own prediction re-enters context unmasked, per item 6 above).
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

Results do not claim official FloodCastBench benchmark performance, official
DIFF-SPARSE TideWatch reproduction, superiority over persistence/FNO+ until a
full trained evaluation says so, or uncertainty calibration.
