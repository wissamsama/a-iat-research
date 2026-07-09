# DIFF-SPARSE V2 — Design Notes

Performance-optimized evolution of the reference-faithful V1
(`reports/diff_sparse_v1_design.md`), targeting maximum benchmark results with
**propagation path IoU** (newly inundated pixels, per rollout step) as the
headline metric. V1's mandate was fidelity to Islam et al. (AAAI 2026) and its
released code; V2's mandate is performance: the DIFF-SPARSE identity is kept
(masked conditional x0-DDPM, terminal SNR 0, noise masking, temporal-token
cross-attention, one-step training + autoregressive rollout) and everything
else — context length, diffusion steps, target space, conditioning pathways,
capacity, training procedure, decision rules — is treated as a performance
lever.

Prepared 2026-07-05 while the V1 full retrain queue was running; V2 lives in
entirely separate files and **shares no V1 training/eval code path** (the V1
dataset class is subclassed read-only).

## Files

| Component | Path |
|---|---|
| Model | `models/diff_sparse_v2.py` |
| Dataset | `datasets/floodcastbench_diff_sparse_v2_dataset.py` (subclass of the frozen V1 dataset) |
| Training | `tools/train_floodcastbench_diff_sparse_v2.py` |
| Evaluation | `tools/evaluate_floodcastbench_diff_sparse_v2.py` |
| Delta stats | `tools/compute_floodcastbench_diff_sparse_v2_delta_stats.py` → `outputs/floodcastbench_normalization/diff_sparse_v2_delta_stats.json` |
| Config | `configs/floodcastbench_diff_sparse_v2_highfid_60m.yaml` |
| Tests | `tests/test_diff_sparse_v2_smoke.py` (18 tests, CPU) |

## What stays (the DIFF-SPARSE identity)

x0-parameterized conditional DDPM, N=20 steps, linear beta [1e-4, 1.0]
(terminal SNR exactly 0), noise masking of sparse sensor context (Algorithm 1),
temporal-token cross-attention conditioning through
`diffusers.UNet2DConditionModel` with CrossAttn at the 2 middle levels,
raw-beta_t reverse variance, one-step training with autoregressive rollout
inference, 2/8 val/test scenarios, the seeded 10-mask eval bank, and the
train-only standardization stats.

## Changes and their evidence

### 0. Delta prediction (biggest single lever)

V2 diffuses `x0 = (next_frame − base) / delta_scale` instead of the absolute
field. Measured on the train split
(`tools/compute_floodcastbench_diff_sparse_v2_delta_stats.py`): the std of
consecutive-frame water-depth differences is **0.0007 m vs 0.29 m** for the
absolute field — the actual per-step signal is ~400x smaller than what an
absolute-target model spends its capacity re-encoding (this is also why
oracle persistence is such a strong baseline at 300 s spacing: RMSE ~0.004
normalized). In delta space, "beat persistence" becomes "predict better than
all-zeros", and the newly-inundated-pixel signal — the propagation-path
population — IS the target rather than a 1e-3-relative perturbation of it.

Base definition (mask-aware, degrades gracefully under sparsity): the last
context frame as observed — true values at sensor cells, train-mean fill
(= 0 normalized) elsewhere. Under `missing_rate=0` this is exactly the true
last frame; from rollout step 2 onward it is the model's own dense previous
prediction (matching the reference's rollout, which feeds dense predictions
back); only rollout step 1 under sparsity uses the partially-filled base, at
which pixels the delta target reduces to the absolute anomaly. Physical
clamping maps to a per-pixel tensor floor `(0_depth − base)/scale` inside the
sampler.

### 1. Hybrid conditioning (biggest expected win on all metrics)

`SpatialContextEncoder`: a shallow padded conv stack over the full per-pixel
context stack [12 masked water frames, sensor mask, DEM, 12 context rainfall
frames, target-step rainfall] producing 16 full-resolution feature maps
concatenated with `x_noisy` at the UNet input. The temporal-token
cross-attention pathway is kept unchanged alongside it.

Evidence: direct, in-repo, and strong. Pure temporal-token conditioning gives
the UNet zero per-pixel access to the context; the concat-style variant
converged to dense one-step val_loss ~0.005 in the early V1 experiments while
attention-only conditioning stalled around ~3.7 for 70+ epochs (a ~700x gap in
the training objective), and the diagnostic isolation showed the attention-only
landscape is genuinely slow rather than broken. Flood-front localization —
exactly what path/propagation IoU measures — is a per-pixel problem.

### 2. Target-step rainfall forcing (aimed directly at propagation path IoU)

The rain falling *during* the predicted interval is fed to the spatial
pathway. It is exogenous forcing (same standing as FNO+'s rainfall inputs),
available at every rollout step (the V1 dataset already returns rainfall for
context + prediction frames), and it is the direct causal driver of *newly*
inundated pixels: V1 predicts next-frame water without knowing the rain that
produces it.

### 3. Physically-bounded sampling (the propagation-IoU noise fix)

Water depth cannot be negative, yet ~24-29% of both FNO+ and DIFF-SPARSE V1
predictions are (`reports/diff_sparse_v1_vs_fno_plus_shared_metrics.md`).
Worse for the mask metrics: DDPM sampling noise oscillating around the tiny
γ=0.001 m threshold creates enormous spurious "newly flooded" pixel counts
each step — V1's measured propagation path IoU was ~1e-4 with ~414k false
positives vs 38 true positives at h216. V2 clamps x0_hat to ≥ 0 physical
depth (normalized floor = (0 − mean)/std = −0.3644, verified equal to the
dataset's observed normalized minimum) at **every** reverse-diffusion step,
plus a final clamp on forecasts. A pixel at the floor sits at exactly 0 m and
can only cross γ when the model actually predicts water.

### 4. Rollout-RMSE checkpoint selection + early stopping

The reference implementation selects checkpoints by real generative quality
(`val_masked_nrmse` from ancestral sampling); V1 used a cheap one-step
denoising proxy, which measures a different quantity and misled LR scheduling
once already. V2's `RolloutValidator` rolls out a small fixed set of val
tiles (2 windows × 4 tiles, 1 scenario, fixed RNG, full 12-step
autoregression) every 5 epochs; `checkpoint_best` is selected on this rollout
RMSE. Early stopping (default 60 epochs without improvement) implements the
previously-flagged compute-saving follow-up. The one-step val_loss is kept
per-epoch to drive the LR scheduler (patience 15, V1's validated setting).

### 5. Capacity, context, and diffusion resolution

- UNet 32/64/64/128 (≈4x V1's paper-Table-3 sizing), context embedding 64.
  Small models are a fidelity constraint inherited from Table 3, not a
  performance choice; ~5M parameters is still tiny for a 16 GB GPU.
- **Context 24 frames (2 h)** instead of 12: window length 36 → 2285 train /
  13 val / 13 test eligible windows (split frame ranges unchanged).
- **Diffusion steps 40** instead of 20: finer reverse process; the
  linear-to-1.0 beta schedule keeps terminal SNR exactly 0 (the identity
  property), inference cost scales ~2x.

### 6. Training procedure

- **EMA (0.999)** of weights; rollout validation and final evaluation use the
  EMA weights — the standard free sample-quality win in diffusion.
- **Dihedral augmentation** (8 flips/rotations, all spatial channels
  transformed together — physically consistent since gravity enters only
  through the DEM channel): the benchmark is a single event over a single
  region, so spatial augmentation is the main regularization available.
- **Pushforward exposure-bias training** (fraction 0.25): one extra no-grad
  forward approximates the model's step-1 prediction (terminal-step x0_hat
  from pure noise — exactly the first reverse-sampling step), which replaces
  the true frame in the context; the gradient step then trains step 2
  relative to that imperfect base. One-step teacher forcing never exercises
  error compounding; the autoregressive rollout is nothing but.
- **Change-weighted loss** (`loss.change_weight: 3.0`): pixels whose wet/dry
  state (γ=0.001 m) differs between base and target get 4x weight — exactly
  the propagation-path population, otherwise a vanishing fraction of the MSE.
- **bf16 autocast** on the training forward (sampling and validation stay
  fp32): ~1.5-2x throughput → more epochs per wall-clock hour.
- Rollout-RMSE checkpoint selection + early stopping as described above.

### 7. Evaluation upgrades

- `MultiHorizonPathAccumulator`: path IoU (cumulative newly-flooded area) and
  per-step propagation IoU at **every** rollout step, merged into the
  per-step official CSV — fixes V1's final-horizon-only gap flagged earlier.
- **Scenario-majority decision masks** (`*_median` columns): flood masks and
  path/propagation IoU computed from the per-pixel **median** of scenarios.
  For every threshold γ simultaneously, median > γ ⇔ the majority of
  scenarios exceed γ — the optimal mask decision rule under the model's own
  uncertainty, instead of thresholding the front-smearing mean. Standard
  mean-forecast columns are kept alongside for comparability with V1/FNO+.
- Default tile stride 32 (was 48): stronger Hann overlap, fewer seam
  artifacts in flood-front geometry (knob; costs ~2x tiles).
- `rollout_remask: false` default (matches the reference's rollout, resolved
  in the V1 investigation).
- Optional one-model-for-all-sparsities training via
  `masking.missing_rate_range` (off by default; would collapse the 9-run
  queue to 3 runs if adopted).

### 8. Ablation knobs included but OFF by default

- `loss.snr_gamma`: Min-SNR-style per-timestep loss weighting adapted to
  x0-parameterization (normalized to mean 1, floored at 0.05 so the terminal
  pure-noise step keeps signal — the sampler's first reverse step consumes
  that prediction). Off by default: unvalidated on this task.
- `training.consistency_loss_weight`: hydraulic spatial-coherence penalty
  ported from the reference repo's own (unused, weight-0) consistency_loss.py
  — penalizes a low cell's water surface rising above an adjacent dry higher
  cell. Applied to a t=0 x0 prediction in physical units. Off by default.

## Deliberately not done

- **v-prediction / eps-prediction**: x0-parameterization is a central,
  named design choice of the paper — changing it leaves DIFF-SPARSE.
- **Log-transform of depth**: equivalent pressure on shallow/front pixels is
  applied via the change-weighted loss without breaking the linear
  normalized↔physical mapping every metric and baseline relies on.

## Verification status (2026-07-05)

- 18/18 V2 smoke tests pass on CPU: model contract, scalar AND per-pixel
  tensor clamp floors honored by the sampler, DeltaSpec round trip +
  sampler-floor mapping, change-weight map semantics, EMA update/swap math,
  dihedral transform distinctness, Min-SNR weights, consistency-loss
  semantics, target-rain slicing, multi-horizon path accumulator
  hand-checked cases, real-config forward, optimizer-registration guard.
- Full CPU dry-run of the trainer against the real dataset: delta space
  active (scale 0.00239 normalized), context tokens [B,24,64], spatial
  features [B,16,64,64], pixel_weights present, timestep range confirms 40
  steps, terminal SNR 0, loss finite.
- V1 test suite still green; no V1 file was modified.
- NOT yet GPU-piloted or trained: the V1 retrain queue owns the GPU. Before
  the first full V2 queue: run a short (~40-60 epoch) pilot to sanity-check
  convergence speed (delta space changes the loss landscape entirely),
  rollout-validator overhead, and bf16 stability — then the standard
  3-seed × 3-sparsity protocol with the V2 evaluator.

## Non-claims

No performance claim of any kind until trained and evaluated; V2 is prepared,
not validated. Not an official FloodCastBench benchmark result, not a paper
reproduction (that is V1's role), not evidence of superiority over V1/FNO+/
persistence until the numbers exist.
