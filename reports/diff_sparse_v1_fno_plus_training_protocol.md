# DIFF-SPARSE v1 / FNO+ Training Protocol (fixed 2026-07-04)

## Why "same epoch count" does not mean "same training budget"

FNO+ official-v1 and DIFF-SPARSE v1 use different batching and windowing:

| | Batch size | Windows/epoch | Windowing |
|---|---:|---:|---|
| FNO+ official-v1 | 1 | 116 | non-overlapping (canonical) |
| DIFF-SPARSE v1 | 32 | 2301 | sliding, stride=1 |

At FNO+'s published protocol (100 epochs):

| Axis | FNO+ (100 epochs) | DIFF-SPARSE at 40 epochs (previous default) |
|---|---:|---:|
| Samples seen | 11,600 | 92,040 (~8x more) |
| Gradient steps | 11,600 | 2,876 (~4x fewer) |
| Wall-clock (uncontended) | ~90 min | ~14 min |

No single epoch count equalizes all three axes simultaneously — matching
samples-seen would require ~5 epochs (far too few to converge), matching
wall-clock would require ~257 epochs. These are not interchangeable
definitions of "fair," they are different questions.

## Fixed protocol: equalize gradient steps

DIFF-SPARSE v1 is set to **`training.epochs: 161`** everywhere (base config
and both seed variants), chosen so that:

```
161 epochs x 2301 windows/epoch / 32 batch_size ≈ 11,600 gradient steps
```

matching FNO+'s 100 epochs x 116 windows / batch_size=1 = 11,600 gradient
steps exactly. Rationale:

- Gradient steps (parameter updates) is the standard ML unit for "training
  budget," more defensible than raw epoch count when windowing/batching
  differ this much between two pipelines.
- It is cheap for DIFF-SPARSE (~56 min uncontended for 161 epochs), so this
  choice does not trade off convergence for parity.
- It is directionally consistent with independent convergence evidence
  already collected: the 40-epoch dense checkpoint had not converged
  (val RMSE improved ~21% from 40 to 80 epochs), so more epochs than 40 was
  already motivated on its own before this parity argument was added.

`batch_size` is left at 32 and is not treated as a comparison axis — it is a
throughput/memory choice for a diffusion model with a very different
per-step cost (a stochastic forward pass through the noise schedule) than
FNO+'s deterministic regression step, not a scientific parameter to force
into agreement.

## Scope

This protocol applies to **every** DIFF-SPARSE v1 training going forward,
across all seeds (42, 7, 123) and all sparsity levels (0.0, 0.5, 0.95), so
every number that lands on the shared dashboard/reports comes from the same
training budget definition. Superseded results:

- Dense (missing_rate=0.0) 40-epoch and 80-epoch seed=42 checkpoints
  (`03-07-2026_15-51-43_...` and `04-07-2026_02-50-24_...`) predate this
  protocol and are kept on disk for the convergence-check writeup in
  `reports/diff_sparse_v1_sparsity_ablation.md`, but are not "official" going
  forward.
- Sparse 0.5/0.95 seed=42 40-epoch checkpoints
  (`03-07-2026_22-55-12_...`, `04-07-2026_00-53-04_...`) and the abandoned
  seed 7/123 40-epoch sparse runs are superseded by the 161-epoch reruns
  documented below once complete.

## Runs under this protocol

All 9 trainings completed 2026-07-04, queue finished
2026-07-05T00:09:58+08:00 (started 2026-07-04T10:10:40+08:00, ~14h wall-clock
for all 9 trainings + 30 evals, uncontended GPU). See `reports/diff_sparse_v1_sparsity_ablation.md` for the full multi-seed
results table and per-run convergence discussion.

| Seed | missing_rate | Run directory | Best epoch (val_loss) | Status |
|---:|---:|---|---:|---|
| 42 | 0.0 | `04-07-2026_10-10-42_fcb_diff_sparse_v1_highfid_60m` | 100 | done |
| 7 | 0.0 | `04-07-2026_14-50-59_fcb_diff_sparse_v1_seed7_highfid_60m` | 79 | done |
| 123 | 0.0 | `04-07-2026_19-29-46_fcb_diff_sparse_v1_seed123_highfid_60m` | 89 | done |
| 42 | 0.5 | `04-07-2026_11-28-30_fcb_diff_sparse_v1_highfid_60m` | 137 | done |
| 7 | 0.5 | `04-07-2026_16-08-49_fcb_diff_sparse_v1_seed7_highfid_60m` | 159 | done |
| 123 | 0.5 | `04-07-2026_20-47-41_fcb_diff_sparse_v1_seed123_highfid_60m` | 133 | done |
| 42 | 0.95 | `04-07-2026_13-09-59_fcb_diff_sparse_v1_highfid_60m` | 58 | done |
| 7 | 0.95 | `04-07-2026_17-48-58_fcb_diff_sparse_v1_seed7_highfid_60m` | 68 | done |
| 123 | 0.95 | `04-07-2026_22-28-24_fcb_diff_sparse_v1_seed123_highfid_60m` | 68 | done |

## 2026-07-05 revision: paper-faithful v1 retrain (supersedes the 9 runs above)

Two paper-fidelity corrections were applied to all v1 configs
(`prediction_length: 8 -> 12` per paper Table 2, and
`conditioning: cross_attention_concat -> cross_attention`, the paper's
attention-only conditioning). The parity arithmetic is essentially unchanged
by the new window length (2297 sliding train windows at 24 frames instead of
2301 at 20 frames: 161 x ceil(2297/32) = 11,592 gradient steps ~ 11,600).
Eligible eval windows drop from 14 to 13 per split.

A first attempt at this retrain (seed42 dense) plateaued completely --
`train_loss` stuck at ~0.4 for all 161 epochs instead of dropping to ~0.0005
like the concat variant. Root-caused (at the time) to `training.lr_patience: 3`
being too aggressive for the hand-rolled attention-only conditioning's harder
optimization landscape, and fixed by raising `lr_patience` to 15, verified on
a 60-epoch pilot (LR stays at 1e-3 throughout, breaks through at epoch ~37,
`val_loss` 3.7 -> 1.65 by epoch 60).

Before committing that fix to a full 9-run retrain, the official reference
code (`github.com/KAI10/Diff-Sparse`) was cloned and read directly, which
found the real problem: the hand-rolled cross-attention conditioning
(pixel-aligned spatial map + fixed 2D positional encodings, no self-attention)
was not what the paper's authors built at all — the reference uses a
temporal token sequence (one summary per context timestep) fed to
`diffusers.UNet2DConditionModel` with cross-attention at the 2 *middle* UNet
levels (self- and cross-attention combined). `models/diff_sparse_v1.py` was
rewritten around this exact mechanism (see `reports/diff_sparse_v1_design.md`
for the full diagnostic and architecture description); the earlier 9 runs and
the seed42 partial retrains (both the `lr_patience=3` and `lr_patience=15`
attempts) are superseded and architecture-incompatible with the current
configs — any re-evaluation of them must use the config snapshot saved inside
each run directory.

`lr_patience` was first reverted to the paper's own value (3) under the new
architecture on the hypothesis that the earlier fix was tied to the (wrong)
hand-rolled attention mechanism. Two pilots (40 and 60 epochs) under the
*correct* reference-matching architecture showed the identical plateau
signature at patience=3, ruling that hypothesis out; `lr_patience=15` was
restored. A further 70-epoch pilot at patience=15 (LR pinned at 1e-3 the
whole time, never reduced) still showed only slow improvement
(val_loss 6.58 -> 3.7, clearly decelerating but not fully flat) — no hard
ceiling, just slow.

A decisive single-batch diagnostic then isolated the cause definitively: at a
**fixed** t=0 diffusion timestep (the easy case), the architecture converges
cleanly to loss ~0.0007 in 300 steps — proof the architecture, gradient flow,
and TemporalContextEncoder are all correct, no bug. At **random** timesteps
(matching real training exactly), the same fixed 8-example batch only reaches
loss ~0.17 in the same 300 steps, clearly still falling. Conclusion: this is
not a bug, it is a genuinely slow optimization landscape for the paper's
cross-attention-only (no-concat) conditioning mechanism across the full
random-timestep objective, requiring substantially more gradient steps than
161 epochs (11,592 steps) provides. `training.epochs` is raised to **300**
(~21,600 gradient steps) for the full retrain, explicitly abandoning FNO+
gradient-step parity as a goal for this specific run (a deliberate trade-off,
chosen directly over further piloting once the "not a bug, just slow"
diagnosis was confirmed).

## Follow-up: early stopping recommended, not yet implemented

The 161-epoch ceiling was sized for gradient-step parity, not measured
convergence. In practice: dense converges by epoch ~80-100, missing_rate=0.5
uses nearly the full budget (133-159), and missing_rate=0.95 converges very
early and consistently (epoch 58-68 in all 3 seeds) — 90-100 epochs of
compute per 0.95 run were spent past the selected checkpoint with no benefit
(and a mild val_loss regression in seed 42's case). A proper early-stopping
criterion (distinct from the LR scheduler's patience=3, e.g. a separate
patience of ~25-30 epochs with no val_loss improvement) would reach the same
selected checkpoints at a fraction of the wall-clock cost, without changing
any reported result, since `checkpoint_best.pth` is already selected by
val_loss regardless of how long the loop keeps running. Not yet implemented
in `tools/train_floodcastbench_diff_sparse_v1.py` — the 9 runs above all ran
the full fixed 161 epochs.
