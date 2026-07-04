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

Filled in as the 161-epoch queue completes (dense/0.5/0.95 x seeds 42/7/123,
9 trainings total).

| Seed | missing_rate | Run directory | Status |
|---:|---:|---|---|
| 42 | 0.0 | pending | pending |
| 7 | 0.0 | pending | pending |
| 123 | 0.0 | pending | pending |
| 42 | 0.5 | pending | pending |
| 7 | 0.5 | pending | pending |
| 123 | 0.5 | pending | pending |
| 42 | 0.95 | pending | pending |
| 7 | 0.95 | pending | pending |
| 123 | 0.95 | pending | pending |
