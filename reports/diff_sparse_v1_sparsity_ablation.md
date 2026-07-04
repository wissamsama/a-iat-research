# DIFF-SPARSE v1 Sparsity Ablation

This report is reserved for the sparse-sensor acceptance table requested for
the FloodCastBench high-fidelity 60m DIFF-SPARSE v1 adaptation.

## Scientific Scope

- This is a normalized/physical engineering sanity comparison inside the local
  FloodCastBench adaptation.
- It is not official FloodCastBench benchmark performance.
- It is not an official DIFF-SPARSE TideWatch reproduction.
- It is not uncertainty calibration.
- It is not a claim of superiority over FNO+.

## Baselines

- Model: DIFF-SPARSE v1 checkpoint trained at the listed missing_rate.
- Sparse persistence: last true context frame at observed sensor cells, with
  unobserved cells filled by the train water mean, equal to 0 after
  standardization.
- Oracle dense persistence: dense last true context frame, retained as a
  conservative historical reference. Under sparsity this is an oracle baseline,
  not a fair sensor-limited baseline.

## Multi-Seed Acceptance Table (161-epoch protocol, seeds 42/7/123)

Superseding table. Every (seed x missing_rate) combination in this section is
an independent training under the fixed protocol in
`reports/diff_sparse_v1_fno_plus_training_protocol.md` (epochs=161, gradient-
step parity with FNO+ official-v1). 9 trainings total, run 2026-07-04,
finished 2026-07-05T00:09:58+08:00. Values are normalized-space RMSE,
h13..h20 pooled, `checkpoint_best.pth` selected by val_loss.

### Convergence (best epoch by val_loss, out of a 161-epoch ceiling)

| missing_rate | seed 42 | seed 7 | seed 123 |
|---:|---:|---:|---:|
| 0.0 (dense) | 100 | 79 | 89 |
| 0.5 | 137 | 159 | 133 |
| 0.95 | 58 | 68 | 68 |

Convergence speed is sparsity-dependent, consistently across all 3 seeds:
dense and 0.5 use most or all of the 161-epoch budget, while 0.95 plateaus
(and for seed 42, mildly regresses afterward) by epoch ~60-70 in every seed.
At 95% missing sensors there is little real signal left to keep learning
from, so the model saturates far earlier than at lower sparsity — this
generalizes across seeds, it is not particular to one run. Practical
implication: an early-stopping criterion (e.g. patience ~25-30 epochs with no
val_loss improvement) would have saved roughly 90-100 epochs of compute per
0.95 run without changing which checkpoint gets selected.

### RMSE vs sparse persistence (mean ± std across 3 seeds)

| Split | missing_rate | Model RMSE (mean ± std) | CV | Sparse persistence RMSE | Mean win margin |
|---|---:|---:|---:|---:|---:|
| val | 0.0 | 0.06469 ± 0.00342 | 5.3% | 0.004677 | model loses, ~14x worse |
| test | 0.0 | 0.06773 ± 0.00325 | 4.8% | 0.002534 | model loses, ~27x worse |
| val | 0.5 | **0.57559 ± 0.01663** | 2.9% | 1.585139 | **model wins, 2.75x** |
| test | 0.5 | **0.54888 ± 0.01834** | 3.3% | 1.594029 | **model wins, 2.90x** |
| val | 0.95 | **1.11620 ± 0.12942** | 11.6% | 2.181420 | **model wins, 1.95x** |
| test | 0.95 | **1.36432 ± 0.28547** | 20.9% | 2.193758 | **model wins, 1.61x** |

Per-seed win margin (persistence RMSE / model RMSE), confirming the result
holds for **every individual seed**, not just the average:

| | val 0.5 | test 0.5 | val 0.95 | test 0.95 |
|---|---:|---:|---:|---:|
| seed 42 | 2.80x | 2.96x | 1.81x | 1.38x |
| seed 7 | 2.80x | 2.97x | 2.25x | 2.10x |
| seed 123 | 2.67x | 2.80x | 1.86x | 1.51x |

**Reading the variance**: at missing_rate=0.5 the win is tight and highly
reproducible (CV 2.9-3.3%, margins clustered 2.67-2.97x). At missing_rate=0.95
the win is real and holds in all 6 seed x split cells, but the *margin* is
seed-sensitive (CV 11.6-20.9%, margins ranging 1.38x-2.25x) — consistent with
convergence happening so early (epoch 58-68) that the model can settle into
seed-dependent optima of different quality when there is very little real
signal to constrain training. Report the 0.95 win as "real but with a wide
margin," not as a precise multiplier.

## Superseded Single-Seed Table (40-epoch protocol, seed ~42 only)

Kept for the convergence-check narrative below; superseded by the multi-seed
161-epoch table above for any headline claim.

Each model is a separate 40-epoch training at its listed `missing_rate`
(dense reused from the earlier 40-epoch run; sparse levels trained
03-07/04-07-2026). All values are normalized-space RMSE/MAE, h13..h20 pooled.

| Split | Missing rate | Model RMSE | Sparse persistence RMSE | Oracle persistence RMSE | Model MAE | Sparse persistence MAE | Oracle persistence MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| val | 0.0 | 0.089557 | 0.004677 | 0.004677 | 0.050236 | 0.001439 | 0.001439 |
| test | 0.0 | 0.093462 | 0.002534 | 0.002534 | 0.051656 | 0.000788 | 0.000788 |
| val | 0.5 | **0.794588** | 1.585139 | 0.004677 | **0.331212** | 0.612903 | 0.001439 |
| test | 0.5 | **0.747043** | 1.594029 | 0.002534 | **0.314398** | 0.615356 | 0.000788 |
| val | 0.95 | **1.208136** | 2.181420 | 0.004677 | **0.600436** | 1.162097 | 0.001439 |
| test | 0.95 | **1.335291** | 2.193758 | 0.002534 | **0.709795** | 1.167317 | 0.000788 |

At `missing_rate=0.0` sparse and oracle persistence are identical (the sensor
mask is all-ones, so there is nothing to distinguish). Bold marks the model
beating sparse persistence.

## Result (superseded by the multi-seed table above)

**DIFF-SPARSE v1 beats sparse persistence at both non-trivial sparsity
levels, on both splits, confirmed across all 3 seeds — see "Reading the
variance" above for the caveat on 0.95's margin.**

- missing_rate=0.5 (single-seed 40-epoch numbers below): model RMSE is
  ~2.0-2.1x lower than sparse persistence (0.79 vs 1.59 on val; 0.75 vs 1.59
  on test). The 161-epoch multi-seed mean pushes this to **~2.75-2.90x**.
- missing_rate=0.95 (single-seed 40-epoch numbers below): model RMSE is
  ~1.6-1.8x lower than sparse persistence. The 161-epoch multi-seed mean is
  **~1.61-1.95x**, individually ranging 1.38x-2.25x across seeds.

This is the first result in the DIFF-SPARSE v1 adaptation where the model
outperforms a persistence baseline, and it is now seed-checked: every one of
the 6 (seed x split) cells at each sparsity level beats sparse persistence,
not just the average. It is consistent with the paper's own framing: sparse
persistence has no way to fill unobserved cells except a constant (here, the
train water mean = 0 normalized), while DIFF-SPARSE's masked training lets it
condition on DEM and rainfall to infer plausible values where sensors are
absent.

Against **oracle** (dense, full-field) persistence, the model still loses
badly at every sparsity level (oracle RMSE ~0.0025-0.0047 vs model RMSE
0.75-1.34) — expected, since oracle persistence assumes information
(the true dense field) that a real sparse-sensor deployment would not have.
Oracle persistence is not a fair baseline under sparsity; it is retained only
as the same fixed historical reference point used throughout this project.

## Convergence Check (missing_rate=0.0: 40 -> 80 -> 161 epochs, seed 42)

| Epochs | Val model RMSE (normalized) | Val persistence RMSE |
|---:|---:|---:|
| 40 | 0.089557 | 0.004677 |
| 80 | 0.070519 | 0.004677 |
| 161 (best epoch 100) | 0.060772 | 0.004677 |

Extending training keeps improving dense val RMSE: ~21% relative gain from 40
to 80 epochs, ~32% total from 40 to 161. It remains far behind persistence at
this short h13..h20 horizon regardless — more training narrows but does not
close this particular gap (persistence is simply an unusually strong
baseline at 5-minute spacing with no missing sensors).

Across all 3 seeds at 161 epochs, dense `checkpoint_best.pth` was actually
selected at epoch 79-100 (see the multi-seed convergence table above) — the
161-epoch ceiling was never the binding constraint for dense, it was already
converged with room to spare. An early-stopping criterion would have reached
the same selected checkpoint at a fraction of the wall-clock cost.

## Notes

- The dense (missing_rate=0.0) checkpoint loses to persistence through
  h13..h20 at both 40 and 80 epochs — persistence is an unusually strong
  baseline at 5-minute spacing with no missing sensors, so a diffusion
  model's sampling variance is a real cost with no offsetting advantage.
- Under sparsity, the comparison flips: persistence has no mechanism to
  fill missing sensors at all, and DIFF-SPARSE's masked training turns that
  into a genuine, reproducible advantage on both splits.
- Sparse-sensor evaluation is the scientifically relevant DIFF-SPARSE framing
  per the paper; this table is the first evidence in this repo that the
  adaptation delivers on that framing, not just a working pipeline.
