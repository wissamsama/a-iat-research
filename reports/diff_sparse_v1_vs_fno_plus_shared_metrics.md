# DIFF-SPARSE v1 vs FNO+ Shared Metrics

This note compares DIFF-SPARSE v1 dense rollout against the existing official-v1
FNO+ best checkpoint and oracle persistence on the shared h13..h20 test window.
Metrics are computed in physical water-depth space after inverse-transforming
DIFF-SPARSE normalized tensors with the shared train water statistic.

## Multi-Seed Update (161-epoch protocol, 2026-07-05)

Superseding section. DIFF-SPARSE is now the mean ± std across 3 independent
seeds (42/7/123, all dense/missing_rate=0.0, 161-epoch gradient-step-parity
protocol — see `reports/diff_sparse_v1_fno_plus_training_protocol.md`), the
same values embedded in the dashboard
(`scripts/build_fno_plus_metric_dashboard.py` /
`experiments/FloodCastBench/fno_plus_metric_dashboard_scientific.html`).

**FNO+ remains single-seed (42) for this specific per-step h13..h20 breakdown**
— the expensive long-horizon rollout tool that produces individual per-step
metrics has only ever been run for seed 42; seeds 7/123 only have the pooled
h2..h20 aggregate (see `reports/fno_plus_multiseed_results.md`), which is not
broken down by individual step and so cannot be restricted to h13..h20 alone.
This asymmetry (FNO+ single-seed vs DIFF-SPARSE 3-seed) is intentional and
disclosed, not an oversight — closing it would require running FNO+'s
long-horizon rollout for 2 more seeds, a separate, expensive task not done
here.

| Horizon | Series | classical_rmse (m) | current_relative_rmse | nse | pearson_r | csi_gamma_0_001 | csi_gamma_0_01 |
|---|---|---:|---:|---:|---:|---:|---:|
| h13 | DIFF-SPARSE v1 dense (mean±std, N=3) | 0.01483 ± 0.00080 | 0.02108 ± 0.00113 | 0.99939 ± 0.00007 | 0.99971 ± 0.00003 | 0.7430 ± 0.0279 | 0.9046 ± 0.0162 |
| h13 | FNO+ official-v1 best (seed 42 only) | 0.00392 | 0.00557 | 0.99996 | 0.99998 | 0.9258 | 0.9931 |
| h20 | DIFF-SPARSE v1 dense (mean±std, N=3) | 0.02500 ± 0.00138 | 0.03553 ± 0.00196 | 0.99827 ± 0.00019 | 0.99915 ± 0.00009 | 0.7368 ± 0.0231 | 0.8498 ± 0.0219 |
| h20 | FNO+ official-v1 best (seed 42 only) | 0.00703 | 0.00999 | 0.99986 | 0.99994 | 0.8737 | 0.9886 |

**Reading this**: FNO+ still leads DIFF-SPARSE on every shared metric at both
horizons. DIFF-SPARSE's own seed-to-seed std on classical/relative RMSE is
modest and consistent (~5.4-5.5% of the mean at both h13 and h20) — the gap
to FNO+ (DIFF-SPARSE RMSE ~3.6-3.8x FNO+'s) is far outside that noise band,
not an artifact of an unlucky seed. The gap narrows
slightly from h13 to h20 in relative terms (classical_rmse ratio ~3.8x at h13
vs ~3.6x at h20), consistent with FNO+'s own error growing faster with
horizon than a short 8-step DIFF-SPARSE rollout's does — but DIFF-SPARSE
still trails throughout. See the top-level structural discussion (dense vs
sparse vs long-horizon axes) already recorded in this project's conversation
history for why closing this specific dense/short-horizon gap is not the
recommended next objective.

An independent cross-check of the underlying DIFF-SPARSE physical-unit
numbers (computed separately in `reports/floodcastbench_final_comparison_summary.md`
and `reports/floodcastbench_final_comparison_fno_diff_sparse.csv`) agrees
with the classical_rmse values above to within rounding.

## Superseded Single-Seed Section (40-epoch protocol, kept for history)

## Sources

- DIFF-SPARSE eval:
  `/home/wissam/utem-workspace/experiments/FloodCastBench/03-07-2026_15-51-43_fcb_diff_sparse_v1_highfid_60m/eval_rollout_test_03-07-2026_18-11-41`
- FNO+ per-step metrics:
  `/home/wissam/utem-workspace/experiments/FloodCastBench/28-06-2026_15-59-18_fcb_fno_plus_official_v1_normalized_100epoch_highfid_60m/long_horizon_rollout_eval_dense_v2/checkpoint_best/long_horizon_metrics_per_step.csv`

## h13

| Series | classical_rmse | current_relative_rmse | nse | pearson_r | csi_gamma_0_001 | csi_gamma_0_01 | negative_prediction_ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| DIFF-SPARSE v1 dense | 0.020454 | 0.029069 | 0.998845 | 0.999459 | 0.804622 | 0.907196 | 0.253549 |
| FNO+ official-v1 best | 0.003923 | 0.005575 | 0.999958 | 0.999984 | 0.925814 | 0.993077 | 0.285966 |
| Oracle persistence | 0.000147 | 0.000208 | 1.000000 | 1.000000 | 0.999720 | 0.999953 | n/a (not computed) |

## h20

| Series | classical_rmse | current_relative_rmse | nse | pearson_r | csi_gamma_0_001 | csi_gamma_0_01 | negative_prediction_ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| DIFF-SPARSE v1 dense | 0.032314 | 0.045921 | 0.997117 | 0.998756 | 0.777650 | 0.857596 | 0.235653 |
| FNO+ official-v1 best | 0.007030 | 0.009991 | 0.999864 | 0.999939 | 0.873709 | 0.988644 | 0.266116 |
| Oracle persistence | 0.001167 | 0.001658 | 0.999996 | 0.999998 | 0.997089 | 0.999635 | n/a (not computed) |

## Interpretation

On this short h13..h20 dense window, DIFF-SPARSE v1 trails both FNO+ and oracle
persistence on the shared deterministic physical metrics. The profile shape is
consistent with the earlier DIFF-SPARSE diagnostics: the rollout remains
spatially meaningful, but its dense stochastic mean has larger depth error and
lower flood-mask CSI than FNO+ at the same horizons. Oracle persistence remains
extremely strong at 5-minute spacing, so this comparison should not be read as a
long-horizon or sparse-sensor verdict.

One metric where the two models are *not* clearly differentiated:
`negative_prediction_ratio`. DIFF-SPARSE (~24-25%) is actually slightly
*better* than FNO+ (~27-29%) at both horizons — both architectures produce a
substantial fraction of physically-impossible negative water depths, since
neither has an output positivity constraint. This is not a point in favor of
either model; it is a shared limitation worth flagging for both, and a
candidate follow-up (e.g. softplus/relu output head, or clipping at inference)
independent of which architecture is chosen. Persistence trivially avoids this
failure mode by construction (it copies observed physical values, which the
hydrodynamic simulator never produced as negative), so it is not a meaningful
three-way comparison point and is marked n/a above.

## Non-Claims

- Not official FloodCastBench benchmark performance.
- Not physical proof of DIFF-SPARSE TideWatch reproduction.
- Not sparse-sensor robustness.
- Not uncertainty calibration.
- Not evidence of FNO+ or DIFF-SPARSE superiority outside this shared dense
  h13..h20 diagnostic window.
