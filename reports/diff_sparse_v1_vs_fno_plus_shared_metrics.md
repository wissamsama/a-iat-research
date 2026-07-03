# DIFF-SPARSE v1 vs FNO+ Shared Metrics

This note compares DIFF-SPARSE v1 dense rollout against the existing official-v1
FNO+ best checkpoint and oracle persistence on the shared h13..h20 test window.
Metrics are computed in physical water-depth space after inverse-transforming
DIFF-SPARSE normalized tensors with the shared train water statistic.

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
