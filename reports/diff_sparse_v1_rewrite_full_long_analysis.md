# DIFF-SPARSE v1 rewrite full queue - long analysis

## Executive status
- Queue completed: `2026-07-08T11:19:54+08:00`.
- Scope completed: 9 trainings, 30 rollout evaluations, 3 seeds (`seed42`, `seed7`, `seed123`), missing rates `0.0`, `0.5`, `0.95`.
- No claim here is an official FloodCastBench benchmark result or a full original DIFF-SPARSE reproduction.
- This rewrite queue uses the reference-style architecture path noted in the log: `diffusers UNet2DConditionModel + temporal-token conditioning`.
- Important horizon caveat: this queue evaluates `h13:h24` (`prediction_length=12`), while the older protocol-161 aggregate evaluated `h13:h20`; only explicit horizon fields such as `h20` are safer to compare.

## Run completeness
| seed | missing_rate | epochs | best_epoch | best_val_loss | mask_mean_first_batch | evals |
|---|---:|---:|---:|---:|---:|---:|
| seed42 | 0.00 | 300 | 207 | 3.631656 | 1.000 | 2 |
| seed42 | 0.50 | 300 | 300 | 2.603761 | 0.500 | 4 |
| seed42 | 0.95 | 300 | 168 | 3.205375 | 0.050 | 4 |
| seed7 | 0.00 | 300 | 218 | 3.623225 | 1.000 | 2 |
| seed7 | 0.50 | 300 | 252 | 3.056184 | 0.500 | 4 |
| seed7 | 0.95 | 300 | 182 | 3.215099 | 0.050 | 4 |
| seed123 | 0.00 | 300 | 156 | 3.672655 | 1.000 | 2 |
| seed123 | 0.50 | 300 | 167 | 3.225657 | 0.500 | 4 |
| seed123 | 0.95 | 300 | 103 | 3.277412 | 0.050 | 4 |

## Aggregate metrics by condition
Values are means over 3 seeds. Model RMSE is the evaluator model overall RMSE in meters over h13:h24. h20 and h24 are explicit horizon slices.

| split | missing | baseline | model RMSE m | persistence RMSE m | improvement vs persistence | h20 RMSE m | h24 RMSE m | h24 CSI 0.01 | path IoU 0.001 |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| val | 0 | oracle | 0.644679 | 0.002013 | -31926.67% | 0.644199 | 0.646676 | 0.587497 | 0.004086 |
| val | 0.5 | sparse | 0.649043 | 0.461304 | -40.70% | 0.651656 | 0.728882 | 0.593248 | 0.004058 |
| val | 0.5 | oracle | 0.649043 | 0.002013 | -32143.43% | 0.651656 | 0.728882 | 0.593248 | 0.004058 |
| val | 0.95 | sparse | 0.745087 | 0.634918 | -17.35% | 0.762225 | 0.778361 | 0.590445 | 0.004100 |
| val | 0.95 | oracle | 0.745087 | 0.002013 | -36914.78% | 0.762225 | 0.778361 | 0.590445 | 0.004100 |
| test | 0 | oracle | 0.606249 | 0.001083 | -55883.70% | 0.606239 | 0.609600 | 0.587079 | 0.001337 |
| test | 0.5 | sparse | 0.631738 | 0.463942 | -36.17% | 0.638513 | 0.704148 | 0.589158 | 0.001335 |
| test | 0.5 | oracle | 0.631738 | 0.001083 | -58237.54% | 0.638513 | 0.704148 | 0.589158 | 0.001335 |
| test | 0.95 | sparse | 0.708406 | 0.638573 | -10.94% | 0.724392 | 0.735049 | 0.587082 | 0.001338 |
| test | 0.95 | oracle | 0.708406 | 0.001083 | -65317.42% | 0.724392 | 0.735049 | 0.587082 | 0.001338 |

## Reading the numbers
- Dense `missing_rate=0.0`: the model remains far worse than oracle persistence. This is expected to be a brutal baseline because h1-to-future persistence is extremely strong in this FloodCastBench setup.
- Sparse `missing_rate=0.5`: the rewrite model is worse than sparse persistence on both val and test in the 3-seed mean, and massively worse than oracle persistence.
- Sparse `missing_rate=0.95`: the rewrite model is also worse than sparse persistence in the 3-seed mean, and again massively worse than oracle persistence.
- Path IoU and propagation Path IoU remain very small. This does not support a claim of learned propagation-front quality.
- The model often keeps flood area/classification CSI in a moderate range while RMSE/NSE are poor, indicating broad flooded-area overlap can coexist with bad depth magnitudes and weak dynamics.

## Rewrite versus previous protocol-161
This table uses h20 where possible because it is horizon-matched. Overall columns are included only as nonmatched diagnostics because old overall is h13:h20 and rewrite overall is h13:h24.

| split | missing | baseline | old h20 RMSE m | rewrite h20 RMSE m | h20 change | old PathIoU | rewrite PathIoU |
|---|---:|---|---:|---:|---:|---:|---:|
| test | 0 | oracle | 0.024999 | 0.606239 | -2325.05% | 0.003438 | 0.001337 |
| val | 0 | oracle | 0.024057 | 0.644199 | -2577.81% | 0.002691 | 0.004086 |
| test | 0.5 | oracle | 0.164730 | 0.638513 | -287.61% | 0.003696 | 0.001335 |
| test | 0.5 | sparse | 0.164730 | 0.638513 | -287.61% | 0.003696 | 0.001335 |
| val | 0.5 | oracle | 0.171752 | 0.651656 | -279.42% | 0.002905 | 0.004058 |
| val | 0.5 | sparse | 0.171752 | 0.651656 | -279.42% | 0.002905 | 0.004058 |
| test | 0.95 | oracle | 0.432397 | 0.724392 | -67.53% | 0.003978 | 0.001338 |
| test | 0.95 | sparse | 0.432397 | 0.724392 | -67.53% | 0.003978 | 0.001338 |
| val | 0.95 | oracle | 0.343307 | 0.762225 | -122.02% | 0.003117 | 0.004100 |
| val | 0.95 | sparse | 0.343307 | 0.762225 | -122.02% | 0.003117 | 0.004100 |

Interpretation of h20 comparison: positive change means rewrite reduced RMSE; negative means rewrite is worse. The rewrite is clearly worse than the earlier local protocol on scalar h20 RMSE across all listed conditions.

## FNO+ context
| reference | T+20 RMSE m | T+20 CSI 0.001 | T+20 CSI 0.01 | note |
|---|---:|---:|---:|---|
| Official Table 4 FNO+ | 0.003941 | 0.939638 | 0.984588 | Context only; not same protocol as DIFF-SPARSE rewrite. |
| Official-v1 Best (main baseline) | 0.006480 | 0.927199 | 0.993997 | Context only; not same protocol as DIFF-SPARSE rewrite. |

FNO+ remains the stronger practical reference in the existing artifacts, but this report does not present a strict FNO+ versus DIFF-SPARSE benchmark because the protocols and horizon scopes are not identical.

## Scientific interpretation
- The rewrite queue validates that the full training/evaluation machinery can complete across seeds and missing rates.
- It does not validate sparse-sensor robustness in the strong sense, because oracle persistence still dominates and high missing-rate performance is weak.
- It does not validate long-horizon forecasting skill, uncertainty calibration, or superiority over FNO+.
- The most honest result is negative/diagnostic: the reference-style rewrite did not produce a convincing performance breakthrough; it mostly exposes that architecture/protocol changes alone do not overcome the persistence and FNO+ baselines.

## Recommended next steps
1. Stop launching more full 3-seed queues until a smaller diagnostic improves h20/h24 RMSE against sparse persistence and oracle persistence.
2. Inspect saved or newly generated maps for `m95` to determine whether the model is over-flooding, under-flooding, or collapsing to smooth depth fields.
3. Run a tiny ablation on the best/worst seed: lower beta_end or revise x0/delta target scaling before any expensive multiseed repeat.
4. Build a final dashboard from these aggregate CSVs only after marking the protocol mismatch (`h13:h24` vs `h13:h20`) in the UI.

## Files produced
- `/home/wissam/utem-workspace/code/a-iat-research/reports/diff_sparse_v1_rewrite_full_eval_aggregate.csv`
- `/home/wissam/utem-workspace/code/a-iat-research/reports/diff_sparse_v1_rewrite_full_eval_aggregate_by_condition.csv`
- `/home/wissam/utem-workspace/code/a-iat-research/reports/diff_sparse_v1_rewrite_full_train_runs.csv`
- `/home/wissam/utem-workspace/code/a-iat-research/reports/diff_sparse_v1_rewrite_vs_protocol161_h20_comparison.csv`
- `/home/wissam/utem-workspace/code/a-iat-research/reports/diff_sparse_v1_rewrite_full_summary.json`
- `/home/wissam/utem-workspace/code/a-iat-research/reports/diff_sparse_v1_rewrite_full_long_analysis.md`
