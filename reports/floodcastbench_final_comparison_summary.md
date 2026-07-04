# FloodCastBench final process and protocol comparison summary

Status: no active FloodCastBench/FNO+/DIFF-SPARSE processes were detected at report creation time.

## Completed queues
- fno_plus_multiseed: exists=True, complete=True, start=FNO+ multi-seed cross-validation queue started at 2026-07-03T17:43:03+08:00, finish=FNO+ multi-seed cross-validation queue finished at 2026-07-04T03:59:26+08:00
- diff_sparse_protocol161_full: exists=True, complete=True, start=DIFF-SPARSE v1 protocol-161 full queue started at 2026-07-04T10:10:39+08:00, finish=DIFF-SPARSE v1 protocol-161 full queue finished at 2026-07-05T00:09:58+08:00
- diff_sparse_seed_ablation_partial: exists=True, complete=False, start=DIFF-SPARSE v1 seed-ablation queue started at 2026-07-04T09:47:29+08:00, finish=None
- diff_sparse_followup_retry3: exists=True, complete=True, start=DIFF-SPARSE v1 follow-up queue started at 2026-07-03T20:07:31+08:00, finish=DIFF-SPARSE v1 follow-up queue finished at 2026-07-04T03:31:55+08:00

## Main result, with caveats
- FNO+ official-v1 normalized runs completed for seed7 and seed123. Their eval JSONs report pooled h2:h20 physical metrics after inverse transform.
- DIFF-SPARSE v1 protocol-161 completed 9 trainings and 30 evaluations across seeds seed42, seed7, seed123 and missing rates 0.0, 0.5, 0.95.
- The partial seed-ablation run from 04-07-2026_09-47-32 is excluded because the queue did not finish the full requested matrix.
- Direct FNO+ pooled h2:h20 versus DIFF-SPARSE overall h13:h20 is not a strict horizon-matched comparison. Use it as context only.

## FNO+ pooled h2:h20 mean of 2 seeds
| split | RMSE m | current_relative_rmse | NSE | Pearson r |
|---|---:|---:|---:|---:|
| val | 0.004517 | 0.006452 | 0.999943 | 0.999972 |
| test | 0.004558 | 0.006478 | 0.999943 | 0.999972 |

## DIFF-SPARSE v1 protocol-161 mean of 3 seeds
| split | missing_rate | persistence | model RMSE m | persistence RMSE m | improvement pct | h20 RMSE m | h20 CSI 0.01 | path IoU 0.001 |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| val | 0 | oracle | 0.018832 | 0.001362 | -1283.02 | 0.024057 | 0.854630 | 0.002691 |
| val | 0.5 | sparse | 0.167576 | 0.461490 | 63.69 | 0.171752 | 0.703717 | 0.002905 |
| val | 0.5 | oracle | 0.167576 | 0.001362 | -12206.52 | 0.171752 | 0.703717 | 0.002905 |
| val | 0.95 | sparse | 0.324966 | 0.635088 | 48.83 | 0.343307 | 0.676725 | 0.003117 |
| val | 0.95 | oracle | 0.324966 | 0.001362 | -23765.01 | 0.343307 | 0.676725 | 0.003117 |
| test | 0 | oracle | 0.019720 | 0.000738 | -2573.49 | 0.024999 | 0.849758 | 0.003438 |
| test | 0.5 | sparse | 0.159798 | 0.464078 | 65.57 | 0.164730 | 0.696757 | 0.003696 |
| test | 0.5 | oracle | 0.159798 | 0.000738 | -21564.18 | 0.164730 | 0.696757 | 0.003696 |
| test | 0.95 | sparse | 0.397201 | 0.638680 | 37.81 | 0.432397 | 0.660587 | 0.003978 |
| test | 0.95 | oracle | 0.397201 | 0.000738 | -53749.33 | 0.432397 | 0.660587 | 0.003978 |

## Interpretation
- Dense missing_rate=0.0 DIFF-SPARSE-style loses badly to oracle persistence. The h1-to-h20 oracle persistence baseline is extremely strong in this protocol.
- Sparse missing_rate=0.5 and 0.95 DIFF-SPARSE-style beats sparse persistence in normalized/physical sanity metrics, but still loses badly to oracle persistence.
- Path IoU remains near zero, so this should not be described as successful flood-front propagation modeling.
- FNO+ pooled h2:h20 metrics are much smaller than DIFF-SPARSE v1 physical rollout errors, but the horizon scopes differ. Treat this as a strong warning sign and not as a final paper-grade claim.

## Recommended next step
1. Generate horizon-matched FNO+ h13:h20 and h20-only metrics from the existing FNO+ checkpoints, then rebuild the comparison table.
2. For DIFF-SPARSE, inspect maps for sparse m50/m95 and dense m0 to determine whether errors are bias/oversmoothing/noise rather than only scalar loss differences.
3. Keep oracle persistence as the primary hard baseline for any h20 claim.

## Files generated
- /home/wissam/utem-workspace/code/a-iat-research/reports/floodcastbench_final_comparison_fno_diff_sparse.csv
- /home/wissam/utem-workspace/code/a-iat-research/reports/floodcastbench_final_run_manifest.json
- /home/wissam/utem-workspace/code/a-iat-research/reports/floodcastbench_final_comparison_summary.md
- /home/wissam/utem-workspace/code/a-iat-research/reports/floodcastbench_final_comparison_dashboard.html
