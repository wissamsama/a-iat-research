# DIFF-SPARSE v1 Protocol-161 Aggregate

Generated: 2026-07-05 local workspace audit.

Scope: protocol-161 full queue only. The partial `diff_sparse_v1_seed_ablation_queue_04-07-2026_09-47-11` is intentionally excluded from the aggregate because it did not finish and was superseded by the full queue.

Rows found: 30 evaluation summaries.

Artifacts:
- `/home/wissam/utem-workspace/code/a-iat-research/reports/diff_sparse_v1_protocol161_eval_aggregate.csv`
- `/home/wissam/utem-workspace/code/a-iat-research/reports/diff_sparse_v1_protocol161_eval_aggregate_by_condition.csv`

Scientific status: normalized/physical transformed rollout sanity evaluation for local DIFF-SPARSE v1 adaptation. This is not an official FloodCastBench benchmark claim, not full DIFF-SPARSE TideWatch reproduction, and not uncertainty calibration.

## Aggregate By Condition

| missing | split | mode | n | model RMSE m mean | RMSE std | persistence RMSE m | improv % vs persistence | h20 RMSE m | h20 CSI@0.01 | PathIoU@0.001 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.0 | val | oracle | 3 | 0.018832 | 0.000813 | 0.001362 | -1.283e+03 | 0.024057 | 0.854630 | 0.002691 |
| 0.0 | test | oracle | 3 | 0.019720 | 0.000772 | 0.000738 | -2.573e+03 | 0.024999 | 0.849758 | 0.003438 |
| 0.5 | val | sparse | 3 | 0.167576 | 0.003954 | 0.461490 | 63.69 | 0.171752 | 0.703717 | 0.002905 |
| 0.5 | val | oracle | 3 | 0.167576 | 0.003954 | 0.001362 | -1.221e+04 | 0.171752 | 0.703717 | 0.002905 |
| 0.5 | test | sparse | 3 | 0.159798 | 0.004360 | 0.464078 | 65.57 | 0.164730 | 0.696757 | 0.003696 |
| 0.5 | test | oracle | 3 | 0.159798 | 0.004360 | 0.000738 | -2.156e+04 | 0.164730 | 0.696757 | 0.003696 |
| 0.95 | val | sparse | 3 | 0.324966 | 0.030765 | 0.635088 | 48.83 | 0.343307 | 0.676725 | 0.003117 |
| 0.95 | val | oracle | 3 | 0.324966 | 0.030765 | 0.001362 | -2.377e+04 | 0.343307 | 0.676725 | 0.003117 |
| 0.95 | test | sparse | 3 | 0.397201 | 0.067860 | 0.638680 | 37.81 | 0.432397 | 0.660587 | 0.003978 |
| 0.95 | test | oracle | 3 | 0.397201 | 0.067860 | 0.000738 | -5.375e+04 | 0.432397 | 0.660587 | 0.003978 |

## Test Split Per Seed

| seed | missing | mode | model RMSE m | persistence RMSE m | improv % | h20 RMSE m | h20 CSI@0.01 | PathIoU@0.001 | ckpt epoch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| seed42 | 0.0 | oracle | 0.018721 | 0.000738 | -2.438e+03 | 0.023441 | 0.874972 | 0.003152 | 100 |
| seed42 | 0.5 | sparse | 0.157070 | 0.464078 | 66.15 | 0.160943 | 0.705289 | 0.003794 | 137 |
| seed42 | 0.5 | oracle | 0.157070 | 0.000738 | -2.119e+04 | 0.160943 | 0.705289 | 0.003794 | 137 |
| seed42 | 0.95 | sparse | 0.463589 | 0.638680 | 27.41 | 0.519297 | 0.686179 | 0.004395 | 58 |
| seed42 | 0.95 | oracle | 0.463589 | 0.000738 | -6.275e+04 | 0.519297 | 0.686179 | 0.004395 | 58 |
| seed7 | 0.0 | oracle | 0.020600 | 0.000738 | -2.693e+03 | 0.026064 | 0.836069 | 0.003550 | 79 |
| seed7 | 0.5 | sparse | 0.156373 | 0.464078 | 66.30 | 0.160872 | 0.702634 | 0.003600 | 159 |
| seed7 | 0.5 | oracle | 0.156373 | 0.000738 | -2.110e+04 | 0.160872 | 0.702634 | 0.003600 | 159 |
| seed7 | 0.95 | sparse | 0.303990 | 0.638680 | 52.40 | 0.327015 | 0.667341 | 0.003924 | 68 |
| seed7 | 0.95 | oracle | 0.303990 | 0.000738 | -4.111e+04 | 0.327015 | 0.667341 | 0.003924 | 68 |
| seed123 | 0.0 | oracle | 0.019839 | 0.000738 | -2.590e+03 | 0.025492 | 0.838234 | 0.003610 | 89 |
| seed123 | 0.5 | sparse | 0.165951 | 0.464078 | 64.24 | 0.172376 | 0.682348 | 0.003695 | 133 |
| seed123 | 0.5 | oracle | 0.165951 | 0.000738 | -2.240e+04 | 0.172376 | 0.682348 | 0.003695 | 133 |
| seed123 | 0.95 | sparse | 0.424023 | 0.638680 | 33.61 | 0.450879 | 0.628240 | 0.003615 | 68 |
| seed123 | 0.95 | oracle | 0.424023 | 0.000738 | -5.739e+04 | 0.450879 | 0.628240 | 0.003615 | 68 |

## Best Test Rows By Model RMSE

| seed | missing | split | mode | model RMSE m | h20 RMSE m | overall NSE | overall CSI@0.01 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| seed42 | 0.0 | test | oracle | 0.018721 | 0.023441 | 0.999032 | 0.899940 |
| seed123 | 0.0 | test | oracle | 0.019839 | 0.025492 | 0.998913 | 0.867541 |
| seed7 | 0.0 | test | oracle | 0.020600 | 0.026064 | 0.998828 | 0.864231 |
| seed7 | 0.5 | test | sparse | 0.156373 | 0.160872 | 0.932473 | 0.710498 |
| seed7 | 0.5 | test | oracle | 0.156373 | 0.160872 | 0.932473 | 0.710498 |

## Notes

- Dense `missing_rate=0.0` has only oracle-mode evaluation in the protocol because sparse/oracle persistence are identical with an all-ones mask.
- Sparse `missing_rate=0.5` and `0.95` each have sparse and oracle persistence references.
- Positive improvement means model RMSE is lower than the chosen persistence reference; negative means persistence is better.
- PathIoU values are extremely small across conditions and should be interpreted as a diagnostic, not a success claim.
