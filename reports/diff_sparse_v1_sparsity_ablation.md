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

## Acceptance Table

Pending long-running train/eval jobs.

| Split | Missing rate | Model RMSE norm | Sparse persistence RMSE norm | Oracle persistence RMSE norm | Model MAE norm | Sparse persistence MAE norm | Oracle persistence MAE norm | Eval directory |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| val | 0.0 | pending | pending | pending | pending | pending | pending | pending |
| test | 0.0 | pending | pending | pending | pending | pending | pending | pending |
| val | 0.5 | pending | pending | pending | pending | pending | pending | pending |
| test | 0.5 | pending | pending | pending | pending | pending | pending | pending |
| val | 0.95 | pending | pending | pending | pending | pending | pending | pending |
| test | 0.95 | pending | pending | pending | pending | pending | pending | pending |

## Notes

- The existing dense 40-epoch checkpoint loses to oracle persistence at
  missing_rate=0.0 through h20.
- Sparse-sensor evaluation is the relevant DIFF-SPARSE framing; dense
  missing_rate=0.0 is a difficult short-lead comparison against a very strong
  copy-last-frame baseline.
