# FNO+ Pipeline & Dashboard Audit — 2026-07-03

Scope: the official-v1 normalized FNO+ pipeline (the repo's most faithful
FloodCastBench Table 4 FNO+ reproduction attempt) and the scientific metric
dashboard `experiments/FloodCastBench/fno_plus_metric_dashboard_scientific.html`,
audited against the FloodCastBench paper (Xu et al., Scientific Data 2025,
doi:10.1038/s41597-025-04725-2).

## Headline result: the Table 4 "RMSE" definition is now PROVEN

The paper's eq. (2) is ambiguous as written, and this repo's earlier reports
(`fno_plus_metric_diagnosis.md` §5, §10.5) flagged the metric definition as the
main open reproduction question. It is now resolved:

**Table 4 "RMSE" = pooled `sqrt(SSE / sum(y^2))` — exactly this repo's
`current_relative_rmse`.**

Proof: `NSE = 1 - SSE/TotVar` and `relRMSE^2 = SSE/sum(y^2)` imply
`NSE = 1 - relRMSE^2 / (TotVar/sum(y^2))`. The dataset ratio
`TotVar/sum(y^2)` for the Australia-60m test targets is a fixed data property,
computable from any completed run's paired (relRMSE, NSE):

| Source run | implied TotVar/sum(y^2) |
|---|---|
| v0 internal run `27-06-2026_14-00-18` | 0.731329 |
| official-v1 run `28-06-2026_15-59-18` | 0.731329 |

With ratio 0.731329, the published Table 4 (RMSE, NSE) pairs are reproduced
for **all three** official models:

| Model | Table 4 RMSE | NSE predicted from RMSE | NSE published | diff |
|---|---:|---:|---:|---:|
| U-Net | 0.566626 | 0.560984 | 0.560984 | 2e-7 |
| FNO   | 0.004258 | 0.999975 | 0.999975 | 2e-7 |
| FNO+  | 0.003941 | 0.999979 | 0.999979 | 2e-7 |

Corollaries:
- The per-pixel reading of eq. (2) (`mean((y-p)^2/y^2)`) is definitively wrong
  — it explodes on dry cells (repo recompute measured 90.0 for a model with
  relRMSE 0.012; see `reports/fno_plus_official_metric_recompute_internal_best.json`).
- The honest same-definition comparison is:
  **official FNO+ 0.003941 vs this repo's official-v1 best 0.006694**
  (pooled t=2..20 test protocol) — a real ×1.70 gap, same metric.

## Pipeline conformity vs the paper (verified)

| Paper spec | Repo implementation | Verdict |
|---|---|---|
| One-shot space-time FNO, inputs (X,Y,T,h1[,DEM,rain]), output t=2..20 (Fig 6a) | `models/fno_plus_official.py` `FNOPlusOfficial3d`, 6 input channels, 3D rfftn spectral conv, output 19 steps | conform |
| 4 Fourier layers, 12 modes, width 20 | identical | conform |
| Adam β=(0.9, 0.999), weight decay 1e-4 | identical | conform |
| Cosine LR 0.001 → 0, 100 epochs, batch size 1 | identical | conform |
| Australia 60m splits: 116/14/14 non-overlapping 20-frame windows | identical | conform |
| Metrics eqs. 2-5 (relRMSE, NSE, r, CSI@{0.001,0.01}) | pooled accumulators, physical units after inverse transform | conform (relRMSE proven above) |
| Preprocessing/normalization | train-only standardization of h1/DEM/rain/target (X,Y,T untouched) | **unknown in paper** — repo choice; empirically moved CSI@0.001 from 0.724 (v0 raw) to 0.909 (v1 normalized), toward official 0.9396 |
| Metric aggregation granularity | pooled over all pixels/steps/samples | consistent with the RMSE/NSE proof (pooled), CSI granularity unverifiable |

Long-horizon rollout tool
(`tools/evaluate_floodcastbench_fno_plus_official_v1_long_horizon_rollout.py`):
protocol is sound — physical-space autoregressive rollout, last predicted map
re-broadcast as next initial depth, rainfall/DEM/coords rebuilt at the advanced
frame index, no clipping/post-processing, valid-start bookkeeping per horizon,
per-step CSVs dense over steps 1..216. Values are consistent across its three
output variants (consistent_v1 / dense_v1 / dense_v2 agree at shared horizons).

## Dashboard defects found (2026-06-29 hand-written version) and fixes

1. **Official 0.003941 plotted on the "Classical RMSE" (meters) curve.**
   Provably the wrong metric family (see proof). → Moved to the relative-RMSE
   metric; deliberately no official reference on classical RMSE.
2. **Official Table 4 values pinned at the per-step T+20 x-position.** They are
   pooled t=2..20 aggregates, not a step-20 error. → Drawn as reference lines
   spanning steps 1..19, next to the repo's own same-protocol pooled values
   (`test_metrics_checkpoint_best_official_v1_normalized.json`), so the
   comparison is same-protocol. The repo's pooled values were previously absent
   from the dashboard entirely.
3. **X-axis sub-labels claimed "T+N = N hours".** Steps are 300 s; T+216 = 18 h,
   not 216 h (×12 error). → Numeric step axis with correct wall-clock labels.
4. **The "T+20" tick was actually rollout step 19** (the tool labels it
   `T+20_paper_t20_direct`) while T+5/T+10 were true steps 5/10. → Dense
   numeric axis; step 19 annotated as the paper's t=20 (`19*`).
5. **No generator script existed** (hand-written HTML, errors untraceable).
   → `scripts/build_fno_plus_metric_dashboard.py` rebuilds the dashboard from
   run artifacts; the dashboard now embeds its data lineage.
6. Sparse 12-point curves replaced by the dense per-step data (216 steps) that
   already existed in `long_horizon_metrics_per_step.csv`; per-step valid test
   start counts (14 → 3) surfaced in tooltips and the table.

Backup of the previous dashboard:
`experiments/FloodCastBench/fno_plus_metric_dashboard_scientific_pre_audit_backup.html`.
The older `fno_plus_metric_dashboard.html` (2026-06-29 19:40 variant) is left
untouched as a historical artifact.

## Honest current standing vs official FNO+ (same pooled protocol, test split)

| Metric | Official FNO+ | Official-v1 best (this repo) | Gap |
|---|---:|---:|---|
| relRMSE (Table 4 def.) | 0.003941 | 0.006694 | ×1.70 worse |
| NSE | 0.999979 | 0.999939 | worse |
| Pearson r | 0.999990 | 0.999971 | worse |
| CSI@0.001 | 0.939638 | 0.909009 | worse |
| CSI@0.01 | 0.984588 | 0.993807 | better |

Residual, non-resolvable-here uncertainties: official preprocessing, CSI
aggregation granularity, seeds. The remaining gap is real but the pipeline is
now metric-faithful and protocol-explicit; claiming "official reproduction"
remains off-limits (`does_not_claim` discipline).

## Regeneration

```bash
python scripts/build_fno_plus_metric_dashboard.py            # default run + output
python scripts/build_fno_plus_metric_dashboard.py --run-dir <run> --output <html>
```
