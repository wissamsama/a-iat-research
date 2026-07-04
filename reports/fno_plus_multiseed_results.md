# FNO+ Official-v1 Multi-Seed Cross-Validation

Three independent 100-epoch trainings of the official-v1 normalized FNO+
reproduction, identical config except `training.seed`, evaluated with the
official pooled t=2..20 test protocol.

## Runs

| Seed | Run directory | Best checkpoint epoch |
|---:|---|---:|
| 42 | `28-06-2026_15-59-18_fcb_fno_plus_official_v1_normalized_100epoch_highfid_60m` | 71 |
| 7 | `03-07-2026_17-43-06_fcb_fno_plus_official_v1_normalized_highfid_60m` | 80 |
| 123 | `03-07-2026_22-38-40_fcb_fno_plus_official_v1_normalized_seed123_highfid_60m` | 75 |

## Test-split pooled metrics

| Seed | relRMSE | NSE | Pearson r | CSI@0.001 | CSI@0.01 |
|---:|---:|---:|---:|---:|---:|
| 42 | 0.006694 | 0.999939 | 0.999971 | 0.909009 | 0.993807 |
| 7 | 0.006426 | 0.999944 | 0.999972 | 0.810893 | 0.996355 |
| 123 | 0.006530 | 0.999942 | 0.999972 | 0.887944 | 0.995453 |
| **mean ± std** | **0.006550 ± 0.000135** | 0.999941 ± 0.000002 | 0.999972 ± 0.000000 | **0.869282 ± 0.051652** | 0.995205 ± 0.001292 |

## Does the gap to official Table 4 survive seed noise?

Official Table 4 FNO+ relRMSE: **0.003941**.

Seed-to-seed standard deviation on relRMSE (0.000135) is only **~5% of the
gap** between this repo's mean (0.006550) and the published value
(gap = 0.002609). The ~1.66x reproduction gap is real and well outside
seed-to-seed noise for this metric — it reflects a genuine difference
(most likely unpublished preprocessing/training details, not run-to-run
variance), not a single unlucky training.

CSI@0.001 behaves differently: its seed-to-seed spread (std=0.0517, range
0.81-0.91) is much larger relative to its own scale than relRMSE's — flood-
boundary classification at the shallowest threshold is markedly more
seed-sensitive than the continuous depth error. Any single-seed CSI@0.001
number in this repo's other reports should be read with that variability in
mind.

## Does Not Claim

- Not an explanation of *why* the gap exists (preprocessing, seed count, or
  protocol details beyond what's already audited in
  `reports/fno_plus_dashboard_audit.md`).
- Not a claim that 3 seeds fully characterize the true variance — it rules
  out "one unlucky run" as the explanation for the relRMSE gap, nothing more.
