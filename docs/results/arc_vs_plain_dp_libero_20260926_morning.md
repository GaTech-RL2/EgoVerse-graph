# ARC versus plain DP: September 26 morning

Verified trial audit: 2026-09-26T16:18:35.595146+00:00 (09:18 Pacific).

Goal / U-Net now has a full baseline evaluation: 2142/2500 successes, or **85.68%**. All six ARC configurations below also have 2500 trials, with matching task, repetition, trial, seed and initial-state identities. Each model completed its full 5001-epoch training budget. Five evaluation repetitions use one trained checkpoint, not five training seeds.

| Goal / U-Net | Success rate | Difference from plain DP |
| --- | ---: | ---: |
| Plain DP | 85.68% | — |
| ARC STK1 | 82.20% | -3.48 pp |
| ARC STK2 | 84.64% | -1.04 pp |
| ARC DUR1 | 89.56% | +3.88 pp |
| ARC DUR2 | 87.28% | +1.60 pp |
| ARC shared STK | 84.84% | -0.84 pp |
| ARC shared DUR | 88.96% | +3.28 pp |

All DUR variants beat the plain-DP baseline on Goal. All STK variants are below it. This is one full-suite comparison with one training seed; it does not establish an advantage across every suite or backbone.

## Remaining comparisons are partial

Spatial / OAT-DP now has 2270 shared trials: plain DP 67.31%, ARC STK1 76.56% (+9.25 points). These evaluations progress in task order, so partial percentages are not final suite scores and should not replace the unfinished cells in the full table.

| Suite | Backbone | ARC | Trials | Plain DP | ARC | Difference | Both evaluations complete |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| libero_spatial | unet | STK1 | 292 | 26.37% | 62.33% | +35.96 pp | No |
| libero_object | unet | STK1 | 1971 | 51.04% | 54.03% | +2.99 pp | No |
| libero_goal | unet | STK1 | 2500 | 85.68% | 82.20% | -3.48 pp | Yes |
| libero_spatial | unet | STK2 | 292 | 26.37% | 58.56% | +32.19 pp | No |
| libero_object | unet | STK2 | 1971 | 51.04% | 53.32% | +2.28 pp | No |
| libero_goal | unet | STK2 | 2500 | 85.68% | 84.64% | -1.04 pp | Yes |
| libero_10 | unet | STK2 | 1603 | 56.83% | 59.39% | +2.56 pp | No |
| libero_spatial | unet | DUR1 | 292 | 26.37% | 50.34% | +23.97 pp | No |
| libero_object | unet | DUR1 | 1971 | 51.04% | 53.98% | +2.94 pp | No |
| libero_goal | unet | DUR1 | 2500 | 85.68% | 89.56% | +3.88 pp | Yes |
| libero_10 | unet | DUR1 | 1603 | 56.83% | 64.82% | +7.99 pp | No |
| libero_spatial | unet | DUR2 | 292 | 26.37% | 53.42% | +27.05 pp | No |
| libero_object | unet | DUR2 | 1971 | 51.04% | 67.22% | +16.18 pp | No |
| libero_goal | unet | DUR2 | 2500 | 85.68% | 87.28% | +1.60 pp | Yes |
| libero_10 | unet | DUR2 | 1603 | 56.83% | 64.94% | +8.11 pp | No |
| libero_spatial | unet | shared STK | 292 | 26.37% | 57.88% | +31.51 pp | No |
| libero_object | unet | shared STK | 1971 | 51.04% | 58.95% | +7.91 pp | No |
| libero_goal | unet | shared STK | 2500 | 85.68% | 84.84% | -0.84 pp | Yes |
| libero_10 | unet | shared STK | 1603 | 56.83% | 63.76% | +6.92 pp | No |
| libero_spatial | unet | shared DUR | 292 | 26.37% | 65.07% | +38.70 pp | No |
| libero_object | unet | shared DUR | 1971 | 51.04% | 62.71% | +11.67 pp | No |
| libero_goal | unet | shared DUR | 2500 | 85.68% | 88.96% | +3.28 pp | Yes |
| libero_10 | unet | shared DUR | 1603 | 56.83% | 61.26% | +4.43 pp | No |
| libero_10 | unet | STK1 | 1603 | 56.83% | 57.64% | +0.81 pp | No |
| libero_spatial | oat_dp | STK1 | 2270 | 67.31% | 76.56% | +9.25 pp | No |
| libero_goal | oat_dp | STK1 | 524 | 94.85% | 92.94% | -1.91 pp | No |
| libero_10 | oat_dp | STK1 | 346 | 28.90% | 18.79% | -10.12 pp | No |
| libero_spatial | oat_dp | STK2 | 2270 | 67.31% | 75.99% | +8.68 pp | No |
| libero_object | oat_dp | STK2 | 954 | 0.00% | 0.00% | +0.00 pp | No |
| libero_goal | oat_dp | STK2 | 524 | 94.85% | 94.66% | -0.19 pp | No |
| libero_10 | oat_dp | STK2 | 346 | 28.90% | 29.48% | +0.58 pp | No |
| libero_spatial | oat_dp | DUR1 | 2270 | 67.31% | 74.54% | +7.22 pp | No |
| libero_10 | oat_dp | DUR1 | 346 | 28.90% | 23.41% | -5.49 pp | No |
| libero_goal | oat_dp | DUR2 | 524 | 94.85% | 92.75% | -2.10 pp | No |
| libero_10 | oat_dp | DUR2 | 346 | 28.90% | 27.17% | -1.73 pp | No |
| libero_10 | oat_dp | shared STK | 346 | 28.90% | 24.57% | -4.34 pp | No |
| libero_10 | oat_dp | shared DUR | 346 | 28.90% | 22.54% | -6.36 pp | No |

Object anomalies remain unresolved. No aggregate across incomplete suites is reported.

[CSV](arc_vs_plain_dp_libero_20260926_morning.csv) · [Source audit, per-task counts and artifact hashes](arc_vs_plain_dp_libero_20260926_morning.json) · [Full table](arc_vs_oat_libero_20260925.md)
