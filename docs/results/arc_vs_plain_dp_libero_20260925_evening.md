# ARC versus plain DP: matched preliminary LIBERO trials

Artifact audit: 2026-09-26T05:02:37.935838+00:00 (September 25, 22:02 Pacific).

Each comparison uses the same policy backbone and exact task/repetition/trial/seed identities, with matching simulator initial-state hashes. The plain DP policy predicts raw actions without ARC or OAT tokenization. All checkpoints have completed their full training budget. None of the baseline evaluations is complete in this snapshot.

Evaluations progress in task order. These success rates describe the completed trial subset, with unequal task coverage; they are not full-suite estimates or confidence intervals. ARC ranges below span the completed, preselected configurations on that same subset. Five evaluation repetitions use one trained checkpoint, not five training seeds.

| Suite | Backbone | Matched trials | Plain DP | ARC range | Delta range (pp) |
| --- | --- | ---: | ---: | ---: | ---: |
| libero_spatial | OAT-release DP | 2094 | 68.9% | 75.2–77.0% | +6.3 to +8.1 |
| libero_object | U-Net | 1683 | 46.8% | 49.4–65.5% | +2.6 to +18.7 |
| libero_goal | U-Net | 2418 | 85.6% | 82.2–89.7% | -3.4 to +4.1 |
| libero_10 | U-Net | 1372 | 57.1% | 57.7–65.5% | +0.7 to +8.5 |

Spatial/OAT-DP STK1 is ahead by 8.1 percentage points. Goal/U-Net is configuration-dependent: DUR1 is ahead by 4.1 points, whereas STK1 is behind by 3.4 points. Object/U-Net DUR2 is ahead by 18.7 points on the available subset. These results do not establish an advantage across all suites or both backbones.

Other subsets are less informative: U-Net Spatial has only 163 first-task trials; OAT-DP LIBERO-10 has only 180 first-task trials and mixed ARC deltas. OAT-DP Object has 0/787 successes for both its raw baseline and ARC STK2, alongside the unresolved native-OAT Object anomaly. OAT-DP Goal has no baseline trials in this snapshot.

## Every matched configuration

| Suite | Backbone | ARC configuration | Trials | Plain DP | ARC | Delta (pp) | ARC evaluation complete |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| libero_spatial | unet | STK1 | 163 | 18.40% | 61.96% | +43.56 | Yes |
| libero_object | unet | STK1 | 1683 | 46.76% | 50.21% | +3.45 | Yes |
| libero_goal | unet | STK1 | 2418 | 85.61% | 82.22% | -3.39 | Yes |
| libero_spatial | unet | STK2 | 163 | 18.40% | 57.06% | +38.65 | Yes |
| libero_object | unet | STK2 | 1683 | 46.76% | 49.38% | +2.61 | Yes |
| libero_goal | unet | STK2 | 2418 | 85.61% | 85.03% | -0.58 | Yes |
| libero_10 | unet | STK2 | 1372 | 57.07% | 60.20% | +3.13 | Yes |
| libero_spatial | unet | DUR1 | 163 | 18.40% | 49.69% | +31.29 | Yes |
| libero_object | unet | DUR1 | 1683 | 46.76% | 50.15% | +3.39 | Yes |
| libero_goal | unet | DUR1 | 2418 | 85.61% | 89.74% | +4.14 | Yes |
| libero_10 | unet | DUR1 | 1372 | 57.07% | 65.52% | +8.45 | Yes |
| libero_spatial | unet | DUR2 | 163 | 18.40% | 49.69% | +31.29 | Yes |
| libero_object | unet | DUR2 | 1683 | 46.76% | 65.48% | +18.72 | Yes |
| libero_goal | unet | DUR2 | 2418 | 85.61% | 87.22% | +1.61 | Yes |
| libero_10 | unet | DUR2 | 1372 | 57.07% | 65.45% | +8.38 | Yes |
| libero_spatial | unet | shared STK | 163 | 18.40% | 58.28% | +39.88 | Yes |
| libero_object | unet | shared STK | 1683 | 46.76% | 55.73% | +8.97 | Yes |
| libero_goal | unet | shared STK | 2418 | 85.61% | 85.44% | -0.17 | Yes |
| libero_10 | unet | shared STK | 1372 | 57.07% | 63.48% | +6.41 | Yes |
| libero_spatial | unet | shared DUR | 163 | 18.40% | 66.26% | +47.85 | Yes |
| libero_object | unet | shared DUR | 1683 | 46.76% | 60.31% | +13.55 | Yes |
| libero_goal | unet | shared DUR | 2418 | 85.61% | 89.00% | +3.39 | Yes |
| libero_10 | unet | shared DUR | 1372 | 57.07% | 62.32% | +5.25 | Yes |
| libero_10 | unet | STK1 | 1372 | 57.07% | 57.73% | +0.66 | Yes |
| libero_spatial | oat_dp | STK1 | 2094 | 68.91% | 77.03% | +8.12 | Yes |
| libero_10 | oat_dp | STK1 | 180 | 35.56% | 22.78% | -12.78 | Yes |
| libero_spatial | oat_dp | STK2 | 2094 | 68.91% | 76.55% | +7.64 | Yes |
| libero_object | oat_dp | STK2 | 787 | 0.00% | 0.00% | +0.00 | Yes |
| libero_10 | oat_dp | STK2 | 180 | 35.56% | 38.33% | +2.78 | Yes |
| libero_spatial | oat_dp | DUR1 | 2094 | 68.91% | 75.21% | +6.30 | Yes |
| libero_10 | oat_dp | DUR1 | 180 | 35.56% | 25.00% | -10.56 | Yes |
| libero_10 | oat_dp | DUR2 | 180 | 35.56% | 31.67% | -3.89 | Yes |
| libero_10 | oat_dp | shared STK | 180 | 35.56% | 31.11% | -4.44 | No |
| libero_10 | oat_dp | shared DUR | 180 | 35.56% | 25.00% | -10.56 | No |

Baseline interruption was verified as cluster GPU quota enforcement. Completed trial artifacts and videos are retained for evaluation recovery; no training-budget reduction or success filtering is used.

[CSV](arc_vs_plain_dp_libero_20260925_evening.csv) · [Full audit with per-task counts, run identities and artifact hashes](arc_vs_plain_dp_libero_20260925_evening.json) · [Full-evaluation table](arc_vs_oat_libero_20260925.md)
