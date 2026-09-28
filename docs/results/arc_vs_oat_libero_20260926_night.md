# ARC versus OAT: LIBERO, September 26 night

Verified artifact snapshot: 2026-09-27T04:18:54.878961+00:00 (21:18 Pacific). 51/60 requested cells have complete evaluations.

| Method | Spatial | Object | Goal | LIBERO-10 |
| --- | ---: | ---: | ---: | ---: |
| Native OAT | 61.68% | 0.36%* | 75.52% | 59.44% |
| ARC U-Net / STK1 | 67.32% | 52.80% | 82.20% | 51.88% |
| ARC U-Net / STK2 | 68.36% | 52.36% | 84.64% | 52.96% |
| ARC U-Net / DUR1 | 66.32% | 53.64% | 89.56% | 55.16% |
| ARC U-Net / DUR2 | 68.32% | 63.72% | 87.28% | 55.44% |
| ARC U-Net / shared STK | 66.36% | 58.88% | 84.84% | 57.16% |
| ARC U-Net / shared DUR | 68.08% | 59.64% | 88.96% | 53.52% |
| ARC OAT-DP / STK1 | 76.16% | Pending | 87.00% | 67.88% |
| ARC OAT-DP / STK2 | 76.16% | 0.00%* | 81.32% | 70.12% |
| ARC OAT-DP / DUR1 | 73.76% | Pending | 82.68% | 68.76% |
| ARC OAT-DP / DUR2 | Pending | Pending | 83.48% | 71.68% |
| ARC OAT-DP / shared STK | 74.88% | Pending | Pending | 67.64% |
| ARC OAT-DP / shared DUR | Pending | Pending | Pending | 68.56%† |
| Plain DP / U-Net | 61.44% | 52.88% | 85.68% | 50.72% |
| Plain DP / OAT-release | 64.52% | 0.00%* | 83.12% | 60.52% |

All completed cells use 2500 episodes from one trained checkpoint, with five evaluation repetitions. 38 of 39 completed ARC cells pass matching starting-state checks against OAT. All eight plain-DP cells pass the same data, trial and starting-state checks. The original 60-cell campaign has 51 full evaluations; FAST adds four pending cells, giving 51/64 including FAST.

* OAT Object (9/2500), ARC OAT-DP STK2 Object (0/2500), and plain OAT-DP Object (0/2500) are complete but unusually low. Their causes remain unresolved.

† ARC OAT-DP shared DUR / LIBERO-10 completed all 2500 trials, but 29 starting-state hashes differ from OAT. Its 68.56% is flagged and excluded from paired claims. A fresh full evaluation of the same checkpoint is running in `arc-eval-paired-20260927-shared-dur-10-1`.

Native OAT uses its learned tokenizer and autoregressive policy. ARC U-Net uses the original conditional U-Net; ARC OAT-DP uses the released diffusion Transformer. Plain DP predicts raw actions without either tokenizer. Pending cells have no verified full-evaluation SR.

STK1: R192/D1.6/M36. STK2: R192/D0.8/M32. DUR1: R192/D1.6/M24. DUR2: R384/D0.8/M32. Shared STK and DUR: R384/D1.6/M36.

[Heatmap (PNG)](arc_vs_oat_libero_20260926_night.png) · [Vector plot (SVG)](arc_vs_oat_libero_20260926_night.svg) · [CSV](arc_vs_oat_libero_20260926_night.csv) · [Complete source audit](arc_vs_oat_libero_20260926_night.json)

FAST has no full success-rate scores yet. [Current training and recovery progress](libero_progress_20260926_night.md).
