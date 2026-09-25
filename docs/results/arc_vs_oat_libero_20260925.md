# ARC versus OAT: LIBERO, September 25

Verified artifact snapshot: 2026-09-25T20:02:07.143558+00:00 (13:02 Pacific). 35/60 requested cells have complete evaluations; no additional full scores since the morning snapshot.

| Method | Spatial | Object | Goal | LIBERO-10 |
| --- | ---: | ---: | ---: | ---: |
| Native OAT | 61.68% | 0.36%* | 75.52% | 59.44% |
| ARC U-Net / STK1 | 67.32% | 52.80% | 82.20% | 51.88% |
| ARC U-Net / STK2 | 68.36% | 52.36% | 84.64% | 52.96% |
| ARC U-Net / DUR1 | 66.32% | 53.64% | 89.56% | 55.16% |
| ARC U-Net / DUR2 | 68.32% | 63.72% | 87.28% | 55.44% |
| ARC U-Net / shared STK | 66.36% | 58.88% | 84.84% | 57.16% |
| ARC U-Net / shared DUR | 68.08% | 59.64% | 88.96% | 53.52% |
| ARC OAT-DP / STK1 | Pending | Pending | 87.00% | 67.88% |
| ARC OAT-DP / STK2 | Pending | 0.00%* | 81.32% | 70.12% |
| ARC OAT-DP / DUR1 | 73.76% | Pending | Pending | 68.76% |
| ARC OAT-DP / DUR2 | Pending | Pending | Pending | Pending |
| ARC OAT-DP / shared STK | Pending | Pending | Pending | Pending |
| ARC OAT-DP / shared DUR | Pending | Pending | Pending | Pending |
| Plain DP / U-Net | Pending | Pending | Pending | Pending |
| Plain DP / OAT-release | Pending | Pending | Pending | Pending |

All completed cells use 2500 episodes from one trained checkpoint, with five evaluation repetitions. All 31 completed ARC comparisons pass matching starting-state checks against OAT. The eleven earlier unpaired results have been superseded.

* OAT Object (9/2500) and ARC OAT-DP STK2 Object (0/2500) are complete but unusually low. Their causes remain unresolved; do not infer a settled representation advantage from Object.

Native OAT uses its learned tokenizer and autoregressive policy. ARC U-Net uses the original conditional U-Net; ARC OAT-DP uses the released diffusion Transformer. Plain DP predicts raw actions without either tokenizer. Pending cells have no verified full-evaluation SR.

STK1: R192/D1.6/M36. STK2: R192/D0.8/M32. DUR1: R192/D1.6/M24. DUR2: R384/D0.8/M32. Shared STK and DUR: R384/D1.6/M36.

[Heatmap (PNG)](arc_vs_oat_libero_20260925_morning.png) · [Vector plot (SVG)](arc_vs_oat_libero_20260925_morning.svg) · [CSV](arc_vs_oat_libero_20260925_morning.csv) · [Latest complete source audit](arc_vs_oat_libero_20260925_afternoon.json)

## Work still in progress

Evaluation episode counts are from the 13:02 Pacific collector snapshot. Training counters and scheduler state were refreshed at 2026-09-25T20:03:52.503302+00:00. These episode counts are progress, not success rates.

| Suite | Plain DP / U-Net | Plain DP / OAT-release |
| --- | --- | --- |
| spatial | Evaluation replacement running; no recorded episodes yet | Evaluating: 1668/2500 episodes |
| object | Evaluating: 1437/2500 episodes | Evaluating: 477/2500 episodes |
| goal | Evaluating: 1824/2500 episodes | Recovery queued (1680/280056 updates saved) |
| 10 | Training: 469310/605121 updates (77.6%) | Training: 592380/605121 updates (97.9%) |

Original ARC/U-Net is complete on all 24 cells, and native OAT on all four. ARC with OAT's diffusion Transformer is complete on seven of 24 cells; 10 are training, 1 is evaluating and 6 are queued.

The U-Net Spatial evaluation failed to start after training completed. Its final checkpoint was downloaded, SHA-256 verified and confirmed to contain all 5001 epochs / 270054 optimizer and EMA updates. A separate one-L40S evaluation, `dp-eval-r1-20260925-unet-spatial-1`, now runs that exact checkpoint. The Goal U-Net evaluation recovery and OAT-DP Object training recovery are also evaluating.

The current CPU collector is `libero-table-afternoon-20260925-1`, publishing every five minutes under `s3://rldb/experiments/arc-oat-20260919/campaigns/libero-table-afternoon-20260925/`. It follows all accepted recovery IDs. The previous collector was retired only after the successor published a verified 60-cell report.

All current jobs use L40S. Training budgets remain 5001 epochs / global batch 1024; complete evaluations require 2500 episodes. LIBERO-90 and ARC+OAT remain deferred. The September 25 morning completion target was missed.

[Progress, scheduler and recovery evidence](libero_progress_20260925_afternoon.json)
