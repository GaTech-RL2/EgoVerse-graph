# ARC versus OAT: LIBERO, September 25

Verified artifact snapshot: 2026-09-26T04:56:16.893402+00:00 (21:56 Pacific). 39/60 requested cells have complete evaluations.

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
| ARC OAT-DP / DUR1 | 73.76% | Pending | Pending | 68.76% |
| ARC OAT-DP / DUR2 | Pending | Pending | 83.48% | 71.68% |
| ARC OAT-DP / shared STK | Pending | Pending | Pending | Pending |
| ARC OAT-DP / shared DUR | Pending | Pending | Pending | Pending |
| Plain DP / U-Net | Pending | Pending | Pending | Pending |
| Plain DP / OAT-release | Pending | Pending | Pending | Pending |

All completed cells use 2500 episodes from one trained checkpoint, with five evaluation repetitions. All 35 completed ARC comparisons pass matching starting-state checks against OAT. The eleven earlier unpaired results have been superseded.

* OAT Object (9/2500) and ARC OAT-DP STK2 Object (0/2500) are complete but unusually low. Their causes remain unresolved; do not infer a settled representation advantage from Object.

Native OAT uses its learned tokenizer and autoregressive policy. ARC U-Net uses the original conditional U-Net; ARC OAT-DP uses the released diffusion Transformer. Plain DP predicts raw actions without either tokenizer. Pending cells have no verified full-evaluation SR.

STK1: R192/D1.6/M36. STK2: R192/D0.8/M32. DUR1: R192/D1.6/M24. DUR2: R384/D0.8/M32. Shared STK and DUR: R384/D1.6/M36.

[Heatmap (PNG)](arc_vs_oat_libero_20260925_evening.png) · [Vector plot (SVG)](arc_vs_oat_libero_20260925_evening.svg) · [CSV](arc_vs_oat_libero_20260925_evening.csv) · [Complete source audit](arc_vs_oat_libero_20260925_evening.json)

## Plain DP comparison and recovery

[Matched preliminary ARC versus plain-DP comparisons](arc_vs_plain_dp_libero_20260925_evening.md) are available separately. They compare exact shared trials within each backbone and do not fill unfinished cells above.

All eight plain-DP baselines have finished the full 5001-epoch budget. Their final checkpoints were downloaded and verified for SHA-256, optimizer updates and EMA updates. Seven evaluations have saved trials; OAT-DP Goal finished training before its evaluation could begin. All eight baseline jobs were interrupted by cluster GPU quota enforcement.

The saved baseline trials at the 22:02 Pacific audit are:

| Suite | U-Net | OAT-release DP |
| --- | ---: | ---: |
| Spatial | 163/2500 | 2094/2500 |
| Object | 1683/2500 | 787/2500 |
| Goal | 2418/2500 | 0/2500 |
| LIBERO-10 | 1372/2500 | 180/2500 |

Eight evaluation-only replacements were submitted on L40S. Seven resume saved trials and run only missing episodes, after validating the immutable checkpoint, exact protocol and artifact hashes. The first seven resume attempts restored the records but failed at a command-line output-directory guard before running new trials. That guard was fixed in `d8213f9c`, tested through the command-line entry point, and those seven attempts were replaced under `dp-eval-r3-20260926-*`. OAT-DP Goal uses `dp-eval-r2-20260926-oat-goal` and starts its evaluation fresh. Restored trials are never counted twice.

Eleven interrupted ARC/OAT-DP training runs were also resubmitted under `arc-dpr2-20260926-*` from verified checkpoints on four L40S each, retaining optimizer/EMA state, global batch 1024 and the original total update budget. Two ongoing shared-configuration LIBERO-10 evaluations were left running. No other agents' workflows were changed.

All campaign jobs use L40S at normal queue priority. LIBERO-90 and ARC+OAT remain deferred. The September 25 morning completion target was missed. Queueing and further preemption remain unresolved completion risks.

The successor CPU collector `libero-table-evening-20260926-1` published a SHA-verified 60-cell report at 22:27 Pacific, following all accepted recovery IDs. Only then was the previous CPU collector retired. It publishes every five minutes under `s3://rldb/experiments/arc-oat-20260919/campaigns/libero-table-evening-20260926/`.

At 22:27 Pacific, all eight baseline evaluation replacements were running; two ARC training replacements were running and nine queued. Running status includes startup and is not itself evidence of new optimizer updates or episodes. [Scheduler, recovery and final-checkpoint evidence](libero_progress_20260925_evening.json).

At 22:29 Pacific, GPU rollouts had advanced beyond restored trials in U-Net Goal (2418→2421), U-Net LIBERO-10 (1372→1380), OAT-DP LIBERO-10 (180→186), and OAT-DP Spatial (2094→2095). The other three resumed baselines had restored their saved counts; OAT-DP Goal had 124 new trials. These progress counts do not update the frozen 22:02 paired comparison above.
