# ARC versus OAT: LIBERO, September 26 morning

Verified artifact snapshot: 2026-09-26T16:13:53.325224+00:00 (09:13 Pacific). 40/60 requested cells have complete evaluations.

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
| Plain DP / U-Net | Pending | Pending | 85.68% | Pending |
| Plain DP / OAT-release | Pending | Pending | Pending | Pending |

All completed cells use 2500 episodes from one trained checkpoint, with five evaluation repetitions. All 35 completed ARC comparisons pass matching starting-state checks against OAT. The eleven earlier unpaired results have been superseded.

* OAT Object (9/2500) and ARC OAT-DP STK2 Object (0/2500) are complete but unusually low. Their causes remain unresolved; do not infer a settled representation advantage from Object.

Native OAT uses its learned tokenizer and autoregressive policy. ARC U-Net uses the original conditional U-Net; ARC OAT-DP uses the released diffusion Transformer. Plain DP predicts raw actions without either tokenizer. Pending cells have no verified full-evaluation SR.

STK1: R192/D1.6/M36. STK2: R192/D0.8/M32. DUR1: R192/D1.6/M24. DUR2: R384/D0.8/M32. Shared STK and DUR: R384/D1.6/M36.

[Heatmap (PNG)](arc_vs_oat_libero_20260926_morning.png) · [Vector plot (SVG)](arc_vs_oat_libero_20260926_morning.svg) · [CSV](arc_vs_oat_libero_20260926_morning.csv) · [Complete source audit](arc_vs_oat_libero_20260926_morning.json)

## First complete plain-DP comparison

[Goal / U-Net, all 2500 matching trials](arc_vs_plain_dp_libero_20260926_morning.md): plain DP **85.68%**; ARC STK1 **82.20%**, STK2 **84.64%**, DUR1 **89.56%**, DUR2 **87.28%**, shared STK **84.84%**, shared DUR **88.96%**. All DUR configurations lead the baseline; all STK configurations trail it. This is one full-suite comparison using one training seed.

The same audit has 2270 matching Spatial/OAT-DP trials: plain DP **67.31%** versus ARC STK1 **76.56%** (+9.25 points). That comparison remains partial and task coverage is uneven. The full table above contains only completed evaluations.

## Overnight interruption and remaining work

Only one additional final score completed overnight, bringing the table from 39 to 40 cells. Twenty other owned jobs were canceled by cluster GPU quota enforcement around 23:00 Pacific on September 25, including jobs still queued. They did not keep training overnight. All eight raw-DP models have completed full training, but only the U-Net Goal evaluation is complete. Thirteen ARC/OAT-DP cells remain: eleven models need more training, and two shared LIBERO-10 models need only evaluation.

The retained baseline trial counts before this morning's recovery are:

| Suite | U-Net | OAT-release DP |
| --- | ---: | ---: |
| Spatial | 292/2500 | 2270/2500 |
| Object | 1971/2500 | 954/2500 |
| Goal | **2500/2500 complete** | 524/2500 |
| LIBERO-10 | 1603/2500 | 346/2500 |

Both shared ARC/OAT-DP LIBERO-10 checkpoints have completed 5001 epochs and 605121 optimizer/EMA updates. Shared STK retains 1727 evaluation trials and shared DUR 977. New standalone evaluations use one L40S each, releasing the eight-GPU training allocation their previous inline evaluation jobs used. Recovery validates the saved checkpoint, protocol, trial identities, hashes and videos before skipping completed episodes. Older inline ARC runs reconstruct their immutable request from checksummed training provenance.

All training and evaluation budgets remain unchanged: 5001 epochs, global batch 1024 and 2500 evaluation trials. Recovery uses L40S at NORMAL cluster priority. Further queueing and quota preemption remain completion risks; no completion time is promised. LIBERO-90 and ARC+OAT remain deferred. The September 25 morning deadline was missed.

Twenty recovery workflows were accepted at NORMAL priority in `groot-l40s-03`: seven plain-DP evaluations, two shared ARC LIBERO-10 evaluations, and eleven ARC training resumes. The scheduler snapshot at 2026-09-26T16:33:52.908048+00:00 reports {'PENDING': 13, 'RUNNING': 7}. A running scheduler status includes startup; uploaded records separately verify new baseline trials beyond restored counts. [Recovery, scheduler, checkpoint and trial evidence](libero_progress_20260926_morning.json).

The new CPU collector `libero-table-morning-20260926-1` published a SHA-verified 60-cell report at 09:34 Pacific, following all twenty accepted recovery IDs. The old evening collector was retired only after that check. Live snapshots publish every five minutes under `s3://rldb/experiments/arc-oat-20260919/campaigns/libero-table-morning-20260926/`.
