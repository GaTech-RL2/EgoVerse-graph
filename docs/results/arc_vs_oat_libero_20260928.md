# ARC, OAT, FAST and plain DP: LIBERO results, September 28

Verified artifact snapshot: 2026-09-28T16:09:30.995417+00:00 (September 28, 09:09 Pacific). **57/60 original results are complete, or 57/64 including FAST.** All 57 completed cells have the full evaluation budget and pass the applicable protocol, data, episode-identity and starting-state checks.

Success rate (%). Every number uses 10 tasks × 50 trials × 5 evaluation repetitions = 2500 episodes from one training seed. Pending cells have no full-suite score.

| Method | Spatial | Object | Goal | LIBERO-10 |
| --- | ---: | ---: | ---: | ---: |
| Native OAT | 61.68 | 0.36* | 75.52 | 59.44 |
| Native FAST | Pending | Pending | Pending | Pending |
| Plain DP / U-Net | 61.44 | 52.88 | 85.68 | 50.72 |
| ARC / U-Net / STK1 | 67.32 | 52.80 | 82.20 | 51.88 |
| ARC / U-Net / STK2 | 68.36 | 52.36 | 84.64 | 52.96 |
| ARC / U-Net / DUR1 | 66.32 | 53.64 | 89.56 | 55.16 |
| ARC / U-Net / DUR2 | 68.32 | 63.72 | 87.28 | 55.44 |
| ARC / U-Net / shared STK | 66.36 | 58.88 | 84.84 | 57.16 |
| ARC / U-Net / shared DUR | 68.08 | 59.64 | 88.96 | 53.52 |
| Plain DP / OAT-DP | 64.52 | 0.00* | 83.12 | 60.52 |
| ARC / OAT-DP / STK1 | 76.16 | Pending | 87.00 | 67.88 |
| ARC / OAT-DP / STK2 | 76.16 | 0.00* | 81.32 | 70.12 |
| ARC / OAT-DP / DUR1 | 73.76 | 0.00* | 82.68 | 68.76 |
| ARC / OAT-DP / DUR2 | 75.84 | 0.00* | 83.48 | 71.68 |
| ARC / OAT-DP / shared STK | 74.88 | Pending | 85.00 | 67.64 |
| ARC / OAT-DP / shared DUR | 74.76 | Pending | 83.28 | 68.56 |

* Object: OAT 0.36%, the completed ARC/OAT-DP Object scores 0.00%, and plain OAT-DP Object 0.00% remain anomalous. These are completed evaluations; their cause is unresolved. Do not interpret them as a settled tokenizer advantage.

The shared DUR / OAT-DP / LIBERO-10 rerun completed at 68.56% and now passes starting-state matching (zero mismatches). It replaces the previously flagged evaluation of the same checkpoint.

New full ARC/OAT-DP scores since the September 26 night snapshot: DUR2 Spatial 75.84; shared DUR Spatial 74.76; DUR1 Object 0.00; DUR2 Object 0.00; shared STK Goal 85.00; shared DUR Goal 83.28.

All original ARC/OAT/plain-DP policy training has finished. Three ARC/OAT-DP Object evaluations were interrupted by cluster GPU quota enforcement: STK1 saved 2257/2500 trials, shared STK 1933/2500, shared DUR 1792/2500. FAST Goal completed all 280056 optimizer/EMA updates and saved 542/2500 evaluation trials before quota interruption. FAST Spatial, Object and LIBERO-10 training were interrupted at approximately 56.6%, 48.0%, and 61.8% of their budgets. Partial evaluations are not included as full-suite SR.

Policy training retains 5001 epochs, global batch 1024, seed 42, and EMA. Evaluation observes two frames, predicts 32 actions, executes 16 before replanning, and caps episodes at 550 steps. Native OAT and FAST use autoregressive policies; OAT-DP denotes the released diffusion Transformer. LIBERO-90 and the ARC+OAT tokenizer hybrid remain deferred.

ARC configurations are fixed across the four suites: STK1 R192/D1.6/M36; STK2 R192/D0.8/M32; DUR1 R192/D1.6/M24; DUR2 R384/D0.8/M32; shared STK/DUR R384/D1.6/M36. One training seed does not establish variation across training seeds; selecting the best configuration separately per suite is a sweep result.

[Heatmap (PNG)](arc_vs_oat_libero_20260928.png) · [Vector plot (SVG)](arc_vs_oat_libero_20260928.svg) · [Full table CSV](arc_vs_oat_libero_20260928.csv) · [Per-cell audit CSV](arc_vs_oat_libero_20260928_cells.csv) · [Per-task scores CSV](arc_vs_oat_libero_20260928_per_task.csv) · [Complete source audit](arc_vs_oat_libero_20260928.json)

[Accepted recoveries and durable training progress](libero_progress_20260928.md).
