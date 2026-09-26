# ARC versus OAT: LIBERO Spatial

Snapshot: 2026-09-24T06:41:15.095049+00:00.

All seven policies completed 2,500 episodes each: ten tasks, 50 trials per task, five evaluation repetitions. The highest observed ARC score is **69.24% for DUR2**, versus **61.68% for OAT**, a raw difference of **7.56 percentage points**.

**The full-suite paired comparison remains provisional.** ARC and OAT starting-state hashes differ in 200 episodes, all from the top-drawer task in repetitions 1–4. The task/seed plans, observations/dataset identities, horizon, replanning interval, simulator revision and EMA settings match. All six ARC configurations have identical starting-state hashes to each other. The cause of the ARC/OAT state differences is unresolved. Strict native paired validation correctly rejects the full comparison.

| Method | R | D | M | Full-suite successes | Full-suite SR | Difference from OAT |
|---|---:|---:|---:|---:|---:|---:|
| OAT | — | — | — | 1,542/2,500 | 61.68% | — |
| ARC STK1 | 192 | 1.6 | 36 | 1,700/2,500 | 68.00% | +6.32 pp |
| ARC STK2 | 192 | 0.8 | 32 | 1,711/2,500 | 68.44% | +6.76 pp |
| ARC DUR1 | 192 | 1.6 | 24 | 1,708/2,500 | 68.32% | +6.64 pp |
| ARC DUR2 | 384 | 0.8 | 32 | 1,731/2,500 | 69.24% | +7.56 pp |
| ARC shared STK | 384 | 1.6 | 36 | 1,648/2,500 | 65.92% | +4.24 pp |
| ARC shared DUR | 384 | 1.6 | 36 | 1,693/2,500 | 67.72% | +6.04 pp |

R is the rotation limit in degrees; D is the translation limit in meters; M is the support count. The shared configuration is evaluated with both STK and DUR. These are observed scores of one trained checkpoint per configuration.

## Comparison on matched starting states

Excluding the entire affected drawer task leaves nine tasks and 2,250 episodes per policy, with balanced coverage across all five repetitions. Every recorded starting-state hash matches across ARC and OAT on this subset. It is a diagnostic nine-task comparison, not the full Spatial benchmark.

| Method | Successes on nine matched tasks | Success rate |
|---|---:|---:|
| OAT | 1,384/2,250 | 61.51% |
| ARC STK1 | 1,494/2,250 | 66.40% |
| ARC STK2 | 1,532/2,250 | 68.09% |
| ARC DUR1 | 1,539/2,250 | 68.40% |
| ARC DUR2 | 1,544/2,250 | 68.62% |
| ARC shared STK | 1,464/2,250 | 65.07% |
| ARC shared DUR | 1,540/2,250 | 68.44% |

DUR2 leads OAT by 7.11 percentage points on the nine-task subset. On all 2,300 individually matched episodes (including the first repetition of the drawer task), DUR2 succeeds in 1,583/2,300 (68.83%) and OAT in 1,413/2,300 (61.43%). The latter subset has unequal task coverage and is also diagnostic.

## Validation and provenance

The native reader verified complete, unique episode identities and valid outcomes for each individual full run. The released full-protocol validator passed for every policy. Artifact SHA-256 metadata was checked on download. Native paired validation was attempted and rejected the starting-state mismatches; these checks were not bypassed to publish a paired full-suite result.

Evaluation uses two observations, a 32-action prediction horizon, 16 executed actions per replan, and up to 550 steps per episode. ARC uses the recorded STK/DUR representation, and OAT uses the native predictor and frozen tokenizer. ARC has 62,739,860 total policy parameters; OAT has 33,229,721. This experiment does not match total parameter counts.

OAT ran five repetitions concurrently; these ARC runs used their existing sequential evaluator. The OAT recovery also showed outcome differences from its earlier canceled partial run despite matching starting states; see the [OAT result and reproducibility notes](oat_libero_spatial_20260924.md). The effect of execution layout on exact reproducibility remains unresolved.

[Machine-readable results, per-task scores, run IDs and artifact hashes](arc_vs_oat_libero_spatial_20260924.json).
