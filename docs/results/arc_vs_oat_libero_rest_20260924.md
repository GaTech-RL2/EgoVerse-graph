# ARC versus OAT: LIBERO Object, Goal and LIBERO-10

Artifact snapshot: 2026-09-24T17:14:36.505269+00:00.

These scores use the original ARC conditional 1D U-Net and native OAT autoregressive policy. They are separate from the newly launched ARC runs using the released OAT diffusion-Transformer configuration. Each complete score covers 2,500 episodes: 10 tasks, 50 trials per task, five evaluation repetitions of one trained checkpoint.

| Method | Object SR | Goal SR | LIBERO-10 SR |
| --- | ---: | ---: | ---: |
| OAT | 0.36% ‡ | 75.52% | 59.44% † |
| ARC STK1 | 52.80% | 82.20% | Evaluating (606/2,500) |
| ARC STK2 | 52.36% | 84.64% | 52.24% † |
| ARC DUR1 | 53.64% | 89.56% | 54.76% † |
| ARC DUR2 | 63.72% | 87.28% | 57.44% † |
| ARC shared STK | 58.88% | 84.84% | 55.44% † |
| ARC shared DUR | 59.64% | 88.96% | 56.20% † |

Goal: the highest observed ARC score is DUR1, 2,239/2,500 (89.56%), versus OAT 1,888/2,500 (75.52%), a 14.04 percentage-point difference. All six Goal comparisons pass native paired validation, including starting-state hashes.

‡ Object: OAT succeeded in 9/2,500 episodes (0.36%). DUR2 is the highest observed ARC result at 1,593/2,500 (63.72%). The unusually low OAT score is flagged for investigation and should not be presented as a settled ARC advantage. Its correct Object checkpoint, suite/data identity, 5,001 completed epochs, 325,065 optimizer updates and matching EMA count were verified. Recorded policy losses are finite. All six Object comparisons pass native paired validation. The cause of the low OAT result remains unresolved.

† LIBERO-10: OAT succeeded in 1,486/2,500 (59.44%); the best completed ARC run is DUR2 at 1,436/2,500 (57.44%). STK1 is still evaluating, with 606 uploaded episodes at the snapshot; its partial rate is not used as a full-suite score. All five completed comparisons have 600 differing starting-state hashes, so the full-suite differences are provisional. The task/seed plans and other compared protocol fields match. Strict native paired validation correctly rejects each full comparison.

The affected LIBERO-10 tasks are the bottom-drawer-and-close task, the microwave-and-close task, and the two-moka-pots task. Each has 200 mismatches, covering repetitions 1–4. Excluding these entire tasks leaves seven equally weighted tasks and 1,750 episodes per policy with matching starting states. This is a diagnostic subset, not the LIBERO-10 benchmark:

| Method | SR on seven matched tasks |
| --- | ---: |
| OAT | 55.94% |
| ARC STK2 | 48.97% |
| ARC DUR1 | 53.09% |
| ARC DUR2 | 55.83% |
| ARC shared STK | 53.71% |
| ARC shared DUR | 54.29% |

On this seven-task subset, DUR2 is 55.83% and OAT is 55.94%; the raw full-suite two-point difference does not carry over to this diagnostic subset.

Fixed representation configurations (R in degrees, D in meters, M supports): STK1 = 192/1.6/36; STK2 = 192/0.8/32; DUR1 = 192/1.6/24; DUR2 = 384/0.8/32; shared STK and shared DUR = 384/1.6/36.

All three OAT policies completed the full 5,001 epochs at global batch 1,024: Object 325,065, Goal 280,056 and LIBERO-10 605,121 optimizer/EMA updates. Evaluation uses two observations, predicts 32 actions, executes 16 before replanning, and allows at most 550 episode steps. All complete individual records passed native episode coverage/outcome checks and the full protocol validator. Downloaded artifact SHA-256 metadata was verified. The JSON includes all per-task scores, checkpoint identities, run IDs, artifact hashes, training evidence and comparison results.

The new ARC DP campaign has no complete evaluation scores at 2026-09-24T17:12:42.141806+00:00: 7 workflows running, 16 canceled and 1 failed. Checked cancellation reasons include GPU quota reclamation; the failed STK1 Spatial task reports a backend error. Uploaded checkpoints remain preserved. This status supersedes the launch-time allocation snapshot. LIBERO-90 remains deferred.

[Machine-readable results and per-task scores](arc_vs_oat_libero_rest_20260924.json). [Earlier Spatial results and their separate validation caveat](arc_vs_oat_libero_spatial_20260924.md).
