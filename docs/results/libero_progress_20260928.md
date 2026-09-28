# LIBERO progress and recovery: September 28

Verified at 2026-09-28T16:08:21.464240+00:00. All 60 original ARC/OAT/plain-DP policy trainings are complete. Including FAST, 61/64 policy trainings and 57/64 full evaluations are complete.

Seven unfinished jobs were canceled by cluster GPU quota enforcement on September 27 at approximately 11:37–11:42 UTC. They have accepted successors in `groot-l40s-01`, at NORMAL priority, using L40S. No other agents’ jobs or worktrees were changed.

| Run | Durable optimizer updates / target | Training complete | Saved eval trials | Successor status |
| --- | ---: | ---: | ---: | --- |
| ARC OAT-DP stk_1 stk / libero_object | 325,065 / 325,065 | 100% | 2257 / 2500 | RUNNING |
| ARC OAT-DP shared stk / libero_object | 325,065 / 325,065 | 100% | 1933 / 2500 | RUNNING |
| ARC OAT-DP shared dur / libero_object | 325,065 / 325,065 | 100% | 1792 / 2500 | RUNNING |
| FAST / libero_spatial | 152,280 / 270,054 | 56.4% | 0 / 2500 | RUNNING |
| FAST / libero_object | 155,350 / 325,065 | 47.8% | 0 / 2500 | RUNNING |
| FAST / libero_goal | 280,056 / 280,056 | 100% | 542 / 2500 | RUNNING |
| FAST / libero_10 | 372,680 / 605,121 | 61.6% | 0 / 2500 | RUNNING |

RUNNING includes environment/data setup. Checkpoint restore and optimizer advancement on the new tasks are separate verification steps. The resume points above are the exact durable optimizer/EMA counts; the last logs before cancellation were slightly ahead. No ETA is inferred from canceled-job logs.

Accepted workflow names:

- `arc-eval-r2-20260928-stk1-object-1` — 1 L40S, evaluation.
- `arc-eval-r2-20260928-shared-stk-object-1` — 1 L40S, evaluation.
- `arc-eval-r2-20260928-shared-dur-object-1` — 1 L40S, evaluation.
- `fast-libero-r2-20260928-spatial-1` — 4 L40S, resume training.
- `fast-libero-r2-20260928-object-1` — 4 L40S, resume training.
- `fast-eval-r1-20260928-goal-1` — 1 L40S, evaluation.
- `fast-libero-r2-20260928-10-1` — 4 L40S, resume training.

The three ARC Object policies and FAST Goal resume evaluation with the same final checkpoint and saved trials. FAST Spatial/Object/LIBERO-10 resume optimizer, EMA and embedded BPE from their verified checkpoints. All policy runs retain 5001 epochs, global batch 1024, seed 42, and 2500 evaluation episodes.

The fresh shared DUR / OAT-DP / LIBERO-10 evaluation completed at 68.56% and has zero initial-state mismatches against OAT. The old flagged run is preserved as historical evidence.

The new CPU collector follows all 64 cells and their accepted recovery aliases. Its environment includes the FAST tokenizer dependencies required for native final-checkpoint validation. The successor published the expected 64-cell table and its prior CPU collector was retired after verification.

[Full result table](arc_vs_oat_libero_20260928.md) · [Checkpoint and scheduler evidence](libero_progress_20260928.json)

Observed after restoration: ARC Object STK1 2416/2500 trials, shared STK 2069/2500, shared DUR 1928/2500; FAST Goal 621/2500. These counts are progress, not full-suite SR. FAST LIBERO-10 has verified checkpoint restoration and entered training; Spatial and Object are still in setup at this snapshot.
