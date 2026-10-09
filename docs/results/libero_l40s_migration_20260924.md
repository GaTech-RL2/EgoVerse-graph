# L40S migration

Verified snapshot: 2026-09-24T19:51:16.420355+00:00.

The September 24 user instruction moves this campaign to L40S only. All seven owned H100 workflows are canceled after durable checkpoint verification. Other agents’ workflows and worktrees are unchanged. The four required replacements are accepted on `groot-l40s-01` at NORMAL priority; allocation and actual training resumption remain pending at this snapshot.

| Replacement | L40S GPUs | Status | Saved optimizer/EMA updates |
| --- | ---: | --- | ---: |
| `arc-eval-l40s-20260924-stk1-10-1` | 1 | PENDING | 605,121 / 605,121 |
| `arc-dp-l40s-20260924-stk1-10-1` | 4 | PENDING | 527,560 / 605,121 |
| `arc-dp-l40s-20260924-stk2-10-1` | 4 | PENDING | 538,450 / 605,121 |
| `arc-dp-l40s-20260924-dur1-10-1` | 4 | PENDING | 516,670 / 605,121 |

Three DP policies resume their saved weights, optimizer, EMA and normalization state. SHA-256 was recomputed from each downloaded checkpoint; optimizer step and EMA counts equal its global step. Four L40S GPUs use microbatch 256 and one accumulation step, retaining global batch 1,024 and the original 5,001 epochs / 605,121 optimizer updates. The resume fix preserves the saved `oat_dp` backbone metadata; 117 focused CPU checks passed and four CUDA-only checks were skipped locally. Real L40S restoration and advancing counters are still to be verified.

The original STK1/LIBERO-10 U-Net checkpoint completed all 605,121 updates. Its 781 uploaded partial episodes remain intact at the original source prefix. The replacement restarts the complete 2,500-episode evaluation using five repetition workers on one L40S, preserving the checkpoint, task/seed plan and 32/16 replanning. It does not merge the old partial outcomes into the replacement. Initial-state pairing still requires verification before a final comparison.

The three retired OAT source pipelines already have complete canonical 2,500-episode OAT evaluations. Their additional legacy ARC stages/checkpoints are preserved but deferred; they are separate from the selected six-profile ARC campaign. All completed OAT scores remain available.

Priority order remains original ARC versus OAT evaluations, then ARC with OAT’s released diffusion policy, then ARC+OAT. The Object-score anomaly and Spatial/LIBERO-10 starting-state audit remain unresolved. LIBERO-90 and the hybrid runs remain deferred.

[Machine-readable run IDs and immutable checkpoint receipts](libero_l40s_migration_20260924.json). [Priority policy](../LIBERO_EXPERIMENT_PRIORITIES.md).
