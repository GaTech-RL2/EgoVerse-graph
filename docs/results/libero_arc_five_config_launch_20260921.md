# Five-configuration LIBERO ARC launch

All **30 full L40S workflows were submitted**. Snapshot: 2026-09-21T22:20:46.402780+00:00.
Scheduler state: 30 RUNNING.
A running scheduler state includes environment setup and replay validation; it is not a policy-training score.

Source is pinned to `60b54d22252cdc4cb53d7fb5b097558f4f7c0ac7`. Each policy uses 5,001 epochs and global batch 1,024 (microbatch 256 × accumulation 4).
The five distinct R/D/M triples expand to six mode/configuration combinations and five suites.
The original five-suite OAT/ARC campaign continues separately. Its OAT models are the intended shared references; these new jobs train ARC only.

| Configuration | R° | D m | M | Spatial | Object | Goal | 10 | 90 |
| --- | ---: | ---: | ---: | --- | --- | --- | --- | --- |
| stk_1 / stk | 192 | 1.6 | 36 | [spatial](https://us-west-2-aws.osmo.nvidia.com/workflows/arc-five-v2-20260921-stk1-libero-spatial-1) | [object](https://us-west-2-aws.osmo.nvidia.com/workflows/arc-five-v2-20260921-stk1-libero-object-1) | [goal](https://us-west-2-aws.osmo.nvidia.com/workflows/arc-five-v2-20260921-stk1-libero-goal-1) | [10](https://us-west-2-aws.osmo.nvidia.com/workflows/arc-five-v2-20260921-stk1-libero-10-1) | [90](https://us-west-2-aws.osmo.nvidia.com/workflows/arc-five-v2-20260921-stk1-libero-90-1) |
| stk_2 / stk | 192 | 0.8 | 32 | [spatial](https://us-west-2-aws.osmo.nvidia.com/workflows/arc-five-v2-20260921-stk2-libero-spatial-1) | [object](https://us-west-2-aws.osmo.nvidia.com/workflows/arc-five-v2-20260921-stk2-libero-object-1) | [goal](https://us-west-2-aws.osmo.nvidia.com/workflows/arc-five-v2-20260921-stk2-libero-goal-1) | [10](https://us-west-2-aws.osmo.nvidia.com/workflows/arc-five-v2-20260921-stk2-libero-10-1) | [90](https://us-west-2-aws.osmo.nvidia.com/workflows/arc-five-v2-20260921-stk2-libero-90-1) |
| dur_1 / dur | 192 | 1.6 | 24 | [spatial](https://us-west-2-aws.osmo.nvidia.com/workflows/arc-five-v2-20260921-dur1-libero-spatial-1) | [object](https://us-west-2-aws.osmo.nvidia.com/workflows/arc-five-v2-20260921-dur1-libero-object-1) | [goal](https://us-west-2-aws.osmo.nvidia.com/workflows/arc-five-v2-20260921-dur1-libero-goal-1) | [10](https://us-west-2-aws.osmo.nvidia.com/workflows/arc-five-v2-20260921-dur1-libero-10-1) | [90](https://us-west-2-aws.osmo.nvidia.com/workflows/arc-five-v2-20260921-dur1-libero-90-1) |
| dur_2 / dur | 384 | 0.8 | 32 | [spatial](https://us-west-2-aws.osmo.nvidia.com/workflows/arc-five-v2-20260921-dur2-libero-spatial-1) | [object](https://us-west-2-aws.osmo.nvidia.com/workflows/arc-five-v2-20260921-dur2-libero-object-1) | [goal](https://us-west-2-aws.osmo.nvidia.com/workflows/arc-five-v2-20260921-dur2-libero-goal-1) | [10](https://us-west-2-aws.osmo.nvidia.com/workflows/arc-five-v2-20260921-dur2-libero-10-1) | [90](https://us-west-2-aws.osmo.nvidia.com/workflows/arc-five-v2-20260921-dur2-libero-90-1) |
| shared / stk | 384 | 1.6 | 36 | [spatial](https://us-west-2-aws.osmo.nvidia.com/workflows/arc-five-v2-20260921-shared-stk-libero-spatial-1) | [object](https://us-west-2-aws.osmo.nvidia.com/workflows/arc-five-v2-20260921-shared-stk-libero-object-1) | [goal](https://us-west-2-aws.osmo.nvidia.com/workflows/arc-five-v2-20260921-shared-stk-libero-goal-1) | [10](https://us-west-2-aws.osmo.nvidia.com/workflows/arc-five-v2-20260921-shared-stk-libero-10-1) | [90](https://us-west-2-aws.osmo.nvidia.com/workflows/arc-five-v2-20260921-shared-stk-libero-90-1) |
| shared / dur | 384 | 1.6 | 36 | [spatial](https://us-west-2-aws.osmo.nvidia.com/workflows/arc-five-v2-20260921-shared-dur-libero-spatial-1) | [object](https://us-west-2-aws.osmo.nvidia.com/workflows/arc-five-v2-20260921-shared-dur-libero-object-1) | [goal](https://us-west-2-aws.osmo.nvidia.com/workflows/arc-five-v2-20260921-shared-dur-libero-goal-1) | [10](https://us-west-2-aws.osmo.nvidia.com/workflows/arc-five-v2-20260921-shared-dur-libero-10-1) | [90](https://us-west-2-aws.osmo.nvidia.com/workflows/arc-five-v2-20260921-shared-dur-libero-90-1) |

Each workflow reuses the audited calibration for its single frozen candidate, runs selection demos 2–9, and confirms on fresh demos **10–24**. The same reset, coverage and numerical controls apply. It automatically starts training after the replay result passes the existing checks; measured gaps to raw are reported under the established protocol. There is no retuning using confirmation outcomes.

Training is followed by 50 policy rollouts per task over five repeats, using the same seeds, initial states and EMA protocol as OAT. ARC metrics are recorded independently. Paired OAT comparisons remain incomplete until both sets of records exist and pass protocol/reset matching.

## Original campaign configurations

These are the earlier per-suite replay selections configured in the original full jobs. R is degrees, D metres, and M support rows. At launch time those jobs were still training OAT tokenizers; their ARC stages had not started.

| Suite | STK (R, D, M) | DUR (R, D, M) |
| --- | --- | --- |
| libero_spatial | (96, 1.6, 36) | (128, 0.8, 32) |
| libero_object | (48, 1.6, 36) | (48, 0.8, 36) |
| libero_goal | (192, 1.6, 36) | (192, 1.6, 36) |
| libero_10 | (128, 1.6, 32) | (192, 0.8, 32) |
| libero_90 | (128, 0.8, 36) | (128, 1.6, 36) |

## Validation and launch receipts

115 focused tests passed across targeted invocations, including replay-before-training sequencing, frozen-setting enforcement, full and smoke ARC-only evaluation, and the exact replay-file parser round trip. Ruff, shell syntax and whitespace checks passed.
Two additional L40S smokes exercise the shared STK M=36 and DUR1 M=24 policies. Their status and the complete run/parent/reference map are recorded in the [machine-readable launch receipt](libero_arc_five_config_launch_20260921.json).

The first full attempt stopped before replay or training because JSON exponent notation was parsed as text by the YAML reader. The replacement source writes YAML and tests the real parser round trip. All 30 first-attempt workflows are terminal; no training checkpoints were lost.

The protected older LIBERO-90 training job was retired at 21:59 UTC only after its V6 replacement advanced and uploaded a new verified checkpoint. Other agents’ workflows, worktrees and data were not changed.
