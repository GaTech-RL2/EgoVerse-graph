# Pusher sim migration to main arrival clocks — 2026-11-05

Runtime branch: `codex/pusher-main-arrival-20261105`.
Main source: `c7e110461650d09a88b2c2dff8f02db54b1deace`, verified via GitHub at completion.
The canonical `cartesian_arc.py` is copied byte-for-byte. Seven reused legacy geometry helpers were checked AST-identical to main; the tests pin those hashes.

## Why and what changed

The previous study used first-delay clocks. This migration follows current main: measured current actuator pose is a separate anchor; row i is a future target at (i+1)*dt; velocity is an incoming-segment rate; duration is an absolute arrival timestamp. Both training and rollout now use the last raw simulator observation, before normalization. The existing loader still resolves pre-step offset1 and post-step offset2 source records. The anchor-to-first segment does not consume the future D/R budget.

The pusher adapter retains XY/cos(yaw)/sin(yaw)/padded-gripper geometry. It bypasses main's metric speed constraint and does not use projection offsets. No grippers enter this pusher study. Independent XY/rotation clocks, D80 native units/R26deg/M56, 30Hz, raw80, fixed8 execution, DP16, model, frozen data selection, batch128 and 240k updates are preserved.

New checkpoint/config/normalizer metadata identifies this incompatible arrival contract. Old same-width first-delay checkpoints and normalizers must not be reused. New campaign: `egomimic/hydra_configs/benchmark/pusher_main_arrival.yaml`; new R2 prefix: `experiments/pusher-main-arrival-20261105/`.

## Findings during verification

Velocity clocks in main collapse zero-motion waiting time. A synthetic gripper event consequently occurred one tick earlier than the raw event sequence; the adapter matches main exactly. Duration preserves the event timestamp. The original generic first-control replay assertion was too strict for stationary incoming velocity streams. Preflight now checks first-support source alignment, checks first-control equality for moving incoming streams, and records first-control error and full-window RMSE. It does not change main's pause behavior or add a pause-aware ablation.

55 CPU checks passed: canonical/helper source parity; native2/3/4 adapters; velocity and absolute duration; stationary and paused streams; angular wrap; unbounded isotonic correction; explicit anchor requirements; batched decoding; synthetic pre/post-step dataset resolution including end padding; raw state snapshot before in-place normalization; fixed8 queue history; incompatible normalizer rejection; all six arm/regime Hydra recipes; train/valid transforms for every pusher; and existing planar rollout regressions. Changed adapter/rollout files pass Ruff; training launcher and canonical codec compile; git diff --check is clean.

The six workflows were rendered in memory only: 18 training tasks and 558 one-level evaluation tasks, with four L40S per training task and one L40S per evaluation. See launch-plan.json. No jobs were submitted, no GPU training smoke ran, and there are no new success-rate results. Real cached demonstration replay could not be completed because reads of the Desktop cache stalled; validation used synthetic episode caches with deliberately distinct states/actions. The launcher retains real training-only replay auditing and the four-step training smoke before the full run.

## Reproduce

From repository root on the validation branch:

```bash
source emimic/bin/activate
pytest -q validation/test_pusher_main_arrival.py tests/test_planar_rollout.py
```

Tests and this report live only on `codex/pusher-main-arrival-validation-20261105`; the runtime branch contains code/configs/AGENTS changes only.
