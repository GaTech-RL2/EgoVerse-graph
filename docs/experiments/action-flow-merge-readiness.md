# Action-flow stack: merge handoff

Date: 2026-09-06. Branch: `codex/action-flow-stopgrad-ablation-20260906`.

## Fix and regression guard

The previous ICE smoke sent `SIGUSR2` while Python was importing, before the
trainer registered its handler; no checkpoint was produced. Commit
`7eb8904b281cc7b8e1d6cf3cf5434f91db23b467` installs the handler before NumPy,
PyTorch, and model imports and retains the request through initialization.
Serialization remains outside the handler, after a completed optimizer step.
Request counters prevent a new signal during a save from being cleared.

`tests/test_synthetic_trainer_signals.py` permanently checks:

- A real `SIGUSR2` at the first NumPy import produces an unscheduled step-1
  checkpoint, despite a 50,000-step checkpoint interval.
- Restarting from that checkpoint exactly matches uninterrupted nonlinear
  action-velocity training: model, optimizer, RNG state, and validation metrics.
- Failed atomic writes preserve the previous checkpoint and clean temporary files.
- Requests arriving during a save remain pending.

No objectives, loss weights, or gradient-routing defaults changed. Full gradients
remain the default; stop-gradient modes remain explicit ablations.

## Validation

- Fix commit: **405 repository tests passed**, 5 dependency warnings.
- Changed-file Ruff and `git diff --check`: passed.
- Current upstream `main`: `3d60f06c25d9bd7edbc8fd9577f44a17944f7b27`.
- Required Pipeline, Paper-DP, and UNITE base commits: verified ancestors of main.
- `git merge-tree`: no conflicts.
- Temporary combined-tree commit
  `4800a38b7975909e701d5b7afa1e58cd477eb23e`: **411 tests passed**, 5 dependency
  warnings. This detached validation commit is not part of the published branch.
- Validation used the project `.venv` on macOS; the restart test explicitly used
  CPU and exercised optimizer steps followed by the actual synthetic evaluator.

## Scope and remaining operational gate

The startup regression is fixed and covered by executable tests. This does not
claim successful live Slurm signal forwarding, scheduler requeue, or recovery
within an arbitrary grace period. The old failed ICE smoke remains failed;
run a readiness-aware scheduled smoke before relying on preemption recovery for
a new cluster run. No training jobs or merge into main were requested here.

GitHub access uses Skynet's authenticated `ElmoPA` identity through `sky2`, the
working fallback after `sky1` stalled. Publication includes the previously
unpublished gradient audit, sphere/cube experiment, stop-gradient ablation,
and requeue commits. Existing experimental checkpoints remain bound to their
original source revisions.
