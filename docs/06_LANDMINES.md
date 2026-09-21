# Landmines

Ordered by how much time each cost. Every one is a real incident.

## 1. `FAILED_EVICTED` means disk, not preemption

**Symptom:** jobs die at seemingly random points; the status column says
`FAILED_EVICTED`, which looks exactly like being preempted.

**Reality:** the container overran its ephemeral storage. Only
`osmo workflow events <id>` shows the reason.

Two instances: a 192 GB wholesale checkpoint-prefix copy, and `save_top_k: -1`
projecting 1.3 TB. Details in `docs/04_INFRA.md`.

**Cost:** hours of shrinking chunk sizes and migrating pools, plus cutting
`storage` 400Gi -> 100Gi, which made it *worse*.

**Rule:** read the events, then do the storage arithmetic, before any theory
about contention.

## 2. The action-target off-by-one

**Symptom:** none. Training succeeds and produces plausible numbers.

`get_planar_keymap` fetches `action_horizon + action_target_offset` steps and
relies on `SliceActionTarget` to drop `a_t`. Factories that don't *name*
`action_target_offset` swallow it via `**_kwargs`, so the tokenizer consumes all
`H+1` steps starting at `a_t`.

Affected exactly 2 of 43 configs — and they were the two ARC arms of the 260M
headline comparison. Fixed in `61bf2bf`.

**Rule:** a factory taking `action_target_offset` must also take
`raw_action_horizon`. Re-run the audit in `docs/01_ARC_CODEC.md` after touching
any factory.

## 3. `normalize()` returns data UNCHANGED when stats are missing

**Symptom:** none. Raw pixels into a network trained on normalized input,
producing a plausible number.

No exception, no warning. Detected only by an explicit delta check
(`state_agent_obj was NOT normalized (delta 0)`).

Compounding trap: `normalize()` iterates `key_types[emb_id]` built from the
**datasets**, so pooled stats alone are insufficient — the embodiment needs a
dataset entry too. That is why `artic_all9_*.yaml` exists.

**Rule:** for any held-out embodiment, pool stats *and* give it a dataset entry.
Verify with `tools/check_norm_stats.py`.

## 4. `--set-env` does not override the `environment:` block

**Symptom:** the job runs with the wrong configuration and never says so. A
BC eval would have silently used the nine-domain data config.

Its `--help` explicitly claims it overrides. A `--dry-run` shows it does not.

**Rule:** use `{{template params}}` + a `default-values:` block and
`--set-string`. Verify with `--dry-run` before submitting.

Related: a second `--set` **replaces** the first (`nargs="+"` + plain store), so
combine into one.

## 5. Hydra validates on construction, and construction is late

A one-character config mistake surfaces after cloning, installing and staging
tens of GB — 30+ minutes in, sometimes hours.

**Rule:** run `tools/preflight_experiment_configs.py` every time. It composes,
instantiates, builds the graph and pushes a real token through the denoiser in
seconds.

## 6. `**_kwargs` silently eats unknown arguments

`get_usocket_arc_velocity_transform_list` swallowed `rotation_distance_unit`,
leaving the angular budget uncapped — **every R cell in that sweep tokenized
identically**, making the whole angular comparison a null experiment.

**Rule:** if a config sets a knob, assert the factory names it. Prefer explicit
parameters over `**_kwargs` on anything a sweep varies.

## 7. The eval datamodule's domain set must match the model's

Three separate failures, same root cause:

- `STAGE` omitted the held-out `umi` -> `ValueError: No valid collection names
  from local filtering` on every scored cell.
- Norm-stat pooling keyed off `EMBODIMENTS` -> a suction-only job skipped pooling
  -> `no entry for embodiment id 23`.
- A single-domain BC model given the nine-domain config -> same error.

Relaxing `expected_episode_count=null` does **not** help: zero episodes is a hard
error, not a count assertion.

**Rule:** the eval data config, the model's `action_dims`, and the available
norm stats must describe the same domain set. `EVAL_DATA_MODE` selects which.

## 8. Data configs must NOT carry `# @package _global_`

Experiment configs must; data configs must not. With it, the contents land at
the config root and `cfg.data` silently disappears.

## 9. `planar.action_dims` silently decides the model's shape

`StagesUniteSeparate` builds one head per key. A BC baseline inheriting the
co-train base would quietly stay a seven-domain model and measure nothing.

**Rule:** assert `train domains == expected` in preflight.

## 10. YAML block-scalar breaks

Multi-line `python -c` or nested heredocs inside `contents: |` break the scalar
and the YAML parses into nonsense. Hit four times.

**Rule:** one-line embedded python; `bash -n` the extracted body; run
`tools/lint_osmo_launcher.py`.

## 11. zsh is not bash

- Arrays are **1-indexed**.
- Unquoted variables do **not word-split** — `$EXPS` arrives as one argument.
- Backticks inside `git commit -m "..."` get command-substituted; use
  `git commit -F -`.

## 12. Staging path and link naming

The per-shard episode index restarts at zero, so the shard name must be in the
link name or 24 shards collapse onto 125 episodes. Links must start with
`episode_` because `episode_budget.py` globs `episode_*.zarr`.

**Guard with the same glob the consumer uses.** A guard counting
`find -mindepth 1` reported 3,000 while the consumer saw none.

## 13. Smaller traps

| trap | note |
|---|---|
| wandb `Name` | 128-char limit; preflight checks it |
| wandb entity | **`rl2-group`**. `nvidia-gear` does not exist |
| checkpoint names | `epoch_epoch=2399.ckpt` — the doubled `epoch` is real |
| `find \| head` | under `set -o pipefail` exits 141 |
| flat S3 wildcard listings | truncate at 168,096 objects |
| `semantic_blocks` | hardcoded width-5 in `eval_planar_v2`; width-7 needs `[[0,2],[2,3],[3,5],[5,6],[6,7]]` |
| `PipelineAlgo` | exposes the graph as `.pipeline.stages`, not `.stages` |
| `Tsimulation/__init__.py` | swallows `ModuleNotFoundError`; a missing dep surfaces much later as a confusing import error |
| `EnergyScore` | `accuracy` is a **distance** — lower is better |
| `monitor: None` | so there is no best-checkpoint selection; keeping 48 checkpoints bought nothing |
| preflight on CPU | `PipelineAlgo.__init__` moves nets to cuda; derive device from `next(net.parameters()).device` |

## 14. Reasoning failures worth copying down

Not tooling — process. Each produced a wrong statement to the user:

- **Called stability from one sample.** Saw 5 checkpoints all at epoch 1399 and
  concluded `save_top_k=0` was pruning. It was mid-write; it retains. Take two
  readings before declaring a trend.
- **Inferred cause from a status column.** `FAILED_EVICTED` -> "preemption", for
  hours, without running `osmo workflow events`.
- **Generalised from the cheap cells.** "All five arms fail the task" was true
  of the 21 cells that finished first — which were exactly the low-coverage
  ones. Ordering is not a random sample.
- **Trusted a hang as evidence of emptiness.** A slow `s5cmd ls` was auth
  failure, not an empty prefix, and the conclusion drawn from it happened to be
  right for the wrong reason.
- **Asserted a `set -e` hazard without testing.** Claimed `[ -n "$X" ] && Y=z`
  would abort under `set -e`; it does not. Tested, retracted, removed the false
  comment from the file.
