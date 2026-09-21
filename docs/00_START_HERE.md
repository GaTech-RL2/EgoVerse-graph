# Start here

Handover for the ARC action-codec study. Written 2026-09-21 by the agent that ran
it. Read this file, then the one you need.

| file | what it holds |
|---|---|
| `/Users/rpunamiya/Desktop/GEAR/sim_run/wt_codec/docs/00_START_HERE.md` | this: orientation, repo map, current state |
| `/Users/rpunamiya/Desktop/GEAR/sim_run/wt_codec/docs/01_ARC_CODEC.md` | what the codec is, every token layout, decoders, invariants |
| `/Users/rpunamiya/Desktop/GEAR/sim_run/wt_codec/docs/02_TRAINING.md` | hydra configs, the datamodule/sampler, the training launcher |
| `/Users/rpunamiya/Desktop/GEAR/sim_run/wt_codec/docs/03_EVAL.md` | rollout harness, metrics, budgets, chunking, collection |
| `/Users/rpunamiya/Desktop/GEAR/sim_run/wt_codec/docs/04_INFRA.md` | OSMO and R2: the operational facts that cost the most time |
| `/Users/rpunamiya/Desktop/GEAR/sim_run/wt_codec/docs/05_RESULTS.md` | every number, with its confounds stated |
| `/Users/rpunamiya/Desktop/GEAR/sim_run/wt_codec/docs/06_LANDMINES.md` | the traps, each with its symptom |
| `/Users/rpunamiya/Desktop/GEAR/sim_run/wt_codec/HANDOVER.md` | the older long-form doc; still accurate, superseded in places by these |

## The one-paragraph version

The question is whether an **arc-length action codec (ARC)** beats a plain
diffusion-policy action chunk (DP) for planar pushing. On a nine-embodiment
articulated corpus, co-trained on seven and holding two out, **ARC beats DP by
+0.046 peak coverage in-domain (95% CI [+0.018, +0.074])**, concentrated in the
tools where the policy works at all. Separately, **multi-embodiment co-training
is worth about three times more than the codec**: dropping the other six tools
costs −0.115 pooled. The stacked-velocity arms transfer to unseen tools; the
duration arms collapse to ~0.001 on `umi`. There is one large confound you must
not report around: coverage tracks episode length at **r = +0.984**.

## Two repos, and you need both

| path | holds | missing |
|---|---|---|
| `/Users/rpunamiya/Desktop/GEAR/sim_run/wt_codec` | ARC modules, configs, policy, launchers, tools. Branch `codec-replay-rotfix`. **Nearly all work is here.** | the simulator |
| `/Users/rpunamiya/Desktop/GEAR/EgoVerse` | `Tsimulation` — the PushShapes simulator | the ARC modules |

The worktree is a checkout of `GaTech-RL2/EgoVerse-graph`. Open PR: **#128**
(`codec-replay-rotfix` -> `main`), 92 commits.

Rollout jobs need both repos; `docs/03_EVAL.md` explains the PYTHONPATH dance.

## Local python

There is no repo venv. Use:

```
/Users/rpunamiya/Desktop/GEAR/sim_run/venv/bin/python
export PYTHONPATH="/Users/rpunamiya/Desktop/GEAR/sim_run/wt_codec:/Users/rpunamiya/Desktop/GEAR/sim_run/stubs"
```

The stubs directory supplies a fake `projectaria_tools`, which
`egomimic/rldb/zarr/action_chunk_transforms.py` imports at module scope. Without
it every hydra instantiate fails with `ModuleNotFoundError: projectaria_tools`
and it looks like a config bug. It is not.

## State as of 2026-09-21

**Done and trustworthy**

- 45/45 co-train rollout cells (5 arms x 9 embodiments, 40 episodes each).
- 6/7 single-embodiment BC baselines scored.
- Figures and CSV committed under
  `/Users/rpunamiya/Desktop/GEAR/sim_run/wt_codec/results/`.

**Unfinished**

- `spring` BC cell never completed. Its eval job was cancelled twice
  (`bce-spring-1`, `bce2-spring-1`). Relaunching it is the cheapest open item —
  one 4-GPU job, ~1 h. See `docs/03_EVAL.md`.
- `articbc-3-1` (BC training) reached epoch 2399/2400 and then **exited 1**. All
  seven `last.ckpt` are banked and complete, so the checkpoints are usable and
  were used — but nobody has identified which of the seven runs returned
  non-zero. The log is flooded with `flipper` bounds-violation warnings.
- The three 260M runs in `HANDOVER.md` §3a are **confounded** by an action-target
  off-by-one (see `docs/06_LANDMINES.md`). The code is fixed; the runs were
  trained before the fix and would need retraining. The user deliberately
  deferred this.

**Do not re-derive these; they are settled and written down**

- `FAILED_EVICTED` means disk, not preemption. Twice.
- `--set-env` does not override the OSMO `environment:` block.
- R2 needs `R2_*` remapped to `AWS_*`, region `auto`, no session token.

## If you do one thing

Finish the `spring` BC cell and regenerate
`/Users/rpunamiya/Desktop/GEAR/sim_run/wt_codec/results/figures/conditions_full.png`.
That completes the only figure with a hole in it.

## How the user works

- **Absolute paths in replies, always.** The GEAR tree has several look-alike
  roots and relative paths are ambiguous.
- They ask for status often and want it short and honest. If a thing failed, say
  so and say why.
- They care about confounds being stated on the figure, not in a footnote.
- They will accept "I was wrong about X" without drama; they will not accept a
  number reported as settled when it was one sample.
