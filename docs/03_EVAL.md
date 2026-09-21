# Eval: the rollout harness

Launcher: `/Users/rpunamiya/Desktop/GEAR/sim_run/wt_codec/osmo/articulated_rollout.yaml`
Evaluator: `/Users/rpunamiya/Desktop/GEAR/sim_run/wt_codec/egomimic/eval/sim_rollout_planar_eval.py`
Config:    `/Users/rpunamiya/Desktop/GEAR/sim_run/wt_codec/egomimic/hydra_configs/evaluator/sim_rollout_planar_v2.yaml`

## NON-PROTOCOL — say this whenever you quote a number

The canonical `bf_eval_par.sbatch` lives on Skynet and there is no registered
inference contract for the ARC family. This harness follows the protocol's
**substance** (p99 frame budget, PEAK coverage, SR@0.80/SR@0.95, level-0
rollouts on seeds `0..n-1`, sim_v2) so numbers are internally comparable and
nothing more. Every summary line is stamped
`NON-PROTOCOL_REPO_LOCAL_ROLLOUT_NOT_COMPARABLE` on purpose. **Never pool these
with canonical results.**

## What the evaluator actually does

```python
episodes = sorted(self.dataset_dir.glob("episode_*.zarr"))
store = zarr.open_group(str(episodes[0]), mode="r")   # env_args ONLY
...
for ep in range(self.n_episodes):
    seed = self.seed_base + ep
    env.reset(seed=seed)
    for t in range(budget):
        ... peak = max(peak, info["coverage"])
```

Three consequences, all load-bearing:

1. **Initial states come from `env.reset(seed)`, not from the corpus.** The
   staged episodes are read only for `episodes[0]`'s `env_args`.
2. **The corpus's only other job is the p99 frame budget.** So with the budget
   cached, one shard is score-identical to 24 — that is what makes thin staging
   legitimate rather than an approximation.
3. **Episodes are independent.** Seeds `0..39` scored as four chunks of ten are
   *the same forty episodes*. Chunking is exact.

## Metrics

`summary` includes `peak_coverage_mean`, `peak_coverage_median`, `SR@0.80`,
`SR@0.95`, `seed_base`, and **`ep_coverages`** — the full per-episode list. That
last field is what makes paired statistics possible: every arm is scored on the
same seeds, so arm-vs-arm is a paired per-episode difference, not a difference
of means.

`EnergyScore` (training-time val) is different: `accuracy = mean distance from
samples to target`, so **lower is better**, and `score = accuracy - 0.5*diversity`.

## Budgets

`/Users/rpunamiya/Desktop/GEAR/sim_run/wt_codec/scripts/eval/usocket_arc_rollout/episode_budget.py`
writes `{"by_level": {"0": {"budget": N, "n": ...}}, "episode_count": M}`.

Budgets are **deterministic** — `flipper` computed 4842 across 27 independent
runs, every copy identical. They are cached in R2 at
`s3://rldb/staged/articulated_rollout/_pool/budgets/<emb>.json` and honoured
**only** when `episode_count == 3000`, so a budget computed over a thin stage can
never masquerade as a real one.

| embodiment | budget |
|---|---|
| u_socket / gripper / chain_gripper / umi | 626 / 636 / 659 / 752 |
| suction | 2,218 |
| triangle / flipper / spring / scoop | 4,826 / 4,842 / 5,055 / 5,365 |

**This ~7x spread is the study's main confound.** See `docs/05_RESULTS.md`.

## Runtime, measured

| cell | 40 episodes |
|---|---|
| light (budget ~640) | ~74 min |
| suction (2,218) | ~144 min |
| heavy (~4,900) | ~3.5 h |

`ep_s ~= (budget/replan_every) * s_per_call`, with `s_per_call ~= 0.51 s` on
H100. Heavy cells cannot finish inside a preemption window, which is why
chunking exists.

## Chunking

`CHUNK_EPISODES` (default 10) splits 40 episodes into seed windows via
`evaluator.seed_base`. Each chunk banks separately to a shared pool keyed
`<experiment>__<embodiment>__s<base>n<count>.log`. A restart skips what is
already banked — `run_one` probes with a **single GET**, not a list, because
listing an empty R2 prefix costs minutes of retries.

Only a log containing `[sim] SUMMARY` is banked or honoured, so a truncated cell
cannot poison the pool permanently.

Verified: the chunk arithmetic tiles `0..N-1` with no gap or overlap for every N
and chunk size tried, and the collector's merge reproduces the pre-chunking
table exactly.

## EVAL_DATA_MODE — critical for single-domain models

| value | eval data config | for |
|---|---|---|
| `all9` (default) | `pusht/artic_all9_<kind>` | co-train models: 7 heads, need a key map spanning every embodiment |
| `per_emb` | `pusht/artic_ideal_<kind>_<emb>` | **BC baselines**: 1 head, 1-domain `norm_stats.json` |

Handing a BC model the nine-domain config reproduces
`norm_stats file has no entry for embodiment id 23`. Pooling is also skipped in
`per_emb` mode because a single-domain baseline needs none.

`held_out` takes the sentinel **`none`**, not an empty string — an empty
`--set-string` value is not reliably parseable. The script maps `none -> ""`
before anything reads it.

## Launching

Overridable via `default-values` + `--set-string` (**not** `--set-env`, which
does not override the `environment:` block — verified by dry-run):

```bash
cd /Users/rpunamiya/Desktop/GEAR/sim_run/wt_codec
osmo workflow submit osmo/articulated_rollout.yaml -p groot-l40-04 --priority HIGH \
  --set num_gpu=4 cpu=32 n_episodes=40 replan_every=8 chunk_start=0 shards_per_cell=24 \
  --set-string job_name=bce2-spring branch=codec-replay-rotfix sim_commit=f952ca0d \
    eval_data_mode=per_emb train_emb=spring held_out=none \
    experiments=artic_bc1_arc_dur_D80_M56_R26deg_spring "embodiments=spring" \
    ckpt_job=articbc-3 memory=200Gi storage=150Gi
```

**That exact command is the outstanding `spring` BC cell.** Run it.

Co-train cells use one job per (arm, embodiment), 4 GPUs, spread across
`groot-l40s-03` / `groot-l40-04` / `groot-l40-03`. All 24 completed with zero
evictions once the checkpoint pull was narrowed.

## Two repos at rollout time

The rollout needs `egomimic` from EgoVerse-graph and `Tsimulation` from EgoVerse.
Putting the whole EgoVerse repo on `PYTHONPATH` **shadows** EgoVerse-graph's
`egomimic` and you get `No module named 'egomimic.eval.checkpoint_loading'`.
Symlink only what you need:

```bash
ln -sfn /workspace/EgoVerse/Tsimulation /workspace/sim_import/Tsimulation
export PYTHONPATH="/workspace/sim_import:/workspace/EgoVerse-graph"
```

`sim_v2.pushshapes` imports pymunk/gymnasium/pygame/shapely at module scope, and
`Tsimulation/__init__.py` **swallows `ModuleNotFoundError`** in its aliasing
loop — so a missing dep surfaces fifteen minutes later as
`No module named 'Tsimulation.pushshapes'`. The launcher installs
`pymunk==7.3.0 gymnasium pygame shapely` and runs a deep import assertion that
also asserts `SIM_VERSION == 2`.

`sim_commit=f952ca0d` is the simulator the corpus was collected under. The repo
has two simulator lineages; only `sim_v1`/`sim_v2` has the nine agents.

## Collecting results

```bash
cd /Users/rpunamiya/Desktop/GEAR/sim_run/wt_codec
/Users/rpunamiya/Desktop/GEAR/sim_run/venv/bin/python tools/collect_artic_results.py \
  --jobs _pool arp-1 arp-2 ... --csv /tmp/out.csv
```

Reads **R2 directly**, not `osmo workflow logs` — a log fetch on these jobs runs
past ten minutes and times out. It merges seed-window chunks on the `seed_base`
inside each `SUMMARY` and marks partial cells with `*`.

`_pool` is the shared chunk store:
`s3://rldb/staged/articulated_rollout/_pool/cells/`.
