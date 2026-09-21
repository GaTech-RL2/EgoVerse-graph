# Training: configs, data, launcher

## Hydra layout

Root: `/Users/rpunamiya/Desktop/GEAR/sim_run/wt_codec/egomimic/hydra_configs`

| kind | directory | carries `# @package _global_`? |
|---|---|---|
| experiment | `experiment/pusht/` | **yes** |
| data | `data/pusht/` | **NO — this breaks everything** |
| model | `model/bf/` | n/a |
| evaluator | `evaluator/` | n/a |
| callbacks | `callbacks/` | n/a |

A data config with `# @package _global_` lands its contents at the config root
and `cfg.data` silently disappears. Cost hours once.

### The config families

| family | file pattern | what it is |
|---|---|---|
| articulated co-train | `experiment/pusht/artic_cotrain7_*.yaml` | 5 arms, 7 embodiments |
| articulated BC | `experiment/pusht/artic_bc1_arc_dur_D80_M56_R26deg_<emb>.yaml` | 7 single-embodiment baselines |
| per-embodiment data | `data/pusht/artic_ideal_{arc,dp}_<emb>.yaml` | one domain each |
| co-train data | `data/pusht/artic_cotrain7_{arc,dp}.yaml` | the 7 training domains |
| eval-only data | `data/pusht/artic_all9_{arc,dp}.yaml` | **all nine** — eval only, never train |

`artic_all9_*` exists because `normalize()` iterates `key_types[emb_id]` built
from the DATASETS, so the held-out ids need a dataset entry to get a key map.
Pooled stats alone were not enough.

## How a batch is drawn

`/Users/rpunamiya/Desktop/GEAR/sim_run/wt_codec/egomimic/pl_utils/pl_data_utils.py:164`

```python
return CombinedLoader(iterables, "max_size_cycle")
```

One `DataLoader` per embodiment, each `shuffle=True`, each with its own
`batch_size`. Every optimizer step gets a **dict with one batch from every
domain**.

| property | value |
|---|---|
| `planar.batch_size` | 32 per embodiment |
| training domains (co-train) | 7 |
| **effective batch/step** | **224** |
| mixing ratio | fixed 1:1:…, **not** proportional to dataset size |

All seven corpora are exactly 3,000 episodes (2,970 train after
`valid_ratio: 0.01`), so `max_size_cycle` causes no repetition here. **With
unequal corpora it silently oversamples the small ones** to match the largest.

This is why the BC baseline is a fair comparison: each embodiment contributed
`240k x 32 = 7.68M` samples to the co-trained run, and a solo run at 240k steps
sees the same 7.68M of its own data. Exposure is matched by construction; the
only variable is whether the other six tools were in the mixture.

## `planar.action_dims` decides the model's shape

`StagesUniteSeparate` (`egomimic/pipeline/stages_unite_separate.py:41`) builds
**one head per key**. So:

- `artic_cotrain7_base.yaml` pins all seven -> seven-domain model.
- `artic_bc1_base.yaml` deliberately declares **no** `action_dims`; each BC
  experiment supplies exactly one.

If a BC config had inherited the co-train base it would have quietly remained a
seven-domain model and measured nothing. Preflight asserts `train domains == 1`.

**The model never conditions on embodiment id.** The graph is
`FusedObsEncoder -> ActionTargetBuilder -> DiffusionNoisingStage ->
DiffusionDenoiserStage -> DiffusionEpsilonLossStage` and none of those read it.
The id only selects normalization statistics.

## Normalization

- `norm_mode` is **quantile** (`quantile_1` / `quantile_99`), not mean/std.
  Assuming mean/std produced four confident wrong verdicts once.
- Stats are keyed by numeric embodiment id and are per-`(row, channel)`,
  horizon-shaped.
- **`normalize()` returns the tensor UNCHANGED when it finds no entry for an
  id.** No error, no warning — raw pixels into a network trained on normalized
  input, producing a plausible-looking number. This is the single most dangerous
  silent failure in the codebase.
- Held-out embodiments have no stats by definition. Use
  `/Users/rpunamiya/Desktop/GEAR/sim_run/wt_codec/tools/pool_norm_stats.py` to
  pool the training ids onto the held-out ids.

Guard: `/Users/rpunamiya/Desktop/GEAR/sim_run/wt_codec/tools/check_norm_stats.py`
composes an experiment, reads its own `data.train_datasets`, maps to ids and
requires exactly those present and non-empty. It does **not** hardcode a count —
a literal `== 7` rejected every single-domain BC baseline as if truncated, which
is what killed `articbc-1-1`.

## Preflight — run this before every submission

```bash
cd /Users/rpunamiya/Desktop/GEAR/sim_run/wt_codec
export PYTHONPATH="/Users/rpunamiya/Desktop/GEAR/sim_run/wt_codec:/Users/rpunamiya/Desktop/GEAR/sim_run/stubs"
/Users/rpunamiya/Desktop/GEAR/sim_run/venv/bin/python tools/preflight_experiment_configs.py \
  --holdout umi scoop -- pusht/artic_cotrain7_arc_dur_D80_M56_R26deg
```

It composes, instantiates every transform list, builds the pipeline graph,
**pushes a real token through the denoiser**, checks the wandb name length, and
asserts no held-out embodiment appears in any train/valid dataset. Hydra
validates on construction and construction happens after tens of GB of staging,
so without this a one-character mistake costs hours.

zsh note: `$VAR` holding a space-separated list does **not** word-split. Pass
experiment names as separate literal arguments or you get one long bogus name.

## The training launcher

`/Users/rpunamiya/Desktop/GEAR/sim_run/wt_codec/osmo/articulated_cotrain_sweep.yaml`

Phases: stage corpora -> norm-stats pass per experiment -> guard -> train
`GPUS_PER_RUN` GPUs each, all experiments in parallel -> background R2 sync.

Parameters (`--set` for numbers, `--set-string` for strings):

| param | note |
|---|---|
| `experiments` | space-separated bare names, group-qualified internally by `EXP_GROUP: pusht` |
| `gpus_per_run` | 1 for these; 7 runs fit on an 8-GPU node |
| `max_steps` | 240000 |
| `ckpt_every` | 5000 |
| `val_batches` / `val_interval` | 8 / 20000 — see below |
| `resume_job` | previous job name; pulls each run's `last.ckpt` |

Working example (the BC sweep):

```bash
cd /Users/rpunamiya/Desktop/GEAR/sim_run/wt_codec
EXPS="artic_bc1_arc_dur_D80_M56_R26deg_u_socket ... _spring"
osmo workflow submit osmo/articulated_cotrain_sweep.yaml -p groot-h100-01 --priority HIGH \
  --set num_gpu=8 cpu=64 gpus_per_run=1 max_steps=240000 ckpt_every=5000 batch_size=32 \
        val_batches=8 val_interval=20000 \
  --set-string job_name=articbc-3 branch=codec-replay-rotfix \
    "experiments=$EXPS" wandb_entity=rl2-group resume_job=articbc-2 \
    memory=800Gi storage=800Gi
```

`wandb_entity` is **`rl2-group`**. `nvidia-gear` does not exist; I invented it
once and it failed late.

### Validation cost

Originally `limit_val_batches=80`, `val_check_interval=10000`: measured **255
min per pass**, 12.8 h of a 17.5 h run, projecting 139 h. Now 8 / 20000:
**20-21 min per pass, 13% of wall-clock, ~30 h projected.** Do not raise these
without measuring.

### Checkpoint retention — the disk leak

`callbacks/checkpoints.yaml` ships `save_top_k: -1`, keeping **every** periodic
checkpoint at 3.92 GB each. 48 per run x 7 runs = **1.3 TB**, against an 800Gi
ephemeral limit. `articbc-2-1` died at 53% having pushed 728 GB.

`save_top_k=0` is **not** sufficient — observed keeping both epoch 1399 and 1599
per run. The launcher now prunes explicitly in the R2 sync loop, **after** the
sync so the remote copy is safe:

```bash
ls -1t "$d"/epoch_*.ckpt 2>/dev/null | tail -n +2 | xargs -r rm -f
```

Steady state ~2 files per run. Do not remove this.

### Resume

`resume_job=<name>` pulls `s3://rldb/staged/articulated_cotrain/<job>/<RN>/checkpoints/last.ckpt`
where `RN` is the config's `name:` field (**not** the filename — 
`artic_cotrain7_x` -> `artic_c7_x`). A missing checkpoint under a non-empty
`resume_job` is FATAL by design. Verified working: `articbc-3` resumed at
global_step 120000-125000.
