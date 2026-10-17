# Mixed-speed training handoff for Elmo

Branch: `aidan/elmo-mixed-config-main-20261013`, directly based on remote main
`7043c0251a350578c6da5a316a79c5ed7d21033d` (synced 2026-10-13).
This is an isolated consolidation of the maintained stattempo source at
`61b699e5704614316d7d67b5b52e938dbfe56d97`, including its native YAM source-window
fix and frozen towel splits. Existing run/checkpoint branches stay immutable.

## Select a recipe

All selectors below are relative to `experiment/yam_arc_grid/`. Pass one with
`+experiment=yam_arc_grid/<selector>` to `train_zarr_cartesian`.

| Family | Selector | Data and purpose |
| --- | --- | --- |
| Matched mixed-speed stationery | `scratch_rl2_stattempo_mixed_{time,arcdur}` | 133 train episodes: 44 slow / 44 medium / 45 fast; common 24 validation episodes, eight per tempo |
| Matched single-tempo controls | `scratch_rl2_stattempo_{slow,medium,fast}_{time,arcdur}` | 133 train episodes per cell; the same 24 validation episodes as mixed |
| Full stationery pool | `scratch_rl2_stattempo_all_{time,arcdur}` | Frozen full-pool control; larger data exposure, not a matched-size mixed comparison |
| Original slowpace baseline | `scratch_rl2_stattempo_slowpace_time` | Original 217-train / 24-validation TIME control; distinct from the 366 split |
| Slowpace 366 stream variants | `bc_rl2_stattempo_slowpace366_{arcdurhyb,arcvelhyb,arcdurtri,arcveltri,arcdurevt,arcvelevt,arcdurref,arcvelref}_r360` | 366 robot train / 24 validation; uncapped 2π rotation; hybrid, tri, event and reference layouts |
| Slowpace Aria cotrain | `cotrain_rl2_stattempo_slowpace366_aria_{arcdurhyb,arcvelhyb,arcdurtri,arcveltri}_r72` | Recorded 72° family, 366 robot train / 24 robot validation / 38 human train; not a 2π BC twin |
| Towels HPT300 | `towels_{since0924,all}_{time,arcdur,arcvel}_hpt300` | Frozen recent and full RL2 towel pools; all three representations share each pool's exact split |

These are RL2 mixed-speed recipes. They do not combine ABC with RL2. The older
ABC+RL2 balanced-data experiments are a separate historical family. The
uncommitted `EgoVerse-graph-mixed-speed` prototype is not the source of this
handoff.

The latest earbuds note records a different `wireless_earbuds_updated` pool
from the cancelled 10-11 runs. No stale earbuds selector is presented as a
fresh-run recipe; freeze that updated pool separately before starting earbuds.

## Shared configuration

`_base/stattempo` owns the 180M HPT h640/t8 + flow d384x10, AdamW 1e-4,
10k warmup then cosine, 240k optimizer steps, seed 42, bf16 and global batch 32
on one GPU. `_base/stattempo_openloop`, `_base/stattempo_aria` and
`_base/stattempo_reference` own their evaluator/domain/normalization differences.
`_base/towels_hpt300` inherits the HPT300 control (h840/t19 + flow d320x6).
Leaves select data, representation, dimensions and run identity through Hydra
defaults. Model graph stages and observation/inference contracts are inherited.

The remaining flattened medium Lambda TIME/ARC-duration and eval TIME recipes
now inherit those maintained parents as well. Their fully resolved behavior is
unchanged, including historical logger IDs, checkpoint retention and evaluator
settings. Do not reuse a historical W&B ID for a new run.

Source manifests: `scripts/e1/stationery_tempo_manifest.json`,
`stationery_slowpace366_manifest.json`, `stationery_elmoaidan_split.json`, and
`towels_20261011_manifest.json`. Keep these memberships fixed when comparing
codecs. Builders emit defaults plus differences, never flattened run YAML.

## Configure local paths and identity

SQL/S3 recipes retain their original dataset mount and resolver configuration.
Setting `paths.dataset_dir` alone does **not** relocate an explicit
`data.*_datasets.yam_bimanual.resolver.folder_path`; override both train and
validation resolver paths when using a different mount. The towel manifests
also support the existing `LocalFolderEpisodeResolver` via membership lists and
count/hash pins. Verify local episode inventory before using it.

For a fresh run, set unique `paths.output_dir`, `hydra.run.dir`,
`norm_stats.save_cache_dir`, `logger.wandb.id`, and `logger.wandb.name`.
Keep `logger.wandb.entity=rl2-group`, `logger.wandb.offline=false`,
`logger.wandb.resume=never`, and `WANDB_MODE=online`. Fit normalization on that
row's training split; do not share a different codec's normalizer.

Compose without launching (replace paths/ID for your host):

```bash
export WANDB_MODE=online
python -m egomimic.trainHydra --config-name=train_zarr_cartesian \
  +experiment=yam_arc_grid/scratch_rl2_stattempo_mixed_arcdur \
  paths.output_dir=/tmp/elmo-mixed-config-compose \
  paths.work_dir=/tmp/elmo-mixed-config-compose \
  data.train_datasets.yam_bimanual.resolver.folder_path=/YOUR/DATASET/MOUNT \
  data.valid_datasets.yam_bimanual.resolver.folder_path=/YOUR/DATASET/MOUNT \
  logger.wandb.id=elmo-mixed-arcdur-UNIQUE \
  +logger.wandb.name=elmo-mixed-arcdur-UNIQUE \
  logger.wandb.resume=never logger.wandb.offline=false \
  logger.wandb.entity=rl2-group --cfg job --resolve
```

Use the selected host's maintained launcher after data, normalization, model,
optimizer, checkpoint budget and optimizer-plus-validation smoke preflight.
`scripts/e1/stationery_ft.sbatch` is the existing Phoenix entrypoint with
`EXPERIMENT`, `WT`, `PY`, `OUT`, `ARM` and `EXTRA` interfaces; its scheduler
headers and H100/H200 guard are Phoenix-specific. Skynet/ICE must use their
maintained launchers and host bindings. Do not start Python on PACE login nodes.
No job submission or new GPU/robot verification is part of this handoff.

## Verification

```bash
python scripts/e1/validate_stattempo_configs.py
pytest -q tests/test_stattempo_consolidation.py tests/test_data_context_contract.py
```

The pinned checks cover 50 original resolved recipes, codec/clock parity,
model state schemas, inference declarations and normalization restoration.
All 56 experiment leaves are also compared before/after this cleanup using
complete resolved configs, including the six towel HPT300 selectors.
CPU checks establish source/config compatibility; each actual new run still
needs its exact host/data/source preflight and optimizer-plus-validation smoke.
Historical checkpoint binding fingerprints belong to their recorded source;
use that source for historical reload/resume.
