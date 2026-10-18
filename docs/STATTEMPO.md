# Aidan stationery source and recipes

New ARC/stattempo source work starts from `aidan/stattempo-consolidated-main-20261011`
in `GaTech-RL2/EgoVerse-graph`. This is the main-based consolidation of
`aidan/stattempo-streams-ref-20261011` at `51cd2bf8bc52380757130b209ae1f686cbe42b6f`.
The starting main commit is `7043c0251a350578c6da5a316a79c5ed7d21033d`.
Resolve and record the branch's exact current commit before new work.

This is a squashed source migration onto main, with main's graph, data-context,
checkpoint, initialization and test infrastructure retained. It does not replay
obsolete consolidation commits or remove main's unrelated model families.
The original streams/ref/r72 and arc-bc-consolidated branches remain historical
source records. Existing jobs and checkpoints keep their recorded source;
this PR is not authorization to restart them or to bypass a training preflight.

## Current stack (2026-10-18): one branch for the three YAM task runs

`aidan/yam-tasks-consolidated-20261018` stacks on Aniketh's
`codex/absolute-log-clock6` (PR #259) through a single-parent catch-up of that
branch onto main `a525d738`, then the four consolidation commits above and the
earbuds / towels / timing / M-D recipe commits that previously lived only in
Phoenix worktrees. Every YAML composes under `scripts/audit_hydra_configs.py`
and the 50 stationery recipes still match the pinned `51cd2bf8` contracts.

All three tasks launch through the same one-GPU launcher; the leaf name is the
only required input, and a relaunch changes just its W&B identity:

```bash
sbatch --export=ALL,ARM=<tag>,EXPERIMENT=yam_arc_grid/<leaf>,PY=<venv python>,OUT=<run root>,"EXTRA=logger.wandb.id=<leaf>_<stamp>_s42" \
  --job-name=<tag> --cpus-per-task=8 --mem=160G scripts/e1/stationery_ft.sbatch
```

| Task | Frozen pool | Leaves under `experiment/yam_arc_grid/` | Small change per run |
| --- | --- | --- | --- |
| Stationery (sort, slow-pace 366) | `scripts/e1/stationery_slowpace366_manifest.json` | `bc_rl2_stattempo_slowpace366_{arcdur,arcvel}`, `..._arcdur_abstime`, `..._arcvel_nosmooth`, `..._m200_d033` twins, `..._arc_projection_{dur,vel}[_m200_d033]`, plus the hyb/tri/evt/ref r72 / r360 builder recipes | `logger.wandb.id`; pick the variant leaf (or `_base/m200_d033` for M 200 / D 0.33) |
| Earbuds (`wireless_earbuds_updated`) | `scripts/e1/earbuds_updated_manifest.json` (refresh with `scripts/e1/build_earbuds_updated.py`) | `earbuds_updated_{time,arcdur,arcvel}_hpt300`, `earbuds_updated_arcdur_abstime_{m200,d03}_hpt300`, `earbuds_updated_arcvel_nosmooth_{m200,d03}_hpt300`; `earbuds_all_*_hpt300` for the unfrozen pool | `logger.wandb.id`; a new pool is one builder run plus the data leaf it writes |
| Towels (`fold_towels`) | `scripts/e1/towels_plus74_20261015_manifest.json`, `scripts/e1/towels_20261011_manifest.json` | `towels_{new290plus74,since0924plus74}_{time,arcdur,arcvel}_hpt300` (HPT300, `_base/towels_hpt300`) | `logger.wandb.id`; `towels_pool` + the data leaf for another frozen pool |

`e1.variant` selects the codec and its inference contract in every leaf
(`time`, `arcdur`, `arcvel`, `..._abstime`, `..._nosmooth`, hybrid / tri / evt
/ ref, projection); `e1.M` / `e1.D` are threaded through the data transforms,
tokenizer and graph by `_base/m200_d033`. Aniketh's `experiment/abc_arc`
recipes (RL2 organize cotrain, ABC + RL2 towels multistream and log clocks,
ABC-DiT) stay launchable from the same tree.

## Configuration structure

[ARC Projection](ARC_PROJECTION.md) adds duration and velocity model leaves
whose single per-arm clock uses a point 2 cm along each end-effector-local axis.
Its recipes inherit the existing BC infrastructure and frozen slowpace366 pool.

All 50 existing stattempo experiment selectors remain available under
`experiment/yam_arc_grid/`. Their shared parents are:

- `_base/stattempo`: FT optimizer/scheduler, h640/t8/d384x10 dimensions,
  240k steps, checkpoint cadence, data paths and logging defaults.
- `_base/stattempo_openloop`: complete-episode robot validation.
- `_base/stattempo_aria`: the human source and cotrain graph selection.
- `_base/stattempo_reference`: 24D reference-stream normalization and tick-prefix
  evaluation. Leaves supply the duration/rate ranges and run identity.

`model/e1/hpt_flow_wrists_ft.yaml` owns the graph. Its named `model.graph`
components resolve into the original ordered `pipeline.stages` list, preserving
checkpoint parameter names. Cotrain adds its human stem/domain; reference models
set `clip_sample_range=1.0`. Neither duplicates the stage list.
Observation/inference fields are inherited from `model_contract/e1_wrists`.
Data codec leaves inherit the same pool and loader. The YAM keymap and transform
explicitly pin 100 source frames; main's other recipes retain their own horizons.

Builders write Hydra defaults and actual differences, never flattened resolved
configs. The existing slowpace366 commands remain:

```bash
python scripts/e1/build_slowpace366.py
python scripts/e1/build_slowpace366.py --rotation-deg 360 --kinds bc --stamp 20261011 --fixes --variants arcdurhyb arcvelhyb arcdurtri arcveltri arcdurevt arcvelevt
python scripts/e1/build_slowpace366.py --rotation-deg 360 --kinds bc --stamp 20261011 --variants arcdurref arcvelref
```

On PACE, run Python through a scheduled allocation and use the existing project
venv. These commands generate configs; they do not launch jobs. Launch wrappers
still use the frozen manifest, reject uncommitted regenerated configs, and send
smokes and training to W&B online. The 366 train / 24 validation / 38 optional
human episode lists are unchanged in `scripts/e1/stationery_slowpace366_manifest.json`.

## Preserved runtime and compatibility

The port includes hybrid, tri, event and reference tokenization; the 72-degree
and 360-degree recipes; decode speed floors; whole-token tick-prefix evaluation;
optional quantile widening; fixed action ranges; and clipped flow sampling.
Fixed-range contexts reject incompatible restored scales in either direction.
Main's partial-normalization recovery remains available; ambiguous fixed-range
stats-only recovery is rejected. Every declared E1 decoder now accepts its
18D/20D/24D layout, including event and reference profiles that the old robot
adapter did not implement.

Main's default lab open-loop evaluator retains its fixed control-prefix behavior.
Stationery's evaluator group explicitly selects its recorded execution modes.
Main's other source families, tests, docs, and runtime APIs are retained.
Later commits found only on the old `arc-bc-consolidated` tip (Lambda launch packs,
auxiliary validation-loss tools) are historical tooling, not additional source
for the current streams/ref runs. They are not reintroduced as duplicate launch
menus; consult their recorded branch for those historical runs.

## Validation and limits

`tests/fixtures/stattempo_contracts.json` pins the original source's resolved
configuration hashes, 36 numeric codec cases, and five full model state schemas.
The config comparison removes only documented structural differences: the new
named graph (already expanded in pipeline.stages), explicit null sampling clip,
explicit 100-frame transform pin, factored reference ranges, and main's retired
`eval_logger_enabled=false` flag. Dataset selection, network structure,
optimizer/scheduler, callbacks, evaluation settings and logging identity remain
in the comparison.

```bash
python scripts/e1/validate_stattempo_configs.py
pytest -q tests/test_stattempo_consolidation.py tests/test_data_context_contract.py
```

CPU tests cover token/clock parity, inference declaration and decoder construction,
normalizer restoration and rejection, small-graph backward/sampling, and full
state-dict name/shape and parameter-count parity. They do not claim trained-weight
reload, GPU throughput, or live robot validation. Historical checkpoint binding
fingerprints still belong to their original source; do not rewrite them to make
an old checkpoint appear native to this new branch.
