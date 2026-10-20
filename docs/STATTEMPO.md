# Aidan stationery source and recipes

> Historical source record: this cleanup stack removes the E1 ARC codec and
> its recipes. Use [ARC_CLOCK_CONTRACT.md](ARC_CLOCK_CONTRACT.md) for new ARC
> work. Existing checkpoints require their original pinned source.

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
| Stationery (sort, slow-pace 366) | `scripts/e1/stationery_slowpace366_manifest.json` (366 train / 24 validation) | `bc_rl2_stattempo_slowpace366_{arcdur,arcvel,arcdur_abstime,arcvel_nosmooth,arc_projection_dur,arc_projection_vel}_m200_d033` | `logger.wandb.id`; pick the variant leaf |
| Earbuds (`wireless_earbuds_updated`) | `scripts/e1/earbuds_updated_manifest.json` (refresh with `scripts/e1/build_earbuds_updated.py`) | `earbuds_updated_time_hpt300`, `earbuds_updated_{arcdur,arcvel,arcdur_abstime,arcvel_nosmooth}_m200_d033_hpt300` | `logger.wandb.id`; a new pool is one builder run plus the data leaf it writes |
| Towels (`fold_towels`) | `scripts/e1/towels_plus74_20261015_manifest.json` (pools `since0924plus74`, `new290plus74`) | `towels_{since0924plus74,new290plus74}_time_hpt300`, `towels_{since0924plus74,new290plus74}_{arcdur,arcvel}_m200_d033_hpt300` | `logger.wandb.id`; `towels_pool` + the data leaf for another frozen pool |

Every ARC leaf trains at M 200 waypoints and D 0.33 m (`_base/m200_d033` for the
stationery recipes, `_base/m200_d033_hpt300` for the HPT-300M towels / earbuds
recipes); the time leaves have no M or D. `e1.variant` selects the codec and its
inference contract (`time`, `arcdur`, `arcvel`, `..._abstime`, `..._nosmooth`,
projection). The hybrid / tri / event / reference codecs, the 72 / 360 degree
recipes, the tempo-bucket and Elmo+Aidan scratch recipes, the Aria cotrain
recipes and the M 100 twins were removed from this branch on 2026-10-18 at
Aidan's request; their runtime (codecs, decoders, evaluator modes) stays, so
`scripts/e1/build_slowpace366.py` can regenerate those recipes from the manifest
when a run needs them. Aniketh's `experiment/abc_arc` recipes stay launchable
from the same tree.

## Configuration structure

[ARC Projection](ARC_PROJECTION.md) adds duration and velocity model leaves
whose single per-arm clock uses a point 2 cm along each end-effector-local axis.
Its recipes inherit the existing BC infrastructure and frozen slowpace366 pool.

The shared parents under `experiment/yam_arc_grid/_base/` are:

- `_base/stattempo`: FT optimizer/scheduler, h640/t8/d384x10 dimensions,
  240k steps, checkpoint cadence, data paths and logging defaults, on the frozen
  slow-pace 366 pool (`data/abc_arc/stationery_slowpace366_bc_time` over the
  shared `data/abc_arc/stationery_yam_loader`).
- `_base/stattempo_openloop`: complete-episode robot validation.
- `_base/arc_projection`: the projection model and data over the same pool.
- `_base/m200_d033`: M 200 / D 0.33 threaded through the data transforms,
  tokenizer, token-row horizons and the open-loop evaluator.
- `_base/towels_hpt300`: the HPT-300M recipe (embed 840, 19 trunk blocks,
  flow 6 x 320) with the tempo evaluator, on the `since0924plus74` towel pool;
  `_base/earbuds_updated_hpt300` selects the frozen earbuds pool over it;
  `_base/m200_d033_hpt300` threads M 200 / D 0.33 through that family.

`model/e1/hpt_flow_wrists_ft.yaml` owns the graph. Its named `model.graph`
components resolve into the original ordered `pipeline.stages` list, preserving
checkpoint parameter names. Observation/inference fields are inherited from
`model_contract/e1_wrists`. Data codec leaves inherit the same pool and loader.
The YAM keymap and transform explicitly pin 100 source frames; main's other
recipes retain their own horizons.

Builders write Hydra defaults and actual differences, never flattened resolved
configs. `scripts/e1/build_slowpace366.py` regenerates the hybrid / tri / event /
reference slow-pace 366 recipes from the frozen manifest when they are needed
again; `scripts/e1/build_earbuds_updated.py` refreshes the earbuds pool. These
commands generate configs; they do not launch jobs. The 366 train / 24
validation episode lists are unchanged in
`scripts/e1/stationery_slowpace366_manifest.json`.

## Preserved runtime and compatibility

The codecs keep hybrid, tri, event and reference tokenization, decode speed
floors, whole-token tick-prefix evaluation, optional quantile widening, fixed
action ranges and clipped flow sampling, so the removed recipes can be rebuilt
without a code change. Fixed-range contexts reject incompatible restored scales
in either direction. Main's partial-normalization recovery remains available;
ambiguous fixed-range stats-only recovery is rejected. Every declared E1 decoder
accepts its 16D/18D/20D/24D layout.

Main's default lab open-loop evaluator retains its fixed control-prefix behavior.
Stationery's evaluator group explicitly selects its recorded execution modes.
Main's other source families, tests, docs, and runtime APIs are retained.

## Validation and limits

`tests/fixtures/stattempo_contracts.json` pins the resolved configuration hashes
of the six stationery M 200 / D 0.33 recipes as they were authored on 2026-10-17
(source `8e80f090`, the branch tip before the rollout consolidation), 36 numeric
codec cases, and the six models' full state schemas. The config comparison
removes only documented structural differences: the new named graph (already
expanded in pipeline.stages), explicit null sampling clip, explicit 100-frame
transform pin, factored reference ranges, main's retired
`eval_logger_enabled=false` flag, and the deploy-only decoder controls added on
2026-10-18. Dataset selection, network structure, optimizer/scheduler,
callbacks, evaluation settings and logging identity remain in the comparison.

```bash
python scripts/e1/validate_stattempo_configs.py
pytest -q tests/test_stattempo_consolidation.py tests/test_stationery_m200_d033.py tests/test_hpt300_m200_d033.py
```

CPU tests cover token/clock parity, inference declaration and decoder construction,
normalizer restoration and rejection, small-graph backward/sampling, and full
state-dict name/shape and parameter-count parity. They do not claim trained-weight
reload, GPU throughput, or live robot validation. Historical checkpoint binding
fingerprints still belong to their original source; do not rewrite them to make
an old checkpoint appear native to this new branch.
