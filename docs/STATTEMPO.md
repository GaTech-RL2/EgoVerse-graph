# Aidan stationery source and recipes

Elmo's isolated main-based handoff is documented in [ELMO_MIXED_CONFIG.md](ELMO_MIXED_CONFIG.md).

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

## Configuration structure

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
