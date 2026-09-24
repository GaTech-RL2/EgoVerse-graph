# Data navigation

The root AGENTS.md applies here. Dataset/experiment selection belongs in
`../hydra_configs/data/` and `../hydra_configs/experiment/`.

- `zarr/zarr_dataset_multi.py`: episode resolvers, `ZarrDataset`, `MultiDataset`,
  normalization, saved schema, bounds checking and annotation cutoffs.
- `zarr/data_module.py`: configured data lifecycle, immutable normalization
  context restoration and source validation. `resolve_memo.py` scopes resolver
  reuse to one preparation call; it is not a persistent cache of SQL results.
- `embodiment/human.py`, `eva.py`, `yam.py`: raw keymaps and geometric frame
  conventions. `action_mode`, `coord_frame`, `rotation_mode`, camera mappings,
  local-frame corrections and calibration inputs come from YAML.
- `embodiment/bimanual_arc.py`: raw-window horizons and time/ARC target choices
  used by the E1 recipes. `e1_fold.py` is an old import name only.
- `zarr/action_chunk_transforms.py`: shared pose, rotation, gripper, interpolation
  and key-routing transforms. Add general transform behavior here.
- `zarr/arc_length_tokenizer.py`, `zarr/e1_arc_tokenizer.py`: the two ARC timing
  representations. Check their layouts before interpreting the last channels.
- `zarr/e1_resolvers.py`, `zarr/e1_anchor_sampler.py`: metadata overrides and
  configurable anchor sampling used by the tempo recipes.
- `filters.py`: dataset query builder. Filter values belong in the selected YAML.
- `zarr/selection.py`: pinned random/paired/custom diagnostic selections and
  evenly spaced episode sampling; preserve real frame indices.
- `scripts/viz_language.py`, `scripts/check_data.py` (under `egomimic/`): use
  the DataModule's preview capabilities without fitting normalization. See
  [data tools](../../docs/integration/DATA_TOOLS.md).

Transforms produce native values. `MultiDataset` owns normalization; graph
training/inference consumes normalized values, and evaluation/rollout adapters
unnormalize predictions once. Preserve normalization mode, key types, shapes and
embodiment IDs when loading a cache. Live rollouts require the full
`normalizer_state`, not just numeric bounds.

Do not change stored episode metadata to rename an embodiment. Canonicalize the
metadata at the loader boundary. Test geometric changes with
`test_action_chunk_transforms.py`, `test_wrist6d_roundtrip.py`, `test_human_arc.py`,
`test_abc_yam.py`, and the relevant ARC/PI numeric tests in the root `tests/`.
