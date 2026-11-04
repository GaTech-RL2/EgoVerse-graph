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
  used by the current Cartesian task recipes.
- `zarr/action_chunk_transforms.py`: shared pose, rotation, gripper, interpolation
  and key-routing transforms. Add general transform behavior here.
- `zarr/cartesian_arc.py` (stable exports in `arc_length_tokenizer.py`): Cartesian
  ARC with `clock_mode` velocity or duration (absolute arrival seconds,
  anchor-inclusive speed-bounded isotonic decode). `streams` supports
  full_separation (M×20), translation_rotation (M×18), translation or rotation
  (M×16, one clock per arm). Row zero is current measured pose to first future
  target, never a start delay. The finite same-frame 14D `arc.current_action`
  anchor is required separately; source rows begin at idx+1 / time dt, and
  decoded controls are sampled at dt, 2dt, … . Every PW row is an incoming
  speed. No synthetic 400-frame holds or projection mode are supported.
  A zero-progress resampling fallback spans actual source times; fully static
  PW needs the caller's explicit horizon because zero rates contain no time.
  D/R budgets cover the future source path, excluding the incoming anchor
  segment. See `docs/ARC_CLOCK_CONTRACT.md`.
- `zarr/embodiment_resolvers.py`, `zarr/progress_sampler.py`: metadata overrides and
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
metadata at the loader boundary. Keep regression checks and their fixtures outside
the repository; never commit tests. Record local verification results in PRs.
