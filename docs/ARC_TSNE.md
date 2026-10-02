# ARC representation explorer

Generate an offline HTML t-SNE report directly from a trusted data config. No
checkpoint, training job, W&B session, web server, CDN, or GPU is required.
These are **native action/token feature projections, not learned model latents**.

From the repository root, in the project environment:

```sh
export EGOVERSE_ABC_DATASET_DIR=/path/to/local/episodes
python -m egomimic.scripts.data_visualization.arc_tsne \
  --data-config egomimic/hydra_configs/visualization/arc_tsne_yam.yaml \
  --max-episodes 12 --samples-per-episode 64 \
  --clock both --output output/arc-tsne
```

Open `output/arc-tsne/index.html` directly. The HTML embeds its complete report;
copying that one file is sufficient. `report.json` contains coordinates, anchor
metadata and provenance; `features.npz` contains the exact pre-scaling feature
matrices in the same row order. Existing output directories are never overwritten.

## Configs and sampling

Accepts a standalone data YAML with `train_datasets` / `valid_datasets`, or a
full Hydra config with those fields under `data`. Hydra defaults are composed
relative to the supplied config directory; use a top-level config when its
defaults reference other groups. External interpolations (e.g. `abc.*`) must
come from that full config or repeatable `--override` arguments. Only the
selected resolver/filter is instantiated: no model or data module setup.

- `--dataset yam_bimanual`: required when the selected split has multiple datasets.
- `--split train|valid`: preserves configured validation fraction and split seed.
  The supplied RL2-oriented example uses **5% validation**; existing configs are
  never silently changed. Episode-count/hash pins are checked when configured.
- `--representations both|arc|baseline`: baseline-only also accepts ordinary
  Cartesian configs. ARC requires the current training transform
  `TokenizeBimanualArcLengthCartesian`; legacy E1 codecs are deliberately rejected.
- `--clock include|exclude|both`: removes **all timing rows/channels before**
  scaling, PCA, and t-SNE. Handles mean stacked `(M+1,14)`, granular stacked
  `(2M,14)`, and wide `(M,28)` layouts. Explicit clock removal is not a guarantee
  of time invariance: pose/gripper geometry and endpoint coverage can retain
  timing correlations.
- `--baseline-rows 100`: a separate fixed-time read of the same anchor, using
  the configured pre-tokenization geometric transforms and repeat-last padding
  at episode tails. Action key horizons are changed on a shallow leaf copy, not
  the ARC leaf. It is not ARC detokenization. The comparison
  holds anchors and geometric transforms fixed, not the temporal coverage:
  ARC is distance/rotation capped, baseline is time capped. `source_rows` exposes
  the ARC source-window length in the report. Configs with explicit interpolation
  before tokenization retain that interpolation; use the supplied YAM recipe
  for native-cadence comparisons.
- Sampling is deterministic, uniform without replacement over episodes and
  then frames (defaults: 12 episodes × 64 anchors). Short episodes contribute
  fewer points. No point is silently substituted after a loader fallback.
  Invalid/nonfinite/sentinel samples cause an error instead of selective dropping.
- Multistream/rotation/D/M/dt come from the actual training tokenizer. The example
  explicitly selects multistream, D=0.4m, R=24°, M=100, per-waypoint wide clocks.
  Existing configs retain their mode rather than being silently migrated.

Resolver discovery may still inspect the complete local inventory to preserve
the split; sampling limits apply to tokenization, not inventory discovery.
There is no recursive scan or parallel loader. Non-local resolvers are rejected
unless `--allow-remote-resolution` is explicitly passed. That option invokes
the configured resolver and **can query SQL or download episodes**, so use a
local config for lightweight inspection. A partial local cache produces a split
of that local inventory, not necessarily the original full training split;
use matching pinned inventories for exact training-split parity. Do not relabel
a control-rate resolver as a plain local resolver: doing so changes cadence.

Only use trusted configs: Hydra targets and filter lambdas execute Python.
Reports include episode/task identifiers, paths, and selected resolved config;
review these before sharing outside your team.

## Projection and HTML

Each representation has its own fit: constant features removed, optional
per-feature standardization (`--scaling standard|none`), PCA to at most 50
dimensions, then seeded 2D t-SNE. Perplexity is capped to `(N-1)/3` for small
samples; actual values, dimensions and KL divergence are recorded. Native
units without scaling can overweight some feature types. Euler angles retain
the training representation's wrap discontinuities. All-constant input fails
instead of showing invented structure. Computation uses one numerical thread.

Linked selection highlights the same anchor across panels. Filter by episode,
color by episode/task/embodiment, select via canvas or keyboard-accessible point
controls/table, inspect source XY trajectories, or download coordinates. Filters
do not refit t-SNE. Source trajectories retain the configured coordinate frame;
they are not camera overlays. Images/video and checkpoint-learned latents remain
the responsibility of the existing Dash latent inspector.

Do not compare axes or absolute distances between independent t-SNE fits. The
viewer is for exploratory neighborhoods, not a quantitative representation score.

## Tests

### Bounded SQL-inventory exports

`arc_tsne_inventory` reads a configured SQL inventory in a read-only transaction,
applies the 5% split to the full matching inventory, then opens only sampled
episodes. The RL2 organizing recipes select exact `lab=rl2`,
`task=organize_stationary`, and human/YAM bimanual embodiment. They explicitly
preserve native source cadence before modern multistream tokenization and omit
all camera keys. Missing sampled episodes cause an error unless `--action-cache`
opts into sparse staging: only pose/action arrays and Zarr metadata are fetched,
one object at a time, with a default 64 MiB total download limit. Shared caches
are never overwritten; no missing episode is replaced with a different sample.

```sh
python -m egomimic.scripts.data_visualization.arc_tsne_inventory \
  --data-config egomimic/hydra_configs/visualization/rl2_organize_human_bimanual.yaml \
  --max-episodes 12 --samples-per-episode 64 --output /path/to/new/report
```

Use `rl2_organize_yam_bimanual.yaml` for YAM. Both export baseline, ARC-clock and
ARC-no-clock panels. Reports retain the full split IDs, selected IDs, transfer
counts, and exact tokenizer configuration. Human gripper channels are zero-padded
because those recordings have no robot gripper signal.

```sh
python -m pytest tests/test_arc_tsne.py tests/test_arc_chunking_modes.py -q
```
