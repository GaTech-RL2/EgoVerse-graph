# Training orchestration agent rules

The repository-root `AGENTS.md` applies. Before changing Lightning wrappers,
DataModules, normalization binding, or train/eval orchestration, read
`../../docs/GENERIC_PIPELINE_CONTRACT.md`.

- Shared training code composes configured interfaces. It must not inspect
  concrete dataset classes, resolver keymaps, storage suffixes, embodiment
  registries, model families, or evaluator-specific fields.
- The DataModule/data adapter owns dataset construction, normalization,
  collation, samplers, source/group identity, episode identity, and ordering.
- `ModelWrapper` treats source names as opaque and delegates specialized
  optimization or diagnostics through configured behavior/provider interfaces.
- Evaluator data requirements and trainer overrides are explicit interface
  methods. Do not add another implicit attribute convention.
- Existing `MultiDataset`, embodiment-registry, and E1-sampler dependencies are
  documented migration debt. Changes must preserve or reduce that debt, never
  spread it into new shared modules.
- During EgoVerse merge-back, validate every retained HPT/PI/data recipe through
  this one orchestration path. Temporary compatibility aliases may delegate to
  it, but a second trainer/model lifecycle is not allowed to survive cutover.
