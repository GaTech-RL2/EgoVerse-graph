# Pipeline agent rules

The repository-root `AGENTS.md` applies. Before changing shared graph behavior,
read `../../docs/GENERIC_PIPELINE_CONTRACT.md`.

## Generic stage contract

- A reusable stage is a configured `dict -> dict` transform. Declare complete
  `reads`, `writes`, and mode-specific contracts; validate local tensor and
  semantic invariants inside the stage.
- Parameterize keys, shapes, horizons, encodings, and modes that vary between
  recipes. Do not branch on task, dataset, embodiment, robot, experiment, or
  neighboring concrete stage type in shared execution code.
- Cross-stage discovery uses a declared capability, stable configured stage
  identifier, or explicitly injected object. Do not add `isinstance`, class-name,
  import-path, or `_target_` dispatch to generic runners or exporters.
- Keep family semantics in explicitly named family stages or adapters. Adding a
  specialized stage is preferable to teaching `Pipeline`, `PipelineAlgo`, or a
  generic artifact writer about another family.
- Model-owned YAML declares inference inputs/history, native output, decoder,
  timing, canonical output, and typed runtime controls. Generic code validates
  and serializes the declaration; it does not reverse-engineer those semantics.
- New ARC stages must preserve per-waypoint or per-segment timing. Chunk-level
  mean timing remains legacy/diagnostic and is not a valid full-trajectory
  rollout contract.

Existing hardcoded cases listed in the architecture document are migration
debt. Do not use them as examples for new functionality.

