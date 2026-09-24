# Checkpoint and inference migration

Owner: graph integration. The destination runtime is `PipelineAlgo`.

Graph training resumes require a complete, strict state-dictionary match.
Saved normalizers are restored before model binding; normalization hashes must
match on resume. Standalone evaluation never recomputes normalization from the
training corpus. Export the full normalizer schema and statistics if an older
graph checkpoint predates embedded data contexts.

Inference artifact schema 2 binds the resolved model pipeline, explicit model
declaration, normalizer schema, tokenizer and decoder. Schema 1 inferred model
semantics by class name and is intentionally rejected. To migrate an older
graph run, add and review the exact model-owned declaration against its saved
training configuration, then run `python -m egomimic.pipeline.inference_config
--training-config <resolved.yaml> --output <new-sidecar.yaml>`. Preserve the
original config and sidecar. Do not substitute a current recipe or overwrite
historical artifacts. Exporting a declaration does not prove a trained codec
or timing representation compatible.

Historical chunk-mean ARC is diagnostic-only. Its checkpoints remain retained;
they cannot be made into interval-duration policies by changing a decoder.

Legacy HPT checkpoints are reproduced with immutable EgoVerse source
`ec5c903c067bf1b29bbff781bc414c6df2bcf1f2` and its lock. There is no general
HPT checkpoint converter. PI's explicit weights-only initializer accepts one
complete backend namespace with exact key/shape matching; it does not resume
optimizer state or infer action semantics from tensor widths. Its supported
source paths and tests remain part of the PI parity gate.

Temporary `cartesian_pi` keymap aliases delegate to canonical Cartesian
keymaps and warn. Remove them only when saved configurations have been migrated.
The deprecated `configure_flow_inference_steps` helper requires an explicit
`sampler` stage ID and delegates to that setting; migrate callers to typed
controls before removing the alias. Neither alias restores family dispatch.

No checkpoint, branch, dataset, or historical source has been deleted.
