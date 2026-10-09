# Checkpoint and inference migration

Owner: graph integration. The destination runtime is `PipelineAlgo`.

Graph training resumes require a complete, strict state-dictionary match.
Saved data contexts are restored before model binding. Checkpoints from the
integration runtime bind the resolved pipeline, inference declaration and the
**entire** data-context snapshot. That snapshot includes normalization mode,
numeric statistics, schema, and the data adapter's keymap/transform/frame-rate
configuration. Standalone evaluation never recomputes normalization from the
training corpus. Moving validation data or selecting different held-out episodes
is allowed; changing preprocessing under a saved checkpoint is rejected.

Inference artifact schema 2 binds the resolved model pipeline, explicit model
declaration, normalizer schema, tokenizer and decoder. Schema 1 inferred model
semantics by class name and is intentionally rejected. Normal training emits
`inference-config.yaml` and `data-context.json` next to its checkpoints; keep
them with the saved **resolved** training configuration. The high-level resume
and inference loaders reject checkpoints lacking this binding. This is a new
fail-closed boundary, not a claim of automatic migration for older graph runs.
Preserve their exact source/environment/config/data context until an explicitly
verified migration exists. No general converter has been added.

The exporter CLI without checkpoint/context arguments writes an **unbound
declaration** for review; that file cannot authorize deployment. To re-export a
sidecar for an already-bound checkpoint, pass `--checkpoint <file.ckpt>` and
`--data-context <data-context.json>` alongside `--training-config <resolved.yaml>`
and `--output <new-sidecar.yaml>`. It verifies every binding before writing and
refuses to overwrite different artifacts. Exporting a declaration does not prove
a learned codec or timing representation compatible.

`pipeline.inference_session.InferenceSession` consumes already-assembled
model-facing observation windows and returns canonical tensor action sequences
without opening hardware or datasets. Use `normalize_inputs=False` only for
observations already normalized by the training DataModule. The robot policy
uses the same bound graph loader, adding station camera/frame conversion and
history buffering. Both paths decode the same declared time grid and expose the
model's typed sampling/replanning controls.

Strict restoration constructs the architecture while suppressing external
parameter initialization. It then loads every checkpoint key strictly. HPT
vision/text and PI constructors honor this scoped policy; tokenizer and model
configuration resources are still required. This avoids downloading or opening
obsolete pretraining files just to overwrite their parameters from a checkpoint.
Fresh training retains its configured initialization, with overlapping
initialization targets rejected before any parameter is changed.

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
