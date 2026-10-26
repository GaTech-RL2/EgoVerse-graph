# Elmo deployed source import (2026-10-26)

This change imports the verified source snapshot used by the Elmo image,
proprioception and action model into maintained Git history. It has the real
parent `255fdaf5137f7043b35df7ef48a3c1b8adc70815`, published on
`codex/joint-modality-ablation-20261019-v1` in `GaTech-RL2/EgoVerse-graph`.

The deployment recorded source label
`153d4a5484bb61eb113d9ac7a7b7a5cbabbbf920`. Its original Git object and ancestry
could not be recovered. **This import does not claim that label is a parent or
ancestor, or reconstruct its missing author/commit history.** The new import
receives its own Git identity. Existing runs and checkpoints retain their
original source records.

The independently retained inputs are:

| Input | SHA-256 |
| --- | --- |
| `source-153d4a5484bb61eb113d9ac7a7b7a5cbabbbf920.tar.gz` (2,061,765 bytes) | `5e3f11a43ae33411591d4149c77432ad355c0671f9764a68e18fdddbf180aa2f` |
| `source-153d4a5484bb61eb113d9ac7a7b7a5cbabbbf920.files.sha256` | `208ba5fa51d0514fd27d00f0f202cbf73f6e31ab7f3537e26ca55004e4fe3f99` |

All 1,030 manifest paths are reproduced byte-for-byte with archive executable
bits: 988 already matched this parent, 25 are updated, and 17 are added. Every
parent path outside the export is retained. This change imports the deployed
baseline only; the larger-model configuration, fused objective and boundary
sampling experiments belong in separate changes.

Runtime packaging must use the exact 1,030-path snapshot allowlist, plus only
explicitly reviewed experimental additions. Retaining older recipes and tests
in Git does not select them for execution or certify their compatibility. The
native model is `egomimic.models.joint_image_proprio.JointImageProprioModel`;
the Elmo selectors live under `experiment/pixel_proprio/native_joint/`.
Operational services remain separately hash-bound; do not accidentally use
older operational modules retained from this parent.

Byte equivalence and local CPU checks do not certify cluster readiness. A new
immutable source export/readback, explicit runtime/data bindings and matching
allocated optimization, validation and full-state/RNG smoke are required for
new experimental runs. No historical run identity is relabeled by this import.
