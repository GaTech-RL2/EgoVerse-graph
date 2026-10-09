# Three-source integration — validation in progress

The human authorized one common codebase for these source lines:

- PR #224: `40a0d1032a5b63c29192e31417ad894176b49b9f`.
- Stationary AF/DP: requested `f1fca415567af5c66a3d80a87be1d401a3ee58b0`,
  with its native-window repair `c69ee2aebdf14bd19455abd42f6be75a46c1eeb7`.
- Eight-run LIBERO training: `7c253e5aae5e6d08dace5951c9ed98b716ad36a2`.

Current-main integration baseline is
`3d42edfbe506a357f6a034ddb1a5750890c496f4`.
The shared implementation is not an instruction to replace any running source,
resume a historical checkpoint on a different commit, or launch new training.

## Preserved distinctions

Stationary DP retains AdamW, its own learning-rate schedule, and disabled EMA.
The LIBERO original-OAT DP recipe retains its separate optimizer and released
EMA behavior. Action Flow retains its own objective and optimizer. Matching data
and observations does not mean these algorithms or all augmentation contracts
are identical.

The explicit `fm_field_execution` setting preserves the LIBERO separate-field
execution and the stationary shared-field gradient contract. Optional moment
objectives stay optional. Opaque codec routes retain heterogeneous-source
rejection. Historical native-speed conditioning remains a distinct compatibility
API; reference-free new conditioning uses the dimensionless retiming multiplier.
Strict state-dict contract buffers prevent treating these weights as interchangeable.

Main's generic data-context adapter now carries the existing proportional loader
and native normalization receipt gates. Saved data contexts record the declared
normalizer configuration, not merely statistics, and reject configuration drift.
Partial normalization resumes remain normalization-only. Ordered validation
cannot silently random-fallback to another sample.

## Validation and remaining gates

Focused stationary/LIBERO reconciliation: 54 passed. Main reconciliation:
64 passed after repairing a merged test import. Loader/adapter union: 9 passed.
PR #224/conditioning/native sample-axis union: 61 passed after fixing normalized
versus typed metric identity and the explicit float64 diagnostic input boundary.
Combined context/DP/native-artifact union: 45 passed. These are overlapping suites,
not additive coverage counts.

All 372 shipped YAML configurations resolve with explicit, scoped offline audit
inputs. Synthetic audit paths and hashes are never opened or accepted as launch
evidence. Static checks and formatting pass. Vendored OAT implementation bytes
are excluded from first-party formatting; the first-party adapters remain checked.

These results use a local Python 3.11 / Torch 2.2 CPU environment, not a cluster
training-runtime proof. The full suite initially stopped at collection, exposing
obsolete cross-test imports, two lost main-code transforms, and local missing
Zarr 3 / AV dependencies. The imports and transforms were repaired. An isolated
environment supplies the pinned Zarr 3.1.5 and AV 12.0 without changing the
pre-existing environment. The next full run remains a required gate.

Inherited candidate-verifier tests also expose stale constructor/topology/smoke
expectations. Production approval lists must not be expanded just to pass tests.
Publication through authenticated Skynet Graphite and the final GitHub merge are
still incomplete. No final integration, deployment, training or GPU parity claim
is made by this progress record.
