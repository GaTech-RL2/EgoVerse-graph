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
The repaired candidate gate suite passes 42 tests. It distinguishes the private
codec routes, correct stage counts, campaign-required manifests and explicitly
approved smoke recipes. The validator uses method-specific encoder parameter
counts rather than certifying likelihood and graph codecs as ordinary codecs.

The unrestricted-thread full CPU suite stalled in the local Rosetta libiomp5
runtime; its log and process sample are retained outside the source checkout.
Only that exact local test process was stopped. The replacement uses
`scripts/run_local_cpu_validation.sh`'s single-thread CPU envelope and 60-second
stack diagnostics. That attempt exposed a separate native segfault in tslearn's
Numba SoftDTW routine. The local runner also selects the single-thread Numba
workqueue rather than an additional OpenMP pool; the targeted real alignment
parity retest passes seven cases. The next full run reached a terminal result:
2,219 passed and 180 failed, with 17 skips and 28 subtests. That run overlapped
ongoing repairs and is not a final-head gate. Its failures remain recorded.
This runner does not
change cluster execution or establish exact-runtime or historical-run parity.
Publication through authenticated Skynet Graphite and the final GitHub merge are
still incomplete. No final integration, deployment, training or GPU parity claim
is made by this progress record.

Subsequent focused results: typed framework/launcher contracts 120 passed;
native recipe/context/cluster union 134 passed and one failed; compile,
higher-order DiT-half and world-size resume union 35 passed; retained HPT/PI
and OAT optimizer/checkpoint lifecycle union 19 passed. These overlapping local
suites are not additive coverage or certification of historical full-size runs.

The remaining native recipe failure exposed a real merged H384 base-config
regression: the H640 denoiser override and configured checkpoint policy were
ignored. The base now carries the explicit independent denoiser dimensions,
optional zero-weight objective, and checkpoint setting again; default U-Socket
projection parameter names are retained. Chain's explicit action-model width
and selector alias are supported, with the original 4D default when that field
is absent. The focused config/typed artifact/static-buffer union passes 24
tests. Offline resolution passes all shipped YAML after this correction.
The shared data preflight also honors native resolvers' explicit omission of
extra transforms without reading replay data. Strict saved Hydra tests bind the
actual local version and distinct logical corpus digest, not receipt bytes.

The local runner fails before work when fewer than 4 GiB of temporary storage
is available. Only completed, independently identified synthetic pytest fixture
directories are removed for headroom; logs and XML receipts remain outside the
source checkout. No experiment checkpoint or running job is changed.

The subsequent full constructor audit is still failing and is not waived.
Native evaluator templates need persisted artifact identities and explicit
same-pass profiles; original-OAT policy templates need a trained tokenizer.
Their production gates remain fail-closed. A selected six-context audit now
confirms native data schemas (zero extra transforms), original tokenizer and ARC
constructors, and retains four missing-binding failures as actual limitations.
The deterministic OAT/FSQ scalar-buffer meta construction is isolated to the
offline audit, with real default-tokenizer layout/RNG and restoration tests.
Native metric-schema tests now use valid v2 tensor-reference metadata instead
of making every negative case fail at an obsolete v1 schema: 28 passed,
including wrong sample axes and legacy metric-only rejection. These metadata
fixtures are not payload-byte, episode, or optimizer evidence.

Pinned MuJoCo 3.4.0 / Mink 1.1.0 wheels were installed only in the isolated
local environment. The robot union passes 65 cases; two numeric-interface
cases cannot execute because MuJoCo rejects x86 Python on Apple Silicon. Do
not reinterpret that runtime limitation as success or bypass its architecture
check. Exact Linux/server or native-arm execution remains required.

The isolated native-arm Python 3.11.14 environment now successfully installs
the exact project lock (Torch 2.7.1, alignment and diagnostics extras). Both
previously blocked MuJoCo numeric-interface tests execute and pass there.
The LIBERO evaluator boundary fixtures now supply the explicit native profile,
test missing/mismatched-profile rejection, and isolate wrapper dispatch from
the separately tested numeric diagnostic implementation. The boundary,
v2 artifact metadata, and shared numeric diagnostic union passes 56 tests.
No production artifact-identity or diagnostic gate was relaxed. These CPU
tests are not all historical checkpoint, server-runtime, or GPU certification;
the complete final-head constructor/test/CI and merge gates remain incomplete.
