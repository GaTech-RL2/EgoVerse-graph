# ARC Throughput Optimization With Exact Token Parity

## Status

Approved in chat on 2026-09-24.

## Goal

Remove the dominant CPU and launch overhead from ARC training without changing
the ARC representation. For every source sample and tokenizer configuration,
the optimized implementation must produce the same token bytes as the current
implementation.

## Non-negotiable compatibility contract

- ARC token shape, dtype, row order, values, timing payload, gripper values,
  preserved source actions, and detokenized behavior remain unchanged.
- `translation_horizon_mode="joint"` and `"race"` both remain supported.
- `M=100` with per-waypoint velocity remains a `(200, 14)` token.
- Dynamic source horizons return the same integer row count as before.
- Token parity is bit-for-bit: `np.array_equal(reference, optimized)` must pass.
  A numerical tolerance is not an acceptable substitute.
- If a proposed vectorized rotation operation changes any output bit, retain
  the existing rotation operation and optimize only the surrounding lookup,
  allocation, and data flow.
- Existing normalization statistics and checkpoints remain semantically valid
  because the action representation does not change.

## Current bottlenecks

The current YAM key map attaches the same dynamic horizon specification to four
action keys. `ZarrDataset.__getitem__` resolves each occurrence separately, so
one sample performs four identical horizon scans and each scan rereads both arm
pose streams across as many as 600 rows.

Hybrid tokenization then performs multiple scalar target searches and repeated
position, gripper, and rotation interpolation. It first creates the legacy
per-arm waypoints and then recomputes hybrid waypoints, discarding work. A local
600-row, `M=100` benchmark measured about 37.6 ms per joint-hybrid sample.

The active Lambda recipes also use four loader workers, non-persistent workers,
FP32, and one-rank `ddp_find_unused_parameters_true`. This design changes loader
and one-GPU strategy settings, but deliberately leaves precision unchanged.

## Design

### 1. Resolve a shared dynamic horizon once

During sample loading, collect all dynamic horizon dictionaries from the key
map and canonicalize their contents. Resolve each unique specification once,
then reuse its integer horizon for every key that references an equivalent
specification.

The resolver API and returned integer do not change. Conflicting dynamic
horizon specifications still raise an error when they resolve to different
lengths. A reader-spy regression test must prove that the four YAM action keys
cause one two-arm horizon read rather than four.

This optimization changes only call multiplicity. It must not cache horizons
across dataset samples or epochs because episode data and fallback indices are
part of the sample context.

### 2. Fuse and vectorize interpolation under a reference oracle

Freeze the pre-optimization tokenizer as a test oracle before changing
production code. The corpus covers:

- straight and curved bimanual motion;
- left-first, right-first, simultaneous, fractional, and absent race crossings;
- joint translation clocks;
- stationary intervals and one stationary arm;
- hybrid rotation caps and Euler wraparound;
- changing grippers;
- short source tails;
- mean, duration, and per-waypoint timing layouts where currently supported.

Vectorize scalar bracket searches with one `np.searchsorted` operation over all
targets. Reuse cumulative translation and rotation arrays instead of rebuilding
them in waypoint and timing passes. Fill position and gripper arrays in batches
while preserving the current arithmetic order for each output element.

Construct hybrid output directly instead of first materializing waypoint blocks
that hybrid mode immediately replaces. Shared intermediate arrays may be reused
only when doing so preserves token bytes.

Rotation interpolation is optimized conservatively. Batch or reuse SciPy
rotation objects only if every golden case remains bit-identical. Otherwise,
the current pairwise SLERP call remains the reference implementation.

### 3. ARC loader settings

Retained ARC experiment configurations use these training-loader defaults:

- `num_workers: 12`
- `persistent_workers: true`
- `prefetch_factor: 4`
- `pin_memory: true`

Validation uses eight workers with the same persistent, prefetch, and pinned
memory settings. Baseline recipes are not changed by this feature.

Hydra composition tests verify the effective settings. A dataset parity test
loads the same deterministic sample with zero workers and with the optimized
worker settings and requires identical ARC token arrays. Different worker
counts may alter the order in which future random augmentations consume RNG,
but they may not alter a token for a fixed source sample.

### 4. Remove one-rank DDP overhead

Single-GPU launches use `trainer.strategy=auto`, which selects Lightning's
single-device path and avoids NCCL initialization and unused-parameter graph
traversal. Multi-GPU launches retain their explicitly reviewed DDP strategy;
this feature does not silently rewrite a multi-GPU launch.

The launcher/preflight layer resolves strategy from world size:

- world size `1`: `auto`;
- world size greater than `1`: the configured multi-GPU strategy.

A one-GPU smoke test verifies that no distributed process group is initialized.
This change cannot affect ARC token construction because it occurs after data
loading, but it does change training execution and therefore requires a fresh
W&B identity when used for a new run.

## Data flow after optimization

1. The sampler selects one dataset index.
2. The dataset resolves each unique dynamic horizon specification once.
3. All action keys read the same resolved source window.
4. The transform computes cumulative clocks once.
5. Batched target lookup and interpolation produce the ARC token.
6. The golden parity gate compares the token to the frozen reference behavior.
7. The loader prefetches subsequent samples while the GPU trains.

## Validation and performance gates

Correctness gates:

- exact token parity across the golden corpus;
- exact preserved-source and horizon parity;
- existing tokenizer, resolver, evaluator, and Hydra composition suites pass;
- fixed-sample parity across loader worker settings;
- no existing run or checkpoint is modified during development.

Performance gates:

- one shared YAM horizon scan per sample instead of four;
- optimized tokenizer median time is at most 70% of the reference median on the
  checked-in synthetic benchmark corpus;
- report loader batches/second for 4 and 12 workers before adopting defaults;
- report one-GPU steps/second with and without one-rank DDP.

If an optimization misses its performance gate, it is not enabled by default.
If it misses any parity gate, it is rejected regardless of speed.

## Rollout safety

- Development occurs only on `codex/arc-race-mode`.
- Lambda job `405895` remains untouched.
- No cluster checkout, training run, W&B run, or normalization cache is changed
  by this implementation task.
- Cluster synchronization and relaunch require separate explicit approval after
  local tests and benchmarks pass.

## Out of scope

- BF16 or other precision changes;
- offline token caches;
- changing `D`, `R`, `M`, source horizon, or execution cap semantics;
- changing validation metrics or videos;
- restarting current training jobs.
