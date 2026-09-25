# ARC Throughput With Exact Token Parity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the dominant CPU and one-rank distributed overhead from ARC training while preserving every ARC token byte for both `joint` and `race` translation-horizon modes.

**Architecture:** Freeze the current tokenizer as a golden reference, then optimize one boundary at a time: deduplicate per-sample horizon resolution, reuse cumulative clocks and batched bracket lookups inside the tokenizer, increase retained ARC recipe loader concurrency, and select Lightning's single-device strategy for one-GPU runs. Exact `np.array_equal` tests remain the release gate; scalar pairwise SLERP stays in place wherever SciPy's batched rotation path is not bit-identical.

**Tech Stack:** Python 3, NumPy, SciPy Rotation/Slerp, PyTorch DataLoader, Lightning, Hydra/OmegaConf, pytest.

**Spec:** `docs/superpowers/specs/2026-09-24-arc-throughput-token-parity-design.md`

## Global Constraints

- ARC token shape, dtype, row order, values, timing rows, gripper values, preserved source actions, and detokenized behavior must remain unchanged.
- `translation_horizon_mode="joint"` and `translation_horizon_mode="race"` must both remain supported.
- `M=100` with per-waypoint velocity must remain a `(200, 14)` token.
- Dynamic source horizons must return the same integer row count as the reference implementation.
- Correctness requires `np.array_equal(reference, optimized)`; tolerance-based comparisons are insufficient.
- Preserve the existing scalar pairwise SLERP if a batched rotation implementation changes any output bit.
- Do not change `D`, `R`, `M`, source horizon, execution cap, precision, validation, checkpoint, normalization, or W&B semantics.
- Do not modify or restart job `405895`, synchronize clusters, or mutate remote checkouts in this implementation.
- Retained ARC train loaders use 12 workers, persistent workers, prefetch factor 4, and pinned memory; retained ARC validation loaders use the same settings with 8 workers.
- Single-GPU runs use `trainer.strategy=auto`; multi-GPU runs retain their explicitly configured strategy.
- The optimized tokenizer median must be at most 70% of the frozen reference median on the deterministic synthetic benchmark corpus before the implementation is enabled.

## File Structure

- `tests/arc_token_parity_fixtures.py`: deterministic, hand-defined ARC source trajectories and frozen SHA256/token-byte expectations shared by parity and benchmark tests.
- `tests/test_arc_token_parity.py`: exact token, preserved-source, shape, dtype, horizon, and benchmark gates for joint/race and all supported timing layouts.
- `tests/test_zarr_dynamic_horizon_dedup.py`: reader-spy regression coverage for once-per-unique-spec horizon resolution and conflict preservation.
- `tests/test_arc_loader_runtime.py`: composed Hydra recipe assertions, fixed-sample worker parity, and single-device strategy smoke coverage.
- `egomimic/rldb/zarr/zarr_dataset_multi.py`: per-sample canonicalization and deduplication of equivalent dynamic-horizon specs.
- `egomimic/rldb/zarr/arc_length_tokenizer.py`: shared ARC sampling state, batched bracket/time computation, direct hybrid waypoint assembly, and conservative rotation interpolation.
- `egomimic/pl_utils/pl_data_utils.py`: DataLoader parameter validation needed to make persistent workers safe when a test or override selects zero workers.
- `egomimic/trainHydra.py`: world-size-aware trainer strategy resolution before Trainer instantiation.
- `egomimic/hydra_configs/data/abc_arc/*.yaml`: loader defaults for the four retained hybrid ARC experiment families only.
- `scripts/benchmark_arc_tokenizer.py`: deterministic reference-versus-optimized median timing report used by the performance gate.

---

### Task 1: Freeze the ARC Reference Oracle and Golden Corpus

**Files:**
- Create: `tests/arc_token_parity_fixtures.py`
- Create: `tests/test_arc_token_parity.py`
- Create: `scripts/benchmark_arc_tokenizer.py`
- Read: `egomimic/rldb/zarr/arc_length_tokenizer.py`

**Interfaces:**
- Consumes: `TokenizeBimanualArcLengthCartesian.transform(batch: dict) -> dict` and `truncate_bimanual_translation_race(actions, distance, epsilon) -> np.ndarray`.
- Produces: `ARC_CASES: tuple[ArcParityCase, ...]`, `reference_transform(case) -> tuple[np.ndarray, np.ndarray | None]`, and `benchmark_cases() -> tuple[ArcParityCase, ...]` for later tasks.

- [ ] **Step 1: Add deterministic source-trajectory fixtures**

  Define an immutable `ArcParityCase` dataclass with `name`, `actions`, `distance`, `rotation_distance`, `M`, `velocity_mode`, `translation_horizon_mode`, `preserve_rows`, `expected_token_sha256`, and `expected_preserved_sha256`. Construct literal-seeded trajectories covering straight and curved motion, stationary intervals, one stationary arm, changing grippers, Euler wraparound, short tails, joint mode, and race mode with left-first, right-first, simultaneous, fractional, and absent crossings. Include `mean`, `duration`, and `per_waypoint` where supported.

- [ ] **Step 2: Freeze current outputs as independent byte fixtures**

  Run the unmodified tokenizer once for every case and record SHA256 values of `dtype.str`, `shape`, and C-order bytes as literals in `ARC_CASES`. `reference_transform` must invoke an unmodified reference copy captured before Task 3, return copied arrays, and never call optimized helpers.

- [ ] **Step 3: Write exact characterization tests**

  For every case assert the output dtype, output shape, literal SHA256, `np.array_equal(actual, reference)`, preserved-source SHA256, and `(200, 14)` for `M=100/per_waypoint`. Add literal horizon expectations for all race crossing cases.

- [ ] **Step 4: Run the characterization tests against current production code**

  Run: `pytest -q tests/test_arc_token_parity.py`

  Expected: PASS, proving the corpus accurately freezes existing behavior before optimization.

- [ ] **Step 5: Add the benchmark harness without a pass/fail claim yet**

  Implement `python scripts/benchmark_arc_tokenizer.py --repeat 15 --json` to warm each case, measure per-case and aggregate medians with `time.perf_counter_ns`, and report `reference_ms`, `optimized_ms`, and `ratio`. The script must import the same deterministic corpus and compare arrays before timing.

- [ ] **Step 6: Commit the oracle**

  ```bash
  git add tests/arc_token_parity_fixtures.py tests/test_arc_token_parity.py scripts/benchmark_arc_tokenizer.py
  git commit -m "test(arc): freeze exact tokenizer parity corpus"
  ```

### Task 2: Resolve Equivalent Dynamic Horizons Once Per Sample

**Files:**
- Create: `tests/test_zarr_dynamic_horizon_dedup.py`
- Modify: `egomimic/rldb/zarr/zarr_dataset_multi.py:2250-2285`

**Interfaces:**
- Consumes: `ZarrDataset._resolve_dynamic_horizon(start_idx: int, spec: dict) -> int`.
- Produces: `_canonical_dynamic_horizon_spec(spec: dict) -> tuple` and `_resolve_sample_dynamic_horizons(start_idx: int) -> dict[tuple, int]`; `__getitem__` looks up each action key's canonical spec in that per-sample mapping.

- [ ] **Step 1: Write the failing reader-spy test**

  Build a minimal real `ZarrDataset` instance with four YAM action keys sharing equivalent-but-distinct horizon dictionaries and an episode reader that records pose-window reads. Fetch one sample and assert one two-arm horizon read rather than four. Name the break: restoring per-key `_resolve_dynamic_horizon` calls makes the count become four.

- [ ] **Step 2: Verify the reader-spy test fails for the expected reason**

  Run: `pytest -q tests/test_zarr_dynamic_horizon_dedup.py::test_equivalent_yam_horizons_scan_pose_streams_once`

  Expected: FAIL with four resolver pose reads observed instead of one.

- [ ] **Step 3: Add canonical per-sample resolution**

  Canonicalize nested dictionaries, lists/tuples, NumPy scalar values, and pose-key order into a hashable tuple without mutating the source spec. Resolve each unique canonical spec once inside the current retry-loop iteration, map every dynamic key to that result, and discard the mapping whenever fallback changes `idx`.

- [ ] **Step 4: Preserve conflict behavior with a second failing test**

  Add two genuinely different dynamic specs that resolve to different lengths and assert the existing `ValueError("multiple dynamic horizon specs resolved to different lengths ...")`. Run the isolated test before implementation and verify it fails if unique-spec handling silently chooses one result.

- [ ] **Step 5: Implement and verify conflict handling**

  Compare all unique resolved lengths before reads; preserve the exact error condition while allowing equivalent specs to share work.

- [ ] **Step 6: Run horizon and ARC characterization suites**

  Run: `pytest -q tests/test_zarr_dynamic_horizon_dedup.py tests/test_yam_hybrid_arc.py tests/test_arc_token_parity.py`

  Expected: PASS with exact golden hashes unchanged.

- [ ] **Step 7: Commit horizon deduplication**

  ```bash
  git add egomimic/rldb/zarr/zarr_dataset_multi.py tests/test_zarr_dynamic_horizon_dedup.py
  git commit -m "perf(data): resolve shared ARC horizon once per sample"
  ```

### Task 3: Reuse ARC Clocks and Batch Translation Interpolation

**Files:**
- Modify: `tests/test_arc_token_parity.py`
- Modify: `egomimic/rldb/zarr/arc_length_tokenizer.py:1183-1648`
- Modify: `scripts/benchmark_arc_tokenizer.py`

**Interfaces:**
- Consumes: `_bracket_segments(cumdist, targets) -> tuple[np.ndarray, np.ndarray]`, scalar `_interp_ypr_at_s`, and the Task 1 oracle.
- Produces: `_ArcSamplingState` containing source arrays, cumulative translation/rotation clocks, targets, bracket indices, alphas, and fractional source times; `_linear_segments(values, indices, alpha) -> np.ndarray`; hybrid waypoint and velocity methods accept/reuse this state.

- [ ] **Step 1: Add failing lookup-count tests**

  Instrument cumulative-clock and bracket helpers while transforming a joint-hybrid and race-hybrid case. Assert each required cumulative clock is built once and each target array is bracketed once. Verify the tests fail because current waypoint and velocity paths rebuild clocks and loop through scalar lookups.

- [ ] **Step 2: Add exact helper parity tests before replacing callers**

  Test `_linear_segments` against literal scalar formulas for endpoint, fractional, repeated-distance, and zero-length segments with `np.array_equal`. Add a rotation probe that compares `_slerp_segments_ypr` to repeated `_interp_ypr_at_s` using `np.array_equal`; the production path may use the batched helper only for cases where this exact test passes.

- [ ] **Step 3: Verify the helper tests fail because the sampling-state API is absent**

  Run: `pytest -q tests/test_arc_token_parity.py -k 'sampling_state or linear_segments or rotation_probe'`

  Expected: FAIL on missing `_ArcSamplingState`/`_linear_segments`, with no fixture or import errors.

- [ ] **Step 4: Implement a once-per-transform sampling state**

  Build the state after race truncation and before output assembly. Compute only clocks needed by the configured mode. Use `_bracket_segments` once per target set and derive source times as `(indices + alpha) * dt` with the same float64 operation order as the scalar code.

- [ ] **Step 5: Assemble joint-hybrid waypoints directly**

  For joint hybrid, skip `BimanualArcLengthTokenizer.tokenize(raw)` because its translation waypoints are discarded. Batch xyz and gripper interpolation with `_linear_segments`. Keep repeated scalar `_interp_ypr_at_s` calls unless the exact rotation probe proves the batched implementation identical on the full corpus.

- [ ] **Step 6: Reuse state for joint-hybrid velocity rows**

  Pass cached translation and rotation source times to `_hybrid_per_waypoint_velocity`; preserve existing `np.diff`, safe-division, final-row repetition, and angular-rate operation order.

- [ ] **Step 7: Reuse state for race-hybrid output without changing per-arm semantics**

  Retain the base tokenization required for independent per-arm translation waypoints, replace repeated per-arm timing searches with cached bracket/time arrays, and share the single rotation clock across both arms. Preserve truncated raw actions and preserved-source rows exactly.

- [ ] **Step 8: Run exact parity after each caller conversion**

  Run: `pytest -q tests/test_arc_token_parity.py tests/test_arc_length_tokenizer.py tests/test_yam_hybrid_arc.py tests/test_arc_eval_detokenize.py`

  Expected: PASS; any changed SHA256 or failed `np.array_equal` requires reverting that specific vectorized operation to its scalar reference path.

- [ ] **Step 9: Enforce the performance gate**

  Run: `python scripts/benchmark_arc_tokenizer.py --repeat 15 --json`

  Expected: every parity check passes and aggregate `ratio <= 0.70`. If the ratio is greater than `0.70`, do not enable the optimized production path; retain the benchmark output and continue profiling instead of weakening the gate.

- [ ] **Step 10: Commit tokenizer optimization**

  ```bash
  git add egomimic/rldb/zarr/arc_length_tokenizer.py tests/test_arc_token_parity.py scripts/benchmark_arc_tokenizer.py
  git commit -m "perf(arc): reuse clocks and batch exact interpolation"
  ```

### Task 4: Enable Retained ARC Loader Concurrency Safely

**Files:**
- Create: `tests/test_arc_loader_runtime.py`
- Modify: `egomimic/pl_utils/pl_data_utils.py:243-304`
- Modify: `egomimic/hydra_configs/data/abc_arc/stationery_rl2_hpt_arc_hybrid_D40_M100_R24deg.yaml`
- Modify: `egomimic/hydra_configs/data/abc_arc/abc_towels_hpt180_arc_D40_M100.yaml`
- Modify: `egomimic/hydra_configs/data/abc_arc/abc_mecka_fold_multitask_arc_cotrain_D40_M100.yaml`
- Modify: `egomimic/hydra_configs/data/abc_arc/mecka_fold_clothes_40h_human_hybrid_D40_M100_R24deg.yaml`

**Interfaces:**
- Consumes: `MultiDataModuleWrapper.train_dataloader()` and `val_dataloader()` plus retained Hydra experiment composition.
- Produces: `_normalized_loader_params(params: Mapping) -> dict` that removes `persistent_workers` and `prefetch_factor` only when `num_workers == 0`, preserving explicit multi-worker values.

- [ ] **Step 1: Write failing Hydra composition assertions**

  Compose all four retained ARC experiments and assert every train source has `num_workers=12`, `persistent_workers=true`, `prefetch_factor=4`, and `pin_memory=true`; assert every validation source/group has the corresponding values with `num_workers=8`.

- [ ] **Step 2: Verify composition assertions fail on current six-worker recipes**

  Run: `pytest -q tests/test_arc_loader_runtime.py::test_retained_arc_recipes_use_prefetch_loader_defaults`

  Expected: FAIL showing `num_workers=6` and missing persistent/prefetch/pin fields.

- [ ] **Step 3: Update only retained ARC data recipes**

  Add explicit loader settings to each train and validation source, including every nested validation group in the multitask recipe. Do not alter baseline recipes or batch sizes.

- [ ] **Step 4: Write and verify worker-zero safety test**

  Instantiate a datamodule with `num_workers=0`, `persistent_workers=true`, and `prefetch_factor=4`; assert loader construction succeeds with those multiprocessing-only options removed. Verify the test fails before adding `_normalized_loader_params`.

- [ ] **Step 5: Implement loader normalization and fixed-sample parity test**

  Normalize params immediately before every DataLoader constructor. Load the same deterministic ARC sample once through a zero-worker loader and once through the optimized worker settings, then assert `np.array_equal` for token and preserved-source arrays.

- [ ] **Step 6: Run loader, composition, and token suites**

  Run: `pytest -q tests/test_arc_loader_runtime.py tests/test_abc_arc_experiment_layout.py tests/test_arc_token_parity.py`

  Expected: PASS with identical fixed-sample bytes.

- [ ] **Step 7: Commit loader changes**

  ```bash
  git add egomimic/pl_utils/pl_data_utils.py egomimic/hydra_configs/data/abc_arc tests/test_arc_loader_runtime.py
  git commit -m "perf(arc): prefetch retained ARC data loaders"
  ```

### Task 5: Select Single-Device Strategy for One-GPU Runs

**Files:**
- Modify: `tests/test_arc_loader_runtime.py`
- Modify: `egomimic/trainHydra.py:620-670`

**Interfaces:**
- Consumes: composed `cfg.trainer.devices`, `cfg.trainer.num_nodes`, and configured `cfg.trainer.strategy`.
- Produces: `_resolve_trainer_strategy(cfg: DictConfig) -> str`, called before `hydra.utils.instantiate(cfg.trainer, ...)`.

- [ ] **Step 1: Write failing strategy-resolution tests**

  Assert one device on one node returns and writes `auto`; assert two devices or two nodes preserve the exact configured multi-GPU strategy string. Name the break: unconditional `auto` on multi-GPU or retained one-rank DDP must fail.

- [ ] **Step 2: Verify the strategy tests fail because the resolver is absent**

  Run: `pytest -q tests/test_arc_loader_runtime.py -k trainer_strategy`

  Expected: FAIL on missing `_resolve_trainer_strategy`.

- [ ] **Step 3: Implement world-size-aware strategy resolution**

  Resolve integer device count and node count from the composed config, reject non-positive values, set `cfg.trainer.strategy = "auto"` only when their product is one, and otherwise leave the configured strategy untouched. Call this after eval-mode trainer overrides and before Trainer instantiation.

- [ ] **Step 4: Add a one-device Lightning smoke test**

  Instantiate a CPU `Trainer` through the resolved config with logging/checkpointing disabled, assert its strategy is Lightning's single-device strategy, and assert `torch.distributed.is_initialized()` is false. Also instantiate or inspect a two-device config and assert its DDP strategy value remains unchanged without launching processes.

- [ ] **Step 5: Run trainer and launch-config suites**

  Run: `pytest -q tests/test_arc_loader_runtime.py tests/test_train_hydra_slurm_environment.py tests/test_lambda_launcher.py`

  Expected: PASS; no distributed process group exists after the one-device smoke test.

- [ ] **Step 6: Commit strategy selection**

  ```bash
  git add egomimic/trainHydra.py tests/test_arc_loader_runtime.py
  git commit -m "perf(train): avoid DDP strategy for one GPU"
  ```

### Task 6: Full Verification and Performance Report

**Files:**
- Modify if required by measured evidence: `docs/superpowers/specs/2026-09-24-arc-throughput-token-parity-design.md`
- Verify: all files changed in Tasks 1-5

**Interfaces:**
- Consumes: exact parity corpus, benchmark harness, Hydra compositions, and strategy smoke test.
- Produces: reproducible verification evidence and a clean branch ready for review; no remote mutation.

- [ ] **Step 1: Run the focused correctness suite**

  Run: `pytest -q tests/test_arc_token_parity.py tests/test_zarr_dynamic_horizon_dedup.py tests/test_arc_loader_runtime.py tests/test_arc_length_tokenizer.py tests/test_yam_hybrid_arc.py tests/test_arc_eval_detokenize.py tests/test_abc_arc_experiment_layout.py`

  Expected: PASS with zero failures and no token-hash drift.

- [ ] **Step 2: Run the broader repository regression suite**

  Run: `pytest -q`

  Expected: PASS. If unrelated pre-existing failures occur, record exact node IDs and rerun every affected ARC/trainer test independently before reporting status.

- [ ] **Step 3: Run deterministic tokenizer benchmark**

  Run: `python scripts/benchmark_arc_tokenizer.py --repeat 15 --json`

  Expected: exact parity for every case and aggregate optimized/reference median ratio `<= 0.70`.

- [ ] **Step 4: Measure loader scaling locally**

  Run the deterministic fixed-sample loader benchmark at 4 and 12 workers for enough batches to exclude startup, report batches/second, and confirm token SHA256 remains unchanged. Do not claim a cluster throughput improvement from this local measurement.

- [ ] **Step 5: Review the compatibility contract line by line**

  Confirm exact token bytes, shape, dtype, row order, timing rows, gripper values, preserved source, horizon count, joint/race coverage, `(200, 14)` per-waypoint layout, loader fixed-sample parity, and single-/multi-GPU strategy behavior. Inspect `git diff --check` and `git status --short`.

- [ ] **Step 6: Commit any verification-only documentation**

  ```bash
  git add docs/superpowers/specs/2026-09-24-arc-throughput-token-parity-design.md
  git commit -m "docs(arc): record throughput parity verification"
  ```

  Skip this commit when no documentation changed.

