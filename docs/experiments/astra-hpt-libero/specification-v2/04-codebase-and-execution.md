# 04 — Codebase, implementation order, and resources

## One repository and one simulator integration

Implement within `AstraExps`. Keep the consolidated plan under `plans/` and upstream repositories as pinned dependencies or explicitly recorded narrow patches. Avoid scattering active changes across the existing EgoVerse experiment checkouts. The following layout is proposed; these modules do not yet exist.

```text
AstraExps/
  plans/astra-hpt-libero-signs-of-life/
  pyproject.toml
  locks/{simulator,learner}.lock
  configs/
    experiment/hpt_libero_pilot.yaml
    simulator/libero_push.yaml
    learner/hpt_scratch.yaml
    assets/push_primitives.yaml
  src/astra_vla/
    schemas/{scene,task,teacher,episode,decision,report}.py
    simulator/{libero_push,assets,compiler,bundles,actions}.py
    semantics/{referents,success,novelty,splits}.py
    teachers/{program,push_controller,collector}.py
    generation/{provider,prompts,validation,archive}.py
    data/{hdf5_writer,hpt_dataset,sampling,normalization}.py
    learner/{hpt_adapter,byte_language,masked_flow,init_audit}.py
    learner/{train,policy_server,checkpoints}.py
    feedback/{control_probes,reports,gradient_diagnostic}.py
    orchestration/{coordinator,ledger,budgets,resume}.py
    evaluation/{paired_runner,metrics,bootstrap,report}.py
    cli.py
  tests/{semantics,interfaces,integrity}/
  scripts/{profile_simulator,profile_learner,run_pilot}.py
  artifacts/<run_id>/
    locks/ scenes/ decisions/ episodes/ manifests/
    checkpoints/ probes/ evaluation/ reports/ ledger.sqlite
```

HPT remains the sole learner implementation. Reuse its trunk/stems/head primitives rather than creating another policy family. New adapters cover the random language stem, masked flow loss, seven-dimensional actions, data, and checkpoint/provenance requirements. There is no generic multi-simulator framework in this pilot.

## Processes and dependency boundaries

Use a synchronous coordinator and one simulator worker. Keep the legacy LIBERO/MuJoCo stack and PyTorch/HPT training stack in separately locked Python environments. The simulator sends versioned image/proprioception/instruction arrays to a local learner-policy process and receives a `[10, 7]` action chunk; it executes the first command. Use local sockets or subprocess pipes with JSON metadata and non-pickled arrays. A minimal transport fixture must check shapes, types, timestamps, request IDs, and timeout behavior.

Training and simulation may share the planned GPU sequentially. The scheduler policy and EGL rendering choice must be inspected during implementation; no account, partition, GPU availability, or measured memory fit is assumed here. Record environment/package versions, offscreen rendering settings, hardware, driver, and submitted job IDs. Do not assume a working HPT environment implies a working simulator environment.

Keep SQLite single-writer ownership with the coordinator. Workers emit immutable artifacts and completion records. Publish checkpoints and data manifests atomically. A failed stage cannot erase incurred cost or turn a partial file into a committed artifact.

## Round state machine

```text
checkpoint published -> control report frozen -> decision validated
-> scenes compiled and witnessed -> quotas collected -> data frozen
-> learner updated -> probes complete -> round committed
```

Every transition records input/output hashes and the manifest version. An attempt gets an ID before reset. Replaying an already committed operation cannot increment accepted episode counts. A provider retry has its own transport-attempt ID under a logical request ID. Recovery uses the saved agent response and exact accepted data, not a fresh proposal that changes the experiment retrospectively.

## Ordered work items and acceptance checks

| Work | Build | Evidence required before completion |
|---|---|---|
| W01 | Resolve sources/dependencies, instantiate random HPT, establish actual Astra access | Locked revisions/hashes, offline initialization audit, one real forward/backward update, offscreen render, valid structured provider response |
| W02 | Implement primitive assets and `AstraPush` with canonical actions | All three stages instantiate; 20 reset seeds/stage; camera direction, controller frame/scale, gripper polarity, and state restore checks |
| W03 | Trusted semantics and frozen split manifests | Correct witnesses pass; adversarial wrong-object/target, transient-goal, move-then-return, and lift fixtures fail; split IDs cannot leak |
| W04 | Compiler, novelty, teacher execution, commissioning | Six novel bundles, two per stage; thirty accepted commissioning rollouts within 120 attempts; standalone bundle reload; frozen proprioception statistics |
| W05 | HDF5 recording and HPT loader | Real episode round trip, real mixed-stage batch, instruction pairing, padding mask, student-feature allowlist |
| W06 | Seed collection and warm-start | 120 successful S1 episodes within 480 attempts; fresh audited learner initialized from scratch; 1,000 updates; complete fork checkpoint; 36 control probes |
| W07 | Decision/report contracts and blind uniform arm | Contrasting fixture reports elicit valid agent decisions; malformed quotas are rejected; U never receives learner performance; no deterministic promotion logic |
| W08 | Four-round runner and exact resume | Both arms meet quotas/updates and novelty gates, or explicit incomplete-run artifacts; induced interruption resumes without duplicate acquisitions |
| W09 | Frozen paired evaluation and report | Shared warm-start/U/A evaluated; raw counts, uncertainty, videos, actual cost, scene novelty, and decision evidence |
| W10 | Gradient diagnostic | Twelve scores and controlled twenty-update branches, transfer-loss deltas, source provenance, compute costs, numerical validation |

W07 fixtures can be developed before seed training. W08 requires W01–W07. W01's real-batch check can use a minimal commissioning fixture to expose interface errors; it does not bypass W04's six-scene gate or become production learner training.

For validation, prioritize the scientific failure modes: pretrained-weight leakage, privileged state in learner batches, wrong language/action pairing, ignored padding, task semantics, false scene novelty, budget bypass through retries, and incomplete checkpoint restoration. Do not add large generic test suites that do not resolve those risks.

## Resource and cost plan

Target one GPU with at least 40 GB VRAM, batch size four, and one simulator worker. This is a resource planning choice rather than a benchmarked minimum. Profile memory, step time, simulator steps/second, render latency, policy latency, teacher yield, and disk bytes/episode before the production run. Fix an OOM or compatibility issue as a recorded engineering change; changing architecture or budgets creates a new manifest version.

Use measured values to estimate:

`training time = 5,240 × measured update time`

`collection time <= 3,000 × 150 / measured simulator control steps per second`

`evaluation time <= 708 × 150 × measured evaluation seconds per control step`

The collection bound assumes every attempt runs to horizon; reset/compile/provider overhead is additional. GPU evaluation throughput includes policy inference and need not equal teacher throughput. W02 reset checks, W01 integration checks, retries, and debugging costs are separate ledger categories. The planning estimate is not a wall-clock reservation request.

Raw two-camera storage at full horizon is about 45.2 MB per episode (decimal units) before compression: `150 × 2 × 224 × 224 × 3`. At the cap of 750 accepted commissioning/training episodes, raw camera arrays alone are about 33.9 GB. Saving every failed attempt at full horizon could raise collection camera storage to about 135.5 GB; state records, videos, checkpoint copies, and evaluation recordings are additional. Record the actual retention/compression policy before running; do not silently drop failure evidence to improve reported yield.

## Proposed command sequence

The following command names describe the intended CLI, not runnable commands delivered by this planning task:

```text
avc doctor --manifest ...
avc scenes commission --manifest ...
avc data audit --manifest ...
avc learner warm-start --manifest ...
avc experiment run --manifest ...
avc evaluate --split sealed-test --run RUN_ID
avc diagnostic gradients --run RUN_ID
avc report --run RUN_ID
```

The final report must show both successes and failed gates. Lack of provider access, hardware, or a valid compiler is a concrete implementation blocker, not permission to invent a model ID, substitute a pretrained learner, or label synthetic fixtures as real experiment results.
