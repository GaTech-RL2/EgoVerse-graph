# Native image/proprio temporal-pair admission

Fresh image/proprio runs with discrete and continuous temporal representations do not satisfy the existing modality-factorial or optimizer-parent comparison contracts. `verify_temporal_pair.py` adds an explicit contract for this pair without fabricating either parent proof.

The v1 profile pins scientific source `535d4fc`, seed 42, four ranks with eight examples each, global batch 32, constant AdamW LR `1e-4`, and 300k updates. Checkpointing and validation occur every 20k updates, retaining all 15 immutable checkpoints. An exact override allowlist prevents unrequested scientific changes, including changes shared by both arms. Discrete/continuous pose dimensions 36/9 and token limits 3,216/12,864 differ; every other composed scientific value must match after checked output and identity substitutions.

Both actual GPU smoke proofs are independently revalidated, including 50 optimizer updates, 100 scheduled validation anchors, every rank's LR and precision checks, strict reload, and sampling proofs. All eight ranks must share the same initial image weights. Training-draw hashes, whole-model initial weights and data-context hashes are retained separately; they are not claimed to be paired across temporal representations.

Two immutable full-request prototypes omit `temporal_comparison`. The verifier's receipt binds their original files and canonical complete request hashes. Final requests add only that receipt binding. Admission revalidates both prototypes, both smoke proof sets, both configurations and the complete receipt before requiring exact request membership. The driver requires exactly one comparison contract.

This profile supports fresh single-allocation requests with identical runtime and data-cache bindings. It rejects continuations and allocation-boundary requests. The temporal binding is included in continuation stable fields for future extensions, but recovery and different node/cache bindings still require separately reviewed admission support.

The 49 focused tests use configurations composed from frozen source `535d4fc` and synthetic smoke evidence fixtures. They do not certify GPU execution, runtime readiness, deployment or production training. Before deployment, bind a published service export; complete allocated runtime, data and model preflight; obtain two new exact four-rank smoke passes at LR `1e-4`; report measured timings; and refresh ownership, queues, claims, storage, W&B identities and output-collision checks. The scientific archive remains independent and unchanged.
