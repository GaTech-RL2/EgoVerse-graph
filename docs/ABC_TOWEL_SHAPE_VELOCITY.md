# ABC towel shape and velocity ablations

These recipes compare whether the ARC shape stream benefits from its own
parameters or from one-way attention when the model predicts per-waypoint
velocity. All experiments use the ABC `fold and stack the towels` task and the
existing bimanual ARC tokenizer. Its public action is `(M=100, 28)`: 14 shape
values followed by 14 velocity values at each waypoint. The denoisers split
the streams internally and return the same wide token for the existing
detokenizer and evaluator.

| Recipe | Stream processing | Flow block allocation |
| --- | --- | --- |
| `abc_towels_hpt180_hybrid_velocity_decoder_visual_openloop` | Shared HPT trunk; a separate condition decoder and FM head predict velocity | Shape 6 + velocity 4 |
| `abc_towels_hpt180_hybrid_shared_directional_visual_openloop` | Shared transformer parameters; velocity reads shape tokens, shape is blocked from velocity | Shared 10 |
| `abc_towels_hpt180_hybrid_mot_parallel_visual_openloop` | Independent shape and velocity transformers, followed by a shared bidirectional FM head | Shape 3 + velocity 3 + fusion 4 |
| `abc_towels_hpt180_hybrid_mot_directional_visual_openloop` | Same MoT branches, with velocity-to-shape access only in the shared FM head | Shape 3 + velocity 3 + fusion 4 |
| `abc_towels_hpt180_baseline_param_matched_visual_openloop` | Cartesian HPT with no ARC tokenizer | Shared 10 |

All five use a 640-wide, 19-block HPT trunk. The FM capacity is budgeted as ten
384-wide `CrossBlock`s in every model; the velocity-decoder model also has a
small condition MLP, and stream models have learned stream identities. The
launch manifest records instantiated trainable parameter counts so any
remaining difference is visible.

ARC runs keep the same joint flow-matching objective as full-token MSE. The
ARC-specific loss stage logs separate shape and waypoint-velocity MSEs; their
arithmetic mean is the trained loss, so the objective remains unchanged.
Normalization is precomputed separately for Cartesian and ARC actions with
the same task filter, split seed, and 10% sample fraction before training.
