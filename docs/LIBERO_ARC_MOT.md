# ARC modality-specific Transformer comparison

The previous shared-decoder control splits shape and timing into different
tokens but shares the Transformer parameters. These three MoT variants give
each modality its own self-attention Q/K/V/output projections, observation
cross-attention, feed-forward layers, and normalization. Projected action
keys/values are combined for attention across modalities, following the
[Mixture-of-Transformers design](https://github.com/facebookresearch/Mixture-of-Transformers).
The observation/time memory is common and never updated from action tokens.

| Variant | Action modalities | Cross-modality attention | Width / FFN | Action parameters |
| --- | --- | --- | --- | ---: |
| `mot_xyz_rot_gripper` | xyz + translation timing; rotation + rotation timing; gripper | All directions, causal by waypoint | 160 / 536 | 4,790,284 |
| `mot_shape_velocity` | xyz + rotation + gripper; translation + rotation timing | Both directions, causal by waypoint | 192 / 672 | 4,787,532 |
| `mot_shape_velocity_masked` | Same two modalities | Shape queries cannot read velocity keys; velocity can read shape | 192 / 672 | 4,787,532 |

All three have four layers per modality and four attention heads. All modules
are active; there is no unused parameter padding. The existing split-token
control has 4,791,564 action parameters, so these models match that budget
within 0.1%. The observation encoder remains 22,394,248 parameters. Width and
FFN size change to hold total capacity approximately fixed: this is a
comparison of architectures at a common parameter budget, not a comparison
with identical layer dimensions. The two shape/velocity arms differ only in
the mask and have identical parameter initialization for the same seed.

The external ARC tensor remains `(B, M, 12)` in the existing checkpoint order:
`[xyz3, translation_timing, rotation6d6, rotation_timing, gripper]`. The word
velocity refers to the existing two STK timing/progress channels, not new
Cartesian velocity labels. The xyz/rotation/gripper arm uses `3M` internal
tokens; the other arms use `2M`. Attention masks use waypoint indices, so
different modalities at the same waypoint can communicate in either token
order. The mask applies at every layer and every diffusion step.

## Frozen Spatial protocol

- STK2 tokenizer: R=192 degrees, D=0.8 m, M=32, unchanged physical ARC decoding.
- Audited replay: `arc-five-v2-20260921-stk2-libero-spatial-replay`.
- Seed 42; 5,001 epochs; 270,054 optimizer updates; global batch 1,024.
- Eight L40S GPUs per arm, microbatch 128, using the decoded replay loader.
- Same observation encoder, epsilon loss, AdamW groups and EMA schedule as
  the existing OAT-DP ARC comparison.
- Two observation frames, horizon 32, ten DDIM steps; execute 16 actions then
  replan. Evaluation uses final EMA weights.
- Each training workflow includes an eight-GPU training/checkpoint smoke
  test and a short simulator rollout on all ten Spatial tasks before the full
  training budget starts. Its dependent evaluation starts automatically when
  the final checkpoint is verified.
- Evaluation: ten tasks, 50 trials per task, five repetitions = 2,500 episodes,
  maximum 550 steps. One training seed, five evaluation repetitions.
- Pair against `arc-dpr-20260925-stk2-spatial` using identical episode IDs,
  initial-state hashes, observation/data context, and rollout protocol.

Existing completed Spatial references are original joint-token ARC 76.16%,
split-token shared decoder 70.40%, separate decoders 76.04%, and split-token
shared decoder with shape-to-timing attention blocked 77.12%. MoT success
rates remain pending until their full paired evaluations complete.

## Implementation and reproduction

The generic model is `egomimic/models/modality_diffusion.py`. Modality partitions,
mask directions and capacity settings live in `hydra_configs/arc_decoder/`.
The shared graph, trainer, replay loader, physical decoder and rollout runner
are reused. Checkpoint validation checks the exact saved modality partition,
mask and model dimensions before evaluation or resuming training.

Select `+experiment=oat/libero_arc_mot_policy` and
`arc_decoder=mot_xyz_rot_gripper`, `mot_shape_velocity`, or
`mot_shape_velocity_masked`. The existing OSMO launcher accepts these names
through `--arc-decoder-variant` with `--arc-profile stk_2 --arc-modes stk`.
Use the audited replay and paired reference run above for full runs.

The [launch manifest](arc_mot_runs_20260930.json) records the immutable source,
OSMO workflow links, resource allocations, specification checksums and paired
reference checksums for the three submitted runs. A separate CPU collector
publishes `latest.json`, `table.md` and final `results.json` under
`s3://rldb/experiments/arc-oat-20260919/campaigns/arc-mot-20260930-spatial/`.
It checks the exact runtime architecture, full optimizer/EMA budget and
reference episode hashes before accepting a completed score.

Tests compare the attention operation and gradients with PyTorch's decoder
when experts have identical weights, exercise all cross-modality paths and
forbidden gradients, check temporal causality and output channel order, test
all ten DDIM steps for masking leaks, and exercise actual shared training,
optimizer groups, EMA checkpoint resume and physical-action inference.
All 75 focused CPU tests pass. GPU/simulator preflights are enforced by each
workflow before full training, and their receipts are uploaded with the run.
