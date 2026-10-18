# Images-only pixel video pilots

The initial experiment uses LIBERO gripper RGB only: two cameras at 160×160,
with 16×16 RGB patches (100 tokens per camera). There is no pretrained image
encoder, point track input, proprioception conditioning, state head, or action
head in stage1. Camera IDs, spatial positions, and the selected time encoding
remain. Each stage1 model has one image expert, width 768, depth 8, six attention
heads, and 88,178,688 parameters. The current model uses per-frame AdaLN-Zero
noise conditioning and spatial/temporal RoPE (64 time, 32 x, 32 y channels per
head); it has no additive XY or learned phase embedding.

The two recipes keep the same frozen 450-training/50-validation episode split,
with 45/5 episodes per task. Discrete training selects a complete 64-frame
window and selects 16 timestamps at stride 4, producing 3,200 tokens. Continuous
training samples 2–min(64, episode length) timestamps across the whole episode,
producing at most 12,800 tokens. This caps the number of selected timestamps,
not their temporal span. The unchanged model capacity ceiling remains 25,600
tokens; deferred stage2 retains its separately configured 128-timestamp cap.
Both cameras are retained at each selected
timestamp. Training anchors are uniform over eligible recorded frames. Equal
optimizer-step budgets therefore do not imply equal token or attention work.

The discrete and continuous full-training recipes use AdamW at LR 5e-5,
weight decay 0.01, and 300,000 optimizer steps each. The first major quality
review is at 100,000 steps; it does not automatically stop either run.
Only these two recipe defaults change the total budget. The shared pilot base,
explicit smoke step overrides, and deferred stage2 budgets remain unchanged.
The allocation profile `+pixel_execution=ddp4` runs each pilot on four GPUs on
one node. Its reusable default is batch 1 per GPU; the selected full runs use
the explicit override `data.batch_size=8`, retaining global batch 32 with no
gradient accumulation or learning-rate scaling. Two separate four-GPU runs share the allocated eight-GPU
node. DDP uses one image expert with all trainable parameters used; automatic
sampler replacement is disabled because the pixel adapter owns sharding.
The explicit TorchElastic environment reads ranks from each run's `torchrun`
process group, while Slurm IDs remain available for artifact provenance.
Validation covers two fixed anchors from each of the 50 held-out
episodes every 20,000 optimizer steps, with identical 16-timestamp evaluation grids in both arms.
The 100 global validation batches are divided into four contiguous shards of
25 without padding or duplicated examples. Sampling noise uses global batch
IDs so rank assignment does not change it. Metrics sum squared errors and
valid-target counts across ranks before dividing; RGB diagnostics are written
only by rank zero, whose shard contains both forecast and completion examples.
Retain the 20,000-step checkpoints: a fresh run produces 15 scheduled
checkpoints through step 300,000, including the 100,000-step review point.
`save_last: link` makes `last.ckpt` a symbolic link to the latest saved
checkpoint on the local filesystem, including mounted NFS paths. Removing
older checkpoints outside these milestones is a separate operation after
verifying the resumed trainer and retained checkpoints; this configuration
does not delete existing files. Changing these defaults does not update an
active trainer. A resumed run must explicitly select the new validation and
checkpoint intervals and linked `last` while restoring its optimizer, step,
sampler, and per-rank RNG states.
FP32 weights and two AdamW moment tensors alone use approximately 1.06 GB per
checkpoint for this model, or 15.9 GB per fresh run and 31.7 GB for both runs at
this cadence. These are decimal raw-tensor estimates; additional checkpoint
metadata and diagnostic artifacts require more space. A successful `last`
symlink adds no duplicate tensor payload. Validation reports held-out
image MSE and RGB diagnostics; it is not a policy-success evaluation.

The default execution profile is `world_model_attention=flash4_bf16_mixed`,
with Lightning `trainer.precision=bf16-mixed`. Standard autocast uses BF16 for
eligible operations, including the linear layers and FA4 attention. Stored
parameters and AdamW state remain FP32, image MSE is evaluated in FP32, and
time coordinates retain FP32 precision. Some residuals and other activations
remain FP32; this is not an all-BF16 activation graph. TF32 is disabled,
queries are unchunked, and whole-block activation checkpointing is off.
The earlier `flash4_bf16` profile retains attention-only BF16 with
`trainer.precision=32-true`; `native` and `unchunked_fp32` also retain `32-true`.
FA4 requires the separately verified runtime documented in `WORLD_MODEL.md`.
CPU integration tests select `native` and do not certify CUDA capacity or
performance.

`data.image_only=true` emits only the image stream and does not load measured
state/action arrays or pose-normalization statistics. The model uses
`phase_a_condition_on_proprio=false`, `pose_dims=null`, and `action_dims=null`.
The reusable prepared manifest may still retain state/action provenance for
later work; those fields are not stage1 inputs.

Stage2 is deferred until review of the generated videos. Its separate model
may introduce proprioception plus measured-state and control-command MSE
heads, initialized from the audited stage1 image weights. Selecting these
stage1 recipes does not activate or schedule stage2 training. The deferred
readout recipes retain their 1,000-step budget, 250-step validation and
checkpoint intervals, and `save_last: true`. The shared 1,000-step stage1 pilot still validates at
the end through `val_at_end=true`.

## Source and publication provenance

The reviewed training tree is derived from local snapshot
`252275a903b9f444a297dda303fbb6a1ca9ddf93`. The publication branch is a genuine
Git child of `cb95574e7f3dcdec8c0b0374aecbb1fefaf5f616`
([PR #261](https://github.com/GaTech-RL2/EgoVerse-graph/pull/261)), retaining its
parent relationship to
[PR #260](https://github.com/GaTech-RL2/EgoVerse-graph/pull/260). Consolidation
preserves the upstream `tests/fixtures/legacy_hpt_trunk/trunk.pth` fixture,
which was absent from the isolated local snapshot. The original local history
and the frozen experiment source remain available under their recorded IDs.

The running experiments and their checkpoints retain exact source
`3785f94392b9d900e21d9948fbc5bf94a6d975b8`, with explicit run overrides for
300,000 steps, batch 8 per rank on four ranks, and the continuous arm's
64-timestamp sampling cap. Publishing this branch does not replace that source
or relabel existing checkpoint provenance. As recorded on 2026-10-18, the
continuous arm uses allocation `411469` on `gpu19`, while the discrete resume
uses allocation `411503` on `gpu21`; they are separate four-GPU allocations.
Subsequent resumes must retain their own exact source, checkpoint, and
allocation receipts rather than infer identity from this publication branch.
