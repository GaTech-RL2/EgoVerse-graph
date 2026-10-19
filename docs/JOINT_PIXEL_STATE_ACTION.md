# Joint pixel, measured-state and command diffusion

The October 19 recipes train all three MoT experts together, from scratch. They
replace the deferred frozen-image readout experiment for the new runs. Historical
image-only checkpoints and run identities remain separate experiments.

| Setting | Discrete | Continuous |
| --- | --- | --- |
| Experiment | `pixel_proprio/joint_discrete` | `pixel_proprio/joint_continuous` |
| GPUs / batch per GPU | 2 / 16 | 4 / 8 |
| Global batch / accumulation | 32 / 1 | 32 / 1 |
| Recorded times | 16 stride-four anchors, covering 64 raw frames | 2–64 timestamps over the full episode |
| Measured-state width | 36: four chronological 9D states per anchor | 9 |
| Command width | 28: four chronological 7D commands per anchor | 7 |
| Image expert parameters | 88,178,688 | 88,178,688 |
| Proprio expert parameters | 13,956,388 | 13,942,537 |
| Action expert parameters | 13,952,284 | 13,941,511 |
| Parameters, total and trainable | 116,087,360 | 116,062,736 |
| Optimizer | AdamW, **3,000-update linear warmup to 5e-6**, then constant, weight decay 0.01 | same |
| Budget | 300,000 optimizer updates | same |

## Confirmed frame and vector layout

The LIBERO measurements and commands are recorded at **20 Hz**, with 0.05 s
between raw timestamps. Discrete image anchors use stride four: **5 Hz**, with
0.20 s between images. Each image anchor starts a group of four chronological
raw states and commands. For an anchor at `t`, the group uses `t`, `t+0.05`,
`t+0.10` and `t+0.15`; the next image anchor is at `t+0.20`. Sixteen groups
therefore contain 16 image timestamps and 64 state/command timestamps.
Only mark a historical state chunk observed once all four measurements are
available; a chunk is an interval rather than a single instantaneous state.

One measured state has **9 values**: end-effector XYZ in meters, a scalar-first
orientation quaternion `[qw, qx, qy, qz]`, and the two measured gripper finger
joint positions. Four such states flatten to 36 values. One recorded command
has **7 values**: three translation deltas, three rotation deltas and one
gripper command. Four commands flatten to 28 values. These are vector widths,
not the number of transformer tokens: each anchor has one state token and one
command token, alongside the image patch tokens.

Continuous training retains irregular full-episode sampling of 2–64 raw
timestamps and one 9D state/7D command per sampled timestamp. It has no fixed
frame interval. Both recipes retain the same 3,000-update LR warmup above.

Both use gripper-only LIBERO, the existing balanced 450/50 episode split, two
160×160 RGB camera streams and 16×16 RGB patches. Each camera contributes
100 patch tokens per timestamp. The pixel denoising dimension and image residual
width remain 768. Proprio and action each use residual width **256**, eight blocks
and MLP ratio four (FFN hidden width 1024, versus 3072 for images). Each vector
expert is below the human's 20M parameter budget, including all input/output,
noise-conditioning and AdaLN parameters. No image parameters or patch dimensions
were reduced.

The scheduler lives beside the optimizer in model configuration, outside the
graph nodes. Update one uses `5e-6 / 3000`; update 3,000 reaches `5e-6`, held
through update 300,000. The shared trainer steps the scheduler after every
optimizer update. Full checkpoints retain scheduler state together with AdamW,
the sampler and all-rank RNG, including when resuming during warmup. The smoke's
ten untimed compilation updates are distinct from this LR warmup.

Each layer has separate image/proprio/action QKV projections, output projections,
FFNs and AdaLN-Zero modulation, with a single joint full-attention operation.
Attention uses six heads of width 128 for all streams. A vector expert projects
its 256-wide normalized state into Q/K/V of width 768, participates in the same
attention, then projects the 768-wide result back to its private width 256.
Shared attention therefore does not require equal expert residual/FFN widths.
Camera embeddings apply to images. Images use spatial plus temporal RoPE;
vectors use the same temporal frequencies and zero spatial phase. Discrete uses
grid indices; continuous uses signed physical seconds with an arbitrary origin.
No phase, modality, embodiment or track embedding is added. Embodiment names
select the declared vector layout only.

## Corruption and losses

At every recorded timestamp, both image views and measured proprio are either
observed or queried together. A queried state draws one uniform strength `s_t`;
it is shared by all image patches/views and the corresponding proprio token.
Gaussian noise samples are independent across values. Command tokens draw their
own strengths independently of state strengths. For every modality:

```
x_s = (1 - s) * x_clean + s * epsilon
```

Sharing the coefficients does not force equal empirical SNR across modalities:
normalized proprio and RGB values can have different variances.

Observed inputs have strength zero and are clamped during sampling. Unobserved
values are not read from the ground truth during inference. Noise strength
conditions each expert through AdaLN, independently of physical-time RoPE.

The graph contains separate noising, denoising, image loss, proprio loss, action
loss and sampling nodes. All three targets are clean values (`x0`) with FP32 plain
MSE, averaged per example and then summed with weights 1/1/1. The objective accepts
noise/time metadata so later SNR weighting can replace the loss without modifying
noising or attention. Proprio uses the existing train-split normalization;
commands retain their recorded controller units. Forecast command targets exclude
past anchors. Unlabelled command positions have no loss and supply pure noise.

BF16 mixed precision keeps parameters, AdamW moments, MSE and sampler integration
in FP32. Attention uses FlashAttention 4. Valid-token execution packs all three
streams by example before attention, preserving example boundaries. Padded and
packed reference paths share identical parameters. Regional compilation is
enabled; no activation checkpointing or gradient clipping is added.

## Reproduce a run

Use `codex/joint-state-action-diffusion-20261019` and record the exact commit in
each launch request. Keep Hydra's full `.hydra/config.yaml` alongside the
model-only `smoke-config.yaml`. The postflight verifier checks that their model
and provenance agree, uses the full snapshot to restore data/trainer settings,
and preserves the minimal model construction tree for strict checkpoint binding.

Use the repository's dependencies plus the attention overlay versions recorded
in `scripts/world_model/joint_runtime_versions.json`. The GPU runtime uses
PyTorch 2.9.1+cu126 and FlashAttention 4.0.0b35; the older local CPU environment
does not validate compiled CUDA autocast. Start these commands **inside an
allocated node**, with the pinned runtime activated and node-local JIT caches.
Provide the prepared manifest and a unique output directory and W&B ID.

```bash
torchrun --standalone --nnodes=1 --nproc-per-node=2 --max-restarts=0 \
  egomimic/trainHydra.py +experiment=pixel_proprio/joint_discrete \
  hydra/launcher=basic hydra.job.chdir=false \
  data.manifest_path="$MANIFEST" paths.output_dir="$OUTPUT" \
  logger.wandb.id="$RUN_ID" +logger.wandb.resume=never
```

For continuous, select `pixel_proprio/joint_continuous` and four processes. Keep
the recipe's per-rank batch size: 16 discrete or 8 continuous. New runs use new
W&B identities, not the historical image-only IDs. The two recipes have matched
global batches and optimizer budgets. Their different GPU counts mean wall-time
comparisons need the resource difference reported. Vector chunk widths and frame
counts differ intentionally. Both use seed42; differing parameter shapes mean
the entire tensor initialization is not identical between the two models.

## Mandatory smoke and checkpoint policy

Before production, add `+joint_execution=smoke`, `logger=[csv]`, `~logger.wandb`
and `evaluator.pixel_diagnostics.log_wandb=false`, with a separate smoke output.
Keep the exact production GPU count and per-rank batch. The smoke runs 50 real
optimizer updates, then all 100 scheduled held-out anchors, including generated
images, proprio and commands. It records precision, gradients, DDP weight
agreement, actual peak memory and warmed update timing. Its terminal checkpoint
is diagnostic only. After successful zero-exit training, run on allocated compute:

```bash
python scripts/world_model/verify_joint_smoke.py "$SMOKE_OUTPUT" --world-size 2
```

Use world size four for continuous. Postflight strictly reloads the full wrapper,
checks optimizer, sampler and all-rank RNG state, then generates a 20-step real
sample with all hidden targets replaced by NaNs. A fit receipt alone is not a
passing smoke. Full submission additionally requires the exact matched source,
configuration, passing postflight and zero native/host exit. Report expected
training time from measured throughput before production launch.

Before the fit smoke, `scripts/world_model/check_joint_capacity.py` takes the
resolved configuration and an output JSON path. It checks a real batch with the
maximum frame count, actual optimization and sampling, using the exact per-rank
batch size. This prevents typical short continuous batches from hiding a
maximum-length memory failure. The check uses a disposable model instance.

Production saves and validates every **20,000 global optimizer updates**. Retain
all 15 immutable epoch-plus-step checkpoints and the `last.ckpt` symlink; there
are no extra epoch, wall-clock, signal or terminal saves. Estimated full-state
size is about 1.4 GB each, about 20.9 GB per run before overhead. Preserve every
historical checkpoint as well. Verify destination free space before submission.

GPU memory fit and compiled CUDA execution are launch gates, not claims made by
the CPU regression tests. On Phoenix, every production allocation remains at
most 24 hours, with a fresh verified full-state resume if interrupted.
