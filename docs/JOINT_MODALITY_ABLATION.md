# Discrete image quality: proprioception × action factorial experiment

The first authorized experiment is four fresh discrete runs, each for100000
optimizer updates. It tests whether adding the proprioception stream, the action
stream, or their interaction changes image forecast quality under the current
joint-training recipe. It does not use the historical image-only run as a causal
control: that run had a different learning rate and its preview had more context
and a shorter future horizon.

| Experiment suffix | Active streams | Total/trainable parameters |
| --- | --- | ---: |
| image_only | image | 88178688 |
| image_proprio | image, proprio | 102135076 |
| image_action | image, action | 102130972 |
| image_proprio_action | image, proprio, action | 116087360 |

Select `+experiment=pixel_proprio/ablation/discrete_SUFFIX`. Each declared stream
has its expert, attention tokens and unit-weight plain x0-MSE loss. Removed
streams have no parameters, tokens, losses or sampler updates. All retained
parameters are trainable from scratch. The full bank is constructed before
pruning, preserving identical common image weights and constructor RNG at seed42.
The shared full data/noising contract is retained internally to pair image frame
selection, observed masks and random draws; inactive values never enter attention.
The sampler likewise consumes the same initial RGB noise across all four arms.
The full three-stream default remains compatible with the parent checkpoint keys.

All arms use the same immutable500-episode LIBERO gripper corpus and450/50 split,
two160×160 cameras,16×16 RGB patches,16stride4 anchors and four native20Hz vector
slots per anchor. Effective denoising dimensions are image768, proprio36 (4×9)
and action28 (4×7), where present. Native action width is7. Image private width
is768; vector private residual/FFN/AdaLN width256; common QKV width768,6heads×128,
8blocks. No tokenizer, pretrained image weights, freeze or gradient clipping.
Image/proprio observation masks and noise strengths remain tied; action strengths
remain independent. The complete noiser is used in every arm to pair RNG draws.

AdamW peak5e-6, weight decay0.01,3000optimizer-update linear warmup then constant:
update1=5e-6/3000 and update3000=5e-6. Global batch32,16/rank on two dedicatedH100s,
accumulation1,FA4,BF16 mixed with FP32 stored weights, AdamW moments and MSE.
Same clean source, runtime bindings and complete read-only data cache in all arms.
The dataset/normalization/provenance checks remain enabled.

Checkpoint and scheduled validation cadence is20000global optimizer updates.
Retain all five immutable epoch+global-step files per arm, `save_top_k=-1`, with
`last.ckpt` as a symlink. No extra epoch, wall-clock, signal or terminal production
save. Separate directories, stable new W&B IDs and fresh `resume=never`; never
initialize from a smoke or existing campaign checkpoint. Estimated checkpoint
payloads are12bytes/parameter; total across20files is24511925760bytes plus small
metadata/log overhead. Check destination free space again at allocated admission.

## Evaluation and interpretation

The primary endpoint is held-out forecast image normalized MSE at100k, using the
same deterministic100scheduled validation anchors (two per held-out episode),
query masks,20-step sampler and evaluation seeds in every arm. Completion image
MSE is a secondary endpoint. Retain the20k/40k/60k/80k results to distinguish
optimization rate from final-budget quality; do not select a favorable checkpoint
independently for each model. Compare the same camera images side by side.

For qualitative fixed-window forecasts, use observed indices[0,4,8,12] and query
indices[16,20,...,60], hence0.6seconds observed span and2.4seconds future. Use
identical held-out episodes/windows and initial inference seeds for all arms.
No hidden query label may enter sampling: replace it with NaN and verify finite
predictions and exact observed-value clamping. Show ground truth and hold-last
image baseline beside predictions. Report per-camera and per-horizon image errors
from retained predictions; present action/proprio only when those outputs exist.
All final sampling comparisons use one inference backend/precision across arms.
These are offline forecasts, not simulator rollouts or policy scores.

For lower-is-better image error Y, compute paired effects:

- proprio without action: Y(image+proprio)−Y(image)
- proprio with action: Y(image+proprio+action)−Y(image+action)
- action without proprio: Y(image+action)−Y(image)
- action with proprio: Y(image+proprio+action)−Y(image+proprio)
- interaction: Y(all)−Y(image+proprio)−Y(image+action)+Y(image)

Positive differences indicate worse image MSE. Use per-episode paired differences
and an episode-level paired bootstrap interval when individual predictions have
been exported. A single seed is a screening experiment, not evidence of
seed-robust causality. A detected effect requires replicate seeds before a strong
claim. Removing a whole stream changes tokens, expert parameters, supervision and
any corresponding observed conditioning together. Loss-only and conditioning-only
follow-ups are needed to identify which mechanism caused an effect. Equal updates
and image exposure are controlled; FLOPs and total parameter count differ as a
necessary consequence, so this is not a compute-matched comparison.

## Admission and operation

`egomimic/pl_utils/joint_smoke.py` and `scripts/world_model/verify_joint_smoke.py`
provide the real-data gate for every active-stream subset. Each exact two-rank
smoke must optimize50actual updates, validate all100scheduled anchors, prove
all-rank FP32 weights/gradients/MSE and BF16 output, every actual warmup LR, strict
full wrapper/AdamW/scheduler/data-context/sampler/all-rank RNG restoration and
finite20-step hidden-NaN samples. Check the real maximum16-anchor shape separately.
A FIT receipt, CPU test or successful allocation alone does not constitute PASS.
Require host exit0 and native COMPLETED0:0.

The maintained fresh training service is `scripts/train/joint/run.py`, promoted
and generalized from the accepted Lambda full/smoke drivers. Requests bind clean
source/archive/manifest hashes, exact scientific config, runtime/data guards,
unique paths and evidence. Immutable scheduler/service bindings stay outside the
source checkout. The original production operations remain immutable.

`scripts/world_model/audit_modality_ablation.py` constructs the four real-size
models on CPU and verifies common image initialization and constructor RNG.
`scripts/train/joint/verify_factorial.py` compares initial image weights and the
first three actual RGB training batches/noise/masks per rank from all four GPU
smokes. All four must pass before any full run is admitted. Full requests bind
this paired receipt and their own exact smoke, then compare resolved configs,
recheck fresh W&B identity, context, resource ownership and checkpoint storage.
The full verifier accepts request-bound100k/300k budgets and checks every retained
milestone and finite scheduled validation event without saving another checkpoint.

Never rerun a consumed submitter, overwrite an existing output or blindly retry an
unknown sbatch result. Inspect queue and immutable claim receipts first. No current
production job is cancelled or changed for this experiment. Preserve all historical
checkpoints and audits. Phoenix GPU namespace mapping is unresolved; this protocol
does not grant a Phoenix launch or reuse any previous race grants.
