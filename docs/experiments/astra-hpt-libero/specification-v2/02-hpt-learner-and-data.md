# 02 — EgoVerse HPT from scratch and its data interface

## Meaning of from scratch

All learner weights start randomly: image encoders, language embeddings/transformer, modality projections, HPT trunk, action tokens, and flow-matching head. All learnable parameters participate in optimization. No HPT, ImageNet, Qwen, T5, robotics, teacher-distilled, or other pretrained weight file is loaded. No external demonstration dataset is used. Resuming this experiment's own checkpoints is permitted and must preserve ancestry.

This requirement applies to the learner. Astra remains a pretrained generator; the simulator and teacher controller contain engineered knowledge. Report those sources of prior knowledge explicitly.

The inspected EgoVerse checkout contains `ResNet(weights="DEFAULT")` and language encoders calling `from_pretrained`. Its model-level `pretrained: false` only avoids the HPT checkpoint path. The required adapters below therefore cannot be replaced by toggling that single flag.

## Fixed architecture to implement

| Component | Pilot specification |
|---|---|
| Backend | PyTorch, EgoVerse HPT, one domain named `libero_push` |
| Observation horizon | One current observation; no frame stack |
| Vision | Two independent ResNet-18 encoders, `weights=None`, `freeze_backbone=False`; 256-dimensional projected spatial features |
| Visual stems | HPT MLP/cross-attention stems, 256 dimensions, 16 latents per view, 8 attention heads, head dimension 64 |
| Proprioception | Nine floats: end-effector position 3, orientation quaternion 4, gripper joint positions 2; one 256-dimensional HPT MLP/cross-attention stem with 16 latents |
| Language | New byte-level stem: 256 byte IDs plus PAD/BOS/EOS, vocabulary 259, at most 256 tokens including boundaries |
| Text network | Random embeddings and learned positions, two bidirectional transformer layers, width 256, 8 heads, feed-forward width 1024, dropout 0.1; HPT-compatible cross-attention to 16 latents |
| Trunk | Existing HPT transformer structure, width 256, 16 blocks, 8 heads, action-token postprocessing, one domain embedding, drop-path 0.1 |
| Head | EgoVerse `FMPolicy` with `CrossTransformer`: 6 blocks, condition width 256, hidden width 128, 4 heads, dropout 0.1, MLP settings matching the inspected example (`mlp_layers=4`, `mlp_ratio=4`) |
| Action output | Horizon 10, dimension 7 everywhere; 50 flow integration steps; execute only action 0 and replan |
| Flow training | Upstream beta time distribution, parameters (1.5, 1.0), mapped to [0.001, 1]; Gaussian action noise |
| Initialization | Seed 17 with named derived streams; no loaded weights; preserve parameter initialization code and config hash |

These are concrete proposed settings based on EgoVerse's local interfaces, not a verified named upstream LIBERO recipe. The existing example uses different embodiment/action dimensions and inconsistent-looking trunk/head horizons for this intended application; override every relevant horizon to 10 and every action dimension to 7. Set the upstream pose-conversion option `6dof: false`; the action is a normalized OSC delta command, not an absolute wrist pose. Do not enable cotraining, shared action heads, representation freezing, or optimal-transport losses.

The random text stem is a new integration module implementing HPT's modality-stem interface. Encode UTF-8 bytes directly; no pretrained tokenizer or corpus-derived vocabulary is needed. Reject overlength instructions rather than truncating a referent or destination. PAD is masked in self-attention and latent cross-attention. The frozen task grammar supplies short English instructions; success would demonstrate bounded grounding, not open-domain language understanding.

## Pretraining audit and initialization gate

In W01, construct the learner with network access and model-cache reads disabled. Instrument weight-loading entry points so unexpected pretrained loads fail. Save resolved constructor arguments, all parameter/buffer names and shapes, initialization seed streams, optimizer membership, and the initial checkpoint hash.

Confirm that changing the initialization seed changes weights, restoring a seed reproduces the initialization, and no trainable component is frozen or absent from the optimizer. One real mixed-stage batch must produce finite loss and gradients reaching image encoders, text stem, trunk, and action head. A single tensor can legitimately have zero gradient; inspect component-level gradient participation across the fixture batch rather than demanding every scalar move.

Do not let validation updates alter the production initial model. W01 fixtures and update results are discarded; recreate the audited seed-17 initialization before common warm-start training. Synthetic fixture updates and their compute are logged as engineering checks, not included as study training steps.

## Action and loss conventions

Canonical actions are seven bounded normalized `OSC_POSE` inputs: translation delta 3, axis-angle rotation delta 3, and gripper command 1. Train directly in those units with identity action normalization. The adapter applies the simulator's recorded controller transform exactly once. Clip sampled actions to the valid command range at execution and record clipping. Confirm the frame, units, rotation convention, and gripper polarity using physical commissioning fixtures.

Use upstream flow interpolation `x_t = t * noise + (1 - t) * action`, target velocity `noise - action`, and masked mean squared velocity error. The mask counts valid scalar action elements. A horizon window at an episode boundary pads within that episode and excludes all padded timesteps from the loss; it cannot borrow another episode's actions.

The inspected `DenoisingPolicy.loss_fn` is unmasked MSE. Add a local `MaskedFMPolicy` adapter that threads the action-validity mask through loss computation and optionally accepts explicit noise/time draws for diagnostic replay. Reject mismatched shapes rather than slicing dimensions to fit. Keep the upstream loss unchanged outside this experiment adapter.

Use AdamW at `1e-4`, weight decay 0.01, betas (0.9, 0.999), epsilon `1e-8`, gradient clipping 1.0, batch size 4, and no accumulation. Warm up over the first 100 common updates, then hold the learning rate constant. Maintain FP32 parameters, optimizer moments, and reductions with BF16 autocast. No EMA, color jitter, or crop augmentation. Training uses dropout; evaluation disables it. These choices and all budgets must be identical across the arms.

## Student observation and storage contract

Store raw external/wrist images as uint8 arrays, nine raw proprioception values, timestamps, natural-language instruction, and executed actions in HDF5. An episode JSON record references the task/scene/teacher, initial-state hash, source round, split, acceptance result, and artifact checksums. Privileged state and semantic labels live in a separate audit namespace.

The HPT adapter yields only:

| Tensor or field | Shape / representation |
|---|---|
| External RGB | `[B, 1, 1, 3, 224, 224]`, scaled with `x / 127.5 - 1` |
| Wrist RGB | Same, independently encoded |
| Proprioception | `[B, 1, 9]`, fixed commissioning mean/std, minimum divisor `1e-6` |
| Text | Byte IDs `[B, L]` and attention mask; raw instruction retained for audit |
| Target actions | `[B, 10, 7]` |
| Action validity | `[B, 10]` boolean |
| Domain | Literal `libero_push` |

Object IDs/poses, desired target coordinates, reward terms, teacher state, task stage, and scene names cannot serve as student features. An allowlist should reject accidental privileged keys rather than trusting downstream code to ignore them.

Use a direct HDF5-to-PyTorch adapter, not a required conversion through LeRobot/RLDS. This limits integration work without changing the canonical record. Pair observation t with the action executed after that observation and preserve timestamps. Verify one real episode round trip and one real learner batch before collection at scale.

## Replay, branching, and resume

During common warm-start, sample S1 episodes uniformly, then sample windows within the selected episode. During each branch round, draw exactly two batch windows from that round's newly accepted pool and two from historical data. Historical data include the common seed pool and earlier rounds, but exclude the current round to keep the split unambiguous. Sampling is episode-balanced within each pool. Commissioning episodes do not enter either pool.

After 1,000 common updates, fork the complete learner state into U and A. This includes model parameters and buffers, AdamW moments and step counters, learning-rate scheduler, RNGs, data sampler state, normalization artifact, data manifests, and checkpoint ancestry. Log subsequent arm-specific random streams; common initialization does not imply that divergent datasets consume identical random draws.

Save final checkpoints at branch update 2,000 and retain branch update 1,000 for the diagnostic. Publish checkpoint manifests atomically, and resume model, optimizer, replay, curriculum decisions, and ledger together. Replaying a stored Astra decision must not require another generation call.

## Upstream integration boundary

Reuse HPT's architecture and flow head from the inspected source. Add the byte language stem, masked head, single-arm LIBERO observation/action adapter, initialization audit, and direct episode loader in this experiment's repository. Pin the upstream revision and relevant source hashes; record any needed patch separately. Do not describe planned adapters as existing EgoVerse features or reuse the bimanual example's action conventions silently.
