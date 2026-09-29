# ARC with the released OAT diffusion-policy backbone

The original ARC runs use EgoVerse's conditional U-Net and 100 DDIM sampling
steps. The `oat_dp` option trains ARC from scratch with the diffusion Transformer
and sampler specified in OAT's pinned
[`train_diffpolicy.yaml`](https://github.com/Chaoqi-LIU/oat/blob/1da92695ef12c23b7000a0b1a76cab0aef4750e6/oat/config/train_diffpolicy.yaml).
It preserves the previously selected STK/DUR representations and replay evidence.

The native Transformer source and positional embedding are tracked in
`egomimic/models/oat/UPSTREAM.json`. The only graph adapter reshapes flattened
observation features back into the two observation tokens expected upstream.
No upstream Python package is required at runtime.

| Setting | Released DP and ARC DP |
| --- | --- |
| Backbone | 4 Transformer decoder layers, width 256, 4 attention heads |
| Feedforward width / dropout | 1024 / 0.1 |
| Attention | Released causal action and observation-memory masks |
| Observation encoder | Separate ResNet-18 encoders for two cameras, spatial softmax, robot state and task ID |
| Observation history | 2 frames, each with 138 features |
| Diffusion | Epsilon MSE, 100 training timesteps, cosine beta schedule |
| Sampling | 10 DDIM steps; released clipping, offset and alpha settings |
| Optimizer | AdamW, action LR 5e-5, observation LR 1e-5, betas (0.9, 0.95), zero weight decay |
| Schedule | Constant LR, 5001 epochs, effective batch 1024, released EMA schedule |
| Dense action prediction / execution | 32 / 16 steps |

The representation requires two changes in model dimensions: 7 raw-action
channels become 12 ARC channels, and 32 raw-action tokens become M ARC supports
(24, 32 or 36). Hidden width, depth, masks, conditioning and sampling settings
are preserved. As with the previous native port, this uses EgoVerse's training
runtime and pinned dependencies; it is not a claim of identical training
trajectories across framework versions or device layouts.

| Model | Action-network parameters | Total with observation encoder |
| --- | ---: | ---: |
| Released DP, 7 channels, horizon 32 | 4,788,231 | 27,182,479 |
| ARC DP, M24 | 4,788,748 | 27,182,996 |
| ARC DP, M32 | 4,790,796 | 27,185,044 |
| ARC DP, M36 | 4,791,820 | 27,186,068 |

The shared observation encoder contains 22,394,248 parameters, including 72
fixed normalization entries. ARC encoding/decoding adds no learned parameters.
The diffusion Transformer's internal condition encoder receives the action LR;
only the camera/state encoder receives the observation LR.

## Launch and provenance

Use `--arc-backbone oat_dp` with a standalone `--arc-profile` and one ARC mode
in `scripts/benchmarks/launch_libero_osmo.py`. The new experiment is
`oat/libero_arc_oat_dp_policy`; the existing default remains the U-Net.

A fresh policy may reuse `--arc-replay-runs-file` without `--resume-from-run`.
The worker still checks the complete replay confirmation, chosen R/D/M,
dataset identity and unchanged codec source hashes. This reuses demonstration
reconstruction evidence, not trained weights. Resuming across backbone types
is rejected. Each new run records its backbone and immutable source commit.

Full ARC DP runs automatically evaluate all 2500 episodes after training using
the same five independent repetition workers as the completed OAT evaluation.
The strict merge checks all episode identities, seeds and checkpoint metadata.
Starting-state hashes must still be checked against OAT before claiming a fully
paired comparison; the previous U-Net evaluation's drawer mismatch is not
retroactively resolved by this change.

## Verification

`tests/test_oat_diffusion.py` compares the full released network and graph
adapter against the pinned upstream source: outputs, epsilon-loss gradients,
and all 10 DDIM steps must match exactly for M24/M32/M36. It also checks the
released YAML values, parameter counts, optimizer assignment, STK/DUR shared
training, EMA checkpoint reload, resume and inference without action targets.
`tests/test_libero_arc_sweep.py` covers reuse of audited replay for a fresh
backbone. Set `OAT_REFERENCE_ROOT` to the pinned checkout for parity tests.
