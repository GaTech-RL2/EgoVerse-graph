# LIBERO-10: uniform ARC versus true Fourier and Chebyshev

This experiment changes geometry representation, not the trainer or simulator.
It reuses the native LIBERO command bridge, OAT diffusion Transformer, dataset,
normalizer, evaluator, checkpoint reload and closed-loop rollout paths.

## Matched representation

The three `oat/libero_arc_global_{uniform,fourier,chebyshev}` recipes share:

- 32 future 7-channel OSC commands, two observation frames, execute 16 commands;
- independent translation-metre and SO(3)-geodesic progress, capped at 1.6 m and
  384 degrees, with world-frame rotation composition inherited from the bridge;
- 32 geometry slots for xyz and continuous rotation6d; no rotation-log chart cut;
- two identical 31-coefficient monotone local-speed clocks, from the human ARC
  implementation at `8eea7328`, plus two durations;
- 32 global cosine grip coefficients, exactly fitted on native control ticks;
- 384 floats, packed into 32 x 12 without unused slots, and fixed physical scales;
- no learned decoder and no chunk-level mean speed.

Uniform geometry uses linear xyz / SO(3) SLERP. Fourier is an open-curve linear
trend plus sine/cosine residuals. Chebyshev is an anchored polynomial series.
All start anchors and fitting grids are deterministic. Finite bases approximate
curves; they do not make arbitrary trajectories uniquely encoded or remove
phase, chunk-origin, timing, or rotation6d projection ambiguity.

The diffusion Transformer has 27,185,044 total / 27,184,972 trainable parameters
at this packing, including the shared observation encoder. Its state is 32x12;
hidden embedding width is 256, not a separately learned trajectory bottleneck.
Four layers/four heads, epsilon prediction, 100 training diffusion timesteps and
10 DDIM inference steps are unchanged. Sample clipping is disabled for all three
arms because global coefficients are not bounded native supports. This is a
controlled new comparison, not numerical parity with released OAT clipping.

## Gates and interpretation

`scripts/benchmarks/compare_libero_arc_bases.py` samples episode-disjoint train
and validation windows from the released, hash-verified LIBERO-10 replay. It
reports native command error, geometry-only path error and time-indexed path
error separately. Its plots are not simulator or trained-policy success rates.

`benchmark/libero_arc_global_replay.yaml` freezes all three candidates before
physics replay. The maintained `egomimic.benchmarks.libero.replay` runner uses
raw, repeated-raw, source-precision and dense-codec controls on the same saved
initial states. `--local-only` retains evidence on cluster storage and does not
instantiate the object-store uploader or require bucket credentials. No basis
is discarded or chosen as the winner from confirmation results. Failure of a
raw control is a simulator/protocol diagnostic, not evidence against a basis.

Training still requires a matching real optimizer plus scheduled validation
smoke, finite native metrics and strict checkpoint reload. Full training budget,
retention/storage and runtime checks are resolved before submission. Do not use
the old combined cluster runner's top-k=1 override for these experiments.

All three arms must retain identical dataset/split, native normalization, clock,
grip, backbone, optimizer groups, seeds, budget and evaluation conditions. This
matched uniform control is not a restart or replacement of the older STK/DUR
runs. Existing OAT experiments, outputs and unrelated jobs are left untouched.
