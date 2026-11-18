# Native B-spline Policy for LIBERO

This adds B-spline Policy (BSP) to the same native graph, replay loader, training
loop, checkpoint format and rollout harness used by the ARC/OAT/DP comparison.
It is a learned spline-parameter policy, not post-hoc smoothing of a raw DP output.

Source: [official implementation](https://github.com/B-spline-policy/bspline-policy/tree/61ed5f42fced971d50a89b46417493790876ccd1),
[paper](https://arxiv.org/pdf/2607.09648), especially Sections 3–4 and Appendix A.1.
Adapted source is MIT-licensed; its notice ships in `egomimic/models/bspline/LICENSE`.

## Representation and training

- Fit one cubic spline jointly to all seven native delta OSC channels of each
  demonstration, including gripper. FITPACK inserts knots until the maximum
  absolute component error is below 0.002. This follows the released code's
  componentwise criterion; the paper writes a vector-norm criterion.
- At each observation frame, select the nearest future knot and its neighboring
  support, exactly as upstream. Shift knot times by the observation frame index.
  The default target is **16 × 8**: `[knot_time, cx, cy, cz, crx, cry, crz, cgrip]`.
  Six knot rows provide cubic boundary support. The final four coefficient rows
  are padding, as in upstream. Rows are not sampled actions or 16 timesteps.
- Observation history ends at the current frame. The existing raw action target
  still starts at that frame. No observations from the future are introduced.
- Full-episode fits are cached atomically under `<replay>.bspline/`. Keys include
  exact native action bytes and codec settings. File locks serialize competing
  workers/ranks. Training batches only gather the local parameters; they do not
  rerun spline fitting.
- Parameter channels use limits measured over the training episodes, shared over
  token positions as in upstream. Raw actions and observations retain the
  existing baseline normalization. Both parameter statistics and codec identity
  are saved in the normalizer/checkpoint.

The two recipes preserve their corresponding DP observation encoders, optimizers,
EMA, diffusion schedules and network settings. The predicted horizon becomes 16
spline-parameter rows and the output width becomes 8. OAT-DP uses its existing
four-layer, width-256 transformer and 10 DDIM steps; U-Net retains 100 denoising
steps. Full LIBERO runs retain 5001 epochs, global batch 1024 and seed 42.

## Execution contract and deliberate adaptations

LIBERO remains at its existing simulator rate and **16-command replanning**.
Decode on the original frame-time grid; do not stretch the spline's variable
time span into an equally spaced fixed-horizon action sequence. Predicted knots
are corrected by a forward pass that repairs decreases without sorting control
points. Execution speed is 1 and segment alignment is disabled, consistent with
the paper's treatment of delta actions.

If a predicted segment is shorter than the replanning interval, hold its final
command. Upstream's nearest-future-knot convention can put time zero before the
valid span; this implementation uses the first polynomial there, matching its
deployment wrapper. The paper instead pipelines replanning around segment ends.
**These are matched-protocol LIBERO baselines, not a reproduction of the paper's
100 Hz controller or accelerated execution results.**

The generic codec exposes temporal rescaling and an absolute-action phase-search
helper. Neither is enabled in the LIBERO graph. Gripper is a spline dimension,
not an ARC event stream. Episodes shorter than four frames are endpoint-padded
for cubic fitting. A failed fitting tolerance raises instead of silently accepting
an inaccurate fit. Repeated terminal knots in upstream can make SciPy return zero
at the exact endpoint; decoding evaluates the left limit to preserve the terminal
action and gripper state.

## Training and evaluation

After activating `emimic`, install the selected extras:

```bash
uv pip install -e '.[oat,libero,bspline]'
python -m egomimic.trainHydra \
  +experiment=oat/libero_bspline_oat_dp_policy hydra/launcher=basic \
  benchmark.suite=libero_spatial benchmark.dataset=/path/to/replay.zarr
```

Use `libero_bspline_unet_policy` for U-Net. The existing checkpoint rollout command
automatically identifies `bspline_oat` or `bspline_unet` and records the representation.
It requires no demonstration dataset or spline cache at evaluation time.

To render an L40S workflow without submitting it:

```bash
python scripts/benchmarks/launch_libero_osmo.py \
  --commit <full-runtime-commit> --run-id bsp-spatial-oat \
  --suite libero_spatial --bspline --dp-backbone oat_dp \
  --mode full --gpus 4 --gpu-type L40S --output /tmp/bsp-spatial-oat.yaml
```

Full runs execute a fresh two-update GPU/simulator preflight, train, release
training GPUs, then evaluate 2500 episodes on one GPU. The final checkpoint,
optimizer budget and EMA count are verified before evaluation. No new training
or simulator results are implied by adding these recipes.

## Validation and limitations

`tests/unit/test_bspline_policy.py` covers upstream target/decoder parity, all
seven channels, short episodes, endpoint gripper preservation, actual knot timing,
temporal scaling, decreasing/collapsed predicted knots, split/statistics/cache
behavior, protocol guards, and real CPU train/resume/EMA reload with both DP heads.
Reload tests remove the replay and spline cache before inference. The golden
fixture was generated with the pinned upstream functions and SciPy 1.16.3; it
does not require upstream code, downloads or a simulator during the test.

Keep the complete fixture, tests and this document on the companion validation
branch, referenced by runtime `.github/validation-ref`. Run `pytest tests/unit`
plus the affected existing LIBERO tests before committing. Full GPU training,
simulator success rates and parameter/timing comparisons remain experiment work.
The runtime branch's pre-existing global `uv lock` resolution fails because its
PI05/OpenPI extra requires `av>=14.2.0` while base dependencies pin `av==12.0.0`;
use the selected-extra install above, as the existing LIBERO launcher does.

Validation on 2026-11-18: all 14 BSP unit tests passed, including the two real
CPU bf16 train/resume/reload cases. The combined BSP + DP/LIBERO regression run
passed 151 cases before encountering two existing performance-test failures
(`test_target_cache_is_exact_bounded_and_invalidated_by_codec_changes[stk/dur]`).
Both failures were reproduced on untouched runtime base
`6138cd6ff4e9f3e80d7817a541406756f607df95`: the legacy fixtures omit the required
`libero_osc_main_arrival_v2_unbounded` clock contract. No ARC implementation was
modified. Separate OAT diffusion checks passed 3 cases and skipped 5 CUDA-only
cases on this CPU host. A wheel build included the BSP code, license and recipes.

The additional existing-CI pass reached 248 passing cases and three failures.
One failure was caused by changing the legacy dependency-install entrypoint and
was fixed by installing BSP's extra dependencies conditionally; all 12 recovery
checks then passed. The other two observed failures (`test_graph_cache_and_representation_record_the_stream_layout`
and `test_native_graph_normalizes_each_variant_and_decodes_bfloat16[stk]`) were
also reproduced on untouched base 6138cd6 and use the same obsolete ARC fixtures.
No historical fixture was skipped or weakened.
