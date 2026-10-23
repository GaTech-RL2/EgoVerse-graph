# Cartesian ARC: one codec, two clocks, four stream layouts

This contract replaces the historical Cartesian mean/wide/stacked/clock/E1
families. It builds on PR #263 and preserves the local first-row-delay
implementation. It does not change the separate planar or Libero codecs.

New run presets are grouped by task under `hydra_configs/experiment/arc_real/`:
`organize_stationary`, `fold_and_stack_towels`, `wireless_earbuds`, and
`multitask`, each with `arc_vel`, `arc_dur`, and `baseline`.
See the [experiment menu](../egomimic/hydra_configs/experiment/README.md)
for models, frozen data pools, budgets, retained variants and migration.

```yaml
_target_: egomimic.rldb.zarr.arc_length_tokenizer.TokenizeBimanualArcLengthCartesian
clock_mode: velocity                 # velocity | duration
streams: full_separation              # full_separation | translation_rotation | translation | projection
resampled_vector_length: 100
min_distance_unit: 0.42               # meters, per arm
rotation_distance_unit: 1.7453292519943295  # radians, per arm (100 degrees)
dt: 0.03333333333333333
hold_window_frames: 400
gripper_stream_mode: events           # events | distance; separate streams only
gripper_distance_unit: null           # required positive for distance mode
```

## Shape and clock order

Geometry always occupies the first 14 columns:
`[left xyz, left YPR, left gripper, right xyz, right YPR, right gripper]`.

| streams | Shape | Remaining columns |
| --- | --- | --- |
| full_separation | M × 20 | T-left, T-right, R-left, R-right, G-left, G-right |
| translation_rotation | M × 18 | T-left, T-right, R-left, R-right |
| translation | M × 16 | T-left, T-right (rotation and gripper ride the translation supports) |
| projection | M × 16 | as translation, with progress measured at `p + R @ projection_offset_m` (2 cm per local axis) |

Each arm and stream spends its own spatial budget. Supports from different
streams in the same row are not necessarily simultaneous.

In translation_rotation, grippers are sampled at their arm's translation
supports and interpolated with its translation clock. There is no independent
gripper termination racer or gripper distance budget. A fully stationary
translation stream retains its initial gripper target; use full_separation to
represent independent gripper movement.

## Timing rows

Row zero of every timing column represents that stream's initial delay.
The meaning of later rows depends on `clock_mode`: incoming rates for velocity,
absolute waypoint seconds for duration. There is no repeated final padding clock.

- velocity: row zero is seconds; later rows are the distance between emitted
  supports divided by the source interval's seconds. Units are m/s, rad/s,
  and gripper units/s.
- duration: every row is the absolute seconds from the chunk anchor to that
  waypoint (row zero is the start delay). Decoding projects the rows once onto
  the nearest non-decreasing clock by isotonic regression on the slack above
  the fastest clock allowed by the 5 m/s translation max speed (no min speed:
  rotation and gripper ride slow arms), so a backwards pair is averaged and no local
  error accumulates into later rows. Provenance `timing_layout` is
  `first_delay_absolute_v1`; velocity keeps `first_delay_incoming_v1`.

Translation distances are Euclidean, rotation distances are SO(3) geodesic,
and gripper distances are absolute value differences. Velocity-mode rates are
not smoothed or physically speed-clamped. Duration-mode clock projection does
enforce its explicit 5 m/s translation bound. Neither mode adds an arbitrary
decoder-duration floor.

An initial wait retains constant geometry before the first movement. Rotation
and distance-gripper target times use first arrival, so a final stationary
plateau does not stretch the last moving interval.

Full-window holds retain constant waypoints. Velocity interval rates are zero.
Duration absolute clocks span the configured hold_window_frames × dt. Constant
streams do not terminate a chunk. Internal dwells are not explicitly allocated
duplicate supports; zero speed alone cannot encode a zero-distance dwell time.

Separate event grippers are held commands, not E1's fitted aperture ramps.
Command-change supports and endpoint padding retain the prior local semantics.

## Evaluation and rollout

Both use the same codec and `control_steps()`:

1. Keep an exact integer waypoint prefix, e.g. 30 of M100.
2. Decode every stream using its selected clock representation: start delay
   plus distance/rate intervals, or absolute waypoint seconds.
3. Replan when the first moving stream reaches its prefix endpoint.
4. Ignore fully constant streams in this race.

Validation is an open-loop segment simulation against recorded observations.
Ground truth and videos cover the executed control steps. It does not apply a
second fixed-tick E1 regime. Distance-prefix token truncation is unsupported.

The robot adapter pads its fixed output tensor after the executed prefix and
reports the actual replan boundary via `execution_steps()`. No hardware is
started by constructing a decoder.

Malformed/nonfinite tokens fail. A moving stream with unusable timing cannot
win the termination race; an explicit caller horizon can bound a stalled clock.

## Migration and checkpoint safety

The old keyword options velocity_mode, velocity_layout and velocity_norm are
not accepted by the Cartesian codec. TokenizeBimanualArcLengthE1 and its
projection subclass are removed, along with their selectable ARC recipes and
codec-specific diagnostics. Baseline time-chunk retiming and unrelated model,
dataset, planar and Libero implementations remain separate.

Old serialized configs/checkpoints must use their original recorded source.
Do not reinterpret them by token width, rename their normalizers, or resume
training after changing this representation. Generate fresh normalization
statistics and a fresh run. The sparse row-zero delay may require existing
quantile widening (`norm_stats.widen_degenerate_quantiles: true`).

The first-delay six-stream refactor was compared numerically against its
preceding local implementation across 32 synthetic clock/gripper/hold cases,
including token, detokenizer, and termination parity. Regression tests cover
clock/stream combinations and validation/rollout agreement.

Historical verification for the initial two-layout migration: 536 focused CPU
tests passed, including
normalization-cache rejection, baseline graph forward/backward, source adapters,
and planar ARC regressions. All 379 Hydra YAML files shipped at that revision
composed; these are historical counts, not the current configuration inventory.
The 32 six-stream comparisons used predecessor
`f364923a3e675106882823da17f840a9505ab972`, with token tolerance 1e-10,
decoded-trajectory tolerance 1e-9, and identical execution-step counts.
These checks do not establish GPU throughput, full pretrained-checkpoint
reload compatibility, or hardware safety. No cluster or robot process was run.
