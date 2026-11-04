# Cartesian ARC: explicit-current-pose arrival clocks

This is the current Cartesian contract. It is a new representation relative to
the former first-delay token: row zero now describes the current measured pose
to the first future target. No row has a special start-delay meaning.
Separate planar and Libero codecs are unchanged.

Select a complete experiment from the
[experiment menu](../egomimic/hydra_configs/experiment/README.md). The example
below describes the codec API, not a universal experiment's D/R/M settings:

```yaml
_target_: egomimic.rldb.zarr.arc_length_tokenizer.TokenizeBimanualArcLengthCartesian
initial_action_key: arc.current_action
clock_mode: velocity                 # velocity | duration
streams: full_separation              # full_separation | translation_rotation | translation | rotation
resampled_vector_length: 100
min_distance_unit: 0.42               # meters, per arm
rotation_distance_unit: 1.7453292519943295  # radians, per arm (100 degrees)
dt: 0.03333333333333333
gripper_stream_mode: events           # events | distance; separate streams only
gripper_distance_unit: null           # required positive for distance mode
```

## Current state and future source

The transform requires two distinct inputs:

- `arc.current_action`: the finite 14D measured current pose at time zero,
  transformed into exactly the same coordinate frame, YPR convention and
  gripper convention as the action geometry.
- `actions_cartesian`: future rows beginning at source index `idx+1`.
  Raw row i arrives at `(i+1) * dt`; the current observation row is not
  duplicated into this window.

The current pose is supplied separately to tokenization and every decoding
entrypoint. It must never be guessed from token row zero or replaced by a zero
pose except in an explicitly constructed synthetic example whose current pose
really is zero. Timing uses the configured control-grid dt, not irregular
measured timestamps.

## Geometry, supports and clock order

Geometry occupies the first 14 columns:
`[left xyz, left YPR, left gripper, right xyz, right YPR, right gripper]`.

| streams | Shape | Timing columns |
| --- | --- | --- |
| full_separation | M × 20 | T-left, T-right, R-left, R-right, G-left, G-right |
| translation_rotation | M × 18 | T-left, T-right, R-left, R-right |
| translation | M × 16 | T-left, T-right |
| rotation | M × 16 | R-left, R-right |

Translation-only samples XYZ, YPR and gripper on the translation supports.
Rotation-only samples all three on the rotation supports. In
translation_rotation, the gripper follows translation. Full separation gives
each component its own supports and clock. A token row therefore need not
represent simultaneous source times across separate streams.

D and R bound cumulative travel **within the supplied future trajectory**.
The measured-current-to-first-future-target segment is included in incoming
rates and decoding, but is not retroactively included in those support budgets.
Anchor-inclusive emitted travel can exceed the source budget by that incoming
segment. Translation uses Euclidean distances; rotation uses SO(3) geodesic
distances; gripper distance uses absolute command differences.

Exactly zero-progress geometry uses supports distributed over the actual
source indices. This retains the real available time span without inventing
movement or extending the trajectory to a configured hold duration.

## The same timing meaning in every row

Let q[-1] be the measured current pose, t[-1]=0, and q[i] the emitted future
waypoint with source arrival time t[i].

### Velocity

Every row, including row zero, stores the incoming segment speed:

```text
v[i] = distance(q[i-1], q[i]) / (t[i] - t[i-1])
```

Thus row zero is measured-current-to-first-target speed, not seconds and not
a start delay. The numerator is computed between emitted supports, matching
the segment reconstructed by the decoder. Units are m/s, rad/s or gripper
units/s. Tiny zero intervals use numerical guards. Rates are not smoothed or
physically speed-clamped; decode uses the rate magnitude.

Normalization provenance identifies this timing layout as `incoming_rate_v2`.

### Duration

Every row stores its raw absolute arrival time from the current observation:

```text
clock[i] = t[i]
```

There is no logarithm, exponentiation, or special row-zero delay. For future
targets at 0.1, 0.2 and 0.3 seconds, the token stores `[0.1, 0.2, 0.3]`.
The decoder derives incoming intervals with `diff([0, corrected_times])`.

Predicted times are projected to a non-decreasing absolute clock by isotonic
regression. Translation has a 5 m/s maximum-speed bound that includes the
current-pose-to-row-zero segment:

```text
fastest[i] = sum(incoming_distance[0:i+1]) / 5
corrected_times = fastest + max(isotonic(predicted_times - fastest), 0)
```

Rotation and gripper clocks receive monotone projection without a translation
speed bound. No arbitrary minimum chunk duration is added.
Normalization provenance uses `absolute_arrival_seconds_v2`.

## Pauses and grippers

There is no motion-start-delay extractor, synthetic 400-frame hold clock,
or dedicated internal-pause support allocation.

Duration can preserve a stationary span when it is represented in the emitted
support times. Fully static duration tokens retain the actual source span;
they are not extended to 400 frames. Velocity is zero on a zero-distance
interval and cannot encode how long that dwell lasted. This includes terminal
gripper dwells and all-static chunks.

For separate event grippers, supports remain initial future command, command
changes, source endpoint and repeated endpoint padding. If there are more
events than M supports, they are downsampled. Reconstruction is step/hold
command behavior, not fitted aperture ramps. Distance grippers use their
independent cumulative-travel budget.

Single-stream velocity has the same limitation: if the selected component
does not move but another component does, that selected component's zero rate
cannot encode the other component's timing. Use a separated clock layout or
duration when that timing must be represented.

## Decode and execution boundary

Both evaluation and rollout provide the measured current pose explicitly:

```python
codec.control_steps(token_prefix, initial_action=current_pose, max_steps=horizon)
codec.detokenize(token_prefix, horizon, initial_action=current_pose)
```

The codec starts at the measured pose at t=0 and emits future controls at
`dt, 2*dt, ..., horizon*dt`. XYZ uses linear interpolation, rotations use
SO(3) interpolation, and event grippers use held commands.

For first-stream execution:

1. Keep an exact integer prefix of future waypoint rows.
2. Decode all incoming intervals, including current-to-row-zero.
3. Replan when the first moving stream finishes that prefix.
4. Exclude constant streams from the race while another stream moves.

A fully static velocity token contains no duration. `control_steps` therefore
requires the caller's positive `max_steps` and returns that explicit bound.
For fully static duration tokens it uses the earliest final absolute stream
arrival. A moving PW stream with unusable rates cannot win the termination
race; the explicit caller horizon can bound it. Nonfinite tokens or anchors
fail rather than being repaired silently.

`clock_token_seconds` requires the same explicit anchor when converting
duration columns. Sixteen-column helpers also require explicit `streams`;
shape alone cannot distinguish translation-only from rotation-only.

## Migration and checkpoint safety

The former mean/wide/stacked/log-clock/E1 APIs are not supported.
Projection is outside the main contract. `hold_window_frames`,
`projection_offset_m`, `velocity_mode`, `velocity_layout` and
`velocity_norm` are not codec constructor options.

First-delay checkpoints and normalization statistics are incompatible with
these arrival clocks despite matching tensor shapes. Preserve their recorded
source for historical reloads; do not infer the layout from token width or
resume an old run after this change. New runs require fresh normalization
statistics and the new timing-layout provenance.

Checks and fixtures remain outside the application repository under the
[verification policy](VERIFICATION_POLICY.md). Local numerical checks do not
establish GPU throughput, pretrained-checkpoint compatibility or hardware
safety. This source update does not start a cluster or robot process.
