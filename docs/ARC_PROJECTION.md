# ARC Projection

ARC Projection measures progress at one point rigidly attached to each end
effector. It uses the existing HPT + flow network and the single-stream E1
duration/velocity codecs. Each arm has one clock shared by xyz, rotation and
gripper. The two arms have separate clocks queried on the same control-time grid.

For end-effector position `p` and orientation `R`:

```text
offset = (0.02, 0.02, 0.02) metres, in end-effector-local axes
q(t) = p(t) + R(t) @ offset
s(t) = cumulative sum of ||q(t + 1) - q(t)||
```

This is one point, 2 cm along each local axis, with a total offset of about
3.46 cm. It is not three independently timed points. Rotation uses the repo's
intrinsic ZYX Euler convention. Under an end-effector-relative action frame,
`p` and `R` are already relative to the anchor; the offset still rotates with
the instantaneous end effector.

## Supports and clocks

The tokenizer places M supports uniformly over `min(D, s[-1])`, using the
projected path to find fractional source frames. It samples the original
end-effector xyz, rotation and gripper at those same frames. Tokens therefore
remain `(M, 16)`:

```text
[left xyz, left ypr, left grip, right xyz, right ypr, right grip,
 left timing, right timing]
```

The output pose positions are actual end-effector positions, not `q`. The
decoder reconstructs `q` from each predicted pose to measure the waypoint
path. It then queries the arm's clock at each control tick, interpolating xyz
and grip linearly and rotation with SLERP.

- **`arc_projection_dur` / `e1_projection_dur`:** elapsed seconds in rows
  `0..M-2`; the last row repeats the last interval and is ignored by the clock.
  Decode clamps intervals nonnegative and cumulatively sums them. The existing
  whole-clock mean-speed bounds apply using projected distance.
- **`arc_projection_vel` / `e1_projection_vel`:** each waypoint segment's mean
  projected speed in m/s: its projected length over the true seconds it took
  (`speed_encoding: interval`, the default since 2026-10-20). Decode is
  `dt_i = ds_i / v_i`, so on ground truth the clock equals
  `arc_projection_dur`'s exactly and pauses are kept. The original encoding,
  the raw path speed sampled at each support and integrated with the
  trapezoidal rule, is `speed_encoding: point`; it is kept only to decode
  checkpoints trained before 2026-10-20, and it is the only one that reads
  `model.projection_speed_smooth_frames`.

The projected speed floor is explicitly `1e-6 m/s`, shared by training-side
decode, validation and rollout. The original translation floor of `0.01 m/s`
would artificially speed up slow rotations of a 2 cm lever. This lower floor
preserves those motions; near-zero predicted velocity can still create long
intervals, as with any inverse-speed clock.

The maintained leaves use `D=0.33 m`, `M=200` and 30 Hz. D measures
projected-point travel, so it couples translation and rotation in one budget.
There is no independent rotation cutoff. Validation decodes the whole token and
scores a fixed control-time prefix (25 ticks, tick-prefix regime). The rollout
decoder executes the first 50 % of the waypoints and replans when the first arm
reaches its last kept waypoint (`first_stream: 1`, `execute_percent: 50`), with
the dashboard's replay-speed controls. Absolute waypoint timing is not
supported for projection layouts; `arc_projection_dur` is delta timing only.

## Configuration

The reusable model base is `model/e1/hpt_flow_wrists_ft_arc_projection`.
Its two leaves are:

- `model/e1/hpt_flow_wrists_ft_arc_projection_dur`
- `model/e1/hpt_flow_wrists_ft_arc_projection_vel`

Complete experiment selectors are:

- `yam_arc_grid/bc_rl2_stattempo_slowpace366_arc_projection_dur_m200_d033`
- `yam_arc_grid/bc_rl2_stattempo_slowpace366_arc_projection_vel_m200_d033`

They inherit the maintained stattempo BC graph, optimizer, scheduler and frozen
366/24 stationery split. HPT is width 640 with eight blocks; flow is width 384
with ten blocks. The token head is 16-wide, matching ordinary single-stream
ARC. No new network modules are introduced.

`model.projection_offset_m`, `model.projection_min_speed` and
`model.projection_speed_smooth_frames` own the geometry and speed settings.
Training/validation transforms and model-owned inference declarations refer to
these fields. Geometry, speed floor and timing variant are included in the
inference compatibility contract: existing xyz-only ARC checkpoints and
normalizers cannot be silently relabeled as ARC Projection. Fit fresh action
normalization and train a new model for this representation.

The new selectors use the shared open-loop evaluator. The older E1 tempo
accumulator is not a supported evaluator for these variants; its translation-
progress clock metrics would need a separate projected-progress definition.

## Representation limits

A single point is not a complete pose metric. Rotation about the offset vector
does not move it; translation can also cancel its rotational motion. Entirely
stationary projected paths inherit the single-stream codec's decoded hold.
Gripper-only motion cannot advance this clock. When projected motion ends, the
gripper shares the final pose hold.

This change retains existing single-stream dwell/initial-delay semantics. In
particular, leading stationary time is not encoded as a separate delay channel,
and the velocity profile cannot exactly represent a finite dwell at zero
distance. Intermediate source holds can be spread between supports. This is a
change of geometric progress coordinate, not a new event or dwell codec.

On curved projected paths, source-frame chords and decoded waypoint chords
approximate the same curve at different resolutions; velocity roundtrips can
therefore have a small discretization error. Shorter projected paths use all
M supports by default; fixed-spacing behavior remains available through the
existing option.

## Verification

`tests/test_arc_projection.py` covers local-axis geometry, rigid-frame
invariance, pure rotation with a stationary end effector, synchronized gripper
timing, translation-only parity, combined motion, projected-distance cutoffs,
the single-point blind axis, parameter validation, checkpoint contract identity,
training/evaluator/rollout decode parity, and a small CPU graph backward/sample.
The source-pinned stattempo tests guard the existing recipes and codecs.

No training job or physical rollout is implied by these checks. Before a full
run, use the normal training preflight, dataset/normalizer checks and GPU smoke
for the exact new recipe and source commit.
