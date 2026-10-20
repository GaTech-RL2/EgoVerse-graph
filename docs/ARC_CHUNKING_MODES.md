# Bimanual ARC chunking modes

Every retained `abc_arc/robot_bc` and `abc_arc/human_bc` visual experiment
accepts `abc.arc_chunking_mode=race`, `multistream`, or `joint_distance`.
The shared default is `joint_distance`. ARC recipes use the hybrid action
representation, a positive rotation target R, and `per_waypoint` timing.
Cartesian baseline recipes accept the override for evaluation and provenance;
their training targets remain Cartesian.

Supported experiment entrypoints are listed in
`egomimic/hydra_configs/experiment/abc_arc/README.md`. Unreferenced legacy ABC
Qwen, diffusion-policy, non-hybrid, and cotrain templates were removed; the
historical versions remain recoverable from Git. Other experiment families and
the data component still consumed by `experiment/abc` are unchanged.

Retained recipes default to `norm_stats.sample_frac=0.10`. Cached statistics are
immutable artifacts: this default does not convert a historical 20% cache into
a 10% cache. Use a verified matching cache or compute a new one. An intentional
historical-cache launch must explicitly declare its actual sampling fraction;
the loader's ARC action-contract check is not a dataset/sampling provenance check.

| Mode | Translation target and timing |
| --- | --- |
| `race` | End when the first of four streams spends its budget: either arm travelling D, or either arm's rotation turning R. Interpolate all four streams through that shared end frame. |
| `multistream` | Each arm targets D with its own translation clock and resamples its available path to M waypoints. An arm with less than D travels its available distance, then holds its last absolute pose. |
| `joint_distance` | Sum both arms' travel to D using one shared translation clock. |

For example, with D=40 cm and M=100, an arm that travels only 10 cm within
the bounded source window resamples that actual 10 cm across **all 100 motion
waypoints**. There is no null-waypoint grid padding. Detokenization uses the
predicted velocities to execute the 10 cm, then holds the last absolute pose
through the remaining bounded rollout horizon. The two arms resample their
own paths independently to the same M.

All three modes keep the **separate R target**, and in all three that target is
spent **per arm** on that arm's own SO(3) travel. Left and right rotation are
never summed into one clock. An arm that does not rotate therefore keeps
collapsed rotation waypoints instead of inheriting the other arm's angular
travel, and two arms each turning 0.3 rad under R=0.5 each keep their own 0.3
rather than jointly hitting the cap at 0.25 apiece.

Independent means each stream spends its own budget, not that a stream may read
outside the chunk. Under `multistream` and `joint_distance` no frame ends the
chunk, so reaching D does not truncate the raw rotation trajectory.

`race` is a stopping time rather than a budget, so it does define an end frame,
and rotation there is also a racer. Four streams compete: left translation for
D, right translation for D, left rotation for R, right rotation for R. The first
budget spent ends the chunk, and all four streams are then interpolated on their
own cumulative length through that frame. A race chunk consequently gives up
some translation when a rotation wins, and some rotation when a translation
wins, which is what keeps a waypoint's orientation inside the frames its
position came from. An arm that never spends its budget inside the window does
not enter the race.

The generic Zarr horizon resolver reads the bounded native source buffer and
selects enough rows to cover both translation and rotation endpoints. A missing
endpoint uses the available bounded prefix. The default buffer remains 600
frames; episode bounds and the existing repeat-last tail behavior still apply.
Explicit Human ARC modes retain native rows and use `dt=1/30`, bypassing legacy
stride/interpolation. Their untokenized evaluation copy has 100 native rows,
with repeat-last padding at short tails.

The public embodiment APIs and tokenizer use `arc_chunking_mode`. Omission
remains `None` when forwarded to the codec: R present infers `joint_distance`;
no R infers `multistream`. Any explicitly supplied mode requires a positive
finite R and `velocity_mode` of `per_waypoint` or `duration`. Legacy omitted-mode no-R APIs remain
usable. Human configs that omit both mode and R retain
their existing fixed source window and stride settings.
Old no-R resolver specs with `require_all_arms: false` retain race selection.

The shared visual recipe saves representation, mode, D, R, waypoint count,
timing mode, and control interval in `run_provenance.action_contract`, which
training retains in Lightning checkpoint configuration. The full Hydra config
also records the mode passed to each resolver, transform, and evaluator.

Changing modes changes action semantics even when tensor shapes are equal.
The existing cache had no transform-config hash. Retained ARC recipes now pass
their resolved action contract from `trainHydra` to normalization. New caches
store it per embodiment in `provenance.action_contracts`; normalizer state also
retains it across checkpoint/state round trips and cache rewrites.

When loading precomputed statistics, a tagged contract must match the requested
representation, mode, D, R, waypoint count, timing mode, and control interval.
A mismatch raises before any statistics are applied. Race and multistream
reject untagged caches. Existing normalization-mode and key-set checks still
apply.

Legacy **joint_distance** caches without action-contract metadata remain
eligible, with a warning, so established compatible runs do not need to
recompute normalization. Their existing data, D/R, and representation
compatibility must already have been verified; missing metadata cannot prove
those facts. Baseline and old configurations without an ARC contract retain
their prior behavior.

For migration to race or multistream, compute statistics with that recipe into
a new cache directory (omit `norm_stats.precomputed_norm_path` for that pass),
then use the resulting `norm_stats/norm_stats.json`. Do not relabel a joint
cache as a different mode. Existing compatible joint caches can continue using
their current path; a subsequent cache write records the configured contract.

External caches of transformed targets and resume signatures must also
distinguish the complete action contract. Pure image feature caches need no
action-mode distinction if they contain no transformed actions or
action-dependent sample selection.

## Open-loop decoding

### Opt-in six-stream first-row start delays

The non-E1 Cartesian tokenizer supports two versioned layouts:

| `velocity_mode` | `velocity_layout` | Row 0 of each timing column | Rows 1 through M-1 |
| --- | --- | --- | --- |
| `per_waypoint` | `rate_clock6_start_delay` | Start delay in seconds | Speed for the incoming interval i-1 → i |
| `duration` | `log_clock6_start_delay` | `log(start_delay + 0.001)` | `log(incoming_interval_seconds + 0.001)` |

The token remains M × 20: `[L xyz/ypr/grip, R xyz/ypr/grip,
T_L, T_R, R_L, R_R, G_L, G_R]`. Geometry row 0 is the initial
pose; there is no extra waypoint and no repeated final timing entry. Each
stream waits at its first waypoint until its own delay, then interpolates on
its own interval clock. Translation and rotation use linear and SO(3)
interpolation respectively; event grippers retain held-command semantics.

The delay is the time of the last stationary source frame preceding the first
moving source interval. Times use fractional source indices times configured
`dt`, not measured timestamps. No smoothing or control-period duration floor
is added. Rotation and distance-gripper endpoint clocks use first crossing, so
a trailing stationary plateau does not stretch the final moving interval.
Full holds have zero delay: duration intervals carry the configured window
time, while PW intervals are zero. Numerical SO(3) noise on fully held source
streams is not emitted as a nonzero rate.

For example, 4 waypoints spaced by 0.01 m after a 0.6 s initial wait and three
0.1 s moves have raw duration timings `[0.6, 0.1, 0.1, 0.1]`. Their waypoint
times are `[0.6, 0.7, 0.8, 0.9]`; before 0.6 s the first pose is held. In PW
the incoming rates happen to be 0.1 m/s in this example, but row 0 remains
seconds. A log token is exponentiated exactly once after unnormalization.

These layouts require `arc_chunking_mode: multistream` and
`arc_execution_cap_mode: waypoints`. Fractional distance-prefix execution is
explicitly rejected, not approximated by folding the delay into a movement.
Keeping K waypoint rows also keeps the start delay and exactly K-1 incoming
intervals. Replanning is at the earliest delay-plus-prefix-duration of a
moving stream, including grippers; stationary streams cannot win. If all six
streams hold, the configured hold-window convention applies. A caller's
explicit remaining-episode/output-horizon bound is still respected.

`OpenLoopSimEval` and `FirstStreamArcDecoder` share
`TokenizeBimanualArcLengthCartesian.first_delay_control_steps`. The
`BimanualIntervalArcDecoder` also accepts the new layouts for full-token
decoding. Declare `velocity_layout`, gripper mode/budget and D/R/M/dt in the
checkpoint-bound tokenizer, evaluator, rollout decoder and action contract;
do not infer the layout from the 20-column shape.

This is an opt-in target change, not a migration of old checkpoint tensors.
Existing `rate_clock`, `log_clock6`, absolute clocks and E1 codecs are unchanged.
New runs need fresh normalization statistics and a new experiment identity.
With quantile normalization, enable the existing
`norm_stats.widen_degenerate_quantiles: true` option and inspect row-zero
statistics: rare nonzero delays can otherwise be amplified by a zero-width
quantile range. The option falls back to observed min/max; it cannot cover
delays absent from the statistics sample, so check held-out normalized targets
as well. Never reuse legacy timing statistics merely because shapes match.
This change only separates leading holds; internal dwells still need suitable
supports, and PW cannot encode a zero-distance dwell duration or terminal
gripper dwell by speed alone.

### Legacy layouts

This is the decoder every open-loop evaluation, DTW score and validation video
uses.

1. Keep the first `execute_fraction * M` waypoints of the predicted token
   (`arc_execution_cap_mode: waypoints`).
2. Interpolate each stream at the 30 Hz control rate on its own clock: each
   arm's translation (D) and each arm's rotation (R). There is no fixed frame
   cap; an ARC chunk covers a constant distance in a variable number of frames.
3. Translation timing: a moving interval is timed by its position step over its
   position rate (`per_waypoint`) or by its stored duration (`duration`,
   `clock`, `log_clock`). The gripper rides the arm's translation clock and
   sets the time only on a hold interval, where the arm's position does not
   move. Do not take the longer of the position and gripper durations: a
   predicted gripper twitch (a tiny step over a near-zero rate) then stretches
   the whole arm's clock.
4. `multistream` executes until the first stream (any arm, translation or
   rotation) exhausts its retained waypoints, then replans. A stream that does
   not move in the prefix cannot end the chunk. `race` and `joint_distance`
   execute until every retained stream finishes.
5. Ground truth for a chunk is the recorded episode over exactly the executed
   frames, expressed in the replan frame's wrist frame. Distance DTW
   (`arc_chunking_global_dtw_v5`) concatenates the executed chunks and aligns
   them with the episode in world XYZ.
6. Validation videos (`video_overlay_mode: executed_chunk`) draw the executed
   chunk and its ground truth at the replan frame and hold that pair while the
   video advances through the chunk, then replan.
