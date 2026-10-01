# Consolidated ARC contract

This contract describes the integration of #197, #159 and #160. The numerical
implementations are `rldb/zarr/arc_length_tokenizer.py` (lab and hybrid ARC) and
`rldb/zarr/e1_arc_tokenizer.py` (E1). Evaluators and robot adapters call those
codecs. An identical tensor shape alone is never checkpoint compatibility.

## Representations

The Cartesian pose order is `[left xyz, left yaw/pitch/roll, left gripper,
right xyz, right yaw/pitch/roll, right gripper]`: 14 columns, metres and intrinsic
ZYX radians. The frame is declared independently; the YAM grid uses each arm's
observed end-effector frame. A camera-frame checkpoint cannot be substituted.
Normalization happens once in MultiDataset and is undone once before decoding.

| Representation | Native shape | Timing and padding | Decoder / rollout disposition |
| --- | --- | --- | --- |
| Uniform time | `(H,14)` | Native control samples; YAM baseline is 100 frames at 30 Hz | Cartesian adapter; visual recipes declare their time-grid decoder |
| Lab mean | `(M+1,14)` | M poses, one mean-rate row; cannot retain local dwell | Lab codec for diagnostics; no interval rollout approval |
| Lab interval velocity | `(2M,14)` stacked or `(M,28)` wide | M poses plus M rate rows; final rate repeats the last interval and is not consumed | Cartesian ARC codec; explicit interval adapter layout |
| Lab interval duration | `(2M,14)` stacked or `(M,28)` wide | Per-arm interval seconds occupy the arm's first timing slot | Same codec, explicitly `velocity_mode=duration` |
| E1 arcdur | `(M,16)` | Pose14, then left/right absolute interval seconds | E1 codec, `token_layout=e1_dur` |
| E1 arcvel | `(M,16)` | Pose14, then left/right path-speed profiles in m/s | E1 codec, `token_layout=e1_profile` |
| E1 arclogdur | `(M,16)` | Row 0 log mean slowness; subsequent rows log relative interval durations | E1 codec, `token_layout=e1_logdur` |
| Hybrid ARC | `(M,28)` default; explicit `(2M,14)` compatibility | Pose14 followed by rate14, with separate left/right translation and SO(3) clocks | Same training/eval/interval codec; automatic visual-policy rollout remains explicitly unsupported pending a validated profile |

For wide lab/hybrid tokens, columns 14–27 are rates, not another pose.
Per-arm rate fields are xyz velocity, angular rate, and gripper rate. Hybrid
angular rates use relative SO(3) rotation vectors; old non-hybrid interval
encoding retains its original Euler-difference convention. Narrow/wide
packing changes storage only. E1's two timing columns are a distinct format;
`e1_dur` and `e1_profile` cannot be inferred from their identical shape.

The source layouts remain explicit: legacy B/C lab configs pin `stacked`;
canonical visual/hybrid A recipes use `wide`. Mean tokens, E1 16-column tokens,
and hybrid 28-column tokens have separate declarations. Shape, finite-value,
frame, and saved model/data/codec bindings fail closed at their boundaries.

## Distances, clocks and reconstruction

`D` is a translation distance in metres, `R` a per-arm rotation distance in
radians, and `M` the number of waypoint rows. Explicit hybrid modes require a
positive finite R and `per_waypoint` velocity. Omitting hybrid mode and R retains
the old non-hybrid representation; it does not invent a rotation budget.

| Hybrid mode | Translation clock | Rotation clock | Source boundary |
| --- | --- | --- | --- |
| `race` | Per-arm D | Per-arm R | First of the four streams reaches its budget; all streams crop at that raw time |
| `multistream` | Each arm independently reaches D | Each arm independently reaches R | Preserve each stream and hold completed streams |
| `joint_distance` | Sum of both arms' travel reaches D | Each arm independently reaches R | Shared translation timing, independent rotation timing |

A translation cap must not truncate another arm's still-moving rotation.
In particular, R is never the sum of left and right rotation. The obsolete
#160 shared-rotation `_hybrid_waypoints` and `_hybrid_per_waypoint_velocity`
definitions were removed: an automatic merge had placed them after the newer
methods, silently overriding them. A duplicate-class/method AST audit now
finds no such overrides in project code/tests.

Waypoint progress is mapped to fractional **source** frame times, rather than
to chord lengths along the resampled path. Decode uses the same declared dt,
per-arm duration helpers and endpoint holds. Shape and timing endpoints remain
aligned during execution-prefix and distance/rotation capping. Open-loop
evaluation executes a prefix, then replans from the recorded observation at
the next boundary; it is not a physical closed-loop success-rate experiment.
DTW retains its M28 shape handling, per-arm rotation caps, metric-frame
conversion and explicit complexity limits.

Zero/short/held streams retain meaningful anchors, orientation and gripper
behavior; rectangular timing padding is not a presence flag. E1 deliberately
retains its historical 200-source-step translation hold rule. A 400-frame
window can therefore hold a sufficiently slow arm, even when another arm
reaches D. This is distinct from the hybrid codec's hold rules and is not
silently changed to improve replay scores. E1 fixed-spacing/log-duration
padding is inferred from repeated geometry; it is not an interchangeable
hybrid null token.

## Source windows and reproducibility

`bimanual_arc.get_keymap(..., embodiment=yam, keymap_mode=cartesian)` uses
100 source frames unless `yam_source_frames` explicitly selects another
positive integer. `400` is honored in the w400 cell. Booleans, non-integral or
nonpositive values, and an override for a different embodiment/mode are errors.
Dynamic ARC distance horizons are not coerced to integer windows. Canonical
YAM hybrid loading keeps its 200-frame source buffer, and explicit human ARC
uses native 30 Hz samples; legacy human stride behavior remains separate.

The 64 #160 grid cells keep their pools, split/count/hash pins, seeds, steps,
normalization fractions, optimizer and network tensors. Two cells explicitly
select sample-mean loss to retain C's reduction default. Packaged allowlists
replace absolute paths into the author's checkout; their contents and
pre-split membership semantics are unchanged. Full saved normalization contexts
include preprocessing/source-window metadata. Same-embodiment sources share
one pooled normalizer; different action/frame/preprocessing contracts reject.

## Independent evidence and compatibility limits

- `test_consolidated_arc_contract.py` decodes hand-authored linear/rotating
  paths through training, open-loop evaluation and the numeric robot adapter
  in both storage layouts, checking independent expected positions and angles
  at `1e-7` absolute tolerance. Same-shape mode changes invalidate checkpoint
  bindings.
- `test_yam_source_frames.py` slices raw WXYZ poses at fixed indices 17 and
  203, applies real keymaps/transforms and independently checks geometry and
  time targets at `1e-6`. Seventeen output arrays match pinned C exactly;
  see `evidence/consolidation/yam-window-source-parity.json`.
- Existing E1, hybrid, hold, zero/short, M28 DTW, cap, frame, normalization,
  visual-stem, HPT/PI and checkpoint tests remain in the assembled suite.
- Lossy geometry is not claimed to round-trip arbitrary curves exactly.
  Tests use preserved invariants and explicit tolerances, not only a codec's
  own encode/decode agreement.

Bound checkpoints must use their saved configuration and normalization state.
Source C/direct-ResNet visual checkpoints do not acquire A's pretrained
ResNet+MLP+jitter architecture merely by selecting the same recipe name today.
Shared-rotation checkpoints cannot be relabeled per-arm R checkpoints. These
changes require original-runtime reproduction or an explicitly verified
migration; no checkpoint conversion or hardware deployment was performed.
