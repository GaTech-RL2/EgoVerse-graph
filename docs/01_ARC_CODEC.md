# The ARC codec

## The idea

A diffusion policy normally predicts a chunk of actions sampled at a fixed
**time** step. ARC resamples the same trajectory by **arc length** instead: take
the native action window, walk along it accumulating distance, and emit `M`
waypoints spaced evenly in distance rather than in time. Slow, dense parts of a
trajectory stop consuming most of the token.

Two clocks run independently:

```
translation_arc = cumsum(|dxy|)   capped at D   ->  M samples
angle_arc       = cumsum(|dtheta|) capped at R  ->  M samples
```

Both produce `M` samples, then concatenate on the feature axis. **A row index is
a waypoint index, not a timestamp.** That is the single most important thing to
internalise: row `i` of a translation stream and row `i` of a rotation stream
are *not* the same instant.

`D` is in pixels, `M` is a count, `R` is in **radians**. The configs write
`26 degrees` as `0.4537856055185257`. Getting this wrong silently uncaps the
angular budget and every R cell in a sweep tokenizes identically — that has
already wasted one whole sweep.

## Token layouts

Source: `/Users/rpunamiya/Desktop/GEAR/sim_run/wt_codec/egomimic/rldb/zarr/planar_arc.py`

| name | shape | columns | constant |
|---|---|---|---|
| common-five (native, padded) | `[H,5]` | `x, y, cos, sin, grip` | `PLANAR_ACTION_DIM = 5` |
| row-split | `[2M,5]` | translation rows then rotation rows | — |
| stacked velocity | `[M,6]` | `x, y, v_xy, cos, sin, omega` | `PLANAR_ARC_STACKED_DIM = 6` |
| duration | `[M,6]` | `x, y, dt_tr, cos, sin, dt_rot` | `PLANAR_ARC_STACKED_DIM = 6` |
| **grip-carrying (articulated)** | `[M,7]` | `x, y, timing_t, cos, sin, timing_r, grip` | `PLANAR_ARC_GRIP_DIM = 7` |

Width-7 is what the articulated study uses. Columns 0-5 are **bit-identical** to
the width-6 token; column 6 is the addition. That was verified, not assumed.

### The grip channel

Decided with the user, and the reasoning matters:

- Grip **rides the translation clock**. It has no budget of its own and does not
  get a third arc.
- It is **interpolated**, not rate-decoded — grip is a position-like quantity,
  not a velocity.
- It is **clamped to [0,1]** on decode.
- 3-DOF tools (`u_socket`, `triangle`, `scoop`) have no grip. `PadPlanarAction`
  pads them with a constant 0, and the decoder slices back to 3 via
  `native_action_dim`.

Round-trip quality, measured: **0.001 px on xy, grip exact.**

## Where the code lives

| what | absolute path |
|---|---|
| tokenizers | `/Users/rpunamiya/Desktop/GEAR/sim_run/wt_codec/egomimic/rldb/zarr/planar_arc.py` |
| detokenize stages | `/Users/rpunamiya/Desktop/GEAR/sim_run/wt_codec/egomimic/pipeline/stages_arc.py` |
| articulated factories + decoders | `/Users/rpunamiya/Desktop/GEAR/sim_run/wt_codec/egomimic/rldb/embodiment/articulated_arc.py` |
| u_socket factories | `/Users/rpunamiya/Desktop/GEAR/sim_run/wt_codec/egomimic/rldb/embodiment/usocket_arc_velocity.py` |
| keymap, SliceActionTarget, PadPlanarAction | `/Users/rpunamiya/Desktop/GEAR/sim_run/wt_codec/egomimic/rldb/embodiment/pushshapes.py` |
| embodiment ids | `/Users/rpunamiya/Desktop/GEAR/sim_run/wt_codec/egomimic/rldb/embodiment/embodiment.py` |

### The width-7 classes

```python
class _ArcGripMixin:
    token_width = PLANAR_ARC_GRIP_DIM          # 7
    def _grip_waypoints(self, actions, cumulative, end): ...

class TokenizeArcVelocityStackedGrip(_ArcGripMixin, TokenizeUSocketArcVelocityStacked): ...
class TokenizeArcDurationGrip(_ArcGripMixin, TokenizeUSocketArcDuration): ...
```

Both set `token[:, 6] = self._grip_waypoints(actions, translation_arc, translation_end)`.

Decoders, in `stages_arc.py`:

```python
grip = self._decode_stream(tokens[..., 6:7], xy_distance, rate).clamp(0.0, 1.0)      # velocity
grip = self._interpolate_at_times(tokens[..., 6:7], xy_duration).clamp(0.0, 1.0)     # duration
```

## The transform list — and the trap in it

A dataset's `transform_list` is built by a factory. The articulated ones are:

```python
_prefix(keys, action_target_offset, raw_action_horizon) + [Tokenize...]
# _prefix == [SliceActionTarget(start=offset, horizon=raw_H)] if offset else []
#            + [PadPlanarAction(keys)]
```

`get_planar_keymap` fetches `action_horizon + action_target_offset` steps and
**relies on `SliceActionTarget` to drop `a_t`**, aligning the target with the
last observation in the window.

The u_socket ARC factories originally took **neither** argument, so
`action_target_offset: 1` vanished into `**_kwargs`: the loader handed over 81
steps and the tokenizer consumed all 81 starting at `a_t`. Fixed in `61bf2bf`;
all factories now honour the offset and the guard is on a non-zero offset so
`offset=0` configs are byte-identical. Full story in `docs/06_LANDMINES.md`.

**Audit command** — re-run this after touching any factory:

```bash
cd /Users/rpunamiya/Desktop/GEAR/sim_run/wt_codec
export PYTHONPATH="/Users/rpunamiya/Desktop/GEAR/sim_run/wt_codec:/Users/rpunamiya/Desktop/GEAR/sim_run/stubs"
# compose every experiment, instantiate its transform_list, assert:
#   offset != 0  =>  SliceActionTarget present
```
It was 0 risks across 43 configs at `61bf2bf`.

## Embodiment ids

Norm stats are keyed by these **numeric ids**, not names. Append-only — renumbering
invalidates every stored `norm_stats.json`.

| id | embodiment | DOF | grip | mechanism |
|---|---|---|---|---|
| 19 | u_socket | 3 | no | latch (prehensile) |
| 20 | chain_gripper | 4 | yes | grasp |
| 21 | gripper | 4 | yes | grasp |
| 22 | suction | 3 | engage | adhesion |
| 23 | umi | 4 | yes | grasp — **held out** |
| 24 | triangle | 3 | no | push |
| 25 | scoop | 3 | no | push — **held out** |
| 26 | flipper | 4 | yes | push |
| 27 | spring | 4 | yes | push |

Mechanism column is from the simulator's own agent docstrings in
`/Users/rpunamiya/Desktop/GEAR/EgoVerse/Tsimulation/sim_v2/pushshapes/agents.py`,
not inferred from names.

## Invariants worth asserting in any change

1. Width-7 columns 0-5 equal the width-6 token exactly.
2. Round-trip xy error stays ~0.001 px.
3. `R` is radians.
4. A tokenizer factory that accepts `action_target_offset` must also accept
   `raw_action_horizon`, or the slice horizon is wrong.
5. `token_width` and the model's expected action dim must agree — the preflight
   tool checks this by pushing a real token through the denoiser.
