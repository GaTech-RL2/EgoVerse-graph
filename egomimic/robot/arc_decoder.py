"""Decode native ARC policy outputs before robot frame transforms.

Ported from aidan/shorts-extreme (55932d99), rollout-arc.py. This module has no
robot I/O. Call it after action unnormalization, then apply the controller's
camera/base frame transforms to the returned canonical poses.

Codec
-----
Every token is decoded by the tokenizer that trains it: the live codecs in
``egomimic.rldb.zarr`` (``e1_arc_tokenizer``, ``arc_projection_tokenizer``,
``arc_length_tokenizer``). No checkpoint-era copies are carried; a bundle trained
by another revision of a tokenizer needs that revision's runtime.

Replay tempo
------------
An arc token is a path plus a clock, so the path can be replayed at a tempo the
training data never contained without touching the geometry. ``speed`` multiplies
the tempo of the MOVING phases and ``hold_speed`` the tempo of the slow phases
(holds, grasp dwell, careful placement: every instant at which both arms' decoded
path speed is under ``hold_threshold`` m/s). 1.0 / 1.0 is the demonstrated tempo
and takes the unmodified decode path. The two are separate because a hold is where
the gripper physically closes, and that time does not shrink with the arm's.

The warp is ONE monotone map from wall time to token-clock time, applied to both
arms, so bimanual coordination is exactly what the token encoded; only the rate
at which the pair advances changes. It acts on the clock, before the codec's own
arc-length resampling, so rotation still goes through the codec's SLERP. Tokens
without a per-waypoint clock (lab and the cartesian layouts) take a uniform speed
only, as a codec whose control period is scaled.

Token shapes (M = resampled_vector_length):
  e1_dur / e1_logdur / e1_profile  (M, 16)  E1 arcdur / arclogdur / arcvel
  e1_projection_dur / _vel         (M, 16)  the same clocks on a projected progress
                                            coordinate
  e1_durhyb                        (M, 18)  E1 hybrid arcdur: independent per-arm
                                            translation and rotation clocks
  e1_profhyb                       (M, 18)  E1 arcvelhyb: the same streams, timed by
                                            interval speeds
  e1_durtri / e1_proftri           (M, 20)  E1 arcdurtri / arcveltri: the gripper is
                                            a third stream per arm with its own clock
  e1_durevt / e1_stkevt            (M, 20)  Ryan's stream codec with an event-list
                                            gripper
  e1_durref / e1_stkref            (M, 24)  two clocks per arm over a rotation6d
                                            layout
  lab                              (M+1, 14) the bimanual mean-speed token

``FirstStreamArcDecoder`` decodes the multistream interval tokens of
``arc_length_tokenizer`` (wide (M, 28), stacked (2M, 14), four-clock duration
(M, 18)) and executes them as validation scores them: the first N % of the
waypoints, replanning when the first moving stream ends.
``BimanualIntervalArcDecoder`` decodes the same tokens whole.

``TimeChunkRetimer`` gives a time-indexed policy (no token, no decoder) the same
control: its (H, 14) chunk is a path on a uniform clock, read through the same warp.
speed == hold_speed is the naive baseline speed-up (2x = every other row).
"""

import numpy as np
import torch
from scipy.spatial.transform import Rotation, Slerp

from egomimic.rldb.zarr import arc_length_tokenizer as training_cartesian
from egomimic.rldb.zarr.arc_length_tokenizer import (
    TokenizeBimanualArcLengthCartesian as TrainingCartesianCodec,
)
from egomimic.rldb.zarr.arc_length_tokenizer import (
    bimanual_arc_token_shape,
    cumulative_arc_length,
)
from egomimic.rldb.zarr.arc_projection_tokenizer import (
    PROJECTION_LAYOUTS,
    PROJECTION_OFFSET_M,
    TokenizeBimanualArcProjection,
)
from egomimic.rldb.zarr.e1_arc_tokenizer import (
    ARM_LAYOUT,
    DEFAULT_ROTATION_DISTANCE_UNIT,
    REF_ARM_LAYOUT,
    TokenizeBimanualArcLengthE1,
)
from egomimic.robot.arc_speed import ARC_SPEED_RANGE, validate_arc_speed  # noqa: F401

# Every E1 layout and the codec mode it decodes with.
E1_VELOCITY_MODE = {
    **PROJECTION_LAYOUTS,
    "e1_dur": "dur",
    "e1_logdur": "logdur",
    "e1_profile": "profile",
    "e1_durhyb": "durhyb",
    "e1_profhyb": "profhyb",
    "e1_durtri": "durtri",
    "e1_proftri": "proftri",
    "e1_durevt": "durevt",
    "e1_stkevt": "stkevt",
    "e1_durref": "durref",
    "e1_stkref": "stkref",
}
# Multi-clock layouts whose row M-1 carries each stream's start delay.
HYBRID_LAYOUTS = {"e1_durhyb", "e1_profhyb", "e1_durtri", "e1_proftri"}
# Layouts whose streams run on more than one clock per arm (they take the
# hybrid rotation budget R).
E1_HYBRID_LAYOUTS = set(E1_VELOCITY_MODE) - {
    "e1_dur",
    "e1_logdur",
    "e1_profile",
    *PROJECTION_LAYOUTS,
}
REF_LAYOUTS = {"e1_durref", "e1_stkref"}
ARC_TOKEN_LAYOUTS = ("lab", *E1_VELOCITY_MODE)
# Token-clock seconds the tempo eases over at a hold <-> moving transition, so the
# commanded speed does not step when an arm crosses ``hold_threshold``.
RATE_RAMP_S = 0.2
_WARP_OVERSAMPLE = 4  # warp grid points per control step


class BimanualArcRoundTrip:
    """Temporary ARC encode/decode around a canonical Cartesian chunk.

    The input is an ordinary baseline ``(B, H, 14)`` prediction.  The codec
    uses M=100 and computes D from that prediction on every call.  Because the
    non-hybrid tokenizer applies one cap independently to each arm, the scalar
    D is the larger of the two arm path lengths; the shorter arm is clipped at
    its own exact path length and then represented with trailing holds.
    """

    def __init__(
        self,
        resampled_vector_length=100,
        dt=1 / 30,
        velocity_mode="per_waypoint",
        distance_mode="predicted_span",
    ):
        if int(resampled_vector_length) != 100:
            raise ValueError("The temporary ARC round trip requires M=100")
        if velocity_mode != "per_waypoint":
            raise ValueError(
                "The temporary ARC round trip requires per_waypoint velocities"
            )
        if distance_mode != "predicted_span":
            raise ValueError("Unsupported temporary ARC round-trip distance mode")
        if not np.isfinite(float(dt)) or float(dt) <= 0:
            raise ValueError("ARC round-trip dt must be positive and finite")
        self.M = 100
        self.dt = float(dt)
        self.velocity_mode = velocity_mode
        self.distance_mode = distance_mode
        self.last_distances = ()

    @staticmethod
    def _distance(row):
        left = cumulative_arc_length(row[:, 0:3])[-1]
        right = cumulative_arc_length(row[:, 7:10])[-1]
        return max(float(left), float(right), 1e-6)

    def __call__(self, actions):
        if torch.is_tensor(actions):
            actions = actions.detach().double().cpu().numpy()
        values = np.asarray(actions, dtype=np.float64)
        if values.ndim == 2:
            values = values[None]
        if values.ndim != 3 or values.shape[-1] != 14 or values.shape[1] < 2:
            raise ValueError("ARC round trip expects canonical actions shaped (B,H,14)")
        if not np.isfinite(values).all():
            raise ValueError("ARC round-trip input must be finite")

        decoded = []
        distances = []
        for row in values:
            distance = self._distance(row)
            codec = TrainingCartesianCodec(
                min_distance_unit=distance,
                resampled_vector_length=self.M,
                dt=self.dt,
                velocity_mode=self.velocity_mode,
            )
            tokens = codec.transform({"actions_cartesian": row.copy()})[
                "actions_cartesian"
            ]
            decoded.append(codec.detokenize(tokens, action_horizon=len(row)))
            distances.append(distance)
        self.last_distances = tuple(distances)
        return np.stack(decoded)


def check_execute_percent(num_waypoints, percent):
    """First-stream execution keeps an exact whole prefix of >= 2 waypoints."""
    if type(percent) is not int or not 0 < percent <= 100:
        raise ValueError("execute_percent must be an integer in (0, 100]")
    if num_waypoints * percent % 100 or num_waypoints * percent // 100 < 2:
        raise ValueError(
            f"{percent} % of M={num_waypoints} waypoints is not a whole prefix of at least two"
        )
    return percent


class ChunkTermination:
    """Dashboard-selectable chunk termination shared by the ARC decoders.

    ``first_stream`` 1: multistream fastest-stream termination -- cap the token at
    ``execute_percent`` % of its waypoints, detokenize, and replan when the first
    moving stream reaches its cap (``last_stats["replan_steps"]``).
    ``first_stream`` 0: the original method -- decode the whole token and let the
    policy execute a fixed ``replan_every`` actions.
    """

    @property
    def first_stream(self):
        return self._first_stream

    @first_stream.setter
    def first_stream(self, value):
        if type(value) is not int or value not in (0, 1):
            raise ValueError(
                "first_stream must be 0 (fixed repredict) or 1 (fastest stream)"
            )
        self._first_stream = value

    @property
    def execute_percent(self):
        return self._execute_percent

    @execute_percent.setter
    def execute_percent(self, percent):
        self._execute_percent = check_execute_percent(self.M, percent)

    def check_execute_percent(self, percent):
        return check_execute_percent(self.M, percent)

    def execution_steps(self):
        """Generic prefix capability consumed equally by sequence and robot inference."""
        return (self.last_stats or {}).get("replan_steps")


class ReplayTempo:
    """speed / hold_speed state and the wall-time -> clock warp shared by the ARC
    decoder and the time-chunk retimer. Subclasses set dt, action_horizon,
    hold_threshold and ``holds`` (whether hold_speed can act on their output)."""

    def _init_tempo(self, speed, hold_speed, hold_threshold):
        if (
            isinstance(hold_threshold, bool)
            or not isinstance(hold_threshold, (int, float))
            or not np.isfinite(hold_threshold)
            or hold_threshold < 0
        ):
            raise ValueError("ARC hold_threshold must be a finite speed >= 0 in m/s")
        self.hold_threshold = float(hold_threshold)
        self.last_stats = None
        self.set_speed(speed, hold_speed)

    def set_speed(self, speed, hold_speed=None):
        """Set the tempo multipliers used from the NEXT decoded token on.

        ``hold_speed=None`` keeps holds in step with ``speed`` (a uniform speed-up).
        Both values are validated before either is stored.
        """
        speed = validate_arc_speed(speed, "speed")
        hold_speed = (
            speed
            if hold_speed is None
            else validate_arc_speed(hold_speed, "hold_speed")
        )
        self.speed, self.hold_speed = speed, hold_speed

    # Integer percent views for the rollout profile's typed inference controls.
    @property
    def speed_percent(self):
        return int(round(self.speed * 100))

    @speed_percent.setter
    def speed_percent(self, value):
        self.set_speed(value / 100, self.hold_speed)

    @property
    def hold_speed_percent(self):
        return int(round(self.hold_speed * 100))

    @hold_speed_percent.setter
    def hold_speed_percent(self, value):
        self.set_speed(self.speed, value / 100)

    def _warp(self, clocks, cums, end=np.inf):
        """Token-clock time to read each of the H control ticks at, and the share of
        the decoded span that was a hold. rate = d(clock) / d(wall). ``end`` is the
        clock time at which the token's path runs out: nothing after it is a hold,
        it is just the endpoint being repeated, so it is left out of the share."""
        wall = self.dt * np.arange(self.action_horizon, dtype=np.float64)
        top = max(self.speed, self.hold_speed)
        n = (
            int(np.ceil(self.action_horizon * top * _WARP_OVERSAMPLE))
            + _WARP_OVERSAMPLE
            + 1
        )
        tau = (self.dt / _WARP_OVERSAMPLE) * np.arange(
            n, dtype=np.float64
        )  # covers wall[-1] * top
        speed = np.zeros(n - 1)
        for clock, cum in zip(clocks, cums):
            if cum[-1] >= 1e-9:  # a stationary arm never makes the pair "moving"
                speed = np.maximum(
                    speed, np.diff(np.interp(tau, clock, cum)) / np.diff(tau)
                )
        moving = speed >= self.hold_threshold
        rate = np.where(moving, self.speed, self.hold_speed)
        width = max(1, int(round(RATE_RAMP_S / (self.dt / _WARP_OVERSAMPLE))))
        if width > 1 and self.speed != self.hold_speed:
            padded = np.pad(rate, (width // 2, width - 1 - width // 2), mode="edge")
            rate = np.convolve(padded, np.ones(width) / width, mode="valid")
        elapsed = np.concatenate(
            ([0.0], np.cumsum(np.diff(tau) / rate))
        )  # wall time at each tau
        times = np.interp(wall, elapsed, tau)
        used = tau[1:] <= min(times[-1], end)
        return times, (float(1.0 - moving[used].mean()) if used.any() else None)


class BimanualArcDecoder(ReplayTempo, ChunkTermination):
    def __init__(
        self,
        token_layout="e1_dur",
        min_distance_unit=0.4,
        resampled_vector_length=100,
        dt=1 / 30,
        action_horizon=100,
        rotation_distance_unit=None,
        speed=1.0,
        hold_speed=1.0,
        hold_threshold=0.05,
        execute_percent=None,
        first_stream=0,
        projection_offset_m=None,
        projection_min_speed=1e-6,
        duration_timing="delta",
        speed_floors=None,
        # A bundle exported before 2026-10-20 names no speed encoding and was trained on
        # point speeds; current contracts declare "interval".
        speed_encoding="point",
    ):
        if token_layout not in ARC_TOKEN_LAYOUTS:
            raise ValueError(f"token_layout must be one of {ARC_TOKEN_LAYOUTS}")
        self.token_layout, self.dt = token_layout, float(dt)
        self.holds = (
            token_layout in E1_VELOCITY_MODE
        )  # only a per-waypoint clock separates holds
        # Every ARC token is executed the same way whatever its stream count:
        # cap, detokenize, replan when the first moving stream reaches its cap.
        # The dashboard can switch back to the fixed repredict (first_stream=0).
        self.first_stream = first_stream
        self._init_tempo(speed, hold_speed, hold_threshold)
        self.M = int(resampled_vector_length)
        self.action_horizon = int(action_horizon)
        if self.M < 2 or self.action_horizon < 1 or dt <= 0 or min_distance_unit <= 0:
            raise ValueError("Invalid ARC distance, time, horizon or waypoint count")
        if duration_timing != "delta" and token_layout not in (
            "e1_dur",
            "e1_projection_dur",
        ):
            raise ValueError(
                "Absolute waypoint timing requires token_layout 'e1_dur' or 'e1_projection_dur'"
            )
        projection = token_layout in PROJECTION_LAYOUTS
        if projection_offset_m is not None and not projection:
            raise ValueError("projection_offset_m requires an ARC Projection layout")
        if speed_floors is not None and token_layout not in E1_VELOCITY_MODE:
            raise ValueError("speed_floors apply to the E1 layouts only")
        if execute_percent is None:
            # 50 % unless M has no whole 50 % prefix; the policy then sets it.
            valid = self.M * 50 % 100 == 0 and self.M * 50 // 100 >= 2
            execute_percent = 50 if valid else 100
        self.execute_percent = execute_percent
        kwargs = dict(
            min_distance_unit=min_distance_unit, resampled_vector_length=self.M, dt=dt
        )
        self._uniform_codec = None  # (speed, codec) for a uniform speed-up
        if token_layout in E1_VELOCITY_MODE:
            if token_layout in E1_HYBRID_LAYOUTS:
                rotation_distance_unit = float(
                    DEFAULT_ROTATION_DISTANCE_UNIT
                    if rotation_distance_unit is None
                    else rotation_distance_unit
                )
                if (
                    not np.isfinite(rotation_distance_unit)
                    or rotation_distance_unit <= 0
                ):
                    raise ValueError(
                        "Hybrid ARC rotation distance must be finite and positive"
                    )
                kwargs["rotation_distance_unit"] = rotation_distance_unit
            elif rotation_distance_unit is not None:
                raise ValueError(
                    "A rotation distance requires an E1 hybrid token layout"
                )
            kwargs.update(
                velocity_norm="path",
                velocity_mode=E1_VELOCITY_MODE[token_layout],
                duration_timing=duration_timing,
                speed_encoding=speed_encoding,
            )
            if speed_floors is not None:
                kwargs["speed_floors"] = speed_floors
            if projection:
                self.codec = TokenizeBimanualArcProjection(
                    **kwargs,
                    projection_offset_m=PROJECTION_OFFSET_M
                    if projection_offset_m is None
                    else projection_offset_m,
                    min_speed=projection_min_speed,
                )
            else:
                self.codec = TokenizeBimanualArcLengthE1(**kwargs)
            self.shape = (self.M, self.codec.token_dim)
        else:  # lab: the (M+1, 14) mean-speed token
            if rotation_distance_unit is not None:
                raise ValueError("The lab layout has no rotation budget")
            self.shape = (self.M + 1, 14)
            self._uniform_kwargs = kwargs
            self.codec = TrainingCartesianCodec(**kwargs)

    def __call__(self, native_tokens):
        if torch.is_tensor(native_tokens):
            native_tokens = native_tokens.detach().double().cpu().numpy()
        values = np.asarray(native_tokens, dtype=np.float64)
        if values.ndim == 2:
            values = values[None]
        if values.ndim != 3 or values.shape[1:] != self.shape:
            raise ValueError(
                f"Expected native tokens (B,{self.shape[0]},{self.shape[1]}), got {values.shape}"
            )
        if not len(values) or not np.isfinite(values).all():
            raise ValueError("Native tokens must be nonempty and finite")
        return np.stack([self._decode(row) for row in values])

    def _decode(self, row):
        """Cap, detokenize, and end the chunk at the first stream to reach its cap.

        Every layout runs the same three steps whatever its stream count:
        keep the first ``execute_percent`` % of the waypoints (``_cap``), decode
        that capped token, and set ``last_stats["replan_steps"]`` to the tick at
        which the fastest MOVING stream reaches its last capped waypoint. The
        policy executes exactly those rows. Each stream holds at its own cap.
        With ``first_stream`` 0 the whole token is decoded and the policy's fixed
        ``replan_every`` decides the replan, as before this mode existed.
        """
        h = self.action_horizon
        token = self._cap(row) if self.first_stream else row
        ends = self._stream_ends(token)
        if self.token_layout in E1_VELOCITY_MODE:
            clocks, cums = self._translation_clocks(token)
            times, hold_fraction = self._warp(clocks, cums, max(ends, default=0.0))
            if self.speed == 1.0 and self.hold_speed == 1.0:
                decoded = self.codec.detokenize(token, action_horizon=h)
            else:
                decoded = self.codec.detokenize(token, action_horizon=h, times=times)
        else:
            # The lab token has no per-waypoint clock to warp: holds cannot be
            # told apart, so ``hold_speed`` has nothing to act on. A uniform tempo
            # is exactly a codec whose control period is scaled, rebuilt only when
            # ``speed`` moves.
            times, hold_fraction = (
                self.dt * self.speed * np.arange(h, dtype=np.float64),
                None,
            )
            if self.speed == 1.0:
                decoded = self.codec.detokenize(token, action_horizon=h)
            else:
                if self._uniform_codec is None or self._uniform_codec[0] != self.speed:
                    kwargs = {**self._uniform_kwargs, "dt": self.dt * self.speed}
                    self._uniform_codec = (self.speed, TrainingCartesianCodec(**kwargs))
                decoded = self._uniform_codec[1].detokenize(token, action_horizon=h)
        first = min(ends, default=0.0)
        # Ticks read strictly before the first stream's cap; at 1x, ceil(end / dt).
        steps = int(np.searchsorted(times, first - 1e-9 * self.dt, side="left"))
        valid = int(np.searchsorted(times, max(ends), side="right")) if ends else 0
        self.last_stats = {
            "speed": self.speed,
            "hold_speed": self.hold_speed if self.holds else None,
            "horizon": h,
            "valid_steps": min(valid, h),
            "hold_fraction": hold_fraction,
            "stream_ends_s": ends,
        }
        if self.first_stream:
            self.last_stats.update(
                replan_steps=min(max(1, steps), h),
                execute_waypoints=self.M * self.execute_percent // 100,
            )
        return decoded

    def _arm_xyz(self, token, k):
        """One arm's waypoint positions, whatever the layout's column order."""
        layout = REF_ARM_LAYOUT if self.token_layout in REF_LAYOUTS else ARM_LAYOUT
        offset = layout[k][0]
        return token[:, offset : offset + 3]

    def _translation_clocks(self, token):
        """Per arm, the translation clock at the waypoints and the cumulative path
        length there: what the tempo warp separates holds from motion with."""
        clocks = [
            arm["translation"]
            for arm in self.codec.stream_clocks(token, self.action_horizon)
        ]
        cums = [cumulative_arc_length(self._arm_xyz(token, k)) for k in range(2)]
        return clocks, cums

    def _cap(self, row):
        """The first ``execute_percent`` % of the waypoints, as a token of the same
        layout whose clocks equal the full token's clocks on those waypoints."""
        n = self.M * self.execute_percent // 100
        layout = self.token_layout
        if layout == "lab":
            # One chord rate per arm, which the codec converts to an arc rate with
            # the token's own arc/chord ratio: rescale so the arc rate is kept.
            waypoints, rate = row[: self.M], row[self.M].copy()
            for off in (0, 7):
                rate[off : off + 3] *= _arc_over_chord(
                    waypoints[:, off : off + 3]
                ) / _arc_over_chord(waypoints[:n, off : off + 3])
            return np.concatenate((waypoints[:n], rate[None]))
        token = row[:n].copy()  # every E1 layout: wide per-waypoint rows
        if layout in HYBRID_LAYOUTS:
            token[-1, 14:] = row[-1, 14:]  # row M-1 is each stream's start delay
        elif layout == "e1_logdur":
            # Row 0 owns the whole token's time: give the prefix exactly its share.
            for k, clock in enumerate(self.codec.clock_at_waypoints(row)):
                span = float(cumulative_arc_length(self._progress(row[:n], k))[-1])
                if span > 1e-9 and clock[n - 1] > 1e-12:
                    token[0, 14 + k] = np.log(clock[n - 1] / span)
        # Event / reference layouts repeat their last interval row and have no start
        # delay, and absolute waypoint times are already per row: the prefix stands.
        return token

    def _progress(self, token, k):
        """The coordinate whose arc length the single-clock E1 codecs measure progress by."""
        off, ypr = ARM_LAYOUT[k][0], ARM_LAYOUT[k][1]
        return self.codec.progress_coordinates(
            token[:, off : off + 3], token[:, ypr : ypr + 3]
        )

    def _stream_ends(self, token):
        """Seconds each MOVING stream takes to reach its last capped waypoint."""
        if self.token_layout in E1_VELOCITY_MODE:
            return self.codec.stream_ends(token, self.action_horizon)
        ends = []  # lab: one chord rate per arm, converted to the arc rate
        for off in (0, 7):
            total = float(cumulative_arc_length(token[:-1, off : off + 3])[-1])
            speed = np.linalg.norm(token[-1, off : off + 3]) * _arc_over_chord(
                token[:-1, off : off + 3]
            )
            if total >= 1e-9 and speed >= 1e-8:  # otherwise the codec holds the arm
                ends.append(total / speed)
        return ends


def _arc_over_chord(xyz):
    """Arc length over chord, the factor the lab codec turns a chord rate into an arc rate by."""
    total = float(cumulative_arc_length(xyz)[-1])
    chord = float(np.linalg.norm(xyz[-1] - xyz[0]))
    return total / chord if chord > 1e-6 else 1.0


class FirstStreamArcDecoder(ChunkTermination):
    """Multistream execution, ported from the validation decode
    (eval/open_loop_sim.py ``arc_prefix_control_steps``).

    Keep the first ``execute_percent`` % of the M waypoints and their timing rows,
    decode every stream on its own clock, and stop at the FIRST moving stream to
    run out of retained waypoints: either arm's translation or either arm's
    rotation. A stream that does not move in the prefix cannot end the chunk.
    ``last_stats["replan_steps"]`` is that boundary and the policy executes exactly
    that many rows. Past it the returned chunk holds the last pose so the
    (B, H, 14) output contract is unchanged.

    ``execute_percent`` and ``first_stream`` are dashboard controls; with
    ``first_stream`` 0 the whole token is decoded and ``replan_every`` decides.
    """

    def __init__(
        self,
        velocity_mode,
        velocity_layout,
        min_distance_unit,
        rotation_distance_unit,
        resampled_vector_length=100,
        dt=1 / 30,
        action_horizon=100,
        arc_chunking_mode="multistream",
        execute_percent=50,
        first_stream=1,
    ):
        if arc_chunking_mode != "multistream":
            raise ValueError("FirstStreamArcDecoder executes multistream tokens only")
        if rotation_distance_unit is None:
            raise ValueError("FirstStreamArcDecoder needs the hybrid rotation budget R")
        self._lib = training_cartesian
        self.M, self.dt = int(resampled_vector_length), float(dt)
        self.action_horizon = int(action_horizon)
        if self.M < 2 or self.action_horizon < 1 or not self.dt > 0:
            raise ValueError("Invalid ARC horizon, waypoint count or control period")
        self.velocity_mode = self._lib.validate_bimanual_velocity_mode(velocity_mode)
        if self.velocity_mode not in ("per_waypoint", "duration"):
            raise ValueError("First-stream execution needs per-interval timing")
        self.shape = self._lib.bimanual_arc_token_shape(
            self.M, self.velocity_mode, velocity_layout
        )
        # Tokens are restacked once in _decode, so the codec reads the stacked form.
        self.codec = self._lib.TokenizeBimanualArcLengthCartesian(
            min_distance_unit=min_distance_unit,
            rotation_distance_unit=rotation_distance_unit,
            resampled_vector_length=self.M,
            dt=self.dt,
            velocity_mode=self.velocity_mode,
            velocity_layout="stacked",
            arc_chunking_mode="multistream",
        )
        self.last_stats = None
        self.execute_percent = execute_percent
        self.first_stream = first_stream

    def __call__(self, native_tokens):
        if torch.is_tensor(native_tokens):
            native_tokens = native_tokens.detach().double().cpu().numpy()
        values = np.asarray(native_tokens, dtype=np.float64)
        if values.ndim == 2:
            values = values[None]
        # One replan boundary per call: the rollout executes a single plan.
        if values.shape != (1, *self.shape) or not np.isfinite(values).all():
            raise ValueError(
                f"Expected one finite native token (1,{self.shape[0]},{self.shape[1]}), got {values.shape}"
            )
        return self._decode(values[0])[None]

    def _stream_durations(self, waypoints, timing):
        """Seconds each of the four streams takes over the retained prefix. A
        moving interval with no usable timing costs more than the horizon."""
        h = self.action_horizon
        stalled = self.dt * (h + 1)
        durations = []
        for offset in (0, 7):
            for rotation in (False, True):
                columns = (
                    slice(offset + 3, offset + 6)
                    if rotation
                    else slice(offset, offset + 3)
                )
                travel = (
                    np.diff(self._lib.cumulative_rotation_length(waypoints[:, columns]))
                    if rotation
                    else np.linalg.norm(np.diff(waypoints[:, columns], axis=0), axis=1)
                )
                if self.velocity_mode == "duration":
                    rate = timing[:-1, offset + 3 if rotation else offset]
                else:
                    rate = np.linalg.norm(timing[:-1, columns], axis=1)
                invalid = (travel > 1e-12) & ((rate <= 1e-8) | ~np.isfinite(rate))
                clock = self.codec._arm_durations(
                    waypoints, timing, offset, h, rotation=rotation
                )
                durations.append(float(np.sum(np.where(invalid, stalled, clock))))
        return durations

    def _decode(self, row):
        h = self.action_horizon
        token = self._lib.stack_arc_token(row)
        if not self.first_stream:
            # Original method: the whole token, and replan_every decides the replan.
            self.last_stats = {
                "speed": 1.0,
                "hold_speed": None,
                "horizon": h,
                "valid_steps": None,
                "hold_fraction": None,
            }
            return self.codec.detokenize(token, action_horizon=h)
        count = self.M * self.execute_percent // 100
        waypoints, timing = token[:count], token[self.M : self.M + count]
        durations = self._stream_durations(waypoints, timing)
        moving = [d for d in durations if d > 1e-9]
        steps = max(1, int(np.ceil(min(moving) / self.dt - 1e-9))) if moving else 1
        steps = min(steps, h)
        decoded = self.codec.detokenize(
            np.concatenate((waypoints, timing)), action_horizon=steps
        )
        self.last_stats = {
            "speed": 1.0,
            "hold_speed": None,
            "horizon": h,
            "valid_steps": steps,
            "replan_steps": steps,
            "hold_fraction": None,
            "execute_waypoints": count,
            "stream_durations_s": durations,
        }
        return np.concatenate((decoded, np.repeat(decoded[-1:], h - steps, axis=0)))


class TimeChunkRetimer(ReplayTempo):
    """Replay tempo for a time-indexed policy's canonical (H, 14) Euler chunk.

    Row k is the pose at k * dt, so the chunk is a path on a uniform clock and the
    ARC decoder's warp applies unchanged: speed == hold_speed is the naive uniform
    speed-up (2x replays every other row), hold_speed < speed keeps holds at the
    demonstrated tempo. Past the last row the final pose is held, as an ARC path's
    endpoint is. Rotation is SLERPed in intrinsic ZYX, the convention pose_matrix
    reads. 1.0 / 1.0 returns the chunk untouched.
    """

    holds = True

    def __init__(
        self,
        action_horizon=100,
        dt=1 / 30,
        speed=1.0,
        hold_speed=1.0,
        hold_threshold=0.05,
    ):
        self.action_horizon, self.dt = int(action_horizon), float(dt)
        if self.action_horizon < 2 or not self.dt > 0:
            raise ValueError("Invalid chunk horizon or control period")
        self.shape = (self.action_horizon, 14)
        self._init_tempo(speed, hold_speed, hold_threshold)

    def __call__(self, actions):
        h = self.action_horizon
        if self.speed == 1.0 and self.hold_speed == 1.0:
            self.last_stats = {
                "speed": 1.0,
                "hold_speed": 1.0,
                "horizon": h,
                "valid_steps": h,
                "hold_fraction": None,
            }
            return actions  # the unmodified path, byte for byte
        if torch.is_tensor(actions):
            actions = actions.detach().double().cpu().numpy()
        values = np.asarray(actions, dtype=np.float64)
        if values.ndim == 2:
            values = values[None]
        if (
            values.ndim != 3
            or values.shape[1:] != self.shape
            or not np.isfinite(values).all()
        ):
            raise ValueError(
                f"Expected a finite time chunk (B,{h},14), got {values.shape}"
            )
        return np.stack([self._retime(row) for row in values])

    def _retime(self, row):
        h = self.action_horizon
        clock = self.dt * np.arange(h, dtype=np.float64)
        cums = [cumulative_arc_length(row[:, off : off + 3]) for off, *_ in ARM_LAYOUT]
        times, hold_fraction = self._warp([clock, clock], cums, clock[-1])
        valid = int(np.searchsorted(times, clock[-1], side="right"))
        self.last_stats = {
            "speed": self.speed,
            "hold_speed": self.hold_speed,
            "horizon": h,
            "valid_steps": min(valid, h),
            "hold_fraction": hold_fraction,
        }
        times = np.minimum(times, clock[-1])
        out = np.empty_like(row)
        for col in range(14):
            out[:, col] = np.interp(times, clock, row[:, col])
        for _xyz, ypr, _grip, _vsl in ARM_LAYOUT:
            spin = Slerp(clock, Rotation.from_euler("ZYX", row[:, ypr : ypr + 3]))
            out[:, ypr : ypr + 3] = spin(times).as_euler("ZYX")
        return out


class BimanualIntervalArcDecoder:
    """Decode an explicitly declared interval layout using the training codec."""

    def __init__(
        self,
        velocity_mode,
        min_distance_unit,
        resampled_vector_length,
        dt,
        action_horizon,
        velocity_layout="stacked",
        rotation_distance_unit=None,
        arc_chunking_mode=None,
    ):
        if velocity_mode not in {"per_waypoint", "duration"}:
            raise ValueError(
                "Deployment requires per-waypoint timing, not chunk-mean timing"
            )
        self.action_horizon = int(action_horizon)
        self.M = int(resampled_vector_length)
        if self.action_horizon < 1 or self.M < 2:
            raise ValueError("Invalid ARC output horizon or waypoint count")
        # Preserve the existing main contract. Checkpoint-era extensions above
        # use explicitly pinned compatibility codecs, not this training codec.
        self.codec = TrainingCartesianCodec(
            min_distance_unit=min_distance_unit,
            resampled_vector_length=self.M,
            dt=dt,
            velocity_mode=velocity_mode,
            velocity_layout=velocity_layout,
            rotation_distance_unit=rotation_distance_unit,
            arc_chunking_mode=arc_chunking_mode,
        )

        self.shape = bimanual_arc_token_shape(self.M, velocity_mode, velocity_layout)

    def __call__(self, native_tokens):
        if torch.is_tensor(native_tokens):
            native_tokens = native_tokens.detach().cpu().numpy()
        values = np.asarray(native_tokens)
        if (
            values.ndim != 3
            or values.shape[1:] != self.shape
            or not np.isfinite(values).all()
        ):
            raise ValueError(
                f"Expected finite (B,{self.shape[0]},{self.shape[1]}) interval ARC tokens"
            )
        return np.stack(
            [
                self.codec.detokenize(row, action_horizon=self.action_horizon)
                for row in values
            ]
        )


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Decode saved native ARC tokens; no robot I/O"
    )
    parser.add_argument("tokens", help="Input .npy file, already unnormalized")
    parser.add_argument("output", help="New .npy file for (B,H,14) canonical poses")
    parser.add_argument(
        "--arc-token-layout", choices=ARC_TOKEN_LAYOUTS, default="e1_dur"
    )
    parser.add_argument("--arc-min-distance-unit", type=float, default=0.4)
    parser.add_argument("--arc-resampled-vector-length", type=int, default=100)
    parser.add_argument("--arc-dt", type=float, default=1 / 30)
    parser.add_argument("--arc-rollout-horizon", type=int, default=100)
    parser.add_argument("--arc-rotation-distance-unit", type=float, default=None)
    parser.add_argument(
        "--arc-duration-timing", choices=("delta", "absolute"), default="delta"
    )
    parser.add_argument(
        "--arc-speed-encoding", choices=("interval", "point"), default="point"
    )
    parser.add_argument(
        "--arc-first-stream",
        type=int,
        choices=(0, 1),
        default=0,
        help="1 caps the token and ends the chunk at the first moving stream",
    )
    parser.add_argument("--arc-execute-percent", type=int, default=None)
    parser.add_argument(
        "--arc-speed",
        type=float,
        default=1.0,
        help="tempo multiplier for moving phases (1 = demonstrated)",
    )
    parser.add_argument(
        "--arc-hold-speed",
        type=float,
        default=None,
        help="tempo multiplier for holds; default follows --arc-speed",
    )
    parser.add_argument(
        "--arc-hold-threshold",
        type=float,
        default=0.05,
        help="path speed in m/s under which both arms count as holding",
    )
    args = parser.parse_args()
    decoder = BimanualArcDecoder(
        args.arc_token_layout,
        args.arc_min_distance_unit,
        args.arc_resampled_vector_length,
        args.arc_dt,
        args.arc_rollout_horizon,
        rotation_distance_unit=args.arc_rotation_distance_unit,
        hold_threshold=args.arc_hold_threshold,
        execute_percent=args.arc_execute_percent,
        first_stream=args.arc_first_stream,
        duration_timing=args.arc_duration_timing,
        speed_encoding=args.arc_speed_encoding,
    )
    decoder.set_speed(args.arc_speed, args.arc_hold_speed)
    result = decoder(np.load(args.tokens, allow_pickle=False))
    with open(args.output, "xb") as output:
        np.save(output, result, allow_pickle=False)


if __name__ == "__main__":
    main()
