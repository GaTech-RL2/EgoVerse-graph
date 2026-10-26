"""Native measured-anchor ARC decoding; no hardware I/O or historical inference."""

import numpy as np
import torch
from scipy.spatial.transform import Rotation, Slerp

from egomimic.rldb.zarr.arc_length_tokenizer import cumulative_arc_length
from egomimic.rldb.zarr.cartesian_arc import TokenizeBimanualArcLengthCartesian
from egomimic.robot.arc_speed import validate_arc_speed

ARM_LAYOUT = ((0, 3, 6, None), (7, 10, 13, None))
RATE_RAMP_S = 0.2
_WARP_OVERSAMPLE = 4


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
    """speed / hold_speed state and the clock warp shared by the time-chunk
    retimer and the first-stream ARC decoder (rollout-only, never trained).
    Subclasses set dt, action_horizon,
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

    def _warp(self, clocks, cums, end=np.inf, start=0):
        """Token-clock time to read each of the H control ticks at, and the share of
        the decoded span that was a hold. rate = d(clock) / d(wall). ``end`` is the
        clock time at which the token's path runs out: nothing after it is a hold,
        it is just the endpoint being repeated, so it is left out of the share.
        ``start`` is the first tick (1 for next-frame ARC rows)."""
        wall = self.dt * np.arange(start, start + self.action_horizon, dtype=np.float64)
        top = max(self.speed, self.hold_speed)
        n = (
            int(np.ceil((start + self.action_horizon) * top * _WARP_OVERSAMPLE))
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


class TimeChunkRetimer(ReplayTempo):
    """Replay tempo for a time-indexed policy's canonical (H, 14) Euler chunk.

    Row k is the pose at k * dt, so the chunk is a path on a uniform clock.
    speed == hold_speed is the naive uniform
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
    """Decode the whole token with its explicitly configured training codec."""

    context_keys = ("initial_action",)

    def __init__(
        self,
        clock_mode="velocity",
        streams="full_separation",
        min_distance_unit=0.4,
        rotation_distance_unit=1.7453292519943295,
        resampled_vector_length=100,
        dt=1 / 30,
        action_horizon=100,
        gripper_distance_unit=None,
        gripper_stream_mode="events",
    ):
        self.action_horizon = int(action_horizon)
        if self.action_horizon < 1:
            raise ValueError("action_horizon must be positive")
        self.M, self.dt = int(resampled_vector_length), float(dt)
        self.codec = TokenizeBimanualArcLengthCartesian(
            clock_mode=clock_mode,
            streams=streams,
            min_distance_unit=min_distance_unit,
            rotation_distance_unit=rotation_distance_unit,
            resampled_vector_length=self.M,
            dt=dt,
            gripper_distance_unit=gripper_distance_unit,
            gripper_stream_mode=gripper_stream_mode,
        )
        self.shape = self.codec.shape
        self.last_stats = None

    def __call__(self, native_tokens, *, initial_action):
        if torch.is_tensor(native_tokens):
            native_tokens = native_tokens.detach().double().cpu().numpy()
        values = np.asarray(native_tokens, dtype=np.float64)
        if values.ndim == 2:
            values = values[None]
        if (
            values.ndim != 3
            or values.shape[1:] != self.shape
            or not len(values)
            or not np.isfinite(values).all()
        ):
            raise ValueError(
                f"Expected finite nonempty (B,{self.shape[0]},{self.shape[1]}) ARC tokens"
            )
        if torch.is_tensor(initial_action):
            initial_action = initial_action.detach().double().cpu().numpy()
        anchors = np.asarray(initial_action, dtype=np.float64)
        if anchors.shape == (14,) and len(values) == 1:
            anchors = anchors[None]
        if anchors.shape != (len(values), 14) or not np.isfinite(anchors).all():
            raise ValueError(
                "ARC decoding requires one finite native (14,) measured anchor per token"
            )
        return np.stack(
            [self._decode(row, anchor) for row, anchor in zip(values, anchors)]
        )

    def _decode(self, row, initial_action):
        return self.codec.detokenize(
            row, self.action_horizon, initial_action=initial_action
        )


class FirstStreamArcDecoder(BimanualIntervalArcDecoder, ChunkTermination, ReplayTempo):
    """Keep a waypoint prefix; replan when the first moving stream ends.

    speed / hold_speed replay the token's translation clock faster or slower
    (moving vs held, as TimeChunkRetimer); 1.0 / 1.0 is the plain decode."""

    holds = True

    def __init__(
        self,
        *,
        execute_percent=30,
        first_stream=1,
        speed=1.0,
        hold_speed=1.0,
        hold_threshold=0.05,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.execute_percent, self.first_stream = execute_percent, first_stream
        self._init_tempo(speed, hold_speed, hold_threshold)

    def _decode(self, row, initial_action):
        token = (
            row[: self.M * self.execute_percent // 100] if self.first_stream else row
        )
        h, times, hold_fraction = self.action_horizon, None, None
        if self.speed != 1.0 or self.hold_speed != 1.0:
            clocks, cums = [], []
            for clock, cum in self.codec.stream_clocks(
                token, initial_action=initial_action
            ):
                finite = np.isfinite(clock)  # a stalled velocity stream ends there
                clocks.append(clock[finite])
                cums.append(cum[finite])
            end = max(clock[-1] for clock in clocks)
            times, hold_fraction = self._warp(clocks, cums, end, start=1)
        if not self.first_stream:
            steps = h
        elif times is None:
            steps = self.codec.control_steps(
                token, initial_action=initial_action, max_steps=h
            )
        else:
            end = self.codec.control_duration(token, initial_action=initial_action)
            steps = (
                h
                if end is None
                else min(h, int(np.searchsorted(times, end - 1e-9)) + 1)
            )
        decoded = self.codec.detokenize(
            token,
            steps,
            initial_action=initial_action,
            times=None if times is None else times[:steps],
        )
        if steps < self.action_horizon:
            decoded = np.concatenate(
                (decoded, np.repeat(decoded[-1:], self.action_horizon - steps, axis=0))
            )
        self.last_stats = {
            "replan_steps": steps if self.first_stream else None,  # 0: replan_every
            "horizon": self.action_horizon,
            "execute_waypoints": len(token),
            "speed": self.speed,
            "hold_speed": self.hold_speed,
            "hold_fraction": hold_fraction,
        }
        return decoded
