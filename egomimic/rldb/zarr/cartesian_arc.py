"""Cartesian ARC future targets with explicit current-pose arrival clocks."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from scipy.optimize import isotonic_regression

from egomimic.rldb.zarr.arc_length_tokenizer import (
    _first_crossing_brackets,
    _slerp_segments_ypr,
    cumulative_arc_length,
    cumulative_rotation_length,
)

CLOCK_MODES = ("velocity", "duration")
STREAM_MODES = ("full_separation", "translation_rotation", "translation", "rotation")
STREAM_NAMES = {
    "full_separation": ("translation", "rotation", "gripper"),
    "translation_rotation": ("translation", "rotation"),
    "translation": ("translation",),
    "rotation": ("rotation",),
}
STREAM_DIMS = {
    "full_separation": 20,
    "translation_rotation": 18,
    "translation": 16,
    "rotation": 16,
}
SIX_CLOCK_COLUMNS = (0, 7, 3, 10, 6, 13)
CLOCK_COLUMNS = (0, 7, 3, 10)
SINGLE_CLOCK_COLUMNS = (0, 7)
# Only absolute translation clocks have this speed bound, including anchor->row 0.
MAX_SPEED = 5.0
_NUMERICAL_EPS = 1e-12


def validate_clock_mode(mode):
    if mode not in CLOCK_MODES:
        raise ValueError(f"clock_mode must be one of {CLOCK_MODES}, got {mode!r}")
    return mode


def validate_streams(streams):
    if streams not in STREAM_MODES:
        raise ValueError(f"streams must be one of {STREAM_MODES}, got {streams!r}")
    return streams


def bimanual_arc_token_shape(
    resampled_vector_length, clock_mode="velocity", streams="full_separation"
):
    validate_clock_mode(clock_mode)
    validate_streams(streams)
    m = int(resampled_vector_length)
    if m < 2:
        raise ValueError("resampled_vector_length must be at least two")
    return m, STREAM_DIMS[streams]


def _initial_actions(initial_action, leading_shape=()):
    value = np.asarray(initial_action, dtype=np.float64)
    if value.ndim < 1 or value.shape[-1] != 14 or not np.isfinite(value).all():
        raise ValueError("initial_action must contain finite 14D current poses")
    try:
        return np.broadcast_to(value, tuple(leading_shape) + (14,))
    except ValueError as exc:
        raise ValueError(
            "initial_action batch dimensions must match the ARC token"
        ) from exc


def _token_streams(width, streams):
    if streams is None:
        streams = {20: "full_separation", 18: "translation_rotation"}.get(width)
        if streams is None:
            raise ValueError(
                "16D ARC tokens require explicit translation or rotation streams"
            )
    if STREAM_DIMS[validate_streams(streams)] != width:
        raise ValueError(f"{streams} tokens have {STREAM_DIMS[streams]} columns")
    return streams


def stream_distances(waypoints, off, stream, streams, *, initial_action=None):
    """Distance between emitted supports, optionally including current->row 0."""
    validate_streams(streams)
    values = np.asarray(waypoints, dtype=np.float64)
    if initial_action is not None:
        anchor = _initial_actions(initial_action)
        values = np.concatenate((anchor[None], values[:, :14]), axis=0)
    if stream == "translation":
        return np.linalg.norm(np.diff(values[:, off : off + 3], axis=0), axis=1)
    if stream == "rotation":
        return np.diff(cumulative_rotation_length(values[:, off + 3 : off + 6]))
    if stream == "gripper":
        return np.abs(np.diff(values[:, off + 6]))
    raise ValueError(f"Unknown ARC stream: {stream!r}")


def absolute_times_to_seconds(times, distances, max_speed=None):
    """Absolute arrival seconds -> incoming intervals, including anchor->row 0.

    Project absolute times onto a non-decreasing clock. A translation clock is
    projected above its fastest possible clock, including the distance from the
    measured current pose at t=0. There is no synthetic initial hold interval.
    """
    t = np.asarray(times, dtype=np.float64)
    distance = np.asarray(distances, dtype=np.float64)
    if (
        t.ndim != 1
        or not len(t)
        or t.shape != distance.shape
        or not np.isfinite(t).all()
        or not np.isfinite(distance).all()
        or np.any(distance < 0)
    ):
        raise ValueError(
            "Absolute times require one finite arrival and distance per waypoint"
        )
    if max_speed is not None and (not np.isfinite(max_speed) or max_speed <= 0):
        raise ValueError("max_speed must be positive and finite")
    fastest = np.cumsum(distance) / max_speed if max_speed else np.zeros_like(t)
    slack = np.maximum(isotonic_regression(t - fastest).x, 0.0)
    clock = fastest + slack
    return np.diff(np.r_[0.0, clock])


def clock_token_seconds(token, clock_mode, streams=None, *, initial_action):
    """Convert absolute duration columns once to incoming interval seconds.

    Velocity columns remain rates in every row, including row zero. Both modes
    require an explicit current pose; token width never selects a 16D clock.
    """
    validate_clock_mode(clock_mode)
    value = np.array(token, dtype=np.float64)
    if value.ndim < 2 or value.shape[-1] not in (16, 18, 20):
        raise ValueError("Expected an ARC token with 16, 18 or 20 columns")
    if not np.isfinite(value).all():
        raise ValueError("ARC tokens must be finite")
    streams = _token_streams(value.shape[-1], streams)
    anchors = _initial_actions(initial_action, value.shape[:-2])
    if clock_mode == "duration":
        rows = value.reshape(-1, *value.shape[-2:])
        for row, anchor in zip(rows, anchors.reshape(-1, 14)):
            for index, name in enumerate(STREAM_NAMES[streams]):
                for arm, off in enumerate((0, 7)):
                    col = 14 + 2 * index + arm
                    row[:, col] = absolute_times_to_seconds(
                        row[:, col],
                        stream_distances(
                            row, off, name, streams, initial_action=anchor
                        ),
                        MAX_SPEED if name == "translation" else None,
                    )
        value = rows.reshape(value.shape)
    return value


def stack_arc_token(token, streams=None):
    """Internal geometry/timing view, not a selectable serialized layout."""
    value = np.asarray(token, dtype=np.float64)
    if value.ndim < 2 or value.shape[-1] not in (16, 18, 20):
        raise ValueError("Only the explicit 16D/18D/20D ARC contracts are supported")
    streams = _token_streams(value.shape[-1], streams)
    cols = {
        "full_separation": SIX_CLOCK_COLUMNS,
        "translation_rotation": CLOCK_COLUMNS,
        "translation": SINGLE_CLOCK_COLUMNS,
        "rotation": (3, 10),
    }[streams]
    timing = np.zeros(value.shape[:-1] + (14,))
    timing[..., cols] = value[..., 14:]
    return np.concatenate((value[..., :14], timing), axis=-2)


class TokenizeBimanualArcLengthCartesian:
    """M future targets with incoming rates or absolute arrival seconds.

    The measured current pose is supplied separately as initial_action at t=0.
    Raw source row i is a future action at (i+1)*dt; it must not include the
    current observation row. Row zero has the same clock meaning as every other
    row: current->target 0, then target i-1->target i.

    Translation-only and rotation-only streams use that stream's supports and
    clock for all seven pose/gripper components of the arm. No legacy layout,
    initial-delay row, projection mode or synthetic hold window is inferred.
    """

    def __init__(
        self,
        action_key="actions_cartesian",
        output_action_key="actions_cartesian",
        min_distance_unit=0.60,
        rotation_distance_unit=1.7453292519943295,
        gripper_distance_unit=None,
        gripper_stream_mode="events",
        resampled_vector_length=100,
        dt=1 / 30,
        zero_dist_epsilon=1e-6,
        preserve_action_key=None,
        preserve_action_rows=None,
        clock_mode="velocity",
        streams="full_separation",
        initial_action_key="arc.current_action",
    ):
        self.clock_mode = validate_clock_mode(clock_mode)
        self.streams = validate_streams(streams)
        self.action_key, self.output_action_key = action_key, output_action_key
        if not isinstance(initial_action_key, str) or not initial_action_key:
            raise ValueError("initial_action_key must name the measured current pose")
        self.initial_action_key = initial_action_key
        self.preserve_action_key, self.preserve_action_rows = (
            preserve_action_key,
            preserve_action_rows,
        )
        self.shape = bimanual_arc_token_shape(
            resampled_vector_length, clock_mode, streams
        )
        for name, value in (
            ("D", min_distance_unit),
            ("R", rotation_distance_unit),
            ("dt", dt),
            ("zero_dist_epsilon", zero_dist_epsilon),
        ):
            if value is None or not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if preserve_action_rows is not None and int(preserve_action_rows) < 1:
            raise ValueError("preserve_action_rows must be positive")
        if gripper_stream_mode not in ("events", "distance"):
            raise ValueError("gripper_stream_mode must be events or distance")
        if streams == "full_separation" and gripper_stream_mode == "distance":
            if (
                gripper_distance_unit is None
                or not np.isfinite(gripper_distance_unit)
                or gripper_distance_unit <= 0
            ):
                raise ValueError(
                    "Separate distance gripper streams require positive gripper_distance_unit"
                )
        self.rotation_distance_unit = float(rotation_distance_unit)
        self.gripper_distance_unit = gripper_distance_unit
        self.gripper_stream_mode = gripper_stream_mode
        self.zero_dist_epsilon = float(zero_dist_epsilon)
        self.tokenizer = SimpleNamespace(
            M=self.shape[0],
            config=SimpleNamespace(
                dt=float(dt),
                min_distance_unit=float(min_distance_unit),
                zero_dist_epsilon=self.zero_dist_epsilon,
            ),
        )

    @property
    def M(self):
        return self.shape[0]

    @property
    def _stream_names(self):
        return STREAM_NAMES[self.streams]

    def transform(self, batch):
        raw = np.asarray(batch[self.action_key], dtype=np.float64)
        if (
            raw.ndim != 2
            or raw.shape[1] != 14
            or len(raw) < 2
            or not np.isfinite(raw).all()
        ):
            raise ValueError("ARC future source must be finite (T>=2, 14)")
        if self.initial_action_key not in batch:
            raise ValueError(
                f"ARC requires the measured current pose in {self.initial_action_key!r}"
            )
        anchor = _initial_actions(batch[self.initial_action_key])
        if self.preserve_action_key is not None:
            preserved = raw.copy()
            if self.preserve_action_rows is not None:
                n = int(self.preserve_action_rows)
                preserved = preserved[:n]
                if len(preserved) < n:
                    preserved = np.concatenate(
                        (
                            preserved,
                            np.repeat(preserved[-1:], n - len(preserved), axis=0),
                        )
                    )
            batch[self.preserve_action_key] = preserved
        waypoints, arrival_times = self._future_supports(raw)
        clocks = []
        for index, name in enumerate(self._stream_names):
            for arm, off in enumerate((0, 7)):
                times = arrival_times[name, arm]
                if self.clock_mode == "duration":
                    clock = times
                else:
                    intervals = np.diff(np.r_[0.0, times])
                    distance = self._distances(waypoints, off, name, anchor)
                    clock = np.divide(
                        distance,
                        intervals,
                        out=np.zeros_like(distance),
                        where=intervals > _NUMERICAL_EPS,
                    )
                clocks.append(clock)
        batch[self.output_action_key] = np.concatenate(
            (waypoints, np.stack(clocks, axis=1)), axis=1
        )
        return batch

    def _resample_coordinates(self, cumulative, budget):
        """Return source brackets; zero-length geometry keeps the actual span."""
        if cumulative[-1] <= _NUMERICAL_EPS:
            positions = np.linspace(0.0, len(cumulative) - 1, self.M)
            lower = np.minimum(positions.astype(np.int64), len(cumulative) - 2)
            return lower, positions - lower
        targets = np.linspace(0.0, min(float(cumulative[-1]), budget), self.M)
        return _first_crossing_brackets(cumulative, targets)

    @staticmethod
    def _linear(values, lower, alpha):
        return values[lower] + alpha[:, None] * (values[lower + 1] - values[lower])

    def _future_supports(self, raw):
        """Resample future geometry; every source row i arrives at (i+1)*dt."""
        arms, times = [], {}
        dt = self.tokenizer.config.dt
        for arm, off in enumerate((0, 7)):
            translation = self._resample_coordinates(
                cumulative_arc_length(raw[:, off : off + 3]),
                self.tokenizer.config.min_distance_unit,
            )
            rotation = self._resample_coordinates(
                cumulative_rotation_length(raw[:, off + 3 : off + 6]),
                self.rotation_distance_unit,
            )
            primary = rotation if self.streams == "rotation" else translation
            xyz = self._linear(raw[:, off : off + 3], *primary)
            rotation_supports = (
                rotation if "rotation" in self._stream_names else primary
            )
            ypr = _slerp_segments_ypr(raw[:, off + 3 : off + 6], *rotation_supports)
            if self.streams == "full_separation":
                grip_values = raw[:, off + 6 : off + 7]
                if self.gripper_stream_mode == "events":
                    frames = self._gripper_event_frames(grip_values[:, 0])
                    grip = grip_values[frames]
                    times["gripper", arm] = (frames + 1) * dt
                else:
                    cumulative = np.r_[
                        0.0, np.cumsum(np.abs(np.diff(grip_values[:, 0])))
                    ]
                    supports = self._resample_coordinates(
                        cumulative, self.gripper_distance_unit
                    )
                    grip = self._linear(grip_values, *supports)
                    times["gripper", arm] = (supports[0] + supports[1] + 1) * dt
            else:
                grip = self._linear(raw[:, off + 6 : off + 7], *primary)
            for name, supports in (
                ("translation", translation),
                ("rotation", rotation),
            ):
                if name in self._stream_names:
                    times[name, arm] = (supports[0] + supports[1] + 1) * dt
            arms.append(np.concatenate((xyz, ypr, grip), axis=1))
        return np.concatenate(arms, axis=1), times

    def _gripper_event_frames(self, values):
        """Initial future command, command changes, source endpoint and padding."""
        changes = np.flatnonzero(np.abs(np.diff(values)) > 1e-9) + 1
        required = np.unique(np.r_[0, changes, len(values) - 1]).astype(np.int64)
        if len(required) > self.M:
            required = required[
                np.rint(np.linspace(0, len(required) - 1, self.M)).astype(int)
            ]
        return np.r_[required, np.repeat(required[-1], self.M - len(required))]

    def _distances(self, waypoints, off, stream, initial_action):
        return stream_distances(
            waypoints, off, stream, self.streams, initial_action=initial_action
        )

    def _native(self, token, initial_action, clocks_in_seconds=False):
        value = np.asarray(token, dtype=np.float64)
        if value.ndim != 2 or value.shape[1] != self.shape[1] or len(value) < 1:
            raise ValueError(f"Expected (M>=1, {self.shape[1]}) ARC token")
        if not np.isfinite(value).all():
            raise ValueError("ARC tokens must be finite")
        _initial_actions(initial_action)
        if clocks_in_seconds and self.clock_mode != "duration":
            raise ValueError("clocks_in_seconds is only valid for duration clocks")
        return (
            value.copy()
            if clocks_in_seconds
            else clock_token_seconds(
                value, self.clock_mode, self.streams, initial_action=initial_action
            )
        )

    def _intervals(self, distance, stored):
        if self.clock_mode == "duration":
            return np.maximum(stored, 0)
        moving, usable = distance > _NUMERICAL_EPS, np.abs(stored) > 1e-8
        intervals = np.divide(
            distance,
            np.abs(stored),
            out=np.zeros_like(distance),
            where=moving & usable,
        )
        intervals[moving & ~usable] = np.inf
        return intervals

    def control_steps(
        self, token, *, initial_action, max_steps=None, clocks_in_seconds=False
    ):
        if max_steps is not None and (int(max_steps) != max_steps or max_steps < 1):
            raise ValueError("max_steps must be a positive integer")
        value = self._native(token, initial_action, clocks_in_seconds)
        moving_ends, static_ends = [], []
        for index, name in enumerate(self._stream_names):
            for arm, off in enumerate((0, 7)):
                distance = self._distances(value, off, name, initial_action)
                end = self._intervals(distance, value[:, 14 + index * 2 + arm]).sum()
                if np.any(distance > _NUMERICAL_EPS):
                    moving_ends.append(end)
                elif self.clock_mode == "duration":
                    static_ends.append(end)
        if moving_ends:
            duration = min(moving_ends)
        elif static_ends:
            duration = min(static_ends)
        else:
            if max_steps is None:
                raise ValueError(
                    "A fully static velocity token has no duration; supply max_steps"
                )
            return int(max_steps)
        if not np.isfinite(duration):
            if max_steps is None:
                raise ValueError(
                    "ARC token timing does not define a finite replan boundary"
                )
            return int(max_steps)
        steps = max(1, int(np.ceil(duration / self.tokenizer.config.dt - 1e-9)))
        return min(steps, int(max_steps)) if max_steps is not None else steps

    def _clock_indices(self, durations, action_horizon):
        elapsed = np.r_[0.0, np.cumsum(durations)]
        query = self.tokenizer.config.dt * np.arange(1, action_horizon + 1)
        upper = np.clip(
            np.searchsorted(elapsed, query, side="right"), 1, len(elapsed) - 1
        )
        lower = upper - 1
        start, end = elapsed[lower], elapsed[upper]
        width = end - start
        alpha = np.divide(
            query - start,
            width,
            out=(query >= end).astype(np.float64),
            where=(width > 0) & np.isfinite(width),
        )
        return lower, upper, np.clip(alpha, 0.0, 1.0), elapsed

    def detokenize(
        self, arc_actions, action_horizon, *, initial_action, clocks_in_seconds=False
    ):
        value = self._native(arc_actions, initial_action, clocks_in_seconds)
        h = int(action_horizon)
        if h < 1:
            raise ValueError("action_horizon must be positive")
        anchor = _initial_actions(initial_action)
        geometry = np.concatenate((anchor[None], value[:, :14]), axis=0)
        query = np.arange(1, h + 1) * self.tokenizer.config.dt
        arms = []
        for arm, off in enumerate((0, 7)):
            indices, clocks = {}, {}
            for index, name in enumerate(self._stream_names):
                durations = self._intervals(
                    self._distances(value, off, name, anchor),
                    value[:, 14 + index * 2 + arm],
                )
                lower, upper, alpha, elapsed = self._clock_indices(durations, h)
                indices[name] = (lower, upper, alpha)
                clocks[name] = elapsed
            primary = indices.get("translation", indices.get("rotation"))
            lo, hi, a = primary
            xyz = geometry[lo, off : off + 3] + a[:, None] * (
                geometry[hi, off : off + 3] - geometry[lo, off : off + 3]
            )
            rlo, _, ra = indices.get("rotation", primary)
            ypr = _slerp_segments_ypr(geometry[:, off + 3 : off + 6], rlo, ra)
            grip = geometry[:, off + 6 : off + 7]
            if (
                self.streams == "full_separation"
                and self.gripper_stream_mode == "events"
            ):
                clock = clocks["gripper"]
                finite_end = clock[np.isfinite(clock)][-1]
                tolerance = 8 * np.finfo(np.float32).eps * max(1.0, finite_end)
                gi = np.searchsorted(clock, query + tolerance, side="right") - 1
                grip = grip[np.clip(gi, 0, len(grip) - 1)]
            else:
                glo, ghi, ga = indices.get("gripper", primary)
                grip = grip[glo] + ga[:, None] * (grip[ghi] - grip[glo])
            arms.append(np.concatenate((xyz, ypr, grip), axis=1))
        return np.concatenate(arms, axis=1)
