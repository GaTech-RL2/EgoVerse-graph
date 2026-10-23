"""Cartesian ARC future targets with explicit current-pose arrival clocks."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from egomimic.rldb.zarr.arc_length_tokenizer import (
    _first_crossing_brackets,
    _slerp_segments_ypr,
    cumulative_arc_length,
    cumulative_rotation_length,
)

CLOCK_MODES = ("velocity",)
STREAM_MODES = ("translation_rotation", "translation", "rotation")
STREAM_NAMES = {
    "translation_rotation": ("translation", "rotation"),
    "translation": ("translation",),
    "rotation": ("rotation",),
}
STREAM_DIMS = {
    "translation_rotation": 18,
    "translation": 16,
    "rotation": 16,
}
CLOCK_COLUMNS = (0, 7, 3, 10)
SINGLE_CLOCK_COLUMNS = (0, 7)
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
    resampled_vector_length, clock_mode="velocity", streams="translation_rotation"
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
        streams = {18: "translation_rotation"}.get(width)
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
    raise ValueError(f"Unknown ARC stream: {stream!r}")


def clock_token_seconds(token, clock_mode, streams=None, *, initial_action):
    """Validate incoming speed columns and the separate measured current pose."""
    validate_clock_mode(clock_mode)
    value = np.array(token, dtype=np.float64)
    if value.ndim < 2 or value.shape[-1] not in (16, 18):
        raise ValueError("Expected an ARC token with 16 or 18 columns")
    if not np.isfinite(value).all():
        raise ValueError("ARC tokens must be finite")
    _token_streams(value.shape[-1], streams)
    _initial_actions(initial_action, value.shape[:-2])
    return value


def stack_arc_token(token, streams=None):
    """Internal geometry/timing view, not a selectable serialized layout."""
    value = np.asarray(token, dtype=np.float64)
    if value.ndim < 2 or value.shape[-1] not in (16, 18, 20):
        raise ValueError("Only the explicit 16D/18D ARC contracts are supported")
    streams = _token_streams(value.shape[-1], streams)
    cols = {
        "translation_rotation": CLOCK_COLUMNS,
        "translation": SINGLE_CLOCK_COLUMNS,
        "rotation": (3, 10),
    }[streams]
    timing = np.zeros(value.shape[:-1] + (14,))
    timing[..., cols] = value[..., 14:]
    return np.concatenate((value[..., :14], timing), axis=-2)


class TokenizeBimanualArcLengthCartesian:
    """M future targets with incoming per-waypoint speeds.

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
        resampled_vector_length=100,
        dt=1 / 30,
        zero_dist_epsilon=1e-6,
        preserve_action_key=None,
        preserve_action_rows=None,
        clock_mode="velocity",
        streams="translation_rotation",
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
        self.rotation_distance_unit = float(rotation_distance_unit)
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
            grip = self._linear(raw[:, off + 6 : off + 7], *primary)
            for name, supports in (
                ("translation", translation),
                ("rotation", rotation),
            ):
                if name in self._stream_names:
                    times[name, arm] = (supports[0] + supports[1] + 1) * dt
            arms.append(np.concatenate((xyz, ypr, grip), axis=1))
        return np.concatenate(arms, axis=1), times

    def _distances(self, waypoints, off, stream, initial_action):
        return stream_distances(
            waypoints, off, stream, self.streams, initial_action=initial_action
        )

    def _native(self, token, initial_action):
        value = np.asarray(token, dtype=np.float64)
        if value.ndim != 2 or value.shape[1] != self.shape[1] or len(value) < 1:
            raise ValueError(f"Expected (M>=1, {self.shape[1]}) ARC token")
        if not np.isfinite(value).all():
            raise ValueError("ARC tokens must be finite")
        return clock_token_seconds(
            value, self.clock_mode, self.streams, initial_action=initial_action
        )

    def _intervals(self, distance, stored):
        moving, usable = distance > _NUMERICAL_EPS, np.abs(stored) > 1e-8
        intervals = np.divide(
            distance,
            np.abs(stored),
            out=np.zeros_like(distance),
            where=moving & usable,
        )
        intervals[moving & ~usable] = np.inf
        return intervals

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

    def detokenize(self, arc_actions, action_horizon, *, initial_action):
        value = self._native(arc_actions, initial_action)
        h = int(action_horizon)
        if h < 1:
            raise ValueError("action_horizon must be positive")
        anchor = _initial_actions(initial_action)
        geometry = np.concatenate((anchor[None], value[:, :14]), axis=0)
        arms = []
        for arm, off in enumerate((0, 7)):
            indices = {}
            for index, name in enumerate(self._stream_names):
                durations = self._intervals(
                    self._distances(value, off, name, anchor),
                    value[:, 14 + index * 2 + arm],
                )
                lower, upper, alpha, _ = self._clock_indices(durations, h)
                indices[name] = (lower, upper, alpha)
            primary = indices.get("translation", indices.get("rotation"))
            lo, hi, a = primary
            xyz = geometry[lo, off : off + 3] + a[:, None] * (
                geometry[hi, off : off + 3] - geometry[lo, off : off + 3]
            )
            rlo, _, ra = indices.get("rotation", primary)
            ypr = _slerp_segments_ypr(geometry[:, off + 3 : off + 6], rlo, ra)
            grip = geometry[:, off + 6 : off + 7]
            grip = grip[lo] + a[:, None] * (grip[hi] - grip[lo])
            arms.append(np.concatenate((xyz, ypr, grip), axis=1))
        return np.concatenate(arms, axis=1)
