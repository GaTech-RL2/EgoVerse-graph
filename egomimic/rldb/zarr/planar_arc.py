"""Small, explicit SE(2) arc tokenizer used by Planar PushShapes models."""

from __future__ import annotations

import math

import numpy as np
from scipy.interpolate import CubicSpline

PLANAR_ACTION_DIM = 5  # [x, y, cos(theta), sin(theta), grip]

# How a Planar arc token carries timing.
#
#   "mean"         M waypoints + ONE timing row holding the chunk's mean arc
#                  speed. Exact when the chunk is traversed at constant speed,
#                  and lossy otherwise: one scalar cannot express accelerating,
#                  dwelling, or stop-and-go motion.
#   "per_waypoint" M waypoints + M rate rows, one local SE(2) rate per interval,
#                  measured from the bracketing source frames. Recovers the
#                  elapsed-time parameterization of a non-uniform chunk.
#   "duration"     M waypoints + M duration rows, one elapsed-time (seconds)
#                  per interval. Same layout width as per_waypoint, but stores
#                  Δt directly instead of arc_distance / Δt.
#
# ``mean`` remains readable only for checkpoint-bound legacy artifacts. New
# codecs default to interval duration because a chunk-level mean cannot recover
# acceleration, dwell, or stop-and-go timing.
VELOCITY_MODES = ("mean", "per_waypoint", "duration")
_MEAN, _PER_WAYPOINT, _DURATION = VELOCITY_MODES
DEFAULT_VELOCITY_MODE = _DURATION


def validate_velocity_mode(velocity_mode: str) -> str:
    if velocity_mode not in VELOCITY_MODES:
        raise ValueError(
            f"velocity_mode must be one of {VELOCITY_MODES}, got {velocity_mode!r}"
        )
    return velocity_mode


def arc_token_rows(
    resampled_vector_length: int,
    velocity_mode: str = DEFAULT_VELOCITY_MODE,
) -> int:
    """Rows in one Planar arc token.

    Single source of truth: the tokenizer, both graph stages, the native
    decoder and the configs all size themselves from this, so a mode change
    cannot leave one of them expecting the other layout.
    """
    validate_velocity_mode(velocity_mode)
    num_waypoints = int(resampled_vector_length)
    if num_waypoints < 2:
        raise ValueError("resampled_vector_length must be at least two")
    return num_waypoints + 1 if velocity_mode == _MEAN else 2 * num_waypoints


def lambda_for_radius(radius: float) -> float:
    """Convert a physical rotation radius into the SE(2) metric weight."""
    radius = float(radius)
    if radius < 0:
        raise ValueError("radius must be non-negative")
    return 2.0 * math.sqrt(2.0) * radius


def rotation_step_metric_planar(theta: np.ndarray) -> np.ndarray:
    """Return the scaled chordal rotation distance between adjacent angles."""
    theta = np.unwrap(np.asarray(theta, dtype=np.float64))
    return math.sqrt(2.0) * np.sin(np.abs(np.diff(theta)) / 4.0)


def planar_step_distance(
    xy: np.ndarray,
    theta: np.ndarray | None = None,
    lambda_rot: float = 0.0,
) -> np.ndarray:
    """Return adjacent SE(2) distances: translation plus weighted rotation."""
    xy = np.asarray(xy, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError(f"xy must have shape (T, 2), got {xy.shape}")
    if len(xy) < 2:
        return np.zeros(0, dtype=np.float64)
    distance = np.linalg.norm(np.diff(xy, axis=0), axis=-1)
    if lambda_rot:
        if lambda_rot < 0 or theta is None or len(theta) != len(xy):
            raise ValueError("positive lambda_rot requires one theta per xy")
        distance += float(lambda_rot) * rotation_step_metric_planar(theta)
    return distance


def _bracket_segment(cumulative: np.ndarray, target: float) -> tuple[int, float]:
    if target <= cumulative[0]:
        return 0, 0.0
    if target >= cumulative[-1]:
        return len(cumulative) - 2, 1.0
    index = int(np.searchsorted(cumulative, target, side="left") - 1)
    index = max(0, min(index, len(cumulative) - 2))
    span = cumulative[index + 1] - cumulative[index]
    alpha = 0.0 if span <= 1e-12 else (target - cumulative[index]) / span
    return index, float(alpha)


def _interpolate(values: np.ndarray, cumulative: np.ndarray, target: float):
    if target <= cumulative[0]:
        return values[0].copy()
    if target >= cumulative[-1]:
        return values[-1].copy()
    index, alpha = _bracket_segment(cumulative, target)
    return (1.0 - alpha) * values[index] + alpha * values[index + 1]


def _unique_curve_support(
    xy: np.ndarray,
    cumulative: np.ndarray,
    end: float,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return strictly increasing support through the exact arc-window end."""
    inside = cumulative < end - epsilon
    parameter = np.concatenate((cumulative[inside], np.array([end])))
    points = np.concatenate(
        (xy[inside], _interpolate(xy, cumulative, end)[None]), axis=0
    )
    keep = np.concatenate((np.ones(1, dtype=bool), np.diff(parameter) > epsilon))
    return parameter[keep], points[keep]


def curvature_adaptive_curve_samples(
    xy: np.ndarray,
    cumulative: np.ndarray,
    end: float,
    num_waypoints: int,
    *,
    dense_samples: int = 257,
    curvature_floor: float | None = None,
    epsilon: float = 1e-9,
) -> tuple[np.ndarray, np.ndarray]:
    """Allocate supports using a natural cubic curvature estimate.

    The returned targets stay in the tokenizer's cumulative arc coordinate so
    per-segment durations can be recovered from the original source frames.
    XY uses the same source-polyline interpolation as uniform allocation:
    fitting a different geometric path would confound the allocation ablation.
    """
    xy = np.asarray(xy, dtype=np.float64)
    cumulative = np.asarray(cumulative, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) != len(cumulative):
        raise ValueError("xy and cumulative must have matching (T,2)/(T,) shapes")
    if num_waypoints < 2:
        raise ValueError("num_waypoints must be at least two")
    if dense_samples < max(17, num_waypoints):
        raise ValueError("dense_samples must be at least max(17, num_waypoints)")
    if curvature_floor is not None and (
        not math.isfinite(curvature_floor) or curvature_floor <= 0
    ):
        raise ValueError("curvature_floor must be null or finite and positive")
    if end <= epsilon:
        return np.repeat(xy[:1], num_waypoints, axis=0), np.zeros(num_waypoints)

    support_s, support_xy = _unique_curve_support(xy, cumulative, end, epsilon)
    if len(support_s) < 3:
        targets = np.linspace(0.0, end, num_waypoints)
        return (
            np.stack([_interpolate(xy, cumulative, point) for point in targets]),
            targets,
        )

    spline = CubicSpline(support_s, support_xy, axis=0, bc_type="natural")
    dense_s = np.linspace(0.0, end, dense_samples)
    first = spline(dense_s, 1)
    second = spline(dense_s, 2)
    curve_speed = np.linalg.norm(first, axis=-1)
    numerator = np.abs(first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0])
    speed_scale = max(float(np.max(curve_speed)), epsilon)
    regularized_speed = np.maximum(curve_speed, speed_scale * 1e-6)
    curvature = numerator / np.power(regularized_speed, 3)
    curvature = np.nan_to_num(curvature, nan=0.0, posinf=0.0, neginf=0.0)
    effective_floor = 1.0 / end if curvature_floor is None else curvature_floor
    density = np.power(effective_floor**2 + curvature**2, 1.0 / 5.0)

    delta_s = np.diff(dense_s)
    weighted_arc = density * curve_speed
    importance = np.concatenate(
        (
            np.zeros(1),
            np.cumsum(0.5 * (weighted_arc[:-1] + weighted_arc[1:]) * delta_s),
        )
    )
    if importance[-1] <= epsilon:
        targets = np.linspace(0.0, end, num_waypoints)
    else:
        targets = np.interp(
            np.linspace(0.0, importance[-1], num_waypoints),
            importance,
            dense_s,
        )
    targets[0], targets[-1] = 0.0, end
    return np.stack([_interpolate(xy, cumulative, point) for point in targets]), targets


class PadPlanarAction:
    """Widen native ``[x,y[,theta[,grip]]]`` actions to a common five-vector."""

    def __init__(self, keys: list[str] | None = None):
        self.keys = list(keys or ["actions"])

    def transform(self, batch: dict) -> dict:
        for key in self.keys:
            value = np.asarray(batch[key])
            if value.ndim != 2 or value.shape[-1] not in (2, 3, 4):
                raise ValueError(
                    f"PadPlanarAction expects (T, 2|3|4) for {key!r}, got {value.shape}"
                )
            work = value.astype(np.float64, copy=False)
            theta = work[:, 2] if work.shape[1] >= 3 else np.zeros(len(work))
            grip = work[:, 3] if work.shape[1] == 4 else np.zeros(len(work))
            output = np.column_stack((work[:, :2], np.cos(theta), np.sin(theta), grip))
            dtype = (
                value.dtype if np.issubdtype(value.dtype, np.floating) else np.float32
            )
            batch[key] = output.astype(dtype, copy=False)
        return batch


class TokenizePlanarArcLength:
    """Encode a future native action chunk as M curve supports plus timing.

    Duration timing is the default: one interval duration is stored per
    waypoint row. The first waypoint is copied from timestep zero rather than
    recovered via an arc lookup; this preserves stationary grip transitions.

    ``hybrid_rotation_unit`` optionally adds the hybrid Cartesian/angular
    window rule used by the arc sweep.  The regular SE(2) cumulative clock
    still supplies interpolation positions.  A separate, unweighted angular
    clock then limits the fraction of the available Cartesian window.  This
    is intentionally a dataset-bound Planar transform: generic PipelineAlgo
    stages do not need to know what an angle or an action represents.
    """

    def __init__(
        self,
        action_key: str = "actions",
        output_action_key: str = "actions",
        min_distance_unit: float = 200.0,
        resampled_vector_length: int = 100,
        dt: float = 1.0 / 30.0,
        rotation_radius: float = 0.0,
        hybrid_rotation_unit: float | None = None,
        velocity_mode: str = DEFAULT_VELOCITY_MODE,
        waypoint_sampling: str = "uniform",
        curvature_dense_samples: int = 257,
        curvature_floor: float | None = None,
        zero_dist_epsilon: float = 1e-9,
    ):
        if min_distance_unit <= 0 or dt <= 0:
            raise ValueError("min_distance_unit and dt must be positive")
        if resampled_vector_length < 2:
            raise ValueError("resampled_vector_length must be at least two")
        if rotation_radius < 0:
            raise ValueError("rotation_radius must be non-negative")
        if hybrid_rotation_unit is not None and (
            not math.isfinite(hybrid_rotation_unit) or hybrid_rotation_unit <= 0
        ):
            raise ValueError("hybrid_rotation_unit must be finite and positive")
        if waypoint_sampling not in {"uniform", "curvature"}:
            raise ValueError("waypoint_sampling must be 'uniform' or 'curvature'")
        if curvature_dense_samples < max(17, resampled_vector_length):
            raise ValueError(
                "curvature_dense_samples must be at least max(17, resampled_vector_length)"
            )
        if curvature_floor is not None and (
            not math.isfinite(curvature_floor) or curvature_floor <= 0
        ):
            raise ValueError("curvature_floor must be null or finite and positive")
        self.action_key = str(action_key)
        self.output_action_key = str(output_action_key)
        self.distance = float(min_distance_unit)
        self.num_waypoints = int(resampled_vector_length)
        self.dt = float(dt)
        self.rotation_radius = float(rotation_radius)
        self.velocity_mode = validate_velocity_mode(velocity_mode)
        self.hybrid_rotation_unit = (
            None if hybrid_rotation_unit is None else float(hybrid_rotation_unit)
        )
        self.waypoint_sampling = str(waypoint_sampling)
        self.curvature_dense_samples = int(curvature_dense_samples)
        self.curvature_floor = (
            None if curvature_floor is None else float(curvature_floor)
        )
        self.zero_dist_epsilon = float(zero_dist_epsilon)

    @staticmethod
    def _components(actions: np.ndarray):
        xy = actions[:, :2]
        theta = (
            np.unwrap(actions[:, 2])
            if actions.shape[1] >= 3
            else np.zeros(len(actions))
        )
        grip = actions[:, 3] if actions.shape[1] == 4 else np.zeros(len(actions))
        return xy, theta, grip

    def _window_end(self, cumulative: np.ndarray, theta: np.ndarray) -> float:
        """Return the cumulative-distance endpoint for this token window."""
        end = min(self.distance, float(cumulative[-1]))
        if self.hybrid_rotation_unit is None:
            return end

        rotation = np.concatenate(
            (
                np.zeros(1, dtype=np.float64),
                np.cumsum(rotation_step_metric_planar(theta)),
            )
        )
        translation_span = float(cumulative[-1])
        rotation_span = float(rotation[-1])
        if (
            translation_span > self.zero_dist_epsilon
            and rotation_span > self.zero_dist_epsilon
        ):
            rotation_fraction = min(1.0, self.hybrid_rotation_unit / rotation_span)
            end = min(end, translation_span * rotation_fraction)
        return end

    def _interval_times(
        self, cumulative: np.ndarray, targets: np.ndarray
    ) -> np.ndarray:
        """Elapsed time at each arc target, from the bracketing source frames.

        Recovered as ``(index + alpha) * dt`` rather than by dividing the
        window by a single duration. That preserves local speed changes and
        stationary frames before motion, both of which a chunk-level mean
        erases.
        """
        times = np.empty(self.num_waypoints, dtype=np.float64)
        times[0] = 0.0
        for index, target in enumerate(targets[1:], start=1):
            source_index, alpha = _bracket_segment(cumulative, float(target))
            times[index] = (source_index + alpha) * self.dt
        return times

    def _interval_rates(
        self, cumulative: np.ndarray, targets: np.ndarray
    ) -> np.ndarray:
        """Local SE(2) rate per waypoint interval, from the source frames.

        Returns one rate per waypoint; the final value repeats the last
        interval so the array lines up with the waypoint rows.
        """
        times = self._interval_times(cumulative, targets)
        delta_time = np.diff(times)
        delta_arc = np.diff(targets)
        interval_rate = np.divide(
            delta_arc,
            delta_time,
            out=np.zeros_like(delta_arc),
            where=delta_time > self.zero_dist_epsilon,
        )
        rates = np.zeros(self.num_waypoints, dtype=np.float64)
        rates[:-1] = interval_rate
        rates[-1] = interval_rate[-1]
        return rates

    def _interval_durations(
        self, cumulative: np.ndarray, targets: np.ndarray
    ) -> np.ndarray:
        """Elapsed seconds per waypoint interval, from the source frames.

        Returns one duration per waypoint; the final value repeats the last
        interval so the array lines up with the waypoint rows.
        """
        times = self._interval_times(cumulative, targets)
        delta_time = np.diff(times)
        durations = np.zeros(self.num_waypoints, dtype=np.float64)
        durations[:-1] = delta_time
        durations[-1] = delta_time[-1]
        return durations

    def _stationary_support_frames(
        self, cumulative: np.ndarray, targets: np.ndarray
    ) -> np.ndarray | None:
        """Reserve both ends of stationary intervals in the duration clock.

        An arc coordinate alone cannot distinguish arriving at a position
        from leaving it after a hold. Both allocation modes retain those time
        boundaries, then allocate the remaining supports by their respective
        arc-target density. Refuse a budget that cannot represent the holds
        rather than silently deleting elapsed time from a training target.
        """
        stopped = np.diff(cumulative) <= self.zero_dist_epsilon
        if not np.any(stopped):
            return None
        last_index, alpha = _bracket_segment(cumulative, float(targets[-1]))
        end_frame = last_index + alpha
        starts = np.flatnonzero(stopped & np.r_[True, ~stopped[:-1]])
        ends = np.flatnonzero(stopped & np.r_[~stopped[1:], True]) + 1
        boundaries = [0.0, end_frame]
        for start, end in zip(starts, ends, strict=True):
            if start < end_frame:
                boundaries.extend((float(start), min(float(end), end_frame)))
        boundaries = np.unique(boundaries)
        if len(boundaries) == 2:
            return None
        if len(boundaries) > self.num_waypoints:
            raise ValueError(
                f"duration ARC needs at least {len(boundaries)} waypoints to "
                f"preserve stationary intervals, configured {self.num_waypoints}"
            )

        frames = np.arange(len(cumulative), dtype=np.float64)
        boundary_arc = np.interp(boundaries, frames, cumulative)
        target_rank = np.linspace(0.0, 1.0, self.num_waypoints)
        boundary_rank = np.interp(boundary_arc, targets, target_rank)
        weights = np.diff(boundary_rank)
        extra = self.num_waypoints - len(boundaries)
        quotas = extra * weights / weights.sum()
        counts = np.ones(len(weights), dtype=int) + np.floor(quotas).astype(int)
        remaining = self.num_waypoints - 1 - counts.sum()
        order = np.argsort(-(quotas - np.floor(quotas)), kind="stable")
        counts[order[:remaining]] += 1

        support_frames = [0.0]
        for index, count in enumerate(counts):
            if boundary_arc[index + 1] - boundary_arc[index] <= self.zero_dist_epsilon:
                interior = np.linspace(
                    boundaries[index], boundaries[index + 1], count + 1
                )[1:-1]
                support_frames.extend(interior.tolist())
            else:
                ranks = np.linspace(
                    boundary_rank[index], boundary_rank[index + 1], count + 1
                )[1:-1]
                for point in np.interp(ranks, target_rank, targets):
                    source_index, fraction = _bracket_segment(cumulative, float(point))
                    support_frames.append(source_index + fraction)
            support_frames.append(boundaries[index + 1])
        return np.asarray(support_frames)

    def tokenize(self, actions: np.ndarray) -> np.ndarray:
        xy, theta, grip = self._components(actions)
        weight = lambda_for_radius(self.rotation_radius)
        steps = planar_step_distance(xy, theta, weight)
        cumulative = np.concatenate((np.zeros(1), np.cumsum(steps)))
        end = self._window_end(cumulative, theta)

        if end <= self.zero_dist_epsilon:
            xy_waypoints = np.repeat(xy[:1], self.num_waypoints, axis=0)
            theta_waypoints = np.repeat(theta[0], self.num_waypoints)
            grip_waypoints = np.repeat(grip[0], self.num_waypoints)
            speed = 0.0
            rates = np.zeros(self.num_waypoints, dtype=np.float64)
            durations = np.zeros(self.num_waypoints, dtype=np.float64)
            if self.velocity_mode == _DURATION:
                # No geometric distance does not imply no native action:
                # gripper opening and radius-zero rotation still evolve.
                # Both allocation arms use the same temporal fallback.
                frames = np.arange(len(actions), dtype=np.float64)
                support_frames = np.linspace(0.0, frames[-1], self.num_waypoints)
                theta_waypoints = np.interp(support_frames, frames, theta)
                grip_waypoints = np.interp(support_frames, frames, grip)
                durations[:-1] = np.diff(support_frames) * self.dt
                durations[-1] = durations[-2]
        else:
            if self.waypoint_sampling == "curvature":
                xy_waypoints, targets = curvature_adaptive_curve_samples(
                    xy,
                    cumulative,
                    end,
                    self.num_waypoints,
                    dense_samples=self.curvature_dense_samples,
                    curvature_floor=self.curvature_floor,
                    epsilon=self.zero_dist_epsilon,
                )
            else:
                targets = np.linspace(0.0, end, self.num_waypoints)
                xy_waypoints = np.stack(
                    [_interpolate(xy, cumulative, point) for point in targets]
                )
            theta_waypoints = np.array(
                [
                    _interpolate(theta[:, None], cumulative, point)[0]
                    for point in targets
                ]
            )
            grip_waypoints = np.array(
                [_interpolate(grip[:, None], cumulative, point)[0] for point in targets]
            )
            last_index = int(np.searchsorted(cumulative, end, side="left"))
            speed = end / (max(1, last_index) * self.dt)
            rates = self._interval_rates(cumulative, targets)
            durations = self._interval_durations(cumulative, targets)
            if self.velocity_mode == _DURATION:
                support_frames = self._stationary_support_frames(cumulative, targets)
                if support_frames is not None:
                    frames = np.arange(len(actions), dtype=np.float64)
                    xy_waypoints = np.column_stack(
                        [
                            np.interp(support_frames, frames, xy[:, axis])
                            for axis in (0, 1)
                        ]
                    )
                    theta_waypoints = np.interp(support_frames, frames, theta)
                    grip_waypoints = np.interp(support_frames, frames, grip)
                    durations[:-1] = np.diff(support_frames) * self.dt
                    durations[-1] = durations[-2]

        xy_waypoints[0] = xy[0]
        theta_waypoints[0] = theta[0]
        grip_waypoints[0] = grip[0]
        waypoints = np.column_stack(
            (
                xy_waypoints,
                np.cos(theta_waypoints),
                np.sin(theta_waypoints),
                grip_waypoints,
            )
        )
        if self.velocity_mode == _PER_WAYPOINT:
            # One rate row per waypoint. Column 0 carries the local SE(2) rate;
            # the rest stay zero and are reserved, so the block keeps the same
            # width as the waypoint rows and the token is a plain 2-D array.
            rate_rows = np.zeros(
                (self.num_waypoints, PLANAR_ACTION_DIM), dtype=np.float64
            )
            rate_rows[:, 0] = rates
            return np.concatenate((waypoints, rate_rows), axis=0)
        if self.velocity_mode == _DURATION:
            # One duration row per waypoint. Column 0 carries Δt in seconds;
            # the rest stay zero. Same rectangular layout as per_waypoint.
            duration_rows = np.zeros(
                (self.num_waypoints, PLANAR_ACTION_DIM), dtype=np.float64
            )
            duration_rows[:, 0] = durations
            return np.concatenate((waypoints, duration_rows), axis=0)
        timing = np.zeros((1, PLANAR_ACTION_DIM), dtype=np.float64)
        timing[0, 0] = speed
        return np.concatenate((waypoints, timing), axis=0)

    def transform(self, batch: dict) -> dict:
        value = np.asarray(batch[self.action_key])
        if value.ndim != 2 or value.shape[1] not in (2, 3, 4):
            raise ValueError(
                f"TokenizePlanarArcLength expects (T, 2|3|4), got {value.shape}"
            )
        if len(value) < 2 or not np.isfinite(value).all():
            raise ValueError("planar actions need at least two finite timesteps")
        output = self.tokenize(value.astype(np.float64, copy=False))
        expected_rows = arc_token_rows(self.num_waypoints, self.velocity_mode)
        if output.shape != (expected_rows, PLANAR_ACTION_DIM):
            raise AssertionError(
                f"{type(self).__name__} produced {output.shape}, expected "
                f"{(expected_rows, PLANAR_ACTION_DIM)} for velocity_mode="
                f"{self.velocity_mode!r}"
            )
        dtype = value.dtype if np.issubdtype(value.dtype, np.floating) else np.float32
        batch[self.output_action_key] = output.astype(dtype, copy=False)
        return batch
