"""Small, explicit SE(2) arc tokenizer used by Planar PushShapes models."""

from __future__ import annotations

import math

import numpy as np
from scipy.interpolate import CubicSpline

PLANAR_ACTION_DIM = 5  # [x, y, cos(theta), sin(theta), grip]


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
        (
            xy[inside],
            _interpolate(xy, cumulative, end)[None],
        ),
        axis=0,
    )
    keep = np.concatenate(
        (np.ones(1, dtype=bool), np.diff(parameter) > epsilon)
    )
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
    """Sample a natural cubic curve using L2 chord-error knot density.

    The curve is parameterized by the tokenizer's existing cumulative clock.
    Density is integrated against the fitted curve's Cartesian arc length:
    ``rho(s) = (curvature_floor**2 + curvature(s)**2)**(1/5)``.
    This approaches the L2-optimal ``|curvature|**(2/5)`` density away from
    the finite floor, while retaining uniform arc sampling on straight spans.

    Returns the sampled XY points and their positions in the existing clock.
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
    arc_integrand = density * curve_speed
    importance = np.concatenate(
        (
            np.zeros(1),
            np.cumsum(
                0.5 * (arc_integrand[:-1] + arc_integrand[1:]) * delta_s
            ),
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
    return spline(targets), targets


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
    """Encode a future native action chunk as M arc waypoints plus timing.

    The final row has only its first field populated with mean arc speed. The
    first waypoint is copied from timestep zero rather than recovered via an
    arc lookup; this preserves stationary grip transitions exactly.

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
        self.hybrid_rotation_unit = (
            None
            if hybrid_rotation_unit is None
            else float(hybrid_rotation_unit)
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
            rotation_fraction = min(
                1.0, self.hybrid_rotation_unit / rotation_span
            )
            end = min(end, translation_span * rotation_fraction)
        return end

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
        dtype = value.dtype if np.issubdtype(value.dtype, np.floating) else np.float32
        batch[self.output_action_key] = output.astype(dtype, copy=False)
        return batch
