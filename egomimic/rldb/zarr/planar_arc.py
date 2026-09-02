"""Small, explicit SE(2) arc tokenizer used by Planar PushShapes policies."""

from __future__ import annotations

import math

import numpy as np

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
    """

    def __init__(
        self,
        action_key: str = "actions",
        output_action_key: str = "actions",
        min_distance_unit: float = 200.0,
        resampled_vector_length: int = 100,
        dt: float = 1.0 / 30.0,
        rotation_radius: float = 0.0,
        zero_dist_epsilon: float = 1e-9,
    ):
        if min_distance_unit <= 0 or dt <= 0:
            raise ValueError("min_distance_unit and dt must be positive")
        if resampled_vector_length < 2:
            raise ValueError("resampled_vector_length must be at least two")
        if rotation_radius < 0:
            raise ValueError("rotation_radius must be non-negative")
        self.action_key = str(action_key)
        self.output_action_key = str(output_action_key)
        self.distance = float(min_distance_unit)
        self.num_waypoints = int(resampled_vector_length)
        self.dt = float(dt)
        self.rotation_radius = float(rotation_radius)
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

    def tokenize(self, actions: np.ndarray) -> np.ndarray:
        xy, theta, grip = self._components(actions)
        weight = lambda_for_radius(self.rotation_radius)
        steps = planar_step_distance(xy, theta, weight)
        cumulative = np.concatenate((np.zeros(1), np.cumsum(steps)))
        end = min(self.distance, float(cumulative[-1]))

        if end <= self.zero_dist_epsilon:
            xy_waypoints = np.repeat(xy[:1], self.num_waypoints, axis=0)
            theta_waypoints = np.repeat(theta[0], self.num_waypoints)
            grip_waypoints = np.repeat(grip[0], self.num_waypoints)
            speed = 0.0
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
