"""Duration-timed ARC on LIBERO's single-arm SE(3) command path.

LIBERO actions are *delta controller commands*, not Cartesian waypoints. This
bridge integrates and differences commands using OSC's translation/rotation
scales. It extends the native duration ARC representation to xyz + SO(3) +
gripper; it does not feed 7-D controls into the planar tokenizer.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, Slerp

from egomimic.rldb.zarr.planar_arc import lambda_for_radius


def _rotation6d(rotation):
    matrix = rotation.as_matrix()
    return np.concatenate((matrix[..., :, 0], matrix[..., :, 1]), axis=-1)


def _from_rotation6d(values):
    a, b = values[..., :3].copy(), values[..., 3:].copy()
    bad = np.linalg.norm(a, axis=-1) < 1e-8
    a[bad] = (1, 0, 0)
    a /= np.linalg.norm(a, axis=-1, keepdims=True)
    b -= (a * b).sum(axis=-1, keepdims=True) * a
    bad = np.linalg.norm(b, axis=-1) < 1e-8
    if bad.any():
        fallback = np.eye(3)[np.argmin(np.abs(a[bad]), axis=-1)]
        b[bad] = fallback - (fallback * a[bad]).sum(axis=-1, keepdims=True) * a[bad]
    b /= np.linalg.norm(b, axis=-1, keepdims=True)
    return Rotation.from_matrix(np.stack((a, b, np.cross(a, b)), axis=-1))


class LiberoArcCodec:
    """M rows of [xyz(3), rotation6d(6), gripper(1), interval_seconds(1)].

    The spatial clock uses ARC's translation + scaled chordal rotation metric.
    A small gripper term gives grip-only motion geometric support. Stationary
    interval endpoints are retained when the waypoint budget permits; timing
    is decoded as a cumulative duration clock, never inverted speed.
    """

    def __init__(
        self,
        num_waypoints=16,
        horizon=32,
        dt=0.05,
        translation_scale=0.05,
        rotation_scale=0.5,
        rotation_radius=0.05,
        gripper_radius=0.01,
    ):
        self.num_waypoints, self.horizon = int(num_waypoints), int(horizon)
        self.dt = float(dt)
        self.translation_scale, self.rotation_scale = (
            float(translation_scale),
            float(rotation_scale),
        )
        self.rotation_radius, self.gripper_radius = (
            float(rotation_radius),
            float(gripper_radius),
        )
        if self.num_waypoints < 2 or self.horizon < 1:
            raise ValueError(
                "ARC requires at least two waypoints and a positive horizon"
            )
        values = (
            dt,
            translation_scale,
            rotation_scale,
            rotation_radius,
            gripper_radius,
        )
        if not all(np.isfinite(v) and v > 0 for v in values):
            raise ValueError("ARC scales/radii/dt must be finite and positive")

    def _progress(self, xyz, rotations, grip):
        angles = (rotations[1:] * rotations[:-1].inv()).magnitude()
        step = np.linalg.norm(np.diff(xyz, axis=0), axis=-1)
        step += (
            lambda_for_radius(self.rotation_radius) * np.sqrt(2) * np.sin(angles / 4)
        )
        step += self.gripper_radius * np.abs(np.diff(grip))
        return np.r_[0.0, np.cumsum(step)]

    def encode(self, actions):
        actions = np.asarray(actions, dtype=np.float64)
        if actions.shape != (self.horizon, 7) or not np.isfinite(actions).all():
            raise ValueError(f"Expected finite ({self.horizon},7) LIBERO actions")
        xyz = np.vstack(
            (np.zeros(3), np.cumsum(actions[:, :3] * self.translation_scale, axis=0))
        )
        rotations = [Rotation.identity()]
        for increment in actions[:, 3:6]:
            # robosuite OSC applies the delta rotation in the world frame.
            rotations.append(
                Rotation.from_rotvec(increment * self.rotation_scale) * rotations[-1]
            )
        rotations = Rotation.from_quat(np.stack([r.as_quat() for r in rotations]))
        grip = np.r_[actions[0, 6], actions[:, 6]]
        progress = self._progress(xyz, rotations, grip)
        time = np.arange(self.horizon + 1, dtype=np.float64) * self.dt
        if self.num_waypoints >= len(time):
            selected = np.r_[time, np.repeat(time[-1], self.num_waypoints - len(time))]
        elif progress[-1] <= 1e-12:
            selected = np.linspace(0, time[-1], self.num_waypoints)
        else:
            # Retain both edges of dwell intervals and each grip transition.
            holds = np.diff(progress) <= 1e-12
            edges = np.flatnonzero(np.diff(np.r_[False, holds, False]))
            changes = np.flatnonzero(np.diff(grip) != 0)
            required = np.unique(np.r_[0, self.horizon, edges, changes, changes + 1])
            if len(required) > self.num_waypoints:
                required = required[
                    np.rint(
                        np.linspace(0, len(required) - 1, self.num_waypoints)
                    ).astype(int)
                ]
            selected = time[required].tolist()
            # Split the greatest remaining ARC interval; use time for a hold.
            while len(selected) < self.num_waypoints:
                selected.sort()
                s = np.interp(selected, time, progress)
                scores = np.diff(s) + np.diff(selected) * 1e-12
                index = int(np.argmax(scores))
                if s[index + 1] - s[index] <= 1e-12:
                    point = (selected[index] + selected[index + 1]) / 2
                else:
                    point = float(
                        np.interp((s[index] + s[index + 1]) / 2, progress, time)
                    )
                selected.append(point)
            selected = np.sort(selected)
        selected = np.asarray(selected)
        positions = np.stack(
            [np.interp(selected, time, xyz[:, i]) for i in range(3)], axis=-1
        )
        orientation = _rotation6d(Slerp(time, rotations)(selected))
        # Gripper is a piecewise constant controller command, not an angle.
        grip_values = grip[
            np.searchsorted(time, selected + 1e-10, side="right").clip(1, len(time)) - 1
        ]
        durations = np.r_[np.diff(selected), 0.0]
        return np.column_stack((positions, orientation, grip_values, durations)).astype(
            np.float32
        )

    def decode(self, tokens):
        tokens = np.asarray(tokens, dtype=np.float64)
        if tokens.shape != (self.num_waypoints, 11) or not np.isfinite(tokens).all():
            raise ValueError(f"Expected finite ({self.num_waypoints},11) ARC tokens")
        clock = np.r_[0.0, np.cumsum(np.maximum(tokens[:-1, 10], 0.0))]
        keep = np.r_[True, np.diff(clock) > 1e-10]
        clock, tokens = clock[keep], tokens[keep]
        if len(clock) < 2:
            result = np.zeros((self.horizon, 7), dtype=np.float32)
            result[:, 6] = tokens[0, 9]
            return result
        query = np.minimum(np.arange(self.horizon + 1) * self.dt, clock[-1])
        rotations = _from_rotation6d(tokens[:, 3:9])
        progress = self._progress(tokens[:, :3], rotations, tokens[:, 9])
        q_progress = np.interp(query, clock, progress)
        unique = np.r_[True, np.diff(progress) > 1e-12]
        if unique.sum() >= 3:
            xyz = CubicSpline(
                progress[unique], tokens[unique, :3], axis=0, bc_type="natural"
            )(q_progress)
        elif unique.sum() == 2:
            xyz = np.stack(
                [
                    np.interp(q_progress, progress[unique], tokens[unique, i])
                    for i in range(3)
                ],
                axis=-1,
            )
        else:
            xyz = np.repeat(tokens[:1, :3], len(query), axis=0)
        # Interpolation in the duration clock also handles zero-translation rotation.
        rot = Slerp(clock, rotations)(query)
        delta_rot = (rot[1:] * rot[:-1].inv()).as_rotvec() / self.rotation_scale
        # Durations are stored as float32. Accumulation can put an exact 20 Hz
        # transition a few ulps after its query; do not delay the grip by a frame.
        clock_tolerance = 8 * np.finfo(np.float32).eps * max(1.0, clock[-1])
        grip_indices = (
            np.searchsorted(clock, query[1:] + clock_tolerance, side="right").clip(
                1, len(clock)
            )
            - 1
        )
        return np.column_stack(
            (
                np.diff(xyz, axis=0) / self.translation_scale,
                delta_rot,
                tokens[grip_indices, 9],
            )
        ).astype(np.float32)
