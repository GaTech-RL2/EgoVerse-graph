"""Independent translation/rotation ARC streams for LIBERO OSC commands.

Ports the ``stk`` (velocity) and ``dur`` (duration) contracts used by
TokenizePlanarArcTimed / PlanarArcTimedNativeDecoder. The reference is the
EgoVerse-graph obstacle campaign at 02db41a01cfdd018785b59c06a1485c2717ad683.
XYZ and SO(3) have separate supports, budgets and clocks; gripper commands
follow the translation clock. LIBERO-specific adaptations are OSC integration,
SO(3) SLERP, rotation6d, and held discrete gripper commands.

The earlier shared-clock LiberoArcCodec remains a separate, unchanged baseline.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from egomimic.rldb.zarr.libero_arc import (
    LiberoArcCodec,
    _from_rotation6d,
    _rotation6d,
)


def _bracket(cumulative, target):
    if target <= cumulative[0]:
        return 0.0
    if target >= cumulative[-1]:
        return float(len(cumulative) - 1)
    lower = int(np.searchsorted(cumulative, target, side="left")) - 1
    span = cumulative[lower + 1] - cumulative[lower]
    return lower + (target - cumulative[lower]) / span


def _support_frames(cumulative, end, count, mode, epsilon=1e-9):
    """Uniform arc allocation, reserving longest dwells for duration tokens.

    The dwell reservation and quota allocation follow duration_support_frames
    in the pinned native planar codec, including its fixed support budget.
    """
    frames = np.arange(len(cumulative), dtype=np.float64)
    if end <= epsilon:
        return np.linspace(0, frames[-1], count)
    targets = np.linspace(0, end, count)
    uniform = np.array([_bracket(cumulative, t) for t in targets])
    if mode == "stk":
        return uniform
    stopped = np.diff(cumulative) <= epsilon
    if not stopped.any():
        return uniform
    starts = np.flatnonzero(stopped & np.r_[True, ~stopped[:-1]])
    ends = np.flatnonzero(stopped & np.r_[~stopped[1:], True]) + 1
    end_frame = uniform[-1]
    holds = [
        (float(a), min(float(b), end_frame))
        for a, b in zip(starts, ends)
        if a < end_frame
    ]
    selected = {0.0, end_frame}
    for start, stop in sorted(holds, key=lambda hold: (hold[0] - hold[1], hold[0])):
        candidate = selected | {start, stop}
        if len(candidate) <= count:
            selected = candidate
    boundaries = np.array(sorted(selected))
    if len(boundaries) == 2:
        return uniform
    boundary_arc = np.interp(boundaries, frames, cumulative)
    ranks = np.linspace(0, 1, count)
    boundary_rank = np.interp(boundary_arc, targets, ranks)
    weights = np.diff(boundary_rank)
    quotas = (count - len(boundaries)) * weights / weights.sum()
    counts = 1 + np.floor(quotas).astype(int)
    remaining = count - 1 - counts.sum()
    counts[np.argsort(-(quotas - np.floor(quotas)), kind="stable")[:remaining]] += 1
    supports = [0.0]
    for index, pieces in enumerate(counts):
        if boundary_arc[index + 1] - boundary_arc[index] <= epsilon:
            supports.extend(
                np.linspace(boundaries[index], boundaries[index + 1], pieces + 1)[1:-1]
            )
        else:
            rank = np.linspace(
                boundary_rank[index], boundary_rank[index + 1], pieces + 1
            )[1:-1]
            supports.extend(
                _bracket(cumulative, t) for t in np.interp(rank, ranks, targets)
            )
        supports.append(boundaries[index + 1])
    return np.asarray(supports)


class LiberoArcTimedCodec(LiberoArcCodec):
    """M x 12: [xyz(3), xyz_timing, rotation6d(6), rot_timing, gripper].

    ``stk`` timings are m/s and rad/s; ``dur`` timings are interval seconds.
    Rates describe distances between the encoded supports, matching decoding.
    A velocity of zero cannot represent a stationary dwell; this is an inherent
    limitation of stk and is measured, not patched with hidden time channels.
    D and R independently cap accumulated translation metres and SO(3) degrees.
    """

    def __init__(self, mode="dur", **kwargs):
        super().__init__(**kwargs)
        if mode not in {"stk", "dur"}:
            raise ValueError("Independent ARC mode must be stk or dur")
        self.mode = mode

    def encode(self, actions):
        actions = np.asarray(actions, dtype=np.float64)
        if actions.shape != (self.horizon, 7) or not np.isfinite(actions).all():
            raise ValueError(f"Expected finite ({self.horizon},7) LIBERO actions")
        xyz = np.vstack(
            (np.zeros(3), np.cumsum(actions[:, :3] * self.translation_scale, axis=0))
        )
        rotations = [Rotation.identity()]
        for increment in actions[:, 3:6]:
            rotations.append(
                Rotation.from_rotvec(increment * self.rotation_scale) * rotations[-1]
            )
        rotations = Rotation.from_quat(np.stack([r.as_quat() for r in rotations]))
        translation = np.r_[
            0.0, np.cumsum(np.linalg.norm(np.diff(xyz, axis=0), axis=-1))
        ]
        angular = np.r_[
            0.0, np.cumsum((rotations[1:] * rotations[:-1].inv()).magnitude())
        ]
        t_end = (
            translation[-1]
            if self.max_translation is None
            else min(translation[-1], self.max_translation)
        )
        r_end = (
            angular[-1]
            if self.max_rotation_degrees is None
            else min(angular[-1], np.deg2rad(self.max_rotation_degrees))
        )
        t_frames = _support_frames(translation, t_end, self.num_waypoints, self.mode)
        r_frames = _support_frames(angular, r_end, self.num_waypoints, self.mode)
        frames = np.arange(self.horizon + 1)
        points = np.column_stack(
            [np.interp(t_frames, frames, xyz[:, i]) for i in range(3)]
        )
        orientation = Slerp(frames, rotations)(r_frames)
        t_time, r_time = np.diff(t_frames) * self.dt, np.diff(r_frames) * self.dt
        if self.mode == "stk":
            t_time = np.divide(
                np.linalg.norm(np.diff(points, axis=0), axis=-1),
                t_time,
                out=np.zeros_like(t_time),
                where=t_time > 1e-9,
            )
            r_time = np.divide(
                (orientation[1:] * orientation[:-1].inv()).magnitude(),
                r_time,
                out=np.zeros_like(r_time),
                where=r_time > 1e-9,
            )
        grip = np.r_[actions[0, 6], actions[:, 6]]
        grip_values = grip[np.floor(t_frames + 1e-10).astype(int)]
        return np.column_stack(
            (
                points,
                np.r_[t_time, t_time[-1]],
                _rotation6d(orientation),
                np.r_[r_time, r_time[-1]],
                grip_values,
            )
        ).astype(np.float32)

    def _durations(self, distance, timing):
        active = distance > 1e-9
        stop = self.dt * (self.horizon + 1)
        if self.mode == "dur":
            return np.where(active & (timing <= 1e-9), stop, np.maximum(timing, 0))
        usable = np.abs(timing) > 1e-9
        duration = np.divide(
            distance, np.abs(timing), out=np.zeros_like(distance), where=active & usable
        )
        return np.where(active & ~usable, stop, duration)

    def clocks(self, tokens):
        tokens = np.asarray(tokens, dtype=np.float64)
        if tokens.shape != (self.num_waypoints, 12) or not np.isfinite(tokens).all():
            raise ValueError(f"Expected finite ({self.num_waypoints},12) ARC tokens")
        rotation = _from_rotation6d(tokens[:, 4:10])
        distances = (
            np.linalg.norm(np.diff(tokens[:, :3], axis=0), axis=-1),
            (rotation[1:] * rotation[:-1].inv()).magnitude(),
        )
        return tuple(
            np.r_[0.0, np.cumsum(self._durations(d, tokens[:-1, index]))]
            for d, index in zip(distances, (3, 10))
        )

    def represented_seconds(self, tokens):
        clocks = self.clocks(tokens)
        # An inactive stream does not limit execution. Grip-only stk is active
        # but has a zero translation clock and therefore zero timing coverage.
        xyz_active = np.any(np.abs(np.diff(tokens[:, [0, 1, 2, 11]], axis=0)) > 1e-9)
        rotation = _from_rotation6d(tokens[:, 4:10])
        rot_active = np.any((rotation[1:] * rotation[:-1].inv()).magnitude() > 1e-9)
        return min(
            [c[-1] for c, active in zip(clocks, (xyz_active, rot_active)) if active]
            or [self.horizon * self.dt]
        )

    def decode(self, tokens):
        tokens = np.asarray(tokens, dtype=np.float64)
        t_clock, r_clock = self.clocks(tokens)
        queries = np.arange(self.horizon + 1) * self.dt

        def brackets(clock):
            upper = np.searchsorted(clock, queries, side="right").clip(
                1, self.num_waypoints - 1
            )
            lower = upper - 1
            fraction = (
                (queries - clock[lower]) / np.maximum(clock[upper] - clock[lower], 1e-9)
            ).clip(0, 1)
            return lower, upper, fraction

        lower, upper, fraction = brackets(t_clock)
        xyz = (
            tokens[lower, :3] * (1 - fraction[:, None])
            + tokens[upper, :3] * fraction[:, None]
        )
        xyz[0] = tokens[0, :3]
        r_lower, r_upper, fraction = brackets(r_clock)
        orientation = _from_rotation6d(tokens[:, 4:10])
        relative = orientation[r_upper] * orientation[r_lower].inv()
        rotation = (
            Rotation.from_rotvec(relative.as_rotvec() * fraction[:, None])
            * orientation[r_lower]
        )
        quaternions = rotation.as_quat()
        quaternions[0] = orientation[0].as_quat()
        rotation = Rotation.from_quat(quaternions)
        tolerance = 8 * np.finfo(np.float32).eps * max(1.0, t_clock[-1])
        grip_index = (
            np.searchsorted(t_clock, queries[1:] + tolerance, side="right").clip(
                1, self.num_waypoints
            )
            - 1
        )
        return np.column_stack(
            (
                np.diff(xyz, axis=0) / self.translation_scale,
                (rotation[1:] * rotation[:-1].inv()).as_rotvec() / self.rotation_scale,
                tokens[grip_index, 11],
            )
        ).astype(np.float32)


def make_libero_arc_codec(mode="joint_dur", **kwargs):
    if mode == "joint_dur":
        return LiberoArcCodec(**kwargs)
    return LiberoArcTimedCodec(mode=mode, **kwargs)


def codec_source_files(mode):
    """All local codec implementation dependencies for replay/training gating."""
    paths = ["egomimic/rldb/zarr/libero_arc.py", "egomimic/rldb/zarr/planar_arc.py"]
    if mode in {"stk", "dur"}:
        paths.append("egomimic/rldb/zarr/libero_arc_timed.py")
    elif mode != "joint_dur":
        raise ValueError(f"Unknown ARC mode {mode!r}")
    return paths
