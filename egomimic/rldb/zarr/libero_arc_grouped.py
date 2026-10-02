"""Controlled support, rate and clock ablations for LIBERO ARC.

The shape precedes the timing channels. XYZ uses metres, SO(3) uses rotation6d,
and gripper remains a held command. The optional angular-driver control stores
the integral of world-frame angular commands, *not* Euler angles; its decoder
composes rotations at every clock breakpoint, preserving noncommutativity.
No source timestamps, episode history or unexecuted actions enter decoding.
"""

from __future__ import annotations

import copy

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from egomimic.rldb.zarr.libero_arc import _from_rotation6d, _rotation6d
from egomimic.rldb.zarr.libero_arc_timed import LiberoArcTimedCodec, _support_frames


def _partition(value, wanted, name):
    groups = tuple(tuple(int(i) for i in group) for group in value)
    if not groups or any(not group for group in groups):
        raise ValueError(f"{name} requires nonempty groups")
    if sorted(i for group in groups for i in group) != list(wanted):
        raise ValueError(f"{name} must partition {list(wanted)} exactly once")
    return groups


def _interp(values, clock, queries):
    """Right-continuous supports, linear interpolation, then hold the endpoint."""
    upper = np.searchsorted(clock, queries, side="right").clip(1, len(clock) - 1)
    lower = upper - 1
    alpha = np.clip(
        (queries - clock[lower]) / np.maximum(clock[upper] - clock[lower], 1e-9),
        0,
        1,
    )
    result = values[lower] * (1 - alpha) + values[upper] * alpha
    result[queries == 0] = values[0]
    return result


class LiberoArcGroupedCodec(LiberoArcTimedCodec):
    """Use one explicit stream specification for encoding and decoding.

    ``translation_groups`` partitions x/y/z. ``gripper`` is ``translation``
    (the legacy first translation stream), or ``separate`` (its own
    event-distance stream).
    Rates are group norms or signed physical components. Group clocks use
    norm(displacement)/norm(velocity); component clocks invert each component.
    Duration targets encode elapsed seconds directly, including zero-motion
    intervals. Component clocks are distinct from component support sampling.

    D/R cap each selected stream in metres/radians. Gripper distance is total
    variation in controller command units; it is not capped by D or R.
    """

    def __init__(self, stream_spec, mode="dur", **kwargs):
        super().__init__(mode=mode, **kwargs)
        spec = dict(stream_spec)
        allowed = {
            "translation_groups",
            "rotation_groups",
            "rotation_representation",
            "gripper",
            "timing",
            "clocks",
        }
        if set(spec) - allowed:
            raise ValueError(f"Unknown stream options: {sorted(set(spec) - allowed)}")
        self.translation_groups = _partition(
            spec.get("translation_groups", [[0, 1, 2]]), range(3), "translation"
        )
        self.rotation_groups = _partition(
            spec.get("rotation_groups", [[0, 1, 2]]), range(3), "rotation"
        )
        self.rotation_representation = spec.get("rotation_representation", "so3")
        self.gripper = spec.get("gripper", "translation")
        self.timing = spec.get("timing", "group")
        self.clock_mode = spec.get("clocks", "group")
        if self.rotation_representation not in {"so3", "angular_driver"}:
            raise ValueError("Rotation representation must be so3 or angular_driver")
        if self.rotation_representation == "so3" and self.rotation_groups != (
            (0, 1, 2),
        ):
            raise ValueError(
                "SO(3) supports cannot be split into Euler/rotation6d axes"
            )
        if self.gripper not in {"translation", "separate"}:
            raise ValueError("Unknown gripper stream")
        if self.timing not in {"group", "component"} or self.clock_mode not in {
            "group",
            "component",
        }:
            raise ValueError("Timing and clocks must be group or component")
        if self.clock_mode == "component" and self.timing != "component":
            raise ValueError("Independent component clocks require component timing")
        if self.timing == "component" and self.gripper != "separate":
            raise ValueError("Component timing uses the separate-gripper control")
        # Canonical JSON-ready metadata is also part of target-cache identity.
        self.stream_spec = {
            "translation_groups": [list(g) for g in self.translation_groups],
            "rotation_groups": [list(g) for g in self.rotation_groups],
            "rotation_representation": self.rotation_representation,
            "gripper": self.gripper,
            "timing": self.timing,
            "clocks": self.clock_mode,
        }
        self.shape_dim = 10 if self.rotation_representation == "so3" else 7
        self.grip_column = self.shape_dim - 1
        self.groups = []
        for index, axes in enumerate(self.translation_groups):
            self.groups.append((f"xyz{index}", "xyz", axes))
        for index, axes in enumerate(self.rotation_groups):
            self.groups.append((f"rot{index}", self.rotation_representation, axes))
        if self.gripper == "separate":
            self.groups.append(("grip", "grip", (0,)))
        self.timing_slices = {}
        offset = self.shape_dim
        for name, _, axes in self.groups:
            width = len(axes) if self.timing == "component" else 1
            self.timing_slices[name] = slice(offset, offset + width)
            offset += width
        self.action_dim = offset

    def representation_context(self):
        return {
            "version": 1,
            "layout": "shape_then_timing",
            "stream_spec": copy.deepcopy(self.stream_spec),
            "shape_dim": self.shape_dim,
            "action_dim": self.action_dim,
            "gripper_distance": "absolute_command_variation_uncapped",
            "component_rotation_basis": "world_frame_angular_increments",
        }

    def token_scale(self):
        scale = np.ones(self.action_dim, dtype=np.float64)
        scale[:3] = self.translation_scale * self.horizon
        if self.rotation_representation == "angular_driver":
            scale[3:6] = self.rotation_scale * self.horizon
        for name, kind, axes in self.groups:
            if self.mode == "dur":
                bound = self.dt * self.horizon
            else:
                unit = {
                    "xyz": self.translation_scale,
                    "so3": self.rotation_scale,
                    "angular_driver": self.rotation_scale,
                    "grip": 2.0,
                }[kind]
                bound = unit / self.dt
                if self.timing == "group":
                    bound *= np.sqrt(len(axes))
            scale[self.timing_slices[name]] = bound
        return scale

    def _vectors(self, values, group):
        name, kind, axes = group
        if kind == "so3":
            rot = _from_rotation6d(values[:, 3:9])
            return (rot[1:] * rot[:-1].inv()).as_rotvec()
        columns = (
            (self.grip_column,)
            if kind == "grip"
            else tuple(a + (3 if kind == "angular_driver" else 0) for a in axes)
        )
        vectors = np.diff(values[:, columns], axis=0)
        return vectors

    def _supports(self, values, group):
        _, kind, _ = group
        vectors = self._vectors(values, group)
        cumulative = np.r_[0.0, np.cumsum(np.linalg.norm(vectors, axis=-1))]
        budget = (
            self.max_translation
            if kind == "xyz"
            else np.deg2rad(self.max_rotation_degrees)
            if kind in {"so3", "angular_driver"}
            and self.max_rotation_degrees is not None
            else None
        )
        end = cumulative[-1] if budget is None else min(cumulative[-1], budget)
        if kind == "grip":
            # A discrete command has events, not intermediate apertures. Keep
            # event times, then pad; rate tokens can time the jump to an event,
            # but still cannot carry a final stationary dwell.
            changes = np.flatnonzero(np.abs(vectors[:, 0]) > 1e-9) + 1
            required = np.unique(np.r_[0, changes, self.horizon]).astype(float)
            if len(required) > self.num_waypoints:
                required = required[
                    np.rint(
                        np.linspace(0, len(required) - 1, self.num_waypoints)
                    ).astype(int)
                ]
            return np.r_[
                required, np.repeat(required[-1], self.num_waypoints - len(required))
            ]
        return _support_frames(cumulative, end, self.num_waypoints, self.mode)

    def encode(self, actions):
        actions = np.asarray(actions, dtype=np.float64)
        if actions.shape != (self.horizon, 7) or not np.isfinite(actions).all():
            raise ValueError(f"Expected finite ({self.horizon},7) LIBERO actions")
        source = np.zeros((self.horizon + 1, self.shape_dim))
        source[1:, :3] = np.cumsum(actions[:, :3] * self.translation_scale, axis=0)
        if self.rotation_representation == "so3":
            rotations = [Rotation.identity()]
            for action in actions:
                rotations.append(
                    Rotation.from_rotvec(action[3:6] * self.rotation_scale)
                    * rotations[-1]
                )
            rotation = Rotation.from_quat([r.as_quat() for r in rotations])
            source[:, 3:9] = _rotation6d(rotation)
        else:
            source[1:, 3:6] = np.cumsum(actions[:, 3:6] * self.rotation_scale, axis=0)
        source[:, self.grip_column] = np.r_[actions[0, 6], actions[:, 6]]
        result = np.zeros((self.num_waypoints, self.action_dim))
        frames = np.arange(self.horizon + 1)
        supports = {}
        for group in self.groups:
            name, kind, axes = group
            selected = self._supports(source, group)
            supports[name] = selected
            if kind == "so3":
                result[:, 3:9] = _rotation6d(Slerp(frames, rotation)(selected))
            elif kind == "grip":
                result[:, self.grip_column] = source[
                    np.floor(selected + 1e-10).astype(int), self.grip_column
                ]
            else:
                for axis in axes:
                    column = axis + (3 if kind == "angular_driver" else 0)
                    result[:, column] = np.interp(selected, frames, source[:, column])
            if name == "xyz0" and self.gripper != "separate":
                result[:, self.grip_column] = source[
                    np.floor(selected + 1e-10).astype(int), self.grip_column
                ]
        for group in self.groups:
            name, _, _ = group
            elapsed = np.diff(supports[name]) * self.dt
            width = self.timing_slices[name].stop - self.timing_slices[name].start
            if self.mode == "dur":
                timing = np.repeat(elapsed[:, None], width, axis=1)
            else:
                vectors = self._vectors(result, group)
                distances = (
                    np.linalg.norm(vectors, axis=-1, keepdims=True)
                    if self.timing == "group"
                    else vectors
                )
                timing = np.divide(
                    distances,
                    elapsed[:, None],
                    out=np.zeros_like(distances),
                    where=elapsed[:, None] > 1e-9,
                )
            result[:, self.timing_slices[name]] = np.vstack((timing, timing[-1]))
        return result.astype(np.float32)

    def _validate(self, tokens):
        values = np.asarray(tokens, dtype=np.float64)
        if (
            values.shape != (self.num_waypoints, self.action_dim)
            or not np.isfinite(values).all()
        ):
            raise ValueError(
                f"Expected finite ({self.num_waypoints},{self.action_dim}) ARC tokens"
            )
        return values

    def clocks(self, tokens):
        values = self._validate(tokens)
        result = {}
        for group in self.groups:
            name, _, _ = group
            vectors = self._vectors(values, group)
            timing = values[:-1, self.timing_slices[name]]
            if self.clock_mode == "group":
                distance = np.linalg.norm(vectors, axis=-1, keepdims=True)
                if self.timing == "component":
                    timing = (
                        timing.mean(axis=-1, keepdims=True)
                        if self.mode == "dur"
                        else np.linalg.norm(timing, axis=-1, keepdims=True)
                    )
            else:
                distance = np.abs(vectors)
            intervals = self._durations(distance, timing)
            result[name] = np.vstack(
                (np.zeros(intervals.shape[1]), np.cumsum(intervals, axis=0))
            )
        return result

    def represented_seconds(self, tokens):
        values = self._validate(tokens)
        clocks = self.clocks(values)
        active_ends = []
        for group in self.groups:
            name, _, _ = group
            vectors = self._vectors(values, group)
            active = np.any(np.abs(vectors) > 1e-9, axis=0)
            if self.clock_mode == "group":
                active = np.array([active.any()])
                if name == "xyz0" and self.gripper != "separate":
                    active[0] |= np.any(
                        np.abs(np.diff(values[:, self.grip_column])) > 1e-9
                    )
            active_ends.extend(clocks[name][-1, active].tolist())
        return min(active_ends or [self.horizon * self.dt])

    def _angular_path(self, values, clocks, queries):
        """Integrate at the union of all breakpoints; never add Euler angles."""
        if self.rotation_representation == "so3":
            orientation = _from_rotation6d(values[:, 3:9])
            increments = (orientation[1:] * orientation[:-1].inv()).as_rotvec()
            driver = np.vstack((np.zeros(3), np.cumsum(increments, axis=0)))
            initial = orientation[0]
        else:
            driver = values[:, 3:6]
            initial = Rotation.identity()
        axis_clocks = [None] * 3
        for name, kind, axes in self.groups:
            if kind not in {"so3", "angular_driver"}:
                continue
            for index, axis in enumerate(axes):
                axis_clocks[axis] = clocks[name][
                    :, index if self.clock_mode == "component" else 0
                ]
        events = np.unique(
            np.concatenate(
                [queries] + [c[(c > 0) & (c < queries[-1])] for c in axis_clocks]
            )
        )
        path = np.column_stack(
            [_interp(driver[:, a], c, events) for a, c in enumerate(axis_clocks)]
        )
        rotations = [initial]
        for delta in np.diff(path, axis=0):
            rotations.append(Rotation.from_rotvec(delta) * rotations[-1])
        rotations = Rotation.from_quat([r.as_quat() for r in rotations])
        return rotations[np.searchsorted(events, queries)]

    def decode(self, tokens):
        values = self._validate(tokens)
        clocks = self.clocks(values)
        queries = np.arange(self.horizon + 1) * self.dt
        xyz = np.zeros((self.horizon + 1, 3))
        for name, kind, axes in self.groups:
            if kind == "xyz":
                for index, axis in enumerate(axes):
                    clock = clocks[name][
                        :, index if self.clock_mode == "component" else 0
                    ]
                    xyz[:, axis] = _interp(values[:, axis], clock, queries)
        if self.rotation_representation == "so3" and self.clock_mode == "group":
            # Keep the legacy interpolation calculation, including zero clocks.
            clock = clocks["rot0"][:, 0]
            upper = np.searchsorted(clock, queries, side="right").clip(
                1, self.num_waypoints - 1
            )
            lower = upper - 1
            alpha = (
                (queries - clock[lower]) / np.maximum(clock[upper] - clock[lower], 1e-9)
            ).clip(0, 1)
            orientation = _from_rotation6d(values[:, 3:9])
            relative = orientation[upper] * orientation[lower].inv()
            rotation = (
                Rotation.from_rotvec(relative.as_rotvec() * alpha[:, None])
                * orientation[lower]
            )
            quaternions = rotation.as_quat()
            quaternions[0] = orientation[0].as_quat()
            rotation = Rotation.from_quat(quaternions)
        else:
            rotation = self._angular_path(values, clocks, queries)
        grip_clock = clocks["grip" if self.gripper == "separate" else "xyz0"][:, 0]
        tolerance = 8 * np.finfo(np.float32).eps * max(1.0, grip_clock[-1])
        indices = (
            np.searchsorted(grip_clock, queries[1:] + tolerance, side="right").clip(
                1, self.num_waypoints
            )
            - 1
        )
        return np.column_stack(
            (
                np.diff(xyz, axis=0) / self.translation_scale,
                (rotation[1:] * rotation[:-1].inv()).as_rotvec() / self.rotation_scale,
                values[indices, self.grip_column],
            )
        ).astype(np.float32)
