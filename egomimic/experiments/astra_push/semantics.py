"""Independent language resolution and whole-trajectory success measurements."""

import itertools
import re

import numpy as np

from egomimic.experiments.astra_push.schemas import SceneSpec, TaskSpec


def resolve_instruction(scene, instruction, initial_positions):
    """Resolve student language independently of the teacher's object argument."""
    if scene.stage == "S1" and instruction == "Push the block into the target.":
        return scene.cubes[0].name, "target"
    if scene.stage == "S2":
        match = re.fullmatch(
            r"Push the (red|blue) block to the (left|right) target\. Leave the (red|blue) block in place\.",
            instruction,
        )
        if match and match[1] != match[3]:
            candidates = [c.name for c in scene.cubes if c.color == match[1]]
            if len(candidates) == 1:
                return candidates[0], match[2]
    if scene.stage == "S3":
        match = re.fullmatch(
            r"Push the block (left|right) of the marker to the (left|right) target\. Leave the other block in place\.",
            instruction,
        )
        if match:
            marker = next(f for f in scene.fixtures if f.name == "marker")
            sign = -1 if match[1] == "left" else 1
            candidates = [
                name
                for name, pos in initial_positions.items()
                if sign * (pos[1] - marker.center_xy[1]) >= 0.05
            ]
            if len(candidates) == 1:
                return candidates[0], match[2]
    raise ValueError("Instruction has no unique grounded referent in this reset")


def cube_corners(position, rotation):
    position, rotation = (
        np.asarray(position, dtype=float),
        np.asarray(rotation, dtype=float),
    )
    if (
        position.shape != (3,)
        or rotation.shape != (3, 3)
        or not np.isfinite(position).all()
        or not np.isfinite(rotation).all()
    ):
        raise ValueError("Cube pose must be finite XYZ and 3x3 orientation")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or not np.isclose(
        np.linalg.det(rotation), 1, atol=1e-5
    ):
        raise ValueError("Cube rotation must be proper orthonormal")
    local = np.array(list(itertools.product([-0.02, 0.02], repeat=3)))
    return local @ rotation.T + position


class SuccessEvaluator:
    def __init__(self, scene, task, initial_positions):
        self.scene, self.task = (
            SceneSpec.model_validate(scene),
            TaskSpec.model_validate(task),
        )
        self.task.validate_scene(self.scene)
        if set(initial_positions) != {c.name for c in self.scene.cubes}:
            raise ValueError("Initial state must contain every cube, and only cubes")
        self.initial = {
            k: np.asarray(v, dtype=float).copy() for k, v in initial_positions.items()
        }
        for position in self.initial.values():
            cube_corners(position, np.eye(3))
        self.requested, self.destination = resolve_instruction(
            self.scene, self.task.instruction, self.initial
        )
        self.target = next(t for t in self.scene.targets if t.name == self.destination)
        self.hold = self.max_hold = self.steps = 0
        self.max_lift = self.max_preservation = 0.0
        self.other_target_seen = False

    def update(self, positions, rotations):
        if set(positions) != set(self.initial) or set(rotations) != set(self.initial):
            raise ValueError("Every control step must supply every cube pose")
        corners = {
            name: cube_corners(positions[name], rotations[name])
            for name in self.initial
        }
        self.steps += 1
        requested = np.asarray(positions[self.requested], dtype=float)
        self.max_lift = max(
            self.max_lift, float(requested[2] - self.initial[self.requested][2])
        )
        for name in self.initial:
            if name != self.requested:
                self.max_preservation = max(
                    self.max_preservation,
                    float(
                        np.linalg.norm(np.asarray(positions[name]) - self.initial[name])
                    ),
                )
        inside = np.all(
            np.abs(corners[self.requested][:, :2] - self.target.center_xy)
            <= self.target.half_width + 1e-9
        )
        self.hold = self.hold + 1 if inside else 0
        self.max_hold = max(self.max_hold, self.hold)
        for target in self.scene.targets:
            if target.name != self.destination and np.all(
                np.abs(corners[self.requested][:, :2] - target.center_xy)
                <= target.half_width + 1e-9
            ):
                self.other_target_seen = True
        return self.result()

    def result(self):
        lift = self.max_lift > 0.01 + 1e-9
        preservation = self.max_preservation > 0.015 + 1e-9
        return {
            "success": self.max_hold >= 10 and not lift and not preservation,
            "max_hold_steps": self.max_hold,
            "steps": self.steps,
            "max_lift_m": self.max_lift,
            "max_nonrequested_displacement_m": self.max_preservation,
            "lift_violation": lift,
            "preservation_violation": preservation,
            "wrong_object": preservation,
            "wrong_target": self.other_target_seen,
        }
