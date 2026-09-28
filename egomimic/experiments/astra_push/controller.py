"""Fixed closed-loop execution of bounded, typed pushing programs.

This module runs only in the simulator environment. Privileged inputs are
teacher-side and are never returned as student observations.
"""

from pathlib import Path

import numpy as np
import yaml

from egomimic.experiments.astra_push.schemas import TeacherProgram

CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "hydra_configs/astra_push/simulator.yaml"
)


def settings():
    return yaml.safe_load(CONFIG_PATH.read_text())


def cube_poses(environment, scene):
    positions, rotations = {}, {}
    for cube in scene.cubes:
        body = environment.env.objects_dict[cube.name]
        body_id = environment.sim.model.body_name2id(body.root_body)
        positions[cube.name] = environment.sim.data.body_xpos[body_id].copy()
        rotations[cube.name] = (
            environment.sim.data.body_xmat[body_id].reshape(3, 3).copy()
        )
    return positions, rotations


def student_observation(observation):
    """The only simulator -> learner conversion; quaternion order is XYZW."""
    state = np.concatenate(
        [
            observation["robot0_eef_pos"],
            observation["robot0_eef_quat"],
            observation["robot0_gripper_qpos"],
        ]
    ).astype(np.float32)
    if state.shape != (9,) or not np.isfinite(state).all():
        raise ValueError("Invalid measured Panda proprioception")
    result = {"proprioception": state}
    for source, target in (
        ("agentview_image", "external_rgb"),
        ("robot0_eye_in_hand_image", "wrist_rgb"),
    ):
        frame = observation[source]
        if frame.shape != (224, 224, 3) or frame.dtype != np.uint8:
            raise ValueError("Invalid raw camera frame")
        result[target] = np.ascontiguousarray(frame[::-1])
    return result


def bounded_vector(vector, bound):
    length = np.linalg.norm(vector)
    return vector * min(1.0, bound / max(length, 1e-12))


class PushController:
    def __init__(self, scene, task, program, *, config=None):
        self.config = config or settings()
        self.fixed = self.config["teacher"]
        self.program = TeacherProgram.model_validate(program)
        self.program.validate_scene(scene, task)
        self.object_name = self.program.skills[0].object
        self.target = np.asarray(
            next(t.center_xy for t in scene.targets if t.name == task.destination)
        )
        self.index = self.elapsed = self.total_steps = 0
        self.finished = False
        self.settle_goal = None
        self.push_direction = None

    def command(self, observation, positions, *, table_height):
        from robosuite.utils.control_utils import orientation_error
        from robosuite.utils.transform_utils import quat2mat

        if self.finished:
            raise RuntimeError("Teacher program already ended")
        cfg = self.fixed
        skill = self.program.skills[self.index]
        eef = np.asarray(observation["robot0_eef_pos"])
        cube = positions[self.object_name]
        displacement = self.target - cube[:2]
        distance = np.linalg.norm(displacement)
        if self.push_direction is None:
            self.push_direction = displacement / max(distance, 1e-9)
        direction = self.push_direction
        behind = cube[:2] - direction * cfg["behind_distance_m"]
        contact = cube[:2] - direction * cfg["contact_distance_m"]
        name = skill.skill
        if name == "approach":
            goal = np.r_[behind, table_height + cfg["approach_height_m"]]
        elif name == "align_behind":
            goal = np.r_[behind, table_height + cfg["contact_height_m"]]
        elif name == "contact":
            goal = np.r_[contact, table_height + cfg["contact_height_m"]]
        elif name == "push_toward":
            goal = np.r_[
                contact
                + bounded_vector(
                    displacement, skill.speed_m_s / self.config["control_hz"]
                ),
                table_height + cfg["contact_height_m"],
            ]
        else:
            if self.settle_goal is None:
                self.settle_goal = eef.copy()
                self.settle_goal[:2] -= direction * cfg["release_distance_m"]
            goal = self.settle_goal

        reached = np.linalg.norm(goal - eef) < cfg["position_tolerance_m"]
        if name == "push_toward":
            reached = distance < cfg["target_center_tolerance_m"]
        elif name == "settle":
            reached = self.elapsed >= cfg["settle_steps"]
        if reached and self.index < 4:
            self.index += 1
            self.elapsed = 0
            return self.command(observation, positions, table_height=table_height)
        if self.elapsed >= skill.timeout_steps:
            raise TimeoutError(f"Teacher skill timed out: {name}")
        translation = bounded_vector(
            goal - eef, skill.speed_m_s / self.config["control_hz"]
        )
        yaw = np.arctan2(direction[1], direction[0])
        c, s = np.cos(yaw), np.sin(yaw)
        yaw_matrix = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        rotation = orientation_error(
            yaw_matrix @ quat2mat(np.asarray(cfg["desired_quaternion_xyzw"])),
            quat2mat(observation["robot0_eef_quat"]),
        )
        rotation = bounded_vector(rotation, cfg["rotation_limit_rad"])
        raw = np.r_[
            translation / cfg["translation_output_scale"],
            rotation / cfg["rotation_output_scale"],
            cfg["closed_gripper"],
        ]
        if not np.isfinite(raw).all():
            raise ValueError("Nonfinite teacher command")
        executed = np.clip(raw, -1, 1).astype(np.float32)
        self.elapsed += 1
        self.total_steps += 1
        self.finished = name == "settle" and reached
        return executed, {
            "skill": name,
            "raw_command": raw.tolist(),
            "saturated_scalars": int(np.count_nonzero(np.abs(raw) > 1)),
            "goal_xyz": goal.tolist(),
        }


def engineering_program(scene, task):
    """Hand-authored calibration fixture, never labeled Astra supervision."""
    if scene.stage == "S1":
        selected = scene.cubes[0].name
    elif scene.stage == "S2":
        selected = next(c.name for c in scene.cubes if c.color == task.referent)
    else:
        marker = next(f for f in scene.fixtures if f.name == "marker")
        selected = next(
            c.name
            for c in scene.cubes
            if (c.placement.center_xy[1] < marker.center_xy[1])
            == (task.referent == "left")
        )
    return TeacherProgram(
        schema_version="astrapush-1",
        skills=[
            {**s, "object": selected, "target": task.destination}
            for s in settings()["teacher"]["engineering_program"]
        ],
    )
