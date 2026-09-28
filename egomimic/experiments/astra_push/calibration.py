"""Bounded physical engineering checks, distinct from commissioning attempts."""

import argparse
import hashlib
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from egomimic.experiments.astra_push.artifacts import named_seed, publish_json
from egomimic.experiments.astra_push.controller import (
    cube_poses,
    settings,
    student_observation,
)
from egomimic.experiments.astra_push.render_probe import starter
from egomimic.experiments.astra_push.sim_state import restore, snapshot


def replay_digest(environment, actions):
    states, images = [], []
    for action in actions:
        observation, _, _, _ = environment.step(action)
        states.append(environment.sim.get_state().flatten().copy())
        frame = student_observation(observation)
        images.append(
            [
                hashlib.sha256(frame[k].tobytes()).hexdigest()
                for k in ("external_rgb", "wrist_rgb")
            ]
        )
    return np.stack(states), images


def verify_fresh_restore(bundle, state, expected, output):
    from egomimic.experiments.astra_push.libero_scene import load_environment

    reference = json.loads(Path(expected).read_text())
    env = load_environment(bundle)
    try:
        env.env.reset()
        restore(env, state)
        states, images = replay_digest(env, np.asarray(reference["actions"]))
        maximum = float(np.max(np.abs(states - reference["states"])))
        assert maximum < 1e-10 and images == reference["images"], (
            maximum,
            images == reference["images"],
        )
        publish_json(
            output,
            {
                "fresh_process_restore_passed": True,
                "maximum_state_error": maximum,
                "camera_frames_identical": True,
            },
        )
    finally:
        env.close()


def calibrate(output):
    from robosuite.utils.camera_utils import (
        get_camera_transform_matrix,
        project_points_from_world_to_camera,
    )

    from egomimic.experiments.astra_push.libero_scene import (
        compile_scene,
        load_environment,
    )

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    cfg = settings()
    records = []
    for stage in ("S1", "S2", "S3"):
        scene, task = starter(stage)
        bundle = compile_scene(scene, task, output / stage / "bundle")
        env = load_environment(bundle)
        try:
            for index in range(20):
                seed = named_seed(f"engineering_reset:{stage}:{index}")
                record = {
                    "stage": stage,
                    "index": index,
                    "seed": seed,
                    "category": "engineering_reset",
                }
                publish_json(output / stage / f"attempt-{index:02d}.json", record)
                try:
                    np.random.seed(seed)
                    random.seed(seed)
                    obs = env.env.reset()
                    for _ in range(cfg["reset_settle_steps"]):
                        obs, _, _, _ = env.step(np.r_[np.zeros(6), 1.0])
                    first, _ = cube_poses(env, scene)
                    for _ in range(5):
                        obs, _, _, _ = env.step(np.r_[np.zeros(6), 1.0])
                    second, _ = cube_poses(env, scene)
                    drift = max(
                        float(np.linalg.norm(first[k] - second[k])) for k in first
                    )
                    valid = student_observation(obs)
                    assert all(
                        valid[k].std() > 1 for k in ("external_rgb", "wrist_rgb")
                    )
                    assert drift <= cfg["reset_max_drift_m"]
                    record.update(status="passed", maximum_drift_m=drift)
                except Exception as exc:
                    record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                publish_json(output / stage / f"reset-{index:02d}.json", record)
                records.append(record)

            # Only run comparisons after a verified reset, with immutable state.
            if records[-1]["status"] != "passed":
                continue
            state_path = output / stage / "full-state"
            state_hash = snapshot(env, state_path)
            original = student_observation(obs)
            actions = np.zeros((8, 7), np.float32)
            actions[:, 6] = 1
            actions[:4, 1] = 0.1
            actions[4:, 0] = 0.1

            def trajectory():
                result = []
                for action in actions:
                    observation, _, _, _ = env.step(action)
                    result.append(
                        (
                            env.sim.get_state().flatten().copy(),
                            student_observation(observation),
                        )
                    )
                return result

            first = trajectory()
            publish_json(
                output / stage / "expected-replay.json",
                {
                    "actions": actions.tolist(),
                    "states": [r[0].tolist() for r in first],
                    "images": [
                        [
                            hashlib.sha256(r[1][k].tobytes()).hexdigest()
                            for k in ("external_rgb", "wrist_rgb")
                        ]
                        for r in first
                    ],
                },
            )
            replay_obs = student_observation(restore(env, state_path))
            assert all(np.array_equal(original[k], replay_obs[k]) for k in original)
            second = trajectory()
            error = max(
                float(np.max(np.abs(a[0] - b[0]))) for a, b in zip(first, second)
            )
            images_equal = all(
                np.array_equal(a[1][k], b[1][k])
                for a, b in zip(first, second)
                for k in ("external_rgb", "wrist_rgb")
            )
            assert error < 1e-10 and images_equal, (error, images_equal)
            restore(env, state_path)
            controller = env.robots[0].controller
            assert np.allclose(controller.output_max, [0.05, 0.05, 0.05, 0.5, 0.5, 0.5])
            # Three independent world-frame translations from identical state.
            displacements = []
            for axis in range(3):
                before = restore(env, state_path)["robot0_eef_pos"].copy()
                action = np.zeros(7)
                action[axis] = 0.1
                action[6] = 1
                for _ in range(5):
                    after, _, _, _ = env.step(action)
                delta = after["robot0_eef_pos"] - before
                assert delta[axis] > 0.001 and np.argmax(np.abs(delta)) == axis
                displacements.append(delta.tolist())
            restore(env, state_path)
            for _ in range(10):
                opened, _, _, _ = env.step(np.r_[np.zeros(6), -1.0])
            open_width = float(np.ptp(opened["robot0_gripper_qpos"]))
            for _ in range(10):
                closed, _, _, _ = env.step(np.r_[np.zeros(6), 1.0])
            closed_width = float(np.ptp(closed["robot0_gripper_qpos"]))
            assert open_width > closed_width + 0.02
            observation = restore(env, state_path)
            world = np.array([[0, -0.12, 0.92], [0, 0.12, 0.92]])
            matrix = get_camera_transform_matrix(env.sim, "agentview", 224, 224)
            pixels = project_points_from_world_to_camera(world, matrix, 224, 224)
            assert pixels[0, 1] < pixels[1, 1], "Negative Y must render to camera-left"
            preview = Image.fromarray(student_observation(observation)["external_rgb"])
            draw = ImageDraw.Draw(preview)
            for label, (y, x) in zip(("-Y left", "+Y right"), pixels):
                draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill="yellow")
                draw.text((x - 22, y + 5), label, fill="black")
            preview.save(output / stage / "frame-calibration.png")
            publish_json(
                output / stage / "calibration.json",
                {
                    "state_sha256": state_hash,
                    "restored_state_max_error": error,
                    "restored_images_equal": images_equal,
                    "positive_axis_displacements": displacements,
                    "open_width_m": open_width,
                    "closed_width_m": closed_width,
                    "closed_command": 1.0,
                    "negative_y_is_camera_left": True,
                    "projected_labels_yx": pixels.tolist(),
                },
            )
        finally:
            env.close()
        subprocess.run(
            [
                sys.executable,
                "-m",
                "egomimic.experiments.astra_push.calibration",
                "--restore-bundle",
                str(bundle),
                "--state",
                str(state_path),
                "--expected",
                str(output / stage / "expected-replay.json"),
                "--output",
                str(output / stage / "fresh-restore.json"),
            ],
            check=True,
            timeout=60,
        )
    receipt = {
        "category": "engineering_calibration",
        "reset_attempts": 60,
        "resets_passed": sum(r["status"] == "passed" for r in records),
        "reset_records": records,
        "seconds": time.monotonic() - started,
        "counts_as_commissioning": False,
    }
    publish_json(output / "receipt.json", receipt)
    print(json.dumps({k: v for k, v in receipt.items() if k != "reset_records"}))
    if receipt["resets_passed"] != 60:
        raise RuntimeError("Reset stability gate failed; no retries hidden")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--restore-bundle")
    parser.add_argument("--state")
    parser.add_argument("--expected")
    args = parser.parse_args()
    if args.restore_bundle:
        verify_fresh_restore(
            args.restore_bundle, args.state, args.expected, args.output
        )
    else:
        calibrate(args.output)
