"""Measured teacher execution shared by engineering and generated commissioning."""

import argparse
import hashlib
import json
import time
import traceback
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from egomimic.experiments.astra_push.artifacts import file_hash, publish_json
from egomimic.experiments.astra_push.controller import (
    CONFIG_PATH,
    PushController,
    cube_poses,
    engineering_program,
    settings,
    student_observation,
)
from egomimic.experiments.astra_push.episodes import write_episode
from egomimic.experiments.astra_push.render_probe import starter
from egomimic.experiments.astra_push.schemas import canonical_hash
from egomimic.experiments.astra_push.semantics import SuccessEvaluator
from egomimic.experiments.astra_push.sim_state import snapshot


def run_teacher_attempt(
    output,
    *,
    scene,
    task,
    program,
    seed,
    episode_id,
    phase,
    teacher_authorship,
    arm="common",
    round_index=0,
    video=True,
):
    from egomimic.experiments.astra_push.libero_scene import (
        compile_scene,
        load_environment,
    )

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    if phase not in {"engineering", "commissioning", "seed", "round"}:
        raise ValueError("Unknown acquisition purpose")
    task.validate_scene(scene)
    program.validate_scene(scene, task)
    cfg = settings()
    started = time.monotonic()
    publish_json(
        output / "attempt.json",
        {
            "category": phase,
            "attempt_id": episode_id,
            "seed": seed,
            "stage": scene.stage,
            "counts_as_commissioning": phase == "commissioning",
        },
    )
    env = None
    rows, actions, audit, replay = [], [], [], []
    evaluator = None
    failure = None
    try:
        bundle = compile_scene(scene, task, output / "bundle")
        env = load_environment(bundle, seed=seed)
        observation = env.env.reset()
        for _ in range(cfg["reset_settle_steps"]):
            observation, _, _, _ = env.step(
                np.r_[np.zeros(6), cfg["teacher"]["closed_gripper"]]
            )
        positions, rotations = cube_poses(env, scene)
        state = output / "initial-full-state"
        state_hash = snapshot(env, state)
        with (bundle / "resolved.xml").open("x") as stream:
            stream.write(env.sim.model.get_xml())
        evaluator = SuccessEvaluator(scene, task, positions)
        controller = PushController(scene, task, program)
        publish_json(output / "teacher.json", program.model_dump(mode="json"))
        for t in range(cfg["horizon"]):
            positions, rotations = cube_poses(env, scene)
            action, control = controller.command(
                observation, positions, table_height=env.env.table_offset[2]
            )
            student_row = student_observation(observation)
            contact = [
                {"geoms": [int(c.geom1), int(c.geom2)], "distance": float(c.dist)}
                for c in env.sim.data.contact[: env.sim.data.ncon]
            ]
            audit.append(
                {
                    **control,
                    "step": t,
                    "executed": False,
                    "cube_positions": {k: v.tolist() for k, v in positions.items()},
                    "eef_xyz": observation["robot0_eef_pos"].tolist(),
                    "contacts": contact,
                    "controller_goal_pos": env.robots[0].controller.goal_pos.tolist(),
                    "controller_goal_ori": env.robots[0].controller.goal_ori.tolist(),
                    "gripper_current_action": env.robots[
                        0
                    ].gripper.current_action.tolist(),
                    "torques": env.robots[0].torques.tolist(),
                }
            )
            observation, _, _, _ = env.step(action)
            audit[-1]["executed"] = True
            rows.append(student_row)
            actions.append(action)
            if t < 8:
                frame = student_observation(observation)
                replay.append(
                    {
                        "state": env.sim.get_state().flatten().tolist(),
                        "images": [
                            hashlib.sha256(frame[k].tobytes()).hexdigest()
                            for k in ("external_rgb", "wrist_rgb")
                        ],
                    }
                )
            positions, rotations = cube_poses(env, scene)
            metrics = evaluator.update(positions, rotations)
            if metrics["lift_violation"] or metrics["preservation_violation"]:
                failure = "trajectory_constraint_violation"
                break
            if controller.finished:
                break
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
        (output / "failure-traceback.txt").write_text(traceback.format_exc())
    finally:
        if env is not None:
            env.close()
    metrics = evaluator.result() if evaluator else {"success": False}
    accepted = bool(metrics["success"] and failure is None)
    receipt = {
        "category": f"{phase}_teacher",
        "stage": scene.stage,
        "seed": seed,
        "teacher_authorship": teacher_authorship,
        "commissioning_accepted": int(phase == "commissioning" and accepted),
        "accepted": accepted,
        "metrics": metrics,
        "failure": failure,
        "executed_steps": len(actions),
        "seconds": time.monotonic() - started,
        "controller_config_sha256": file_hash(CONFIG_PATH),
        "controller_source_sha256": file_hash(
            Path(__file__).with_name("controller.py")
        ),
        "full_state_captured": bool(evaluator),
    }
    if rows:
        observations = {key: np.stack([r[key] for r in rows]) for key in rows[0]}
        observations["timestamps"] = (
            np.arange(len(rows), dtype=np.float64) / cfg["control_hz"]
        )
        record = write_episode(
            output / "episode.hdf5",
            observations=observations,
            actions=np.stack(actions),
            instruction=task.instruction,
            provenance={
                "episode_id": episode_id,
                "scene_hash": canonical_hash(scene),
                "task_hash": canonical_hash(task),
                "teacher_hash": canonical_hash(program),
                "initial_state_hash": state_hash,
                "phase": phase,
                "arm": arm,
                "round": round_index,
                "stage": scene.stage,
                "partition": "engineering" if phase == "engineering" else "training",
                "accepted": accepted,
            },
            audit={"steps": audit, "metrics": metrics, "failure": failure},
        )
        receipt["episode"] = record
        if video:
            with imageio.get_writer(
                output / "preview.mp4",
                fps=cfg["control_hz"],
                codec="libx264",
                quality=7,
                macro_block_size=None,
            ) as writer:
                for row in rows:
                    writer.append_data(
                        np.concatenate([row["external_rgb"], row["wrist_rgb"]], axis=1)
                    )
        publish_json(output / "control-trace.json", audit)
        if len(replay) == 8:
            publish_json(
                output / "expected-replay.json",
                {
                    "actions": np.stack(actions[:8]).tolist(),
                    "states": [r["state"] for r in replay],
                    "images": [r["images"] for r in replay],
                },
            )
    publish_json(output / "receipt.json", receipt)
    return receipt


def probe(output, *, stage="S1", seed=17):
    scene, task = starter(stage)
    receipt = run_teacher_attempt(
        output,
        scene=scene,
        task=task,
        program=engineering_program(scene, task),
        seed=seed,
        episode_id=f"engineering_{stage}_{seed}",
        phase="engineering",
        teacher_authorship="hand_authored_fixture",
    )
    print(json.dumps(receipt))
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stage", choices=["S1", "S2", "S3"], default="S1")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    probe(args.output, stage=args.stage, seed=args.seed)
