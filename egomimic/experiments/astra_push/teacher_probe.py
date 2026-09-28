"""Measured engineering teacher rollout; never counted as generated supervision."""

import argparse
import json
import time
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


def probe(output, *, stage="S1", seed=17):
    from egomimic.experiments.astra_push.libero_scene import (
        compile_scene,
        load_environment,
    )

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    scene, task = starter(stage)
    program = engineering_program(scene, task)
    cfg = settings()
    started = time.monotonic()
    publish_json(
        output / "attempt.json",
        {
            "category": "engineering",
            "attempt": 1,
            "seed": seed,
            "stage": stage,
            "counts_as_commissioning": False,
        },
    )
    bundle = compile_scene(scene, task, output / "bundle")
    env = load_environment(bundle, seed=seed)
    rows, actions, audit = [], [], []
    evaluator = None
    failure = None
    try:
        observation = env.env.reset()
        for _ in range(cfg["reset_settle_steps"]):
            observation, _, _, _ = env.step(
                np.r_[np.zeros(6), cfg["teacher"]["closed_gripper"]]
            )
        positions, rotations = cube_poses(env, scene)
        state = output / "initial-full-state"
        state_hash = snapshot(env, state)
        evaluator = SuccessEvaluator(scene, task, positions)
        controller = PushController(scene, task, program)
        publish_json(output / "teacher.json", program.model_dump(mode="json"))
        for t in range(cfg["horizon"]):
            positions, rotations = cube_poses(env, scene)
            action, control = controller.command(
                observation, positions, table_height=env.env.table_offset[2]
            )
            rows.append(student_observation(observation))
            actions.append(action)
            contact = [
                {"geoms": [int(c.geom1), int(c.geom2)], "distance": float(c.dist)}
                for c in env.sim.data.contact[: env.sim.data.ncon]
            ]
            audit.append(
                {
                    **control,
                    "step": t,
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
            positions, rotations = cube_poses(env, scene)
            metrics = evaluator.update(positions, rotations)
            if metrics["lift_violation"] or metrics["preservation_violation"]:
                failure = "trajectory_constraint_violation"
                break
            if controller.finished:
                break
    except (TimeoutError, ValueError, RuntimeError) as exc:
        failure = f"{type(exc).__name__}: {exc}"
    finally:
        env.close()
    metrics = evaluator.result() if evaluator else {"success": False}
    accepted = bool(metrics["success"] and failure is None)
    receipt = {
        "category": "engineering_teacher",
        "stage": stage,
        "seed": seed,
        "teacher_authorship": "hand_authored_fixture",
        "commissioning_accepted": 0,
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
                "episode_id": f"engineering_{stage}_{seed}",
                "scene_hash": canonical_hash(scene),
                "task_hash": canonical_hash(task),
                "teacher_hash": canonical_hash(program),
                "initial_state_hash": state_hash,
                "phase": "engineering",
                "arm": "common",
                "round": 0,
                "stage": stage,
                "partition": "engineering",
                "accepted": accepted,
            },
            audit={"steps": audit, "metrics": metrics, "failure": failure},
        )
        receipt["episode"] = record
        with imageio.get_writer(
            output / "preview.mp4",
            fps=cfg["control_hz"],
            codec="libx264",
            quality=7,
            macro_block_size=None,
        ) as video:
            for row in rows:
                video.append_data(
                    np.concatenate([row["external_rgb"], row["wrist_rgb"]], axis=1)
                )
        publish_json(output / "control-trace.json", audit)
    publish_json(output / "receipt.json", receipt)
    print(json.dumps(receipt))
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stage", choices=["S1", "S2", "S3"], default="S1")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    probe(args.output, stage=args.stage, seed=args.seed)
