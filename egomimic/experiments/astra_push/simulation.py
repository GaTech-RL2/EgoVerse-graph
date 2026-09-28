"""The sole simulator worker: immutable banks, teachers and policy rollouts."""

import json
import socket
import sys
import time
import traceback
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image

from egomimic.experiments.astra_push.artifacts import (
    file_hash,
    named_seed,
    publish_json,
)
from egomimic.experiments.astra_push.controller import (
    cube_poses,
    settings,
    student_observation,
)
from egomimic.experiments.astra_push.libero_scene import compile_scene, load_environment
from egomimic.experiments.astra_push.partitions import config
from egomimic.experiments.astra_push.schemas import (
    SceneSpec,
    TaskSpec,
    TeacherProgram,
    canonical_hash,
)
from egomimic.experiments.astra_push.semantics import SuccessEvaluator
from egomimic.experiments.astra_push.sim_state import restore, snapshot
from egomimic.experiments.astra_push.teacher_probe import run_teacher_attempt
from egomimic.experiments.astra_push.transport import (
    receive_frame,
    send_frame,
    validate_actions,
)


def freeze_bank(definitions_path, output):
    output = Path(output)
    receipt_path = output / "manifest.json"
    if receipt_path.exists():
        result = json.loads(receipt_path.read_text())
        if result["definitions_sha256"] != file_hash(definitions_path):
            raise ValueError("Frozen bank definitions changed")
        for case in result["cases"]:
            if (
                file_hash(Path(case["snapshot"]) / "snapshot.json")
                != case["initial_state_sha256"]
            ):
                raise ValueError("Frozen reset changed")
        return result
    definitions = json.loads(Path(definitions_path).read_text())
    output.mkdir(parents=True, exist_ok=True)
    cases = []
    for row in definitions["templates"]:
        scene = SceneSpec.model_validate(row["scene"])
        task = TaskSpec.model_validate(row["tasks"][0])
        bundle = output / row["id"] / "bundle"
        if not bundle.exists():
            compile_scene(scene, task, bundle)
        for i, seed in enumerate(row["state_seeds"]):
            root = output / row["id"] / f"state-{i}"
            root.mkdir(parents=True, exist_ok=True)
            state_path = root / "initial-full-state"
            env = load_environment(bundle, seed=seed)
            try:
                observation = env.env.reset()
                for _ in range(settings()["reset_settle_steps"]):
                    observation, _, _, _ = env.step(
                        np.r_[np.zeros(6), settings()["teacher"]["closed_gripper"]]
                    )
                if not state_path.exists():
                    snapshot(env, state_path)
                else:
                    observation = restore(env, state_path)
                initial, _ = cube_poses(env, scene)
                for raw_task in row["tasks"]:
                    SuccessEvaluator(scene, TaskSpec.model_validate(raw_task), initial)
                before = student_observation(observation)
                restored = student_observation(restore(env, state_path))
                if any(not np.array_equal(before[k], restored[k]) for k in before):
                    raise RuntimeError("Frozen reset observation changed after restore")
                if not (root / "preview.png").exists():
                    Image.fromarray(
                        np.concatenate(
                            [before["external_rgb"], before["wrist_rgb"]], axis=1
                        )
                    ).save(root / "preview.png")
                state_hash = file_hash(state_path / "snapshot.json")
                which = (
                    [row["control_instruction_indices"][i]]
                    if row["partition"] == "training-control"
                    else range(2)
                )
                for side in which:
                    case_id = f"{row['id']}_state{i}_instruction{side}"
                    cases.append(
                        {
                            "id": case_id,
                            "template_id": row["id"],
                            "partition": row["partition"],
                            "stage": row["stage"],
                            "pair_id": f"{row['id']}_state{i}",
                            "state_index": i,
                            "instruction_index": side,
                            "scene": row["scene"],
                            "task": row["tasks"][side],
                            "bundle": str(bundle),
                            "snapshot": str(state_path),
                            "initial_state_sha256": state_hash,
                            "reset_seed": seed,
                            "policy_seed": named_seed(f"policy:{row['id']}:{i}"),
                            "family_hash": row["family_hash"],
                        }
                    )
            finally:
                env.close()
            print(
                json.dumps(
                    {"event": "frozen_state", "template": row["id"], "state": i}
                ),
                flush=True,
            )
    counts = {
        p: sum(c["partition"] == p for c in cases)
        for p in definitions["counts_per_checkpoint"]
    }
    if counts != definitions["counts_per_checkpoint"]:
        raise ValueError("Frozen partition sizes differ")
    result = {
        "version": config()["version"],
        "definitions_sha256": file_hash(definitions_path),
        "cases": cases,
        "counts": counts,
    }
    publish_json(receipt_path, result)
    return result


def evaluate_bank(bank_path, output, *, partition, checkpoint_hash, policy_socket):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    bank = json.loads(Path(bank_path).read_text())
    cases = [c for c in bank["cases"] if c["partition"] == partition]
    outcomes = []
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(config()["policy_timeout_seconds"])
        connection.connect(policy_socket)
        for case in cases:
            root = output / case["id"]
            receipt_path = root / "receipt.json"
            if receipt_path.exists():
                saved = json.loads(receipt_path.read_text())
                if saved["checkpoint_hash"] != checkpoint_hash or saved[
                    "case_hash"
                ] != canonical_hash(case):
                    raise ValueError("Evaluation resume changed checkpoint or instance")
                outcomes.append(saved)
                continue
            root.mkdir(parents=True, exist_ok=True)
            attempt = len(list(root.glob("attempt-*.json")))
            publish_json(
                root / f"attempt-{attempt:03d}.json",
                {
                    "checkpoint_hash": checkpoint_hash,
                    "case_hash": canonical_hash(case),
                    "time": time.time(),
                    "reason": "initial"
                    if attempt == 0
                    else "recover_interrupted_rollout",
                    "earlier_attempts_retained": attempt,
                },
            )
            env = load_environment(case["bundle"], seed=case["reset_seed"])
            actions, raw_actions, traces, frames, video_frames = [], [], [], [], []
            begin = time.monotonic()
            try:
                env.env.reset()
                observation = restore(env, case["snapshot"])
                scene, task = (
                    SceneSpec.model_validate(case["scene"]),
                    TaskSpec.model_validate(case["task"]),
                )
                positions, rotations = cube_poses(env, scene)
                evaluator = SuccessEvaluator(scene, task, positions)
                initial_distance = float(
                    np.linalg.norm(
                        positions[evaluator.requested][:2] - evaluator.target.center_xy
                    )
                )
                minimum_distance = initial_distance
                # Fixed representative video selection, independent of success.
                save_video = (
                    partition != "training-control"
                    and case["state_index"] == 0
                    and int(case["template_id"].rsplit("_", 1)[1]) < 2
                )
                for step in range(settings()["horizon"]):
                    row = student_observation(observation)
                    row["proprioception"] = np.asarray(
                        row["proprioception"], dtype=np.float32
                    )
                    if step in config()["evaluation"]["frame_steps"]:
                        frames.append(
                            np.concatenate(
                                [row["external_rgb"], row["wrist_rgb"]], axis=1
                            )
                        )
                    if save_video:
                        video_frames.append(
                            np.concatenate(
                                [row["external_rgb"], row["wrist_rgb"]], axis=1
                            )
                        )
                    request_id = f"{case['id']}:{attempt}:{step}"
                    send_frame(
                        connection,
                        {
                            "version": 1,
                            "kind": "observation",
                            "request_id": request_id,
                            "episode_id": f"{case['id']}:{attempt}",
                            "step": step,
                            "timestamp": step / 10,
                            "policy_seed": named_seed(f"{case['policy_seed']}:{step}"),
                            "instruction": task.instruction,
                        },
                        row,
                    )
                    metadata, array = receive_frame(connection)
                    predicted = validate_actions(metadata, array, request_id)
                    raw = predicted[0].copy()
                    action = np.clip(raw, -1, 1).astype(np.float32)
                    observation, _, _, _ = env.step(action)
                    raw_actions.append(raw)
                    actions.append(action)
                    positions, rotations = cube_poses(env, scene)
                    metrics = evaluator.update(positions, rotations)
                    distance = float(
                        np.linalg.norm(
                            positions[evaluator.requested][:2]
                            - evaluator.target.center_xy
                        )
                    )
                    minimum_distance = min(minimum_distance, distance)
                    traces.append(
                        {
                            "step": step,
                            "positions": {k: v.tolist() for k, v in positions.items()},
                            "rotations": {k: v.tolist() for k, v in rotations.items()},
                            "distance": distance,
                            "metrics": metrics,
                        }
                    )
                    if (
                        metrics["success"]
                        or metrics["lift_violation"]
                        or metrics["preservation_violation"]
                    ):
                        break
                final_frame = student_observation(observation)
                frames.append(
                    np.concatenate(
                        [final_frame["external_rgb"], final_frame["wrist_rgb"]], axis=1
                    )
                )
                metrics = evaluator.result()
            finally:
                env.close()
            with (root / f"actions-{attempt}.npz").open("xb") as stream:
                np.savez_compressed(
                    stream, executed=np.stack(actions), raw=np.stack(raw_actions)
                )
            publish_json(root / f"trace-{attempt}.json", traces)
            frame_path = root / f"frames-{attempt}.png"
            Image.fromarray(np.concatenate(frames, axis=0)).save(frame_path)
            video_path = None
            if video_frames:
                video_path = root / f"preview-{attempt}.mp4"
                with imageio.get_writer(
                    video_path,
                    fps=10,
                    codec="libx264",
                    quality=7,
                    macro_block_size=None,
                ) as writer:
                    for frame in video_frames:
                        writer.append_data(frame)
            result = {
                "case_id": case["id"],
                "case_hash": canonical_hash(case),
                "template_id": case["template_id"],
                "pair_id": case["pair_id"],
                "instruction_index": case["instruction_index"],
                "stage": case["stage"],
                "partition": partition,
                "checkpoint_hash": checkpoint_hash,
                "initial_state_sha256": case["initial_state_sha256"],
                "policy_seed": case["policy_seed"],
                "metrics": metrics,
                "initial_distance_m": initial_distance,
                "minimum_distance_m": minimum_distance,
                "final_distance_m": distance,
                "displacement_failure": not metrics["success"]
                and minimum_distance > evaluator.target.half_width,
                "overshoot_failure": not metrics["success"]
                and metrics["max_hold_steps"] > 0,
                "steps": len(actions),
                "clipped_scalars": int(
                    np.count_nonzero(np.stack(actions) != np.stack(raw_actions))
                ),
                "seconds": time.monotonic() - begin,
                "attempts": attempt + 1,
                "frames_path": str(frame_path),
                "frames_sha256": file_hash(frame_path),
                "video_path": str(video_path) if video_path else None,
                "actions_sha256": file_hash(root / f"actions-{attempt}.npz"),
            }
            publish_json(receipt_path, result)
            outcomes.append(result)
            print(
                json.dumps(
                    {
                        "event": "policy_rollout",
                        "partition": partition,
                        "case": case["id"],
                        "success": metrics["success"],
                        "steps": len(actions),
                        "seconds": result["seconds"],
                    }
                ),
                flush=True,
            )
    return {
        "partition": partition,
        "checkpoint_hash": checkpoint_hash,
        "bank_hash": file_hash(bank_path),
        "episodes": outcomes,
    }


def worker():
    for line in sys.stdin:
        request = json.loads(line)
        try:
            arguments = request["arguments"]
            if request["operation"] == "freeze":
                value = freeze_bank(**arguments)
            elif request["operation"] == "teacher":
                arguments["scene"] = SceneSpec.model_validate(arguments["scene"])
                arguments["task"] = TaskSpec.model_validate(arguments["task"])
                arguments["program"] = TeacherProgram.model_validate(
                    arguments["program"]
                )
                value = run_teacher_attempt(**arguments)
            elif request["operation"] == "evaluate":
                value = evaluate_bank(**arguments)
            else:
                raise ValueError("Unknown worker operation")
            response = {"id": request["id"], "ok": True, "value": value}
        except Exception:
            response = {
                "id": request["id"],
                "ok": False,
                "error": traceback.format_exc(),
            }
        print("ASTRA_WORKER " + json.dumps(response, allow_nan=False), flush=True)


if __name__ == "__main__":
    worker()
