"""Ravens learned-policy preview using checkpoint-native inference and controller.

Simulation and Torch may use different Python ABIs. A private Unix socket carries
only typed arrays and commands; neural inference never receives oracle/object data.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import random
import subprocess
import time
from collections import deque
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def pack(arrays):
    import numpy as np

    result = {}
    for key, value in arrays.items():
        value = np.ascontiguousarray(value)
        if value.dtype not in (np.dtype("uint8"), np.dtype("float32")):
            raise ValueError("Transport permits only uint8 pixels and float32 state")
        result[key] = {
            "dtype": value.dtype.str,
            "shape": list(value.shape),
            "data": base64.b64encode(value.tobytes()).decode("ascii"),
        }
    return result


def unpack(arrays):
    import numpy as np

    result = {}
    for key, value in arrays.items():
        dtype = np.dtype(value["dtype"])
        if dtype not in (np.dtype("uint8"), np.dtype("float32")):
            raise ValueError("Unsupported transport dtype")
        raw = base64.b64decode(value["data"], validate=True)
        shape = tuple(value["shape"])
        if any(type(n) is not int or n < 0 for n in shape) or np.prod(
            shape
        ) * dtype.itemsize != len(raw):
            raise ValueError("Transport shape/byte mismatch")
        result[key] = np.frombuffer(raw, dtype=dtype).reshape(shape).copy()
    return result


def send(connection, message):
    connection.send_bytes(json.dumps(message, allow_nan=False).encode())


def receive(connection):
    if not connection.poll(300):
        raise TimeoutError("Ravens policy/simulator communication stalled")
    return json.loads(connection.recv_bytes(4 * 1024 * 1024))


def validate_scene_selection(request):
    """Keep seen-scene diagnostics separate from unseen-scene previews."""
    seeds = request["seeds"]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Preview requires unique scene seeds")
    if "policy_seed" in request and (
        type(request["policy_seed"]) is not int or request["policy_seed"] < 0
    ):
        raise ValueError("Policy RNG seed must be a nonnegative integer")
    scope = request.get("evaluation_split", "unseen")
    if scope == "unseen":
        if set(seeds) & set(request["collection_seeds"]):
            raise ValueError("Unseen preview seeds overlap the collection")
        return {seed: {"evaluation_split": scope} for seed in seeds}
    if scope != "train":
        raise ValueError("Unknown scene evaluation split")
    for key in ("dataset_manifest", "split_manifest"):
        if sha256(request[key]) != request[key + "_sha256"]:
            raise ValueError(f"Training scene {key} checksum mismatch")
    manifest = json.loads(Path(request["dataset_manifest"]).read_text())
    split = json.loads(Path(request["split_manifest"]).read_text())
    episodes = manifest["episodes"]
    train, valid = split["train"], split["valid"]
    if (
        not manifest.get("complete")
        or split["dataset_manifest_sha256"] != request["dataset_manifest_sha256"]
        or split["seed"] != 42
        or len(train) != 450
        or len(valid) != 50
        or set(train) & set(valid)
        or sorted(train + valid) != [f"episode_{i:06d}" for i in range(500)]
        or len(episodes) != 500
    ):
        raise ValueError("Training scene requires the exact saved450/50 split")
    recorded_seeds = [episode["seed"] for episode in episodes]
    if (
        len(set(recorded_seeds)) != 500
        or recorded_seeds != sorted(recorded_seeds)
        or recorded_seeds != request["collection_seeds"]
    ):
        raise ValueError("Training scene collection mapping mismatch")
    selected = {}
    for seed in seeds:
        if seed not in recorded_seeds:
            raise ValueError("Requested training scene was not recorded")
        index = recorded_seeds.index(seed)
        episode_id = f"episode_{index:06d}"
        if episode_id not in train:
            raise ValueError("Requested scene is validation, not training")
        selected[seed] = {
            "evaluation_split": scope,
            "episode_id": episode_id,
            "demonstration_sha256": episodes[index]["sha256"],
            "demonstration_frames": episodes[index]["frames"],
            "demonstration_reward": episodes[index]["reward"],
        }
    return selected


def validate_request(request):
    if request["protocol"] != "ravens-continuous-policy-preview-v1":
        raise ValueError("Unknown rollout contract")
    validate_scene_selection(request)
    if not 1 <= request["max_actions"] <= 4000:
        raise ValueError("Continuous controller budget must be at most4000")
    if request["precision"] != "bf16-mixed" or request["use_ema"] is not True:
        raise ValueError("This preview requires EMA and BF16 neural autocast")
    if sha256(request["checkpoint"]) != request["checkpoint_sha256"]:
        raise ValueError("Checkpoint checksum mismatch")
    source = Path(__file__).resolve().parents[3]
    head = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if (
        head != request["driver_commit"]
        or subprocess.check_output(
            ["git", "-C", str(source), "status", "--porcelain"], text=True
        ).strip()
    ):
        raise ValueError("Driver requires its exact clean committed source")
    subprocess.run(
        [
            "git",
            "-C",
            str(source),
            "merge-base",
            "--is-ancestor",
            request["model_commit"],
            head,
        ],
        check=True,
    )
    # Only routing/rollout code may differ; model, sampler, normalizer and
    # continuous controller implementations remain checkpoint-bound byte-for-byte.
    changed = subprocess.check_output(
        [
            "git",
            "-C",
            str(source),
            "diff",
            "--name-only",
            request["model_commit"],
            head,
        ],
        text=True,
    ).splitlines()
    allowed = {
        "egomimic/benchmarks/libero/rollout.py",
        "egomimic/benchmarks/ravens/rollout.py",
        "tests/test_ravens_rollout.py",
    }
    if not set(changed).issubset(allowed):
        raise ValueError("Rollout driver changed checkpoint-bound scientific source")
    return source


def resolve_protocol(saved_config, training_config):
    """AF checkpoints save model-only metadata; bind benchmark fields separately."""
    from omegaconf import OmegaConf

    saved = OmegaConf.to_container(saved_config.model, resolve=True)
    trained = OmegaConf.to_container(training_config.model, resolve=True)
    if saved != trained:
        raise ValueError("Training configuration model differs from checkpoint")
    protocol = {
        key: training_config.benchmark[key]
        for key in ("suite", "horizon", "n_obs_steps", "n_action_steps")
    }
    expected = {
        "suite": "ravens-continuous-v1",
        "horizon": 32,
        "n_obs_steps": 2,
        "n_action_steps": 16,
    }
    if protocol != expected:
        raise ValueError("Unexpected trained action/history contract")
    embedded = OmegaConf.select(saved_config, "model.benchmark_protocol")
    if embedded is not None and any(
        embedded.get(key) != value for key, value in protocol.items()
    ):
        raise ValueError("Embedded benchmark protocol conflicts with training")
    return protocol


def load_policy(request, *, device="cuda"):
    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    from egomimic.benchmarks.libero.rollout import GraphPolicy
    from egomimic.benchmarks.ravens.replay import RavensNormalizer
    from egomimic.eval.checkpoint_loading import strict_load_pipeline_checkpoint

    if device == "cuda" and (
        not torch.cuda.is_available() or torch.cuda.device_count() != 1
    ):
        raise RuntimeError("Exactly one allocated CUDA GPU is required")
    if device == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 inference unsupported")
    payload = torch.load(request["checkpoint"], map_location="cpu", weights_only=False)
    if (
        payload["global_step"] != request["checkpoint_step"]
        or payload["ema_num_updates"] != payload["global_step"]
    ):
        raise ValueError("Checkpoint/EMA optimizer step mismatch")
    config = OmegaConf.create(payload["hyper_parameters"]["config_tree"])
    if sha256(request["training_config"]) != request["training_config_sha256"]:
        raise ValueError("Training configuration checksum mismatch")
    full_config = OmegaConf.load(request["training_config"])
    protocol = resolve_protocol(config, full_config)
    algo = instantiate(config.model.pipeline, device=device)
    normalizer = RavensNormalizer(state=payload["normalizer_state"])
    normalizer.assert_tokenizer_context(payload["benchmark_data_context"])
    if request.get("evaluation_split") == "train":
        if (
            sha256(request["training_normalization"])
            != request["training_normalization_sha256"]
        ):
            raise ValueError("Training normalization checksum mismatch")
        native = torch.load(
            request["training_normalization"], map_location="cpu", weights_only=False
        )
        if native["dataset_manifest_sha256"] != request[
            "dataset_manifest_sha256"
        ] or native["split"] != json.loads(Path(request["split_manifest"]).read_text()):
            raise ValueError("Training scene differs from saved normalization dataset")
        normalizer.assert_tokenizer_context(
            RavensNormalizer(state=native["normalizer_state"]).tokenizer_context()
        )
    if (
        normalizer.context.get("suite") != protocol["suite"]
        or normalizer.context.get("action_contract")
        != "world_absolute_xyz_rotvec_suction_20hz"
    ):
        raise ValueError("Normalizer does not describe native Ravens absolute actions")
    algo.bind_data_context(normalizer=normalizer)
    strict_load_pipeline_checkpoint(algo, payload, use_ema=True)
    policy = GraphPolicy(
        algo,
        normalizer,
        n_obs_steps=2,
        n_action_steps=16,
        horizon=32,
        embodiment="ravens_ur5",
    )
    del payload
    return policy, protocol


def serve(request, output):
    from multiprocessing.connection import Listener

    import torch

    validate_request(request)
    policy, protocol = load_policy(request)
    socket = Path(request["socket"])
    if socket.exists():
        raise FileExistsError(socket)
    with Listener(str(socket), family="AF_UNIX") as listener:
        socket.chmod(0o600)
        write(
            output / "MODEL_READY.json",
            {
                "strict_model_and_ema_reload": "PASS",
                "protocol": protocol,
                "checkpoint_step": request["checkpoint_step"],
                "precision": "bf16-mixed",
                "parameters_dtype": str(next(policy.algo.nets.parameters()).dtype),
            },
        )
        with listener.accept() as connection:
            while True:
                message = receive(connection)
                if message["command"] == "close":
                    send(connection, {"status": "closed"})
                    break
                if message["command"] != "predict":
                    raise ValueError("Unknown policy request")
                if message.get("seed") is not None:
                    random.seed(message["seed"])
                    torch.manual_seed(message["seed"])
                    import numpy as np

                    np.random.seed(message["seed"])
                history = [unpack(row) for row in message["history"]]
                if len(history) != 2:
                    raise ValueError("Require exactly two causal observations")
                policy.reset(history[0])
                policy.observe(history[1])
                started = time.perf_counter()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    actions = policy.predict()
                send(
                    connection,
                    {
                        "actions": pack({"actions": actions.astype("float32")}),
                        "seconds": time.perf_counter() - started,
                    },
                )
    # Listener.close() normally removes its Unix socket already.
    socket.unlink(missing_ok=True)


def simulate(request, output):
    from multiprocessing.connection import Client

    import cv2
    import numpy as np

    from egomimic.benchmarks.ravens.continuous import make_environment

    scene_selection = validate_scene_selection(request)
    env = make_environment(
        request["cliport_source"],
        task_module="place_red_in_green",
        task_class="PlaceRedInGreen",
        control_hz=20,
        record=False,
    )
    aliases = {
        "front_rgb": "agentview_rgb",
        "side_rgb": "robot0_eye_in_hand_rgb",
        "eef_pos": "robot0_eef_pos",
        "eef_quat": "robot0_eef_quat",
        "suction_state": "robot0_gripper_qpos",
        "task_uid": "task_uid",
    }

    def observe():
        raw = env.policy_observation()
        return {alias: raw[key] for key, alias in aliases.items()}

    records = []
    try:
        with Client(request["socket"], family="AF_UNIX") as connection:
            with (output / "episodes.jsonl").open("x") as handle:
                for seed in request["seeds"]:
                    env.seeded_reset(seed)
                    history = deque([observe()] * 2, maxlen=2)
                    name = f"seed-{seed}.mp4"
                    camera = env.agent_cams[0]
                    height, width = camera["image_size"]
                    writer = cv2.VideoWriter(
                        str(output / name),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        20,
                        (width, height),
                    )
                    if not writer.isOpened():
                        raise RuntimeError("Video encoder unavailable")
                    trace = []
                    latencies = []
                    reward = 0.0
                    first = True
                    frames = 0
                    started = time.perf_counter()
                    try:
                        while (
                            env.action_count < request["max_actions"] and reward < 0.999
                        ):
                            send(
                                connection,
                                {
                                    "command": "predict",
                                    "seed": request.get("policy_seed", seed)
                                    if first
                                    else None,
                                    "history": [pack(row) for row in history],
                                },
                            )
                            first = False
                            reply = receive(connection)
                            actions = unpack(reply["actions"])["actions"]
                            if (
                                actions.shape != (16, 7)
                                or not np.isfinite(actions).all()
                            ):
                                raise ValueError(
                                    "Invalid finite16x7 native action chunk"
                                )
                            latencies.append(reply["seconds"])
                            for action in actions[
                                : request["max_actions"] - env.action_count
                            ]:
                                env.control(action)
                                history.append(observe())
                                # Like native Ravens.step, score only settled objects;
                                # require released suction so hovering a held block cannot
                                # count as placement under native XY-only pose matching.
                                if (
                                    not env.ee.activated
                                    and env.is_static
                                    and env.task.goals
                                ):
                                    delta, _ = env.task.reward()
                                    reward += float(delta)
                                rng = env._random.get_state()
                                image = env.render_camera(camera)[0]
                                env._random.set_state(rng)
                                writer.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                                frames += 1
                                trace.append(action.copy())
                                if reward >= 0.999:
                                    break
                    finally:
                        writer.release()
                    record = {
                        "seed": seed,
                        "policy_seed": request.get("policy_seed", seed),
                        **scene_selection[seed],
                        "success": reward >= 0.999,
                        "reward": reward,
                        "actions": env.action_count,
                        "video": name,
                        "frames": frames,
                        "inference_seconds": latencies,
                        "wall_seconds": time.perf_counter() - started,
                    }
                    np.savez_compressed(
                        output / f"seed-{seed}-actions.npz", actions=np.asarray(trace)
                    )
                    handle.write(json.dumps(record, allow_nan=False) + "\n")
                    handle.flush()
                    records.append(record)
                    print("ROLLOUT_EPISODE " + json.dumps(record), flush=True)
            send(connection, {"command": "close"})
            receive(connection)
    finally:
        env.close()
    write(
        output / "SUMMARY.json",
        {
            "status": "COMPLETE_PREVIEW",
            "episodes": len(records),
            "successes": sum(r["success"] for r in records),
            "records": records,
            "request": request,
            "metric": "Native task reward>=.999, released suction and static scene; preview, not paper benchmark",
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=["preflight", "preflight-model", "serve", "simulate"],
        required=True,
    )
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    output = Path(request["output"])
    if args.mode == "preflight":
        validate_request(request)
        if output.exists():
            raise FileExistsError(output)
        print(json.dumps({"status": "READY_METADATA", "request": request}), flush=True)
    elif args.mode == "preflight-model":
        validate_request(request)
        _, protocol = load_policy(request, device="cpu")
        write(
            args.request.parent / "MODEL_PREFLIGHT.json",
            {
                "status": "PASS_STRICT_MODEL_EMA_AND_PROTOCOL",
                "protocol": protocol,
                "checkpoint_step": request["checkpoint_step"],
            },
        )
    elif args.mode == "serve":
        serve(request, output)
    else:
        simulate(request, output)


if __name__ == "__main__":
    main()
