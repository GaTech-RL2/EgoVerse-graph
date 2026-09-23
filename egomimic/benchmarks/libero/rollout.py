"""Identical closed-loop protocol for any EgoVerse LIBERO PipelineAlgo."""

from __future__ import annotations

import hashlib
import json
import random
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from egomimic.benchmarks.libero.catalog import (
    LIBERO_COMMIT,
    OAT_COMMIT,
    TASK_IDS,
    TASKS,
    get_tasks,
)


@dataclass(frozen=True)
class RolloutSpec:
    task: str
    repetition: int
    trial: int
    seed: int


def rollout_plan(
    suite, trials_per_task=50, repetitions=5, start_seed=1000, *, repetition_index=None
):
    if min(trials_per_task, repetitions) < 1:
        raise ValueError("Positive trial and repetition counts are required")
    if repetition_index is not None and not 0 <= repetition_index < repetitions:
        raise ValueError("Evaluation repetition is outside the protocol")
    tasks = get_tasks(suite)
    return [
        RolloutSpec(
            task,
            repetition,
            trial,
            start_seed
            + (repetition * trials_per_task + trial) * len(tasks)
            + task_index,
        )
        for repetition in range(repetitions)
        for trial in range(trials_per_task)
        for task_index, task in enumerate(tasks)
        if repetition_index is None or repetition == repetition_index
    ]


class LiberoEnvironment:
    """OAT's ControlEnv observations/actions, with fresh post-settle reset data."""

    def __init__(self, task, image_size=128):
        # Keep simulator/GL dependencies optional for CPU codec and data tests.
        verify_libero_installation()
        from libero.libero import benchmark, get_libero_path
        from libero.libero.benchmark.libero_suite_task_map import libero_task_map
        from libero.libero.envs.env_wrapper import ControlEnv

        if libero_task_map != TASKS:
            raise RuntimeError(
                f"Install LIBERO at pinned commit {LIBERO_COMMIT}; task catalogs differ"
            )
        suite, local_id, self.uid = TASK_IDS[task]
        task_info = benchmark.get_benchmark_dict()[suite]().get_task(local_id)
        if task_info.name != task:
            raise RuntimeError("LIBERO task order differs from OAT")
        path = (
            Path(get_libero_path("bddl_files"))
            / task_info.problem_folder
            / task_info.bddl_file
        )
        self.env = ControlEnv(
            bddl_file_name=str(path),
            camera_names=["agentview", "robot0_eye_in_hand"],
            camera_heights=image_size,
            camera_widths=image_size,
            has_renderer=False,
            use_camera_obs=True,
            has_offscreen_renderer=True,
        )
        controller = self.env.env.robots[0].controller
        # ARC's delta bridge and OAT must see exactly the documented OSC scaling.
        if not np.allclose(controller.output_max[:6], [0.05] * 3 + [0.5] * 3):
            raise RuntimeError("Unexpected LIBERO OSC controller scaling")
        if not getattr(controller, "use_delta", True):
            raise RuntimeError("LIBERO requires delta OSC control")
        self.task = task

    def _observation(self, raw):
        result = {
            key: np.asarray(raw[key], dtype=np.float32)
            for key in ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")
        }
        for camera in ("agentview", "robot0_eye_in_hand"):
            result[f"{camera}_rgb"] = np.flip(raw[f"{camera}_image"], axis=0).copy()
        result["task_uid"] = np.asarray([self.uid], dtype=np.float32)
        return result

    def reset(self, seed):
        self.env.seed(seed)
        raw = self.env.reset()
        for _ in range(10):
            raw, _, _, _ = self.env.step(np.asarray([0.0] * 6 + [-1.0]))
        # Fetch after settling: the upstream README explicitly identifies stale
        # reset observations as a bug. The same correction applies to both methods.
        raw = self.env.env._get_observations(force_update=True)
        self.initial_state_sha256 = hashlib.sha256(
            np.asarray(self.env.sim.get_state().flatten()).tobytes()
        ).hexdigest()
        return self._observation(raw)

    def step(self, action):
        raw, _, terminated, _ = self.env.step(action)
        return self._observation(raw), bool(self.env.check_success()), bool(terminated)

    def close(self):
        self.env.close()


def verify_libero_installation():
    """Do not label an arbitrary LIBERO installation as the pinned benchmark."""
    import importlib.metadata
    import subprocess
    from urllib.parse import unquote, urlparse

    distribution = importlib.metadata.distribution("libero")
    direct = json.loads(distribution.read_text("direct_url.json") or "{}")
    commit = direct.get("vcs_info", {}).get("commit_id")
    if commit is None and direct.get("url", "").startswith("file:"):
        path = unquote(urlparse(direct["url"]).path)
        revision = subprocess.run(
            ["git", "-C", path, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        commit = revision.stdout.strip()
        clean = subprocess.run(
            ["git", "-C", path, "diff", "--quiet", "HEAD", "--"], check=False
        )
        if clean.returncode:
            raise RuntimeError("LIBERO checkout has tracked modifications")
    if commit != LIBERO_COMMIT:
        raise RuntimeError(
            f"LIBERO revision {commit!r}; install pinned revision {LIBERO_COMMIT}"
        )


class GraphPolicy:
    def __init__(self, algo, normalizer, n_obs_steps=2, n_action_steps=16, horizon=32):
        self.algo, self.normalizer = algo, normalizer
        self.n_obs_steps, self.n_action_steps, self.horizon = (
            n_obs_steps,
            n_action_steps,
            horizon,
        )
        self.history = deque(maxlen=n_obs_steps)
        self.algo.nets.eval()

    def reset(self, observation):
        self.history.clear()
        for _ in range(self.n_obs_steps):
            self.history.append(observation)

    def observe(self, observation):
        self.history.append(observation)

    @torch.inference_mode()
    def predict(self):
        from egomimic.rldb.zarr.libero_dataset import EMBODIMENT, OBS_KEYS

        batch = {
            key: torch.from_numpy(np.stack([obs[key] for obs in self.history]))
            .float()
            .unsqueeze(0)
            for key in OBS_KEYS
        }
        batch = self.normalizer.normalize(batch, EMBODIMENT)
        processed = self.algo.process_batch_for_training({"libero_panda": batch})
        result = self.algo.forward_eval(processed)["libero_panda"]["pred_action"]
        if result.shape != (1, self.horizon, 7):
            raise ValueError(f"Invalid policy action shape {tuple(result.shape)}")
        native = self.normalizer.unnormalize({"actions": result}, EMBODIMENT)["actions"]
        if not torch.isfinite(native).all():
            raise ValueError("Non-finite rollout actions")
        return native[0, : self.n_action_steps].cpu().numpy()


def load_policy(checkpoint, *, device="cuda", use_ema=True, use_k_tokens=None):
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    from egomimic.eval.checkpoint_loading import strict_load_pipeline_checkpoint
    from egomimic.rldb.zarr.libero_dataset import LiberoNormalizer

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = OmegaConf.create(payload["hyper_parameters"]["config_tree"])
    # A trained policy checkpoint includes frozen tokenizer weights. Build its
    # architecture from saved metadata, without needing the original tokenizer file.
    config = _self_contained_policy_config(config, payload)
    algo = instantiate(config.model.pipeline, device=device)
    normalizer = LiberoNormalizer(state=payload["normalizer_state"])
    normalizer.assert_tokenizer_context(payload["benchmark_data_context"])
    algo.bind_data_context(normalizer=normalizer)
    strict_load_pipeline_checkpoint(algo, payload, use_ema=use_ema)
    stages = list(algo.pipeline.stages)
    from egomimic.pipeline.stages_oat import OATPolicyStage

    oat = [stage for stage in stages if isinstance(stage, OATPolicyStage)]
    if use_k_tokens is not None:
        if len(oat) != 1:
            raise ValueError("OAT prefix control requires an OAT policy checkpoint")
        oat[0].use_k_tokens = use_k_tokens
    protocol = dict(config.model.benchmark_protocol)
    if protocol["suite"] != normalizer.context["suite"]:
        raise ValueError("Checkpoint protocol/data suite differs")
    return GraphPolicy(
        algo,
        normalizer,
        **{key: protocol[key] for key in ("n_obs_steps", "n_action_steps", "horizon")},
    ), protocol


def _self_contained_policy_config(config, payload):
    from egomimic.models.oat.checkpoint import policy_config_without_external_tokenizer

    return policy_config_without_external_tokenizer(config, payload)


def run_rollouts(
    policy,
    plan,
    output,
    *,
    max_episode_steps=550,
    env_factory=LiberoEnvironment,
    video_trials=0,
    metadata=None,
):
    """Stream every episode record; refuse existing output, never hide failures."""
    plan = list(plan)
    if not plan or max_episode_steps < 1 or video_trials < 0:
        raise ValueError("A nonempty plan and positive episode horizon are required")
    video_specs = set(plan[:video_trials])
    # Keep one environment per task through all trials, while preserving each
    # recorded seed and the balanced protocol plan. Rebuilding for every trial
    # would load the same simulator assets tens of thousands of times.
    execution_order = sorted(
        plan, key=lambda spec: (TASK_IDS[spec.task][2], spec.repetition, spec.trial)
    )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "protocol.json").write_text(
        json.dumps(
            {
                "oat_commit": OAT_COMMIT,
                "libero_commit": LIBERO_COMMIT,
                "max_episode_steps": max_episode_steps,
                "plan": [asdict(spec) for spec in plan],
                **(metadata or {}),
            },
            indent=2,
        )
        + "\n"
    )
    records, env, last_task = [], None, None
    try:
        with (output / "episodes.jsonl").open("x") as handle:
            for index, spec in enumerate(execution_order):
                random.seed(spec.seed)
                np.random.seed(spec.seed)
                torch.manual_seed(spec.seed)
                if env is None or spec.task != last_task:
                    if env is not None:
                        env.close()
                    env, last_task = env_factory(spec.task), spec.task
                observation = env.reset(spec.seed)
                policy.reset(observation)
                frames = []
                success, steps, latency = False, 0, []
                while steps < max_episode_steps:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    started = time.perf_counter()
                    actions = policy.predict()
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    latency.append(time.perf_counter() - started)
                    if (
                        len(actions) < 1
                        or actions.shape[-1] != 7
                        or not np.isfinite(actions).all()
                    ):
                        raise ValueError("Invalid action chunk in rollout")
                    terminated = False
                    for action in actions[: max_episode_steps - steps]:
                        observation, success, terminated = env.step(action)
                        policy.observe(observation)
                        steps += 1
                        if spec in video_specs:
                            frames.append(observation["agentview_rgb"])
                        if success or terminated:
                            break
                    if success or terminated:
                        break
                record = {
                    **asdict(spec),
                    "success": bool(success),
                    "steps": steps,
                    "initial_state_sha256": env.initial_state_sha256,
                    "inference_seconds": latency,
                }
                if frames:
                    import imageio.v2 as imageio

                    name = f"rollout_{index:06d}.mp4"
                    imageio.mimwrite(output / name, frames, fps=20)
                    record["video"] = name
                handle.write(json.dumps(record) + "\n")
                handle.flush()
                records.append(record)
                print(
                    "ROLLOUT_EPISODE "
                    + json.dumps(
                        {
                            "task": spec.task,
                            "repetition": spec.repetition,
                            "trial": spec.trial,
                            "success": bool(success),
                            "steps": steps,
                            "completed": len(records),
                            "planned": len(plan),
                        }
                    ),
                    flush=True,
                )
    finally:
        if env is not None:
            env.close()
    return records
