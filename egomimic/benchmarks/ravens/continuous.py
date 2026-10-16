"""Record and replay continuous actions through one controller.

The native oracle supplies demonstrations only. A policy never receives object
poses, segmentation, oracle waypoints or the pick/place primitive. Actions are
absolute world-frame tip XYZ, rotation vector and suction (+1 on, -1 off).
This is a custom continuous-control adaptation, not the native CLIPort metric.
"""
from __future__ import annotations

import argparse
import faulthandler
import fcntl
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import random
import subprocess
import sys
import types

import numpy as np

UPSTREAM_COMMIT = "2be5c47b5bb9bb7040ad90693288b87b1e18e7ad"


def defer_dependency(name, submodules=()):
    """Defer optional training/UI imports while retaining their actual APIs on use."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.find_spec(name)
    module = types.ModuleType(name)
    if spec and spec.submodule_search_locations:
        module.__path__ = list(spec.submodule_search_locations)
    loaded = None

    def resolve(attribute):
        nonlocal loaded
        if attribute.startswith("__"):
            raise AttributeError(attribute)
        if attribute in submodules:
            return importlib.import_module(f"{name}.{attribute}")
        if loaded is None:
            current = sys.modules.get(name)
            if current is not None and current is not module:
                loaded = current
            else:
                sys.modules.pop(name, None)
                loaded = importlib.import_module(name)
        return getattr(loaded, attribute)

    module.__getattr__ = resolve
    sys.modules[name] = module
    return module


def load_simulator(source, task_module, task_class):
    """Load only upstream simulator modules, avoiding language-agent imports."""
    source = Path(source).resolve()
    actual = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != UPSTREAM_COMMIT:
        raise ValueError(f"Unexpected CLIPort revision: {actual}")
    dirty = subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"], text=True
    ).strip()
    if dirty:
        raise ValueError("CLIPort source has tracked modifications")
    # The upstream utility module also serves neural agents and a live meshcat
    # viewer. Neither is used by the simulator/oracle, so avoid multi-GB eager
    # Torch imports on shared storage. Actual attribute use loads the real API.
    deferred = {name: defer_dependency(name) for name in ("torch", "kornia")}
    defer_dependency("meshcat", submodules=("geometry", "transformations"))
    # Upstream __init__ eagerly imports unrelated CLIP/agent training packages.
    # Namespace packages preserve the exact simulator files without patching them.
    for name in ("cliport", "cliport.tasks", "cliport.utils", "cliport.environments"):
        if name in sys.modules:
            raise RuntimeError(f"Simulator must be loaded once in a fresh process: {name}")
        package = types.ModuleType(name)
        package.__path__ = [str(source.joinpath(*name.split(".")))]
        package.__package__ = name
        sys.modules[name] = package
    env_class = importlib.import_module("cliport.environments.environment").Environment
    selected_class = getattr(importlib.import_module(f"cliport.tasks.{task_module}"), task_class)
    # SciPy's optional array-library detection must not mistake unloaded Torch
    # for an active tensor backend. Upstream utils retains the deferred object.
    for name, proxy in deferred.items():
        if sys.modules.get(name) is proxy and "__getattr__" in vars(proxy):
            sys.modules.pop(name)
    return env_class, selected_class


def make_environment(source, *, task_module, task_class, control_hz=20, record=True):
    import cv2
    import pybullet as p
    from scipy.spatial.transform import Rotation

    env_class, selected_class = load_simulator(source, task_module, task_class)
    if control_hz < 1 or 240 % control_hz:
        raise ValueError("Control frequency must divide 240 Hz physics")

    class ContinuousEnvironment(env_class):
        def __init__(self):
            self.samples = []
            self.record_actions = False
            self.max_actions = 4000
            self.action_count = 0
            super().__init__(str(Path(source) / "cliport/environments/assets"),
                             task=selected_class(), hz=240)

        def tip_pose(self):
            state = p.getLinkState(self.ur5, self.ee_tip, computeForwardKinematics=True)
            return np.asarray(state[4]), np.asarray(state[5])

        def policy_observation(self):
            xyz, quat = self.tip_pose()
            images = []
            for camera in self.agent_cams[:2]:
                # Same FOV/aspect/extrinsics at the model's input resolution.
                # Keep full-resolution oracle rendering completely independent.
                small = dict(camera)
                scale = 128 / camera["image_size"][1]
                small["image_size"] = (round(camera["image_size"][0] * scale), 128)
                intrinsics = np.asarray(camera["intrinsics"]).reshape(3, 3).copy()
                intrinsics[:2] *= scale
                small["intrinsics"] = tuple(intrinsics.reshape(-1))
                image = self.render_camera(small)[0]
                images.append(cv2.resize(image, (128, 128), interpolation=cv2.INTER_AREA))
            return {"front_rgb": images[0], "side_rgb": images[1],
                    "eef_pos": xyz.astype(np.float32),
                    "eef_quat": quat.astype(np.float32),
                    "suction_state": np.array([float(self.ee.activated), 0], np.float32),
                    "task_uid": np.array([0], np.float32)}

        def control(self, action):
            action = np.asarray(action, dtype=np.float64)
            if action.shape != (7,) or not np.isfinite(action).all():
                raise ValueError("Action must contain seven finite values")
            if self.action_count >= self.max_actions:
                raise RuntimeError("Continuous episode action budget exceeded")
            observation = self.policy_observation() if self.record_actions else None
            if action[6] > 0:
                self.raw_activate()
            else:
                self.raw_release()
            pose = (action[:3], Rotation.from_rotvec(action[3:6]).as_quat())
            joints = self.solve_ik(pose)
            p.setJointMotorControlArray(self.ur5, self.joints, p.POSITION_CONTROL,
                                       targetPositions=joints, positionGains=[1.] * len(self.joints))
            for _ in range(240 // control_hz):
                self.step_simulation()
            if self.record_actions:
                self.samples.append({**observation, "action": action.astype(np.float32)})
            self.action_count += 1
            if self.action_count % 100 == 0:
                print(json.dumps({"phase": "record" if self.record_actions else "replay",
                                  "control_steps": self.action_count}), flush=True)

        def movep(self, pose, speed=.01):
            target_pos = np.asarray(pose[0])
            target_rot = Rotation.from_quat(pose[1])
            for index in range(400):
                xyz, quat = self.tip_pose()
                difference = target_pos - xyz
                current_rot = Rotation.from_quat(quat)
                rotvec = (target_rot * current_rot.inv()).as_rotvec()
                if index and np.linalg.norm(difference) < .001 and np.linalg.norm(rotvec) < .03:
                    return False
                # Bounded oracle targets; exactly the same control() drives replay/policies.
                step = difference * min(1., max(float(speed), .003) / max(np.linalg.norm(difference), 1e-12))
                rotation = Rotation.from_rotvec(rotvec * min(1., .15 / max(np.linalg.norm(rotvec), 1e-12))) * current_rot
                action = np.r_[xyz + step, rotation.as_rotvec(), 1. if self.ee.activated else -1.]
                self.control(action)
            xyz, quat = self.tip_pose()
            print(json.dumps({"phase": "motion_timeout", "target_xyz": target_pos.tolist(),
                              "actual_xyz": xyz.tolist(),
                              "position_error": float(np.linalg.norm(target_pos - xyz)),
                              "rotation_error": float(np.linalg.norm(
                                  (target_rot * Rotation.from_quat(quat).inv()).as_rotvec()))}), flush=True)
            return True

        def seeded_reset(self, seed):
            random.seed(seed)
            np.random.seed(seed)
            self.seed(seed)
            self.record_actions = False
            self.action_count = 0
            self.samples = []
            result = super().reset()
            self.raw_activate, self.raw_release = self.ee.activate, self.ee.release
            # Native primitives change suction between movep calls. Capture that
            # decision BEFORE it changes the observed gripper state; otherwise
            # targets leak into observations and the policy never learns grasp/release.
            def suction_action(enabled):
                xyz, quat = self.tip_pose()
                self.control(np.r_[xyz, Rotation.from_quat(quat).as_rotvec(),
                                   1. if enabled else -1.])
            self.ee.activate = lambda: suction_action(True)
            self.ee.release = lambda: suction_action(False)
            self.record_actions = record
            return result

        def close(self):
            p.disconnect()

    return ContinuousEnvironment()


def collect_episode(env, seed):
    observation = env.seeded_reset(seed)
    oracle = env.task.oracle(env)
    total_reward = 0.
    for _ in range(env.task.max_steps):
        action = oracle.act(observation, env.info)
        if action is None:
            break
        observation, reward, done, _ = env.step(action)
        total_reward += reward
        if done:
            break
    # done() can also mean oracle failure, so require complete reward explicitly.
    return total_reward >= .999, total_reward


def save_episode(path, samples):
    arrays = {key: np.stack([sample[key] for sample in samples]) for key in samples[0]}
    validate_suction_transitions(arrays["suction_state"], arrays["action"])
    np.savez_compressed(path, **arrays)
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_suction_transitions(states, actions):
    """Reject label leakage: both grasp and release must have pre-action observations."""
    states, actions = np.asarray(states), np.asarray(actions)
    if states.shape != (len(actions), 2) or actions.shape != (len(states), 7):
        raise ValueError("Invalid action/state shapes")
    if not np.isfinite(states).all() or not np.isfinite(actions).all():
        raise ValueError("Non-finite action/state")
    on = actions[:, 6] > 0
    was_on = states[:, 0] > .5
    if not np.any(on & ~was_on) or not np.any(~on & was_on):
        raise ValueError("Missing pre-action suction transitions; action may leak into observation")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cliport-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--control-hz", type=int, default=20)
    parser.add_argument("--task-module", required=True)
    parser.add_argument("--task-class", required=True)
    parser.add_argument("--replay-proof", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    # Slow network assets or a controller stall must be distinguishable from
    # healthy collection. This is diagnostic only; no RNG/control changes.
    faulthandler.enable()
    faulthandler.dump_traceback_later(180, repeat=True)
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / ".collector.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists() and not args.resume:
        raise FileExistsError("Output already has a manifest; preserve previous collection")
    print(json.dumps({"phase": "construct_environment", "source": str(args.cliport_source)}), flush=True)
    env = make_environment(args.cliport_source, task_module=args.task_module,
                           task_class=args.task_class, control_hz=args.control_hz)
    print(json.dumps({"phase": "environment_ready"}), flush=True)
    manifest = {"upstream_commit": UPSTREAM_COMMIT,
                "task": {"module": args.task_module, "class": args.task_class},
                "contract": "custom continuous absolute XYZ/rotvec/suction",
                "control_hz": args.control_hz, "physics_hz": 240,
                "policy_inputs": "two RGB images, tip pose, suction state; constant task_uid=0",
                "oracle_privileged_inputs": "demonstration generation only",
                "collector_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "complete": False,
                "episodes": [], "failures": []}
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        for key in ("upstream_commit", "task", "contract", "control_hz", "collector_sha256"):
            if existing.get(key) != manifest[key]:
                raise ValueError(f"Resume contract changed: {key}")
        manifest = existing
    try:
        used_seeds = [e["seed"] for e in manifest["episodes"] + manifest["failures"]]
        seed = max(used_seeds, default=args.start_seed - 1) + 1
        while len(manifest["episodes"]) < args.episodes:
            if seed >= args.start_seed + max(10, args.episodes * 5):
                raise RuntimeError("Too many failed oracle demonstrations")
            try:
                print(json.dumps({"phase": "begin_episode", "seed": seed}), flush=True)
                success, reward = collect_episode(env, seed)
            except RuntimeError as exc:
                success, reward = False, 0.
                manifest["failures"].append({"seed": seed, "error": str(exc)})
            if success:
                recorded = env.samples
                actions = np.stack([sample["action"] for sample in env.samples])
                replay = None
                if args.replay_proof:
                    env.seeded_reset(seed)
                    env.record_actions = False
                    for action in actions:
                        env.control(action)
                    replay_reward, _ = env.task.reward()
                    replay = {"reward": float(replay_reward), "success": replay_reward >= .999,
                              "actions": len(actions)}
                    if not replay["success"]:
                        raise RuntimeError(f"Same-controller replay failed: {replay}")
                    # Reset replaced samples; replay proof must preserve the original recording.
                    samples = recorded
                else:
                    samples = env.samples
                path = args.output / f"episode-{seed:06d}.npz"
                sha = save_episode(path, samples)
                manifest["episodes"].append({"seed": seed, "file": path.name, "sha256": sha,
                                              "frames": len(actions), "reward": reward, "replay": replay})
            else:
                manifest["failures"].append({"seed": seed, "reward": reward})
            manifest["complete"] = len(manifest["episodes"]) == args.episodes
            temporary = manifest_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(manifest, indent=2) + "\n")
            temporary.replace(manifest_path)
            print(json.dumps({"seed": seed, "success": success, "collected": len(manifest["episodes"])}), flush=True)
            seed += 1
    finally:
        env.close()
        lock.close()


if __name__ == "__main__":
    main()
