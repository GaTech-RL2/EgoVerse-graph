"""Shared Eva/Yam rollout: local graph inference or recorded Zarr joint replay."""

import argparse
import time
from collections import deque

import numpy as np
from omegaconf import OmegaConf

from egomimic.robot.cameras import CameraView
from egomimic.robot.interface import ARM_OFFSET, create_robot, joint_vector


def load_policy(config):
    config = dict(config)
    kind = config.pop("kind")
    if kind == "graph":
        from egomimic.robot.graph_policy import load_graph_policy

        return load_graph_policy(config)
    if kind == "zarr_replay":
        from egomimic.robot.replay_policy import ZarrReplayPolicy

        return ZarrReplayPolicy(**config)
    raise ValueError(
        "Use kind=graph for inference or kind=zarr_replay for recorded actions"
    )


def run_rollout(robot, policy, config, view=None):
    frequency, max_steps = float(config["frequency"]), int(config["max_steps"])
    execute_steps = int(config["execute_steps"])
    limit = float(config["max_joint_velocity"]) / frequency
    if min(frequency, max_steps, execute_steps, limit) <= 0 or not np.isfinite(limit):
        raise ValueError("Rollout frequency, step counts and velocity must be positive")
    if policy.action_type not in ("joints", "cartesian"):
        raise ValueError("Unknown policy action representation")
    queue, last, step = deque(), None, 0
    waiting_since = None
    view = view or CameraView(robot.camera_res, **config["preview"])
    try:
        while step < max_steps:
            tick = time.monotonic()
            obs = robot.get_obs()
            if view.update(obs) in ("q", "\x1b"):
                break
            if not all(obs.get(name) is not None for name in robot.camera_res):
                waiting_since = tick if waiting_since is None else waiting_since
                if tick - waiting_since > float(config.get("camera_timeout", 30.0)):
                    raise TimeoutError("Rollout cameras are unavailable")
                # Resume from a fresh plan after camera loss; do not execute stale queued actions.
                queue.clear()
                time.sleep(1 / frequency)
                continue
            waiting_since = None
            if last is None:
                last = np.asarray(obs["joint_positions"], dtype=float).copy()
            if not queue:
                try:
                    prediction = np.asarray(policy.predict(obs), dtype=float)
                except StopIteration:
                    break
                if (
                    prediction.ndim != 2
                    or prediction.shape[1] != 14
                    or not len(prediction)
                    or not np.isfinite(prediction).all()
                ):
                    raise ValueError(
                        "Policy must return a nonempty finite (H, 14) action chunk"
                    )
                # Replay consumes its entire chunk; graph plans replan at execute_steps.
                count = (
                    len(prediction)
                    if policy.action_type == "joints"
                    else min(execute_steps, len(prediction))
                )
                queue.extend(prediction[:count])
            row = queue.popleft()
            commands = {}
            for arm in robot.arms:
                offset = ARM_OFFSET[arm]
                target = row[offset : offset + 7]
                if policy.action_type == "cartesian":
                    target = np.r_[robot.solve_ik(target[:6], arm), target[6]]
                command = joint_vector(target)
                if (
                    np.max(np.abs(command[:6] - last[offset : offset + 6]))
                    > limit + 1e-8
                ):
                    raise ValueError(
                        f"{arm} target exceeds the configured joint velocity. Align replay start / check graph calibration before rollout."
                    )
                commands[arm] = command
            # Validate both arms before sending either command.
            for arm, command in commands.items():
                robot.set_joints(command, arm)
                offset = ARM_OFFSET[arm]
                last[offset : offset + 7] = command
            step += 1
            time.sleep(max(0.0, 1 / frequency - (time.monotonic() - tick)))
    finally:
        view.close()
    return step


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--inference-mode", choices=("graph",), default="graph")
    args = parser.parse_args()
    config = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    # Load/validate policy artifacts before opening hardware.
    policy = load_policy(config["policy"])
    robot = create_robot(config["robot"])
    try:
        run_rollout(robot, policy, config)
    except KeyboardInterrupt:
        pass
    finally:
        robot.close()


if __name__ == "__main__":
    main()
