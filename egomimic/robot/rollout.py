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


def validate_rollout_config(config):
    """Validate browser-only rollout settings before creating robot hardware."""
    preview = dict(config.get("preview", {}))
    if preview.get("mode") != "dashboard":
        return
    robot = config.get("robot")
    if not isinstance(robot, dict):
        raise ValueError("Dashboard rollout configuration needs a robot mapping")
    cameras = robot.get("cameras")
    if not isinstance(cameras, dict) or not cameras:
        raise ValueError("Dashboard rollout configuration needs configured cameras")
    from egomimic.robot.rollout_dashboard import validate_rollout_preview

    validate_rollout_preview(preview, cameras=set(cameras))


def create_preview_view(camera_res, preview):
    """Select the legacy OpenCV preview or the local browser dashboard."""
    preview = dict(preview)
    if preview.get("mode") == "dashboard":
        from egomimic.robot.rollout_dashboard import RolloutDashboard

        return RolloutDashboard(camera_res, **preview)
    return CameraView(camera_res, **preview)


def _set_view_status(view, status):
    set_status = getattr(view, "set_status", None)
    if callable(set_status):
        set_status(status)


def run_rollout(robot, policy, config, view=None):
    frequency, max_steps = float(config["frequency"]), int(config["max_steps"])
    execute_steps = int(config["execute_steps"])
    limit = float(config["max_joint_velocity"]) / frequency
    max_velocity_replans = config.get("max_velocity_replans", 0)
    if min(frequency, max_steps, execute_steps, limit) <= 0 or not np.isfinite(limit):
        raise ValueError("Rollout frequency, step counts and velocity must be positive")
    if type(max_velocity_replans) is not int or not 0 <= max_velocity_replans <= 32:
        raise ValueError("max_velocity_replans must be an integer in [0, 32]")
    if policy.action_type not in ("joints", "cartesian"):
        raise ValueError("Unknown policy action representation")
    queue, last, step = deque(), None, 0
    waiting_since, velocity_replans = None, 0
    view = view or create_preview_view(robot.camera_res, config["preview"])
    wait_for_start = bool(config.get("preview", {}).get("wait_for_start", False))
    started = not wait_for_start
    if wait_for_start:
        _set_view_status(view, "Ready — press c to start")
    try:
        while step < max_steps:
            tick = time.monotonic()
            obs = robot.get_obs()
            control = view.update(obs)
            if control in ("q", "\x1b"):
                break
            if control in ("r", "R"):
                # Restart never reuses a queued target. It returns to the
                # explicit ready gate and sends no command until c is pressed.
                queue.clear()
                last, step, waiting_since, velocity_replans, started = (
                    None,
                    0,
                    None,
                    0,
                    False,
                )
                clear_plan = getattr(view, "clear_action_plan", None)
                if callable(clear_plan):
                    clear_plan()
                _set_view_status(view, "Restarted — press c to start")
                time.sleep(max(0.0, 1 / frequency - (time.monotonic() - tick)))
                continue
            if not started:
                if control in ("c", "C"):
                    started = True
                    _set_view_status(view, "Running")
                else:
                    time.sleep(max(0.0, 1 / frequency - (time.monotonic() - tick)))
                    continue
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
                set_plan = getattr(view, "set_action_plan", None)
                if callable(set_plan):
                    # Browser overlays are display-only. The unchanged command queue
                    # below remains the sole source of robot actuation.
                    set_plan(prediction, policy.action_type)
                # Replay consumes its entire chunk; graph plans replan at execute_steps.
                count = (
                    len(prediction)
                    if policy.action_type == "joints"
                    else min(execute_steps, len(prediction))
                )
                queue.extend(prediction[:count])
            row = queue.popleft()
            commands = {}
            violation = None
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
                    violation = arm
                    break
                commands[arm] = command
            if violation is not None:
                # The entire sampled chunk is unsafe from the current pose.
                # Do not send either arm; re-observe and ask a stochastic policy
                # for a fresh plan instead of terminating on its first outlier.
                queue.clear()
                last = np.asarray(obs["joint_positions"], dtype=float).copy()
                velocity_replans += 1
                if velocity_replans > max_velocity_replans:
                    raise ValueError(
                        f"{violation} target exceeds the configured joint velocity "
                        f"after {max_velocity_replans} safe replan attempt(s). "
                        "No unsafe command was sent."
                    )
                _set_view_status(
                    view,
                    f"Rejected velocity-unsafe plan; resampling "
                    f"{velocity_replans}/{max_velocity_replans}",
                )
                time.sleep(max(0.0, 1 / frequency - (time.monotonic() - tick)))
                continue
            # Validate both arms before sending either command.
            for arm, command in commands.items():
                robot.set_joints(command, arm)
                offset = ARM_OFFSET[arm]
                last[offset : offset + 7] = command
            velocity_replans = 0
            step += 1
            time.sleep(max(0.0, 1 / frequency - (time.monotonic() - tick)))
    finally:
        view.close()
    return step


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--inference-mode", choices=("graph",), default="graph")
    parser.add_argument(
        "--check-config",
        action="store_true",
        help="validate dashboard/calibration configuration only; open no devices",
    )
    args = parser.parse_args()
    config = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    validate_rollout_config(config)
    if args.check_config:
        print(
            "Rollout configuration is valid; no robot, CAN, or camera device was opened."
        )
        return
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
