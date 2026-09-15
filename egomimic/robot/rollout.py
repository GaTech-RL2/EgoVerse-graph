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
    for name in ("reset_on_start", "reset_home_on_restart"):
        value = config.get(name, False)
        if type(value) is not bool:
            raise ValueError(f"{name} must be a boolean")
    execute_steps = config.get("execute_steps")
    if type(execute_steps) is not int or not 1 <= execute_steps <= 100:
        raise ValueError("execute_steps must be an integer in [1, 100]")
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


def create_preview_view(camera_res, preview, execute_steps=None):
    """Select the legacy OpenCV preview or the local browser dashboard."""
    preview = dict(preview)
    if preview.get("mode") == "dashboard":
        from egomimic.robot.rollout_dashboard import RolloutDashboard

        return RolloutDashboard(camera_res, execute_steps=execute_steps, **preview)
    return CameraView(camera_res, **preview)


def _set_view_status(view, status):
    set_status = getattr(view, "set_status", None)
    if callable(set_status):
        set_status(status)


def _velocity_decision(view, details):
    """Resolve an explicit response to a whole-plan velocity-limit rejection."""
    choose = getattr(view, "choose_velocity_action", None)
    if not callable(choose):
        return "resample"
    decision = choose(details)
    if decision not in {"execute", "resample", "restart", "stop"}:
        raise ValueError(f"Unknown velocity-limit decision: {decision!r}")
    return decision


def run_rollout(robot, policy, config, view=None):
    frequency, max_steps = float(config["frequency"]), int(config["max_steps"])
    execute_steps = config["execute_steps"]
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
    paused = False
    view = view or create_preview_view(
        robot.camera_res, config["preview"], execute_steps=execute_steps
    )
    reset_on_start = config.get("reset_on_start", False)
    reset_home_on_restart = config.get("reset_home_on_restart", False)
    wait_for_start = bool(config.get("preview", {}).get("wait_for_start", False))
    started = not wait_for_start

    def reset_to_ready():
        nonlocal last, step, waiting_since, velocity_replans, started, paused
        queue.clear()
        last, step, waiting_since, velocity_replans, started = (
            None,
            0,
            None,
            0,
            False,
        )
        paused = False
        clear_plan = getattr(view, "clear_action_plan", None)
        if callable(clear_plan):
            clear_plan()
        if reset_home_on_restart:
            _set_view_status(view, "Resetting YAM to configured home")
            robot.set_home()
        _set_view_status(view, "Ready — press c to start")

    try:
        if reset_on_start:
            _set_view_status(view, "Resetting YAM to configured home")
            robot.set_home()
        if wait_for_start:
            _set_view_status(view, "Ready — press c to start")
        while step < max_steps:
            tick = time.monotonic()
            obs = robot.get_obs()
            control = view.update(obs)
            if control in ("q", "\x1b"):
                break
            if control in ("r", "R"):
                # Restart never reuses a queued target. The HPT profile also
                # returns both followers to their configured home before c can
                # begin the next policy rollout.
                reset_to_ready()
                time.sleep(max(0.0, 1 / frequency - (time.monotonic() - tick)))
                continue
            is_paused = getattr(view, "is_paused", None)
            requested_pause = bool(is_paused()) if callable(is_paused) else False
            if requested_pause != paused:
                queue.clear()
                clear_plan = getattr(view, "clear_action_plan", None)
                if callable(clear_plan):
                    clear_plan()
                if requested_pause:
                    # Replace any previously commanded target with the measured
                    # pose once. This holds both followers without advancing the
                    # policy queue or retaining an unsafe stale plan.
                    last = np.asarray(obs["joint_positions"], dtype=float).copy()
                    for arm in robot.arms:
                        offset = ARM_OFFSET[arm]
                        robot.set_joints(last[offset : offset + 7], arm)
                    _set_view_status(view, "Paused — holding current joint positions")
                else:
                    # A resumed rollout always infers from a fresh observation.
                    last = None
                    _set_view_status(view, "Running")
                paused = requested_pause
            if paused:
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
                    inference_started = time.perf_counter()
                    prediction = np.asarray(policy.predict(obs), dtype=float)
                    inference_seconds = time.perf_counter() - inference_started
                except StopIteration:
                    break
                record_inference = getattr(view, "record_inference", None)
                if callable(record_inference):
                    record_inference(inference_seconds)
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
                # Replay consumes its entire chunk; graph plans replan at the
                # dashboard-selected interval (or config default).
                get_execute_steps = getattr(view, "get_execute_steps", None)
                plan_steps = (
                    get_execute_steps()
                    if callable(get_execute_steps)
                    else execute_steps
                )
                if type(plan_steps) is not int or not 1 <= plan_steps <= 100:
                    raise ValueError(
                        "Dashboard execute steps must be an integer in [1, 100]"
                    )
                count = (
                    len(prediction)
                    if policy.action_type == "joints"
                    else min(plan_steps, len(prediction))
                )
                queue.extend(prediction[:count])
            row = queue.popleft()
            commands = {}
            violations = []
            for arm in robot.arms:
                offset = ARM_OFFSET[arm]
                target = row[offset : offset + 7]
                if policy.action_type == "cartesian":
                    target = np.r_[robot.solve_ik(target[:6], arm), target[6]]
                command = joint_vector(target)
                joint_step = float(
                    np.max(np.abs(command[:6] - last[offset : offset + 6]))
                )
                if joint_step > limit + 1e-8:
                    violations.append((arm, joint_step))
                commands[arm] = command
            if violations:
                details = {
                    "arms": [arm for arm, _ in violations],
                    "max_joint_step": max(step_size for _, step_size in violations),
                    "limit": limit,
                }
                decision = _velocity_decision(view, details)
                if decision == "stop":
                    break
                if decision == "restart":
                    reset_to_ready()
                    time.sleep(max(0.0, 1 / frequency - (time.monotonic() - tick)))
                    continue
                if decision == "execute":
                    # An operator explicitly accepted this one complete paired
                    # target. It is never selected automatically.
                    _set_view_status(view, "Executing operator-approved plan")
                else:
                    assert decision == "resample"
                    # Do not send either arm; re-observe and request a fresh plan.
                    queue.clear()
                    last = np.asarray(obs["joint_positions"], dtype=float).copy()
                    velocity_replans += 1
                    if velocity_replans > max_velocity_replans:
                        raise ValueError(
                            f"{details['arms']} target exceeds the configured joint "
                            f"velocity after {max_velocity_replans} safe replan "
                            "attempt(s). No unsafe command was sent."
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
