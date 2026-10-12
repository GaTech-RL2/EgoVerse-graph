"""Control-rate loop and scoring, independent of checkpoint loading and I/O."""
import time

import numpy as np


class EpisodeWatchdog(TimeoutError):
    pass


class StaticContactMonitor:
    """Observe static overlap before each simulator clamp without changing physics.

    A control step counts once if any substep has a pusher or object overlap
    deeper than tolerance. Arena walls and internal obstacles both count.
    """
    def __init__(self, env, tolerance):
        self.env = env
        self.tolerance = float(tolerance)
        self.contact_steps = []

    def __getattr__(self, name):
        return getattr(self.env, name)

    def step(self, action):
        contact = False
        original = self.env._clamp_pusher_to_static

        def observed_clamp():
            nonlocal contact
            if not contact:
                depth = max(self.env._object_static_penetration_depth(),
                    self.env._shapes_static_penetration_depth(self.env._pusher_body,
                        list(self.env.agent.physics_shapes(self.env))))
                contact = depth > self.tolerance
            return original()

        self.env._clamp_pusher_to_static = observed_clamp
        try:
            result = self.env.step(action)
        finally:
            self.env._clamp_pusher_to_static = original
        self.contact_steps.append(bool(contact))
        return result


def run_episode(queue, env, initial_observation, initial_info, budget, *, on_frame=None, on_step=None):
    queue.reset(initial_observation)
    coverages = [float(initial_info["coverage"])]
    actions = []; replans = []; failure = None
    started = time.monotonic()
    if on_frame: on_frame()
    try:
        for _ in range(int(budget)):
            action = queue.next_action()
            if not np.isfinite(action).all():
                failure = "nonfinite_action"; break
            if queue.prediction_count > len(replans):
                receipt = dict(queue.last_execution)
                if receipt["execution_start_index"] != 0:
                    raise ValueError("A rollout attempted to execute a retrospective action")
                replans.append(receipt)
            observation, _, terminated, truncated, info = env.step(action)
            # Coverage-triggered termination cannot shorten the fixed evaluation budget.
            if truncated: raise RuntimeError("Unexpected simulator truncation")
            del terminated
            actions.append(np.asarray(action).tolist())
            coverage = float(info["coverage"])
            if not np.isfinite(coverage):
                failure = "nonfinite_coverage"; break
            coverages.append(coverage)
            queue.observe(observation)
            if on_frame: on_frame()
            if on_step: on_step(len(actions), coverage, queue.prediction_count)
    except EpisodeWatchdog:
        failure = "watchdog"
    except ValueError as exc:
        text = str(exc).lower()
        if "nonfinite" in text or "non-finite" in text:
            failure = "nonfinite_prediction_or_observation"
        elif text.startswith("invalid decoded native trajectory"):
            failure = "invalid_decoded_trajectory"
        else:
            raise
    observed_peak = max(coverages)
    peak = observed_peak if failure is None else 0.0
    result = {
        "peak_coverage": peak, "observed_peak_before_failure": observed_peak,
        "final_coverage": coverages[-1], "SR80": int(peak >= 0.80), "SR95": int(peak >= 0.95),
        "failure": failure, "control_steps": len(actions), "budget": int(budget),
        "full_budget_executed": len(actions) == int(budget) and failure is None,
        "wall_seconds": time.monotonic() - started, "prediction_count": queue.prediction_count,
        "actions": actions, "coverages": coverages, "replans": replans,
    }
    if isinstance(env, StaticContactMonitor):
        result.update(static_contact_steps=env.contact_steps,
            collision_steps=sum(env.contact_steps),
            collision_rate=float(np.mean(env.contact_steps)) if env.contact_steps else 0.0)
    return result


def summarize(rows):
    if not rows: raise ValueError("No episodes to summarize")
    scores = np.asarray([r["peak_coverage"] for r in rows], dtype=np.float64)
    if not np.isfinite(scores).all(): raise ValueError("Nonfinite score")
    result = {"episodes": len(rows), "mean_peak_coverage": float(scores.mean()),
            "median_peak_coverage": float(np.median(scores)),
            "SR80": float(np.mean(scores >= 0.80)), "SR95": float(np.mean(scores >= 0.95)),
            "failure_count": sum(r["failure"] is not None for r in rows),
            "mean_control_steps": float(np.mean([r["control_steps"] for r in rows]))}
    if all("final_coverage" in r for r in rows):
        result["mean_final_coverage"] = float(np.mean([r["final_coverage"] for r in rows]))
    if all("collision_rate" in r for r in rows):
        result["mean_collision_rate"] = float(np.mean([r["collision_rate"] for r in rows]))
    return result
