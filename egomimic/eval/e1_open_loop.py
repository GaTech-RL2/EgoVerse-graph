"""E1 bimanual ARC tokens in the open-loop distance-budget DTW rollout.

``distance_budget_dtw`` (PR #174) decodes the canonical ARC tokenizer, whose
``(2M, 14)`` token spans ONE joint left-plus-right arc of ``D``. The grid's YAM
arc runs use ``TokenizeBimanualArcLengthE1``: ``(M, 16)`` = 14 waypoint columns
plus one per-arm timing column (``profile``: speed at each waypoint; ``dur``:
seconds of each waypoint interval), and each ARM is resampled independently
over ``min(D, that arm's path inside the fixed source window of W frames)``.

Everything that defines the metric is kept: M-based execution of the first
``m = M * execute_fraction`` waypoints on the predicted clock, the fractional
final prefix (linear interpolation, SLERP for rotation), one endpoint-anchored
global DTW over the concatenated world-frame rollout, and the GT-frame-weighted
reporting. The one change is the anchor schedule. Aniketh's milestones advance
by the constant ``D * (m-1)/(M-1)`` of joint distance, which is exactly what m
waypoints of a canonical token cover. An E1 token covers a variable, per-arm
distance, so the milestone advances by what the executed waypoints of the
**ground-truth** E1 token at that anchor cover:
``(m-1)/(M-1) * sum_arm min(D, path_arm(GT[a : a + W]))``. With canonical
full-``D`` tokens this reduces to the original rule. Anchors come from GT only,
never from predictions. ``oracle_predictions`` scores the GT tokens themselves,
which measures the floor this rollout imposes on a perfect E1 model.
"""

from __future__ import annotations

import math

import numpy as np

E1_ANCHOR_RULE = "gt_token_coverage_v1"
E1_TIMING_MODES = ("profile", "dur")


def build_e1_tokenizer(evaluator):
    from egomimic.rldb.zarr.e1_arc_tokenizer import TokenizeBimanualArcLengthE1

    tokenizer = getattr(evaluator, "_e1_tokenizer_cache", None)
    if tokenizer is None:
        tokenizer = TokenizeBimanualArcLengthE1(
            action_key="actions_cartesian",
            output_action_key="actions_cartesian",
            min_distance_unit=float(evaluator.min_distance_unit),
            resampled_vector_length=int(evaluator.resampled_vector_length),
            dt=float(evaluator.control_dt),
            velocity_norm=str(evaluator.e1_velocity_norm),
            velocity_mode=str(evaluator.e1_velocity_mode),
            min_speed=float(evaluator.e1_min_speed),
        )
        try:
            evaluator._e1_tokenizer_cache = tokenizer
        except AttributeError:
            pass
    return tokenizer


def is_e1_token(prediction: np.ndarray, M: int) -> bool:
    return prediction.ndim == 2 and prediction.shape == (int(M), 16)


def e1_waypoint_prefix(token: np.ndarray, execute_fraction: float) -> np.ndarray:
    """Keep exactly ``M * execute_fraction`` waypoint rows (with their timing)."""
    from egomimic.eval.open_loop_sim import executed_arc_waypoints

    value = np.asarray(token, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 16:
        raise ValueError(f"E1 token must have shape (M, 16), got {value.shape}")
    count = executed_arc_waypoints(len(value), execute_fraction)
    return value[:count].copy()


def e1_fractional_prefix(prefix: np.ndarray, fraction: float, timing_mode: str) -> np.ndarray:
    """Trim an already-capped E1 prefix at fractional source progress.

    Mirrors ``fractional_waypoint_prefix``: only the final partial episode
    budget interpolates a boundary, and the timing is not retimed. ``profile``
    speeds are linear in progress between waypoints, so the terminal speed is
    interpolated; a ``dur`` interval is linear in time along its segment, so
    the truncated interval keeps ``alpha`` of its duration.
    """
    from egomimic.rldb.zarr.arc_length_tokenizer import slerp_pair_ypr

    if timing_mode not in E1_TIMING_MODES:
        raise ValueError(f"E1 timing mode must be one of {E1_TIMING_MODES}")
    value = np.asarray(prefix, dtype=np.float64)
    m = len(value)
    if m < 2 or value.shape[1] != 16 or not 0 < fraction <= 1:
        raise ValueError("Expected an (m>=2, 16) E1 prefix and fraction in (0,1]")
    if fraction == 1:
        return value.copy()
    progress = fraction * (m - 1)
    lower = min(int(math.floor(progress)), m - 2)
    alpha = progress - lower
    if alpha <= 1e-12 and lower >= 1:
        return value[: lower + 1].copy()
    terminal = (1 - alpha) * value[lower] + alpha * value[lower + 1]
    for offset in (3, 10):
        terminal[offset : offset + 3] = slerp_pair_ypr(
            value[lower, offset : offset + 3],
            value[lower + 1, offset : offset + 3],
            np.array([alpha]),
        )[0]
    out = np.concatenate((value[: lower + 1], terminal[None]), axis=0)
    if timing_mode == "dur":
        out[lower, 14:16] = alpha * value[lower, 14:16]
        out[lower + 1, 14:16] = out[lower, 14:16]  # E1 pads the last row with the last interval
    return out


def e1_prefix_control_steps(tokenizer, prefix: np.ndarray, control_dt: float,
                            max_steps: int | None = None) -> int:
    """Control frames needed to play the prefix on its OWN (predicted) clock."""
    from egomimic.rldb.zarr.e1_arc_tokenizer import ARM_LAYOUT, cumulative_arc_length

    clocks = tokenizer.clock_at_waypoints(prefix)
    durations = []
    for clock, (xyz_off, _, _, _) in zip(clocks, ARM_LAYOUT):
        moving = float(cumulative_arc_length(prefix[:, xyz_off : xyz_off + 3])[-1]) > 1e-9
        durations.append(float(clock[-1]) if moving else 0.0)
    duration = max(durations, default=0.0)
    if math.isfinite(duration):
        steps = max(1, int(math.ceil(duration / float(control_dt) - 1e-9)))
    elif max_steps is not None:
        steps = int(max_steps)
    else:
        raise ValueError("E1 prefix has an unusable (non-finite) clock")
    return steps if max_steps is None else min(steps, int(max_steps))


def e1_coverage_windows(gt_xyz: np.ndarray, cumulative: np.ndarray, *, window_frames: int,
                        min_distance_unit: float, waypoint_fraction: float):
    """Milestone anchors whose step is the executed coverage of the GT E1 token.

    Returns ``(anchors, budgets, coverages)``: the first GT frame reaching each
    milestone, the distance budget of that chunk (``min(coverage, remaining)``),
    and the coverage of its m executed GT waypoints. A stationary window
    (coverage ~0) executes one full prefix and the next anchor is the first
    frame that moves, so every paused GT frame is still scored by the DTW.
    """
    xyz = np.asarray(gt_xyz, dtype=np.float64)
    cum = np.asarray(cumulative, dtype=np.float64)
    T, W, D = len(cum), int(window_frames), float(min_distance_unit)
    if xyz.shape != (T, 6) or W < 2 or not 0 < waypoint_fraction <= 1:
        raise ValueError("Invalid GT trajectory, window, or waypoint fraction")
    length = float(cum[-1])
    if length <= 1e-12:
        return np.array([0]), np.array([0.0]), np.array([0.0])
    anchors, budgets, coverages = [], [], []
    milestone = 0.0
    while True:
        a = min(int(np.searchsorted(cum, milestone, side="left")), T - 1)
        window = xyz[a : a + W]
        per_arm = [
            min(D, float(np.linalg.norm(np.diff(window[:, s], axis=0), axis=1).sum()))
            for s in (slice(0, 3), slice(3, 6))
        ]
        coverage = waypoint_fraction * sum(per_arm)
        remaining = length - milestone
        anchors.append(a)
        coverages.append(coverage)
        if coverage <= 1e-12:
            budgets.append(0.0)
            milestone = max(milestone, float(cum[a])) + 1e-9
            if milestone >= length or a >= T - 1:
                break
            continue
        budgets.append(min(coverage, remaining))
        if remaining <= coverage + 1e-12:
            break
        milestone += coverage
    return np.array(anchors), np.array(budgets), np.array(coverages)


def score_e1_distance_dtw_episode(evaluator, records, gt, cumulative) -> dict:
    """One E1 rollout + one global DTW (the canonical tail of the scorer)."""
    from egomimic.eval.distance_budget_dtw import METRIC_FRAME_KEY, METRIC_VERSION, global_dtw, world_xyz
    from egomimic.eval.open_loop_sim import executed_arc_waypoints

    M = int(evaluator.resampled_vector_length)
    m = executed_arc_waypoints(M, evaluator.execute_fraction)
    waypoint_fraction = (m - 1) / (M - 1)
    tokenizer = build_e1_tokenizer(evaluator)
    anchors, budgets, coverages = e1_coverage_windows(
        gt, cumulative, window_frames=int(evaluator.e1_source_frames),
        min_distance_unit=float(evaluator.min_distance_unit), waypoint_fraction=waypoint_fraction,
    )
    predictions, steps, total = [], [], 0
    for anchor, budget, coverage in zip(anchors, budgets, coverages):
        record = records[int(anchor)]
        prefix = e1_waypoint_prefix(record["prediction"], evaluator.execute_fraction)
        if coverage > 1e-12 and budget < coverage:
            prefix = e1_fractional_prefix(prefix, float(budget / coverage), evaluator.e1_velocity_mode)
        n = e1_prefix_control_steps(tokenizer, prefix, evaluator.control_dt)
        if n > evaluator.dtw_max_prediction_steps:
            raise ValueError(f"DTW prefix has {n} samples; increase dtw_max_prediction_steps explicitly")
        decoded = tokenizer.detokenize(prefix, action_horizon=n)
        total += n
        if total * len(gt) > evaluator.dtw_max_cells:
            raise ValueError("Episode exceeds dtw_max_cells; no GT/prediction samples were discarded")
        predictions.append(world_xyz(decoded, record[METRIC_FRAME_KEY]))
        steps.append(n)
    result = global_dtw(np.concatenate(predictions), gt, max_cells=evaluator.dtw_max_cells)
    result.update(
        metric_version=METRIC_VERSION,
        e1_anchor_rule=E1_ANCHOR_RULE,
        e1_source_frames=int(evaluator.e1_source_frames),
        oracle_predictions=bool(getattr(evaluator, "oracle_predictions", False)),
        segments=len(anchors),
        anchor_frames=[int(records[int(i)]["frame"]) for i in anchors],
        segment_control_steps=steps,
        execution_distance_budget_m=None,
        segment_distance_budgets_m=budgets.tolist(),
        segment_gt_coverage_m=coverages.tolist(),
        gt_joint_distance_m=float(cumulative[-1]),
        stationary_fallback=bool(cumulative[-1] <= 1e-12),
        predicted_duration_s=float(total * evaluator.control_dt),
        gt_duration_s=float(len(gt) * evaluator.control_dt),
        duration_ratio=float(total / len(gt)),
        rollout_mode="gt_e1_token_coverage",
    )
    return result
