"""E1 bimanual ARC tokens in the open-loop distance-budget DTW rollout.

``distance_budget_dtw`` (PR #174) decodes the canonical ARC tokenizer, whose
``(2M, 14)`` token spans ONE joint left-plus-right arc of ``D`` with waypoints
evenly spaced in that joint progress. Executing its first ``m`` waypoints means
executing ``(m-1)/(M-1)`` of the token's joint progress, and the GT anchors sit
at the matching joint-distance milestones ``D * (m-1)/(M-1)``.

The grid's YAM arc runs use ``TokenizeBimanualArcLengthE1``: ``(M, 16)`` = 14
waypoint columns plus one per-arm timing column (``profile``: speed at each
waypoint; ``dur``: seconds of each waypoint interval). Each ARM is resampled
over its own ``min(D, path in a fixed W-frame source window)`` and decoded on
its own clock, so waypoint ``m`` of the two arms is reached at different times:
cutting both arms at waypoint ``m`` would hold the earlier arm still while its
recording keeps moving, which penalises even a perfect model.

This port keeps the metric's definition in joint-progress terms:

* **Prefix.** Decode the predicted token on its predicted clocks (no retiming,
  no spatial scaling) and execute it until its joint left-plus-right travel
  reaches ``(m-1)/(M-1)`` of the token's own joint arc length (for a canonical
  token this is exactly its first ``m`` waypoints). The final chunk of an
  episode uses the remaining fraction of that budget.
* **Anchors.** GT joint-distance milestones as in the original rule, but each
  step is what the executed prefix of the **ground-truth** E1 token at that
  anchor covers, ``(m-1)/(M-1) * sum_arm min(D, path_arm(GT[a : a + W]))``
  (constant ``D * (m-1)/(M-1)`` for full-D canonical tokens). Anchors come from
  GT only, never from predictions. A window where both arms are still executes
  one prefix and the next anchor is the first frame that moves. Near the end the
  window is cut by the episode, and the tail stops once a prefix covers < 1 mm.
* One endpoint-anchored global DTW and the GT-frame-weighted reporting are the
  unchanged ``global_dtw`` / ``summarize_distance_dtw``.

``oracle_predictions`` scores the GT tokens themselves: the floor the rollout
imposes on a perfect E1 model (near zero for ``dur``; ``profile`` smooths its
speeds, so its floor is small but not zero).
"""

from __future__ import annotations

import math

import numpy as np

E1_ANCHOR_RULE = "gt_joint_coverage_v3"
E1_TIMING_MODES = ("profile", "dur")
XYZ = (slice(0, 3), slice(7, 10))


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


def waypoint_progress_fraction(M: int, execute_fraction: float) -> float:
    from egomimic.eval.open_loop_sim import executed_arc_waypoints

    m = executed_arc_waypoints(int(M), execute_fraction)
    return (m - 1) / (int(M) - 1)


def joint_progress(trajectory: np.ndarray) -> np.ndarray:
    """Cumulative left-plus-right XYZ travel along a (T, 14) trajectory."""
    value = np.asarray(trajectory, dtype=np.float64)
    steps = sum(np.linalg.norm(np.diff(value[:, s], axis=0), axis=1) for s in XYZ)
    return np.concatenate(([0.0], np.cumsum(steps)))


def e1_execute_prefix(tokenizer, token: np.ndarray, *, progress_fraction: float, control_dt: float,
                      budget_fraction: float = 1.0, max_steps: int | None = None,
                      max_prediction_steps: int = 30_000) -> tuple[np.ndarray, int]:
    """Decode on the token's own clocks; execute until its joint progress budget.

    Returns ``(samples, n)``: the executed control samples (row 0 is the anchor
    pose, rows at ``k * control_dt``) and their count.
    """
    from egomimic.rldb.zarr.e1_arc_tokenizer import cumulative_arc_length

    value = np.asarray(token, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 16:
        raise ValueError(f"E1 token must have shape (M, 16), got {value.shape}")
    paths = [float(cumulative_arc_length(value[:, s])[-1]) for s in XYZ]
    budget = float(progress_fraction) * float(budget_fraction) * sum(paths)
    if sum(paths) <= 1e-9 or budget <= 1e-12:
        samples = tokenizer.detokenize(value, action_horizon=1)
        return samples.astype(np.float64, copy=False), 1
    clocks = tokenizer.clock_at_waypoints(value)
    end = max(float(clock[-1]) for clock, path in zip(clocks, paths) if path > 1e-9)
    if not math.isfinite(end):
        raise ValueError("E1 token has an unusable (non-finite) clock")
    horizon = max(2, int(math.ceil(end / float(control_dt) - 1e-9)) + 1)
    if horizon > max_prediction_steps:
        raise ValueError(f"E1 token decodes to {horizon} samples; raise dtw_max_prediction_steps explicitly")
    full = tokenizer.detokenize(value, action_horizon=horizon).astype(np.float64, copy=False)
    progress = joint_progress(full)
    n = int(np.searchsorted(progress, budget - 1e-12, side="left"))
    n = int(np.clip(n, 1, len(full)))
    if max_steps is not None:
        n = min(n, int(max_steps))
    return full[:n], n


def e1_coverage_windows(gt_xyz: np.ndarray, cumulative: np.ndarray, *, window_frames: int,
                        min_distance_unit: float, progress_fraction: float):
    """Joint-distance milestones stepped by the GT E1 token's executed coverage.

    Returns ``(anchors, budget_fractions, coverages)``: the first GT frame
    reaching each milestone, the fraction of that chunk's budget still inside
    the episode (< 1 only for the final chunk), and the chunk's GT coverage.
    """
    xyz = np.asarray(gt_xyz, dtype=np.float64)
    cum = np.asarray(cumulative, dtype=np.float64)
    T, W, D = len(cum), int(window_frames), float(min_distance_unit)
    if xyz.shape != (T, 6) or W < 2 or not 0 < progress_fraction <= 1:
        raise ValueError("Invalid GT trajectory, window, or progress fraction")
    length = float(cum[-1])
    if length <= 1e-12:
        return np.array([0]), np.array([1.0]), np.array([0.0])
    anchors, fractions, coverages = [], [], []
    milestone = 0.0
    while True:
        a = min(int(np.searchsorted(cum, milestone, side="left")), T - 1)
        window = xyz[a : a + W]
        coverage = progress_fraction * sum(
            min(D, float(np.linalg.norm(np.diff(window[:, s], axis=0), axis=1).sum()))
            for s in (slice(0, 3), slice(3, 6))
        )
        anchors.append(a)
        coverages.append(coverage)
        remaining = length - milestone
        if coverage <= 1e-12:
            fractions.append(1.0)
            milestone = max(milestone, float(cum[a])) + 1e-9
            if milestone >= length or a >= T - 1:
                break
            continue
        if a + W > T and coverage < 1e-3:
            # The GT window is cut by the episode end, so each prefix covers only
            # (m-1)/(M-1) of what is left: stop the geometric tail below 1 mm (the
            # remaining GT frames are still scored against the last samples).
            fractions.append(1.0)
            break
        if remaining <= coverage + 1e-12:
            fractions.append(float(np.clip(remaining / coverage, 1e-6, 1.0)))
            break
        fractions.append(1.0)
        milestone += coverage
    return np.array(anchors), np.array(fractions), np.array(coverages)


def score_e1_distance_dtw_episode(evaluator, records, gt, cumulative) -> dict:
    """One E1 rollout + one global DTW (the canonical tail of the scorer)."""
    from egomimic.eval.distance_budget_dtw import METRIC_FRAME_KEY, METRIC_VERSION, global_dtw, world_xyz

    fraction = waypoint_progress_fraction(evaluator.resampled_vector_length, evaluator.execute_fraction)
    tokenizer = build_e1_tokenizer(evaluator)
    anchors, budget_fractions, coverages = e1_coverage_windows(
        gt, cumulative, window_frames=int(evaluator.e1_source_frames),
        min_distance_unit=float(evaluator.min_distance_unit), progress_fraction=fraction,
    )
    predictions, steps, total = [], [], 0
    for anchor, budget_fraction in zip(anchors, budget_fractions):
        record = records[int(anchor)]
        samples, n = e1_execute_prefix(
            tokenizer, record["prediction"], progress_fraction=fraction, control_dt=evaluator.control_dt,
            budget_fraction=float(budget_fraction), max_prediction_steps=int(evaluator.dtw_max_prediction_steps),
        )
        total += n
        if total * len(gt) > evaluator.dtw_max_cells:
            raise ValueError("Episode exceeds dtw_max_cells; no GT/prediction samples were discarded")
        predictions.append(world_xyz(samples, record[METRIC_FRAME_KEY]))
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
        execution_progress_fraction=fraction,
        segment_budget_fractions=budget_fractions.tolist(),
        segment_gt_coverage_m=coverages.tolist(),
        gt_joint_distance_m=float(cumulative[-1]),
        stationary_fallback=bool(cumulative[-1] <= 1e-12),
        predicted_duration_s=float(total * evaluator.control_dt),
        gt_duration_s=float(len(gt) * evaluator.control_dt),
        duration_ratio=float(total / len(gt)),
        rollout_mode="gt_e1_joint_coverage",
    )
    return result
