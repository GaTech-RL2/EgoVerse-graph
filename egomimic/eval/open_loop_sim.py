"""Episode-level open-loop segment simulation for bimanual cartesian policies.

This evaluator measures a policy over an entire recorded episode rather than
averaging independent action chunks.  Validation samples are observations at
known episode/frame indices.  We cache the prediction made from each
observation, then replay the episode at validation end:

* execute the first ``execute_fraction`` of a baseline control chunk, or cap
  ARC by an exact waypoint prefix (default) or a mode-aware interpolated
  distance prefix;
* compare those control-frequency commands with the ground-truth commands;
* advance to the observation at the resulting frame;
* repeat until the episode ends.

The next observation is the recorded observation at the next boundary.  This
is an oracle-observation open-loop segment rollout: it measures compounding
segment error while keeping the evaluation deterministic and comparable
between baseline and ARC representations.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

from egomimic.eval.bimanual_cartesian_eval import (
    BimanualCartesianEval,
    overlay_annotation_fields,
)
from egomimic.eval.distance_budget_dtw import (
    ARC_DISTANCE_SEMANTICS,
    METRIC_FRAME_KEY,
    METRIC_VERSION,
    evaluator_chunking_mode,
    joint_cumulative_distance,
    per_arm_cumulative_distance,
    resolve_arc_chunking_mode,
    score_distance_dtw_episode,
    summarize_distance_dtw,
)
from egomimic.eval.video import EvalVideo
from egomimic.pl_utils.pl_data_utils import DEFAULT_VALID_GROUP
from egomimic.rldb.zarr.arc_length_tokenizer import (
    bimanual_arc_token_rows,
    cumulative_rotation_length,
    slerp_pair_ypr,
    validate_bimanual_velocity_mode,
)

XYZ_COLS = (0, 1, 2, 7, 8, 9)
YPR_COLS = (3, 4, 5, 10, 11, 12)
GRIP_COLS = (6, 13)
PAIRED_COLS = XYZ_COLS + GRIP_COLS
ARC_EXECUTION_CAP_MODES = ("waypoints", "distance")
ARC_VIDEO_TRAJECTORY_CAP_MODES = ("execution_horizon", "distance", "joint_distance")


def validate_arc_execution_cap_mode(mode: str) -> str:
    """Normalize the two supported ARC execution-cap semantics."""
    value = str(mode).strip().lower()
    if value not in ARC_EXECUTION_CAP_MODES:
        raise ValueError(
            "arc_execution_cap_mode must be one of "
            f"{ARC_EXECUTION_CAP_MODES}, got {mode!r}"
        )
    return value


def validate_arc_video_trajectory_cap_mode(mode: str) -> str:
    """Normalize ARC overlay capping independently of metric semantics."""

    value = str(mode).strip().lower()
    if value not in ARC_VIDEO_TRAJECTORY_CAP_MODES:
        raise ValueError(
            "arc_video_trajectory_cap_mode must be one of "
            f"{ARC_VIDEO_TRAJECTORY_CAP_MODES}, got {mode!r}"
        )
    return value


def executed_control_steps(control_horizon: int, execute_fraction: float) -> int:
    """Return the positive control-step prefix executed from each chunk."""

    horizon = int(control_horizon)
    fraction = float(execute_fraction)
    if horizon < 1:
        raise ValueError("control_horizon must be positive")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("execute_fraction must be in (0, 1]")
    return max(1, min(horizon, int(math.ceil(horizon * fraction))))


def executed_arc_waypoints(num_waypoints: int, execute_fraction: float) -> int:
    """Return an exact integral M-based ARC execution prefix."""
    M = int(num_waypoints)
    fraction = float(execute_fraction)
    if M < 2:
        raise ValueError("num_waypoints must be at least two")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("execute_fraction must be in (0, 1]")
    scaled = M * fraction
    count = int(round(scaled))
    if not math.isclose(scaled, count, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            "M-based ARC execution requires M * execute_fraction to be an "
            f"integer, got {M} * {fraction} = {scaled}"
        )
    if count < 2:
        raise ValueError(
            f"M-based ARC execution must retain at least two waypoints, got {count}"
        )
    return count


def truncate_arc_token_by_waypoints(
    token: np.ndarray,
    execute_fraction: float,
    velocity_mode: str,
) -> np.ndarray:
    """Keep exactly ``execute_fraction * M`` waypoints and timing rows."""
    mode = validate_bimanual_velocity_mode(velocity_mode)
    if mode != "per_waypoint":
        raise ValueError(
            f"M-based ARC execution requires velocity_mode='per_waypoint', got {mode!r}"
        )
    value = np.asarray(token, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 14:
        raise ValueError(f"ARC token must have shape (rows, 14), got {value.shape}")
    rows = int(value.shape[0])
    M = rows // 2
    if rows != bimanual_arc_token_rows(M, mode):
        raise ValueError(
            f"ARC token has {rows} rows, inconsistent with M={M} and mode={mode!r}"
        )
    count = executed_arc_waypoints(M, execute_fraction)
    waypoints = value[:count].copy()
    timing = value[M : M + count].copy()
    return np.concatenate((waypoints, timing), axis=0)


def truncate_arc_token(
    token: np.ndarray,
    execute_fraction: float,
    velocity_mode: str,
    min_distance_unit: float,
    *,
    arc_chunking_mode: str = "joint_distance",
    rotation_distance_unit: float | None = None,
    control_dt: float = 1.0 / 30.0,
) -> np.ndarray:
    """Cap translation at fD using the selected clocks and rotation at fR.

    Independent streams retain their own vertices and timing rows, with
    terminal holds padding shorter streams. Race stops both translations at
    the earliest arm's D time, interpolating on each arm's own clock.
    The legacy standalone helper defaults to joint distance explicitly.
    """

    mode = validate_bimanual_velocity_mode(velocity_mode)
    value = np.asarray(token, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 14:
        raise ValueError(f"ARC token must have shape (rows, 14), got {value.shape}")
    rows = int(value.shape[0])
    granular = mode in ("per_waypoint", "duration")
    M = rows // 2 if granular else rows - 1
    if rows != bimanual_arc_token_rows(M, mode):
        raise ValueError(
            f"ARC token has {rows} rows, inconsistent with M={M} and mode={mode!r}"
        )
    fraction = float(execute_fraction)
    if not 0.0 < fraction <= 1.0:
        raise ValueError("execute_fraction must be in (0, 1]")
    distance = float(min_distance_unit)
    if not math.isfinite(distance) or distance <= 0.0:
        raise ValueError("min_distance_unit must be positive and finite")

    chunking_mode = resolve_arc_chunking_mode(arc_chunking_mode, rotation_distance_unit)
    all_waypoints, all_timing = value[:M], value[M:]
    cumulative = per_arm_cumulative_distance(all_waypoints[:, XYZ_COLS])
    target = fraction * distance
    if chunking_mode == "joint_distance":
        boundaries = [
            _distance_boundary(
                joint_cumulative_distance(all_waypoints[:, XYZ_COLS]), target
            )
        ] * 2
    else:
        boundaries = [
            _distance_boundary(cumulative[:, arm], target) for arm in range(2)
        ]
        if chunking_mode == "race":
            clocks, _ = _arc_clock_durations(
                all_waypoints,
                all_timing,
                mode,
                control_dt,
                distance,
                chunking_mode,
                rotation_distance_unit,
            )
            times = []
            elapsed = [np.concatenate(([0.0], np.cumsum(clock))) for clock in clocks]
            for arm, boundary in enumerate(boundaries):
                lower = min(int(math.floor(boundary)), M - 2)
                alpha = boundary - lower
                times.append(
                    elapsed[arm][lower] + alpha * clocks[arm][lower]
                    if cumulative[-1, arm] >= target and alpha > 0
                    else (
                        elapsed[arm][lower]
                        if cumulative[-1, arm] >= target
                        else math.inf
                    )
                )
            race_time = min(times)
            if math.isfinite(race_time):
                boundaries = [_distance_boundary(clock, race_time) for clock in elapsed]

    streams = []
    for arm, offset in enumerate((0, 7)):
        columns = (
            list(range(offset, offset + 7))
            if rotation_distance_unit is None
            else [offset, offset + 1, offset + 2, offset + 6]
        )
        streams.append((columns, boundaries[arm]))
    if rotation_distance_unit is not None:
        if mode != "per_waypoint":
            raise ValueError(
                "independent rotation clock requires velocity_mode='per_waypoint'"
            )
        rotation_cumulative = cumulative_rotation_length(
            all_waypoints[:, 3:6]
        ) + cumulative_rotation_length(all_waypoints[:, 10:13])
        streams.append(
            (
                list(YPR_COLS),
                _distance_boundary(
                    rotation_cumulative, fraction * float(rotation_distance_unit)
                ),
            )
        )

    count = max(_boundary_rows(boundary) for _, boundary in streams)
    waypoints = np.empty((count, 14), dtype=np.float64)
    timing = np.empty((count if granular else 1, 14), dtype=np.float64)
    for columns, boundary in streams:
        rows = _boundary_rows(boundary)
        prefix = all_waypoints[:rows].copy()
        prefix[-1] = _waypoint_at(all_waypoints, boundary)
        waypoints[:rows, columns] = prefix[:, columns]
        waypoints[rows:, columns] = prefix[-1, columns]
        if granular:
            rates = all_timing[:rows].copy()
            if mode == "duration":
                for offset in (0, 7):
                    if offset in columns:
                        rates[rows - 2, offset] *= boundary - (rows - 2)
            timing[:rows, columns] = rates[:, columns]
            timing[rows:, columns] = rates[-1, columns]
        else:
            timing[:, columns] = all_timing[:, columns]
    return np.concatenate((waypoints, timing), axis=0)


def _distance_boundary(cumulative: np.ndarray, target: float) -> float:
    """Fractional source-row index of a cap, or the final available row."""
    reached = np.flatnonzero(cumulative >= target)
    if not len(reached):
        return float(len(cumulative) - 1)
    upper = int(reached[0])
    if upper == 0:
        return 0.0
    delta = cumulative[upper] - cumulative[upper - 1]
    alpha = (target - cumulative[upper - 1]) / delta if delta > 1e-12 else 1.0
    return float(upper - 1 + np.clip(alpha, 0.0, 1.0))


def _boundary_rows(boundary: float) -> int:
    return max(2, int(math.ceil(boundary - 1e-12)) + 1)


def _waypoint_at(waypoints: np.ndarray, boundary: float) -> np.ndarray:
    lower = min(int(math.floor(boundary)), len(waypoints) - 2)
    alpha = boundary - lower
    result = waypoints[lower] + alpha * (waypoints[lower + 1] - waypoints[lower])
    for columns in (slice(3, 6), slice(10, 13)):
        result[columns] = slerp_pair_ypr(
            waypoints[lower, columns], waypoints[lower + 1, columns], np.array([alpha])
        )[0]
    return result


def _arc_clock_durations(
    waypoints,
    timing,
    velocity_mode,
    dt,
    distance,
    chunking_mode,
    rotation_distance_unit=None,
    max_steps=None,
):
    """Use codec clocks, with unbounded invalid intervals for stride inference.

    The codec uses a horizon-sized hold for a moving interval with no rate.
    Without a caller-provided horizon this has no finite replan boundary.
    """
    from egomimic.rldb.zarr.arc_length_tokenizer import (
        TokenizeBimanualArcLengthCartesian,
    )

    codec = TokenizeBimanualArcLengthCartesian(
        min_distance_unit=distance,
        rotation_distance_unit=rotation_distance_unit,
        resampled_vector_length=len(waypoints),
        dt=dt,
        velocity_mode=velocity_mode,
        arc_chunking_mode=chunking_mode if rotation_distance_unit is not None else None,
    )
    horizon = max_steps if max_steps is not None else 1
    if rotation_distance_unit is not None and chunking_mode == "joint_distance":
        clocks = [
            codec._hybrid_clock_durations(
                waypoints, timing, rotation=False, action_horizon=horizon
            )
        ]
    elif velocity_mode == "per_waypoint":
        clocks = [
            codec._translation_arm_durations(waypoints, timing, offset, horizon)
            for offset in (0, 7)
        ]
    else:
        # Historical nonhybrid mean/duration layouts retain per-arm timing.
        clocks = []
        for offset in (0, 7):
            travel = np.linalg.norm(
                np.diff(waypoints[:, offset : offset + 3], axis=0), axis=1
            )
            if velocity_mode == "duration":
                duration = np.where(travel > 1e-12, timing[:-1, offset], 0.0)
            else:
                speed = np.linalg.norm(timing[0, offset : offset + 3])
                duration = np.divide(
                    travel, speed, out=np.zeros_like(travel), where=speed > 1e-8
                )
            clocks.append(duration)

    def invalid_intervals(rotation=False):
        masks = []
        for offset in (0, 7):
            columns = (
                slice(offset + 3, offset + 6) if rotation else slice(offset, offset + 3)
            )
            travel = (
                np.diff(cumulative_rotation_length(waypoints[:, columns]))
                if rotation
                else np.linalg.norm(np.diff(waypoints[:, columns], axis=0), axis=1)
            )
            if velocity_mode == "duration" and not rotation:
                rate = timing[:-1, offset]
            else:
                rate = np.linalg.norm(
                    (
                        timing[:-1, columns]
                        if velocity_mode == "per_waypoint"
                        else timing[:1, columns]
                    ),
                    axis=1,
                )
            masks.append((travel > 1e-12) & ((rate <= 1e-8) | ~np.isfinite(rate)))
        return masks

    invalid_duration = math.inf if max_steps is None else dt * (max_steps + 1)
    masks = invalid_intervals()
    if len(clocks) == 1:
        masks = [masks[0] | masks[1]]
    clocks = [
        np.where(mask, invalid_duration, clock) for clock, mask in zip(clocks, masks)
    ]
    rotation_clock = None
    if rotation_distance_unit is not None:
        rotation_clock = codec._hybrid_clock_durations(
            waypoints, timing, rotation=True, action_horizon=horizon
        )
        masks = invalid_intervals(rotation=True)
        rotation_clock = np.where(masks[0] | masks[1], invalid_duration, rotation_clock)
    return clocks, rotation_clock


def truncate_cartesian_trajectory_by_joint_distance(
    trajectory: np.ndarray, max_distance: float
) -> np.ndarray:
    """Interpolate a control trajectory at combined left-plus-right travel."""

    value = np.asarray(trajectory, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 14:
        raise ValueError(
            f"cartesian trajectory must have shape (T, 14), got {value.shape}"
        )
    distance = float(max_distance)
    if not math.isfinite(distance) or distance <= 0.0:
        raise ValueError("max_distance must be positive and finite")
    if len(value) < 2:
        return value.copy()

    interval_distance = np.linalg.norm(
        np.diff(value[:, 0:3], axis=0), axis=-1
    ) + np.linalg.norm(np.diff(value[:, 7:10], axis=0), axis=-1)
    cumulative = np.concatenate(([0.0], np.cumsum(interval_distance)))
    crossing = np.flatnonzero(cumulative >= distance)
    if not len(crossing):
        return value.copy()

    end_index = int(crossing[0])
    if end_index < 1:
        return value[:1].copy()
    interval = float(interval_distance[end_index - 1])
    alpha = (
        1.0
        if interval <= 1e-12
        else float(
            np.clip(
                (distance - cumulative[end_index - 1]) / interval,
                0.0,
                1.0,
            )
        )
    )
    result = value[: end_index + 1].copy()
    result[-1] = value[end_index - 1] + alpha * (
        value[end_index] - value[end_index - 1]
    )
    return result


def truncate_cartesian_trajectory_by_joint_clocks(
    trajectory: np.ndarray,
    max_translation_distance: float,
    max_rotation_distance: float,
) -> np.ndarray:
    """Cap bimanual translation and SO(3) rotation independently.

    Both limits use joint cumulative distance (left increment + right
    increment). Translation and gripper hold once D is reached; orientation
    holds once R is reached. The returned trajectory ends after both clocks
    have either reached their cap or exhausted the available prefix.
    """
    value = np.asarray(trajectory, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 14:
        raise ValueError(
            f"cartesian trajectory must have shape (T, 14), got {value.shape}"
        )
    translation_cap = float(max_translation_distance)
    rotation_cap = float(max_rotation_distance)
    if not math.isfinite(translation_cap) or translation_cap <= 0.0:
        raise ValueError("max_translation_distance must be positive and finite")
    if not math.isfinite(rotation_cap) or rotation_cap <= 0.0:
        raise ValueError("max_rotation_distance must be positive and finite")
    if len(value) < 2:
        return value.copy()

    translation_step = np.linalg.norm(
        np.diff(value[:, 0:3], axis=0), axis=-1
    ) + np.linalg.norm(np.diff(value[:, 7:10], axis=0), axis=-1)
    translation_cumulative = np.concatenate(([0.0], np.cumsum(translation_step)))
    rotation_step = np.diff(cumulative_rotation_length(value[:, 3:6])) + np.diff(
        cumulative_rotation_length(value[:, 10:13])
    )
    rotation_cumulative = np.concatenate(([0.0], np.cumsum(rotation_step)))

    def _crossing(cumulative: np.ndarray, cap: float) -> tuple[int, float, bool]:
        reached = np.flatnonzero(cumulative >= cap)
        if not len(reached):
            return len(cumulative) - 1, 1.0, False
        index = int(reached[0])
        if index == 0:
            return 0, 0.0, True
        interval = float(cumulative[index] - cumulative[index - 1])
        alpha = (
            1.0
            if interval <= 1e-12
            else float(
                np.clip(
                    (cap - cumulative[index - 1]) / interval,
                    0.0,
                    1.0,
                )
            )
        )
        return index, alpha, True

    translation_index, translation_alpha, translation_reached = _crossing(
        translation_cumulative, translation_cap
    )
    rotation_index, rotation_alpha, rotation_reached = _crossing(
        rotation_cumulative, rotation_cap
    )
    end_index = max(translation_index, rotation_index)
    result = value[: end_index + 1].copy()

    if translation_reached:
        terminal = value[translation_index - 1] + translation_alpha * (
            value[translation_index] - value[translation_index - 1]
        )
        translation_columns = [0, 1, 2, 6, 7, 8, 9, 13]
        result[translation_index:, translation_columns] = terminal[translation_columns]

    if rotation_reached:
        for ypr_slice in (slice(3, 6), slice(10, 13)):
            terminal_ypr = slerp_pair_ypr(
                value[rotation_index - 1, ypr_slice],
                value[rotation_index, ypr_slice],
                np.array([rotation_alpha]),
            )[0]
            result[rotation_index:, ypr_slice] = terminal_ypr

    return result


def truncate_cartesian_trajectory_by_arc_mode(
    trajectory: np.ndarray,
    max_translation_distance: float,
    arc_chunking_mode: str,
    max_rotation_distance: float | None = None,
) -> np.ndarray:
    """Cap control-frame overlays using the selected D mode and independent R.

    Control frames already share a time axis, so the race boundary is the
    earlier arm crossing. Multistream holds each arm at its own crossing.
    """
    value = np.asarray(trajectory, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 14:
        raise ValueError(
            f"cartesian trajectory must have shape (T, 14), got {value.shape}"
        )
    mode = resolve_arc_chunking_mode(arc_chunking_mode, max_rotation_distance)
    for cap in (max_translation_distance, max_rotation_distance):
        if cap is not None and (not math.isfinite(float(cap)) or cap <= 0):
            raise ValueError("trajectory distance caps must be positive and finite")
    if len(value) < 2:
        return value.copy()
    cumulative = per_arm_cumulative_distance(value[:, XYZ_COLS])
    if mode == "joint_distance":
        boundaries = [
            _distance_boundary(
                joint_cumulative_distance(value[:, XYZ_COLS]), max_translation_distance
            )
        ] * 2
    else:
        boundaries = [
            _distance_boundary(cumulative[:, arm], max_translation_distance)
            for arm in range(2)
        ]
        if mode == "race":
            boundaries = [min(boundaries)] * 2
    streams = []
    for arm, offset in enumerate((0, 7)):
        columns = (
            list(range(offset, offset + 7))
            if max_rotation_distance is None
            else [offset, offset + 1, offset + 2, offset + 6]
        )
        streams.append((columns, boundaries[arm]))
    if max_rotation_distance is not None:
        rotation = cumulative_rotation_length(
            value[:, 3:6]
        ) + cumulative_rotation_length(value[:, 10:13])
        streams.append(
            (list(YPR_COLS), _distance_boundary(rotation, max_rotation_distance))
        )
    count = max(_boundary_rows(boundary) for _, boundary in streams)
    result = value[:count].copy()
    for columns, boundary in streams:
        terminal = _waypoint_at(value, boundary)
        end = _boundary_rows(boundary) - 1
        result[end:, columns] = terminal[columns]
    return result


def arc_execution_prefix(
    token: np.ndarray,
    execute_fraction: float,
    velocity_mode: str,
    min_distance_unit: float,
    arc_execution_cap_mode: str,
    *,
    arc_chunking_mode: str = "joint_distance",
    rotation_distance_unit: float | None = None,
    control_dt: float = 1.0 / 30.0,
) -> np.ndarray:
    """Apply either exact M-based or interpolated distance-based capping."""
    cap_mode = validate_arc_execution_cap_mode(arc_execution_cap_mode)
    if cap_mode == "waypoints":
        return truncate_arc_token_by_waypoints(token, execute_fraction, velocity_mode)
    return truncate_arc_token(
        token,
        execute_fraction,
        velocity_mode,
        min_distance_unit,
        arc_chunking_mode=arc_chunking_mode,
        rotation_distance_unit=rotation_distance_unit,
        control_dt=control_dt,
    )


def arc_prefix_control_steps(
    token: np.ndarray,
    execute_fraction: float,
    velocity_mode: str,
    control_dt: float,
    min_distance_unit: float,
    *,
    rotation_distance_unit: float | None = None,
    arc_chunking_mode: str = "joint_distance",
    arc_execution_cap_mode: str = "waypoints",
    max_steps: int | None = None,
) -> int:
    """Recover the control-frame stride for the configured ARC prefix."""

    dt = float(control_dt)
    if dt <= 0.0:
        raise ValueError("control_dt must be positive")
    if rotation_distance_unit is not None and (
        not math.isfinite(float(rotation_distance_unit))
        or float(rotation_distance_unit) <= 0.0
    ):
        raise ValueError("rotation_distance_unit must be positive and finite")
    partial = arc_execution_prefix(
        token,
        execute_fraction,
        velocity_mode,
        min_distance_unit,
        arc_execution_cap_mode,
        arc_chunking_mode=arc_chunking_mode,
        rotation_distance_unit=rotation_distance_unit,
        control_dt=control_dt,
    )
    mode = validate_bimanual_velocity_mode(velocity_mode)
    granular = mode in ("per_waypoint", "duration")
    M = len(partial) // 2 if granular else len(partial) - 1
    waypoints = partial[:M]
    timing = partial[M:]
    chunking_mode = resolve_arc_chunking_mode(arc_chunking_mode, rotation_distance_unit)
    clocks, rotation_clock = _arc_clock_durations(
        waypoints,
        timing,
        mode,
        dt,
        min_distance_unit,
        chunking_mode,
        rotation_distance_unit,
        max_steps,
    )
    durations = [float(np.sum(clock)) for clock in clocks]
    # The distance prefix already enforces the race endpoint. All retained
    # stream prefixes must finish, including exact M prefixes whose predicted
    # per-arm clocks may disagree; rotation remains independently timed.
    duration = max(durations, default=0.0)
    if rotation_clock is not None:
        duration = max(duration, float(np.sum(rotation_clock)))
    if math.isfinite(duration):
        steps = max(1, int(math.ceil(duration / dt - 1e-9)))
    elif max_steps is not None:
        steps = int(max_steps)
    else:
        raise ValueError("ARC prefix timing does not define a finite replan boundary")
    if max_steps is not None:
        steps = min(steps, int(max_steps))
    return max(1, steps)


class OpenLoopSimEval(BimanualCartesianEval):
    """Compare baseline and ARC policies over complete recorded episodes.

    A baseline executes the requested fraction of its time-indexed action
    chunk. ARC supports two explicit caps before detokenization: ``waypoints``
    keeps exactly ``execute_fraction * M`` waypoint and per-waypoint velocity
    rows; ``distance`` interpolates its terminal waypoint at
    ``execute_fraction * D`` according to the selected translation mode. Token
    timing recovers the corresponding variable control-frame stride.

    The evaluator expects validation to contain every frame of each episode,
    with ``episode_hash`` and ``frame_index`` metadata. It accumulates model
    predictions during the normal validation loop and, when visualization is
    configured, buffers rendered frames until each complete episode closes.
    Video output therefore follows the metric unit: one MP4 per episode. All
    episode MP4s are kept on disk, while only the first MP4 for each
    validation-group/embodiment pair is uploaded to W&B per validation loop.
    """

    def __init__(
        self,
        *,
        action_key: str = "actions_cartesian",
        ground_truth_action_key: str = "actions_cartesian_untokenized",
        execute_fraction: float = 0.30,
        control_horizon: int = 100,
        control_dt: float = 1.0 / 30.0,
        action_mode: str = "auto",
        arc_execution_cap_mode: str = "waypoints",
        arc_video_trajectory_cap_mode: str = "joint_distance",
        min_distance_unit: float = 0.40,
        rotation_distance_unit: float | None = None,
        resampled_vector_length: int = 100,
        velocity_mode: str = "per_waypoint",
        log_step: int | None = None,
        results_path: str | None = None,
        trajectory_snapshot_path: str | None = None,
        video_only: bool = False,
        distance_dtw_enabled: bool = False,
        dtw_max_cells: int = 50_000_000,
        dtw_max_prediction_steps: int = 30_000,
        require_episode_start: bool = True,
        limit_val_episodes: int | None = None,
        requires_ordered_validation: bool = True,
        limit_val_batches: int | float | None = None,
        deterministic_seed: int = 420042,
        obs_pose_key: str = "observations.state.ee_pose",
        image_key: str = "observations.images.front_img_1",
        viz_func: Mapping | None = None,
        revert_transforms: Mapping | None = None,
        video_output_dir: str | None = None,
        video_chunk_frames: int = 1000,
        max_episode_frames: int = 6000,
        viz_every_n_epochs: int = 1,
        viz_max_batches: int | None = None,
        arc_chunking_mode: str | None = None,
        **kwargs,
    ):
        mode = str(action_mode)
        if mode not in ("auto", "baseline", "arc"):
            raise ValueError("action_mode must be auto, baseline, or arc")
        if float(control_dt) <= 0:
            raise ValueError("control_dt must be positive")
        if not math.isfinite(float(min_distance_unit)) or float(min_distance_unit) <= 0:
            raise ValueError("min_distance_unit must be positive and finite")
        if rotation_distance_unit is not None and (
            not math.isfinite(float(rotation_distance_unit))
            or float(rotation_distance_unit) <= 0
        ):
            raise ValueError("rotation_distance_unit must be positive and finite")
        if arc_chunking_mode is not None and rotation_distance_unit is None:
            raise ValueError(
                "explicit arc_chunking_mode requires rotation_distance_unit; "
                "omit arc_chunking_mode for legacy no-R checkpoints"
            )
        validate_bimanual_velocity_mode(velocity_mode)
        self.execute_fraction = float(execute_fraction)
        self.control_horizon = int(control_horizon)
        self.control_dt = float(control_dt)
        self.action_mode = mode
        self.arc_chunking_mode = resolve_arc_chunking_mode(
            arc_chunking_mode, rotation_distance_unit
        )
        self.arc_execution_cap_mode = validate_arc_execution_cap_mode(
            arc_execution_cap_mode
        )
        self.arc_video_trajectory_cap_mode = validate_arc_video_trajectory_cap_mode(
            arc_video_trajectory_cap_mode
        )
        self.ground_truth_action_key = str(ground_truth_action_key)
        self.min_distance_unit = float(min_distance_unit)
        self.rotation_distance_unit = (
            None if rotation_distance_unit is None else float(rotation_distance_unit)
        )
        self.resampled_vector_length = int(resampled_vector_length)
        self.velocity_mode = str(velocity_mode)
        self.execute_arc_waypoints = (
            executed_arc_waypoints(self.resampled_vector_length, self.execute_fraction)
            if mode == "arc" and self.arc_execution_cap_mode == "waypoints"
            else None
        )
        if (
            mode == "arc"
            and self.arc_execution_cap_mode == "waypoints"
            and self.velocity_mode != "per_waypoint"
        ):
            raise ValueError(
                "M-based ARC execution requires velocity_mode='per_waypoint'"
            )
        self.log_step = None if log_step is None else int(log_step)
        if self.log_step is not None and self.log_step < 0:
            raise ValueError("log_step must be nonnegative")
        self.results_path = Path(results_path) if results_path else None
        self.trajectory_snapshot_path = (
            Path(trajectory_snapshot_path) if trajectory_snapshot_path else None
        )
        if (
            self.trajectory_snapshot_path is not None
            and self.trajectory_snapshot_path.suffix != ".npz"
        ):
            raise ValueError("trajectory_snapshot_path must end in .npz")
        self._trajectory_snapshot_written = False
        self._video_seen_frames = set()
        self._video_last_frame = {}
        self.video_only = bool(video_only)
        self.distance_dtw_enabled = bool(distance_dtw_enabled)
        self.dtw_max_cells = int(dtw_max_cells)
        self.dtw_max_prediction_steps = int(dtw_max_prediction_steps)
        if self.dtw_max_cells < 1 or self.dtw_max_prediction_steps < 1:
            raise ValueError("DTW resource limits must be positive")
        self.require_episode_start = bool(require_episode_start)
        self.limit_val_episodes = (
            None if limit_val_episodes is None else int(limit_val_episodes)
        )
        if self.limit_val_episodes is not None and self.limit_val_episodes < 1:
            raise ValueError("limit_val_episodes must be positive")
        self.requires_ordered_validation = bool(requires_ordered_validation)
        if limit_val_batches is not None:
            raise ValueError(
                "open_loop_sim is episode-based; use limit_val_episodes instead "
                "of limit_val_batches"
            )
        self._records: list[dict[str, Any]] = []
        self.last_results = None
        self._arc_tokenizer = None
        self._metric_device = torch.device("cpu")

        # Reuse the graph evaluator's normalizer binding, deterministic model,
        # embodiment resolution, metric-group namespacing, and episode-aware
        # video buffering. Chunk-level ARC metrics remain deliberately disabled
        # here because this evaluator scores the executed control prefix.
        # Existing ABC experiment blocks are merged into a selected evaluator
        # config by Hydra. Consume their legacy ARC-only knobs so selecting
        # this evaluator does not fail on an unrelated ``action_horizon`` (or
        # accidentally enable chunk/video metrics).
        for ignored_key in (
            "arc_metrics",
            "include_reconstruction_loss",
            "arcmatch_distance",
            "arcmatch_points",
            "arc_chunk_rows",
            "action_horizon",
            "rot_lever_m",
            "dtw_max_samples",
        ):
            kwargs.pop(ignored_key, None)
        super().__init__(
            action_key=action_key,
            obs_pose_key=obs_pose_key,
            image_key=image_key,
            viz_func=viz_func,
            revert_transforms=revert_transforms,
            arc_metrics=False,
            deterministic_seed=deterministic_seed,
            limit_val_batches=None,
            video_output_dir=video_output_dir,
            video_chunk_frames=video_chunk_frames,
            max_episode_frames=max_episode_frames,
            viz_every_n_epochs=viz_every_n_epochs,
            viz_max_batches=viz_max_batches,
            **kwargs,
        )
        self._video_enabled = bool(self.viz_func)
        if self.video_only and not self._video_enabled:
            raise ValueError("video_only requires a configured visualization function")
        self.execute_steps = executed_control_steps(
            self.control_horizon, self.execute_fraction
        )

    def on_validation_start(self):
        if self._video_enabled:
            EvalVideo.on_validation_start(self)
        self._records = []
        self.last_results = None
        self._trajectory_snapshot_written = False
        self._video_seen_frames = set()
        self._video_last_frame = {}
        if self.model is not None:
            try:
                self._metric_device = next(self.model.parameters()).device
            except StopIteration:
                pass

    def _decoded_video_predictions(
        self, prediction: torch.Tensor, embodiment_id: int, max_steps: int
    ) -> tuple[torch.Tensor, list[int]]:
        """Decode only each frame's independently executed prediction prefix."""

        native = self._native(prediction, embodiment_id).detach().cpu().numpy()
        decoded_with_steps = [
            self._decode_prediction_with_steps(sample, max_steps=max_steps)
            for sample in native
        ]
        lengths = [steps for _, steps in decoded_with_steps]
        width = max(lengths)
        decoded = []
        for value, steps in decoded_with_steps:
            if steps < width:
                value = np.concatenate(
                    (value, np.repeat(value[-1:], width - steps, axis=0)), axis=0
                )
            decoded.append(value)
        return (
            torch.from_numpy(np.stack(decoded).astype(np.float32, copy=False)),
            lengths,
        )

    def _cap_arc_video_trajectories(
        self,
        prediction: torch.Tensor,
        ground_truth: torch.Tensor,
        prefix_lengths: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor, list[int], list[int]]:
        """Cap ARC overlays at the selected translation and rotation prefixes."""

        cap = self.execute_fraction * self.min_distance_unit
        rotation_distance_unit = getattr(self, "rotation_distance_unit", None)
        pred_np = prediction.detach().cpu().numpy()
        gt_np = ground_truth.detach().cpu().numpy()
        pred_values = []
        gt_values = []
        pred_lengths = []
        gt_lengths = []
        for index, steps in enumerate(prefix_lengths):
            rotation_cap = (
                None
                if rotation_distance_unit is None
                else self.execute_fraction * rotation_distance_unit
            )
            pred = truncate_cartesian_trajectory_by_arc_mode(
                pred_np[index, :steps], cap, evaluator_chunking_mode(self), rotation_cap
            )
            gt = truncate_cartesian_trajectory_by_arc_mode(
                gt_np[index, :steps], cap, evaluator_chunking_mode(self), rotation_cap
            )
            pred_values.append(pred)
            gt_values.append(gt)
            pred_lengths.append(len(pred))
            gt_lengths.append(len(gt))

        width = max(max(pred_lengths), max(gt_lengths))

        def _pad(values: list[np.ndarray]) -> torch.Tensor:
            padded = []
            for value in values:
                if len(value) < width:
                    value = np.concatenate(
                        (value, np.repeat(value[-1:], width - len(value), axis=0)),
                        axis=0,
                    )
                padded.append(value)
            return torch.from_numpy(np.stack(padded).astype(np.float32, copy=False))

        return _pad(pred_values), _pad(gt_values), pred_lengths, gt_lengths

    def _video_fps(self, source_fps=30):
        """Full-frame video uses the control clock, not the DDP shard rate."""
        return max(1, round(1.0 / float(getattr(self, "control_dt", 1.0 / source_fps))))

    def _collective_video_enabled(self, batch_idx=0):
        cadence = self.viz_every_n_epochs
        return (
            cadence > 0
            and (getattr(self.trainer, "current_epoch", 0) + 1) % cadence == 0
            and (self.viz_max_batches is None or batch_idx < self.viz_max_batches)
        )

    def _collect_open_loop_video(
        self, *, source_id, source_batch, prediction, embodiment_id, embodiment_name
    ):
        """Gather interleaved DDP frames; only this experiment's rank zero renders.

        Lightning's ordered DistributedSampler assigns index i*world+rank.
        Interleaving rank batches reconstructs dataset order without assuming
        episode IDs are lexicographically ordered. Sampler padding is deduplicated.
        """

        def cpu(value):
            if isinstance(value, torch.Tensor):
                return value.detach().cpu()
            if isinstance(value, dict):
                return {k: cpu(v) for k, v in value.items()}
            if isinstance(value, (tuple, list)):
                return type(value)(cpu(v) for v in value)
            return value

        # Wrist-camera observations are not used by this front-camera overlay.
        batch = {
            k: v
            for k, v in source_batch.items()
            if not str(k).startswith("observations.images.") or k == self.image_key
        }
        payload = dict(
            source_id=str(source_id),
            source_batch=cpu(batch),
            prediction=cpu(prediction),
            embodiment_id=embodiment_id,
            embodiment_name=embodiment_name,
        )
        distributed = dist.is_available() and dist.is_initialized()
        world = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0
        parts = [None] * world
        if distributed:
            dist.all_gather_object(parts, payload)
        else:
            parts[0] = payload
        if rank != 0:
            return

        if not hasattr(self, "_video_seen_frames"):
            self._video_seen_frames = set()
            self._video_last_frame = {}
        for part in parts:
            if (part["source_id"], part["embodiment_name"]) != (
                str(source_id),
                embodiment_name,
            ):
                raise RuntimeError(
                    "Validation ranks disagree on video source/embodiment"
                )
        for index in range(max(len(part["prediction"]) for part in parts)):
            for part in parts:
                n = len(part["prediction"])
                if index >= n:
                    continue

                def one(value):
                    if (
                        isinstance(value, (torch.Tensor, np.ndarray))
                        and value.ndim
                        and len(value) == n
                    ):
                        return value[index : index + 1]
                    if isinstance(value, (tuple, list)) and len(value) == n:
                        return value[index : index + 1]
                    return value

                item = {k: one(v) for k, v in part["source_batch"].items()}
                episode = str(
                    self._batch_values(item["episode_hash"], 1, "episode_hash")[0]
                )
                frame = int(
                    self._batch_values(item["frame_index"], 1, "frame_index")[0]
                )
                group = self._validation_group or DEFAULT_VALID_GROUP
                episode_key = (group, str(source_id), episode)
                key = (*episode_key, frame)
                if key in self._video_seen_frames:
                    continue
                expected = self._video_last_frame.get(episode_key, -1) + 1
                if frame != expected:
                    raise RuntimeError(
                        f"Full-frame video is missing/out of order: {episode_key}, "
                        f"expected frame {expected}, got {frame}"
                    )
                self._maybe_log_open_loop_video(
                    source_id=source_id,
                    source_batch=item,
                    prediction=part["prediction"][index : index + 1],
                    embodiment_id=embodiment_id,
                    embodiment_name=embodiment_name,
                )
                self._video_seen_frames.add(key)
                self._video_last_frame[episode_key] = frame

    def _maybe_log_open_loop_video(
        self,
        *,
        source_id: str,
        source_batch: Mapping,
        prediction: torch.Tensor,
        embodiment_id: int,
        embodiment_name: str,
    ) -> None:
        """Render one frame per validation sample and buffer by episode hash.

        Like the metric path, each video frame shows only the independently
        executed prediction prefix and its matching control-frequency ground
        truth. ARC uses the configured waypoint- or distance-based cap before
        its decoded timing determines the frame width.
        """

        if not getattr(self, "_video_enabled", False) or not getattr(
            self.trainer, "is_global_zero", True
        ):
            return
        viz_partial = self.viz_func.get(embodiment_name)
        if viz_partial is None or self.obs_pose_key not in source_batch:
            return
        target_key = (
            self.ground_truth_action_key
            if self.ground_truth_action_key in source_batch
            else self.action_key
        )
        gt_native = (
            self._native_key(source_batch[target_key], target_key, embodiment_id)
            .detach()
            .cpu()
        )
        if gt_native.ndim != 3 or gt_native.shape[0] != prediction.shape[0]:
            raise ValueError(
                "open_loop_sim video ground truth must be batched as (B, T, 14), "
                f"got {tuple(gt_native.shape)}"
            )
        pred_native, prefix_lengths = self._decoded_video_predictions(
            prediction, embodiment_id, int(gt_native.shape[1])
        )
        width = int(pred_native.shape[1])
        gt_prefixes = []
        for index, steps in enumerate(prefix_lengths):
            value = gt_native[index, :steps]
            if steps < width:
                value = torch.cat((value, value[-1:].repeat(width - steps, 1)), dim=0)
            gt_prefixes.append(value)
        gt_native = torch.stack(gt_prefixes).to(dtype=pred_native.dtype)

        prediction_lengths = list(prefix_lengths)
        ground_truth_lengths = list(prefix_lengths)
        is_arc_video = self.action_mode == "arc" or (
            self.action_mode == "auto"
            and self._is_arc_prediction(prediction[0].detach().cpu().numpy())
        )
        if is_arc_video and getattr(
            self,
            "arc_video_trajectory_cap_mode",
            "execution_horizon",
        ) in ("distance", "joint_distance"):
            (
                pred_native,
                gt_native,
                prediction_lengths,
                ground_truth_lengths,
            ) = self._cap_arc_video_trajectories(pred_native, gt_native, prefix_lengths)

        obs_pose_native = (
            self._native_pose(source_batch[self.obs_pose_key], embodiment_id)
            .detach()
            .cpu()
        )
        if obs_pose_native.ndim == 3 and obs_pose_native.shape[1] == 1:
            obs_pose_native = obs_pose_native.squeeze(1)
        pred_camframe = self._revert_to_camframe(
            actions=pred_native,
            obs_pose=obs_pose_native,
            embodiment_name=embodiment_name,
        )
        gt_camframe = self._revert_to_camframe(
            actions=gt_native,
            obs_pose=obs_pose_native,
            embodiment_name=embodiment_name,
        )
        if pred_camframe is None or gt_camframe is None:
            return

        self._write_trajectory_snapshot(
            source_id=source_id,
            source_batch=source_batch,
            prediction=pred_camframe,
            ground_truth=gt_camframe,
            prediction_length=prediction_lengths[0],
            ground_truth_length=ground_truth_lengths[0],
        )

        images = source_batch[self.image_key]
        if images.ndim == 5:
            images = images[:, 0]
        images = images.detach().cpu()
        if images.ndim == 4 and images.shape[1] in (1, 3):
            images = images.permute(0, 2, 3, 1)
        flat_predictions = {
            f"{embodiment_name}_{self.action_key}": pred_camframe,
        }
        flat_batch = {
            self.image_key: images,
            self.action_key: gt_camframe,
            "embodiment": source_batch["embodiment"].detach().cpu(),
        }
        if "intrinsics" in source_batch:
            flat_batch["intrinsics"] = source_batch["intrinsics"].detach().cpu()
        flat_batch.update(
            overlay_annotation_fields(
                viz_partial, {**source_batch, "source": source_id}
            )
        )
        try:
            frames = viz_partial(predictions=flat_predictions, batch=flat_batch)
        except Exception as exc:  # noqa: BLE001 -- overlays are best effort
            print(
                f"[OpenLoopSimEval] skipped {embodiment_name} overlay: {exc}",
                flush=True,
            )
            return
        frames = np.asarray(frames)
        if frames.dtype != np.uint8:
            frames = np.clip(frames, 0, 255).astype(np.uint8)
        if frames.ndim == 3:
            frames = frames[None]
        frame_tensor = torch.from_numpy(frames)
        hashes = [
            str(value)
            for value in self._batch_values(
                source_batch["episode_hash"], int(frame_tensor.shape[0]), "episode_hash"
            )
        ]
        group = self._validation_group or DEFAULT_VALID_GROUP
        buf_key = (group, embodiment_name)
        out_dir = self._group_video_dir(group, embodiment_name)
        self._buffer_per_episode(buf_key, out_dir, list(frame_tensor), hashes)

    def _write_trajectory_snapshot(
        self,
        *,
        source_id: str,
        source_batch: Mapping,
        prediction,
        ground_truth,
        prediction_length: int,
        ground_truth_length: int,
    ) -> None:
        """Save the first rendered executed prefix for speed/shape diagnostics."""

        path = getattr(self, "trajectory_snapshot_path", None)
        if path is None or getattr(self, "_trajectory_snapshot_written", False):
            return

        def _numpy(value) -> np.ndarray:
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
            return np.asarray(value)

        pred = _numpy(prediction)
        gt = _numpy(ground_truth)
        if pred.ndim != 3 or gt.ndim != 3 or pred.shape[0] < 1 or gt.shape[0] < 1:
            raise ValueError(
                "trajectory snapshot expects batched (B, T, 14) camera-frame arrays"
            )
        pred_steps = min(int(prediction_length), int(pred.shape[1]))
        gt_steps = min(int(ground_truth_length), int(gt.shape[1]))
        if pred_steps < 1 or gt_steps < 1:
            raise ValueError("trajectory snapshot execution prefix is empty")

        batch_size = int(pred.shape[0])
        episode = self._batch_values(
            source_batch["episode_hash"], batch_size, "episode_hash"
        )[0]
        frame = -1
        if "frame_index" in source_batch:
            frame = int(
                self._batch_values(
                    source_batch["frame_index"], batch_size, "frame_index"
                )[0]
            )

        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            prediction=pred[0, :pred_steps].astype(np.float32, copy=False),
            ground_truth=gt[0, :gt_steps].astype(np.float32, copy=False),
            control_dt=np.asarray(self.control_dt, dtype=np.float64),
            source_id=np.asarray(str(source_id)),
            episode_hash=np.asarray(str(episode)),
            frame_index=np.asarray(frame, dtype=np.int64),
            action_mode=np.asarray(str(self.action_mode)),
            arc_execution_cap_mode=np.asarray(str(self.arc_execution_cap_mode)),
            arc_chunking_mode=np.asarray(evaluator_chunking_mode(self)),
        )
        self._trajectory_snapshot_written = True

    def _log_wandb_videos(self) -> None:
        """Upload only the first episode MP4 for each val loop/panel."""

        if not self._written_paths:
            return
        experiment = self._wandb_logger()
        if experiment is None:
            return
        import wandb  # type: ignore

        first: dict[tuple[str, str], str] = {}
        for group, embodiment_name, path in self._written_paths:
            first.setdefault((group, embodiment_name), path)
        payload = {}
        for (group, embodiment_name), path in first.items():
            prefix = (
                "Val_video" if group == DEFAULT_VALID_GROUP else f"Val_video_{group}"
            )
            payload[f"{prefix}/{embodiment_name}"] = wandb.Video(
                path, fps=self._video_fps(), format="mp4"
            )
        experiment.log(payload, step=self._resolved_log_step())

    def _resolved_log_step(self) -> int:
        log_step = getattr(self, "log_step", None)
        if log_step is not None:
            return int(log_step)
        return int(getattr(self.trainer, "global_step", 0))

    @staticmethod
    def _batch_values(value, batch_size: int, label: str) -> list:
        if isinstance(value, torch.Tensor):
            if value.ndim == 0:
                return [value.item()] * batch_size
            if int(value.shape[0]) != batch_size:
                raise ValueError(
                    f"{label} has batch dimension {tuple(value.shape)}, expected "
                    f"{batch_size}"
                )
            return [item.item() if item.ndim == 0 else item for item in value]
        if isinstance(value, np.ndarray):
            if value.ndim == 0:
                return [value.item()] * batch_size
            if int(value.shape[0]) != batch_size:
                raise ValueError(
                    f"{label} has batch dimension {value.shape}, expected {batch_size}"
                )
            return [item.item() if np.ndim(item) == 0 else item for item in value]
        if isinstance(value, (list, tuple)):
            if len(value) != batch_size:
                raise ValueError(
                    f"{label} has {len(value)} entries, expected {batch_size}"
                )
            return list(value)
        return [value] * batch_size

    def _native_key(self, value: torch.Tensor, key: str, embodiment_id: int):
        if self.normalizer is None:
            raise RuntimeError("open_loop_sim evaluator data context was not bound")
        return self.normalizer.unnormalize({key: value}, embodiment_id).get(key, value)

    def _is_arc_prediction(self, prediction: np.ndarray) -> bool:
        expected = bimanual_arc_token_rows(
            self.resampled_vector_length, self.velocity_mode
        )
        return prediction.ndim == 2 and prediction.shape == (expected, 14)

    def _decode_prediction_with_steps(
        self, prediction: np.ndarray, *, max_steps: int | None = None
    ) -> tuple[np.ndarray, int]:
        is_arc = self._is_arc_prediction(prediction)
        if self.action_mode == "arc" and not is_arc:
            raise ValueError(
                "open_loop_sim action_mode='arc' received a non-ARC prediction "
                f"with shape {prediction.shape}"
            )
        if self.action_mode == "baseline" and is_arc:
            raise ValueError(
                "open_loop_sim action_mode='baseline' received an ARC prediction"
            )
        if not is_arc:
            if prediction.ndim != 2 or prediction.shape[1] != 14:
                raise ValueError(
                    "baseline open_loop_sim predictions must have shape (T, 14), "
                    f"got {prediction.shape}"
                )
            if prediction.shape[0] < self.execute_steps:
                raise ValueError(
                    f"baseline prediction has only {prediction.shape[0]} control "
                    f"steps, needs {self.execute_steps}"
                )
            steps = self.execute_steps
            if max_steps is not None:
                steps = min(steps, int(max_steps))
            return prediction[:steps].copy(), steps

        if self._arc_tokenizer is None:
            from egomimic.rldb.zarr.arc_length_tokenizer import (
                TokenizeBimanualArcLengthCartesian,
            )

            rotation_distance_unit = getattr(self, "rotation_distance_unit", None)
            self._arc_tokenizer = TokenizeBimanualArcLengthCartesian(
                min_distance_unit=self.min_distance_unit,
                rotation_distance_unit=rotation_distance_unit,
                arc_chunking_mode=(
                    evaluator_chunking_mode(self)
                    if rotation_distance_unit is not None
                    else None
                ),
                resampled_vector_length=self.resampled_vector_length,
                dt=self.control_dt,
                velocity_mode=self.velocity_mode,
            )
        partial = arc_execution_prefix(
            prediction,
            self.execute_fraction,
            self.velocity_mode,
            self.min_distance_unit,
            self.arc_execution_cap_mode,
            arc_chunking_mode=evaluator_chunking_mode(self),
            rotation_distance_unit=getattr(self, "rotation_distance_unit", None),
            control_dt=self.control_dt,
        )
        steps = arc_prefix_control_steps(
            prediction,
            self.execute_fraction,
            self.velocity_mode,
            self.control_dt,
            self.min_distance_unit,
            rotation_distance_unit=getattr(self, "rotation_distance_unit", None),
            arc_chunking_mode=evaluator_chunking_mode(self),
            arc_execution_cap_mode=self.arc_execution_cap_mode,
            max_steps=max_steps,
        )
        decoded = self._arc_tokenizer.detokenize(partial, action_horizon=steps).astype(
            np.float64, copy=False
        )
        return decoded, steps

    def _decode_prediction(self, prediction: np.ndarray) -> np.ndarray:
        return self._decode_prediction_with_steps(prediction)[0]

    def _append_source_records(self, source_id: str, source_batch, prediction):
        if not isinstance(prediction, torch.Tensor) or prediction.ndim != 3:
            raise ValueError(
                f"open_loop_sim expects batched predictions, got {type(prediction)} "
                f"with shape {getattr(prediction, 'shape', None)}"
            )
        batch_size = int(prediction.shape[0])
        self._metric_device = prediction.device
        if "episode_hash" not in source_batch or "frame_index" not in source_batch:
            raise KeyError(
                "open_loop_sim requires episode_hash and frame_index in every "
                f"validation source ({source_id!r})"
            )
        embodiment_id, label = self._embodiment(source_batch)
        episode_hashes = self._batch_values(
            source_batch["episode_hash"], batch_size, "episode_hash"
        )
        frame_indices = self._batch_values(
            source_batch["frame_index"], batch_size, "frame_index"
        )
        pred_native = self._native(prediction, embodiment_id).detach().cpu().numpy()
        target_key = (
            self.ground_truth_action_key
            if self.ground_truth_action_key in source_batch
            else self.action_key
        )
        target_value = self._native_key(
            source_batch[target_key], target_key, embodiment_id
        )
        target_native = target_value.detach().cpu().numpy()
        metric_anchors = None
        if getattr(self, "distance_dtw_enabled", False):
            if METRIC_FRAME_KEY not in source_batch:
                raise ValueError(
                    f"Distance DTW needs {METRIC_FRAME_KEY} from updated EEF transforms"
                )
            metric_anchors = source_batch[METRIC_FRAME_KEY]
            if isinstance(metric_anchors, torch.Tensor):
                metric_anchors = metric_anchors.detach().cpu().numpy()
            metric_anchors = np.asarray(metric_anchors)
            if metric_anchors.shape != (batch_size, 2, 4, 4):
                raise ValueError("DTW anchors must have shape (B,2,4,4)")
        if target_native.ndim != 3 or target_native.shape[0] != batch_size:
            raise ValueError(
                f"open_loop_sim ground truth {target_key!r} must be batched, got "
                f"{target_native.shape}"
            )
        for index in range(batch_size):
            episode = str(episode_hashes[index])
            frame = int(frame_indices[index])
            if not episode or frame < 0:
                raise ValueError(
                    f"invalid open_loop_sim episode/frame: {episode!r}/{frame}"
                )
            prediction_value = pred_native[index]
            self._records.append(
                {
                    "group": self._validation_group or DEFAULT_VALID_GROUP,
                    "source": str(source_id),
                    "label": label,
                    "episode": episode,
                    "frame": frame,
                    # ARC keeps its complete token and native ground-truth
                    # window because its timing payload determines a variable
                    # control-frame stride at episode replay time.
                    "prediction": np.asarray(prediction_value, dtype=np.float32),
                    "ground_truth": np.asarray(target_native[index], dtype=np.float32),
                    **(
                        {METRIC_FRAME_KEY: metric_anchors[index].copy()}
                        if metric_anchors is not None
                        else {}
                    ),
                }
            )

    @torch.inference_mode()
    def on_validation_step(self, batch, batch_idx, dataloader_idx=0):
        del dataloader_idx
        result = self._forward_deterministic(batch)
        for source_id, source_batch in batch.items():
            prediction = result[source_id]["pred_action"]
            if not getattr(self, "video_only", False):
                self._append_source_records(source_id, source_batch, prediction)
            if getattr(
                self, "_video_enabled", False
            ) and self._collective_video_enabled(batch_idx):
                embodiment_id, embodiment_name = self._embodiment(source_batch)
                self._collect_open_loop_video(
                    source_id=source_id,
                    source_batch=source_batch,
                    prediction=prediction,
                    embodiment_id=embodiment_id,
                    embodiment_name=embodiment_name,
                )
        return {}

    @staticmethod
    def _merge_records(states: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
        unique: dict[tuple[str, str, str, int], dict[str, Any]] = {}
        for state in states:
            for record in state:
                key = (
                    record.get("group", DEFAULT_VALID_GROUP),
                    record["source"],
                    record["episode"],
                    int(record["frame"]),
                )
                unique.setdefault(key, record)
        return list(unique.values())

    def _all_records(self) -> list[dict[str, Any]]:
        local = self._records
        if not dist.is_available() or not dist.is_initialized():
            return self._merge_records([local])
        states = [None] * dist.get_world_size()
        dist.all_gather_object(states, local)
        return self._merge_records(states)

    def _score_episode(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        records = sorted(records, key=lambda record: int(record["frame"]))
        frames = [int(record["frame"]) for record in records]
        if len(set(frames)) != len(frames):
            raise RuntimeError("open_loop_sim received duplicate episode frames")
        if self.require_episode_start and frames[0] != 0:
            raise RuntimeError(
                f"open_loop_sim requires complete episodes from frame 0; "
                f"episode {records[0]['episode']!r} starts at frame {frames[0]}"
            )
        expected = set(range(frames[0], frames[-1] + 1))
        missing = sorted(expected.difference(frames))
        if missing:
            raise RuntimeError(
                f"open_loop_sim requires every validation frame in episode "
                f"{records[0]['episode']!r}; missing {len(missing)} frames, "
                f"first missing={missing[0]}"
            )
        by_frame = {frame: record for frame, record in zip(frames, records)}
        end_frame = frames[-1] + 1

        sq = {
            "mse": 0.0,
            "xyz_mse": 0.0,
            "ypr_mse": 0.0,
            "grip_mse": 0.0,
            "paired_mse": 0.0,
        }
        executed = 0
        segments = 0
        segment_control_steps = []
        cursor = frames[0]
        while cursor < end_frame:
            record = by_frame.get(cursor)
            if record is None:
                raise RuntimeError(
                    f"open_loop_sim has no observation at frame {cursor}"
                )
            remaining = end_frame - cursor
            ground_truth = record["ground_truth"]
            if ground_truth.ndim != 2 or ground_truth.shape[1] != 14:
                raise ValueError(
                    "open_loop_sim ground truth must be a control-frequency "
                    f"(T, 14) trajectory, got {ground_truth.shape}"
                )
            prediction, n = self._decode_prediction_with_steps(
                record["prediction"], max_steps=min(len(ground_truth), remaining)
            )
            error = prediction[:n] - ground_truth[:n]
            sq["mse"] += float(np.square(error).sum())
            sq["xyz_mse"] += float(np.square(error[:, XYZ_COLS]).sum())
            sq["ypr_mse"] += float(np.square(error[:, YPR_COLS]).sum())
            sq["grip_mse"] += float(np.square(error[:, GRIP_COLS]).sum())
            sq["paired_mse"] += float(np.square(error[:, PAIRED_COLS]).sum())
            executed += n
            segments += 1
            segment_control_steps.append(n)
            cursor += n

        episode_length = end_frame - frames[0]
        denominators = {
            "mse": executed * 14,
            "xyz_mse": executed * len(XYZ_COLS),
            "ypr_mse": executed * len(YPR_COLS),
            "grip_mse": executed * len(GRIP_COLS),
            "paired_mse": executed * len(PAIRED_COLS),
        }
        return {
            "group": records[0].get("group", DEFAULT_VALID_GROUP),
            "episode": records[0]["episode"],
            "source": records[0]["source"],
            "label": records[0]["label"],
            "executed_steps": executed,
            "segments": segments,
            "segment_control_steps": segment_control_steps,
            "coverage": executed / max(episode_length, 1),
            "metrics": {key: sq[key] / max(denominators[key], 1) for key in sq},
            **(
                {"distance_dtw": score_distance_dtw_episode(self, records)}
                if getattr(self, "distance_dtw_enabled", False)
                else {}
            ),
        }

    @staticmethod
    def _summarize_episodes(episodes: list[dict[str, Any]]) -> dict[str, Any]:
        if not episodes:
            raise RuntimeError("open_loop_sim received no validation episodes")

        total_steps = sum(item["executed_steps"] for item in episodes)
        total_segments = sum(item["segments"] for item in episodes)
        micro = {}
        dimensions = {
            "mse": 14,
            "xyz_mse": len(XYZ_COLS),
            "ypr_mse": len(YPR_COLS),
            "grip_mse": len(GRIP_COLS),
            "paired_mse": len(PAIRED_COLS),
        }
        for key, dims in dimensions.items():
            micro[key] = sum(
                item["metrics"][key] * item["executed_steps"] * dims
                for item in episodes
            ) / max(total_steps * dims, 1)
        labels = defaultdict(list)
        for item in episodes:
            labels[item["label"]].append(item)
        return {
            "episodes": len(episodes),
            "executed_control_steps": total_steps,
            **(
                {"distance_dtw": summarize_distance_dtw(episodes)}
                if "distance_dtw" in episodes[0]
                else {}
            ),
            "segments": total_segments,
            "coverage": float(np.mean([item["coverage"] for item in episodes])),
            "micro": micro,
            "macro": {
                key: float(np.mean([item["metrics"][key] for item in episodes]))
                for key in micro
            },
            "per_label": {
                label: {
                    "episodes": len(items),
                    "executed_control_steps": sum(
                        item["executed_steps"] for item in items
                    ),
                    "coverage": float(np.mean([item["coverage"] for item in items])),
                    "micro": {
                        key: float(
                            np.average(
                                [item["metrics"][key] for item in items],
                                weights=[item["executed_steps"] for item in items],
                            )
                        )
                        for key in micro
                    },
                }
                for label, items in labels.items()
            },
            "episode_results": episodes,
        }

    def _compute_results(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        grouped = defaultdict(list)
        for record in records:
            grouped[
                (
                    record.get("group", DEFAULT_VALID_GROUP),
                    record["source"],
                    record["episode"],
                )
            ].append(record)
        if self.limit_val_episodes is not None:
            grouped_by_source = defaultdict(list)
            for key, group in grouped.items():
                grouped_by_source[key[:2]].append((key[2], group))
            grouped = defaultdict(list)
            for (validation_group, source), episode_groups in grouped_by_source.items():
                for episode, group in sorted(episode_groups)[: self.limit_val_episodes]:
                    grouped[(validation_group, source, episode)] = group
        episodes = [self._score_episode(group) for group in grouped.values()]
        if not episodes:
            raise RuntimeError("open_loop_sim received no validation episodes")
        by_group = defaultdict(list)
        for item in episodes:
            by_group[item["group"]].append(item)
        results = self._summarize_episodes(episodes)
        results.update(
            {
                "execute_fraction": self.execute_fraction,
                "distance_dtw_enabled": getattr(self, "distance_dtw_enabled", False),
                "distance_dtw_metric_version": (
                    METRIC_VERSION
                    if getattr(self, "distance_dtw_enabled", False)
                    else None
                ),
                "execute_control_steps": (
                    self.execute_steps if self.action_mode != "arc" else None
                ),
                "arc_execution_cap_mode": (
                    self.arc_execution_cap_mode if self.action_mode == "arc" else None
                ),
                "arc_chunking_mode": evaluator_chunking_mode(self),
                "arc_episode_progress_semantics": ARC_DISTANCE_SEMANTICS[
                    evaluator_chunking_mode(self)
                ],
                "arc_rotation_distance_unit": getattr(
                    self, "rotation_distance_unit", None
                ),
                "arc_rotation_clock": (
                    "shared_independent"
                    if getattr(self, "rotation_distance_unit", None) is not None
                    else None
                ),
                "execute_arc_waypoints": (
                    self.execute_arc_waypoints
                    if self.action_mode == "arc"
                    and self.arc_execution_cap_mode == "waypoints"
                    else None
                ),
                "execute_arc_distance_m": (
                    self.execute_fraction * self.min_distance_unit
                    if self.action_mode == "arc"
                    and self.arc_execution_cap_mode == "distance"
                    else None
                ),
                "arc_distance_semantics": (
                    ARC_DISTANCE_SEMANTICS[evaluator_chunking_mode(self)]
                    if self.action_mode == "arc"
                    and self.arc_execution_cap_mode == "distance"
                    else None
                ),
                "replan_stride_mode": (
                    f"arc_{self.arc_execution_cap_mode}_timing"
                    if self.action_mode == "arc"
                    else "fixed_control_frames"
                ),
                "control_horizon": self.control_horizon,
                "control_dt": self.control_dt,
                "limit_val_episodes": self.limit_val_episodes,
                "groups": sorted(by_group),
                "per_group": {
                    group: self._summarize_episodes(items)
                    for group, items in sorted(by_group.items())
                },
            }
        )
        return results

    def _metric_tensors(self, results: dict[str, Any]) -> dict[str, torch.Tensor]:
        metrics = {}
        for group, summary in results["per_group"].items():
            prefix = (
                "Valid/open_loop_sim"
                if group == DEFAULT_VALID_GROUP
                else f"Valid_{group}/open_loop_sim"
            )
            if "distance_dtw" in summary:
                dtw = summary["distance_dtw"]
                for name, key in {
                    "XYZ_MSE": "xyz_mse",
                    "Episode_XYZ_MSE": "episode_xyz_mse",
                    "GT_Frames": "gt_frames",
                    "Predicted_Samples": "predicted_samples",
                    "Segments": "segments",
                    "GT_Coverage": "gt_coverage",
                    "Prediction_Coverage": "prediction_coverage",
                    "Duration_Ratio": "duration_ratio",
                }.items():
                    metrics[f"{prefix}/Distance_DTW/{name}"] = dtw[key]
            metrics.update(
                {
                    f"{prefix}/MSE": summary["micro"]["mse"],
                    f"{prefix}/Episode_MSE": summary["macro"]["mse"],
                    f"{prefix}/XYZ_MSE": summary["micro"]["xyz_mse"],
                    f"{prefix}/YPR_MSE": summary["micro"]["ypr_mse"],
                    f"{prefix}/Grip_MSE": summary["micro"]["grip_mse"],
                    f"{prefix}/Paired_MSE": summary["micro"]["paired_mse"],
                    f"{prefix}/Coverage": summary["coverage"],
                    f"{prefix}/Episodes": summary["episodes"],
                    f"{prefix}/Executed_Control_Steps": summary[
                        "executed_control_steps"
                    ],
                    f"{prefix}/Segments": summary["segments"],
                }
            )
        return {
            key: torch.tensor(float(value), dtype=torch.float32).to(self._metric_device)
            for key, value in metrics.items()
        }

    def on_validation_end(self):
        if getattr(self, "video_only", False):
            self.last_results = None
            if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
                return None
            if self.trainer is not None and not getattr(
                self.trainer, "is_global_zero", True
            ):
                return None
            EvalVideo.on_validation_end(self)
            return None

        records = self._all_records()
        results = self._compute_results(records)
        self.last_results = results
        if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
            return results
        if self.trainer is not None and not getattr(
            self.trainer, "is_global_zero", True
        ):
            return results
        # Flush episode buffers and upload the first MP4 per panel after all
        # validation frames have arrived.  This must happen before returning
        # on the rank-zero path; otherwise the last episode never gets a file.
        if getattr(self, "_video_enabled", False):
            EvalVideo.on_validation_end(self)
        # This hook is called from LightningModule.on_validation_end(). Calling
        # LightningModule.log_dict() here recursively enters Lightning's
        # validation-hook guard and raises ``MisconfigurationException``.
        # ``_all_records`` has already merged every rank, so direct logger
        # logging on global zero is both sufficient and deterministic.  Keep
        # values scalar so this works for W&B, TensorBoard, and LoggerCollection
        # without asking Lightning to infer validation-step semantics.
        if self.trainer is not None:
            logger = getattr(self.trainer, "logger", None)
            if logger is not None:
                metrics = {
                    key: float(value.detach().cpu().item())
                    for key, value in self._metric_tensors(results).items()
                }
                logger.log_metrics(metrics, step=self._resolved_log_step())
        if self.results_path is not None:
            self.results_path.parent.mkdir(parents=True, exist_ok=True)
            self.results_path.write_text(json.dumps(results, indent=2) + "\n")
        return results
