"""E1 tokens in the distance-budget DTW rollout (egomimic/eval/e1_open_loop.py)."""

from types import SimpleNamespace

import numpy as np
import pytest

from egomimic.eval.distance_budget_dtw import METRIC_FRAME_KEY, joint_cumulative_distance, world_xyz
from egomimic.eval.e1_open_loop import (
    build_e1_tokenizer,
    e1_coverage_windows,
    e1_fractional_prefix,
    e1_prefix_control_steps,
    e1_waypoint_prefix,
    score_e1_distance_dtw_episode,
)


def _evaluator(mode, window, oracle=True):
    return SimpleNamespace(
        resampled_vector_length=100, execute_fraction=0.30, min_distance_unit=0.40,
        control_dt=1.0 / 30.0, dtw_max_cells=50_000_000, dtw_max_prediction_steps=30_000,
        e1_velocity_mode=mode, e1_velocity_norm="path", e1_min_speed=0.01,
        e1_source_frames=window, oracle_predictions=oracle,
    )


def _trajectory(T=900, seed=0):
    """Two arms on smooth curves with speed changes and a pause in the middle."""
    rng = np.random.default_rng(seed)
    t = np.arange(T) / 30.0
    speed = 0.6 + 0.4 * np.sin(0.7 * t)
    speed[300:360] = 0.0
    phase = np.cumsum(speed) / 30.0
    raw = np.zeros((T, 14))
    for off, sign in ((0, 1.0), (7, -1.0)):
        raw[:, off + 0] = 0.25 * np.cos(sign * phase) + 0.02 * rng.normal()
        raw[:, off + 1] = 0.20 * np.sin(1.3 * phase)
        raw[:, off + 2] = 0.05 * phase
        raw[:, off + 3] = 0.3 * np.sin(0.5 * phase)
        raw[:, off + 6] = 0.5 + 0.4 * np.sin(phase)
    return raw


def _window(raw, f, length):
    chunk = raw[f : f + length]
    return np.concatenate([chunk, np.repeat(chunk[-1:], length - len(chunk), 0)]) if len(chunk) < length else chunk


def _records(raw, evaluator):
    tokenizer = build_e1_tokenizer(evaluator)
    eye = np.repeat(np.eye(4)[None], 2, 0)
    records = []
    for f in range(len(raw)):
        batch = tokenizer.transform({"actions_cartesian": _window(raw, f, evaluator.e1_source_frames).copy()})
        records.append({"frame": f, "ground_truth": _window(raw, f, 100), "prediction": batch["actions_cartesian"],
                        METRIC_FRAME_KEY: eye})
    return records


@pytest.mark.parametrize("mode", ["dur", "profile"])
@pytest.mark.parametrize("window", [100, 400])
def test_oracle_rollout_reproduces_gt(mode, window):
    raw = _trajectory()
    evaluator = _evaluator(mode, window)
    records = _records(raw, evaluator)
    gt = np.concatenate([world_xyz(r["ground_truth"][:1], r[METRIC_FRAME_KEY]) for r in records])
    result = score_e1_distance_dtw_episode(evaluator, records, gt, joint_cumulative_distance(gt))
    assert result["rollout_mode"] == "gt_e1_token_coverage"
    assert result["gt_coverage"] == 1.0
    assert result["xyz_mse"] < 2e-5, result["xyz_mse"]          # < ~4.5 mm RMS for a perfect model
    assert 0.8 < result["duration_ratio"] < 1.25, result["duration_ratio"]


def test_prefix_keeps_m_rows_and_steps_follow_the_clock():
    raw = _trajectory()
    evaluator = _evaluator("dur", 100)
    token = _records(raw[:200], evaluator)[10]["prediction"]
    prefix = e1_waypoint_prefix(token, 0.30)
    assert prefix.shape == (30, 16)
    tokenizer = build_e1_tokenizer(evaluator)
    steps = e1_prefix_control_steps(tokenizer, prefix, evaluator.control_dt)
    expected = max(float(c[-1]) for c in tokenizer.clock_at_waypoints(prefix))
    assert steps == max(1, int(np.ceil(expected * 30 - 1e-9)))
    assert e1_prefix_control_steps(tokenizer, prefix, evaluator.control_dt, max_steps=3) == 3


def test_fractional_prefix():
    prefix = np.tile(np.arange(30, dtype=float)[:, None], (1, 16))
    prefix[:, 14:16] = 0.1
    assert np.array_equal(e1_fractional_prefix(prefix, 1.0, "dur"), prefix)
    half = e1_fractional_prefix(prefix, 0.5 / 29 + 3 / 29, "dur")   # 3.5 waypoints in
    assert half.shape == (5, 16)
    assert np.isclose(half[-1, 0], 3.5) and np.allclose(half[3, 14:16], 0.05)
    exact = e1_fractional_prefix(prefix, 3 / 29, "profile")
    assert exact.shape == (4, 16)
    with pytest.raises(ValueError):
        e1_fractional_prefix(prefix, 0.5, "logdur")


def test_stationary_episode_is_one_full_prefix():
    xyz = np.zeros((50, 6))
    anchors, budgets, coverages = e1_coverage_windows(
        xyz, joint_cumulative_distance(xyz), window_frames=100, min_distance_unit=0.4, waypoint_fraction=29 / 99)
    assert anchors.tolist() == [0] and coverages.tolist() == [0.0]


def test_anchor_step_is_gt_token_coverage():
    t = np.arange(400) / 30.0
    xyz = np.zeros((400, 6))
    xyz[:, 0] = 0.3 * t            # left arm 0.3 m/s, right arm still
    cum = joint_cumulative_distance(xyz)
    anchors, budgets, coverages = e1_coverage_windows(
        xyz, cum, window_frames=100, min_distance_unit=0.4, waypoint_fraction=29 / 99)
    # 100 frames at 0.3 m/s = 0.99 m of path, capped at D = 0.4 for the left arm only
    full = anchors + 40 <= len(xyz)   # >= 0.4 m of left-arm path left inside the window
    assert full.sum() > 20 and np.allclose(coverages[full], 0.4 * 29 / 99, atol=1e-9)
    assert np.all(np.diff(anchors) >= 0) and anchors[0] == 0
