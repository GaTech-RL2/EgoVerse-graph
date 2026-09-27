"""E1 tokens in the distance-budget DTW rollout (egomimic/eval/e1_open_loop.py)."""

from types import SimpleNamespace

import numpy as np
import pytest

from egomimic.eval.distance_budget_dtw import METRIC_FRAME_KEY, joint_cumulative_distance, world_xyz
from egomimic.eval.e1_open_loop import (
    build_e1_tokenizer,
    e1_coverage_windows,
    e1_execute_prefix,
    joint_progress,
    score_e1_distance_dtw_episode,
)

FRACTION = 29 / 99


def _evaluator(mode, window, oracle=True):
    return SimpleNamespace(
        resampled_vector_length=100, execute_fraction=0.30, min_distance_unit=0.40,
        control_dt=1.0 / 30.0, dtw_max_cells=50_000_000, dtw_max_prediction_steps=30_000,
        e1_velocity_mode=mode, e1_velocity_norm="path", e1_min_speed=0.01,
        e1_source_frames=window, oracle_predictions=oracle,
    )


def _symmetric(T=900, seed=0):
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


def _asymmetric(T=900):
    """Left arm fast and continuous; right arm slow, pausing, then fast (real bimanual folding)."""
    t = np.arange(T) / 30.0
    raw = np.zeros((T, 14))
    raw[:, 0] = 0.3 * np.cos(0.9 * t)
    raw[:, 1] = 0.2 * np.sin(0.9 * t)
    raw[:, 2] = 0.02 * t
    right_speed = np.where(t < 8, 0.05, np.where(t < 14, 0.0, 0.5))
    phase = np.cumsum(right_speed) / 30.0
    raw[:, 7] = 0.2 + 0.25 * np.sin(phase * 3)
    raw[:, 8] = 0.3 * phase
    raw[:, 13] = np.clip(np.sin(t), 0, 1)
    return raw


def _window(raw, f, length):
    chunk = raw[f : f + length]
    return np.concatenate([chunk, np.repeat(chunk[-1:], length - len(chunk), 0)]) if len(chunk) < length else chunk


def _records(raw, evaluator):
    tokenizer = build_e1_tokenizer(evaluator)
    eye = np.repeat(np.eye(4)[None], 2, 0)
    return [{"frame": f, "ground_truth": _window(raw, f, 100), METRIC_FRAME_KEY: eye,
             "prediction": tokenizer.transform({"actions_cartesian": _window(raw, f, evaluator.e1_source_frames).copy()})["actions_cartesian"]}
            for f in range(len(raw))]


def _oracle(raw, mode, window):
    evaluator = _evaluator(mode, window)
    records = _records(raw, evaluator)
    gt = np.concatenate([world_xyz(r["ground_truth"][:1], r[METRIC_FRAME_KEY]) for r in records])
    return score_e1_distance_dtw_episode(evaluator, records, gt, joint_cumulative_distance(gt))


@pytest.mark.parametrize("window", [100, 400])
@pytest.mark.parametrize("trajectory", [_symmetric, _asymmetric])
def test_oracle_dur_reproduces_gt(trajectory, window):
    result = _oracle(trajectory(), "dur", window)
    assert result["rollout_mode"] == "gt_e1_joint_coverage" and result["gt_coverage"] == 1.0
    assert result["xyz_mse"] < 1.5e-4, result["xyz_mse"]          # < ~12 mm RMS for a perfect model (frame quantisation)
    # E1 clamps each interval to length / min_speed, so a long ONE-arm pause inside
    # the window (the asymmetric case, W=400) plays back compressed: shorter rollout.
    low = 0.9 if trajectory is _symmetric else 0.75
    assert low < result["duration_ratio"] < 1.1, result["duration_ratio"]


@pytest.mark.parametrize("window", [100, 400])
@pytest.mark.parametrize("trajectory", [_symmetric, _asymmetric])
def test_oracle_profile_floor_is_small(trajectory, window):
    result = _oracle(trajectory(), "profile", window)                # smoothed speeds: a small floor
    assert result["xyz_mse"] < 1e-4, result["xyz_mse"]
    assert 0.8 < result["duration_ratio"] < 1.25, result["duration_ratio"]


def test_prefix_stops_at_the_joint_progress_budget():
    evaluator = _evaluator("dur", 100)
    token = _records(_asymmetric()[:200], evaluator)[20]["prediction"]
    tokenizer = build_e1_tokenizer(evaluator)
    samples, n = e1_execute_prefix(tokenizer, token, progress_fraction=FRACTION, control_dt=evaluator.control_dt)
    assert samples.shape == (n, 14) and n > 1
    progress = joint_progress(samples)
    from egomimic.rldb.zarr.e1_arc_tokenizer import cumulative_arc_length
    budget = FRACTION * sum(float(cumulative_arc_length(token[:, s])[-1]) for s in (slice(0, 3), slice(7, 10)))
    assert progress[-1] <= budget + 1e-9 and progress[-1] > 0.8 * budget
    half, n_half = e1_execute_prefix(tokenizer, token, progress_fraction=FRACTION, control_dt=evaluator.control_dt, budget_fraction=0.5)
    assert n_half < n
    assert e1_execute_prefix(tokenizer, token, progress_fraction=FRACTION, control_dt=evaluator.control_dt, max_steps=3)[1] == 3


def test_stationary_episode_is_one_full_prefix():
    xyz = np.zeros((50, 6))
    anchors, fractions, coverages = e1_coverage_windows(
        xyz, joint_cumulative_distance(xyz), window_frames=100, min_distance_unit=0.4, progress_fraction=FRACTION)
    assert anchors.tolist() == [0] and fractions.tolist() == [1.0] and coverages.tolist() == [0.0]


def test_anchor_step_is_gt_token_coverage():
    t = np.arange(400) / 30.0
    xyz = np.zeros((400, 6))
    xyz[:, 0] = 0.3 * t            # left arm 0.3 m/s, right arm still
    anchors, fractions, coverages = e1_coverage_windows(
        xyz, joint_cumulative_distance(xyz), window_frames=100, min_distance_unit=0.4, progress_fraction=FRACTION)
    full = anchors + 40 <= len(xyz)   # >= 0.4 m of left-arm path left inside the window
    assert full.sum() > 20 and np.allclose(coverages[full], 0.4 * FRACTION, atol=1e-9)
    assert np.all(np.diff(anchors) >= 0) and anchors[0] == 0 and 0 < fractions[-1] <= 1
    assert len(anchors) < 60                                    # no geometric tail of repeated end anchors


def test_pause_is_one_prefix_then_resumed_motion():
    xyz = np.zeros((300, 6))
    xyz[:100, 0] = np.linspace(0, 1.0, 100)
    xyz[100:200, 0] = 1.0
    xyz[200:, 0] = 1.0 + np.linspace(0, 1.0, 100)
    anchors, fractions, coverages = e1_coverage_windows(
        xyz, joint_cumulative_distance(xyz), window_frames=100, min_distance_unit=0.4, progress_fraction=FRACTION)
    still = anchors[(coverages == 0) & (anchors < 200)]
    assert len(still) == 1 and 99 <= still[0] < 200
    assert anchors[np.argmax(anchors > still[0])] >= 200
