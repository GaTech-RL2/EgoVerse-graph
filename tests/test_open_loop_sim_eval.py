from types import SimpleNamespace

import numpy as np
import pytest
import torch

from egomimic.eval.open_loop_sim import (
    OpenLoopSimEval,
    executed_control_steps,
    truncate_arc_token,
)


def _baseline_evaluator(execute_steps=2):
    evaluator = OpenLoopSimEval.__new__(OpenLoopSimEval)
    evaluator.execute_fraction = 0.5
    evaluator.control_horizon = 4
    evaluator.control_dt = 1.0 / 30.0
    evaluator.execute_steps = execute_steps
    evaluator.action_mode = "baseline"
    evaluator.resampled_vector_length = 4
    evaluator.velocity_mode = "mean"
    evaluator.min_distance_unit = 0.4
    evaluator._arc_tokenizer = None
    evaluator.require_episode_start = True
    evaluator.limit_val_episodes = None
    return evaluator


def test_execute_fraction_is_a_control_frequency_prefix():
    assert executed_control_steps(100, 0.01) == 1
    assert executed_control_steps(100, 0.25) == 25
    assert executed_control_steps(100, 1.0) == 100
    with pytest.raises(ValueError, match="execute_fraction"):
        executed_control_steps(100, 0.0)


@pytest.mark.parametrize(
    ("mode", "rows", "expected_rows"),
    [("mean", 101, 26), ("per_waypoint", 200, 50), ("duration", 200, 50)],
)
def test_truncate_arc_token_keeps_matching_timing_rows(mode, rows, expected_rows):
    token = np.arange(rows * 14, dtype=np.float64).reshape(rows, 14)
    truncated = truncate_arc_token(token, 0.25, mode)
    assert truncated.shape == (expected_rows, 14)
    M = rows - 1 if mode == "mean" else rows // 2
    K = 25
    np.testing.assert_array_equal(truncated[:K], token[:K])
    if mode == "mean":
        np.testing.assert_array_equal(truncated[K:], token[M : M + 1])
    else:
        np.testing.assert_array_equal(truncated[K:], token[M : M + K])


def test_open_loop_sim_walks_an_episode_in_executed_prefixes():
    evaluator = _baseline_evaluator()
    records = []
    for frame in range(6):
        records.append(
            {
                "source": "human_bimanual",
                "label": "human_bimanual",
                "episode": "episode-a",
                "frame": frame,
                "prediction": np.ones((4, 14)),
                "ground_truth": np.zeros((4, 14)),
            }
        )

    result = evaluator._score_episode(records)
    assert result["executed_steps"] == 6
    assert result["segments"] == 3
    assert result["coverage"] == pytest.approx(1.0)
    assert result["metrics"]["mse"] == pytest.approx(1.0)
    assert result["metrics"]["xyz_mse"] == pytest.approx(1.0)


def test_open_loop_sim_requires_complete_episode_frames():
    evaluator = _baseline_evaluator()
    records = [
        {
            "source": "human_bimanual",
            "label": "human_bimanual",
            "episode": "episode-a",
            "frame": frame,
            "prediction": np.zeros((4, 14)),
            "ground_truth": np.zeros((4, 14)),
        }
        for frame in (0, 1, 3)
    ]
    with pytest.raises(RuntimeError, match="missing 1 frames"):
        evaluator._score_episode(records)


def test_open_loop_sim_keeps_validation_groups_separate():
    evaluator = _baseline_evaluator()
    records = []
    for group, offset in (("valid", 0.0), ("nested", 1.0)):
        for frame in range(2):
            records.append(
                {
                    "group": group,
                    "source": "human_bimanual",
                    "label": "human_bimanual",
                    "episode": f"episode-{group}",
                    "frame": frame,
                    "prediction": np.full((4, 14), offset + 1.0),
                    "ground_truth": np.full((4, 14), offset),
                }
            )

    result = evaluator._compute_results(records)
    assert result["groups"] == ["nested", "valid"]
    assert result["per_group"]["valid"]["micro"]["mse"] == pytest.approx(1.0)
    assert result["per_group"]["nested"]["micro"]["mse"] == pytest.approx(1.0)


def test_open_loop_sim_limits_complete_episodes_not_batches():
    evaluator = _baseline_evaluator()
    evaluator.limit_val_episodes = 1
    records = []
    for episode in ("episode-b", "episode-a"):
        for frame in range(2):
            records.append(
                {
                    "group": "valid",
                    "source": "human_bimanual",
                    "label": "human_bimanual",
                    "episode": episode,
                    "frame": frame,
                    "prediction": np.ones((4, 14)),
                    "ground_truth": np.zeros((4, 14)),
                }
            )

    result = evaluator._compute_results(records)
    assert result["episodes"] == 1
    assert result["episode_results"][0]["episode"] == "episode-a"


def test_open_loop_sim_detokenizes_only_the_executed_arc_prefix():
    evaluator = OpenLoopSimEval.__new__(OpenLoopSimEval)
    evaluator.execute_fraction = 0.25
    evaluator.execute_steps = 2
    evaluator.action_mode = "auto"
    evaluator.resampled_vector_length = 4
    evaluator.velocity_mode = "duration"
    evaluator.min_distance_unit = 0.4
    evaluator.control_dt = 1.0 / 30.0
    evaluator._arc_tokenizer = None

    token = np.zeros((8, 14), dtype=np.float64)
    token[:4, 0] = np.linspace(0.0, 0.3, 4)
    token[:4, 7] = np.linspace(0.0, 0.3, 4)
    token[4:, 0] = 0.1
    token[4:, 7] = 0.1
    decoded = evaluator._decode_prediction(token)
    assert decoded.shape == (2, 14)
    assert np.isfinite(decoded).all()


def test_open_loop_sim_logs_directly_through_trainer_logger():
    evaluator = _baseline_evaluator()
    evaluator._metric_device = torch.device("cpu")
    evaluator.results_path = None
    evaluator._records = [
        {
            "group": "valid",
            "source": "human_bimanual",
            "label": "human_bimanual",
            "episode": "episode-a",
            "frame": frame,
            "prediction": np.ones((4, 14)),
            "ground_truth": np.zeros((4, 14)),
        }
        for frame in range(2)
    ]
    logged = []
    evaluator.trainer = SimpleNamespace(
        is_global_zero=True,
        global_step=17,
        logger=SimpleNamespace(
            log_metrics=lambda metrics, **kwargs: logged.append((metrics, kwargs))
        ),
    )

    evaluator.on_validation_end()

    assert len(logged) == 1
    metrics, kwargs = logged[0]
    assert kwargs == {"step": 17}
    assert metrics["Valid/open_loop_sim/MSE"] == pytest.approx(1.0)
    assert not any("dataloader_idx" in key for key in metrics)
