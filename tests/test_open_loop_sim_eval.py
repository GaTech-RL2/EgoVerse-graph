import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from egomimic.eval.open_loop_sim import (
    OpenLoopSimEval,
    arc_prefix_control_steps,
    executed_control_steps,
    truncate_arc_token,
)
from egomimic.eval.video import EvalVideo


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
    [("mean", 101, 27), ("per_waypoint", 200, 52), ("duration", 200, 52)],
)
def test_truncate_arc_token_keeps_matching_timing_rows(mode, rows, expected_rows):
    token = np.arange(rows * 14, dtype=np.float64).reshape(rows, 14)
    truncated = truncate_arc_token(token, 0.25, mode)
    assert truncated.shape == (expected_rows, 14)
    M = rows - 1 if mode == "mean" else rows // 2
    K = 26
    np.testing.assert_array_equal(truncated[: K - 1], token[: K - 1])
    np.testing.assert_allclose(
        truncated[K - 1], 0.25 * token[24] + 0.75 * token[25]
    )
    if mode == "mean":
        assert truncated[K:].shape == (1, 14)
    else:
        timing = truncated[K:]
        if mode == "per_waypoint":
            np.testing.assert_array_equal(timing, token[M : M + K])
        else:
            expected = token[M : M + K].copy()
            expected[24, (0, 7)] *= 0.75
            np.testing.assert_allclose(timing, expected)


def _duration_arc_token(M: int, interval_duration: float) -> np.ndarray:
    token = np.zeros((2 * M, 14), dtype=np.float64)
    positions = np.linspace(0.0, 0.4, M)
    token[:M, 0] = positions
    token[:M, 7] = positions
    token[M:, 0] = interval_duration
    token[M:, 7] = interval_duration
    return token


def test_arc_prefix_uses_exact_distance_and_recovers_its_frame_stride():
    dt = 1.0 / 30.0
    token = _duration_arc_token(5, dt)
    partial = truncate_arc_token(token, 0.375, "duration")

    # 37.5% lies halfway between waypoint rows 1 and 2.
    np.testing.assert_allclose(partial[2, (0, 7)], [0.15, 0.15])
    # One complete interval plus half an interval spans 1.5 control periods.
    assert arc_prefix_control_steps(token, 0.375, "duration", dt) == 2


def test_arc_prefix_uses_total_bimanual_cumulative_distance():
    token = np.zeros((8, 14), dtype=np.float64)
    token[:4, 0] = [0.0, 0.1, 0.9, 1.0]
    token[:4, 7] = [0.0, 0.1, 0.2, 0.3]
    token[4:, 0] = 1.0 / 30.0
    token[4:, 7] = 1.0 / 30.0

    partial = truncate_arc_token(token, 0.5, "duration")

    # Total bimanual distance is 1.3m. The 0.65m target lies 50% through
    # interval 1 (whose combined distance is 0.9m), not at waypoint index 2.
    np.testing.assert_allclose(partial[2, 0], 0.5)
    np.testing.assert_allclose(partial[2, 7], 0.15)
    np.testing.assert_allclose(partial[4, 0], (1.0 / 30.0) * 0.5)
    np.testing.assert_allclose(partial[4, 7], (1.0 / 30.0) * 0.5)


def test_arc_replan_stride_changes_with_predicted_timing():
    evaluator = OpenLoopSimEval.__new__(OpenLoopSimEval)
    evaluator.execute_fraction = 0.5
    evaluator.execute_steps = 25
    evaluator.control_horizon = 100
    evaluator.control_dt = 1.0 / 30.0
    evaluator.action_mode = "arc"
    evaluator.resampled_vector_length = 5
    evaluator.velocity_mode = "duration"
    evaluator.min_distance_unit = 0.4
    evaluator._arc_tokenizer = None
    evaluator.require_episode_start = True
    evaluator.limit_val_episodes = None

    def records(token):
        return [
            {
                "source": "yam_bimanual",
                "label": "yam_bimanual",
                "episode": "episode-a",
                "frame": frame,
                "prediction": token,
                "ground_truth": np.zeros((100, 14)),
            }
            for frame in range(12)
        ]

    fast = evaluator._score_episode(records(_duration_arc_token(5, evaluator.control_dt)))
    slow = evaluator._score_episode(
        records(_duration_arc_token(5, 3.0 * evaluator.control_dt))
    )

    # The same 50% distance prefix spans 2 frames at the fast timing and 6 at
    # the slow timing, so the oracle-observation replanning boundaries differ.
    assert fast["segments"] == 6
    assert slow["segments"] == 2


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
    assert decoded.shape == (3, 14)
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


def test_open_loop_video_writes_one_full_episode_mp4(tmp_path, monkeypatch):
    evaluator = _baseline_evaluator()
    evaluator.video_output_dir = tmp_path
    evaluator.video_chunk_frames = 1000
    evaluator.max_episode_frames = 100
    evaluator.viz_every_n_epochs = 1
    evaluator.viz_max_batches = None
    evaluator.trainer = SimpleNamespace(
        is_global_zero=True,
        current_epoch=0,
        world_size=1,
        default_root_dir=str(tmp_path),
        logger=None,
    )
    written = []

    def fake_write_video(path, frames, **kwargs):
        written.append((path, tuple(frames.shape), kwargs))

    monkeypatch.setattr("egomimic.eval.video.tvio.write_video", fake_write_video)
    EvalVideo.on_validation_start(evaluator)
    buf_key = ("valid", "yam_bimanual")
    out_dir = evaluator._group_video_dir(*buf_key)
    frame = torch.zeros((4, 4, 3), dtype=torch.uint8)
    evaluator._buffer_per_episode(
        buf_key, out_dir, [frame, frame, frame], ["episode-a"] * 3
    )
    assert written == []
    evaluator._buffer_per_episode(buf_key, out_dir, [frame], ["episode-b"])
    EvalVideo.on_validation_end(evaluator)

    assert [item[0].rsplit("/", 1)[-1] for item in written] == [
        "episode-a.mp4",
        "episode-b.mp4",
    ]
    assert [item[1][0] for item in written] == [3, 1]


def test_open_loop_video_uploads_only_first_episode_per_panel(monkeypatch):
    evaluator = _baseline_evaluator()
    logged = []
    experiment = SimpleNamespace(
        id="run-id",
        log=lambda payload, **kwargs: logged.append((payload, kwargs)),
    )
    evaluator.trainer = SimpleNamespace(
        is_global_zero=True,
        world_size=1,
        global_step=23,
        logger=SimpleNamespace(experiment=experiment),
    )
    evaluator._written_paths = [
        ("valid", "yam_bimanual", "/tmp/episode-a.mp4"),
        ("valid", "yam_bimanual", "/tmp/episode-b.mp4"),
        ("valid", "human_bimanual", "/tmp/episode-c.mp4"),
        ("nested", "yam_bimanual", "/tmp/episode-d.mp4"),
    ]
    fake_wandb = SimpleNamespace(
        Video=lambda path, **kwargs: {"path": path, **kwargs}
    )
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    OpenLoopSimEval._log_wandb_videos(evaluator)

    assert len(logged) == 1
    payload, kwargs = logged[0]
    assert kwargs == {"step": 23}
    assert sorted(payload) == [
        "Val_video/human_bimanual",
        "Val_video/yam_bimanual",
        "Val_video_nested/yam_bimanual",
    ]
    assert payload["Val_video/yam_bimanual"]["path"] == "/tmp/episode-a.mp4"
