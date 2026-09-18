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
def test_truncate_arc_token_uses_combined_distance_and_keeps_timing(
    mode, rows, expected_rows
):
    M = rows - 1 if mode == "mean" else rows // 2
    token = np.zeros((rows, 14), dtype=np.float64)
    token[:M, 0] = np.linspace(0.0, 0.4, M)
    truncated = truncate_arc_token(token, 0.25, mode, 0.4)
    assert truncated.shape == (expected_rows, 14)
    K = 26
    assert truncated[K - 1, 0] == pytest.approx(0.1)
    np.testing.assert_array_equal(truncated[: K - 1], token[: K - 1])
    if mode == "mean":
        np.testing.assert_array_equal(truncated[K:], token[M : M + 1])
    else:
        assert truncated[K:].shape == (K, 14)


def _duration_arc_token(M: int, interval_duration: float) -> np.ndarray:
    token = np.zeros((2 * M, 14), dtype=np.float64)
    positions = np.linspace(0.0, 0.4, M)
    token[:M, 0] = positions
    token[:M, 7] = positions
    token[M:, 0] = interval_duration
    token[M:, 7] = interval_duration
    return token


def _per_waypoint_arc_token(M: int, interval_duration: float) -> np.ndarray:
    token = np.zeros((2 * M, 14), dtype=np.float64)
    positions = np.linspace(0.0, 0.4, M)
    token[:M, 0] = positions
    token[:M, 7] = positions
    interval_speed = (positions[1] - positions[0]) / interval_duration
    token[M:, 0] = interval_speed
    token[M:, 7] = interval_speed
    return token


def test_arc_prefix_executes_30_percent_of_combined_D_and_recovers_stride():
    dt = 1.0 / 30.0
    token = _duration_arc_token(100, dt)
    partial = truncate_arc_token(token, 0.30, "duration", 0.4)

    K = len(partial) // 2
    combined = np.linalg.norm(np.diff(partial[:K, 0:3], axis=0), axis=-1)
    combined += np.linalg.norm(np.diff(partial[:K, 7:10], axis=0), axis=-1)
    assert combined.sum() == pytest.approx(0.12)
    # Both arms move, so combined D is reached halfway as many waypoint rows
    # as the old per-arm interpretation.
    assert K == 16
    assert arc_prefix_control_steps(token, 0.30, "duration", dt, 0.4) == 15


def test_arc_prefix_video_stride_follows_predicted_waypoint_timing():
    dt = 1.0 / 30.0
    token = _per_waypoint_arc_token(100, 2.0 * dt)

    assert arc_prefix_control_steps(token, 0.30, "per_waypoint", dt, 0.4) == 30


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

    fast = evaluator._score_episode(
        records(_duration_arc_token(5, evaluator.control_dt))
    )
    slow = evaluator._score_episode(
        records(_duration_arc_token(5, 3.0 * evaluator.control_dt))
    )

    # The same combined-distance prefix spans 1 frame at the fast timing and
    # 3 at the slow timing, so the oracle-observation boundaries differ.
    assert fast["segments"] == 12
    assert slow["segments"] == 4


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
    assert result["segment_control_steps"] == [2, 2, 2]
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


def test_open_loop_sim_uses_checkpoint_step_for_posthoc_logging():
    evaluator = _baseline_evaluator()
    evaluator.log_step = 40_000
    evaluator.trainer = SimpleNamespace(global_step=0)

    assert evaluator._resolved_log_step() == 40_000


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
    fake_wandb = SimpleNamespace(Video=lambda path, **kwargs: {"path": path, **kwargs})
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


def test_open_loop_video_decodes_full_baseline_chunk_at_every_frame():
    evaluator = _baseline_evaluator()
    evaluator._native = lambda value, embodiment_id: value
    prediction = torch.stack([torch.full((4, 14), float(frame)) for frame in range(3)])

    decoded = evaluator._decoded_video_predictions(
        prediction, embodiment_id=0, max_steps=4
    )

    assert decoded.shape == (3, 4, 14)
    assert decoded[:, 0, 0].tolist() == [0, 1, 2]
    assert decoded[:, -1, 0].tolist() == [0, 1, 2]


def test_open_loop_video_detokenizes_full_arc_chunk():
    evaluator = _baseline_evaluator()
    evaluator.action_mode = "arc"
    evaluator.resampled_vector_length = 4
    evaluator.velocity_mode = "duration"
    evaluator._native = lambda value, embodiment_id: value
    calls = []

    class Tokenizer:
        def detokenize(self, token, action_horizon):
            calls.append((token.copy(), action_horizon))
            return np.full((action_horizon, 14), token[0, 0])

    evaluator._arc_tokenizer = Tokenizer()
    token = torch.zeros((2, 8, 14))
    token[0, 0, 0] = 1.0
    token[1, 0, 0] = 2.0

    decoded = evaluator._decoded_video_predictions(
        token, embodiment_id=0, max_steps=100
    )

    assert decoded.shape == (2, 100, 14)
    assert [action_horizon for _, action_horizon in calls] == [100, 100]
    assert all(value.shape == (8, 14) for value, _ in calls)
    assert decoded[:, 0, 0].tolist() == [1, 2]


def test_video_only_flushes_videos_without_computing_metrics(monkeypatch):
    evaluator = _baseline_evaluator()
    evaluator.video_only = True
    evaluator._video_enabled = True
    evaluator.last_results = "unset"
    evaluator.trainer = SimpleNamespace(is_global_zero=True)
    evaluator._all_records = lambda: pytest.fail("video-only mode scored records")
    flushed = []
    monkeypatch.setattr(
        EvalVideo, "on_validation_end", lambda self: flushed.append(True)
    )

    assert evaluator.on_validation_end() is None
    assert evaluator.last_results is None
    assert flushed == [True]
