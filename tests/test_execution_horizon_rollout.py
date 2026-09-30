"""Regression checks for execution-boundary replanning and held video plans."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from egomimic.eval import distance_budget_dtw as dtw
from tests.test_distance_budget_dtw import evaluator, records, wide_records


def test_dtw_uses_same_temporal_boundaries_as_pointwise_rollout(monkeypatch):
    ev = evaluator("baseline")
    samples = records("baseline")
    for item in samples:
        item["prediction"][:, 0] = item["frame"]
    captured = []
    original = dtw.global_dtw

    def capture(pred, gt, **kw):
        captured.append(pred.copy())
        return original(pred, gt, **kw)

    monkeypatch.setattr(dtw, "global_dtw", capture)
    scored = ev._score_episode(samples)
    score = scored["distance_dtw"]
    assert score["anchor_frames"] == [0, 30, 60]
    assert (
        score["segment_control_steps"] == scored["segment_control_steps"] == [30, 30, 1]
    )
    np.testing.assert_allclose(captured[0][:30, 0], 0)
    np.testing.assert_allclose(captured[0][30:60, 0], 30.12)
    assert score["predicted_samples"] == score["gt_frames"] == 61
    assert score["rollout_mode"] == "execution_horizon"
    assert score["execution_distance_budget_m"] is None


@pytest.mark.parametrize("mode", ["race", "multistream", "joint_distance"])
@pytest.mark.parametrize("speed", [0.04, 0.2, 0.8])
def test_m28_decodes_one_30_waypoint_prefix_per_plan(mode, speed, monkeypatch):
    from egomimic.rldb.zarr.arc_length_tokenizer import (
        TokenizeBimanualArcLengthCartesian,
    )

    ev = evaluator(hybrid=True)
    ev.arc_chunking_mode = mode
    seen = []
    original = TokenizeBimanualArcLengthCartesian.detokenize

    def capture(self, token, *args, **kw):
        seen.append(np.asarray(token).copy())
        return original(self, token, *args, **kw)

    monkeypatch.setattr(TokenizeBimanualArcLengthCartesian, "detokenize", capture)
    score = dtw.score_distance_dtw_episode(ev, wide_records(records(speed=speed)))
    assert len(seen) == score["segments"]
    for token in seen:
        # Every token decoded by the real scorer retains 30 shape/rate pairs.
        assert token.shape == (60, 14)
        assert token[29, 0] == pytest.approx(0.2 * 29 / 99)
        np.testing.assert_allclose(token[30:, 0], speed)
    assert sum(score["segment_control_steps"]) == 61
    assert score["gt_coverage"] == score["prediction_coverage"] == 1
    assert score["anchor_frames"] == [0] + list(
        np.cumsum(score["segment_control_steps"])[:-1]
    )


def test_dtw_replanning_depends_on_predicted_duration_not_gt_distance():
    fast = dtw.score_distance_dtw_episode(evaluator(), records(speed=0.8))
    slow = dtw.score_distance_dtw_episode(evaluator(), records(speed=0.04))
    assert fast["segments"] > slow["segments"]
    assert fast["anchor_frames"] != slow["anchor_frames"]


def test_video_keeps_anchor_prediction_and_gt_until_30_frame_boundary():
    ev = evaluator("baseline")
    ev._video_enabled = True
    ev.trainer = SimpleNamespace(is_global_zero=True)
    ev._validation_group = "valid"
    ev.action_key = ev.ground_truth_action_key = "actions"
    ev.obs_pose_key = "pose"
    ev.image_key = "image"
    ev._native = ev._native_pose = lambda value, embodiment: value
    ev._native_key = lambda value, key, embodiment: value
    ev._video_execution_segments = {}
    calls = []
    ev._render_open_loop_video_frame = lambda **kw: calls.append(kw)
    # Failure if the legacy post-decode distance cap ever re-enters the path.
    ev._cap_arc_video_trajectories = lambda *a: pytest.fail("second video cap")
    for frame in range(61):
        world = torch.eye(4).repeat(1, 2, 1, 1)
        world[:, :, 0, 3] = frame * 0.01
        pose = torch.zeros(1, 14)
        # Moving camera: EEF x in current camera = world x - camera x.
        pose[:, [0, 7]] = frame * 0.01 - frame * 0.002
        batch = dict(
            actions=torch.full((1, 100, 14), float(frame)),
            pose=pose,
            image=torch.tensor([frame]),
            frame_index=torch.tensor([frame]),
            episode_hash=["one"],
            **{dtw.METRIC_FRAME_KEY: world},
        )
        ev._collect_open_loop_video(
            source_id="s",
            source_batch=batch,
            prediction=torch.full((1, 100, 14), float(frame)),
            embodiment_id=7,
            embodiment_name="yam_bimanual",
        )
    ev._flush_video_execution_segments()
    assert len(calls) == 61
    for frame, call in enumerate(calls):
        anchor = frame // 30 * 30
        assert call["executed_prefix"][1] == [min(30, 61 - anchor)]
        assert call["executed_prefix"][0][0, 0, 0] == anchor
        assert call["source_batch"]["actions"][0, 0, 0] == anchor
        assert call["source_batch"]["image"].item() == frame
        assert call["executed_ground_truth"].shape == (1, min(30, 61 - anchor), 14)
        assert call["anchor_pose"][0, 0] == pytest.approx(
            anchor * 0.01 - frame * 0.002, abs=1e-6
        )
    ev.model = None
    ev._video_enabled = False
    ev.on_validation_start()
    assert ev._video_execution_segments == {}


def test_rendered_arc_overlay_preserves_execution_path_beyond_old_distance_cap(
    monkeypatch,
):
    from egomimic.eval import open_loop_sim as module

    ev = evaluator(hybrid=True)
    ev.arc_chunking_mode = "race"
    ev._video_enabled = True
    ev.trainer = SimpleNamespace(is_global_zero=True)
    ev._validation_group = "valid"
    ev.action_key = ev.ground_truth_action_key = "actions"
    ev.obs_pose_key = "pose"
    ev.image_key = "image"
    ev._native = ev._native_pose = lambda value, embodiment: value
    ev._native_key = lambda value, key, embodiment: value
    ev._revert_to_camframe = lambda **kw: kw["actions"]
    ev._write_trajectory_snapshot = lambda **kw: None
    ev._group_video_dir = lambda *args: "/tmp"
    ev._buffer_per_episode = lambda *args: None
    ev.arc_video_trajectory_cap_mode = "distance"  # old saved configuration
    ev._cap_arc_video_trajectories = lambda *args: pytest.fail("second distance cap")
    overlays = []

    def viz(**kw):
        overlays.append(kw)
        return np.zeros((1, 4, 4, 3), dtype=np.uint8)

    ev.viz_func = {"yam_bimanual": viz}
    monkeypatch.setattr(module, "overlay_annotation_fields", lambda *args: {})
    sample = wide_records(records(speed=0.8))[0]["prediction"]
    sample[:, [0, 7]] *= 4  # overshoot must remain visible/scored
    prediction = torch.tensor(sample[None], dtype=torch.float32)
    decoded, lengths = ev._decoded_video_predictions(prediction, 7, 100)
    assert decoded[0, -1, 0] > ev.execute_fraction * ev.min_distance_unit
    batch = dict(
        actions=torch.zeros(1, 100, 14),
        pose=torch.zeros(1, 14),
        image=torch.zeros(1, 3, 4, 4),
        embodiment=torch.tensor([7]),
        episode_hash=["e"],
    )
    ev._render_open_loop_video_frame(
        source_id="s",
        source_batch=batch,
        prediction=prediction,
        embodiment_id=7,
        embodiment_name="yam_bimanual",
        executed_prefix=(decoded, lengths),
    )
    output = overlays[0]["predictions"]["yam_bimanual_actions"]
    torch.testing.assert_close(output, decoded)
    assert overlays[0]["batch"]["actions"].shape == decoded.shape


def test_short_stored_gt_chunks_do_not_shorten_the_execution_horizon():
    ev = evaluator()
    full = records(speed=0.04)
    short = [dict(item, ground_truth=item["ground_truth"][:5]) for item in full]
    expected = dtw.score_distance_dtw_episode(ev, full)
    actual = ev._score_episode(short)
    assert actual["distance_dtw"]["anchor_frames"] == expected["anchor_frames"]
    assert actual["segment_control_steps"] == expected["segment_control_steps"]
    assert actual["segment_control_steps"][0] > 5
    assert actual["distance_dtw"]["xyz_mse"] == pytest.approx(expected["xyz_mse"])
