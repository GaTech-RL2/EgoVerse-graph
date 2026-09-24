from types import SimpleNamespace

import pytest
import torch

import egomimic.eval.open_loop_sim as module
from egomimic.eval.open_loop_sim import OpenLoopSimEval


def evaluator():
    ev = OpenLoopSimEval.__new__(OpenLoopSimEval)
    ev.control_dt = 1 / 30
    ev.trainer = SimpleNamespace(world_size=2, is_global_zero=True, current_epoch=0)
    ev._validation_group = "valid"
    ev.image_key = "observations.images.front_img_1"
    ev.viz_every_n_epochs = 1
    ev.viz_max_batches = None
    return ev


def payload(frames, episode="episode-z"):
    return dict(
        source_id="7",
        source_batch={
            "episode_hash": [episode] * len(frames),
            "frame_index": torch.tensor(frames),
        },
        prediction=torch.zeros(len(frames), 100, 14),
        embodiment_id=7,
        embodiment_name="yam_bimanual",
    )


def fake_group(monkeypatch, other, rank=0):
    monkeypatch.setattr(module.dist, "is_available", lambda: True)
    monkeypatch.setattr(module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(module.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(module.dist, "get_rank", lambda: rank)

    def gather(parts, local):
        parts[rank] = local
        parts[1 - rank] = other

    monkeypatch.setattr(module.dist, "all_gather_object", gather)


def test_rank_zero_collects_every_frame_in_order_and_deduplicates_padding(monkeypatch):
    ev = evaluator()
    calls = []
    ev._maybe_log_open_loop_video = lambda **kw: calls.append(
        int(kw["source_batch"]["frame_index"][0])
    )
    fake_group(monkeypatch, payload([1, 3]))
    ev._collect_open_loop_video(**payload([0, 2]))
    fake_group(monkeypatch, payload([5, 0]))
    ev._collect_open_loop_video(**payload([4, 6]))
    assert calls == list(range(7))
    assert ev._video_fps() == 30


def test_nonzero_rank_participates_but_does_not_write(monkeypatch):
    ev = evaluator()
    ev.trainer.is_global_zero = False
    ev._maybe_log_open_loop_video = lambda **kw: pytest.fail(
        "Nonzero rank wrote a video"
    )
    fake_group(monkeypatch, payload([0, 2]), rank=1)
    assert ev._collective_video_enabled(0)
    ev._collect_open_loop_video(**payload([1, 3]))


def test_single_gpu_uses_all_frames(monkeypatch):
    ev = evaluator()
    calls = []
    monkeypatch.setattr(module.dist, "is_initialized", lambda: False)
    ev._maybe_log_open_loop_video = lambda **kw: calls.append(
        int(kw["source_batch"]["frame_index"][0])
    )
    ev._collect_open_loop_video(**payload([0, 1, 2]))
    assert calls == [0, 1, 2]


def test_missing_frame_fails_instead_of_silently_lowering_fps(monkeypatch):
    ev = evaluator()
    monkeypatch.setattr(module.dist, "is_initialized", lambda: False)
    ev._maybe_log_open_loop_video = lambda **kw: None
    with pytest.raises(RuntimeError, match="expected frame 1, got 2"):
        ev._collect_open_loop_video(**payload([0, 2]))


def test_video_frame_tracking_resets_each_validation():
    ev = evaluator()
    ev._video_enabled = False
    ev.model = None
    ev._video_seen_frames = {("old", 0)}
    ev._video_last_frame = {"old": 99}
    ev.on_validation_start()
    assert ev._video_seen_frames == set()
    assert ev._video_last_frame == {}
