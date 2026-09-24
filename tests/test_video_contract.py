"""Episode identity, native frame timing and distributed video reconstruction."""

from types import SimpleNamespace

import pytest
import torch

from egomimic.eval.bimanual_cartesian_eval import BimanualCartesianEval


def evaluator(tmp_path, **kwargs):
    obj = BimanualCartesianEval(
        sample_id_key="trial", frame_index_key="tick", source_fps=24.0, **kwargs
    )
    obj.trainer = SimpleNamespace(
        is_global_zero=True,
        current_epoch=0,
        default_root_dir=str(tmp_path),
        world_size=2,
        logger=None,
    )
    obj.on_validation_start()
    return obj


def test_rank_frames_are_reassembled_and_padding_deduplicated(tmp_path, monkeypatch):
    obj = evaluator(tmp_path, complete_video_episodes=True)
    key = ("valid", "opaque-source")
    obj._record_video_frames(
        key,
        torch.zeros(2, 4, 4, 3, dtype=torch.uint8),
        {"trial": ["episode", "episode"], "tick": [0, 2]},
    )
    remote = {
        (*key, "episode"): {
            1: torch.ones(4, 4, 3, dtype=torch.uint8),
            3: torch.ones(4, 4, 3, dtype=torch.uint8),
        }
    }
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(
        torch.distributed,
        "all_gather_object",
        lambda output, local: output.__setitem__(
            slice(None), [local, {key: tuple(values) for key, values in remote.items()}]
        ),
    )
    monkeypatch.setattr(
        torch.distributed,
        "gather_object",
        lambda values, output, dst: output.__setitem__(
            slice(None), [values, next(iter(remote.values()))]
        ),
    )
    monkeypatch.setattr(
        torch.distributed, "broadcast_object_list", lambda *a, **k: None
    )
    writes = []
    monkeypatch.setattr(
        obj,
        "_write_episode_stream",
        lambda path, name, frames, **kw: writes.append(
            (torch.stack([torch.as_tensor(f) for f in frames]), kw)
        ),
    )
    obj.on_validation_end()
    assert len(writes) == 1
    frames, settings = writes[0]
    assert settings["fps"] == 24.0
    assert frames[:, 0, 0, 0].tolist() == [0, 1, 0, 1]


def test_sparse_video_uses_actual_indices_not_world_size(tmp_path, monkeypatch):
    obj = evaluator(tmp_path)
    obj._record_video_frames(
        ("valid", "source"),
        torch.zeros(3, 4, 4, 3, dtype=torch.uint8),
        {"trial": ["e"] * 3, "tick": [0, 3, 6]},
    )
    settings = []
    monkeypatch.setattr(
        obj, "_write_episode_stream", lambda *a, **kw: settings.append(kw)
    )
    obj.on_validation_end()
    assert settings[0]["fps"] == 8.0


def test_episode_video_does_not_guess_missing_metadata(tmp_path):
    obj = evaluator(tmp_path)
    with pytest.raises(ValueError, match="trial.*tick"):
        obj._record_video_frames(("valid", "source"), torch.zeros(1, 4, 4, 3), {})


def test_full_episode_video_rejects_gaps(tmp_path):
    obj = evaluator(tmp_path, complete_video_episodes=True)
    obj._record_video_frames(
        ("valid", "source"),
        torch.zeros(2, 4, 4, 3),
        {"trial": ["e"] * 2, "tick": [0, 2]},
    )
    with pytest.raises(ValueError, match="incomplete"):
        obj.on_validation_end()


def test_streamed_video_has_exact_frame_count_and_fractional_fps(tmp_path):
    import av

    obj = evaluator(tmp_path, complete_video_episodes=True)
    obj.source_fps = 30000 / 1001
    frames = torch.stack(
        [torch.full((16, 16, 3), i * 50, dtype=torch.uint8) for i in range(4)]
    )
    obj._record_video_frames(
        ("valid", "source"), frames, {"trial": ["episode"] * 4, "tick": list(range(4))}
    )
    # The corpus is represented by filenames, not resident image tensors.
    assert all(
        path.is_file()
        for records in obj._frame_records.values()
        for path in records.values()
    )
    obj.on_validation_end()
    [path] = list(tmp_path.rglob("episode.mp4"))
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        assert float(stream.average_rate) == pytest.approx(30000 / 1001)
        decoded = [f.to_ndarray(format="rgb24") for f in container.decode(stream)]
    assert len(decoded) == 4
    assert [float(f.mean()) for f in decoded] == pytest.approx([0, 50, 100, 150], abs=3)


def test_nonzero_rank_renders_its_local_samples(tmp_path):
    import numpy as np

    obj = evaluator(tmp_path, complete_video_episodes=True)
    obj.trainer.is_global_zero = False
    obj.viz_func = {"source": lambda **kwargs: np.zeros((1, 16, 16, 3), dtype=np.uint8)}
    obj.revert_transforms = {"source": []}
    obj._native = lambda values, embodiment: values
    obj._native_pose = lambda values, embodiment: values
    obj._viz_source = lambda values, embodiment: values
    obj._maybe_log_overlay(
        embodiment_name="source",
        source_batch={
            "trial": ["e"],
            "tick": [1],
            "embodiment": torch.tensor([7]),
            obj.action_key: torch.zeros(1, 2, 14),
            obj.obs_pose_key: torch.zeros(1, 14),
            obj.image_key: torch.zeros(1, 3, 16, 16),
        },
        predictions={"pred_action": torch.zeros(1, 2, 14)},
        embodiment_id=7,
    )
    assert set(obj._frame_records[("valid", "source", "e")]) == {1}


def test_complete_video_rejects_batch_cap():
    obj = BimanualCartesianEval(
        complete_video_episodes=True, viz_max_batches=1, viz_func={"source": object()}
    )
    with pytest.raises(ValueError, match="viz_max_batches"):
        obj.data_requirements()
