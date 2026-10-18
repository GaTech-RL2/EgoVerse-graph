"""RGB reconstruction, mask semantics, metadata binding and real-time videos."""

import copy
import json
from types import SimpleNamespace

import av
import numpy as np
import pytest
import torch

from egomimic.eval.image_proprio_eval import ImageProprioEval
from egomimic.eval.pixel_diagnostics import (
    PixelPredictionDiagnostics,
    pixel_frame_records,
    unpatchify_rgb,
    write_timestamp_video,
)
from egomimic.rldb.pixel_proprio import patchify_rgb


def test_wandb_media_uses_optimizer_axis_not_history_counter(tmp_path, monkeypatch):
    import wandb
    from lightning.pytorch.loggers import WandbLogger

    calls = []
    experiment = SimpleNamespace(log=lambda *args, **kwargs: calls.append((args, kwargs)))
    logger = WandbLogger(experiment=experiment, offline=True)
    monkeypatch.setattr(wandb, "Image", lambda path: ("image", path))
    diagnostic = PixelPredictionDiagnostics(tmp_path, log_wandb=True)
    diagnostic.written = [("robot", "forecast", "image", tmp_path / "example.png")]
    model = SimpleNamespace(global_rank=0, global_step=250,
                            trainer=SimpleNamespace(loggers=[logger]))
    diagnostic.log_to_wandb(model)
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert "step" not in kwargs
    assert args[0]["trainer/global_step"] == 250
    assert "val_pixel/robot/forecast/image" in args[0]


def batch_fixture(direct=False):
    frames = 1 if direct else 3
    rgb = np.arange(frames * 2 * 4 * 4 * 3, dtype=np.uint8).reshape(frames, 2, 4, 4, 3)
    patches = patchify_rgb(rgb, 2)
    clock = np.array([10.1, 10.3, 11.05])[:frames]
    origin = clock[0] if direct else clock[1]
    xy = np.array([[-0.5, -0.5], [0.5, -0.5], [-0.5, 0.5], [0.5, 0.5]], np.float32)
    known = np.arange(frames) == 0
    image = {
        "value": torch.from_numpy(patches.reshape(1, frames * 8, 12)),
        "time": torch.from_numpy(np.repeat(clock - origin, 8).astype(np.float32)[None]),
        "index": torch.from_numpy(
            np.repeat(np.arange(frames), 8).astype(np.float32)[None]
        ),
        "camera": torch.from_numpy(np.tile(np.repeat(np.arange(2), 4), frames)[None]),
        "xy": torch.from_numpy(np.tile(xy, (frames * 2, 1))[None]),
        "valid": torch.ones(1, frames * 8, dtype=torch.bool),
        "observed": torch.from_numpy(np.repeat(known, 8)[None]),
    }
    proprio = {
        "value": torch.ones(1, frames, 3),
        "time": torch.from_numpy((clock - origin).astype(np.float32)[None]),
        "valid": torch.ones(1, frames, dtype=torch.bool),
        "observed": torch.from_numpy(known[None]),
    }
    metadata = {
        "episode_id": "example-episode",
        "camera_names": ["agentview", "wrist"],
        "frame_indices": [2, 9, 18][:frames],
        "timestamps_seconds": clock.tolist(),
        "time_origin_seconds": float(origin),
        "image_size": [4, 4],
        "patch_size": 2,
        "normalization": "uint8_to_minus_one_one",
        "patch_order": "row_major_hwc",
    }
    item = {
        "world": {"image": image, "proprio": proprio},
        "embodiment": "robot",
        "task_kind": ["direct_action" if direct else "forecast"],
        "pixel_metadata": [metadata],
    }
    if direct:
        item["world"]["action"] = {
            "value": torch.tensor([[[0.1, 0.2]]]),
            "time": torch.zeros(1, 1),
            "valid": torch.ones(1, 1, dtype=torch.bool),
            "observed": torch.zeros(1, 1, dtype=torch.bool),
        }
    return {"robot": item}, rgb


def fake_model(batch, direct=False):
    world = batch["robot"]["world"]
    prediction = (
        {"action": world["action"]["value"] + 0.5}
        if direct
        else {
            "image": world["image"]["value"]
            + (~world["image"]["observed"])[..., None] * 0.2
        }
    )

    def forward_eval(_):
        torch.rand(3)  # Sampling must not advance the training RNG.
        return {"robot": {"image_proprio_prediction": prediction}}

    logged = {}
    model = SimpleNamespace(
        device=torch.device("cpu"),
        current_epoch=1,
        global_step=7,
        global_rank=0,
        trainer=SimpleNamespace(loggers=[]),
        forward_eval=forward_eval,
        log=lambda k, v, **_: logged.update({k: float(v)}),
    )
    return model, prediction, logged


def test_pixel_grid_roundtrip_and_reordered_tokens_use_true_coordinates():
    batch, rgb = batch_fixture()
    item = batch["robot"]
    image = item["world"]["image"]
    for frame in range(3):
        for camera in range(2):
            patches = patchify_rgb(rgb, 2)[frame, camera]
            np.testing.assert_allclose(
                (unpatchify_rgb(patches, [4, 4], 2) + 1) / 2,
                rgb[frame, camera] / 255,
                atol=1e-7,
            )
    permutation = torch.randperm(image["value"].shape[1])
    shuffled = {k: v[:, permutation] for k, v in image.items()}
    result = pixel_frame_records(
        shuffled, shuffled["value"], item["pixel_metadata"][0], 0
    )
    assert list(result) == [0, 1]
    for camera, value in result.items():
        for i, frame in enumerate(value["frames"]):
            np.testing.assert_allclose(frame["target"], rgb[i, camera] / 255, atol=1e-7)
            assert frame["metadata"]["frame_index"] == [2, 9, 18][i]


def test_metadata_time_and_spatial_grid_mismatch_fail_closed():
    batch, _ = batch_fixture()
    item = batch["robot"]
    stream = item["world"]["image"]
    metadata = copy.deepcopy(item["pixel_metadata"][0])
    metadata["timestamps_seconds"][1] += 0.01
    with pytest.raises(ValueError, match="recorded physical timestamp"):
        pixel_frame_records(stream, stream["value"], metadata, 0)
    stream["xy"][:, 1] = stream["xy"][:, 0]
    with pytest.raises(ValueError, match="declared grid"):
        pixel_frame_records(stream, stream["value"], item["pixel_metadata"][0], 0)


def test_video_preserves_uneven_physical_timestamp_intervals(tmp_path):
    from PIL import Image

    path = tmp_path / "uneven.mp4"
    panels = [Image.new("RGB", (64, 64), (i * 40, 0, 0)) for i in range(3)]
    write_timestamp_video(path, panels, [100.0, 100.2, 100.95], nominal_fps=10)
    with av.open(path) as video:
        times = [float(frame.pts * frame.time_base) for frame in video.decode(video=0)]
    np.testing.assert_allclose(times, [0.0, 0.2, 0.95], atol=1e-5)


def test_visual_only_metrics_contact_sheets_errors_and_immutable_files(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    batch, _ = batch_fixture()
    model, _, metrics = fake_model(batch)
    evaluator = ImageProprioEval(
        objective="visual_only",
        pixel_diagnostics={
            "artifact_dir": str(tmp_path),
            "write_video": True,
            "log_wandb": False,
        },
    )
    evaluator.model = model
    evaluator.on_validation_start()
    rng = torch.get_rng_state().clone()
    evaluator.on_validation_step(batch, 0)
    assert torch.equal(torch.get_rng_state(), rng)
    evaluator.log_epoch_metrics()
    assert list(metrics) == ["val/robot/forecast/image_normalized_mse"]
    assert metrics["val/robot/forecast/image_normalized_mse"] == pytest.approx(0.04)
    reports = list(tmp_path.rglob("diagnostic.json"))
    assert len(reports) == 1
    report = json.loads(reports[0].read_text())
    assert report["objective"] == "visual_only"
    assert "action" not in report
    for camera in report["cameras"].values():
        assert camera["frames"][0]["observation"] == "observed"
        assert camera["frames"][0]["query_pixel_mse_rgb01"] is None
        assert camera["frames"][1]["query_pixel_mse_rgb01"] == pytest.approx(0.01)
        assert (reports[0].parent / camera["contact_sheet"]).is_file()
        assert (reports[0].parent / camera["video"]).is_file()
    evaluator.on_validation_step(batch, 1)  # Bounded diagnostics, all metrics continue.
    assert len(list(tmp_path.rglob("diagnostic.json"))) == 1
    evaluator.on_validation_start()
    with pytest.raises(FileExistsError):
        evaluator.on_validation_step(batch, 0)


def test_direct_action_context_never_masquerades_as_pixel_prediction(tmp_path):
    batch, _ = batch_fixture(direct=True)
    model, _, metrics = fake_model(batch, direct=True)
    evaluator = ImageProprioEval(
        objective="direct_action",
        pixel_diagnostics={
            "artifact_dir": str(tmp_path),
            "log_wandb": False,
        },
    )
    evaluator.model = model
    evaluator.on_validation_start()
    evaluator.on_validation_step(batch, 0)
    evaluator.log_epoch_metrics()
    assert metrics == {"val/robot/direct_action/action_command_mse": 0.25}
    report = json.loads(next(tmp_path.rglob("diagnostic.json")).read_text())
    assert report["action"]["squared_error"][0] == pytest.approx([0.25, 0.25])
    assert all(value["video"] is None for value in report["cameras"].values())
    assert not list(tmp_path.rglob("*.mp4"))
