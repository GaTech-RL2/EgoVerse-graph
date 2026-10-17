"""Streaming producer contracts; fake frozen features, no pretrained download."""
import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from egomimic.rldb.world_model import open_prepared_episode
from egomimic.scripts.embedding_process.world_model_cache import extract_episode, sha256


class FakeDino:
    dtype = torch.float32
    device = torch.device("cpu")
    image_side = 32

    def __init__(self):
        self.model = torch.nn.Linear(1, 1).eval().requires_grad_(False)
        self.calls = []

    def _embed_batch(self, images):
        self.calls.append(len(images))
        assert all(image.shape == (32, 32, 3) for image in images)
        self.last_preprocessing = {
            "processed_hw": [32, 32], "patch_grid": [2, 2],
            "precision": "torch.float32", "feature_layer": "last_hidden_state_patch_tokens",
            "frames": [{"original_hw": [32, 32]} for _ in images],
        }
        return np.stack([np.full((4, 768), image[0, 0, 0], np.float32) for image in images])


def fixture(tmp_path, side=32):
    native = tmp_path / "native"
    native.mkdir()
    times = np.array([0., .05, .1], np.float64)
    arrays = {
        "time": times, "source_row": np.arange(3),
        "pose": np.zeros((3, 9)), "xyz": np.zeros((3, 2, 3)),
        "coordinate_available": np.ones((3, 2), bool),
        "reference_grid_xy": np.array([[0, 0], [127, 127]]),
        "valid_grid_indices": np.arange(2), "visibility": np.zeros((3, 2, 2), bool),
    }
    arrays["pose"][:, 3] = 1
    np.savez(native / "aligned.npz", **arrays)
    manifest = {"schema": "libero-same-recorded-state-v1", "cameras": ["front", "wrist"],
                "pose_fields": [f"pose{i}" for i in range(9)], "reference_frame": "recorded simulator world",
                "height": 128, "width": 128,
                "artifacts": {"aligned.npz": sha256(native / "aligned.npz")}}
    (native / "manifest.json").write_text(json.dumps(manifest))
    rgb = tmp_path / "rgb.h5"
    with h5py.File(rgb, "w") as stream:
        pixels = np.zeros((3, 2, side, side, 3), np.uint8)
        for t in range(3):
            for camera in range(2):
                pixels[t, camera] = t * 10 + camera
        stream["rgb"] = pixels
        stream["time"] = times
    receipt = {"schema": "libero-native-rgb-v1", "status": "PASS", "native_render": True,
               "native_geometry_replay_exact": True,
               "source_manifest_sha256": sha256(native / "manifest.json"),
               "source_aligned_sha256": sha256(native / "aligned.npz"),
               "rgb_cache_sha256": sha256(rgb), "height": side, "width": side,
               "cameras": manifest["cameras"], "frames": 3}
    receipt_path = tmp_path / "rgb.json"
    receipt_path.write_text(json.dumps(receipt))
    return dict(native_cache=native, rgb_cache=rgb, rgb_receipt=receipt_path,
                output=tmp_path / "features.h5", embodiment="gripper", episode_id="ep1",
                embedder=FakeDino(), batch_size=2,
                model_identity={"model": "fake-test-only", "revision": "abc", "weights_sha256": "a" * 64,
                                "config_sha256": "b" * 64, "processor_sha256": "c" * 64})


def test_streaming_fp32_cache_geometry_identity_and_reuse(tmp_path):
    args = fixture(tmp_path)
    result = extract_episode(**args)
    assert result["status"] == "PASS" and not result["training_ready"]
    assert result["moments"]["image"]["count"] == 24
    assert result["moments"]["track"]["count"] == 6
    assert result["moments"]["pose"]["count"] == 3
    assert args["embedder"].calls == [2, 1, 2, 1]
    with open_prepared_episode(args["output"]) as saved:
        assert saved["image"].shape == (3, 2, 4, 768)
        assert saved["image"].dtype == np.float32 and saved["image"].chunks[0] == 1
        np.testing.assert_array_equal(saved["image"][:, :, 0, 0], [[0, 1], [10, 11], [20, 21]])
        np.testing.assert_array_equal(saved["track_confidence"][:], np.ones((3, 2)))
        assert not saved["visibility"][:].any()  # occluded does not mean unavailable
        np.testing.assert_array_equal(saved["track_xy"][:], [[-1, -1], [1, 1]])
        np.testing.assert_array_equal(saved["image_xy"][:], [[-.5, -.5], [.5, -.5], [-.5, .5], [.5, .5]])
        assert not saved.attrs["arrays_normalized"]
        values = saved["image"][:].reshape(-1, 768).astype(np.float64)
        np.testing.assert_allclose(result["moments"]["image"]["mean"], values.mean(0))
        np.testing.assert_allclose(result["moments"]["image"]["m2"], ((values - values.mean(0)) ** 2).sum(0))
    args["embedder"].calls.clear()
    assert extract_episode(**args)["reused"]
    assert args["embedder"].calls == []
    changed = {**args, "model_identity": {**args["model_identity"], "processor_sha256": "d" * 64}}
    with pytest.raises(ValueError, match="request differs"):
        extract_episode(**changed)
    with h5py.File(args["output"], "a") as saved:
        saved["image"][0, 0, 0, 0] += 1
    with pytest.raises(ValueError, match="hash mismatch"):
        extract_episode(**args)


def test_reject_upscaled_native_image(tmp_path):
    args = fixture(tmp_path, side=16)
    with pytest.raises(ValueError, match="Native RGB resolution"):
        extract_episode(**args)
    assert args["embedder"].calls == []
    assert not args["output"].exists()


def test_reject_missing_replay_proof(tmp_path):
    args = fixture(tmp_path)
    receipt = json.loads(args["rgb_receipt"].read_text())
    receipt["native_geometry_replay_exact"] = False
    args["rgb_receipt"].write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="exact geometry receipt"):
        extract_episode(**args)


def test_reject_rgb_time_mismatch(tmp_path):
    args = fixture(tmp_path)
    with h5py.File(args["rgb_cache"], "a") as saved:
        saved["time"][1] += .01
    receipt = json.loads(args["rgb_receipt"].read_text())
    receipt["rgb_cache_sha256"] = sha256(args["rgb_cache"])
    args["rgb_receipt"].write_text(json.dumps(receipt))
    with pytest.raises(AssertionError, match="time alignment"):
        extract_episode(**args)


@pytest.mark.parametrize("failure", ["nan", "float16", "grid"])
def test_invalid_feature_batch_never_publishes(tmp_path, failure):
    args = fixture(tmp_path)
    original = args["embedder"]._embed_batch

    def damaged(images):
        values = original(images)
        if failure == "nan":
            values[0, 0, 0] = np.nan
        elif failure == "float16":
            values = values.astype(np.float16)
        else:
            args["embedder"].last_preprocessing["patch_grid"] = [3, 3]
        return values

    args["embedder"]._embed_batch = damaged
    with pytest.raises(ValueError, match="Invalid DINO"):
        extract_episode(**args)
    assert not args["output"].exists()
    assert not Path(str(args["output"]) + ".json").exists()
