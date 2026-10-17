"""Preprocessing/correspondence contracts; fake features are not DINO quality tests."""

import hashlib
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from egomimic.scripts.embedding_process.dinov3_correspondence import (
    cached_features,
    eligible,
    load_manifest,
    match_point,
    patch_geometry,
    run,
)
from egomimic.scripts.embedding_process.dinov3_embedding import DINOv3ImageEmbedding


def test_projected_reference_validity_is_not_visibility(tmp_path):
    point = {"reference_valid": True, "visible": False}
    assert eligible(point, "projected_landmark")
    assert not eligible(point, "visible_manual")
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"reference_mode": "projected_landmark"}))
    with pytest.raises(ValueError, match="projection_provenance"):
        load_manifest(path)


def test_feature_reuse_rejects_changed_identity(tmp_path):
    path = tmp_path / "results.json"
    path.write_text(
        json.dumps({"model": "wrong", "revision": "a" * 40, "precision": "float32"})
    )
    with pytest.raises(ValueError, match="model/revision/precision"):
        cached_features([path], "wanted", "a" * 40)


class Processor:
    image_mean = [0.5, 0.5, 0.5]
    image_std = [0.25, 0.25, 0.25]

    def __call__(self, images, return_tensors, **kwargs):
        assert kwargs == {"do_resize": False, "do_center_crop": False}
        pixels = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).float()
        return {"pixel_values": (pixels / 255 - 0.5) / 0.25}


class Model:
    config = SimpleNamespace(
        image_size=224, patch_size=16, num_register_tokens=4, _commit_hash="test"
    )

    def __call__(self, pixel_values):
        n = pixel_values.shape[-1] * pixel_values.shape[-2] // 256
        tokens = torch.arange(n + 5).float().reshape(1, -1, 1)
        return SimpleNamespace(
            last_hidden_state=tokens.expand(pixel_values.shape[0], -1, 2)
        )


def extractor(side):
    x = DINOv3ImageEmbedding(
        ["image"], image_side=side, device="cpu", dtype=torch.float32
    )
    x.processor, x.model = Processor(), Model()
    return x


@pytest.mark.parametrize(
    "side,patches", [(224, 196), (336, 441), (448, 784), (640, 1600)]
)
def test_actual_processed_grid_not_static_model_config(side, patches):
    x = extractor(side)
    image = np.zeros((480, 640, 3), np.uint8)
    features = x._embed_batch([image])
    assert features.shape == (1, patches, 2)
    assert features[0, 0, 0] == 5
    assert features[0, -1, 0] == patches + 4
    assert x.last_preprocessing["patch_grid"] == [side // 16] * 2
    assert x.last_preprocessing["frames"][0]["original_hw"] == [480, 640]


def test_same_fov_padding_and_native_coordinate_roundtrip():
    x = extractor(640)
    image = np.full((480, 640, 3), 255, np.uint8)
    inputs, geometry = x._prepare([image])
    assert geometry[0] == {
        "original_hw": [480, 640],
        "resized_hw": [480, 640],
        "padding_top_left": [80, 0],
    }
    assert torch.all(inputs["pixel_values"][0, :, 80:560] == 2)
    x._embed_batch([image])
    centers, valid = patch_geometry(x.last_preprocessing)
    assert valid.sum() == 30 * 40
    assert np.array_equal(centers[5 * 40], [7.5, 7.5])
    assert not valid[: 5 * 40].any()


def test_malformed_output_and_fractional_patch_rejected():
    x = extractor(225)
    with pytest.raises(ValueError, match="divisible"):
        x._embed_batch([np.zeros((32, 32, 3), np.uint8)])
    x = extractor(224)
    x.model = lambda **kwargs: SimpleNamespace(last_hidden_state=torch.zeros(1, 196, 2))
    x.model.config = Model.config
    with pytest.raises(ValueError, match="token count"):
        x._embed_batch([np.zeros((32, 32, 3), np.uint8)])


def test_identical_descriptor_grid_matches_same_native_point():
    x = extractor(64)
    x._embed_batch([np.zeros((64, 64, 3), np.uint8)])
    features = np.eye(16, dtype=np.float32)
    pred, cosine = match_point(
        features, features, x.last_preprocessing, x.last_preprocessing, [23.5, 39.5]
    )
    assert np.array_equal(pred, [23.5, 39.5])
    assert cosine == 1


def test_padding_cannot_win_match_even_with_larger_cosine():
    x = extractor(64)
    x._embed_batch([np.zeros((32, 64, 3), np.uint8)])
    source = np.ones((16, 2), np.float32)
    target = np.tile([1.0, 0.0], (16, 1)).astype(np.float32)
    target[:4] = [1.0, 1.0]
    pred, _ = match_point(
        source, target, x.last_preprocessing, x.last_preprocessing, [23.5, 7.5]
    )
    assert 0 <= pred[1] < 32


def test_setup_pins_processor_and_model_to_same_revision(monkeypatch):
    calls = []

    class Factory:
        @staticmethod
        def from_pretrained(name, **kwargs):
            calls.append((name, kwargs))
            return SimpleNamespace(
                to=lambda device: SimpleNamespace(eval=lambda: None),
                requires_grad_=lambda value: None,
            )

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoModel=Factory, AutoImageProcessor=Factory),
    )
    x = DINOv3ImageEmbedding(["image"], device="cpu", revision="a" * 40)
    x.setup()
    assert len(calls) == 2
    assert all(kwargs["revision"] == "a" * 40 for _, kwargs in calls)


def test_source_descriptor_touching_padding_is_rejected():
    x = extractor(64)
    x._embed_batch([np.zeros((32, 64, 3), np.uint8)])
    features = np.ones((16, 2), np.float32)
    with pytest.raises(ValueError, match="padded or mixed"):
        match_point(
            features, features, x.last_preprocessing, x.last_preprocessing, [23.5, 0]
        )


def test_boolean_resolution_rejected():
    with pytest.raises(ValueError, match="positive integer"):
        extractor(True)


@pytest.mark.parametrize("reference_mode", ["visible_manual", "projected_landmark"])
def test_screen_writes_reusable_features_and_does_not_fake_wrong_finger_metric(
    tmp_path, monkeypatch, reference_mode
):
    def fake_setup(self):
        self.processor, self.model = Processor(), Model()
        self.model.config = SimpleNamespace(**vars(Model.config))
        self.model.config._commit_hash = self.revision

    monkeypatch.setattr(DINOv3ImageEmbedding, "setup", fake_setup)
    frames = []
    for key in ["a", "b"]:
        path = tmp_path / f"{key}.png"
        Image.fromarray(np.zeros((32, 32, 3), np.uint8)).save(path)
        frames.append(
            {
                "id": key,
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "episode": "synthetic",
                "points": {
                    "index": {
                        "xy": [15.5, 15.5],
                        "visible": reference_mode == "visible_manual",
                        "reference_valid": True,
                        "annotation_source": "synthetictest",
                    }
                },
            }
        )
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "model": "fake",
                "reference_mode": reference_mode,
                "projection_provenance": {"synthetic": True},
                "revision": "a" * 40,
                "frames": frames,
                "pairs": [["a", "b"]],
            }
        )
    )
    result = run(path, tmp_path / "output", [32, 64], "cpu")
    assert len(result["resolutions"]) == 2
    for r in result["resolutions"]:
        assert r["wrong_labeled_point_fraction"] is None
        for artifact in r["feature_caches"].values():
            file = tmp_path / "output" / artifact["path"]
            assert hashlib.sha256(file.read_bytes()).hexdigest() == artifact["sha256"]
            with np.load(file, allow_pickle=False) as cache:
                assert cache["features"].shape[0] == r["patch_tokens"]
    with pytest.raises(FileExistsError, match="Preserve"):
        run(path, tmp_path / "output", [32], "cpu")
    reused = run(
        path, tmp_path / "reused", [32], "cpu", [tmp_path / "output/results.json"]
    )
    assert reused["reference_mode"] == reference_mode
    assert reused["resolutions"][0]["reused_frames"] == 2
    assert reused["resolutions"][0]["extract_median_seconds_including_transfer"] is None
    assert reused["resolutions"][0]["matches"] == result["resolutions"][0]["matches"]
