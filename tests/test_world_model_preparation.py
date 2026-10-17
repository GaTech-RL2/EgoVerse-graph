"""Real sliced caches, train-only statistics, and one-setup integrity reuse."""

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from egomimic.rldb.world_model import PreparedWorldDataset
from egomimic.rldb.world_model_datamodule import WorldModelDataModule
from egomimic.rldb.world_model_preparation import (
    ArtifactVerificationMemo, finalize, StreamingMoments,
)
from egomimic.rldb.zarr.episode_split import split_dataset_names


def make_inventory(tmp_path):
    paths, schemas = [], {}
    for name, dimension in (("robot", 3), ("human", 4)):
        fields = [f"pose_{i}" for i in range(dimension)]
        schemas[name] = {"dimension": dimension, "fields": fields}
        _, validation = split_dataset_names(["a", "b", "c"], .01, 42)
        for episode in ("a", "b", "c"):
            file = tmp_path / f"{name}-{episode}.h5"
            # Huge held-out values make accidental leakage into moments obvious.
            offset = 10000 if episode in validation else (1 if episode == "a" else 3)
            with h5py.File(file, "w") as cache:
                cache["time"] = np.arange(4, dtype=np.float64) * .1
                cache.create_dataset("image", data=np.ones((4, 1, 2, 2), np.float32) * offset,
                                     chunks=(1, 1, 2, 2))
                cache["image_xy"] = np.zeros((2, 2), np.float32)
                cache["track"] = np.ones((4, 2, 3), np.float32) * offset
                cache["track_xy"] = np.zeros((2, 2), np.float32)
                cache["track_camera"] = np.zeros(2, np.int64)
                cache["track_confidence"] = np.ones((4, 2), np.float32)
                cache["pose"] = np.ones((4, dimension), np.float32) * offset
                cache["pose_valid"] = np.ones(4, bool)
            receipt = Path(str(file) + ".json")
            receipt.write_text(json.dumps({
                "schema": "mot-world-feature-cache-v1", "status": "PASS", "arrays_normalized": False,
                "output_sha256": hashlib.sha256(file.read_bytes()).hexdigest(), "embodiment": name,
                "episode_id": episode, "frames": 4, "pose_fields": fields, "cameras": ["front"],
                "native_source": {"source_demo_sha256": hashlib.sha256(f"{name}-{episode}".encode()).hexdigest(),
                                  "source_demo": str(file), "episode": "demo_0"},
                "dino_identity": {"model": "fixture", "revision": "fixture", "image_size": [1, 2],
                                  "patch_grid": [1, 2], "feature_layer": "last"}}))
            paths.append(receipt.name)
    inventory = tmp_path / "inventory.json"
    inventory.write_text(json.dumps({"receipts": paths, "pose_schemas": schemas,
                                   "track_provenance": {"method": "fixture", "reference_frame": "world",
                                                        "calibration_sha256": "0" * 64,
                                                        "interpolated_endpoints": False}}))
    return inventory


def test_streaming_moments_match_full_population():
    values = np.random.default_rng(4).normal(size=(31, 3))
    moments = StreamingMoments()
    for batch in np.array_split(values, 7):
        moments.update(batch)
    result = moments.result(1e-5)
    np.testing.assert_allclose(result["mean"], values.mean(0), atol=1e-15)
    np.testing.assert_allclose(result["std"], values.std(0), atol=1e-15)


def test_finalizer_merges_bound_train_moments_without_rereading_images(tmp_path, monkeypatch):
    inventory = make_inventory(tmp_path)
    reference_path = tmp_path / "reference.json"
    reference = finalize(inventory, reference_path)
    expected = json.loads((tmp_path / reference["normalization"]["path"]).read_text())
    for receipt_name in json.loads(inventory.read_text())["receipts"]:
        receipt_path = tmp_path / receipt_name
        receipt = json.loads(receipt_path.read_text())
        with h5py.File(str(receipt_path)[:-5]) as cache:
            receipt["moments"] = {}
            for key in ("image", "track", "pose"):
                moments = StreamingMoments()
                moments.update(cache[key][:])
                receipt["moments"][key] = {"count": moments.count, "mean": moments.mean.tolist(),
                                           "m2": moments.m2.tolist()}
        receipt_path.write_text(json.dumps(receipt))
    original = h5py.Dataset.__getitem__

    def read(dataset, index):
        assert dataset.name != "/image", "Bound producer moments should avoid rereading image payloads"
        return original(dataset, index)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", read)
    output = tmp_path / "merged.json"
    actual = finalize(inventory, output)
    observed = json.loads((tmp_path / actual["normalization"]["path"]).read_text())
    assert {k: v for k, v in observed.items() if k != "statistics"} == {
        k: v for k, v in expected.items() if k != "statistics"}
    for key, actual_stats in observed["statistics"].items():
        expected_stats = expected["statistics"][key]
        assert actual_stats["count"] == expected_stats["count"]
        for field in ("mean", "std"):
            np.testing.assert_allclose(actual_stats[field], expected_stats[field], rtol=1e-14, atol=1e-14)


def test_finalizer_has_no_validation_leak_and_normalizes_selected_frames(tmp_path, monkeypatch):
    inventory = make_inventory(tmp_path)
    output = tmp_path / "prepared.json"
    manifest = finalize(inventory, output)
    assert manifest["split_provenance"]["complete_corpus_coverage"]
    stats = json.loads((tmp_path / manifest["normalization"]["path"]).read_text())
    assert all(np.max(np.abs(s["mean"])) < 10 for s in stats["statistics"].values())
    # Same inventory finalization is immutable and deterministic.
    assert finalize(inventory, output) == manifest
    calls = []
    original = h5py.Dataset.__getitem__

    def read(dataset, index):
        if dataset.name == "/image":
            calls.append(index)
        return original(dataset, index)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", read)
    dataset = PreparedWorldDataset(output, split="val", min_frames=2, max_frames=2)
    sample = dataset[0]
    image = sample["world"]["image"]["value"]
    assert torch.isfinite(image).all() and image.min() > 100
    assert len(calls) == 1 and isinstance(calls[0], np.ndarray) and len(calls[0]) == 2


def test_setup_hashes_payloads_once_and_rejects_changed_memo_artifact(tmp_path, monkeypatch):
    inventory = make_inventory(tmp_path)
    output = tmp_path / "prepared.json"
    manifest = finalize(inventory, output)
    count = []
    original = hashlib.file_digest

    def digest(stream, *args, **kwargs):
        count.append(stream.name)
        return original(stream, *args, **kwargs)

    monkeypatch.setattr(hashlib, "file_digest", digest)
    data = WorldModelDataModule(output, min_frames=2, max_frames=2)
    data.setup()
    assert len(count) == len(manifest["episodes"]) + 1
    memo = ArtifactVerificationMemo()
    artifact = tmp_path / manifest["episodes"][0]["path"]
    sha = manifest["episodes"][0]["sha256"]
    memo.verify(artifact, sha)
    with artifact.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        memo.verify(artifact, sha)


@pytest.mark.parametrize("corruption", ["stats_hash", "train_identity", "missing_stats", "raw_without_path"])
def test_raw_normalization_requires_bound_train_statistics(tmp_path, corruption):
    inventory = make_inventory(tmp_path)
    output = tmp_path / "prepared.json"
    manifest = finalize(inventory, output)
    stats_path = tmp_path / manifest["normalization"]["path"]
    if corruption == "stats_hash":
        stats_path.write_text("{}")
    elif corruption == "train_identity":
        stats = json.loads(stats_path.read_text())
        stats["train_episodes"] = []
        stats_path.write_text(json.dumps(stats))
        manifest["normalization"]["sha256"] = hashlib.sha256(stats_path.read_bytes()).hexdigest()
    elif corruption == "missing_stats":
        stats_path.unlink()
    else:
        del manifest["normalization"]["path"]
    output.write_text(json.dumps(manifest))
    with pytest.raises((ValueError, FileNotFoundError)):
        PreparedWorldDataset(output)


def test_finalizer_rejects_duplicate_physical_cache(tmp_path):
    inventory = make_inventory(tmp_path)
    spec = json.loads(inventory.read_text())
    spec["receipts"].append(spec["receipts"][0])
    inventory.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="Duplicate"):
        finalize(inventory, tmp_path / "prepared.json")
