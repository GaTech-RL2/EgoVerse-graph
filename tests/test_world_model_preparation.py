"""Real sliced caches, train-only statistics, and one-setup integrity reuse."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import time

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


def aged_verification_clock(monkeypatch):
    """Simulate already-immutable files without sleeping in the test suite."""
    now = time.time_ns() + 2_000_000_000
    monkeypatch.setattr(ArtifactVerificationMemo, "_now_ns", staticmethod(lambda: now))


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
    # Historical manifests retain their full-hash behavior.
    for episode in manifest["episodes"]:
        episode.pop("verified_fingerprint")
    output.write_text(json.dumps(manifest))
    aged_verification_clock(monkeypatch)
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


def test_finalized_fingerprints_skip_payload_reads_across_fresh_setups(tmp_path, monkeypatch):
    inventory = make_inventory(tmp_path)
    aged_verification_clock(monkeypatch)
    output = tmp_path / "prepared.json"
    calls, original = [], hashlib.file_digest

    def digest(stream, *args, **kwargs):
        calls.append(Path(stream.name).resolve())
        return original(stream, *args, **kwargs)

    monkeypatch.setattr(hashlib, "file_digest", digest)
    manifest = finalize(inventory, output)
    assert len(calls) == len(manifest["episodes"])
    calls.clear()
    for _ in range(2):
        WorldModelDataModule(output, min_frames=2, max_frames=2).setup()
    # Each independent setup still verifies the small train-only statistics.
    assert calls == [(tmp_path / manifest["normalization"]["path"]).resolve()] * 2
    for episode in manifest["episodes"]:
        fingerprint = episode["verified_fingerprint"]
        assert fingerprint["sha256"] == episode["sha256"]
        assert fingerprint["path"] == str((tmp_path / episode["path"]).resolve())


@pytest.mark.parametrize("change", ["missing", "extra", "schema", "sha", "path", "bool_stat"])
def test_malformed_fingerprint_falls_back_to_content_hash(tmp_path, monkeypatch, change):
    path = tmp_path / "artifact"
    path.write_bytes(b"original")
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    aged_verification_clock(monkeypatch)
    fingerprint = ArtifactVerificationMemo().verify(path, expected)
    if change == "missing":
        fingerprint.pop("st_ctime_ns")
    elif change == "extra":
        fingerprint["unchecked"] = True
    elif change == "schema":
        fingerprint["schema"] = "unknown"
    elif change == "sha":
        fingerprint["sha256"] = "0" * 64
    elif change == "path":
        fingerprint["path"] = str(tmp_path / "elsewhere")
    else:
        fingerprint["st_dev"] = True
    calls, original = [], hashlib.file_digest

    def digest(stream, *args, **kwargs):
        calls.append(stream.name)
        return original(stream, *args, **kwargs)

    monkeypatch.setattr(hashlib, "file_digest", digest)
    ArtifactVerificationMemo().verify(path, expected, trusted_fingerprint=fingerprint)
    assert len(calls) == 1


def test_changed_expected_hash_is_not_trusted(tmp_path, monkeypatch):
    path = tmp_path / "artifact"
    path.write_bytes(b"original")
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    aged_verification_clock(monkeypatch)
    fingerprint = ArtifactVerificationMemo().verify(path, expected)
    with pytest.raises(ValueError, match="hash mismatch"):
        ArtifactVerificationMemo().verify(path, "0" * 64, trusted_fingerprint=fingerprint)


def test_same_size_restored_mtime_mutation_fails_persistent_verification(tmp_path, monkeypatch):
    path = tmp_path / "artifact"
    path.write_bytes(b"original")
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    aged_verification_clock(monkeypatch)
    fingerprint = ArtifactVerificationMemo().verify(path, expected)
    before = path.stat()
    path.write_bytes(b"modified")
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert path.stat().st_size == before.st_size
    assert path.stat().st_mtime_ns == before.st_mtime_ns
    # Explicitly exercise the changed-ctime branch even on a coarse-clock tmpfs.
    native_identity = ArtifactVerificationMemo.identity
    monkeypatch.setattr(ArtifactVerificationMemo, "identity", staticmethod(
        lambda candidate: (*native_identity(candidate)[:-1], before.st_ctime_ns + 1)))
    with pytest.raises(ValueError, match="hash mismatch"):
        ArtifactVerificationMemo().verify(path, expected, trusted_fingerprint=fingerprint)


@pytest.mark.parametrize("change", ["replace_inode", "relocate"])
def test_copy_or_relocation_rehashes_even_identical_payload(tmp_path, monkeypatch, change):
    path = tmp_path / "artifact"
    path.write_bytes(b"original")
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    aged_verification_clock(monkeypatch)
    fingerprint = ArtifactVerificationMemo().verify(path, expected)
    destination = tmp_path / "copy"
    if change == "replace_inode":
        shutil.copy2(path, destination)
        destination.replace(path)
        assert path.stat().st_ino != fingerprint["st_ino"]
    else:
        path.rename(destination)
        path = destination
    calls, original = [], hashlib.file_digest

    def digest(stream, *args, **kwargs):
        calls.append(stream.name)
        return original(stream, *args, **kwargs)

    monkeypatch.setattr(hashlib, "file_digest", digest)
    ArtifactVerificationMemo().verify(path, expected, trusted_fingerprint=fingerprint)
    assert len(calls) == 1


def test_recent_hash_cannot_become_trusted_after_same_tick_mutation(tmp_path, monkeypatch):
    path = tmp_path / "artifact"
    path.write_bytes(b"original")
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    identity = ArtifactVerificationMemo.identity(path)
    # Simulate the observed coarse-clock case: the file keeps every stat field.
    monkeypatch.setattr(ArtifactVerificationMemo, "identity", staticmethod(lambda _: identity))
    monkeypatch.setattr(ArtifactVerificationMemo, "_now_ns", staticmethod(lambda: identity[-1]))
    memo = ArtifactVerificationMemo()
    fingerprint = memo.verify(path, expected)
    assert fingerprint is None and not memo.verified
    path.write_bytes(b"modified")
    monkeypatch.setattr(ArtifactVerificationMemo, "_now_ns",
                        staticmethod(lambda: identity[-1] + 2_000_000_000))
    for verifier in (memo, ArtifactVerificationMemo()):
        with pytest.raises(ValueError, match="hash mismatch"):
            verifier.verify(path, expected, trusted_fingerprint=fingerprint)


def test_recent_verification_timestamp_in_fingerprint_is_rejected(tmp_path, monkeypatch):
    path = tmp_path / "artifact"
    path.write_bytes(b"original")
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    aged_verification_clock(monkeypatch)
    fingerprint = ArtifactVerificationMemo().verify(path, expected)
    fingerprint["verification_started_ns"] = fingerprint["st_ctime_ns"]
    # Even after aging, this malformed proof must trigger a new content hash.
    path.write_bytes(b"modified")
    identity = tuple(fingerprint[key] for key in ArtifactVerificationMemo._STAT_FIELDS)
    monkeypatch.setattr(ArtifactVerificationMemo, "identity", staticmethod(lambda _: identity))
    with pytest.raises(ValueError, match="hash mismatch"):
        ArtifactVerificationMemo().verify(path, expected, trusted_fingerprint=fingerprint)


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


def test_sampling_cursor_ignores_prefetched_unconsumed_batches(tmp_path):
    inventory = make_inventory(tmp_path)
    path = tmp_path / "prepared.json"
    finalize(inventory, path)
    data = WorldModelDataModule(path, max_frames=3, checkpointable_sampling=True)
    loader = data.train_dataloader()
    iterator = iter(loader)
    first = next(iterator)
    data.acknowledge_batch(next(iter(first.values()))["sampling_cursor"])
    expected = next(iterator)  # Prefetch without training must not advance saved cursor.
    saved = data.state_dict()
    restored = WorldModelDataModule(path, max_frames=3, checkpointable_sampling=True)
    restored.load_state_dict(saved)
    actual = next(iter(restored.train_dataloader()))
    assert next(iter(actual.values()))["sampling_cursor"] == 2
    for name, stream in next(iter(actual.values()))["world"].items():
        for key, value in stream.items():
            torch.testing.assert_close(value, next(iter(expected.values()))["world"][name][key], rtol=0, atol=0)
    changed = WorldModelDataModule(path, max_frames=3, batch_size=2, checkpointable_sampling=True)
    with pytest.raises(ValueError, match="batch_size"):
        changed.load_state_dict(saved)
    with pytest.raises(ValueError, match="num_workers=0"):
        WorldModelDataModule(path, num_workers=1, checkpointable_sampling=True)


def test_interrupted_world_training_matches_uninterrupted_updates(tmp_path):
    import inspect
    from lightning import Trainer, seed_everything
    from omegaconf import OmegaConf
    from egomimic.pl_utils.pl_model import ModelWrapper
    path = tmp_path / "prepared.json"
    finalize(make_inventory(tmp_path), path)
    cfg = OmegaConf.create({"model": {
        "pipeline": {"_target_": "egomimic.pipeline.algo.PipelineAlgo", "device": "cpu", "stages": [{
            "_target_": "egomimic.pipeline.stages_world_model.WorldModelStage",
            "warmup_steps": 0, "pose_ramp_steps": 0, "sampler_steps": 2,
            "model": {"_target_": "egomimic.models.world_model.MoTWorldModel", "pose_dims": {"robot": 3, "human": 4},
                      "image_dim": 2, "width": 24, "heads": 3, "depth": 1, "max_tracks": 8}}]},
        "training_behavior": {"_target_": "egomimic.pl_utils.world_model_behavior.WorldModelTrainingBehavior"},
        "optimizer": {"_target_": "torch.optim.AdamW", "_partial_": True, "lr": .001}}})

    def make(steps, checkpoint=None):
        seed_everything(123)
        data = WorldModelDataModule(path, max_frames=3, checkpointable_sampling=True)
        context = data.prepare_context(mode="train", normalization=None, normalizer=None)
        model = ModelWrapper(config_tree=cfg)
        model.data_context = context
        context.bind(model.model, None)
        trainer = Trainer(accelerator="cpu", devices=1, max_steps=steps, max_epochs=-1,
                          limit_val_batches=0, logger=False, enable_checkpointing=False,
                          enable_progress_bar=False, num_sanity_val_steps=0)
        kwargs = {"weights_only": False} if "weights_only" in inspect.signature(trainer.fit).parameters else {}
        trainer.fit(model, datamodule=data, ckpt_path=checkpoint, **kwargs)
        return model, trainer, data

    reference, uninterrupted, _ = make(4)
    _, first, data = make(2)
    assert data.consumed_batches == 2
    checkpoint = tmp_path / "resume.ckpt"
    first.save_checkpoint(checkpoint)
    resumed, continued, restored_data = make(4, str(checkpoint))
    assert continued.global_step == 4 and restored_data.consumed_batches == 4
    for name, expected in reference.state_dict().items():
        torch.testing.assert_close(resumed.state_dict()[name], expected, rtol=0, atol=0)
    expected_states = uninterrupted.optimizers[0].state_dict()["state"]
    actual_states = continued.optimizers[0].state_dict()["state"]
    for parameter, expected in expected_states.items():
        for key, value in expected.items():
            torch.testing.assert_close(actual_states[parameter][key], value, rtol=0, atol=0)
