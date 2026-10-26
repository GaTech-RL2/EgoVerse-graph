"""CPU-only source/identity/sampling tests using two synthetic ImageNet classes.

These fixtures certify the adapter contract, not official ImageNet admission,
real Qwen output, model optimization, or GPU generation.
"""

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch

from egomimic.pl_utils.data_context import state_fingerprint
from egomimic.rldb.imagenet_imagefolder import (
    ImageNetImageFolderDataModule, PREPROCESSING,
    build_class_language_manifest, build_imagefolder_manifest,
    build_validation_anchor_manifest, collate_imagenet_images,
    load_class_language, load_data_context,
)
from egomimic.rldb.pixel_proprio import (
    DistributedPixelTrainingBatchSampler, PixelValidationIndex,
)
from egomimic.rldb.world_model import HomogeneousWorldBatchSampler

from imagenet_fixture import make_fixture, put_json, sha


@pytest.fixture
def fixture(tmp_path):
    return make_fixture(tmp_path)


def prepare(fixture, *, mode="train", restored_state=None, **overrides):
    dm = ImageNetImageFolderDataModule(**dict(fixture["datamodule_kwargs"], **overrides))
    context = dm.prepare_context(mode=mode, normalization=None, normalizer=None,
                                 restored_state=restored_state)
    return dm, context


def read(path):
    return json.loads(Path(path).read_text())


def rebind_manifest(fixture, mutate):
    manifest = read(fixture["manifest_path"])
    mutate(manifest)
    for part in manifest["splits"].values():
        part["count"] = len(part["images"])
        part["records_sha256"] = state_fingerprint(part["images"])
    digest = put_json(fixture["manifest_path"], manifest)
    return dict(manifest_sha256=digest)


def test_exact_native_class_and_full_inventory_bindings(fixture):
    classes, cached = load_class_language(
        fixture["class_language_path"], fixture["class_language_sha256"],
        fixture["language_features_path"], fixture["language_features_sha256"],
        expected_classes=2,
    )
    assert set(classes["classes"]) == set(fixture["names"])
    assert cached.instructions == fixture["names"]
    for wnid, vector in cached.features.items():
        expected = hashlib.sha256(vector.numpy().astype("<f4").tobytes()).hexdigest()
        assert classes["classes"][wnid]["feature_sha256"] == expected
        assert classes["classes"][wnid]["class_name"] == fixture["names"][wnid]
        assert vector.shape == (1024,) and vector.dtype == torch.float32
        assert torch.isfinite(vector).all() and torch.linalg.vector_norm(vector).item() == pytest.approx(1)
    dm, context = prepare(fixture)
    manifest = read(fixture["manifest_path"])
    anchors = read(fixture["validation_anchor_path"])
    assert context.state["preprocessing"] == PREPROCESSING
    assert context.state["split_provenance"]["val"]["count"] == 4
    assert len(dm.datasets["train"]) == 12 and len(dm.datasets["val"]) == 2
    assert context.state["validation_anchors"]["manifest_sha256"] == fixture["validation_anchor_sha256"]
    expected = [row for row in manifest["splits"]["val"]["images"] if row["path"].endswith("_000.png")]
    assert anchors["images"] == expected == dm.datasets["val"].records
    assert anchors["official_val_records_sha256"] == manifest["splits"]["val"]["records_sha256"]
    assert context.state["validation_anchors"]["selector"] == "first_sorted_official_val_image_per_wnid"
    assert context.state["validation_anchors"]["count"] == 2


def test_one_image_has_actual_rgb_patches_language_and_no_vector_placeholders(fixture):
    dm, _ = prepare(fixture)
    sample = dm.datasets["train"][0]
    assert sample["embodiment"] == "imagenet" and sample["task_kind"] == "image_generation"
    assert set(sample["world"]) == {"image"}
    stream = sample["world"]["image"]
    assert set(stream) == {"value", "time", "index", "valid", "target_valid", "observed", "xy", "camera"}
    assert stream["value"].shape == (100, 768) and stream["value"].dtype == torch.float32
    assert torch.isfinite(stream["value"]).all()
    assert -1 <= stream["value"].min() <= stream["value"].max() <= 1
    assert stream["valid"].all() and stream["target_valid"].all() and not stream["observed"].any()
    assert stream["camera"].dtype == torch.long and not stream["camera"].any()
    assert not stream["time"].any() and not stream["index"].any()
    assert stream["xy"].shape == (100, 2)
    assert torch.allclose(stream["xy"][0], torch.tensor([-0.9, -0.9]))
    assert torch.allclose(stream["xy"][-1], torch.tensor([0.9, 0.9]))
    metadata = sample["pixel_metadata"]
    assert metadata["image_size"] == [160, 160] and metadata["patch_size"] == 16
    assert metadata["frame_indices"] == [0] and metadata["timestamps_seconds"] == [0.0]
    assert metadata["camera_names"] == ["image"] and metadata["chunk_size"] == 1
    assert metadata["normalization"] == "uint8_to_minus_one_one"
    assert metadata["patch_order"] == "row_major_hwc"
    assert metadata["source_sha256"] == sha(fixture["data_root"] / metadata["sample_id"])
    wnid = metadata["wnid"]
    assert sample["task_id"] == wnid and sample["language_instruction"] == fixture["names"][wnid]
    assert torch.equal(sample["language_features"], torch.tensor(fixture["vectors"][wnid], dtype=torch.float32))
    assert metadata["class_name"] == sample["language_instruction"]
    # Reconstruct pixels from the advertised patch order and compare with the
    # declared source resize/center-crop, independently of the adapter helper.
    image = Image.open(fixture["data_root"] / metadata["sample_id"]).convert("RGB")
    w, h = image.size
    scale = 160 / min(w, h)
    w, h = int(w * scale + 0.5), int(h * scale + 0.5)
    image = image.resize((w, h), Image.Resampling.BICUBIC)
    image = image.crop(((w-160)//2, (h-160)//2, (w-160)//2+160, (h-160)//2+160))
    expected = torch.from_numpy(np.array(image).copy()).float() / 127.5 - 1
    restored = stream["value"].reshape(10, 10, 16, 16, 3).permute(0, 2, 1, 3, 4).reshape(160, 160, 3)
    assert torch.equal(restored, expected)
    sample["language_features"].fill_(0)
    assert torch.equal(dm.datasets["train"][0]["language_features"], torch.tensor(fixture["vectors"][wnid], dtype=torch.float32))


def test_collate_preserves_draw_and_validation_identity(fixture):
    dm, _ = prepare(fixture)
    batch = collate_imagenet_images([dm.datasets["train"][(0, 0, 0)], dm.datasets["train"][(1, 0, 1)]])
    assert set(batch) == {"imagenet"}
    item = batch["imagenet"]
    assert item["world"]["image"]["value"].shape == (2, 100, 768)
    assert item["language_features"].shape == (2, 1024)
    assert item["task_kind"] == ["image_generation", "image_generation"]
    assert item["sampling_cursor"] == 1 and len(item["pixel_metadata"]) == 2
    val = collate_imagenet_images([dm.datasets["val"][PixelValidationIndex(0, 3)]])["imagenet"]
    assert val["validation_batch_index"] == [3]
    with pytest.raises(ValueError, match="identity"):
        collate_imagenet_images([dm.datasets["train"][(0, 0, 0)], dm.datasets["train"][(1, 1, 1)]])
    bad = dm.datasets["train"][1]
    bad["world"]["proprio"] = {}
    with pytest.raises(ValueError, match="independent ImageNet"):
        collate_imagenet_images([bad])


def test_global_draw_slots_are_rank_disjoint_and_repeatable(fixture):
    dm, _ = prepare(fixture)
    dataset = dm.datasets["train"]
    global_draws = list(HomogeneousWorldBatchSampler(dataset, 4, 42, shuffle=True, checkpointable=True))
    shards = [list(DistributedPixelTrainingBatchSampler(dataset, 2, 42, world_size=2, rank=rank)) for rank in (0, 1)]
    assert len(global_draws) == 3
    for cursor, (global_batch, rank0, rank1) in enumerate(zip(global_draws, *shards)):
        assert rank0 + rank1 == global_batch
        assert {draw[2] for draw in rank0} == {0, 1}
        assert {draw[2] for draw in rank1} == {2, 3}
        assert {draw[1] for draw in global_batch} == {cursor}
    assert list(DistributedPixelTrainingBatchSampler(dataset, 2, 42, world_size=2, rank=0)) == shards[0]
    # The source samples with replacement. Distinct slot identity does not
    # imply different image IDs; the test deliberately makes no such claim.


def test_consumed_cursor_restore_skips_only_acknowledged_draws(fixture):
    dm, context = prepare(fixture)
    loader = dm.train_dataloader()
    original = list(loader.batch_sampler)
    next(iter(loader))
    assert dm.state_dict()["consumed_batches"] == 0
    dm.acknowledge_batch(1)
    saved = dm.state_dict()
    assert saved["data_context_sha256"] == context.fingerprint()
    restored, _ = prepare(fixture, restored_state=context.snapshot())
    restored.load_state_dict(saved)
    assert next(iter(restored.train_dataloader().batch_sampler)) == original[1]
    assert restored.consumed_batches == restored.train_sampler.consumed_batches == 1
    with pytest.raises(ValueError, match="Nonconsecutive"):
        dm.acknowledge_batch(3)


@pytest.mark.parametrize("mutation", ["wrong_context", "wrong_schema", "negative", "bool_cursor"])
def test_sampling_restore_rejects_wrong_identity_or_cursor(fixture, mutation):
    dm, _ = prepare(fixture)
    saved = dm.state_dict()
    if mutation == "wrong_context": saved["data_context_sha256"] = "f" * 64
    if mutation == "wrong_schema": saved["schema"] = "other"
    if mutation == "negative": saved["consumed_batches"] = -1
    if mutation == "bool_cursor": saved["consumed_batches"] = True
    with pytest.raises(ValueError):
        dm.load_state_dict(saved)


def test_validation_ranks_cover_each_frozen_anchor_once_without_padding(fixture):
    ids, indices = [], []
    for rank in (0, 1):
        dm, _ = prepare(fixture)
        dm.trainer = SimpleNamespace(global_rank=rank, world_size=2)
        batches = list(dm.val_dataloader())
        assert len(batches) == 1
        item = batches[0]["imagenet"]
        ids.extend(row["sample_id"] for row in item["pixel_metadata"])
        indices.extend(item["validation_batch_index"])
    assert ids == [row["path"] for row in read(fixture["validation_anchor_path"])["images"]]
    assert len(set(ids)) == 2 and indices == [0, 1]
    dm, _ = prepare(fixture, distributed_world_size=4)
    dm.trainer = SimpleNamespace(global_rank=0, world_size=4)
    with pytest.raises(ValueError, match="without padding or dropping"):
        dm.val_dataloader()
    dm.trainer = SimpleNamespace(global_rank=0, world_size=2)
    with pytest.raises(ValueError, match="topology"):
        dm.val_dataloader()


def test_offline_saved_context_never_opens_raw_splits(fixture):
    _, context = prepare(fixture)
    path = fixture["root"] / "checkpoint-context.json"
    put_json(path, {"data_context": context.snapshot()})
    fixture["data_root"].rename(fixture["root"] / "raw-splits-offline")
    restored = load_data_context(path)
    assert restored.fingerprint() == context.fingerprint()
    assert restored.sample_schema == {"world": "image", "language_features": [1024]}
    assert restored.normalizer is None and restored.validation_groups == ("imagenet",)
    restored.state["class_language"]["classes"]["n00000001"]["class_name"] = "changed copy"
    assert context.state["class_language"]["classes"]["n00000001"]["class_name"] != "changed copy"


def test_eval_restores_context_without_train_split_or_training_dataset(fixture):
    _, context = prepare(fixture)
    (fixture["data_root"] / "train").rename(fixture["data_root"] / "train-offline")
    dm, restored = prepare(fixture, mode="eval", restored_state=context.snapshot())
    assert set(dm.datasets) == {"val"} and restored.fingerprint() == context.fingerprint()
    assert len(list(dm.val_dataloader())) == 2
    with pytest.raises(ValueError, match="saved complete"):
        prepare(fixture, mode="eval")


@pytest.mark.parametrize("kind", ["manifest", "classes", "cache", "anchors"])
def test_exact_metadata_bytes_are_required(fixture, kind):
    key = {"manifest": "manifest_sha256", "classes": "class_language_sha256",
           "cache": "language_features_sha256", "anchors": "validation_anchor_sha256"}[kind]
    with pytest.raises(ValueError):
        prepare(fixture, **{key: "f" * 64})


@pytest.mark.parametrize("split", ["train", "val"])
def test_full_split_counts_are_explicit_admission_requirements(fixture, split):
    key = "expected_" + split + "_images"
    expected = fixture["datamodule_kwargs"][key]
    with pytest.raises(ValueError):
        prepare(fixture, **{key: expected + 1})
    kwargs = dict(fixture["datamodule_kwargs"])
    kwargs.pop(key)
    # Production defaults must not admit this small but otherwise complete
    # two-class fixture just because every class appears in both splits.
    with pytest.raises(ValueError):
        ImageNetImageFolderDataModule(**kwargs).prepare_context(
            mode="train", normalization=None, normalizer=None)


@pytest.mark.parametrize("kind", ["unknown_wnid", "wrong_split", "duplicate_path", "missing_class", "noncanonical", "bad_preprocess"])
def test_rehashed_manifest_still_requires_semantic_inventory(fixture, kind):
    def mutate(value):
        rows = value["splits"]["train"]["images"]
        if kind == "unknown_wnid": rows[0]["wnid"] = "n99999999"
        if kind == "wrong_split": rows[0]["path"] = rows[0]["path"].replace("train/", "val/")
        if kind == "duplicate_path": rows[1] = copy.deepcopy(rows[0])
        if kind == "missing_class": value["splits"]["train"]["images"] = [row for row in rows if row["wnid"] == "n00000001"]
        if kind == "noncanonical": rows.reverse()
        if kind == "bad_preprocess": value["preprocessing"]["normalization"] = "identity"
    overrides = rebind_manifest(fixture, mutate)
    with pytest.raises(ValueError):
        prepare(fixture, **overrides)


@pytest.mark.parametrize("kind", ["source", "official_val", "selector", "second_image", "train_image", "count", "records_digest"])
def test_rehashed_anchors_cannot_change_frozen_official_val_subset(fixture, kind):
    value = read(fixture["validation_anchor_path"])
    source = read(fixture["manifest_path"])
    if kind == "source": value["source_manifest_sha256"] = "f" * 64
    if kind == "official_val": value["official_val_records_sha256"] = "f" * 64
    if kind == "selector": value["selector"] = "random"
    if kind == "second_image": value["images"][0] = source["splits"]["val"]["images"][1]
    if kind == "train_image": value["images"][0] = source["splits"]["train"]["images"][0]
    if kind == "count": value["count"] = 4
    if kind == "records_digest": value["records_sha256"] = "f" * 64
    else: value["records_sha256"] = state_fingerprint(value["images"])
    digest = put_json(fixture["validation_anchor_path"], value)
    with pytest.raises(ValueError, match="frozen official-val"):
        prepare(fixture, validation_anchor_sha256=digest)


def test_changed_image_bytes_are_not_silently_used(fixture):
    dm, _ = prepare(fixture)
    image = fixture["data_root"] / dm.datasets["train"].records[0]["path"]
    with image.open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ValueError, match="file changed"):
        dm.datasets["train"][0]


def test_linked_raw_image_component_is_rejected(fixture):
    dm, _ = prepare(fixture)
    image = fixture["data_root"] / dm.datasets["train"].records[0]["path"]
    actual = image.with_suffix(".retained")
    image.rename(actual)
    image.symlink_to(actual)
    with pytest.raises(ValueError, match="Linked ImageNet"):
        dm.datasets["train"][0]


@pytest.mark.parametrize("kind", ["class_text", "feature_hash", "revision", "zero_vector", "nan_vector"])
def test_rehashed_language_metadata_rejects_wrong_text_encoder_or_vectors(fixture, kind):
    classes = read(fixture["class_language_path"])
    cache = read(fixture["language_features_path"])
    wnid = "n00000001"
    if kind == "class_text": classes["classes"][wnid]["class_name"] = "other text"
    if kind == "feature_hash": classes["classes"][wnid]["feature_sha256"] = "f" * 64
    if kind == "revision":
        cache["encoder"]["revision"] = "f" * 40
        classes["encoder"]["revision"] = "f" * 40
    if kind == "zero_vector": cache["tasks"][wnid]["features"] = [0.0] * 1024
    if kind == "nan_vector": cache["tasks"][wnid]["features"][0] = float("nan")
    cache_sha = put_json(fixture["language_features_path"], cache, allow_nan=kind == "nan_vector")
    classes["feature_cache_sha256"] = cache_sha
    class_sha = put_json(fixture["class_language_path"], classes)
    with pytest.raises((ValueError, RuntimeError)):
        load_class_language(fixture["class_language_path"], class_sha,
                            fixture["language_features_path"], cache_sha, expected_classes=2)


@pytest.mark.parametrize("mutation", ["source", "anchors", "preprocess", "decoder", "seed", "topology", "class_text"])
def test_checkpoint_restore_compares_complete_current_context(fixture, mutation):
    _, context = prepare(fixture)
    state = context.snapshot()
    if mutation == "source": state["manifest_sha256"] = "f" * 64
    if mutation == "anchors": state["validation_anchors"]["manifest_sha256"] = "f" * 64
    if mutation == "preprocess": state["preprocessing"]["resize"] = "nearest"
    if mutation == "decoder": state["decoder_version"]["version"] = "other"
    if mutation == "seed": state["sampling"]["seed"] = 43
    if mutation == "topology": state["sampling"]["world_size"] = 1
    if mutation == "class_text": state["class_language"]["classes"]["n00000001"]["class_name"] = "other"
    with pytest.raises(ValueError, match="context mismatch"):
        prepare(fixture, restored_state=state)


@pytest.mark.parametrize("kind", ["duplicate_key", "nonfinite_extra", "missing_anchor", "wrong_qwen", "bad_sampling"])
def test_offline_context_rejects_ambiguous_or_incomplete_metadata(fixture, kind):
    _, context = prepare(fixture)
    state = context.snapshot()
    path = fixture["root"] / "bad-context.json"
    if kind == "duplicate_key":
        raw = json.dumps(state)
        path.write_text('{"kind":"ignored",' + raw[1:])
    else:
        if kind == "nonfinite_extra": state["class_language"]["encoder"]["torch_version"] = float("nan")
        if kind == "missing_anchor": del state["validation_anchors"]
        if kind == "wrong_qwen": state["class_language"]["encoder"]["model_id"] = "unbound/encoder"
        if kind == "bad_sampling": state["sampling"]["global_batch_size"] = 9
        put_json(path, state, allow_nan=kind == "nonfinite_extra")
    with pytest.raises(ValueError):
        load_data_context(path)


def test_preparation_never_overwrites_manifests(fixture):
    with pytest.raises(FileExistsError):
        build_class_language_manifest(fixture["class_catalog_path"], fixture["class_catalog_sha256"],
            fixture["language_features_path"], fixture["language_features_sha256"],
            fixture["class_language_path"], expected_classes=2)
    with pytest.raises(FileExistsError):
        build_imagefolder_manifest(fixture["data_root"], fixture["class_language_path"],
            fixture["class_language_sha256"], fixture["language_features_path"],
            fixture["language_features_sha256"], fixture["manifest_path"], expected_classes=2)
    with pytest.raises(FileExistsError):
        build_validation_anchor_manifest(fixture["manifest_path"], fixture["manifest_sha256"],
            fixture["validation_anchor_path"], expected_classes=2)


@pytest.mark.parametrize("kind", ["sidecar", "split_symlink"])
def test_inventory_preparation_rejects_unclassified_or_linked_split_members(fixture, kind):
    split = fixture["data_root"] / "train"
    if kind == "sidecar":
        (split / "README").write_text("Unclassified source artifact")
    else:
        actual = fixture["root"] / "retained-train"
        split.rename(actual)
        split.symlink_to(actual, target_is_directory=True)
    with pytest.raises(ValueError):
        build_imagefolder_manifest(fixture["data_root"], fixture["class_language_path"],
            fixture["class_language_sha256"], fixture["language_features_path"],
            fixture["language_features_sha256"], fixture["root"] / "new-manifest.json",
            expected_classes=2)


def test_no_temporal_evaluation_or_normalizer_is_introduced(fixture):
    dm, _ = prepare(fixture)
    with pytest.raises(ValueError, match="no normalizer"):
        dm.prepare_context(mode="train", normalization="fitted", normalizer=object())
    with pytest.raises(ValueError, match="Unsupported independent"):
        dm.configure_evaluation(SimpleNamespace(complete_episodes=True, required_keys={"world"}))
    with pytest.raises(ValueError, match="Unsupported independent"):
        dm.configure_evaluation(SimpleNamespace(complete_episodes=False, required_keys={"actions"}))
    dm.configure_evaluation(SimpleNamespace(complete_episodes=False,
        required_keys={"world", "task_kind", "language_features", "pixel_metadata"}))
