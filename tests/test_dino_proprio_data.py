"""CPU contracts for two-stream data and fail-closed, aligned command labels."""

import copy
import hashlib
import json

import h5py
import numpy as np
import pytest
import torch

from egomimic.rldb.dino_proprio import (
    EvenlySpacedEpisodeBatchSampler,
    PreparedDinoProprioDataset,
    collate_dino_proprio,
)
from egomimic.rldb.dino_proprio_datamodule import DinoProprioDataModule


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path, data):
    path.write_text(json.dumps(data))
    return path


def prepared(tmp_path, *, raw=False):
    episodes = []
    for split in ("train", "val"):
        path = tmp_path / f"{split}.h5"
        with h5py.File(path, "w") as data:
            data["time"] = 1e9 + np.arange(6) * 0.125
            data["image"] = np.arange(6 * 2 * 3 * 8, dtype=np.float32).reshape(
                6, 2, 3, 8
            )
            data["image_xy"] = np.zeros((3, 2), np.float32)
            data["pose"] = np.arange(6 * 4, dtype=np.float32).reshape(6, 4)
            data["pose_valid"] = np.ones(6, bool)
        episodes.append(
            dict(
                embodiment="gripper",
                episode_id=split,
                split=split,
                path=path.name,
                frames=6,
                sha256=digest(path),
                frame="fixed_reference",
                pose_kind="measured_pose",
                native_source={
                    "episode": f"demo_{split}",
                    "source_demo_sha256": "a" * 64,
                },
            )
        )
    manifest = dict(
        schema="dino-proprio-prepared-v1",
        episodes=episodes,
        dino_identity={
            "model": "fixture",
            "revision": "fixture",
            "image_size": [16, 48],
            "patch_grid": [1, 3],
            "feature_layer": "last",
        },
        pose_schemas={
            "gripper": {"dimension": 4, "fields": [f"p{i}" for i in range(4)]}
        },
        normalization={
            "scope": "train_only",
            "sha256": "0" * 64,
            "arrays_normalized": not raw,
        },
    )
    if raw:
        norm = dict(
            schema="mot-world-normalization-v1",
            scope="train_only",
            train_episodes=[["gripper", "train", episodes[0]["sha256"]]],
            statistics={
                "image": {"mean": [1.0] * 8, "std": [2.0] * 8, "count": 36},
                "pose/gripper": {"mean": [3.0] * 4, "std": [4.0] * 4, "count": 6},
                # Deliberately unusable: unused legacy stats must not be required/parsed.
                "track": {"not": "a normalizer"},
            },
        )
        normpath = save_json(tmp_path / "normalization.json", norm)
        manifest["normalization"].update(path=normpath.name, sha256=digest(normpath))
    return save_json(tmp_path / "manifest.json", manifest)


def actions(path):
    manifest = json.loads(path.read_text())
    episodes = []
    for ep in manifest["episodes"]:
        target = path.parent / f"actions-{ep['split']}.h5"
        with (
            h5py.File(path.parent / ep["path"], "r") as cache,
            h5py.File(target, "w") as data,
        ):
            data["time"] = cache["time"][:]
            data["action"] = np.arange(18, dtype=np.float32).reshape(6, 3) + 100
            data["valid"] = np.ones(6, bool)
            data["cache_frame_index"] = np.arange(6)
            data["source_state_index"] = np.arange(6) * 2
            data["source_action_index"] = np.arange(6) * 2
        receipt = save_json(
            path.parent / f"alignment-{ep['split']}.json",
            dict(
                schema="dino-proprio-source-alignment-v1",
                feature_sha256=ep["sha256"],
                source_demo_sha256=ep["native_source"]["source_demo_sha256"],
                source_episode=ep["native_source"]["episode"],
                relation="action_t_applied_to_state_t",
                producer_sha256="b" * 64,
                basis={
                    "cache_index_source": "synthetic fixture frame map",
                    "action_index_source": "synthetic fixture command map",
                    "state_action_relation_source": "synthetic fixture definition",
                },
                cache_frame_index=list(range(6)),
                source_state_index=list(range(0, 12, 2)),
                source_action_index=list(range(0, 12, 2)),
            ),
        )
        episodes.append(
            dict(
                embodiment=ep["embodiment"],
                episode_id=ep["episode_id"],
                path=target.name,
                sha256=digest(target),
                feature_sha256=ep["sha256"],
                source_demo_sha256=ep["native_source"]["source_demo_sha256"],
                source_episode=ep["native_source"]["episode"],
                alignment={
                    "relation": "action_t_applied_to_state_t",
                    "cache_frame_mapping": "explicit_source_indices",
                    "producer_sha256": "b" * 64,
                    "source_alignment_path": receipt.name,
                    "source_alignment_sha256": digest(receipt),
                },
            )
        )
    return save_json(
        path.parent / "actions.json",
        dict(
            schema="dino-proprio-actions-v1",
            episodes=episodes,
            action_schemas={
                "gripper": {
                    "dimension": 3,
                    "fields": ["dx", "dy", "grip"],
                    "semantics": "action_command",
                    "value_space": "native_commands",
                }
            },
        ),
    )


def test_two_stream_cache_and_legacy_normalizer_never_require_track_arrays(
    tmp_path, monkeypatch
):
    path = prepared(tmp_path, raw=True)
    old = h5py.Group.__getitem__

    def guarded(group, name):
        assert not str(name).startswith("track")
        return old(group, name)

    monkeypatch.setattr(h5py.Group, "__getitem__", guarded)
    dataset = PreparedDinoProprioDataset(path, split="val", min_frames=4, max_frames=4)
    sample = dataset[0]
    assert set(sample["world"]) == {"image", "proprio"}
    assert set(dataset.normalization) == {"image", "pose/gripper"}
    assert sample["world"]["image"]["value"].shape == (24, 8)
    assert sample["world"]["proprio"]["value"].shape == (4, 4)
    torch.testing.assert_close(
        sample["world"]["image"]["value"][0], (torch.arange(8) - 1.0) / 2
    )
    assert sample["world"]["image"]["observed"].any()
    assert (~sample["world"]["image"]["observed"]).any()


def test_forecast_mask_never_observes_future_and_padding_is_not_supervision(tmp_path):
    path = prepared(tmp_path)
    dataset = PreparedDinoProprioDataset(path, split="val", forecasting_probability=1.0)
    sample = dataset[0]
    image, pose = sample["world"]["image"], sample["world"]["proprio"]
    last_image = image["time"][image["observed"]].max()
    assert (image["time"][~image["observed"]] > last_image).all()
    assert (pose["time"][pose["observed"]] <= last_image).all()
    shorter = copy.deepcopy(sample)
    for stream in shorter["world"].values():
        for key in stream:
            stream[key] = stream[key][:-1]
    batch = collate_dino_proprio([sample, shorter])["gripper"]["world"]
    for stream in batch.values():
        assert not stream["valid"][1, -1]
        assert not stream["target_valid"][1, -1]
        assert not stream["observed"][1, -1]


def test_direct_action_requires_real_labels_and_preserves_native_command_values(
    tmp_path,
):
    path = prepared(tmp_path, raw=True)
    with pytest.raises(ValueError, match="verified action alignment"):
        PreparedDinoProprioDataset(path, task="direct_action")
    apath = actions(path)
    sample = PreparedDinoProprioDataset(
        path, split="val", task="direct_action", action_manifest_path=apath
    )[2]
    assert set(sample["world"]) == {"image", "proprio", "action"}
    image, pose, command = (
        sample["world"][key] for key in ("image", "proprio", "action")
    )
    assert image["value"].shape == (6, 8)
    assert pose["value"].shape == (1, 4)
    assert image["observed"].all() and pose["observed"].all()
    assert not command["observed"].any()
    assert sample["task_kind"] == "direct_action"
    assert collate_dino_proprio([sample])["gripper"]["task_kind"] == ["direct_action"]
    torch.testing.assert_close(command["value"], torch.tensor([[106.0, 107.0, 108.0]]))
    assert all((s["time"] == 0).all() for s in (image, pose, command))
    assert not torch.equal(command["value"], pose["value"][:, :3])


def test_task_kind_survives_mixed_observation_regimes_in_batch_order(tmp_path):
    path = prepared(tmp_path)
    forecast = PreparedDinoProprioDataset(
        path, split="val", forecasting_probability=1.0
    )[0]
    completion = PreparedDinoProprioDataset(
        path, split="val", forecasting_probability=0.0
    )[0]
    assert forecast["task_kind"] == "forecast"
    assert completion["task_kind"] == "completion"
    assert collate_dino_proprio([completion, forecast])["gripper"]["task_kind"] == [
        "completion",
        "forecast",
    ]


def test_validation_subset_spans_every_episode_and_keeps_batches_homogeneous(tmp_path):
    from egomimic.rldb.world_model import HomogeneousWorldBatchSampler

    path = prepared(tmp_path)
    manifest = json.loads(path.read_text())
    original = manifest["episodes"][1]
    for label, embodiment in (("second", "gripper"), ("third", "other")):
        cache = path.parent / f"{label}.h5"
        cache.write_bytes((path.parent / original["path"]).read_bytes())
        episode = copy.deepcopy(original)
        episode.update(episode_id=label, embodiment=embodiment, path=cache.name)
        manifest["episodes"].append(episode)
    manifest["pose_schemas"]["other"] = copy.deepcopy(
        manifest["pose_schemas"]["gripper"]
    )
    save_json(path, manifest)
    data = DinoProprioDataModule(
        path,
        batch_size=2,
        validation_anchors_per_episode=2,
        checkpointable_sampling=True,
    )
    context = data.prepare_context(
        mode="train", normalization="prepared", normalizer=None
    )
    sampler = data.val_dataloader().batch_sampler
    assert isinstance(sampler, EvenlySpacedEpisodeBatchSampler)
    assert list(sampler) == [[0, 5], [6, 11], [12, 17]]
    assert len(sampler) == 3
    assert list(sampler) == list(data.val_dataloader().batch_sampler)
    for indices in sampler:
        assert len({data.datasets["val"][i]["embodiment"] for i in indices}) == 1
    assert isinstance(
        data.train_dataloader().batch_sampler, HomogeneousWorldBatchSampler
    )
    assert context.state["validation_sampling"] == {
        "anchors_per_episode": 2,
        "selection": "deterministic_evenly_spaced_eligible_anchors",
    }
    assert data.state_dict()["validation_anchors_per_episode"] == 2
    changed = DinoProprioDataModule(
        path,
        batch_size=2,
        validation_anchors_per_episode=3,
        checkpointable_sampling=True,
    )
    with pytest.raises(ValueError, match="checkpoint/data context mismatch"):
        changed.prepare_context(
            mode="train",
            normalization="prepared",
            normalizer=None,
            restored_state=context.state,
        )
    with pytest.raises(ValueError, match="validation_anchors_per_episode"):
        changed.load_state_dict(data.state_dict())
    with pytest.raises(ValueError, match="mutually exclusive"):
        DinoProprioDataModule(
            path, validation_anchors_per_episode=2, validation_batches_per_embodiment=2
        )


def test_validation_subset_uses_all_anchors_when_episode_is_short(tmp_path):
    path = prepared(tmp_path)
    dataset = PreparedDinoProprioDataset(path, split="val")
    sampler = EvenlySpacedEpisodeBatchSampler(
        dataset, anchors_per_episode=10, batch_size=4
    )
    assert list(sampler) == [[0, 1, 2, 3], [4, 5]]
    with pytest.raises(ValueError, match="validation only"):
        EvenlySpacedEpisodeBatchSampler(
            PreparedDinoProprioDataset(path), anchors_per_episode=2
        )


@pytest.mark.parametrize(
    "damage",
    [
        "command_index",
        "time",
        "feature_sha",
        "semantics",
        "source_sha",
        "missing_episode",
        "receipt_provenance",
    ],
)
def test_action_alignment_fails_before_training(tmp_path, damage):
    path = prepared(tmp_path)
    apath = actions(path)
    action = json.loads(apath.read_text())
    ep = action["episodes"][0]
    if damage in ("command_index", "time"):
        cache = apath.parent / ep["path"]
        with h5py.File(cache, "a") as data:
            name = "source_action_index" if damage == "command_index" else "time"
            data[name][0] += 1
        ep["sha256"] = digest(cache)
    elif damage == "feature_sha":
        ep["feature_sha256"] = "d" * 64
    elif damage == "semantics":
        action["action_schemas"]["gripper"]["semantics"] = "measured_pose"
    elif damage == "source_sha":
        ep["source_demo_sha256"] = "d" * 64
    elif damage == "receipt_provenance":
        target = apath.parent / ep["alignment"]["source_alignment_path"]
        receipt = json.loads(target.read_text())
        receipt["basis"].pop("state_action_relation_source")
        save_json(target, receipt)
        ep["alignment"]["source_alignment_sha256"] = digest(target)
    else:
        action["episodes"].pop()
    save_json(apath, action)
    with pytest.raises(ValueError, match="alignment|native action|every prepared"):
        PreparedDinoProprioDataset(
            path, task="direct_action", action_manifest_path=apath
        )


def test_feature_and_normalization_checksum_are_enforced(tmp_path):
    path = prepared(tmp_path, raw=True)
    manifest = json.loads(path.read_text())
    target = path.parent / manifest["episodes"][0]["path"]
    with h5py.File(target, "a") as data:
        data["pose"][0, 0] += 1
    with pytest.raises(ValueError, match="Artifact hash mismatch"):
        PreparedDinoProprioDataset(path)


def test_checkpointable_sampling_context_and_action_binding(tmp_path):
    path = prepared(tmp_path)
    apath = actions(path)
    kwargs = dict(
        manifest_path=path,
        task="direct_action",
        action_manifest_path=apath,
        checkpointable_sampling=True,
        batch_size=1,
    )
    data = DinoProprioDataModule(**kwargs)
    context = data.prepare_context(
        mode="train", normalization="prepared", normalizer=None
    )
    assert context.state["kind"] == "dino-proprio-prepared-v1"
    assert context.sample_schema == {"world": "image_proprio_action_labels"}
    first = next(iter(data.train_dataloader()))["gripper"]
    data.acknowledge_batch(first["sampling_cursor"])
    expected = next(iter(data.train_dataloader()))["gripper"]
    state = data.state_dict()
    resumed = DinoProprioDataModule(**kwargs)
    resumed.load_state_dict(state)
    actual = next(iter(resumed.train_dataloader()))["gripper"]
    assert actual["sampling_cursor"] == expected["sampling_cursor"] == 2
    for name in actual["world"]:
        for key in actual["world"][name]:
            torch.testing.assert_close(
                actual["world"][name][key], expected["world"][name][key], rtol=0, atol=0
            )
    changed = json.loads(apath.read_text())
    changed["note"] = "different immutable manifest"
    save_json(apath, changed)
    with pytest.raises(ValueError, match="action_manifest_sha256"):
        DinoProprioDataModule(**kwargs).load_state_dict(state)


@pytest.mark.parametrize(
    "task,objective",
    [("visual_diffusion", "visual_proprio"), ("direct_action", "direct_action")],
)
def test_dataset_batch_passes_stage_training_and_inference(tmp_path, task, objective):
    from egomimic.models.dino_proprio import DinoProprioModel
    from egomimic.pipeline.stages_dino_proprio import DinoProprioStage

    path = prepared(tmp_path, raw=True)
    apath = actions(path) if task == "direct_action" else None
    dataset = PreparedDinoProprioDataset(
        path,
        split="val",
        task=task,
        action_manifest_path=apath,
        forecasting_probability=1.0,
    )
    item = collate_dino_proprio([dataset[0]])["gripper"]
    item["optimizer_step"] = 1
    model = DinoProprioModel(
        objective=objective,
        pose_dims={"gripper": 4},
        action_dims={"gripper": 3} if task == "direct_action" else None,
        image_dim=8,
        width=24,
        depth=1,
        heads=3,
        max_cameras=2,
        max_tokens=100,
    )
    stage = DinoProprioStage(model, warmup_steps=0, pose_ramp_steps=0, sampler_steps=2)
    trained = stage.execute(copy.deepcopy(item), mode="train")
    loss = trained["loss/dino_proprio"]
    assert torch.isfinite(loss) and loss > 0
    loss.backward()
    assert model.image_in.weight.grad is not None
    inferred = stage.execute(copy.deepcopy(item), mode="inference")[
        "dino_proprio_prediction"
    ]
    expected = ("action",) if task == "direct_action" else ("image", "proprio")
    for name in expected:
        assert inferred[name].shape == item["world"][name]["value"].shape
        assert torch.isfinite(inferred[name]).all()
