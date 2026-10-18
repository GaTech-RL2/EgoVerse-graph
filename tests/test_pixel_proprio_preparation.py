"""End-to-end pixel preparation without feature or track dependencies."""

import copy
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from egomimic.rldb.pixel_proprio import PreparedPixelProprioDataset
from scripts.world_model.export_libero_action_sidecars import sha256
from scripts.world_model.prepare_pixel_proprio import (
    derive_split,
    prepare_pixels,
    resize_rgb_area,
)


def save_json(path, value):
    path.write_text(json.dumps(value))
    return path


def native_episode(root, episode, fields, *, pose_offset=0, bad_state_time=False):
    name = episode["episode_id"]
    native = root / name
    native.mkdir()
    n = episode["frames"]
    time = 0.25 + np.arange(n) * 0.05
    pose = pose_offset + np.arange(n)[:, None] + np.arange(len(fields))[None] / 10
    state = np.zeros((n, 20), np.float64)
    state[:, 0] = time + (1 if bad_state_time else 0)
    # Poison unused arrays: preparation must not consume low-res RGB or tracks.
    np.savez(
        native / "aligned.npz",
        pose=pose,
        time=time,
        source_row=np.arange(n),
        state=state,
        track=np.array([np.nan]),
        rgb=np.array([np.nan]),
    )
    native_manifest = save_json(
        native / "manifest.json",
        dict(
            schema="libero-same-recorded-state-v1",
            **episode["native_source"],
            pose_fields=fields,
            cameras=["agentview", "wrist"],
            artifacts={"aligned.npz": sha256(native / "aligned.npz")},
        ),
    )
    rgb_path = root / f"{name}.rgb.h5"
    pixels = (
        (np.arange(n * 2 * 8 * 8 * 3) % 256).astype(np.uint8).reshape(n, 2, 8, 8, 3)
    )
    with h5py.File(rgb_path, "w") as data:
        data.attrs["schema"] = "libero-native-rgb-v1"
        data.attrs["native_render"] = True
        data["rgb"] = pixels
        data["time"] = time
    receipt = save_json(
        root / f"{name}.rgb.json",
        dict(
            schema="libero-native-rgb-v1",
            status="PASS",
            native_render=True,
            native_geometry_replay_exact=True,
            frames=n,
            height=8,
            width=8,
            cameras=["agentview", "wrist"],
            source_manifest_sha256=sha256(native_manifest),
            source_aligned_sha256=sha256(native / "aligned.npz"),
            rgb_cache_sha256=sha256(rgb_path),
        ),
    )
    return (
        dict(
            embodiment=episode["embodiment"],
            episode_id=name,
            native_cache=str(native),
            rgb_cache=str(rgb_path),
            rgb_receipt=str(receipt),
        ),
        pose,
        pixels,
    )


def native_corpus(tmp_path, *, bad_state_time=False):
    episodes, locations, poses, pixels = [], [], {}, {}
    schemas = {
        emb: {"dimension": dim, "fields": [f"coordinate_{i}" for i in range(dim)]}
        for emb, dim in (("libero_gripper", 9), ("libero_sharpa", 29))
    }
    for emb, schema in schemas.items():
        for split in ("train", "val"):
            name = f"{emb}-{split}"
            episode = dict(
                embodiment=emb,
                episode_id=name,
                split=split,
                frames=4,
                path="MISSING_ENCODER_FEATURES.h5",
                sha256="e" * 64,
                native_source={
                    "source_demo": f"/fixture/{emb}.hdf5",
                    "source_demo_sha256": "a" * 64,
                    "episode": split,
                },
            )
            location, pose, rgb = native_episode(
                tmp_path,
                episode,
                schema["fields"],
                pose_offset=0 if split == "train" else 100,
                bad_state_time=bad_state_time,
            )
            episodes.append(episode)
            locations.append(location)
            poses[name], pixels[name] = pose, rgb
    source = save_json(
        tmp_path / "source.json",
        dict(schema="fixture-source-metadata", episodes=episodes, pose_schemas=schemas),
    )
    spec = save_json(tmp_path / "locations.json", {"episodes": locations})
    return source, spec, poses, pixels


def test_area_resize_is_exact_rgb_block_mean_without_crop_or_channel_reordering():
    image = np.arange(2 * 8 * 8 * 3, dtype=np.uint8).reshape(2, 8, 8, 3)
    actual = resize_rgb_area(image, (4, 4))
    expected = np.empty((2, 4, 4, 3), np.uint8)
    for camera in range(2):
        for row in range(4):
            for col in range(4):
                expected[camera, row, col] = np.rint(
                    image[camera, row * 2 : row * 2 + 2, col * 2 : col * 2 + 2].mean(
                        (0, 1)
                    )
                )
    np.testing.assert_array_equal(actual, expected)
    # Document the otherwise ambiguous uint8 half-way rounding.
    half = np.array([0, 1, 1, 0], np.uint8).reshape(2, 2, 1).repeat(3, axis=-1)
    assert np.all(resize_rgb_area(half, (1, 1)) == 0)
    with pytest.raises(ValueError, match="integer downsampling"):
        resize_rgb_area(image, (3, 3))


def test_prepare_mixed_corpus_preserves_split_time_cameras_and_train_only_pose_statistics(
    tmp_path, monkeypatch
):
    source, spec, poses, pixels = native_corpus(tmp_path)
    original = np.lib.npyio.NpzFile.__getitem__

    def guard(data, key):
        assert key in ("pose", "time", "source_row", "state")
        return original(data, key)

    monkeypatch.setattr(np.lib.npyio.NpzFile, "__getitem__", guard)
    output = tmp_path / "prepared"
    result = prepare_pixels(
        source, spec, output, image_size=[4, 4], patch_size=2, resize_batch_size=2
    )
    assert result["episodes"] == 4 and result["frames"] == 16
    manifest = json.loads((output / "manifest.json").read_text())
    stats = json.loads((output / manifest["normalization"]["path"]).read_text())
    assert "dino_identity" not in manifest and "action_schemas" not in manifest
    assert set(stats["statistics"]) == {"pose/libero_gripper", "pose/libero_sharpa"}
    assert manifest["split_provenance"]["counts"] == {
        "libero_gripper": {"train": 1, "val": 1},
        "libero_sharpa": {"train": 1, "val": 1},
    }
    assert [
        (e["embodiment"], e["episode_id"], e["split"]) for e in manifest["episodes"]
    ] == [
        (e["embodiment"], e["episode_id"], e["split"])
        for e in json.loads(source.read_text())["episodes"]
    ]
    for ep in manifest["episodes"]:
        with (
            h5py.File(output / ep["rgb_path"], "r") as rgb,
            h5py.File(output / ep["path"], "r") as data,
        ):
            np.testing.assert_array_equal(
                rgb["rgb"][:], resize_rgb_area(pixels[ep["episode_id"]], (4, 4))
            )
            np.testing.assert_array_equal(rgb["time"][:], data["time"][:])
            assert (
                rgb["cameras"].asstr()[:].tolist()
                == ep["cameras"]
                == ["agentview", "wrist"]
            )
            assert rgb["rgb"].dtype == np.uint8 and "action" not in data
        item = stats["statistics"][f"pose/{ep['embodiment']}"]
        train_pose = poses[f"{ep['embodiment']}-train"]
        np.testing.assert_allclose(
            item["mean"], train_pose.astype(np.float32).mean(0), rtol=1e-6
        )
        assert item["count"] == 4
    dataset = PreparedPixelProprioDataset(output / "manifest.json", split="val")
    assert dataset[0]["world"]["image"]["value"].shape[-1] == 12
    assert dataset[4]["world"]["proprio"]["value"].shape[-1] == 29
    assert (
        dataset[0]["world"]["proprio"]["value"].min() > 80
    )  # Validation poses did not fit stats.
    before = sha256(output / "manifest.json")
    with pytest.raises(FileExistsError, match="immutable"):
        prepare_pixels(source, spec, output, image_size=[4, 4], patch_size=2)
    assert sha256(output / "manifest.json") == before


@pytest.mark.parametrize(
    "damage", ["state_time", "rgb_checksum", "pose_fields", "split"]
)
def test_prepare_rejects_alignment_corruption_or_split_changes_before_output(
    tmp_path, damage
):
    source, spec, _, _ = native_corpus(tmp_path, bad_state_time=damage == "state_time")
    locations = json.loads(spec.read_text())["episodes"]
    if damage == "rgb_checksum":
        with h5py.File(locations[0]["rgb_cache"], "a") as data:
            data["rgb"][0, 0, 0, 0, 0] += 1
    elif damage in ("pose_fields", "split"):
        original = json.loads(source.read_text())
        if damage == "pose_fields":
            original["pose_schemas"]["libero_gripper"]["fields"][0] = (
                "incorrect_coordinate"
            )
        else:
            original["episodes"][1]["split"] = "train"
        save_json(source, original)
    output = tmp_path / "prepared"
    with pytest.raises(ValueError):
        prepare_pixels(source, spec, output, image_size=[4, 4], patch_size=2)
    assert not output.exists()


def test_optional_real_action_export_produces_loader_compatible_labels(tmp_path):
    from tests.test_libero_action_sidecar_export import bundle

    source, raw = bundle(tmp_path)
    manifest = json.loads(source.read_text())
    locations = []
    for episode in manifest["episodes"]:
        location, _, _ = native_episode(
            tmp_path, episode, manifest["pose_schemas"][episode["embodiment"]]["fields"]
        )
        # The raw fixture stores timestamps starting at zero; use its actual
        # rows for all native arrays, then rebind the receipt hashes.
        native = Path(location["native_cache"])
        with np.load(native / "aligned.npz") as data:
            arrays = {key: data[key] for key in data.files}
        arrays["time"] = np.arange(episode["frames"], dtype=np.float64) * 0.05
        arrays["state"][:, 0] = arrays["time"]
        np.savez(native / "aligned.npz", **arrays)
        nmanifest = json.loads((native / "manifest.json").read_text())
        nmanifest["artifacts"]["aligned.npz"] = sha256(native / "aligned.npz")
        save_json(native / "manifest.json", nmanifest)
        with h5py.File(location["rgb_cache"], "a") as data:
            data["time"][:] = arrays["time"]
        receipt = json.loads(Path(location["rgb_receipt"]).read_text())
        receipt.update(
            source_manifest_sha256=sha256(native / "manifest.json"),
            source_aligned_sha256=sha256(native / "aligned.npz"),
            rgb_cache_sha256=sha256(location["rgb_cache"]),
        )
        save_json(Path(location["rgb_receipt"]), receipt)
        locations.append(location)
    spec = save_json(tmp_path / "native.json", {"episodes": locations})
    output = tmp_path / "prepared"
    result = prepare_pixels(
        source, spec, output, image_size=[4, 4], patch_size=2, raw_export_dir=raw
    )
    assert result["action_episodes"] == 2
    dataset = PreparedPixelProprioDataset(
        output / "manifest.json", split="val", task="direct_action"
    )
    np.testing.assert_array_equal(
        dataset[2]["world"]["action"]["value"].numpy(),
        (np.arange(28).reshape(4, 7) / 30)[2:3].astype(np.float32),
    )


def test_gripper_selection_preserves_frozen_split_without_opening_sharpa(tmp_path):
    source, spec, _, _ = native_corpus(tmp_path)
    locations = json.loads(spec.read_text())
    for item in locations["episodes"]:
        if item["embodiment"] == "libero_sharpa":
            item.update(
                native_cache="/nonexistent/sharpa",
                rgb_cache="/nonexistent/sharpa.h5",
                rgb_receipt="/nonexistent/sharpa.json",
            )
    save_json(spec, locations)
    output = tmp_path / "gripper"
    result = prepare_pixels(
        source,
        spec,
        output,
        image_size=[4, 4],
        patch_size=2,
        embodiments=["libero_gripper"],
    )
    assert result["episodes"] == 2
    prepared = json.loads((output / "manifest.json").read_text())
    assert prepared["selected_embodiments"] == ["libero_gripper"]
    assert set(prepared["pose_schemas"]) == {"libero_gripper"}
    assert [(ep["episode_id"], ep["split"]) for ep in prepared["episodes"]] == [
        ("libero_gripper-train", "train"),
        ("libero_gripper-val", "val"),
    ]
    # Selection does not excuse a corrupted frozen source split in the excluded source.
    original = json.loads(source.read_text())
    original["episodes"][-1]["split"] = "train"
    save_json(source, original)
    with pytest.raises(ValueError, match="nonempty train/val"):
        prepare_pixels(
            source,
            spec,
            tmp_path / "bad",
            image_size=[4, 4],
            patch_size=2,
            embodiments=["libero_gripper"],
        )


def task_episodes(tasks=10, episodes_per_task=50):
    return [
        dict(
            embodiment="libero_gripper",
            episode_id=f"task_{task}-demo_{i}",
            frames=75,
            split="val" if task < min(tasks, 5) and i == 0 else "train",
            native_source=dict(
                source_demo=f"/fixture/task_{task}_demo.hdf5",
                source_demo_sha256=f"{task + 1:064x}",
                episode=f"demo_{i}",
            ),
        )
        for task in range(tasks)
        for i in range(episodes_per_task)
    ]


def test_balanced_split_is_order_independent_preserves_old_holdout_and_covers_all_tasks():
    from types import SimpleNamespace

    from egomimic.rldb.pixel_proprio import EvenlySpacedEpisodeBatchSampler

    episodes = task_episodes()
    unchanged = copy.deepcopy(episodes)
    result, policy = derive_split(episodes, 5, 42)
    reversed_result, reversed_policy = derive_split(list(reversed(episodes)), 5, 42)
    assert episodes == unchanged
    assert result == reversed_result and policy == reversed_policy
    assert policy["counts"] == {"libero_gripper": {"train": 450, "val": 50}}
    assert all(
        count["train"] == 45 and count["val"] == 5
        for count in policy["task_counts"].values()
    )
    assert (
        sum(count["retained_original_val"] for count in policy["task_counts"].values())
        == 5
    )
    assert all(
        ep["split"] == "val" for ep in result.values() if ep["original_split"] == "val"
    )
    assert policy["original_identities_sha256"] != policy["derived_identities_sha256"]
    _, other_seed = derive_split(episodes, 5, 43)
    assert (
        other_seed["derived_identities_sha256"] != policy["derived_identities_sha256"]
    )
    heldout = [ep for ep in result.values() if ep["split"] == "val"]
    dataset = SimpleNamespace(
        split="val", episodes=heldout, counts=[ep["frames"] for ep in heldout]
    )
    sampler = EvenlySpacedEpisodeBatchSampler(dataset, 2, 1)
    assert len(sampler) == len(list(sampler)) == 100
    for i in range(50):
        assert list(sampler)[2 * i : 2 * i + 2] == [[i * 75], [i * 75 + 74]]


@pytest.mark.parametrize(
    "damage", ["task_name", "source_hash", "excess_old_val", "empty_train"]
)
def test_balanced_split_rejects_ambiguous_task_or_destructive_split(damage):
    episodes = task_episodes(tasks=2, episodes_per_task=4)
    count = 2
    if damage == "task_name":
        episodes[0]["episode_id"] = "different-task-demo_0"
    elif damage == "source_hash":
        episodes[0]["native_source"]["source_demo_sha256"] = "e" * 64
    elif damage == "excess_old_val":
        episodes[1]["split"] = episodes[2]["split"] = "val"
    else:
        count = 4
    with pytest.raises(ValueError):
        derive_split(episodes, count, 42)


def balanced_native_corpus(tmp_path):
    episodes = task_episodes(tasks=2, episodes_per_task=4)
    schema = dict(dimension=9, fields=[f"p{i}" for i in range(9)])
    derived, _ = derive_split(episodes, 2, 42)
    locations, poses = [], {}
    for ep in episodes:
        ep["frames"] = 4
        key = ep["embodiment"], ep["episode_id"]
        # Every old/new validation episode is extreme, exposing any stats leak.
        offset = 10000 if derived[key]["split"] == "val" else 0
        location, pose, _ = native_episode(
            tmp_path, ep, schema["fields"], pose_offset=offset
        )
        locations.append(location)
        poses[ep["episode_id"]] = pose
    source = save_json(
        tmp_path / "source.json",
        dict(episodes=episodes, pose_schemas={"libero_gripper": schema}),
    )
    spec = save_json(tmp_path / "locations.json", dict(episodes=locations))
    return source, spec, poses


def test_derived_preparation_normalizes_only_new_training_and_records_original_assignments(
    tmp_path,
):
    source, spec, poses = balanced_native_corpus(tmp_path)
    before = sha256(source)
    output = tmp_path / "derived"
    result = prepare_pixels(
        source,
        spec,
        output,
        image_size=[4, 4],
        patch_size=2,
        validation_episodes_per_task=2,
        split_seed=42,
    )
    assert sha256(source) == before
    assert result["split_counts"] == {"libero_gripper": {"train": 4, "val": 4}}
    manifest = json.loads((output / "manifest.json").read_text())
    saved = manifest["split_provenance"]
    assert sha256(output / saved["path"]) == saved["sha256"]
    policy = json.loads((output / saved["path"]).read_text())
    assert policy["kind"] == "balanced_native_libero_tasks_v1" and policy["seed"] == 42
    assert saved["original_identities_sha256"] == policy["original_identities_sha256"]
    assert sum(ep["split"] != ep["original_split"] for ep in manifest["episodes"]) == 2
    norm = json.loads((output / "normalization.json").read_text())
    stats = norm["statistics"]["pose/libero_gripper"]
    expected = np.concatenate(
        [
            poses[ep["episode_id"]]
            for ep in manifest["episodes"]
            if ep["split"] == "train"
        ]
    )
    np.testing.assert_allclose(stats["mean"], expected.mean(0), atol=1e-6)
    assert max(stats["mean"]) < 10 and stats["count"] == 16
    assert len(norm["train_episodes"]) == 4
    data = PreparedPixelProprioDataset(output / "manifest.json", split="val")
    assert len(data.episodes) == 4 and data[0]["world"]["proprio"]["value"].min() > 1000


def test_derived_split_still_rejects_original_raw_command_split_mismatch(
    tmp_path, monkeypatch
):
    from scripts.world_model import prepare_pixel_proprio as module

    source, spec, _ = balanced_native_corpus(tmp_path)
    original = json.loads(source.read_text())["episodes"]
    raw_values = {}
    for ep in original:
        raw = copy.deepcopy(ep)
        raw["split"] = "train" if ep["split"] == "val" else "val"
        raw_values[(ep["embodiment"], ep["episode_id"])] = (
            np.zeros((4, 7), np.float32),
            0.25 + np.arange(4) * 0.05,
            raw,
        )
    # The low-level binary bundle verifier has independent corruption tests;
    # this boundary specifically must compare against ORIGINAL assignments.
    monkeypatch.setattr(module, "_raw_actions", lambda *_: ({}, {}, raw_values))
    output = tmp_path / "bad"
    with pytest.raises(ValueError, match="Raw action source/split/state time"):
        prepare_pixels(
            source,
            spec,
            output,
            image_size=[4, 4],
            patch_size=2,
            raw_export_dir=tmp_path,
            validation_episodes_per_task=2,
            split_seed=42,
        )
    assert not output.exists()
