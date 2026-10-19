"""Raw-pixel ordering, full-episode sampling, provenance and two-stream behavior."""

import copy
import hashlib
import json
import shutil
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch

from egomimic.rldb.pixel_proprio import (
    DistributedPixelTrainingBatchSampler,
    DistributedPixelValidationBatchSampler,
    EvenlySpacedEpisodeBatchSampler,
    PreparedPixelProprioDataset,
    collate_pixel_proprio,
    patchify_rgb,
    sample_discrete_window,
    sample_full_episode_times,
)
from egomimic.rldb.pixel_proprio_datamodule import PixelProprioDataModule
from egomimic.rldb.world_model import HomogeneousWorldBatchSampler


@pytest.mark.parametrize(
    "mode,strategy",
    [("continuous", "full_episode_uniform"), ("discrete", "discrete_window")],
)
def test_joint_state_observation_masks_are_tied_at_data_boundary(
    tmp_path, mode, strategy
):
    manifest = pixel_manifest(tmp_path, include_sharpa=False, frames=16)
    dataset = PreparedPixelProprioDataset(
        manifest,
        task="visual_state_action",
        mode=mode,
        sampling_strategy=strategy,
        min_frames=2,
        max_frames=8,
        discrete_frames=3,
        stride=4,
        couple_state_observations=True,
    )
    kinds = set()
    for draw in range(20):
        sample = dataset[(draw % len(dataset), draw, draw)]
        streams = sample["world"]
        kinds.add(sample["task_kind"])
        observed = streams["image"]["observed"].reshape(-1, 8)
        assert (observed == streams["proprio"]["observed"][:, None]).all()
        assert not streams["action"]["observed"].any()
        assert streams["proprio"]["value"].shape[-1] == (
            36 if mode == "discrete" else 9
        )
    assert kinds == {"forecast", "completion"}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value))
    return path


def pixel_manifest(tmp_path, *, include_sharpa=True, actions=True, frames=6):
    schemas = {
        "libero_gripper": {"dimension": 9, "fields": [f"gripper_{i}" for i in range(9)]}
    }
    if include_sharpa:
        schemas["libero_sharpa"] = {
            "dimension": 29,
            "fields": [f"hand_{i}" for i in range(29)],
        }
    episodes = []
    for embodiment, schema in schemas.items():
        for split in ("train", "val"):
            name = f"{embodiment}-{split}"
            rgb_path, path = tmp_path / f"{name}-rgb.h5", tmp_path / f"{name}.h5"
            time = 1e9 + np.arange(frames) * 2.0
            rgb = (
                (np.arange(frames * 2 * 4 * 4 * 3) % 256)
                .astype(np.uint8)
                .reshape(frames, 2, 4, 4, 3)
            )
            with h5py.File(rgb_path, "w") as data:
                data["rgb"] = rgb
                data["time"] = time
            with h5py.File(path, "w") as data:
                data["time"] = time
                data["pose"] = np.arange(
                    frames * schema["dimension"], dtype=np.float32
                ).reshape(frames, -1)
                data["pose_valid"] = np.ones(frames, bool)
                if actions and embodiment == "libero_gripper":
                    data["action"] = (
                        np.arange(frames * 7, dtype=np.float32).reshape(frames, 7) / 100
                    )
                    data["action_valid"] = np.ones(frames, bool)
            ep = dict(
                embodiment=embodiment,
                episode_id=name,
                split=split,
                frames=frames,
                path=path.name,
                sha256=digest(path),
                rgb_path=rgb_path.name,
                rgb_sha256=digest(rgb_path),
                cameras=["agentview", "wrist"],
                frame="fixed_reference",
                pose_kind="measured_pose",
                native_source={"source_demo_sha256": "a" * 64, "episode": name},
            )
            if actions and embodiment == "libero_gripper":
                proof = save_json(
                    tmp_path / f"{name}-alignment.json",
                    dict(
                        schema="pixel-proprio-action-alignment-v1",
                        relation="action_t_applied_to_state_t",
                        source_demo_sha256="a" * 64,
                        source_episode=name,
                        producer_sha256="b" * 64,
                        raw_export_metadata_sha256="c" * 64,
                        raw_payload_sha256="d" * 64,
                        cache_frame_index=list(range(frames)),
                        source_state_index=list(range(frames)),
                        source_action_index=list(range(frames)),
                        basis={
                            "cache_index_source": "synthetic fixture mapping",
                            "action_index_source": "synthetic fixture commands",
                            "state_action_relation_source": "synthetic fixture relation",
                        },
                    ),
                )
                ep["action_provenance"] = {"path": proof.name, "sha256": digest(proof)}
            episodes.append(ep)
    norm = save_json(
        tmp_path / "normalization.json",
        dict(
            schema="pixel-proprio-normalization-v1",
            scope="train_only",
            train_episodes=sorted(
                [
                    [ep["embodiment"], ep["episode_id"], ep["sha256"]]
                    for ep in episodes
                    if ep["split"] == "train"
                ]
            ),
            statistics={
                f"pose/{emb}": {
                    "mean": [1.0] * schema["dimension"],
                    "std": [2.0] * schema["dimension"],
                    "count": frames,
                }
                for emb, schema in schemas.items()
            },
        ),
    )
    manifest = dict(
        schema="pixel-proprio-prepared-v1",
        pose_schemas=schemas,
        episodes=episodes,
        pixel_identity={
            "image_size": [4, 4],
            "patch_size": 2,
            "channels": 3,
            "patch_order": "row_major_hwc",
            "normalization": "uint8_to_minus_one_one",
        },
        normalization={
            "scope": "train_only",
            "arrays_normalized": False,
            "path": norm.name,
            "sha256": digest(norm),
        },
    )
    if actions:
        manifest["action_schemas"] = {
            "libero_gripper": {
                "dimension": 7,
                "fields": [f"command_{i}" for i in range(7)],
                "semantics": "action_command",
                "value_space": "native_commands",
            }
        }
    return save_json(tmp_path / "manifest.json", manifest)


def test_patch_order_retains_every_raw_hwc_value_and_each_camera():
    rgb = np.arange(1 * 2 * 4 * 4 * 3, dtype=np.uint8).reshape(1, 2, 4, 4, 3)
    patches = patchify_rgb(rgb, 2)
    assert patches.shape == (1, 2, 4, 12) and patches.dtype == np.float32
    for camera in range(2):
        for row in range(2):
            for col in range(2):
                expected = rgb[
                    0, camera, row * 2 : (row + 1) * 2, col * 2 : (col + 1) * 2
                ].reshape(-1)
                np.testing.assert_allclose(
                    patches[0, camera, row * 2 + col], expected / 127.5 - 1, atol=1e-7
                )
    rgb = np.zeros((1, 2, 160, 160, 3), np.uint8)
    rgb[0, 1] = 255
    patches = patchify_rgb(rgb, 16)
    assert patches.shape == (1, 2, 100, 768)
    assert np.all(patches[0, 0] == -1) and np.all(patches[0, 1] == 1)


def test_full_episode_uniform_sampling_clips_maximum_without_time_window():
    all_frames = sample_full_episode_times(
        128, np.random.default_rng(1), 128, 128, anchor=37
    )
    np.testing.assert_array_equal(all_frames, np.arange(128))
    clipped = sample_full_episode_times(5, np.random.default_rng(2), 5, 128, anchor=2)
    np.testing.assert_array_equal(clipped, np.arange(5))
    seen = set()
    rng = np.random.default_rng(21)
    for _ in range(200):
        selected = sample_full_episode_times(20, rng, 2, 2, anchor=0)
        assert len(selected) == len(set(selected)) == 2 and selected[0] == 0
        seen.add(selected[1])
    assert seen == set(range(1, 20))


def test_pixels_and_both_measured_pose_schemas_need_no_encoder_tracks_or_commands(
    tmp_path, monkeypatch
):
    path = pixel_manifest(tmp_path, actions=False)
    original = h5py.Group.__getitem__

    def guard(group, key):
        assert key not in ("image", "dino", "track", "action", "action_valid")
        return original(group, key)

    monkeypatch.setattr(h5py.Group, "__getitem__", guard)
    dataset = PreparedPixelProprioDataset(
        path, split="val", min_frames=6, max_frames=128
    )
    assert set(dataset.normalization) == {"pose/libero_gripper", "pose/libero_sharpa"}
    first, second = dataset[0], dataset[6]
    assert first["world"]["image"]["value"].shape == (48, 12)
    assert first["world"]["proprio"]["value"].shape == (6, 9)
    assert second["world"]["proprio"]["value"].shape == (6, 29)
    torch.testing.assert_close(
        first["world"]["proprio"]["value"][0], (torch.arange(9) - 1) / 2
    )
    assert torch.unique(first["world"]["image"]["time"]).numel() == 6
    assert (
        first["world"]["image"]["time"].max() - first["world"]["image"]["time"].min()
        == 10
    )


@pytest.mark.parametrize("image_only", [False, True])
def test_forecast_and_completion_metadata_padding_and_no_future_observations(
    tmp_path, image_only
):
    path = pixel_manifest(tmp_path, actions=False)
    forecast = PreparedPixelProprioDataset(
        path,
        split="val",
        forecasting_probability=1.0,
        min_frames=6,
        max_frames=6,
        image_only=image_only,
    )[0]
    completion = PreparedPixelProprioDataset(
        path,
        split="val",
        forecasting_probability=0.0,
        min_frames=2,
        max_frames=2,
        image_only=image_only,
    )[0]
    image = forecast["world"]["image"]
    cutoff = image["time"][image["observed"]].max()
    assert (image["time"][~image["observed"]] > cutoff).all()
    if image_only:
        assert set(forecast["world"]) == set(completion["world"]) == {"image"}
    else:
        pose = forecast["world"]["proprio"]
        assert (pose["time"][pose["observed"]] <= cutoff).all()
    batch = collate_pixel_proprio([forecast, completion])["libero_gripper"]
    assert batch["task_kind"] == ["forecast", "completion"]
    assert batch["pixel_metadata"] == [
        forecast["pixel_metadata"],
        completion["pixel_metadata"],
    ]
    metadata = forecast["pixel_metadata"]
    assert metadata["timestamps_seconds"] == (1e9 + np.arange(6) * 2).tolist()
    assert metadata["frame_indices"] == list(range(6))
    assert metadata["camera_names"] == ["agentview", "wrist"]
    for stream in batch["world"].values():
        assert (
            not stream["valid"][1, -1]
            and not stream["observed"][1, -1]
            and not stream["target_valid"][1, -1]
        )


def test_direct_bc_is_one_observed_anchor_and_real_unscaled_command(tmp_path):
    path = pixel_manifest(tmp_path, include_sharpa=False)
    sample = PreparedPixelProprioDataset(path, task="direct_action", split="val")[3]
    assert sample["task_kind"] == "direct_action"
    image, pose, action = (
        sample["world"][name] for name in ("image", "proprio", "action")
    )
    assert image["value"].shape == (8, 12) and pose["value"].shape == (1, 9)
    assert (
        image["observed"].all()
        and pose["observed"].all()
        and not action["observed"].any()
    )
    torch.testing.assert_close(
        action["value"], torch.arange(21, 28)[None].float() / 100
    )
    assert all(
        torch.equal(s["time"], torch.zeros_like(s["time"]))
        for s in (image, pose, action)
    )


def test_direct_mixed_corpus_fails_closed_when_sharpa_commands_are_unverified(tmp_path):
    path = pixel_manifest(tmp_path)
    with pytest.raises(ValueError, match="verified real command labels"):
        PreparedPixelProprioDataset(path, task="direct_action")


@pytest.mark.parametrize(
    "damage",
    ["rgb_checksum", "rgb_time", "action_proof", "image_statistics", "time_window"],
)
def test_pixel_provenance_normalization_and_sampling_fail_closed(tmp_path, damage):
    path = pixel_manifest(tmp_path, include_sharpa=False)
    manifest = json.loads(path.read_text())
    episode = manifest["episodes"][1]
    kwargs = {}
    if damage in ("rgb_checksum", "rgb_time"):
        rgb_path = path.parent / episode["rgb_path"]
        with h5py.File(rgb_path, "a") as data:
            if damage == "rgb_checksum":
                data["rgb"][0, 0, 0, 0, 0] += 1
            else:
                data["time"][0] += 0.25
        if damage == "rgb_time":
            episode["rgb_sha256"] = digest(rgb_path)
    elif damage == "action_proof":
        proof_path = path.parent / episode["action_provenance"]["path"]
        proof = json.loads(proof_path.read_text())
        proof["source_action_index"][0] += 1
        save_json(proof_path, proof)
        episode["action_provenance"]["sha256"] = digest(proof_path)
        kwargs["task"] = "direct_action"
    elif damage == "image_statistics":
        norm_path = path.parent / manifest["normalization"]["path"]
        norm = json.loads(norm_path.read_text())
        norm["statistics"]["image"] = {"mean": [0], "std": [1], "count": 1}
        save_json(norm_path, norm)
        manifest["normalization"]["sha256"] = digest(norm_path)
    else:
        kwargs.update(max_past_seconds=3, max_future_seconds=3)
    save_json(path, manifest)
    with pytest.raises(ValueError):
        PreparedPixelProprioDataset(path, split="val", **kwargs)[0]


@pytest.mark.parametrize("image_only", [False, True])
def test_full_episode_sampler_and_heldout_coverage_survive_exact_resume(
    tmp_path, image_only
):
    path = pixel_manifest(tmp_path, actions=False)
    kwargs = dict(
        manifest_path=path,
        batch_size=1,
        checkpointable_sampling=True,
        validation_anchors_per_episode=2,
        min_frames=2,
        max_frames=128,
        image_only=image_only,
    )
    data = PixelProprioDataModule(**kwargs)
    context = data.prepare_context(
        mode="train", normalization="prepared", normalizer=None
    )
    assert context.state["sampling"]["sampling_strategy"] == "full_episode_uniform"
    assert context.state["kind"] == (
        "pixel-image-prepared-v1" if image_only else "pixel-proprio-prepared-v1"
    )
    assert list(data.val_dataloader().batch_sampler) == [[0], [5], [6], [11]]
    first = next(iter(data.train_dataloader()))
    data.acknowledge_batch(next(iter(first.values()))["sampling_cursor"])
    state = data.state_dict()
    expected = next(iter(data.train_dataloader()))
    restored = PixelProprioDataModule(**kwargs)
    restored.load_state_dict(state)
    actual = next(iter(restored.train_dataloader()))
    assert set(actual) == set(expected)
    key = next(iter(actual))
    assert actual[key]["task_kind"] == expected[key]["task_kind"]
    for name in actual[key]["world"]:
        for field in actual[key]["world"][name]:
            torch.testing.assert_close(
                actual[key]["world"][name][field],
                expected[key]["world"][name][field],
                rtol=0,
                atol=0,
            )
    changed = PixelProprioDataModule(**{**kwargs, "max_frames": 127})
    with pytest.raises(ValueError, match="settings"):
        changed.load_state_dict(state)
    changed = PixelProprioDataModule(**{**kwargs, "image_only": not image_only})
    with pytest.raises(ValueError, match="settings"):
        changed.load_state_dict(state)
    with pytest.raises(ValueError, match="context mismatch"):
        changed.prepare_context(
            mode="train",
            normalization="prepared",
            normalizer=None,
            restored_state=context.state,
        )


@pytest.mark.parametrize("direct", [False, True])
def test_pixel_batches_train_and_inference_ignore_withheld_values(tmp_path, direct):
    from egomimic.models.image_proprio import ImageProprioModel

    path = pixel_manifest(tmp_path, include_sharpa=False)
    task = "direct_action" if direct else "visual_diffusion"
    dataset = PreparedPixelProprioDataset(
        path,
        task=task,
        split="val",
        min_frames=2,
        max_frames=2,
        forecasting_probability=1.0,
    )
    batch = collate_pixel_proprio([dataset[0]])["libero_gripper"]["world"]
    model = ImageProprioModel(
        pose_dims={"libero_gripper": 9},
        action_dims={"libero_gripper": 7} if direct else None,
        objective="direct_action" if direct else "visual_proprio",
        image_dim=12,
        width=24,
        depth=1,
        heads=3,
        max_cameras=2,
        max_tokens=100,
    )
    losses = model.training_losses(batch, "libero_gripper")
    assert torch.isfinite(losses["total"]) and losses["total"] > 0
    losses["total"].backward()
    assert (
        model.image_in.weight.grad is not None and model.image_in.weight.grad.norm() > 0
    )
    changed = copy.deepcopy(batch)
    for stream in changed.values():
        stream["value"][~stream["observed"]] = float("nan")
    torch.manual_seed(7)
    expected = model.sample(batch, "libero_gripper", steps=2)
    torch.manual_seed(7)
    actual = model.sample(changed, "libero_gripper", steps=2)
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)


def test_discrete_window_contains_anchor_and_all_four_state_command_rows(tmp_path):
    path = pixel_manifest(tmp_path, include_sharpa=False, frames=75)
    rng = np.random.default_rng(3)
    for anchor in (0, 30, 74):
        selected = sample_discrete_window(75, rng, 16, 4, anchor)
        assert selected.shape == (16, 4) and anchor in selected
        np.testing.assert_array_equal(np.diff(selected.reshape(-1)), np.ones(63))
    dataset = PreparedPixelProprioDataset(
        path,
        split="val",
        task="visual_state_action",
        mode="discrete",
        sampling_strategy="discrete_window",
        discrete_frames=16,
        stride=4,
        forecasting_probability=1,
    )
    sample = dataset[74]
    meta, world = sample["pixel_metadata"], sample["world"]
    rows = np.asarray(meta["proprio_frame_indices"])
    assert rows.shape == (16, 4) and rows[-1, -1] == 74
    np.testing.assert_array_equal(meta["frame_indices"], rows[:, 0])
    assert world["image"]["value"].shape == (16 * 8, 12)
    assert world["proprio"]["value"].shape == (16, 36)
    assert world["action"]["value"].shape == (16, 28)
    assert not world["action"]["observed"].any()
    expected_pose = (np.arange(75 * 9).reshape(75, 9)[rows].reshape(16, 36) - 1) / 2
    np.testing.assert_allclose(world["proprio"]["value"].numpy(), expected_pose)
    expected_action = (
        np.arange(75 * 7).reshape(75, 7).astype(np.float32)[rows].reshape(16, 28) / 100
    )
    np.testing.assert_allclose(world["action"]["value"].numpy(), expected_action)
    known_images = world["image"]["observed"].reshape(16, -1).all(1).numpy()
    known_pose = world["proprio"]["observed"].numpy()
    cutoff = rows[known_images][-1, -1]
    assert (rows[known_pose] <= cutoff).all()
    assert (rows[~known_images] > cutoff).all()
    assert not known_pose[~known_images].any()
    first_unknown_image = np.asarray(meta["frame_indices"])[~known_images][0]
    assert rows[known_pose].max() < first_unknown_image
    assert world["action"]["valid"].all()
    np.testing.assert_array_equal(
        world["action"]["target_valid"].numpy(), ~known_images
    )
    assert sample["task_kind"] == "forecast"
    assert collate_pixel_proprio([sample])["libero_gripper"]["task_kind"] == [
        "forecast"
    ]


@pytest.mark.parametrize("image_only", [False, True])
def test_shared_validation_grid_masks_and_requirements_across_discrete_and_continuous(
    tmp_path,
    image_only,
):
    from egomimic.pl_utils.data_context import EvaluationDataRequirements

    path = pixel_manifest(tmp_path, include_sharpa=False, actions=False, frames=75)
    shared = dict(
        manifest_path=path,
        validation_anchors_per_episode=2,
        validation_sampling_strategy="discrete_window",
        validation_discrete_frames=16,
        validation_stride=4,
        forecasting_probability=1,
        checkpointable_sampling=True,
        image_only=image_only,
    )
    discrete = PixelProprioDataModule(
        **shared,
        mode="discrete",
        sampling_strategy="discrete_window",
        discrete_frames=16,
        stride=4,
    )
    continuous = PixelProprioDataModule(**shared)
    for data in (discrete, continuous):
        data.configure_evaluation(
            EvaluationDataRequirements(
                required_keys=("world", "embodiment", "task_kind", "pixel_metadata")
            )
        )
        with pytest.raises(ValueError, match="Unsupported"):
            data.configure_evaluation(
                EvaluationDataRequirements(complete_episodes=True)
            )
        with pytest.raises(ValueError, match="Unsupported"):
            data.configure_evaluation(
                EvaluationDataRequirements(required_keys=("missing",))
            )
    for left, right in zip(discrete.val_dataloader(), continuous.val_dataloader()):
        left, right = left["libero_gripper"], right["libero_gripper"]
        assert (
            left["pixel_metadata"][0]["frame_indices"]
            == right["pixel_metadata"][0]["frame_indices"]
        )
        assert (
            left["pixel_metadata"][0]["timestamps_seconds"]
            == right["pixel_metadata"][0]["timestamps_seconds"]
        )
        for key in left["world"]["image"]:
            torch.testing.assert_close(
                left["world"]["image"][key],
                right["world"]["image"][key],
                rtol=0,
                atol=0,
            )
        if image_only:
            assert set(left["world"]) == set(right["world"]) == {"image"}
        else:
            torch.testing.assert_close(
                left["world"]["proprio"]["observed"],
                right["world"]["proprio"]["observed"],
            )
            assert left["world"]["proprio"]["value"].shape[-1] == 36
            assert right["world"]["proprio"]["value"].shape[-1] == 9
    context = continuous.prepare_context(
        mode="train", normalization="prepared", normalizer=None
    )
    assert context.state["validation_overrides"]["stride"] == 4
    changed = PixelProprioDataModule(**{**shared, "validation_stride": 3})
    with pytest.raises(ValueError, match="context mismatch"):
        changed.prepare_context(
            mode="train",
            normalization="prepared",
            normalizer=None,
            restored_state=context.state,
        )
    with pytest.raises(ValueError, match="sampling restore"):
        changed.load_state_dict(continuous.state_dict())


@pytest.mark.parametrize("discrete", [False, True])
def test_forecast_prefix_command_labels_do_not_contribute_to_loss_or_metrics(
    tmp_path, discrete
):
    from types import SimpleNamespace

    from egomimic.eval.image_proprio_eval import ImageProprioEval
    from egomimic.models.image_proprio import ImageProprioModel

    path = pixel_manifest(tmp_path, include_sharpa=False, frames=75)
    settings = dict(
        task="visual_state_action",
        split="val",
        forecasting_probability=1,
        min_frames=4,
        max_frames=4,
    )
    if discrete:
        settings.update(
            mode="discrete",
            sampling_strategy="discrete_window",
            discrete_frames=16,
            stride=4,
        )
    sample = PreparedPixelProprioDataset(path, **settings)[0]
    batch = collate_pixel_proprio([sample])
    world = batch["libero_gripper"]["world"]
    targets = world["action"]["target_valid"]
    assert targets.any() and (~targets).any()
    assert world["action"]["valid"].all() and not world["action"]["observed"].any()
    changed = copy.deepcopy(batch)
    changed["libero_gripper"]["world"]["action"]["value"][~targets] = 1e6
    chunk = 4 if discrete else 1
    model = ImageProprioModel(
        pose_dims={"libero_gripper": 9 * chunk},
        action_dims={"libero_gripper": 7 * chunk},
        objective="visual_state_action",
        image_dim=12,
        width=24,
        depth=1,
        heads=3,
        max_cameras=2,
        max_tokens=400,
    )
    torch.manual_seed(19)
    expected = model.training_losses(world, "libero_gripper")
    torch.manual_seed(19)
    actual = model.training_losses(changed["libero_gripper"]["world"], "libero_gripper")
    torch.testing.assert_close(actual["action"], expected["action"], rtol=0, atol=0)
    torch.testing.assert_close(actual["total"], expected["total"], rtol=0, atol=0)

    def evaluate(value):
        prediction = {
            name: torch.zeros_like(stream["value"]) for name, stream in world.items()
        }
        evaluator = ImageProprioEval(objective="visual_state_action")
        evaluator.model = SimpleNamespace(
            device=torch.device("cpu"),
            global_step=0,
            forward_eval=lambda _: {
                "libero_gripper": {"image_proprio_prediction": prediction}
            },
        )
        evaluator.on_validation_start()
        evaluator.on_validation_step(value, 0)
        return evaluator.totals["val/libero_gripper/forecast/action_command_mse"]

    numerator, denominator = evaluate(batch)
    changed_numerator, changed_denominator = evaluate(changed)
    assert denominator == targets.sum() == changed_denominator
    torch.testing.assert_close(numerator, changed_numerator, rtol=0, atol=0)
    completion = PreparedPixelProprioDataset(
        path, **{**settings, "forecasting_probability": 0}
    )[0]
    assert completion["task_kind"] == "completion"
    assert completion["world"]["action"]["target_valid"].all()


def test_visual_only_discrete_observed_chunks_stop_before_first_unknown_image(tmp_path):
    path = pixel_manifest(tmp_path, include_sharpa=False, actions=False, frames=75)
    sample = PreparedPixelProprioDataset(
        path,
        split="val",
        mode="discrete",
        sampling_strategy="discrete_window",
        discrete_frames=16,
        stride=4,
        forecasting_probability=1,
    )[0]
    world, meta = sample["world"], sample["pixel_metadata"]
    assert set(world) == {"image", "proprio"}
    known_image = world["image"]["observed"].reshape(16, -1).all(1).numpy()
    known_pose = world["proprio"]["observed"].numpy()
    first_unknown_image = np.asarray(meta["frame_indices"])[~known_image][0]
    rows = np.asarray(meta["proprio_frame_indices"])
    assert (rows[known_pose] < first_unknown_image).all()
    assert not known_pose[~known_image].any()


def test_image_only_needs_no_pose_command_or_normalization_artifacts(
    tmp_path, monkeypatch
):
    path = pixel_manifest(tmp_path)
    manifest = json.loads(path.read_text())
    manifest_digest = digest(path)
    for ep in manifest["episodes"]:
        (path.parent / ep["path"]).unlink()
        if "action_provenance" in ep:
            (path.parent / ep["action_provenance"]["path"]).unlink()
    (path.parent / manifest["normalization"]["path"]).unlink()
    original = h5py.Group.__getitem__

    def guard(group, key):
        assert key in ("rgb", "time"), f"Image-only path opened {key}"
        return original(group, key)

    monkeypatch.setattr(h5py.Group, "__getitem__", guard)
    module = PixelProprioDataModule(
        path,
        image_only=True,
        validation_anchors_per_episode=2,
        checkpointable_sampling=True,
    )
    context = module.prepare_context(
        mode="train", normalization="prepared", normalizer=None
    )
    assert context.sample_schema == {"world": "image"}
    assert context.state["manifest"] == manifest
    assert context.state["manifest_sha256"] == manifest_digest
    assert context.state["sampling"]["image_only"] is True
    assert module.datasets["train"].normalization == {}
    assert list(module.val_dataloader().batch_sampler) == [[0], [5], [6], [11]]
    for batch in [next(iter(module.train_dataloader())), *module.val_dataloader()]:
        item = next(iter(batch.values()))
        assert set(item["world"]) == {"image"}
        assert torch.isfinite(item["world"]["image"]["value"]).all()
        assert not any(
            "proprio" in key or "pose" in key for key in item["pixel_metadata"][0]
        )
    module.prepare_context(
        mode="train",
        normalization="prepared",
        normalizer=None,
        restored_state=context.state,
    )


@pytest.mark.parametrize("discrete", [False, True])
@pytest.mark.parametrize("forecasting_probability", [0.0, 1.0])
def test_image_only_preserves_rgb_sampling_masks_and_relative_time(
    tmp_path,
    discrete,
    forecasting_probability,
):
    path = pixel_manifest(tmp_path, include_sharpa=False, actions=False, frames=75)
    settings = dict(forecasting_probability=forecasting_probability)
    if discrete:
        settings.update(
            mode="discrete",
            sampling_strategy="discrete_window",
            discrete_frames=16,
            stride=4,
        )
    else:
        # Two-frame completion exercises all-known/all-unknown image-mask repairs.
        settings.update(min_frames=2, max_frames=2)
    for split in ("train", "val"):
        legacy = PreparedPixelProprioDataset(path, split=split, **settings)
        image_only = PreparedPixelProprioDataset(
            path, split=split, image_only=True, **settings
        )
        for index in range(20):
            expected, actual = legacy[index], image_only[index]
            assert set(actual["world"]) == {"image"}
            assert actual["task_kind"] == expected["task_kind"]
            for field in expected["world"]["image"]:
                torch.testing.assert_close(
                    actual["world"]["image"][field],
                    expected["world"]["image"][field],
                    rtol=0,
                    atol=0,
                )
            for field in (
                "frame_indices",
                "timestamps_seconds",
                "time_origin_seconds",
                "camera_names",
            ):
                assert (
                    actual["pixel_metadata"][field] == expected["pixel_metadata"][field]
                )


@pytest.mark.parametrize("damage", ["checksum", "timestamps", "duplicate"])
def test_image_only_still_validates_rgb_identity_and_time(tmp_path, damage):
    path = pixel_manifest(tmp_path, include_sharpa=False)
    manifest = json.loads(path.read_text())
    ep = manifest["episodes"][1]
    if damage == "duplicate":
        ep["rgb_path"] = manifest["episodes"][0]["rgb_path"]
    else:
        rgb_path = path.parent / ep["rgb_path"]
        with h5py.File(rgb_path, "a") as data:
            if damage == "checksum":
                data["rgb"][0, 0, 0, 0, 0] += 1
            else:
                data["time"][1] = data["time"][0]
        if damage == "timestamps":
            ep["rgb_sha256"] = digest(rgb_path)
    save_json(path, manifest)
    with pytest.raises(ValueError):
        PreparedPixelProprioDataset(path, split="val", image_only=True)[0]


@pytest.mark.parametrize("task", ["direct_action", "visual_state_action"])
def test_image_only_rejects_label_requiring_tasks(tmp_path, task):
    path = pixel_manifest(tmp_path, include_sharpa=False)
    with pytest.raises(ValueError, match="image_only"):
        PreparedPixelProprioDataset(path, task=task, image_only=True)
    with pytest.raises(ValueError, match="image_only"):
        PixelProprioDataModule(path, task=task, image_only=True)


@pytest.mark.parametrize("batch_size", [1, 2])
def test_four_rank_training_is_partition_of_one_global_draw_and_resumes(
    tmp_path, batch_size
):
    path = pixel_manifest(tmp_path, actions=False, frames=20)
    dataset = PreparedPixelProprioDataset(path, image_only=True)
    global_sampler = HomogeneousWorldBatchSampler(
        dataset, batch_size=4 * batch_size, seed=42, checkpointable=True
    )
    global_draws = list(global_sampler)
    ranked = []
    for rank in range(4):
        sampler = DistributedPixelTrainingBatchSampler(
            dataset, batch_size, 42, world_size=4, rank=rank
        )
        ranked.append(list(sampler))
        assert len(sampler) == len(global_sampler)
        sampler.consumed_batches = 1
        assert list(sampler) == ranked[-1][1:]
    for step, expected in enumerate(global_draws):
        actual = [draw for rank in range(4) for draw in ranked[rank][step]]
        assert actual == expected
        # Distinct deterministic RNG slots; replacement may draw the same anchor.
        assert [draw[2] for draw in actual] == list(range(4 * batch_size))
        assert {draw[1] for draw in actual} == {step}
        assert (
            len(
                {
                    dataset.episodes[
                        int(np.searchsorted(dataset.ends, draw[0], side="right"))
                    ]["embodiment"]
                    for draw in actual
                }
            )
            == 1
        )


def _fifty_validation_episode_manifest(tmp_path):
    path = pixel_manifest(tmp_path, include_sharpa=False, actions=False)
    manifest = json.loads(path.read_text())
    template = next(ep for ep in manifest["episodes"] if ep["split"] == "val")
    original_rgb = tmp_path / template["rgb_path"]
    for index in range(1, 50):
        episode = copy.deepcopy(template)
        episode["episode_id"] = f"validation-episode-{index}"
        episode["rgb_path"] = f"validation-episode-{index}-rgb.h5"
        shutil.copyfile(original_rgb, tmp_path / episode["rgb_path"])
        manifest["episodes"].append(episode)
    return save_json(path, manifest)


def _assert_identical_pixel_item(expected, actual):
    assert actual["task_kind"] == expected["task_kind"]
    assert actual["pixel_metadata"] == expected["pixel_metadata"]
    assert set(actual["world"]) == set(expected["world"]) == {"image"}
    for field in actual["world"]["image"]:
        torch.testing.assert_close(
            actual["world"]["image"][field],
            expected["world"]["image"][field],
            rtol=0,
            atol=0,
        )


def test_four_rank_validation_exactly_covers_100_global_draws_and_preserves_masks(
    tmp_path,
):
    path = _fifty_validation_episode_manifest(tmp_path)
    kwargs = dict(
        manifest_path=path,
        image_only=True,
        checkpointable_sampling=True,
        validation_anchors_per_episode=2,
        distributed_world_size=4,
    )
    audit = PixelProprioDataModule(**kwargs)
    context = audit.prepare_context(
        mode="train", normalization="prepared", normalizer=None
    )
    assert context.state["distributed_sampling"]["global_batch_size"] == 4
    assert len(audit.val_dataloader()) == 100  # Unattached audit/reload is global.
    expected = [batch["libero_gripper"] for batch in audit.val_dataloader()]
    visited, episodes = [], []
    for rank in range(4):
        module = PixelProprioDataModule(**kwargs)
        module.trainer = SimpleNamespace(world_size=4, global_rank=rank)
        loader = module.val_dataloader()
        assert len(loader) == 25
        assert (
            module.prepare_context(
                mode="train", normalization="prepared", normalizer=None
            ).state
            == context.state
        )
        kinds = set()
        for batch in loader:
            item = batch["libero_gripper"]
            (global_index,) = item["validation_batch_index"]
            visited.append(global_index)
            episodes.append(item["pixel_metadata"][0]["episode_id"])
            kinds.update(item["task_kind"])
            _assert_identical_pixel_item(expected[global_index], item)
        if rank == 0:
            assert kinds == {"forecast", "completion"}
    assert visited == list(range(100))  # No drops, duplicates, or reordered seeds.
    assert len(set(episodes)) == 50
    assert all(episodes.count(episode) == 2 for episode in set(episodes))


def test_four_rank_checkpoint_cursor_restores_each_local_draw_and_binds_topology(
    tmp_path,
):
    path = pixel_manifest(tmp_path, actions=False, frames=20)
    kwargs = dict(
        manifest_path=path,
        image_only=True,
        checkpointable_sampling=True,
        validation_anchors_per_episode=2,
        distributed_world_size=4,
    )
    states = []
    for rank in range(4):
        data = PixelProprioDataModule(**kwargs)
        data.trainer = SimpleNamespace(world_size=4, global_rank=rank)
        consumed = next(iter(data.train_dataloader()))
        item = next(iter(consumed.values()))
        assert item["sampling_cursor"] == 1
        data.acknowledge_batch(item["sampling_cursor"])
        state = data.state_dict()
        states.append(state)
        expected = next(iter(data.train_dataloader()))
        restored = PixelProprioDataModule(**kwargs)
        restored.load_state_dict(state)
        restored.trainer = SimpleNamespace(world_size=4, global_rank=rank)
        actual = next(iter(restored.train_dataloader()))
        assert actual.keys() == expected.keys()
        key = next(iter(actual))
        _assert_identical_pixel_item(expected[key], actual[key])
        assert actual[key]["sampling_cursor"] == 2
        with pytest.raises(ValueError, match="Nonconsecutive"):
            restored.acknowledge_batch(3)
    assert all(state == states[0] for state in states)  # One global cursor.
    single = PixelProprioDataModule(**{**kwargs, "distributed_world_size": 1})
    with pytest.raises(ValueError, match="distributed topology"):
        single.load_state_dict(states[0])
    with pytest.raises(ValueError, match="distributed topology"):
        PixelProprioDataModule(**kwargs).load_state_dict(single.state_dict())
    context = PixelProprioDataModule(**kwargs).prepare_context(
        mode="train", normalization="prepared", normalizer=None
    )
    with pytest.raises(ValueError, match="context mismatch"):
        single.prepare_context(
            mode="train",
            normalization="prepared",
            normalizer=None,
            restored_state=context.state,
        )


def test_distributed_pixel_loading_rejects_ambiguous_or_lossy_topologies(tmp_path):
    path = pixel_manifest(tmp_path, actions=False)
    with pytest.raises(ValueError, match="checkpointable_sampling"):
        PixelProprioDataModule(path, distributed_world_size=4)
    with pytest.raises(ValueError, match="positive integer"):
        PixelProprioDataModule(path, distributed_world_size=True)
    data = PixelProprioDataModule(
        path, image_only=True, checkpointable_sampling=True, distributed_world_size=4
    )
    data.trainer = SimpleNamespace(world_size=1, global_rank=0)
    with pytest.raises(ValueError, match="world_size"):
        data.train_dataloader()
    dataset = PreparedPixelProprioDataset(path, split="val", image_only=True)
    base = EvenlySpacedEpisodeBatchSampler(dataset, 2)
    with pytest.raises(ValueError, match="divisible"):
        DistributedPixelValidationBatchSampler(base, world_size=3, rank=0)
    with pytest.raises(ValueError, match="topology"):
        DistributedPixelTrainingBatchSampler(dataset, 1, 42, world_size=4, rank=4)


def test_validation_global_batch_index_is_shared_and_preserved_when_collating(tmp_path):
    path = pixel_manifest(tmp_path, include_sharpa=False, actions=False, frames=12)
    data = PreparedPixelProprioDataset(path, split="val", image_only=True)
    base = EvenlySpacedEpisodeBatchSampler(data, 8, batch_size=2)
    shards = [
        DistributedPixelValidationBatchSampler(base, world_size=2, rank=rank)
        for rank in range(2)
    ]
    for rank, shard in enumerate(shards):
        for offset, indices in enumerate(shard):
            item = collate_pixel_proprio([data[index] for index in indices])[
                "libero_gripper"
            ]
            assert item["validation_batch_index"] == [rank * 2 + offset] * 2
    sample = data[0]
    sample["validation_batch_index"] = 0
    with pytest.raises(ValueError, match="global batch index"):
        collate_pixel_proprio([sample, data[1]])
