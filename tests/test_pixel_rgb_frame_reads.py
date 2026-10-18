"""Selected HDF5 reads preserve complete samples and deterministic draw state."""

import json

import h5py
import numpy as np
import pytest
import torch

import egomimic.rldb.pixel_proprio as pixels
from tests.test_pixel_proprio_data import digest, pixel_manifest


@pytest.mark.parametrize("indices", [[1, 4, 9], [9, 1, 4, 1], [], [0]])
def test_frame_chunk_reads_keep_exact_order_bytes_and_two_cameras(
    tmp_path, monkeypatch, indices
):
    rgb = np.random.default_rng(2).integers(
        0, 256, size=(10, 2, 160, 160, 3), dtype=np.uint8
    )
    path = tmp_path / "rgb.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset(
            "rgb", data=rgb, chunks=(1, 1, 160, 160, 3), compression="lzf"
        )
    getitem = h5py.Dataset.__getitem__
    reads = []

    def basic_reads_only(dataset, key, *args, **kwargs):
        assert isinstance(
            key, int
        ), "RGB must not use HDF5 fancy selection or load all frames"
        reads.append(key)
        return getitem(dataset, key, *args, **kwargs)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", basic_reads_only)
    indices = np.asarray(indices, dtype=np.int64)
    with h5py.File(path, "r") as handle:
        actual = pixels.read_selected_rgb_frames(handle["rgb"], indices)
    expected = rgb[indices]
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    assert actual.tobytes() == expected.tobytes()
    assert reads == indices.tolist()


def test_npz_members_keep_numpy_indexing_behavior(tmp_path):
    rgb = np.arange(12 * 2 * 4 * 4 * 3, dtype=np.uint8).reshape(12, 2, 4, 4, 3)
    path = tmp_path / "rgb.npz"
    np.savez_compressed(path, rgb=rgb)
    with np.load(path, allow_pickle=False) as handle:
        for indices in (np.array([9, 2, 9, -1]), np.array([], dtype=int)):
            actual = pixels.read_selected_rgb_frames(handle["rgb"], indices)
            assert actual.dtype == rgb.dtype
            assert actual.tobytes() == rgb[indices].tobytes()


def _equal_tree(actual, expected):
    if torch.is_tensor(expected):
        assert actual.dtype == expected.dtype and torch.equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _equal_tree(actual[key], expected[key])
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected):
            _equal_tree(a, e)
    else:
        assert actual == expected


@pytest.mark.parametrize("storage", ["hdf5", "npz"])
@pytest.mark.parametrize("split", ["train", "val"])
@pytest.mark.parametrize(
    "task,image_only,mode",
    [
        ("visual_diffusion", True, "continuous"),
        ("visual_diffusion", True, "discrete"),
        ("visual_diffusion", False, "continuous"),
        ("visual_diffusion", False, "discrete"),
        ("visual_state_action", False, "continuous"),
        ("visual_state_action", False, "discrete"),
        ("direct_action", False, "continuous"),
    ],
)
def test_full_samples_match_previous_fancy_read_and_rng(
    tmp_path, monkeypatch, storage, mode, split, task, image_only
):
    path = pixel_manifest(tmp_path, include_sharpa=False, frames=75)
    manifest = json.loads(path.read_text())
    for episode in manifest["episodes"]:
        original = path.parent / episode["rgb_path"]
        with h5py.File(original, "r") as handle:
            rgb, time = handle["rgb"][:], handle["time"][:]
        if storage == "npz":
            replacement = original.with_suffix(".npz")
            np.savez_compressed(replacement, rgb=rgb, time=time)
        else:
            replacement = original
            with h5py.File(replacement, "w") as handle:
                handle.create_dataset(
                    "rgb", data=rgb, chunks=(1, 1, 4, 4, 3), compression="lzf"
                )
                handle["time"] = time
        episode["rgb_path"], episode["rgb_sha256"] = (
            replacement.name,
            digest(replacement),
        )
    path.write_text(json.dumps(manifest))
    options = dict(
        split=split,
        task=task,
        image_only=image_only,
        mode=mode,
        min_frames=2,
        max_frames=64,
        discrete_frames=16,
        stride=4,
        sampling_strategy="discrete_window"
        if mode == "discrete"
        else "full_episode_uniform",
    )
    actual, previous = (
        pixels.PreparedPixelProprioDataset(path, **options) for _ in range(2)
    )
    for dataset in (actual, previous):
        dataset.rng = np.random.default_rng(937)
        dataset.worker_seed = torch.initial_seed()
    # Both ordinary worker RNG draws and checkpointed (cursor, global slot)
    # draws must preserve every image, label, time, camera, mask and metadata.
    indices = [0, 37, 74]
    if split == "train":
        indices += [(0, 7, 0), (37, 7, 9), (74, 7, 31)]
    actual_samples = [actual[index] for index in indices]
    with monkeypatch.context() as context:
        context.setattr(pixels, "read_selected_rgb_frames", lambda rgb, rows: rgb[rows])
        previous_samples = [previous[index] for index in indices]
    _equal_tree(actual_samples, previous_samples)
    for start in range(0, len(indices), 3):
        _equal_tree(
            pixels.collate_pixel_proprio(actual_samples[start : start + 3]),
            pixels.collate_pixel_proprio(previous_samples[start : start + 3]),
        )
    assert actual.rng.bit_generator.state == previous.rng.bit_generator.state
