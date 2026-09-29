"""Decoded replay must preserve the benchmark while sharing physical data."""

import multiprocessing
import pickle
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pytest
import torch
import zarr
from torch.utils.data import DataLoader

from egomimic.rldb.zarr.decoded_replay import prepare_decoded_replay
from egomimic.rldb.zarr.libero_dataset import (
    LiberoDataset,
    LiberoNormalizer,
    LiberoReplayResolver,
    keymap,
)
from tests.test_libero_benchmark import make_replay


def build_dataset(path, *, cached, mode="train", n_obs_steps=2, horizon=32):
    resolver = LiberoReplayResolver(
        path, keymap(n_obs_steps, horizon), "libero10", decoded_cache=cached
    )
    dataset = LiberoDataset._from_resolver(resolver, mode=mode)
    normalizer = LiberoNormalizer()
    normalizer.populate_from_datasets({"libero_panda": dataset})
    normalizer.infer_shapes_from_batch(dataset[0])
    normalizer.infer_norm_from_dataset(dataset, "libero_panda")
    dataset.set_norm_stats_from(normalizer)
    return dataset, normalizer


def assert_sample_equal(actual, expected):
    assert actual.keys() == expected.keys()
    for key, value in expected.items():
        if torch.is_tensor(value):
            torch.testing.assert_close(actual[key], value, rtol=0, atol=0)
        else:
            assert actual[key] == value


@pytest.mark.parametrize("n_obs_steps", [0, 2, 8])
def test_all_frames_padding_normalization_and_batch_parity(tmp_path, n_obs_steps):
    path = tmp_path / "replay.zarr"
    make_replay(path)
    reference, old_norm = build_dataset(path, cached=False, n_obs_steps=n_obs_steps)
    cached, new_norm = build_dataset(path, cached=True, n_obs_steps=n_obs_steps)
    assert old_norm.context == new_norm.context
    assert old_norm.tokenizer_context() == new_norm.tokenizer_context()
    assert reference.index_map == cached.index_map
    for i in range(len(reference)):
        assert_sample_equal(cached[i], reference[i])
    for actual, expected in zip(
        DataLoader(cached, batch_size=7), DataLoader(reference, batch_size=7)
    ):
        assert_sample_equal(actual, expected)
    valid, _ = build_dataset(path, cached=True, mode="valid", n_obs_steps=n_obs_steps)
    assert not set(valid.datasets) & set(cached.datasets)
    arrays = cached.resolver.open_arrays()
    assert all(
        isinstance(a, np.memmap) and not a.flags.writeable for a in arrays.values()
    )
    leaf = next(iter(cached.datasets.values()))
    original = leaf[0]
    changed = leaf[0]
    changed["actions"].zero_()
    assert_sample_equal(leaf[0], original)
    with pytest.raises(IndexError):
        leaf[len(leaf)]


def _prepare_worker(path):
    cache = prepare_decoded_replay(path, ["action", "agentview_rgb"])
    return str(cache.directory), cache.open()["action"].copy()


def test_concurrent_ranks_publish_one_cache_and_workers_reopen_it(tmp_path):
    path = tmp_path / "replay.zarr"
    expected = make_replay(path)
    with ProcessPoolExecutor(
        2, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        results = list(pool.map(_prepare_worker, [str(path)] * 2))
    assert results[0][0] == results[1][0]
    for _, actions in results:
        np.testing.assert_array_equal(actions, expected["action"])
    dataset, _ = build_dataset(path, cached=True)
    encoded = pickle.dumps(dataset)
    assert len(encoded) < 100_000  # Never serialize decoded images into each worker.
    restored = pickle.loads(encoded)
    assert restored.resolver._decoded._arrays is None
    for actual, expected_batch in zip(
        DataLoader(
            restored, batch_size=9, num_workers=2, multiprocessing_context="spawn"
        ),
        DataLoader(dataset, batch_size=9),
    ):
        assert_sample_equal(actual, expected_batch)


def test_source_changes_cannot_reuse_old_decoded_values(tmp_path):
    path = tmp_path / "replay.zarr"
    arrays = make_replay(path)
    old = prepare_decoded_replay(path, ["action"])
    old_values = old.open()["action"].copy()
    data = zarr.open_group(str(path), mode="r+")["data"]
    data["action"][0] = arrays["action"][0] + 0.1
    new = prepare_decoded_replay(path, ["action"])
    assert new.directory != old.directory
    np.testing.assert_array_equal(new.open()["action"][0], data["action"][0])
    np.testing.assert_array_equal(old.open()["action"], old_values)


def test_failed_build_is_not_published_and_truncated_cache_is_rejected(tmp_path):
    path = tmp_path / "replay.zarr"
    arrays = make_replay(path)
    data = zarr.open_group(str(path), mode="r+")["data"]
    data["action"][0, 0] = float("nan")
    with pytest.raises(ValueError, match="Non-finite"):
        prepare_decoded_replay(path, ["action"])
    root = path.with_name(path.name + ".decoded")
    assert not list(root.glob("*/manifest.json"))
    assert not list(root.glob(".building-*"))
    data["action"][:] = arrays["action"]
    cache = prepare_decoded_replay(path, ["action"])
    target = cache.directory / cache.manifest["arrays"]["action"]["file"]
    with target.open("r+b") as handle:
        handle.truncate(100)
    with pytest.raises(ValueError, match="Truncated"):
        prepare_decoded_replay(path, ["action"])


def test_cache_location_must_not_pollute_source_identity(tmp_path):
    path = tmp_path / "replay.zarr"
    make_replay(path)
    with pytest.raises(ValueError, match="outside"):
        prepare_decoded_replay(path, ["action"], cache_root=path / "cache")
