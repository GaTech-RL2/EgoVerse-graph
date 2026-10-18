"""Prefetch-safe exact draws, independent validation batching and rank work balance."""

import copy
import json
import os
import sys
from itertools import islice
from types import SimpleNamespace

import pytest
import torch

from egomimic.rldb.pixel_proprio import (
    DistributedPixelTrainingBatchSampler,
    PreparedPixelProprioDataset,
    balance_global_pixel_draws,
)
from egomimic.rldb.pixel_proprio_datamodule import PixelProprioDataModule
from egomimic.rldb.world_model import HomogeneousWorldBatchSampler
from tests.test_pixel_attention_profiles import recipe
from tests.test_pixel_proprio_data import (
    _assert_identical_pixel_item,
    _fifty_validation_episode_manifest,
    pixel_manifest,
)


def close_iterator(iterator):
    # Tests intentionally abandon prefetched samples, as a stopped process does.
    if hasattr(iterator, "_shutdown_workers"):
        iterator._shutdown_workers()


def _protect_spawn_tracker_fd(worker_id):
    """Keep macOS Torch's exec'd shared-memory manager from retaining this pipe.

    Python spawn passes its tracker FD as inheritable. Torch's file_system
    transport can inherit it again into a manager that outlives the worker,
    creating a tracker/manager wait cycle during interpreter shutdown. This
    test-only process hygiene does not change dataset sampling or IPC tensors.
    Linux's default fork loader does not need this workaround.
    """
    del worker_id
    if sys.platform == "darwin":
        from multiprocessing import resource_tracker

        os.set_inheritable(resource_tracker.getfd(), False)


@pytest.mark.parametrize("balance", [False, True])
def test_b2_two_worker_prefetch_resume_replays_only_unconsumed_draws(tmp_path, balance):
    path = pixel_manifest(tmp_path, include_sharpa=False, actions=False, frames=80)
    manifest = json.loads(path.read_text())
    for episode in manifest["episodes"]:
        (path.parent / episode["path"]).unlink()
    (path.parent / manifest["normalization"]["path"]).unlink()
    kwargs = dict(
        manifest_path=path,
        image_only=True,
        checkpointable_sampling=True,
        batch_size=2,
        validation_batch_size=1,
        distributed_world_size=4,
        validation_anchors_per_episode=2,
        num_workers=2,
        prefetch_factor=3,
        persistent_workers=True,
        balance_training_work=balance,
    )
    module = PixelProprioDataModule(**kwargs)
    module.trainer = SimpleNamespace(world_size=4, global_rank=1)
    loader = module.train_dataloader()
    loader.multiprocessing_context = "spawn"
    loader.worker_init_fn = _protect_spawn_tracker_fd
    rng = torch.get_rng_state().clone()
    iterator = iter(loader)
    try:
        for cursor in (1, 2):
            item = next(iterator)["libero_gripper"]
            assert item["sampling_cursor"] == cursor
            module.acknowledge_batch(cursor)
        assert iterator._send_idx > 2  # Worker queue has advanced past consumed data.
        saved = copy.deepcopy(module.state_dict())
        assert saved["consumed_batches"] == 2
        expected = [next(iterator)["libero_gripper"] for _ in range(3)]
        # Iterator construction/prefetch must not consume model Torch RNG.
        assert torch.equal(torch.get_rng_state(), rng)
    finally:
        close_iterator(iterator)
    for workers in (2, 0):
        restored = PixelProprioDataModule(
            **{
                **kwargs,
                "num_workers": workers,
                "prefetch_factor": 2 if workers else None,
                "persistent_workers": bool(workers),
            }
        )
        restored.load_state_dict(saved)
        restored.trainer = SimpleNamespace(world_size=4, global_rank=1)
        loader = restored.train_dataloader()
        if workers:
            loader.multiprocessing_context = "spawn"
            loader.worker_init_fn = _protect_spawn_tracker_fd
        iterator = iter(loader)
        try:
            for cursor, want in enumerate(expected, start=3):
                actual = next(iterator)["libero_gripper"]
                assert actual["sampling_cursor"] == cursor
                _assert_identical_pixel_item(want, actual)
                restored.acknowledge_batch(cursor)
        finally:
            close_iterator(iterator)
        assert restored.consumed_batches == 5


def test_train_b2_prefetch_keeps_all_100_validation_anchors_once_across_four_ranks(
    tmp_path,
):
    path = _fifty_validation_episode_manifest(tmp_path)
    baseline = PixelProprioDataModule(
        path,
        image_only=True,
        checkpointable_sampling=True,
        validation_anchors_per_episode=2,
    )
    expected = [batch["libero_gripper"] for batch in baseline.val_dataloader()]
    visited, episodes = [], []
    for rank in range(4):
        module = PixelProprioDataModule(
            path,
            image_only=True,
            checkpointable_sampling=True,
            batch_size=2,
            validation_batch_size=1,
            distributed_world_size=4,
            validation_anchors_per_episode=2,
            num_workers=2,
            prefetch_factor=2,
            persistent_workers=True,
            balance_training_work=True,
        )
        assert len(module.val_dataloader()) == 100  # Unattached inspection is global.
        module.trainer = SimpleNamespace(world_size=4, global_rank=rank)
        loader = module.val_dataloader()
        loader.worker_init_fn = _protect_spawn_tracker_fd
        assert len(loader) == 25
        iterator = iter(loader)
        try:
            for batch in iterator:
                item = batch["libero_gripper"]
                (index,) = item["validation_batch_index"]
                visited.append(index)
                episodes.append(item["pixel_metadata"][0]["episode_id"])
                _assert_identical_pixel_item(expected[index], item)
        finally:
            close_iterator(iterator)
    assert visited == list(range(100))
    assert len(set(episodes)) == 50
    assert all(episodes.count(episode) == 2 for episode in set(episodes))


def test_loader_execution_options_preserve_legacy_context_but_explicit_batches_bind(
    tmp_path,
):
    path = pixel_manifest(tmp_path, actions=False)
    kwargs = dict(manifest_path=path, image_only=True, checkpointable_sampling=True)
    legacy = PixelProprioDataModule(**kwargs)
    context = legacy.prepare_context(
        mode="train", normalization="prepared", normalizer=None
    )
    state = legacy.state_dict()
    assert (
        "batching" not in context.state and "training_work_balance" not in context.state
    )
    changed_runtime = PixelProprioDataModule(
        **kwargs,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=3,
    )
    changed_runtime.load_state_dict(state)
    assert changed_runtime.state_dict() == state
    assert (
        changed_runtime.prepare_context(
            mode="train",
            normalization="prepared",
            normalizer=None,
            restored_state=context.state,
        ).state
        == context.state
    )
    loader = changed_runtime.train_dataloader()
    assert loader.num_workers == 2 and loader.prefetch_factor == 3
    assert loader.pin_memory and loader.persistent_workers
    explicit = PixelProprioDataModule(**kwargs, batch_size=2, validation_batch_size=1)
    explicit_context = explicit.prepare_context(
        mode="train", normalization="prepared", normalizer=None
    )
    assert explicit_context.state["batching"] == {
        "schema": "pixel-split-batching-v1",
        "train_batch_size": 2,
        "validation_batch_size": 1,
    }
    for invalid in (
        PixelProprioDataModule(**kwargs, batch_size=2),
        PixelProprioDataModule(**kwargs, batch_size=2, validation_batch_size=2),
    ):
        with pytest.raises(ValueError, match="batching"):
            invalid.load_state_dict(explicit.state_dict())
        with pytest.raises(ValueError, match="context mismatch"):
            invalid.prepare_context(
                mode="train",
                normalization="prepared",
                normalizer=None,
                restored_state=explicit_context.state,
            )


@pytest.mark.parametrize(
    "options",
    [
        {"batch_size": False},
        {"batch_size": 0},
        {"validation_batch_size": 0},
        {"validation_batch_size": True},
        {"num_workers": -1},
        {"num_workers": True},
        {"persistent_workers": True},
        {"prefetch_factor": 2},
        {"num_workers": 2, "prefetch_factor": 0},
        {"num_workers": 2, "prefetch_factor": True},
        {"pin_memory": "yes"},
        {"persistent_workers": 1},
        {"balance_training_work": 1},
        {"balance_training_work": True},
    ],
)
def test_loader_options_fail_closed(options):
    with pytest.raises(ValueError):
        PixelProprioDataModule("unused.json", **options)


def test_longest_first_fixed_capacity_preserves_draws_and_reduces_worst_work():
    costs = [100, 81, 64, 49, 16, 9, 4, 1]
    source = SimpleNamespace(predicted_token_work=lambda index, draw: costs[index])
    draws = [(i, 7, i) for i in range(8)]
    balanced = balance_global_pixel_draws(source, draws, 2, 4)
    assert all(len(rank) == 2 for rank in balanced)
    assert sorted(draw for rank in balanced for draw in rank) == draws
    assert max(sum(costs[draw[0]] for draw in rank) for rank in balanced) == 101
    assert (
        max(sum(costs[draw[0]] for draw in draws[i : i + 2]) for i in range(0, 8, 2))
        == 181
    )
    assert balance_global_pixel_draws(source, draws, 2, 4) == balanced
    assert balance_global_pixel_draws(source, draws[:4], 1, 4) == [
        [draw] for draw in draws[:4]
    ]
    equal = SimpleNamespace(predicted_token_work=lambda index, draw: 256)
    assert balance_global_pixel_draws(equal, draws, 2, 4) == [
        draws[i : i + 2] for i in range(0, 8, 2)
    ]
    with pytest.raises(ValueError, match="complete global batch"):
        balance_global_pixel_draws(source, draws[:-1], 2, 4)


@pytest.mark.parametrize("mode", ["continuous", "discrete"])
def test_work_preview_shares_exact_frame_rng_and_balancing_preserves_global_examples(
    tmp_path, mode
):
    path = pixel_manifest(tmp_path, include_sharpa=False, actions=False, frames=80)
    data = PreparedPixelProprioDataset(
        path,
        image_only=True,
        mode=mode,
        discrete_frames=16,
        stride=4,
        sampling_strategy="full_episode_uniform"
        if mode == "continuous"
        else "discrete_window",
    )
    global_draws = list(
        islice(HomogeneousWorldBatchSampler(data, 8, 42, checkpointable=True), 8)
    )
    assert data.rng is None
    work = {
        draw: data.predicted_token_work(draw[0], draw[1:])
        for batch in global_draws
        for draw in batch
    }
    assert data.rng is None  # Preview does not read pixels or initialize worker RNG.
    ranked = []
    for rank in range(4):
        sampler = DistributedPixelTrainingBatchSampler(
            data, 2, 42, world_size=4, rank=rank, balance_training_work=True
        )
        ranked.append(list(islice(sampler, 8)))
        sampler.consumed_batches = 3
        assert list(islice(sampler, 5)) == ranked[-1][3:]
    reassigned = False
    for step, original in enumerate(global_draws):
        combined = [draw for rank in range(4) for draw in ranked[rank][step]]
        assert sorted(combined) == sorted(original)
        reassigned |= combined != original
        max_after = max(
            sum(work[draw] for draw in ranked[rank][step]) for rank in range(4)
        )
        max_before = max(
            sum(work[draw] for draw in original[i : i + 2]) for i in range(0, 8, 2)
        )
        assert max_after <= max_before
        for draw in original:
            expected, actual = (
                data[draw],
                data[next(key for key in combined if key == draw)],
            )
            assert work[draw] == len(actual["world"]["image"]["value"]) ** 2
            assert actual["pixel_metadata"] == expected["pixel_metadata"]
            for field in actual["world"]["image"]:
                assert torch.equal(
                    actual["world"]["image"][field], expected["world"]["image"][field]
                )
    assert reassigned is (mode == "continuous")


def test_work_assignment_choice_is_context_and_resume_bound(tmp_path):
    path = pixel_manifest(tmp_path, include_sharpa=False, actions=False)
    kwargs = dict(
        manifest_path=path,
        image_only=True,
        checkpointable_sampling=True,
        batch_size=2,
        validation_batch_size=1,
        distributed_world_size=4,
    )
    balanced = PixelProprioDataModule(**kwargs, balance_training_work=True)
    state = balanced.state_dict()
    context = balanced.prepare_context(
        mode="train", normalization="prepared", normalizer=None
    )
    assert "rank_seeded" in context.state["training_work_balance"]["model_noise"]
    unbalanced = PixelProprioDataModule(**kwargs)
    with pytest.raises(ValueError, match="work balance"):
        unbalanced.load_state_dict(state)
    with pytest.raises(ValueError, match="context mismatch"):
        unbalanced.prepare_context(
            mode="train",
            normalization="prepared",
            normalizer=None,
            restored_state=context.state,
        )


@pytest.mark.parametrize("arm", ["discrete", "continuous"])
def test_prefetch_execution_profile_keeps_validation_batch_fixed(arm):
    cfg = recipe(arm, execution="ddp4_prefetch")
    assert cfg.data.batch_size == cfg.data.validation_batch_size == 1
    assert cfg.data.distributed_world_size == 4
    assert cfg.data.num_workers == cfg.data.prefetch_factor == 2
    assert cfg.data.pin_memory and cfg.data.persistent_workers
    assert cfg.data.balance_training_work and cfg.data.checkpointable_sampling
    assert not cfg.trainer.use_distributed_sampler
