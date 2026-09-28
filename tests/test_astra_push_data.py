"""Temporal alignment, privileged-state firewall, and resumed replay contracts."""

import numpy as np
import pytest
import torch

from egomimic.experiments.astra_push.data import (
    EpisodeBalancedReplay,
    EpisodeWindows,
    ProprioceptionStats,
    collate_windows,
    write_episode,
)


def episode(tmp_path, name, *, phase="seed", arm="common", round_index=0, count=3):
    observations = {
        "external_rgb": np.full((count, 224, 224, 3), 255, np.uint8),
        "wrist_rgb": np.zeros((count, 224, 224, 3), np.uint8),
        "proprioception": np.arange(count * 9, dtype=np.float32).reshape(count, 9),
        "timestamps": np.arange(count, dtype=np.float64) / 10,
    }
    return write_episode(
        tmp_path / f"{name}.hdf5",
        observations=observations,
        actions=np.repeat(np.arange(count, dtype=np.float32)[:, None] / 10, 7, axis=1),
        instruction="Push the block into the target.",
        provenance={
            "episode_id": name,
            "scene_hash": "1" * 64,
            "task_hash": "2" * 64,
            "teacher_hash": "3" * 64,
            "initial_state_hash": "4" * 64,
            "phase": phase,
            "arm": arm,
            "round": round_index,
            "stage": "S1",
            "partition": "training",
            "accepted": True,
        },
        audit={"privileged_object_pose": [0.1, 0.2, 0.3], "teacher_internal": "secret"},
    )


def norm():
    return ProprioceptionStats(np.zeros(9), np.ones(9), [])


def test_observation_t_targets_action_t_and_never_next_episode(tmp_path):
    records = [episode(tmp_path, name) for name in ("one", "two")]
    dataset = EpisodeWindows(records, norm())
    row = dataset[2]
    assert row["proprioception"][0] == 18
    np.testing.assert_allclose(row["actions"][0], 0.2)
    assert row["action_valid"].sum() == 1
    assert not row["actions"][1:].any()
    assert dataset[3]["action_valid"].sum() == 3
    batch = collate_windows([row, dataset[0]])["libero_push"]
    assert batch["observations.images.external_rgb"].shape == (2, 1, 1, 3, 224, 224)
    assert torch.all(batch["observations.images.external_rgb"] == 1)
    assert torch.all(batch["observations.images.wrist_rgb"] == -1)
    assert batch["action_valid"].dtype == torch.bool
    assert not any("secret" in str(v) or "privileged" in k for k, v in batch.items())
    with pytest.raises(ValueError, match="privileged"):
        collate_windows([{**row, "object_pose": np.zeros(3)}])


def test_commissioning_and_tampered_episodes_cannot_enter_replay(tmp_path):
    record = episode(tmp_path, "commissioning", phase="commissioning")
    with pytest.raises(ValueError, match="imitation"):
        EpisodeWindows([record], norm())
    record = episode(tmp_path, "seed")
    record["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash"):
        EpisodeWindows([record], norm())


def test_two_current_two_history_and_exact_sampler_resume(tmp_path):
    records = [episode(tmp_path, "seed")]
    records.extend(
        episode(tmp_path, f"round-{r}", phase="round", arm="A", round_index=r)
        for r in (1, 2)
    )
    dataset = EpisodeWindows(records, norm())
    sampler = EpisodeBalancedReplay(dataset, seed=17, arm="A", round_index=2)
    for _ in range(10):
        indices = sampler.next_indices()
        assert all(index >= 6 for index in indices[:2])
        assert all(index < 6 for index in indices[2:])
    saved = sampler.state_dict()
    expected = [sampler.next_indices() for _ in range(20)]
    restored = EpisodeBalancedReplay(dataset, seed=123, arm="A", round_index=2)
    restored.load_state_dict(saved)
    assert [restored.next_indices() for _ in range(20)] == expected
    with pytest.raises(ValueError, match="another arm"):
        EpisodeBalancedReplay(dataset, seed=17, arm="U", round_index=2)


def test_published_episode_cannot_be_overwritten(tmp_path):
    original = episode(tmp_path, "same")
    with pytest.raises(FileExistsError):
        episode(tmp_path, "same", count=2)
    assert EpisodeWindows([original], norm()).lengths == [3]
