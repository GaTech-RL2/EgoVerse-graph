"""Native-time TD alignment for matched raw and compressed action prefixes."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from egomimic.pipeline.algo import PipelineAlgo
from egomimic.pipeline.stages_q_chunking import QChunkingStage
from egomimic.rldb.action_codec import ControlChunkCodec
from egomimic.rldb.goal_replay import GoalReplay, goal_window_targets


def replay_batch():
    data = {
        "observations": np.arange(40, dtype=np.float32).reshape(20, 2),
        "actions": np.full((20, 1), .25, dtype=np.float32),
        "terminals": np.array([0] * 9 + [1] + [0] * 9 + [1]),
    }
    replay = GoalReplay(data, backup_horizon=4, discount=.9, include_future_observations=True)
    batch = replay.sample(7, np.random.RandomState(0),
                          indices=[1] * 7, goal_indices=[1, 2, 3, 4, 0, 8, 9])
    return replay, batch


def test_goal_boundary_rewards_endpoints_and_native_discount():
    replay, batch = replay_batch()
    # Current, inside, exactly at boundary, beyond, behind; then longer windows.
    lengths = torch.tensor([2, 2, 2, 2, 2, 3, 4])
    targets = goal_window_targets(batch, lengths, .9)
    torch.testing.assert_close(targets["high_value_backup_horizon"],
                               torch.tensor([0., 1., 2., 2., 2., 3., 4.]))
    torch.testing.assert_close(targets["high_value_rewards"], torch.tensor([1., .9, 0., 0., 0., 0., 0.]))
    torch.testing.assert_close(targets["high_value_masks"], torch.tensor([0., 0., 1., 1., 1., 1., 1.]))
    torch.testing.assert_close(targets["high_value_next_observations"],
        torch.from_numpy(replay.data["observations"][[1, 2, 3, 3, 3, 4, 5]]))
    backup = (targets["high_value_rewards"] + .9 ** targets["high_value_backup_horizon"]
              * targets["high_value_masks"] * .5)
    torch.testing.assert_close(backup, torch.tensor([1., .9, .405, .405, .405, .3645, .32805]))
    # Exact endpoint goal bootstraps, following upstream's half-open STATE reward
    # interval. It is not silently shifted to a transition-arrival reward.
    assert targets["high_value_masks"][2] == 1


def test_opt_in_preserves_fixed_replay_and_random_draws():
    replay, _ = replay_batch()
    fixed = GoalReplay(replay.data, backup_horizon=4, discount=.9)
    first_rng, second_rng = np.random.RandomState(17), np.random.RandomState(17)
    first, second = fixed.sample(128, first_rng), replay.sample(128, second_rng)
    for name, expected in first.items():
        torch.testing.assert_close(second[name], expected, rtol=0, atol=0)
    np.testing.assert_array_equal(first_rng.randint(100, size=20), second_rng.randint(100, size=20))
    assert second["high_value_goal_offset"].dtype == torch.int64
    assert second["high_value_observation_trajectory"].shape == (128, 5, 2)
    # Taking the complete native window reproduces ALL upstream target fields.
    targets = goal_window_targets(second, torch.full((128,), 4), .9)
    for key, actual in targets.items():
        torch.testing.assert_close(actual, first[key], rtol=1e-6, atol=1e-7)
    with pytest.raises(ValueError, match="boundary"):
        replay.sample(1, first_rng, indices=[7], goal_indices=[8])


def test_paired_codecs_share_targets_with_variable_duration_and_holds():
    _, batch = replay_batch()
    native = torch.tensor([[[.9]] * 4, [[.25]] * 4, [[0.]] * 4])
    batch = {key: value[:3] for key, value in batch.items()}
    batch["high_value_action_chunks"] = native
    batch["high_value_goal_offset"] = torch.full((3,), 8)
    results = []
    for kind in ["native_window", "arc"]:
        codec = ControlChunkCodec(1, 4, kind=kind, waypoints=3, distance=.5)
        tokens, lengths = codec.encode(native)
        decoded, decoded_lengths = codec.decode(tokens)
        assert lengths.tolist() == [1, 2, 4]  # holds still consume native time
        torch.testing.assert_close(lengths, decoded_lengths, rtol=0, atol=0)
        for row, length in enumerate(lengths):
            torch.testing.assert_close(decoded[row, :length], native[row, :length])
        results.append(goal_window_targets(batch, lengths, .9))
    for name in results[0]:
        torch.testing.assert_close(results[0][name], results[1][name], rtol=0, atol=0)
    assert results[0]["high_value_backup_horizon"].tolist() == [1., 2., 4.]


@pytest.mark.parametrize("kind", ["native_window", "arc"])
def test_stage_direct_td_ignores_old_teacher_targets_and_logs_durations(kind):
    _, batch = replay_batch()
    codec = ControlChunkCodec(1, 4, kind=kind, waypoints=3, distance=.5)
    stage = QChunkingStage(codec, backup_mode="policy_window", use_chunk_critic=False,
        observation_dim=2, goal_dim=2, action_dim=1, backup_horizon=4,
        discount=.9, hidden_dims=(8,), flow_steps=2, best_of_n=2)
    assert stage.agent.chunk_critic is None
    tokens, lengths = codec.encode(batch["high_value_action_chunks"])
    targets = goal_window_targets(batch, lengths, .9)
    torch.manual_seed(61)
    expected = stage.agent.losses({**batch, **targets}, tokens)
    # Old fixed-window targets must never be consumed in the new mode.
    for name in targets:
        batch[name].fill_(float("nan"))
    torch.manual_seed(61)
    pipeline = PipelineAlgo([stage], device="cpu")
    output = pipeline.forward_training({"ogbench": batch})["ogbench"]
    for name, value in expected.items():
        torch.testing.assert_close(output["loss/" + name], value, rtol=0, atol=0)
    assert set(expected) == {"action_critic", "value", "actor_bc"}
    assert "log/DQC/q_chunk/mean" not in output
    assert output["log/SMDP/duration/mean"] == 2
    assert all(not value.requires_grad for name, value in output.items() if name.startswith("log/"))
    sum(value for name, value in output.items() if name.startswith("loss/")).backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in stage.parameters())
    quiet = {**batch, "log_predictions": False}
    quiet = {key: value for key, value in quiet.items() if not key.startswith(("log/", "loss/"))}
    result = stage.execute(quiet, mode="train")
    assert not any(name.startswith("log/") for name in result)


@pytest.mark.parametrize("invalid", ["teacher", "asymmetric", "mode"])
def test_incompatible_backup_config_fails(invalid):
    args = dict(backup_mode="policy_window", use_chunk_critic=False, kappa_d=.5,
                observation_dim=2, goal_dim=2, action_dim=1, backup_horizon=4, hidden_dims=(4,))
    args.update({"teacher": {"use_chunk_critic": True}, "asymmetric": {"kappa_d": .8},
                 "mode": {"backup_mode": "tokens"}}[invalid])
    with pytest.raises(ValueError):
        QChunkingStage(ControlChunkCodec(1, 4), **args)


@pytest.mark.parametrize("lengths", [torch.tensor([0] * 7), torch.tensor([5] * 7),
                                   torch.tensor([2.] * 7), torch.tensor([2])])
def test_invalid_duration_rejected(lengths):
    _, batch = replay_batch()
    with pytest.raises(ValueError):
        goal_window_targets(batch, lengths, .9)


def test_recipe_overrides_teacher_after_all_domain_settings():
    root = Path(__file__).resolve().parents[1] / "egomimic/hydra_configs/benchmark"
    recipe = OmegaConf.load(root / "smdp_comparison.yaml")
    suite = OmegaConf.load(root / recipe.suite_recipe)
    for domain in suite.domains.values():
        cfg = OmegaConf.load(root / recipe.base_recipe)
        cfg.model.pipeline.stages[0].kappa_d = domain.kappa_d
        for name, value in recipe.overrides.items():
            OmegaConf.update(cfg, name, value)
        assert cfg.include_future_observations
        assert cfg.model.pipeline.stages[0].backup_mode == "policy_window"
        assert not cfg.model.pipeline.stages[0].use_chunk_critic
        assert cfg.model.pipeline.stages[0].kappa_d == .5


@pytest.mark.parametrize("damage", [None, "hash", "scale", "duplicate", "generator"])
def test_manifest_gate_rejects_unready_or_wrong_corpus(damage):
    from egomimic.scripts.data_download.wait_goal_manifest import validate_manifest
    manifest = {"status": "READY", "generator_sha256": "pinned", "transitions": 20,
                "shards": [{"sha256": "a", "array_sha256": "A", "rows": 12, "episodes": 2},
                           {"sha256": "b", "array_sha256": "B", "rows": 12, "episodes": 2}]}
    cfg = {"generator_sha256": "pinned", "expected_transitions": 20, "shards": 2}
    if damage == "scale":
        manifest["transitions"] = 19
    elif damage == "duplicate":
        manifest["shards"][1] = manifest["shards"][0]
    elif damage == "generator":
        manifest["generator_sha256"] = "different"
    raw = json.dumps(manifest).encode()
    metadata = {"sha256": "wrong" if damage == "hash" else hashlib.sha256(raw).hexdigest()}
    if damage:
        with pytest.raises(ValueError):
            validate_manifest(raw, metadata, cfg)
    else:
        assert validate_manifest(raw, metadata, cfg)["state"] == "READY"
