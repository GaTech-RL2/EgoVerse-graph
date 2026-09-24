"""Median duration matching preserves variable native time and fixed DQC TD."""
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from egomimic.eval.control_replay import (
    candidate_rank, duration_statistics, median_spatial_budgets, passes_trace_gates,
)
from egomimic.pipeline.stages_q_chunking import QChunkingStage
from egomimic.rldb.action_codec import ControlChunkCodec
from egomimic.rldb.goal_replay import GoalReplay


def recipe():
    return OmegaConf.load(Path(__file__).resolve().parents[1]
        / "egomimic/hydra_configs/benchmark/dqc_median_comparison.yaml")


def test_spatial_median_fit_retains_speed_distribution_and_stationary_holds():
    speeds = torch.tensor([0., .1, .2, .25, .3, .5, .7])
    actions = speeds[:, None, None].expand(-1, 25, 1)
    codec = ControlChunkCodec(1, 25, kind="arc", waypoints=3)
    budgets, evidence = median_spatial_budgets(codec, actions, 5, [1.])
    assert evidence["distance_at_target_median"] == 1.25
    matched = ControlChunkCodec(1, 25, kind="arc", waypoints=3, **budgets[0])
    latent, lengths = matched.encode(actions)
    decoded, decoded_lengths = matched.decode(latent)
    assert lengths.tolist() == [25, 13, 6, 5, 4, 2, 1]
    assert lengths.float().median() == 5
    assert lengths.unique().numel() > 1
    torch.testing.assert_close(lengths, decoded_lengths, rtol=0, atol=0)
    for index, length in enumerate(lengths):
        torch.testing.assert_close(decoded[index, :length], actions[index, :length], atol=1e-5, rtol=1e-5)
    stats = duration_statistics(lengths, 25)
    assert stats["native_steps_p50"] == 5 and stats["native_steps_histogram"]["25"] == 1
    assert stats["native_cap_fraction"] == pytest.approx(1/7)


def test_joint_translation_rotation_scaling_and_control_space():
    speeds = torch.tensor([.1, .15, .2, .25, .3, .4, .5])
    actions = torch.stack([speeds, speeds.flip(0)], -1)[:, None].expand(-1, 25, -1)
    codec = ControlChunkCodec(2, 25, kind="arc", translation_indices=[0], rotation_indices=[1], rotation=1.)
    choices, _ = median_spatial_budgets(codec, actions, 5, [.5, 1., 2.])
    for choice in choices:
        fitted = ControlChunkCodec(2, 25, kind="arc", translation_indices=[0], rotation_indices=[1], **choice)
        assert fitted.window_lengths(actions).float().median() == 5
    # Torque/control paths integrate differences, rather than treating torques
    # as Cartesian displacements or introducing a meaningless rotation budget.
    torque = torch.linspace(-1, 1, 25)[None, :, None] * speeds[:, None, None]
    control = ControlChunkCodec(1, 25, kind="arc", path_mode="control")
    choices, _ = median_spatial_budgets(control, torque, 5, [.5, 2.])
    assert len(choices) == 1 and choices[0]["rotation"] is None
    fitted = ControlChunkCodec(1, 25, kind="arc", path_mode="control", **choices[0])
    assert fitted.window_lengths(torque).float().median() == 5


def test_unidentifiable_duration_and_fixed_cap_are_rejected():
    codec = ControlChunkCodec(1, 25, kind="arc")
    with pytest.raises(ValueError, match="stationary"):
        median_spatial_budgets(codec, torch.zeros(8, 25, 1), 5, [1.])
    with pytest.raises(ValueError, match="below"):
        median_spatial_budgets(codec, torch.ones(8, 25, 1), 25, [1.])


def test_matching_gate_and_ranking_cannot_reward_longer_chunks():
    cfg = recipe().calibration
    short = {"native_steps_p50": 5., "mean_native_steps": 5.2,
             "encoded_scalars": 18, "scalar_compression_ratio": 1.4, "action_rmse_p90": .1}
    long = {**short, "native_steps_p50": 20., "mean_native_steps": 20., "scalar_compression_ratio": 6.}
    tail = {**short, "mean_native_steps": 12., "scalar_compression_ratio": 3.}
    assert passes_trace_gates(short, cfg) and not passes_trace_gates(long, cfg)
    assert candidate_rank(short, cfg) < candidate_rank(tail, cfg)
    # Existing calibration still uses its original mean/compression objective.
    del cfg.duration_matching
    assert passes_trace_gates(long, cfg)
    assert candidate_rank(long, cfg) < candidate_rank(short, cfg)


@pytest.mark.parametrize("kind", ["native", "native_window", "arc"])
def test_variable_policy_duration_keeps_fixed_teacher_targets(kind):
    data = {"observations": np.arange(120, dtype=np.float32).reshape(60, 2),
            "actions": np.full((60, 1), .1, dtype=np.float32),
            "terminals": np.array([0] * 59 + [1])}
    batch = GoalReplay(data, backup_horizon=25, discount=.999).sample(
        3, np.random.RandomState(0), indices=[1]*3, goal_indices=[50]*3)
    batch["high_value_action_chunks"] = torch.tensor([[[.9]]*25, [[.1]]*25, [[0.]]*25])
    codec = ControlChunkCodec(1, 5 if kind == "native" else 25, kind=kind, waypoints=3, distance=.5)
    lengths = codec.encode(batch["high_value_action_chunks"])[1]
    assert lengths.tolist() == ([5, 5, 5] if kind == "native" else [1, 5, 25])
    stage = QChunkingStage(codec, backup_mode="fixed", use_chunk_critic=True, observation_dim=2,
        goal_dim=2, action_dim=1, backup_horizon=25, discount=.999, hidden_dims=(8,),
        flow_steps=2, best_of_n=2, kappa_d=.8)
    targets = {key: value.clone() for key, value in batch.items() if key.startswith("high_value_")}
    output = stage.execute(batch, mode="train")
    for key, expected in targets.items():
        torch.testing.assert_close(batch[key], expected, rtol=0, atol=0)
    assert batch["high_value_backup_horizon"].tolist() == [25., 25., 25.]
    torch.testing.assert_close(batch["high_value_next_observations"], torch.tensor([[52.,53.]]*3))
    assert "loss/chunk_critic" in output and torch.isfinite(output["loss/chunk_critic"])
    assert output["log/TD/nonterminal_backup_horizon/mean"] == 25
    assert output["log/Chunk/native_duration/p50"] == 5


def test_median_recipe_retains_domain_teacher_and_update_budget():
    cfg = recipe()
    root = Path(__file__).resolve().parents[1] / "egomimic/hydra_configs/benchmark"
    suite = OmegaConf.load(root / cfg.suite_recipe)
    for domain in suite.domains.values():
        base = OmegaConf.load(root / cfg.base_recipe)
        for key in ("q_agg", "kappa_b", "kappa_d"):
            base.model.pipeline.stages[0][key] = domain[key]
        for key, value in cfg.overrides.items():
            OmegaConf.update(base, key, value)
        assert base.steps == 1000000 and base.batch_size == 4096 and base.backup_horizon == 25
        stage = base.model.pipeline.stages[0]
        assert stage.backup_mode == "fixed" and stage.use_chunk_critic
        assert stage.kappa_d == domain.kappa_d and stage.kappa_b == domain.kappa_b
