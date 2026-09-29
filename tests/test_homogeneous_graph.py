"""Shared graph forwards preserve routing, per-source losses and gradients."""

from collections import OrderedDict
from copy import deepcopy

import pytest
import torch
from torch import nn

from egomimic.pipeline.algo import PipelineAlgo
from egomimic.pipeline.core import Pipeline, Stage
from egomimic.pipeline.stages_flow import FlowDenoiserStage, FlowVelocityLossStage
from egomimic.pipeline.stages_hpt import HPTStemStage, HPTTrunkStage
from egomimic.pipeline.stages_sampler import FusedObsEncoder
from egomimic.utils.batch_utils import map_batches


class CountingVelocity(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.3))
        self.calls = []

    def forward(self, action, time, condition):
        self.calls.append(len(action))
        return self.weight * (action + condition[:, None, :2] + time[:, None, None])


def test_flow_combines_unequal_source_batches_and_preserves_loss_gradients():
    head = FlowDenoiserStage(CountingVelocity(), 3, 2, 4)
    graph = PipelineAlgo([head, FlowVelocityLossStage()], device="cpu")
    separate = deepcopy(graph)
    separate.homogeneous_training = False
    batches = OrderedDict(
        (
            name,
            {
                "condition": torch.randn(size, 4),
                "flow/noisy_action": torch.randn(size, 3, 2),
                "flow/time": torch.rand(size),
                "flow/velocity_target": torch.randn(size, 3, 2),
                "metadata": name,
            },
        )
        for name, size in (("human", 2), ("robot", 5))
    )
    grouped = graph.forward_training(batches)
    reference = separate.forward_training(batches)
    assert head.model.calls == [7]
    assert separate.pipeline.stages[0].model.calls == [2, 5]
    for source in batches:
        assert grouped[source]["metadata"] == source
        torch.testing.assert_close(
            grouped[source]["loss/flow_velocity"],
            reference[source]["loss/flow_velocity"],
        )
        assert "flow/predicted_velocity" not in batches[source]
    loss = graph.compute_losses(grouped, batches)["loss"]
    expected = (
        2 * reference["human"]["loss/flow_velocity"]
        + 5 * reference["robot"]["loss/flow_velocity"]
    ) / 7
    torch.testing.assert_close(loss, expected)
    loss.backward()
    expected.backward()
    torch.testing.assert_close(
        head.model.weight.grad, separate.pipeline.stages[0].model.weight.grad
    )


def test_diffusion_combines_batches_and_keeps_scheduler_validation():
    from egomimic.models.ddim_scheduler import DDIMScheduler
    from egomimic.models.diffusion_policy import DiffusionPolicy
    from egomimic.pipeline.stages_diffusion import (
        DiffusionDenoiserStage,
        _scheduler_signature,
    )

    scheduler = DDIMScheduler(num_train_timesteps=10, prediction_type="epsilon")
    network = CountingVelocity()
    stage = DiffusionDenoiserStage(DiffusionPolicy(network, scheduler, 3, 2), 3, 2, 4)
    reference = deepcopy(stage)
    batches = {
        name: {
            "condition": torch.randn(size, 4),
            "diffusion/noisy_action": torch.randn(size, 3, 2),
            "diffusion/timestep": torch.ones(size, dtype=torch.long),
            "diffusion/scheduler_signature": _scheduler_signature(scheduler),
        }
        for name, size in (("a", 2), ("b", 5))
    }
    expected = {
        key: reference.execute(dict(batch), mode="train")
        for key, batch in batches.items()
    }
    results = stage.execute_batches(batches, mode="train")
    assert network.calls == [7]
    for key in results:
        torch.testing.assert_close(
            results[key]["diffusion/predicted_noise"],
            expected[key]["diffusion/predicted_noise"],
        )
    sum(
        row["diffusion/predicted_noise"].square().sum() for row in results.values()
    ).backward()
    sum(
        row["diffusion/predicted_noise"].square().sum() for row in expected.values()
    ).backward()
    torch.testing.assert_close(network.weight.grad, reference.policy.model.weight.grad)
    batches["b"]["diffusion/scheduler_signature"] = ("different",)
    with pytest.raises(ValueError, match="schedulers differ"):
        stage.execute_batches(batches, mode="train")


class CountingStem(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.7))
        self.calls = []

    def compute_latent(self, value):
        self.calls.append(len(value))
        return value[:, None, :] * self.weight


class CountingTrunk(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 4)
        self.calls = []

    def forward(self, tokens):
        self.calls.append(len(tokens))
        return self.proj(tokens)


def test_hpt_batches_shared_stems_and_trunk_after_domain_embedding():
    shared, human, robot = CountingStem(), CountingStem(), CountingStem()
    stem = HPTStemStage(
        {"camera": shared}, {"human": {"state": human}, "robot": {"state": robot}}
    )
    trunk = HPTTrunkStage(
        CountingTrunk(), 4, domains=["human", "robot"], use_domain_embedding=True
    )
    pipeline = Pipeline([stem, trunk])
    reference = deepcopy(pipeline)
    batches = OrderedDict(
        (
            name,
            {
                "camera": torch.randn(size, 4),
                "state": torch.randn(size, 4),
                "embodiment": domain,
            },
        )
        for name, size, domain in (
            ("a", 2, "human"),
            ("b", 3, "robot"),
            ("c", 4, "human"),
        )
    )
    expected = {
        source: reference.execute(batch, mode="train")
        for source, batch in batches.items()
    }
    result = pipeline.execute_batches(batches, mode="train")
    assert shared.calls == [9] and human.calls == [6] and robot.calls == [3]
    assert trunk.trunk.calls == [9]
    for source in batches:
        torch.testing.assert_close(
            result[source]["condition"], expected[source]["condition"]
        )
    sum(row["condition"].square().sum() for row in result.values()).backward()
    sum(row["condition"].square().sum() for row in expected.values()).backward()
    for param, original in zip(pipeline.parameters(), reference.parameters()):
        torch.testing.assert_close(param.grad, original.grad)


def test_hpt_different_token_counts_fall_back_to_separate_groups():
    trunk = HPTTrunkStage(
        CountingTrunk(), 4, token_postprocessing="mean", use_position_embedding=False
    )
    batches = {
        "a": {"hpt/tokens": torch.randn(2, 3, 4)},
        "b": {"hpt/tokens": torch.randn(4, 5, 4)},
    }
    result = trunk.execute_batches(batches, mode="train")
    assert trunk.trunk.calls == [2, 4]
    assert result["a"]["condition"].shape == (2, 4)


def test_map_batches_preserves_dtype_groups_and_rejects_reduced_outputs():
    calls = []
    inputs = {"a": torch.ones(2, 3), "b": torch.ones(1, 3, dtype=torch.float64)}
    result = map_batches(inputs, lambda value: calls.append(value.dtype) or value * 2)
    assert calls == [torch.float32, torch.float64]
    assert result["b"].dtype == torch.float64
    with pytest.raises(ValueError, match="leading sample"):
        map_batches(inputs, lambda value: value.mean())
    with pytest.raises(ValueError, match="leading sample"):
        map_batches(
            {
                "a": {"state": torch.zeros(2, 3), "action": torch.zeros(3, 2)},
                "b": {"state": torch.zeros(3, 3), "action": torch.zeros(2, 2)},
            },
            lambda value: value["action"],
        )


class PackedEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward_packed(self, *, obs_packed, cu_seqlens, T_total, **context):
        self.calls.append((T_total, cu_seqlens.tolist()))
        return obs_packed["state"]


def test_observation_grouping_preserves_episode_boundaries():
    encoder = PackedEncoder()
    stage = FusedObsEncoder(encoder, {"state": "state"}, n_obs_steps=2)
    batches = {
        "a": {"state": torch.randn(2, 2, 4)},
        "b": {"state": torch.randn(3, 2, 4)},
    }
    result = stage.execute_batches(batches, mode="train")
    assert encoder.calls == [(10, [0, 2, 4, 6, 8, 10])]
    assert result["a"]["condition"].shape == (2, 8)


class StatefulCustom(Stage):
    reads = ("x",)
    writes = ("loss/custom",)

    def forward(self, batch):
        batch["loss/custom"] = batch["x"].mean()
        return batch


def test_custom_stage_contracts_and_blocked_graphs_keep_separate_execution():
    pipeline = Pipeline([StatefulCustom()])
    batches = {"a": {"x": torch.ones(2, 1)}, "b": {"x": torch.ones(3, 1) * 2}}
    result = pipeline.execute_batches(batches, mode="train")
    assert result["a"]["loss/custom"] == 1 and result["b"]["loss/custom"] == 2
    with pytest.raises(RuntimeError, match="blocked"):
        pipeline.execute_batches({"a": {"wrong": torch.ones(1)}}, mode="train")
