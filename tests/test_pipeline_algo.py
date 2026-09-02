from collections import OrderedDict

import pytest
import torch

from egomimic.pipeline.algo import PipelineAlgo
from egomimic.pipeline.core import Stage
from egomimic.pipeline.stages_io import ActionTargetBuilder
from egomimic.pipeline.stages_sampler import FusedObsEncoder


class _NormStats:
    _types = {
        "state": "proprio_keys",
        "front_img_1": "camera_keys",
        "actions_cartesian": "action_keys",
    }
    _zarr = {
        "observations.state": "state",
        "observations.images.front": "front_img_1",
        "actions": "actions_cartesian",
    }

    def keys_of_type(self, key_type, embodiment_id):
        assert embodiment_id == 6
        return [key for key, value in self._types.items() if value == key_type]

    def is_key_with_embodiment(self, key, embodiment_id):
        return embodiment_id == 6 and key in self._types

    def zarr_key_to_keyname(self, key, embodiment_id):
        assert embodiment_id == 6
        return self._zarr.get(key)

    def keyname_to_zarr_key(self, key, embodiment_id):
        assert embodiment_id == 6
        return next((raw for raw, name in self._zarr.items() if name == key), None)

    def normalize(self, batch, embodiment_id):
        assert embodiment_id == 6
        return dict(batch)

    def unnormalize(self, batch, embodiment_id):
        assert embodiment_id == 6
        return dict(batch)


class _PackedState(torch.nn.Module):
    def forward_packed(self, *, obs_packed, **kwargs):
        return obs_packed["state"]


class _TinyHead(Stage):
    reads = ["condition", "target", "embodiment"]
    writes = ["loss/fit", "log/fit"]
    reads_by_mode = {"rollout": ["condition", "embodiment"]}
    writes_by_mode = {"rollout": ["pred_action"]}

    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.5))

    def forward(self, batch):
        if self.training:
            loss = (batch["target"] * self.scale).square().mean()
            batch["loss/fit"] = loss
            batch["log/fit"] = loss.detach()
        else:
            batch_size = batch["condition"].shape[0]
            batch["pred_action"] = self.scale.expand(batch_size, 3, 2)
        return batch


class _ObservationAdapter:
    def encode(self, batch):
        output = dict(batch)
        output["observations.state"] = output["raw_state"] + 1
        return output


class _ActionAdapter:
    def __init__(self):
        self.context = None

    def decode(self, actions, context=None):
        self.context = context
        return actions + 2


def _algo():
    return PipelineAlgo(
        stages=[
            FusedObsEncoder(encoder=_PackedState(), n_obs_steps=1),
            ActionTargetBuilder(),
            _TinyHead(),
        ],
        norm_stats=_NormStats(),
        domains=["eva_bimanual"],
        ac_keys={"eva_bimanual": "actions_cartesian"},
        action_horizon=3,
        device="cpu",
    )


def _raw_batch():
    return {
        "eva_bimanual": {
            "observations.state": torch.randn(2, 4),
            "observations.images.front": torch.randn(2, 3, 8, 8),
            "actions": torch.randn(2, 3, 2),
            "ignored": torch.ones(2),
        }
    }


def test_pipeline_algo_maps_loader_keys_and_reduces_domain_losses():
    algo = _algo()
    processed = algo.process_batch_for_training(_raw_batch())

    assert list(processed) == [6]
    assert set(processed[6]) == {"state", "front_img_1", "actions_cartesian"}
    predictions = algo.forward_training(processed)
    losses = algo.compute_losses(predictions, processed)

    assert isinstance(predictions, OrderedDict)
    assert losses["action_loss"].ndim == 0
    assert torch.equal(losses["action_loss"], predictions["6_action_loss"])
    losses["action_loss"].backward()
    assert algo.policy.stages[2].scale.grad is not None
    logged = algo.log_info({"losses": losses})
    assert logged["MSE"] == pytest.approx(losses["action_loss"].item())
    assert logged["MSE/eva_bimanual"] == pytest.approx(losses["6_action_loss"].item())


def test_pipeline_algo_teacher_forced_eval_uses_standard_prediction_names():
    algo = _algo()
    processed = algo.process_batch_for_training(_raw_batch())
    algo.policy.eval()

    predictions = algo.forward_eval(processed)

    expected = torch.full((2, 3, 2), 0.5)
    assert torch.equal(predictions["emb6_actions_cartesian"], expected)
    assert torch.equal(predictions["eva_bimanual_actions_cartesian"], expected)
    assert torch.equal(predictions["actions_cartesian"], expected)


def test_pipeline_algo_rollout_adapts_before_normalizing_and_decodes_after():
    action_adapter = _ActionAdapter()
    algo = PipelineAlgo(
        stages=[
            FusedObsEncoder(encoder=_PackedState(), n_obs_steps=1),
            ActionTargetBuilder(),
            _TinyHead(),
        ],
        norm_stats=_NormStats(),
        domains=["eva_bimanual"],
        ac_keys={"eva_bimanual": "actions_cartesian"},
        rollout_adapters={"eva_bimanual": action_adapter},
        rollout_observation_adapters={"eva_bimanual": _ObservationAdapter()},
        action_horizon=3,
        device="cpu",
    )
    raw = {
        "raw_state": torch.zeros(2, 4),
        "actions": torch.randn(2, 3, 2),
    }

    processed = algo.process_batch_for_rollout({"eva_bimanual": raw})
    algo.policy.eval()
    predictions = algo.forward_rollout(processed, rollout_t=7)

    assert set(processed[6]) == {"state"}
    assert torch.equal(processed[6]["state"], torch.ones(2, 4))
    assert torch.equal(
        predictions["eva_bimanual_actions_cartesian"],
        torch.full((2, 3, 2), 2.5),
    )
    assert torch.equal(
        predictions["eva_bimanual_actions_cartesian_tokens"],
        torch.full((2, 3, 2), 0.5),
    )
    assert action_adapter.context == processed[6]
    assert algo.rollout_adapter_for(6) is action_adapter


def test_pipeline_algo_validates_domain_action_mapping():
    with pytest.raises(ValueError, match="exactly match"):
        PipelineAlgo(
            stages=[],
            norm_stats=_NormStats(),
            domains=["eva_bimanual"],
            ac_keys={},
            device="cpu",
        )
