"""Prevent substituting measured motion for the augmentation multiplier."""

from pathlib import Path

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from egomimic.pipeline.core import Pipeline
from egomimic.pipeline.stages_action_flow import (
    ConditionalVelocityStage,
    LatentBridgeStage,
)
from egomimic.pipeline.stages_speed import (
    SharedSpeedCondition,
    requested_rollout_condition,
)
from egomimic.rldb.zarr.planar_retiming import PlanarCommandRetiming


def test_actual_motion_does_not_determine_multiplier_condition():
    transform = PlanarCommandRetiming((1.0, 2.0))
    stage = SharedSpeedCondition(
        None, condition_dim=4, conditioning_input="retiming_multiplier"
    )
    seen = []
    handle = stage.mlp.register_forward_pre_hook(
        lambda _, args: seen.append(args[0].clone())
    )
    for distance in (0.0, 100.0):
        actions = np.zeros((31, 3))
        actions[:, 0] = np.arange(31) * distance
        result = transform.transform({"actions": actions, "_retiming_view": 1})
        stage(
            {
                "condition": torch.zeros(1, 4),
                "retiming_rate": torch.from_numpy(result["retiming_rate"]).reshape(
                    1, 1
                ),
                "requested_speed": torch.tensor([[distance]]),
            }
        )
    handle.remove()
    assert all(torch.equal(value, torch.tensor([[2.0]])) for value in seen)
    with pytest.raises(KeyError, match="retiming_rate"):
        stage({"condition": torch.zeros(1, 4), "requested_speed": torch.ones(1, 1)})


@pytest.mark.parametrize("value", (0.0, -1.0, float("nan"), float("inf")))
def test_bad_multiplier_fails_closed(value):
    stage = SharedSpeedCondition(None, conditioning_input="retiming_multiplier")
    with pytest.raises(ValueError):
        stage(
            {"condition": torch.zeros(1, 128), "retiming_rate": torch.tensor([[value]])}
        )


def test_strict_reload_cannot_reinterpret_native_speed_checkpoint():
    new = SharedSpeedCondition(None, conditioning_input="retiming_multiplier")
    old_state = dict(new.state_dict())
    old_state.pop("retiming_multiplier_contract")
    old_state["speed_reference"] = torch.tensor(100.0)
    with pytest.raises(RuntimeError, match="retiming_multiplier_contract"):
        new.load_state_dict(old_state, strict=True)
    new.load_state_dict(new.state_dict(), strict=True)


def test_pusht_adapter_rejects_physical_speed_configuration():
    from egomimic.pipeline.stages_speed import build_multiplier_conditioned_pipeline

    with pytest.raises(ValueError, match="raw retiming_multiplier"):
        build_multiplier_conditioned_pipeline([], conditioning_input="native_speed")
    with pytest.raises(TypeError, match="speed_reference"):
        build_multiplier_conditioned_pipeline([], speed_reference=100.0)
    root = Path(__file__).parents[1] / "egomimic/hydra_configs/experiment"
    assert not (root / "pusht/action_flow_cotrain_uc_speed_interpolation.yaml").exists()
    assert not (
        root / "pusht_historical/action_flow_cotrain_uc_speed_interpolation.yaml"
    ).exists()


def test_pusht_retiming_does_not_compute_physical_speed(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("PushT must not calculate a velocity-derived condition")

    monkeypatch.setattr(np.linalg, "norm", forbidden)
    result = PlanarCommandRetiming((2.0,)).transform(
        {
            "actions": np.zeros((31, 3)),
            "_retiming_view": 0,
            "requested_speed": np.asarray([999.0]),
            "requested_speed_value": np.asarray(999.0),
        }
    )
    assert "requested_speed" not in result and "requested_speed_value" not in result
    np.testing.assert_array_equal(result["retiming_rate"], [2.0])


def test_multiplier_reaches_shared_field_in_train_and_inference():
    class Field(torch.nn.Module):
        def forward(self, x, t, condition, **kwargs):
            return condition[:, : x.shape[-1]].unsqueeze(1).expand_as(x)

    condition = SharedSpeedCondition(
        None, condition_dim=16, conditioning_input="retiming_multiplier"
    )
    for layer in (condition.mlp[0], condition.mlp[-1]):
        torch.nn.init.constant_(layer.weight, 0.1)
        torch.nn.init.zeros_(layer.bias)
    graph = Pipeline(
        [
            condition,
            LatentBridgeStage(
                samples_per_content=1,
                condition_key="speed_condition",
                condition_dropout_probability=0,
            ),
            ConditionalVelocityStage(
                Field(),
                num_inference_steps=2,
                inference_method="euler",
                inference_condition_key="speed_condition",
            ),
        ]
    )
    for mode in ("train", "inference"):
        result = graph.execute(
            {
                "condition": torch.zeros(2, 16),
                "retiming_rate": torch.tensor([[1.0], [2.0]]),
                "action_flow/clean_latent": torch.zeros(2, 8, 16),
                "sampler/noise": torch.zeros(2, 8, 16),
            },
            mode=mode,
        )
        key = (
            "action_flow/predicted_velocity"
            if mode == "train"
            else "action_flow/generated_latent"
        )
        assert not torch.equal(result[key][0], result[key][1])
        if mode == "train":
            result[key].square().mean().backward()
            assert condition.mlp[0].weight.grad.abs().sum() > 0


def test_recipe_preserves_compression_and_explicit_rollout_contract(monkeypatch):
    monkeypatch.setenv("PUSHSHAPES_USOCKET_ROOT", "/unused/u")
    monkeypatch.setenv("PUSHSHAPES_CHAIN_GRIPPER_ROOT", "/unused/c")
    root = Path(__file__).parents[1] / "egomimic/hydra_configs"
    with initialize_config_dir(config_dir=str(root), version_base="1.3"):
        cfg = compose(
            config_name="train_zarr_cartesian",
            overrides=[
                "hydra/launcher=basic",
                "+experiment=pusht/action_flow_cotrain_uc_multiplier_interpolation",
            ],
        )
    assert (
        cfg.model.num_latent_tokens,
        cfg.model.latent_dim,
        cfg.model.action_horizon,
    ) == (8, 16, 16)
    assert "speed_reference" not in cfg.model.pipeline
    assert cfg.model.pipeline.conditioning_input == "retiming_multiplier"
    with pytest.raises(ValueError, match="requested_multiplier"):
        requested_rollout_condition(cfg)
    OmegaConf.update(cfg, "deployment.requested_multiplier", 1.0, force_add=True)
    assert requested_rollout_condition(cfg) == ("retiming_rate", 1.0)
    OmegaConf.update(cfg, "deployment.requested_speed", 100.0, force_add=True)
    with pytest.raises(ValueError, match="requested_speed"):
        requested_rollout_condition(cfg)
