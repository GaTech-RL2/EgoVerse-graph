"""Multiplier semantics must not depend on physical speed or future actions."""

import torch

from egomimic.pipeline import stages_speed as m


def test_multiplier_input_contract():
    c = m.SharedSpeedCondition(
        None, condition_dim=256, conditioning_input="retiming_multiplier"
    )
    # Exercise learned dependence rather than only the zero-initialized branch.
    with torch.no_grad():
        c.mlp[0].weight.fill_(1)
        c.mlp[0].bias.zero_()
        c.mlp[-1].weight.fill_(0.1)
    obs = torch.randn(5, 256, requires_grad=True)
    rates = torch.tensor([0.2, 0.4, 0.6, 0.8, 1.0])
    a = c(
        {
            "condition": obs,
            "retiming_rate": rates[:, None],
            "requested_speed": torch.ones(5, 1),
        }
    )["speed_condition"]
    b = c(
        {
            "condition": obs,
            "retiming_rate": rates[:, None],
            "requested_speed": torch.ones(5, 1) * 999,
        }
    )["speed_condition"]
    assert a.shape == (5, 256) and torch.equal(a, b)
    assert not torch.equal(a[0] - obs[0], a[-1] - obs[-1])
    a.sum().backward()
    assert obs.grad is not None and c.mlp[0].weight.grad.abs().sum() > 0
    assert c({"condition": obs.detach()[:, None], "retiming_rate": rates[:, None]})[
        "speed_condition"
    ].shape == (5, 1, 256)
    for bad in (
        torch.zeros(5),
        -rates,
        torch.ones(5) * float("nan"),
        torch.ones(5) * float("inf"),
        torch.ones(5, 2),
    ):
        try:
            c({"condition": obs, "retiming_rate": bad})
        except ValueError:
            pass
        else:
            raise AssertionError("invalid multiplier accepted")
    try:
        c({"condition": obs, "requested_speed": torch.ones(5, 1)})
    except KeyError:
        pass
    else:
        raise AssertionError("physical-speed fallback accepted")


def test_strict_reload_rejects_physical_speed_weights():
    import pytest

    new = m.SharedSpeedCondition(condition_dim=256)
    old_state = dict(new.state_dict())
    old_state.pop("retiming_multiplier_contract")
    old_state["speed_reference"] = torch.tensor(1.0)
    with pytest.raises(RuntimeError):
        new.load_state_dict(old_state, strict=True)
    new.load_state_dict(new.state_dict(), strict=True)
    with pytest.raises(ValueError, match="removed"):
        m.SharedSpeedCondition(1.0)
    with pytest.raises(ValueError, match="removed"):
        m.SharedSpeedCondition(conditioning_input="native_speed")


def test_rollout_requires_explicit_multiplier():
    import pytest
    from omegaconf import OmegaConf

    cfg = OmegaConf.create(
        {
            "model": {
                "pipeline": {
                    "_target_": "egomimic.pipeline.stages_speed.build_speed_conditioned_pipeline",
                    "conditioning_input": "retiming_multiplier",
                }
            },
            "deployment": {"requested_multiplier": 0.4},
        }
    )
    assert m.requested_rollout_condition(cfg) == ("retiming_rate", 0.4)
    cfg.deployment.requested_speed = 12.0
    with pytest.raises(ValueError, match="requested_speed"):
        m.requested_rollout_condition(cfg)


def test_multiplier_reaches_shared_field_in_train_and_inference():
    from egomimic.pipeline.core import Pipeline
    from egomimic.pipeline.stages_action_flow import (
        ConditionalVelocityStage,
        LatentBridgeStage,
    )
    from egomimic.pipeline.stages_speed import SharedSpeedCondition

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


def test_stationary_retimer_emits_multiplier_without_speed():
    import numpy as np

    from egomimic.rldb.zarr.physical_retiming import PhysicalWindowRetiming

    rates = [0.2, 0.4, 0.6, 0.8, 1.0]
    keys = ["left.pose", "right.pose"]
    for view, rate in enumerate(rates):
        tf = PhysicalWindowRetiming(
            rates,
            dict.fromkeys(keys, "pose_wxyz"),
            keys,
            30,
            stride=3,
            timestamp_key="clock",
            conditioning_input="retiming_multiplier",
        )
        km = {k: {"horizon": 30} for k in keys + ["clock"]}
        tf.bind_episode({"fps": 30}, km)
        pose = np.zeros((30, 7))
        pose[:, 0] = np.arange(30)
        pose[:, 3] = 1
        raw = {k: pose.copy() for k in keys}
        raw.update(clock=np.arange(30, dtype=np.int64) * 33333333, _retiming_view=view)
        out = tf.transform(raw)
        assert out["retiming_rate"].shape == (1,)
        np.testing.assert_allclose(out["retiming_rate"], [rate])
        assert "requested_speed" not in out and "requested_speed_value" not in out


def test_stationary_recipe_keeps_selected_rates_and_dimensions():
    from pathlib import Path

    from omegaconf import OmegaConf

    root = Path(__file__).parents[1] / "egomimic/hydra_configs"
    cfg = OmegaConf.load(
        root / "experiment/e1/yam_human_keypoints_multiplier_h816_private512_s42.yaml"
    )
    assert list(cfg.stationary_speed.human_rates) == [0.2, 0.4, 0.6, 0.8, 1.0]
    assert list(cfg.stationary_speed.yam_rates) == [1.0]
    assert cfg.model.pipeline.conditioning_input == "retiming_multiplier"
    assert cfg.model.pipeline.speed_reference is None
    data = OmegaConf.load(
        root / "data/e1/yam_human_keypoints_multiplier_proportional_val01.yaml"
    )
    for source in (
        data.train_datasets.yam_bimanual,
        data.train_datasets.human_bimanual,
        data.valid_datasets.yam.yam_bimanual,
        data.valid_datasets.human.human_bimanual,
    ):
        win = source.resolver.transform_list.window_transform
        assert (
            OmegaConf.to_container(win, resolve=False)["conditioning_input"]
            == "${stationary_speed.conditioning_input}"
        )


def test_multiplier_semantics_preserve_normalization_inputs():
    import numpy as np

    from egomimic.rldb.zarr.physical_retiming import PhysicalWindowRetiming

    for domain, horizon, rates in (
        ("human", 30, [0.2, 0.4, 0.6, 0.8, 1.0]),
        ("robot", 100, [1.0]),
    ):
        fields = {"left": "pose_wxyz", "right": "pose_wxyz", "articulation": "linear"}
        transform = PhysicalWindowRetiming(
            rates,
            fields,
            ["left", "right"],
            horizon,
            embodiment=domain,
            sample_views=5,
            timestamp_key="clock" if domain == "human" else None,
        )
        km = {key: {"horizon": horizon} for key in fields}
        if domain == "human":
            km["clock"] = {"horizon": horizon}
        transform.bind_episode({"fps": 30}, km)
        pose = np.zeros((horizon, 7))
        pose[:, 0] = np.arange(horizon) * 0.01
        pose[:, 3] = 1
        for view in range(5):
            rate = rates[view % len(rates)]
            values = np.arange(horizon * 126, dtype=float).reshape(horizon, 126)
            batch = {
                "left": pose.copy(),
                "right": pose.copy(),
                "articulation": values.copy(),
                "_retiming_view": view,
            }
            if domain == "human":
                batch["clock"] = np.arange(horizon, dtype=np.int64) * 33333333
            out = transform.transform(batch)
            for key in ("left", "right"):
                np.testing.assert_allclose(out[key][:, 0], pose[:, 0] * rate)
                np.testing.assert_array_equal(out[key][:, 3:], pose[:, 3:])
            expected = values[0] + (values - values[0]) * rate
            np.testing.assert_allclose(out["articulation"], expected)
            assert "requested_speed" not in out and out["retiming_rate"].shape == (1,)
