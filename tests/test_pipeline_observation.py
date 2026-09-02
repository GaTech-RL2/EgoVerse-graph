import pytest
import torch
import torch.nn as nn

from egomimic.pipeline.stages_io import ActionTargetBuilder
from egomimic.pipeline.stages_sampler import (
    DPStyleObsEncoder,
    FusedObsEncoder,
    GaussianLatentNoise,
)


class _PackedState(nn.Module):
    def __init__(self):
        super().__init__()
        self.crop_scope = "episode"
        self.seen = None

    def forward_packed(
        self,
        *,
        actions_packed,
        obs_packed,
        cu_seqlens,
        T_total,
        embodiment_id,
        **kwargs,
    ):
        self.seen = {
            "action_shape": tuple(actions_packed.shape),
            "cu": cu_seqlens.tolist(),
            "T_total": T_total,
            "embodiment": embodiment_id,
        }
        return obs_packed["state"]


def test_dp_style_encoder_concatenates_sliced_state_and_image_features():
    image_encoder = nn.Linear(3, 2, bias=False)
    encoder = DPStyleObsEncoder(
        obs_specs={"state": {"input_dim": 2, "input_slice": [1, 3]}},
        img_encoders={"image": image_encoder},
    )
    state = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    image = torch.ones(4, 3)

    output = encoder.forward_packed(
        obs_packed={"state": state, "image": image}, T_total=4
    )

    assert output.shape == (4, 4)
    assert torch.equal(output[:, :2], state[:, 1:3])
    assert torch.equal(output[:, 2:], image_encoder(image))


def test_fused_encoder_packs_history_without_owning_the_action_target():
    encoder = _PackedState()
    stage = FusedObsEncoder(encoder=encoder, n_obs_steps=2)
    state = torch.arange(24, dtype=torch.float32).reshape(3, 2, 4)
    actions = torch.randn(3, 5, 6)

    output = stage(
        {"obs/state": state, "actions": actions, "embodiment": "eva_bimanual"}
    )

    assert output["condition"].shape == (3, 8)
    assert output["actions"] is actions
    assert "target" not in output
    assert encoder.seen == {
        "action_shape": (6, 1),
        "cu": [0, 2, 4, 6],
        "T_total": 6,
        "embodiment": "eva_bimanual",
    }
    assert encoder._episode_cu.tolist() == [0, 2, 4, 6]


def test_action_target_builder_is_a_separate_train_only_node():
    actions = torch.randn(3, 5, 6)
    stage = ActionTargetBuilder()

    output = stage({"actions": actions})

    assert output == {"target": actions}
    assert stage.train_only is True
    assert stage.contract("train") == (("actions",), ("target",))


def test_fused_encoder_rollout_requires_explicit_singleton_history_axis():
    stage = FusedObsEncoder(encoder=_PackedState(), n_obs_steps=1)
    with pytest.raises(ValueError, match="singleton history axis"):
        stage(
            {
                "obs/state": torch.randn(2, 4),
                "embodiment": "eva_bimanual",
                "rollout_t": 0,
            }
        )

    output = stage(
        {
            "obs/state": torch.randn(2, 1, 4),
            "embodiment": "eva_bimanual",
            "rollout_t": 0,
        }
    )
    assert output["condition"].shape == (2, 4)
    assert "target" not in output


def test_required_observations_are_reflected_in_mode_contracts():
    stage = FusedObsEncoder(
        encoder=_PackedState(),
        n_obs_steps=1,
        required_obs_keys=["state", "obs/front_img_1"],
    )
    assert stage.contract("train")[0] == (
        "obs/state",
        "obs/front_img_1",
        "embodiment",
    )
    assert stage.contract("rollout")[0] == stage.contract("train")[0]


def test_gaussian_noise_supports_new_and_legacy_token_names():
    condition = torch.zeros(2, 7)
    assert GaussianLatentNoise(num_tokens=4, latent_dim=3)({"condition": condition})[
        "sampler/noise"
    ].shape == (2, 4, 3)
    legacy = GaussianLatentNoise(action_horizon=5, latent_dim=2)
    assert legacy.action_horizon == 5
    assert legacy({"condition": condition})["sampler/noise"].shape == (2, 5, 2)

    with pytest.raises(ValueError, match="not both"):
        GaussianLatentNoise(num_tokens=4, action_horizon=4)
