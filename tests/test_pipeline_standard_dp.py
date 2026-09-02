from copy import deepcopy
from pathlib import Path

import pytest
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from egomimic.models.ddim_scheduler import DDIMScheduler
from egomimic.models.denoising_nets import ConditionalUnet1D
from egomimic.models.diffusion_policy import DiffusionPolicy
from egomimic.pipeline.core import Pipeline
from egomimic.pipeline.stages_diffusion import (
    DiffusionDenoiserStage,
    DiffusionEpsilonLossStage,
    DiffusionNoisingStage,
)
from egomimic.pipeline.stages_io import ActionTargetBuilder
from egomimic.pipeline.stages_sampler import FusedObsEncoder

CHAIN = "chain"
USOCKET = "u_socket"


def _tiny_scheduler(prediction_type: str = "epsilon", timesteps: int = 8):
    return DDIMScheduler(
        num_train_timesteps=timesteps,
        beta_schedule="squaredcos_cap_v2",
        prediction_type=prediction_type,
    )


def _tiny_policy(domain: str, action_dim: int, prediction_type: str = "epsilon"):
    return DiffusionPolicy(
        model=ConditionalUnet1D(
            input_dim=action_dim,
            cond_dim=5,
            ac_latent_seq=1,
            diffusion_step_embed_dim=16,
            down_dims=[8, 16],
            kernel_size=3,
            n_groups=4,
            cond_predict_scale=True,
        ),
        noise_scheduler=_tiny_scheduler(prediction_type),
        action_horizon=4,
        infer_ac_dims={domain: action_dim},
        num_inference_steps=2,
    )


def _factorized_stages(domain: str = CHAIN, action_dim: int = 6):
    return (
        ActionTargetBuilder(),
        DiffusionNoisingStage(
            noise_scheduler=_tiny_scheduler(),
            action_dim=action_dim,
            action_horizon=4,
        ),
        DiffusionDenoiserStage(
            policy=_tiny_policy(domain, action_dim),
            action_horizon=4,
            action_dim=action_dim,
            condition_input_dim=5,
        ),
        DiffusionEpsilonLossStage(),
    )


def test_factorized_training_uses_epsilon_loss_and_backpropagates():
    stages = _factorized_stages()
    output = {
        "condition": torch.randn(2, 5),
        "actions": torch.randn(2, 4, 6),
    }
    for stage in stages:
        output = stage.train()(output)

    assert output["diffusion/noisy_action"].shape == (2, 4, 6)
    assert output["diffusion/timestep"].shape == (2,)
    assert output["diffusion/predicted_noise"].shape == (2, 4, 6)
    assert output["loss/diffusion_noise"].ndim == 0
    assert torch.isfinite(output["loss/diffusion_noise"])
    output["loss/diffusion_noise"].backward()
    denoiser = stages[2]
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in denoiser.parameters()
    )


def test_factorized_training_is_rng_equivalent_to_diffusion_policy_predict():
    policy = _tiny_policy(CHAIN, 6)
    reference_policy = deepcopy(policy)
    target = torch.randn(2, 4, 6)
    condition = torch.randn(2, 5)

    torch.manual_seed(123)
    expected_prediction, expected_noise = reference_policy.predict(target, condition)
    expected_loss = reference_policy.loss_fn(expected_prediction, expected_noise)

    stages = (
        DiffusionNoisingStage(
            noise_scheduler=_tiny_scheduler(),
            action_dim=6,
            action_horizon=4,
        ),
        DiffusionDenoiserStage(
            policy=policy,
            action_horizon=4,
            action_dim=6,
            condition_input_dim=5,
        ),
        DiffusionEpsilonLossStage(),
    )
    torch.manual_seed(123)
    output = {"condition": condition, "target": target}
    for stage in stages:
        output = stage.train()(output)

    assert torch.equal(output["diffusion/predicted_noise"], expected_prediction)
    assert torch.equal(output["diffusion/noise_target"], expected_noise)
    assert torch.equal(output["loss/diffusion_noise"], expected_loss)


@pytest.mark.parametrize("action_key,width", [(CHAIN, 6), (USOCKET, 4)])
def test_denoiser_rollout_samples_the_common_padded_width(action_key, width):
    stage = DiffusionDenoiserStage(
        policy=_tiny_policy(action_key, width),
        action_horizon=4,
        action_dim=width,
        condition_input_dim=5,
    ).eval()
    with torch.inference_mode():
        output = stage({"condition": torch.randn(2, 5)})

    assert output["pred_action"].shape == (2, 4, width)
    assert torch.isfinite(output["pred_action"]).all()
    assert output["log/diffusion_inference_steps"] == 2.0


def test_nodes_reject_wrong_objective_width_and_scheduler_pair():
    with pytest.raises(ValueError, match="epsilon noise objective"):
        DiffusionNoisingStage(
            noise_scheduler=_tiny_scheduler("sample"),
            action_dim=6,
            action_horizon=4,
        )

    target, noising, denoiser, _ = _factorized_stages(USOCKET, 4)
    with pytest.raises(ValueError, match="must be"):
        noising(
            target(
                {
                    "actions": torch.randn(2, 4, 6),
                }
            )
        )

    noising = DiffusionNoisingStage(
        noise_scheduler=_tiny_scheduler(timesteps=7),
        action_dim=6,
        action_horizon=4,
    )
    mismatched = noising({"target": torch.randn(2, 4, 6)})
    with pytest.raises(ValueError, match="schedulers differ"):
        denoiser = DiffusionDenoiserStage(
            policy=_tiny_policy(CHAIN, 6),
            action_horizon=4,
            action_dim=6,
            condition_input_dim=5,
        )
        denoiser({**mismatched, "condition": torch.randn(2, 5)})


def test_diffusion_pipeline_contracts_plan_factorized_train_and_rollout():
    encoder = FusedObsEncoder(torch.nn.Identity(), n_obs_steps=1)
    target, noising, denoiser, loss = _factorized_stages()
    pipeline = Pipeline([encoder, target, noising, denoiser, loss])

    assert encoder.contract("train") == (
        ("obs/*",),
        ("condition",),
    )
    assert denoiser.contract("rollout") == (
        ("condition",),
        ("pred_action", "log/*"),
    )
    train, train_excluded = pipeline.plan(["obs/state", "actions"], mode="train")
    rollout, rollout_excluded = pipeline.plan(["obs/state"], mode="rollout")
    assert train == [encoder, target, noising, denoiser, loss]
    assert train_excluded == []
    assert rollout == [encoder, denoiser]
    assert rollout_excluded == [
        (target, ["<train-only>"]),
        (noising, ["<train-only>"]),
        (loss, ["<train-only>"]),
    ]


def test_hydra_factorized_dp_fragment_instantiates():
    path = (
        Path(__file__).parents[1]
        / "egomimic"
        / "hydra_configs"
        / "model"
        / "pipeline_standard_dp_stage.yaml"
    )
    cfg = OmegaConf.load(path)
    cfg.stages[1].action_horizon = 4
    cfg.stages[1].noise_scheduler.num_train_timesteps = 8
    cfg.stages[2].action_horizon = 4
    cfg.stages[2].condition_input_dim = 5
    cfg.stages[2].policy.action_horizon = 4
    cfg.stages[2].policy.num_inference_steps = 2
    cfg.stages[2].policy.model.cond_dim = 5
    cfg.stages[2].policy.model.diffusion_step_embed_dim = 16
    cfg.stages[2].policy.model.down_dims = [8, 16]
    cfg.stages[2].policy.model.n_groups = 4
    cfg.stages[2].policy.noise_scheduler.num_train_timesteps = 8

    pipeline = instantiate(cfg)

    assert isinstance(pipeline, Pipeline)
    assert [type(stage).__name__ for stage in pipeline.stages] == [
        "ActionTargetBuilder",
        "DiffusionNoisingStage",
        "DiffusionDenoiserStage",
        "DiffusionEpsilonLossStage",
    ]


def test_generic_diffusion_nodes_ignore_unrelated_route_metadata():
    target = torch.randn(2, 4, 6)
    condition = torch.randn(2, 5)
    first = _factorized_stages()
    second = deepcopy(first)

    torch.manual_seed(91)
    first_output = {"target": target, "condition": condition, "embodiment": CHAIN}
    for stage in first[1:]:
        first_output = stage.train()(first_output)

    torch.manual_seed(91)
    second_output = {
        "target": target,
        "condition": condition,
        "embodiment": "arbitrary-unused-metadata",
    }
    for stage in second[1:]:
        second_output = stage.train()(second_output)

    assert torch.equal(
        first_output["loss/diffusion_noise"],
        second_output["loss/diffusion_noise"],
    )


def test_generic_denoiser_explicit_width_ignores_unrelated_mixed_metadata():
    policy = _tiny_policy(CHAIN, 6)
    policy.infer_ac_dims = {CHAIN: 97, USOCKET: 4}
    stage = DiffusionDenoiserStage(
        policy=policy,
        action_horizon=4,
        action_dim=6,
        condition_input_dim=5,
    ).eval()

    output = stage({"condition": torch.randn(2, 5)})

    assert output["pred_action"].shape == (2, 4, 6)


def test_generic_denoiser_uses_singular_policy_state_prefix():
    stage = DiffusionDenoiserStage(
        policy=_tiny_policy(CHAIN, 6),
        action_horizon=4,
        action_dim=6,
        condition_input_dim=5,
    )
    keys = set(stage.state_dict())
    assert keys
    assert all(key.startswith("policy.") for key in keys)
