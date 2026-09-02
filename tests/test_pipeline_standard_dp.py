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
    DiffusionEpsilonLossStage,
    MultiDomainDiffusionDenoiserStage,
    MultiDomainDiffusionNoisingStage,
    MultiDomainDiffusionPolicyStage,
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
        MultiDomainDiffusionNoisingStage(
            noise_schedulers={domain: _tiny_scheduler()},
            action_dims={domain: action_dim},
            action_horizon=4,
        ),
        MultiDomainDiffusionDenoiserStage(
            policies={domain: _tiny_policy(domain, action_dim)},
            action_horizon=4,
            condition_input_dim=5,
        ),
        DiffusionEpsilonLossStage(),
    )


def test_factorized_training_uses_epsilon_loss_and_backpropagates():
    stages = _factorized_stages()
    output = {
        "condition": torch.randn(2, 5),
        "embodiment": CHAIN,
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


def test_factorized_training_is_rng_equivalent_to_compatibility_stage():
    policy = _tiny_policy(CHAIN, 6)
    legacy_policy = deepcopy(policy)
    target = torch.randn(2, 4, 6)
    condition = torch.randn(2, 5)

    legacy = MultiDomainDiffusionPolicyStage(
        policies={CHAIN: legacy_policy},
        action_horizon=4,
        condition_input_dim=5,
    ).train()
    torch.manual_seed(123)
    legacy_output = legacy(
        {"condition": condition, "embodiment": CHAIN, "target": target}
    )

    stages = (
        MultiDomainDiffusionNoisingStage(
            noise_schedulers={CHAIN: _tiny_scheduler()},
            action_dims={CHAIN: 6},
            action_horizon=4,
        ),
        MultiDomainDiffusionDenoiserStage(
            policies={CHAIN: policy},
            action_horizon=4,
            condition_input_dim=5,
        ),
        DiffusionEpsilonLossStage(),
    )
    torch.manual_seed(123)
    output = {"condition": condition, "embodiment": CHAIN, "target": target}
    for stage in stages:
        output = stage.train()(output)

    assert torch.equal(
        output["loss/diffusion_noise"], legacy_output["loss/diffusion_noise"]
    )


@pytest.mark.parametrize("domain,width", [(CHAIN, 6), (USOCKET, 4)])
def test_denoiser_rollout_samples_the_selected_domain_width(domain, width):
    stage = MultiDomainDiffusionDenoiserStage(
        policies={domain: _tiny_policy(domain, width)},
        action_horizon=4,
        condition_input_dim=5,
    ).eval()
    with torch.inference_mode():
        output = stage({"condition": torch.randn(2, 5), "embodiment": domain})

    assert output["pred_action"].shape == (2, 4, width)
    assert torch.isfinite(output["pred_action"]).all()
    assert output["log/diffusion_inference_steps"] == 2.0


def test_nodes_reject_wrong_objective_width_and_scheduler_pair():
    with pytest.raises(ValueError, match="epsilon noise objective"):
        MultiDomainDiffusionNoisingStage(
            noise_schedulers={CHAIN: _tiny_scheduler("sample")},
            action_dims={CHAIN: 6},
            action_horizon=4,
        )

    target, noising, denoiser, _ = _factorized_stages(USOCKET, 4)
    with pytest.raises(ValueError, match="must be"):
        noising(
            target(
                {
                    "embodiment": USOCKET,
                    "actions": torch.randn(2, 4, 6),
                }
            )
        )

    noising = MultiDomainDiffusionNoisingStage(
        noise_schedulers={CHAIN: _tiny_scheduler(timesteps=7)},
        action_dims={CHAIN: 6},
        action_horizon=4,
    )
    mismatched = noising({"embodiment": CHAIN, "target": torch.randn(2, 4, 6)})
    with pytest.raises(ValueError, match="schedulers differ"):
        denoiser = MultiDomainDiffusionDenoiserStage(
            policies={CHAIN: _tiny_policy(CHAIN, 6)},
            action_horizon=4,
            condition_input_dim=5,
        )
        denoiser({**mismatched, "condition": torch.randn(2, 5)})


def test_diffusion_pipeline_contracts_plan_factorized_train_and_rollout():
    encoder = FusedObsEncoder(torch.nn.Identity(), n_obs_steps=1)
    target, noising, denoiser, loss = _factorized_stages()
    pipeline = Pipeline([encoder, target, noising, denoiser, loss])

    assert encoder.contract("train") == (
        ("obs/*", "embodiment"),
        ("condition",),
    )
    assert denoiser.contract("rollout") == (
        ("condition", "embodiment"),
        ("pred_action", "log/*"),
    )
    train, train_excluded = pipeline.plan(
        ["obs/state", "embodiment", "actions"], mode="train"
    )
    rollout, rollout_excluded = pipeline.plan(
        ["obs/state", "embodiment"], mode="rollout"
    )
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
    cfg.stages[1].action_dims.eva_bimanual = 14
    cfg.stages[1].noise_schedulers.eva_bimanual.num_train_timesteps = 8
    cfg.stages[2].action_horizon = 4
    cfg.stages[2].condition_input_dim = 5
    cfg.stages[2].policies.eva_bimanual.action_horizon = 4
    cfg.stages[2].policies.eva_bimanual.num_inference_steps = 2
    cfg.stages[2].policies.eva_bimanual.model.cond_dim = 5
    cfg.stages[2].policies.eva_bimanual.model.diffusion_step_embed_dim = 16
    cfg.stages[2].policies.eva_bimanual.model.down_dims = [8, 16]
    cfg.stages[2].policies.eva_bimanual.model.n_groups = 4
    cfg.stages[2].policies.eva_bimanual.noise_scheduler.num_train_timesteps = 8

    pipeline = instantiate(cfg)

    assert isinstance(pipeline, Pipeline)
    assert [type(stage).__name__ for stage in pipeline.stages] == [
        "ActionTargetBuilder",
        "MultiDomainDiffusionNoisingStage",
        "MultiDomainDiffusionDenoiserStage",
        "DiffusionEpsilonLossStage",
    ]
