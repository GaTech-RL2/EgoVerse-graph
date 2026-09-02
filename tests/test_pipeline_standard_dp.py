from pathlib import Path

import pytest
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from egomimic.models.ddim_scheduler import DDIMScheduler
from egomimic.models.denoising_nets import ConditionalUnet1D
from egomimic.models.diffusion_policy import DiffusionPolicy
from egomimic.pipeline.core import Pipeline
from egomimic.pipeline.stages_diffusion import MultiDomainDiffusionPolicyStage
from egomimic.pipeline.stages_sampler import FusedObsEncoder

CHAIN = "chain"
USOCKET = "u_socket"


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
        noise_scheduler=DDIMScheduler(
            num_train_timesteps=8,
            beta_schedule="squaredcos_cap_v2",
            prediction_type=prediction_type,
        ),
        action_horizon=4,
        infer_ac_dims={domain: action_dim},
        num_inference_steps=2,
    )


def _tiny_stage():
    return MultiDomainDiffusionPolicyStage(
        policies={
            CHAIN: _tiny_policy(CHAIN, 6),
            USOCKET: _tiny_policy(USOCKET, 4),
        },
        action_horizon=4,
        condition_input_dim=5,
    )


def test_training_uses_epsilon_loss_and_backpropagates():
    stage = _tiny_stage().train()
    output = stage(
        {
            "condition": torch.randn(2, 5),
            "embodiment": CHAIN,
            "target": torch.randn(2, 4, 6),
        }
    )

    assert stage.objective == "epsilon"
    assert "pred_action" not in output
    assert output["loss/diffusion_noise"].ndim == 0
    assert torch.isfinite(output["loss/diffusion_noise"])
    output["loss/diffusion_noise"].backward()
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in stage.parameters()
    )


@pytest.mark.parametrize("domain,width", [(CHAIN, 6), (USOCKET, 4)])
def test_eval_samples_the_selected_domain_width(domain, width):
    stage = _tiny_stage().eval()
    with torch.inference_mode():
        output = stage({"condition": torch.randn(2, 5), "embodiment": domain})

    assert output["pred_action"].shape == (2, 4, width)
    assert torch.isfinite(output["pred_action"]).all()
    assert output["log/diffusion_inference_steps"] == 2.0


def test_stage_rejects_wrong_objective_and_target_width():
    with pytest.raises(ValueError, match="epsilon noise objective"):
        MultiDomainDiffusionPolicyStage(
            policies={CHAIN: _tiny_policy(CHAIN, 6, prediction_type="sample")},
            action_horizon=4,
            condition_input_dim=5,
        )

    with pytest.raises(ValueError, match="must be"):
        _tiny_stage().train()(
            {
                "condition": torch.randn(2, 5),
                "embodiment": USOCKET,
                "target": torch.randn(2, 4, 6),
            }
        )


def test_diffusion_pipeline_contracts_plan_for_train_and_rollout():
    encoder = FusedObsEncoder(torch.nn.Identity(), n_obs_steps=1)
    diffusion = _tiny_stage()
    pipeline = Pipeline([encoder, diffusion])

    assert encoder.contract("train") == (
        ("obs/*", "embodiment", "actions"),
        ("condition", "target"),
    )
    assert diffusion.contract("rollout") == (
        ("condition", "embodiment"),
        ("pred_action", "log/*"),
    )
    train, train_excluded = pipeline.plan(
        ["obs/state", "embodiment", "actions"], mode="train"
    )
    rollout, rollout_excluded = pipeline.plan(
        ["obs/state", "embodiment"], mode="rollout"
    )
    assert train == [encoder, diffusion]
    assert rollout == [encoder, diffusion]
    assert train_excluded == rollout_excluded == []


def test_hydra_standard_dp_fragment_instantiates():
    path = (
        Path(__file__).parents[1]
        / "egomimic"
        / "hydra_configs"
        / "model"
        / "pipeline_standard_dp_stage.yaml"
    )
    cfg = OmegaConf.load(path)
    cfg.action_horizon = 4
    cfg.condition_input_dim = 5
    cfg.policies.eva_bimanual.action_horizon = 4
    cfg.policies.eva_bimanual.num_inference_steps = 2
    cfg.policies.eva_bimanual.model.cond_dim = 5
    cfg.policies.eva_bimanual.model.diffusion_step_embed_dim = 16
    cfg.policies.eva_bimanual.model.down_dims = [8, 16]
    cfg.policies.eva_bimanual.model.n_groups = 4
    cfg.policies.eva_bimanual.noise_scheduler.num_train_timesteps = 8

    stage = instantiate(cfg)

    assert isinstance(stage, MultiDomainDiffusionPolicyStage)
    assert stage.action_horizon == 4
    assert stage.condition_input_dim == 5
    assert stage.action_dims == {"eva_bimanual": 14}
