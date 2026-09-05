from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from egomimic.models.unite_action_decoder import UniteActionDecoder
from egomimic.pipeline.stages_unite_released import (
    ReleasedRecipeUniteLatentPolicy,
    ReleasedRecipeUniteObjective,
)
from egomimic.pipeline.stages_unite_separate import (
    build_configurable_unite_generative_encoder,
)
from egomimic.pipeline.unite_action_velocity import decoder_action_velocity_loss
from egomimic.pl_utils.pl_model_unite_released import ReleasedUniteModelWrapper


DOMAIN = "pushshapes_sim_u_socket"
CONFIG_DIR = Path(__file__).parents[1] / "egomimic/hydra_configs"


def _policy(
    action_velocity_samples: int,
    *,
    gradient_checkpointing: bool = False,
) -> ReleasedRecipeUniteLatentPolicy:
    backbone = {
        "_target_": "egomimic.models.unite_dit.UniteDiTBackbone",
        "input_dim": 4,
        "output_dim": 4,
        "horizon": 2,
        "condition_dim": 4,
        "max_condition_tokens": 1,
        "max_content_tokens": 3,
        "hidden_dim": 8,
        "depth": 1,
        "num_heads": 2,
        "mlp_ratio": 2.0,
        "time_fourier_dim": 8,
        "context_adaln_summary": False,
        "in_context_start": 0,
        "in_context_len": 4,
        "trainable_position_embeddings": False,
        "trainable_register_position_embeddings": True,
        "trainable_content_position_embeddings": False,
        "trainable_condition_position_embeddings": False,
        "trainable_in_context_position_embeddings": True,
        "gradient_checkpointing": gradient_checkpointing,
    }
    encoder = build_configurable_unite_generative_encoder(
        backbone_config=backbone,
        share_encoder_denoiser=True,
        action_dims={DOMAIN: 2},
        condition_input_dim=4,
        latent_dim=4,
        num_latent_tokens=2,
        condition_dim=4,
        denoiser_hidden_dim=8,
        gradient_checkpointing=gradient_checkpointing,
        in_context_start=0,
        in_context_len=4,
    )
    decoder = UniteActionDecoder(
        latent_dim=4,
        action_dim=2,
        num_latent_tokens=2,
        action_horizon=3,
        hidden_dim=8,
        depth=1,
        num_heads=2,
        mlp_ratio=2.0,
        gradient_checkpointing=gradient_checkpointing,
    )
    return ReleasedRecipeUniteLatentPolicy(
        generative_encoder=encoder,
        decoders={DOMAIN: decoder},
        flow_steps_per_reconstruction=2,
        flow_mini_batch=2,
        action_velocity_samples_per_reconstruction=action_velocity_samples,
        reconstruction_noising_probability=0.0,
        condition_dropout_probability=0.0,
    )


def _batch() -> dict:
    return {
        "target": torch.randn(2, 3, 2),
        "condition": torch.randn(2, 4),
        "sampler/noise": torch.randn(2, 2, 4),
        "embodiment": DOMAIN,
    }


def _activate_initially_zero_latent_projection(policy) -> None:
    projection = policy.generative_encoder.denoising_module.proj_d
    nn.init.normal_(projection.weight, std=0.02)


def test_decoder_jvp_matches_linear_action_metric():
    decoder = nn.Linear(3, 2, bias=False)
    decoder.weight.data.copy_(torch.tensor([[1.0, 2.0, 0.0], [0.0, -1.0, 3.0]]))
    latent = torch.randn(4, 3, requires_grad=True)
    residual = torch.randn(4, 3, requires_grad=True)
    expected = (residual @ decoder.weight.T).square().mean()
    actual = decoder_action_velocity_loss(decoder, latent.unsqueeze(1), residual.unsqueeze(1))
    torch.testing.assert_close(actual, expected)


def test_baseline_executes_no_decoder_jvp_and_keeps_zero_av_loss(monkeypatch):
    policy = _policy(0).train()
    monkeypatch.setattr(
        "egomimic.pipeline.stages_unite_released.decoder_action_velocity_loss",
        lambda *_args: pytest.fail("baseline UNITE executed a decoder JVP"),
    )
    output = ReleasedRecipeUniteObjective()(policy(_batch()))
    assert output["unite/action_velocity_sample_count"] == 0
    assert output["loss/unite_action_velocity"] == 0.0


def test_unite_av_is_finite_and_reaches_encoder_denoiser_and_decoder():
    torch.manual_seed(17)
    policy = _policy(1).train()
    _activate_initially_zero_latent_projection(policy)
    output = ReleasedRecipeUniteObjective(action_velocity_weight=1.0)(
        policy(_batch())
    )
    loss = output["loss/unite_action_velocity"]
    assert loss.ndim == 0 and bool(torch.isfinite(loss)) and float(loss) > 0.0
    loss.backward()

    encoder = policy.generative_encoder
    decoder = policy.action_decoder.decoder_for(DOMAIN)
    assert any(
        parameter.grad is not None
        for parameter in encoder.action_context_projections[DOMAIN].parameters()
    )
    assert any(
        parameter.grad is not None
        for parameter in encoder.denoising_module.parameters()
    )
    assert any(parameter.grad is not None for parameter in decoder.parameters())


def test_unite_av_supports_decoder_gradient_checkpointing():
    policy = _policy(1, gradient_checkpointing=True).train()
    _activate_initially_zero_latent_projection(policy)
    output = ReleasedRecipeUniteObjective(action_velocity_weight=1.0)(policy(_batch()))
    loss = output["loss/unite_action_velocity"]
    assert bool(torch.isfinite(loss))
    loss.backward()
    decoder = policy.action_decoder.decoder_for(DOMAIN)
    assert any(parameter.grad is not None for parameter in decoder.parameters())


def test_objective_and_policy_settings_fail_closed():
    with pytest.raises(ValueError, match="samples"):
        _policy(3)
    policy = _policy(0).train()
    with pytest.raises(RuntimeError, match="sampled no AV bridges"):
        ReleasedRecipeUniteObjective(action_velocity_weight=1.0)(policy(_batch()))


def test_unite_av_profile_composes_without_changing_architecture():
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR.resolve())):
        baseline = compose(
            config_name="train_zarr_cartesian",
            overrides=[
                "model=bf/us_unite_register_shared_nt8_s42",
                "+experiment=pusht/unite_usocket_register_sweep_val01_h16",
                "++paths.root_dir=.",
            ],
        )
        unite_av = compose(
            config_name="train_zarr_cartesian",
            overrides=[
                "model=bf/us_unite_av_register_shared_nt8_s42",
                "+experiment=pusht/unite_usocket_register_sweep_val01_h16",
                "++paths.root_dir=.",
            ],
        )
    OmegaConf.resolve(baseline.model)
    OmegaConf.resolve(unite_av.model)
    assert baseline.model.architecture_id == unite_av.model.architecture_id
    assert baseline.model.objective_id == "unite_baseline_v1"
    assert unite_av.model.objective_id == "unite_action_velocity_v1"
    assert (
        baseline.model.pipeline.stages[4].action_velocity_samples_per_reconstruction == 0
    )
    assert (
        unite_av.model.pipeline.stages[4].action_velocity_samples_per_reconstruction == 4
    )
    assert baseline.model.pipeline.stages[5].action_velocity_weight == 0.0
    assert unite_av.model.pipeline.stages[5].action_velocity_weight == 1.0
    assert baseline.model.hidden_dim == unite_av.model.hidden_dim
    assert baseline.model.num_latent_tokens == unite_av.model.num_latent_tokens == 8


def test_unite_av_checkpoint_contract_rejects_baseline_full_resume():
    config = OmegaConf.create(
        {
            "model": {
                "architecture_id": "unite_register_v1",
                "objective_id": "unite_action_velocity_v1",
                "action_velocity_samples_per_reconstruction": 4,
                "action_velocity_weight": 1.0,
            }
        }
    )
    contract = ReleasedUniteModelWrapper._resolve_unite_contract(config)
    wrapper = SimpleNamespace(_unite_contract=contract)
    with pytest.raises(RuntimeError, match="weights-only warm start"):
        ReleasedUniteModelWrapper.on_load_checkpoint(wrapper, {})
    checkpoint = {"unite_contract": dict(contract)}
    ReleasedUniteModelWrapper.on_load_checkpoint(wrapper, checkpoint)
    checkpoint["unite_contract"]["objective_id"] = "unite_baseline_v1"
    with pytest.raises(RuntimeError, match="contract mismatch"):
        ReleasedUniteModelWrapper.on_load_checkpoint(wrapper, checkpoint)
