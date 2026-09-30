from types import SimpleNamespace
from pathlib import Path

import pytest
import torch
from torch import nn
from hydra import compose, initialize_config_dir

from egomimic.pipeline import stages_action_flow
from egomimic.pipeline.action_flow_av_norm import (
    AVOnlyLayerNormCallback,
    differentiable_layer_norm,
    install_av_only,
)
from egomimic.pipeline.stages_action_flow import RoutedContentDecoderStage
from egomimic.pipeline.stages_action_flow import (
    ActionFlowObjectiveStage,
    ConditionalVelocityStage,
    LatentBridgeStage,
    RoutedContentEncoderStage,
)


class _TinyDecoder(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.norms = nn.ModuleList(nn.LayerNorm(2) for _ in range(25))
        self.projection = nn.Linear(2, output_dim)

    def forward(self, value):
        # Residuals avoid the artificial collapse caused by stacking 25 tiny
        # width-two norms directly. Every norm still participates in the JVP.
        for norm in self.norms:
            value = value + 0.01 * norm(value)
        return self.projection(value)


def _decoder(output_dim):
    return _TinyDecoder(output_dim)


def _batch(route):
    return {
        "embodiment": torch.full((2,), route),
        "action_flow/clean_latent": torch.randn(2, 3, 2, requires_grad=True),
        "action_flow/state": torch.randn(2, 3, 2, requires_grad=True),
        "action_flow/velocity_residual": torch.randn(2, 3, 2, requires_grad=True),
    }


def test_routed_av_norm_only_touches_selected_jvp_and_restores_everything():
    us, chain = _decoder(4), _decoder(6)
    stage = RoutedContentDecoderStage(
        decoders={"us": us, "chain": chain}, route_key="embodiment",
        route_aliases={19: "us", 20: "chain"},
    )
    original_jvp = stages_action_flow.jvp
    original_forwards = [m.forward for m in (*us.modules(), *chain.modules())
                         if isinstance(m, nn.LayerNorm)]
    install_av_only(stage)

    for route, selected, other, action_dim in (
        (19, us, chain, 4), (20, chain, us, 6),
    ):
        batch = _batch(route)
        stage._forward_train(batch)
        assert batch[stage.reconstruction_key].shape == (2, 3, action_dim)
        residual = batch[stage.decoded_residual_key]
        assert residual.shape == (2, 3, action_dim)
        selected_parameters = tuple(selected.parameters())
        other_parameters = tuple(other.parameters())
        grads = torch.autograd.grad(
            residual.square().mean(), selected_parameters + other_parameters,
            allow_unused=True,
        )
        active = grads[:len(selected_parameters)]
        inactive = grads[len(selected_parameters):]
        # A synthetic stack of LayerNorms can cancel some affine directions.
        # Require a real, finite AV gradient on the selected route, not a
        # gradient for every mathematically unused fixture parameter.
        assert any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0
                   for g in active)
        assert all(g is None or torch.isfinite(g).all() for g in active)
        assert all(g is None for g in inactive)
        assert stages_action_flow.jvp is original_jvp
        current = [m.forward for m in (*us.modules(), *chain.modules())
                   if isinstance(m, nn.LayerNorm)]
        assert current == original_forwards
        assert all(m.forward.__func__ is not differentiable_layer_norm
                   for m in (*us.modules(), *chain.modules())
                   if isinstance(m, nn.LayerNorm))

    with pytest.raises(ValueError, match="homogeneous"):
        bad = _batch(19)
        bad["embodiment"] = torch.tensor([19, 20])
        stage._forward_train(bad)
    assert stages_action_flow.jvp is original_jvp



def test_routed_av_norm_restores_on_jvp_failure(monkeypatch):
    us, chain = _decoder(4), _decoder(6)
    stage = RoutedContentDecoderStage(
        decoders={"us": us, "chain": chain}, route_key="embodiment",
        route_aliases={19: "us", 20: "chain"},
    )
    original_forwards = [m.forward for m in us.modules() if isinstance(m, nn.LayerNorm)]

    def fail(*args, **kwargs):
        raise RuntimeError("injected JVP failure")

    original_jvp = stages_action_flow.jvp
    monkeypatch.setattr(stages_action_flow, "jvp", fail)
    install_av_only(stage)
    with pytest.raises(RuntimeError, match="injected JVP failure"):
        stage._forward_train(_batch(19))
    assert stages_action_flow.jvp is fail
    assert [m.forward for m in us.modules() if isinstance(m, nn.LayerNorm)] == original_forwards
    monkeypatch.setattr(stages_action_flow, "jvp", original_jvp)


def test_callback_installs_only_for_one_routed_stage():
    stage = RoutedContentDecoderStage(
        decoders={"us": _decoder(4), "chain": _decoder(6)},
        route_key="embodiment", route_aliases={19: "us", 20: "chain"},
    )
    module = SimpleNamespace(model=SimpleNamespace(pipeline=SimpleNamespace(stages=[stage])))
    callback = AVOnlyLayerNormCallback()
    callback.setup(None, module, "validate")
    assert not getattr(stage, "_av_norm_fix_installed", False)
    callback.setup(None, module, "fit")
    assert stage._av_norm_fix_installed
    with pytest.raises(AssertionError, match="already installed"):
        callback.setup(None, module, "fit")


def test_cotrain_av_norm_is_opt_in_and_hydra_reachable(monkeypatch):
    monkeypatch.setenv("PUSHSHAPES_USOCKET_ROOT", "/verified/usocket")
    monkeypatch.setenv("PUSHSHAPES_CHAIN_GRIPPER_ROOT", "/verified/chain")
    config_dir = Path(__file__).parents[1] / "egomimic" / "hydra_configs"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir.resolve())):
        cfg = compose(
            config_name="train_zarr_cartesian",
            overrides=[
                "+experiment=pusht/action_flow_cotrain_uc_latent_fm_sg_unite_h384d12h12_sum14_cfg4_val10k_s42",
                "callbacks=action_flow_avln_cotrain",
                "++paths.root_dir=.",
            ],
        )
    assert cfg.callbacks.av_only_layer_norm._target_.endswith(
        "action_flow_av_norm.AVOnlyLayerNormCallback"
    )
    assert cfg.callbacks.av_only_layer_norm.require_routed is True
    assert cfg.callbacks.model_checkpoint.save_top_k == -1


class _TinyField(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(2, 2)
        self.condition = nn.Linear(5, 2)

    def forward(self, value, time, condition, *, condition_drop_mask=None):
        assert condition_drop_mask is not None
        return (
            self.projection(value)
            + self.condition(condition)[:, None, :]
            + time[:, None, None]
        )


def test_joint_optimizer_step_reaches_shared_field_and_selected_private_codecs():
    torch.manual_seed(42)
    encoders = {"us": nn.Linear(4, 2), "chain": nn.Linear(6, 2)}
    decoders = {"us": _decoder(4), "chain": _decoder(6)}
    field = _TinyField()
    encoder = RoutedContentEncoderStage(
        encoders=encoders, route_key="embodiment",
        route_aliases={19: "us", 20: "chain"},
    )
    decoder = RoutedContentDecoderStage(
        decoders=decoders, route_key="embodiment",
        route_aliases={19: "us", 20: "chain"},
    )
    install_av_only(decoder)
    stages = (
        encoder,
        LatentBridgeStage(samples_per_content=2, condition_dropout_probability=0.0),
        ConditionalVelocityStage(field, num_inference_steps=3),
        decoder,
        ActionFlowObjectiveStage(),
    )
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(field.parameters()) + list(decoder.parameters()),
        lr=1e-3,
    )
    for route, selected, other, action_dim in (
        (19, "us", "chain", 4), (20, "chain", "us", 6),
    ):
        optimizer.zero_grad(set_to_none=True)
        output = {
            "embodiment": torch.full((2,), route),
            "target": torch.randn(2, 3, action_dim),
            "condition": torch.randn(2, 5),
            "sampler/noise": torch.randn(2, 3, 2),
        }
        for stage in stages:
            output = stage(output)
        loss = output["loss/action_flow"]
        assert torch.isfinite(loss)
        assert all(torch.isfinite(output[key]) for key in (
            "log/action_flow_fm",
            "log/action_flow_reconstruction",
            "log/action_flow_action_velocity",
        ))
        loss.backward()
        for module in (encoder.encoder[selected], field, decoder.decoder[selected]):
            assert any(p.grad is not None and torch.isfinite(p.grad).all()
                       and p.grad.abs().sum() > 0 for p in module.parameters())
        for module in (encoder.encoder[other], decoder.decoder[other]):
            assert all(p.grad is None for p in module.parameters())
        optimizer.step()
