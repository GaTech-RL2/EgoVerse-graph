import copy

import pytest
import torch

from egomimic.models.mot_action_codec import make_mot_codecs
from egomimic.pipeline.stages_action_flow import (
    RoutedContentDecoderStage,
    RoutedContentEncoderStage,
)


def codecs(role):
    return make_mot_codecs(
        action_dims={"gripper": 7, "sharpa": 69},
        role=role,
        horizon=4,
        tokens=2,
        latent_dim=4,
        width=16,
        depth=2,
        heads=2,
    )


@pytest.mark.parametrize("role", ["encoder", "decoder"])
def test_two_embodiments_share_latent_parameters_but_not_action_parameters(role):
    bank = codecs(role)
    assert bank["gripper"].shared is bank["sharpa"].shared
    assert bank["gripper"].blocks[0].latent is bank["sharpa"].blocks[0].latent
    assert bank["gripper"].blocks[0].action is not bank["sharpa"].blocks[0].action
    for route, dim in [("gripper", 7), ("sharpa", 69)]:
        bank.zero_grad(set_to_none=True)
        value = torch.randn(2, 4, dim) if role == "encoder" else torch.randn(2, 2, 4)
        result = bank[route](value)
        assert result.shape == ((2, 2, 4) if role == "encoder" else (2, 4, dim))
        result.square().mean().backward()
        shared_gradient = bank[route].shared.layers[0].qkv.weight.grad
        assert torch.isfinite(shared_gradient).all() and shared_gradient.abs().sum() > 0
        other = "sharpa" if route == "gripper" else "gripper"
        assert bank[other].blocks[0].action.qkv.weight.grad is None
    assert len(list(bank.parameters())) == len({id(p) for p in bank.parameters()})
    replacement = codecs(role)
    replacement.load_state_dict(copy.deepcopy(bank.state_dict()), strict=True)
    assert replacement["gripper"].shared is replacement["sharpa"].shared
    for name, parameter in bank.named_parameters():
        assert torch.equal(parameter, dict(replacement.named_parameters())[name])


def test_actual_routed_stages_and_decoder_higher_order_jvp():
    encoder = RoutedContentEncoderStage(encoders=codecs("encoder"), route_key="route")
    decoder = RoutedContentDecoderStage(decoders=codecs("decoder"), route_key="route")
    assert not (
        {id(p) for p in encoder.parameters()} & {id(p) for p in decoder.parameters()}
    )
    for route, dim in [("gripper", 7), ("sharpa", 69)]:
        batch = encoder({"route": route, "target": torch.randn(2, 4, dim)})
        latent = batch["action_flow/clean_latent"]
        selected = decoder.decoder_for(batch)
        output, tangent = torch.func.jvp(
            selected, (latent,), (torch.randn_like(latent),)
        )
        assert output.shape == tangent.shape == (2, 4, dim)
        assert torch.isfinite(tangent).all()
        (output.square().mean() + tangent.square().mean()).backward()
        assert selected.shared.layers[0].qkv.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("role", ["encoder", "decoder"])
@pytest.mark.parametrize("depth", [1, 2])
def test_every_selected_codec_parameter_receives_finite_gradient(role, depth):
    torch.manual_seed(42)
    bank = make_mot_codecs(
        action_dims={"gripper": 7, "sharpa": 69},
        role=role,
        horizon=4,
        tokens=2,
        latent_dim=4,
        width=16,
        depth=depth,
        heads=2,
    )
    for route, dim in [("gripper", 7), ("sharpa", 69)]:
        bank.zero_grad(set_to_none=True)
        value = torch.randn(2, 4, dim) if role == "encoder" else torch.randn(2, 2, 4)
        output = bank[route](value)
        (output - torch.randn_like(output)).square().mean().backward()
        for name, parameter in bank[route].named_parameters():
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
            assert parameter.grad.abs().sum() > 0, name
        inactive = "sharpa" if route == "gripper" else "gripper"
        active_ids = {id(p) for p in bank[route].parameters()}
        for name, parameter in bank[inactive].named_parameters():
            if id(parameter) not in active_ids:
                assert parameter.grad is None, name
    context = bank["gripper"].blocks[-1]
    context = context.action if role == "encoder" else context.latent
    assert hasattr(context, "kv")
    assert not hasattr(context, "qkv") and not hasattr(context, "ffn")


@pytest.mark.parametrize(
    "kwargs", [{"role": "invalid"}, {"action_dims": {"only": 7}}, {"heads": 3}]
)
def test_invalid_configuration_rejected(kwargs):
    options = dict(
        action_dims={"gripper": 7, "sharpa": 69},
        role="encoder",
        horizon=4,
        tokens=2,
        latent_dim=4,
        width=16,
        depth=2,
        heads=2,
    )
    options.update(kwargs)
    with pytest.raises(ValueError):
        make_mot_codecs(**options)


def test_mot_native_half_checkpoint_scope_and_diagnostic_capture():
    from types import SimpleNamespace

    from egomimic.eval.action_flow_diagnostic_forward import _block_outputs
    from egomimic.models import unite_dit
    from egomimic.models.action_flow_unite import UniteActionFlowVelocityField
    from egomimic.models.unite_dit import UniteDiTBackbone
    from egomimic.pipeline.stages_action_flow import ConditionalVelocityStage
    from egomimic.utils.libero_dit_half import HalfCheckpointScope, resolve_backbones

    bank = codecs("encoder")
    for codec in bank.values():
        codec.gradient_checkpointing = True
    encoder = RoutedContentEncoderStage(encoders=bank, route_key="route")
    backbone = UniteDiTBackbone(
        input_dim=4,
        output_dim=4,
        horizon=2,
        condition_dim=5,
        max_condition_tokens=1,
        max_content_tokens=4,
        hidden_dim=16,
        depth=2,
        num_heads=2,
        in_context_start=None,
        in_context_len=0,
        gradient_checkpointing=True,
    )
    velocity = ConditionalVelocityStage(
        UniteActionFlowVelocityField(backbone, 4, 4, 2, 5)
    )
    module = SimpleNamespace(
        model=SimpleNamespace(pipeline=SimpleNamespace(stages=[encoder, velocity]))
    )
    scope = HalfCheckpointScope(
        unite_dit, resolve_backbones(module), expected_depth=2
    ).install()
    try:
        for route, dim in [("gripper", 7), ("sharpa", 69)]:
            result = encoder({"route": route, "target": torch.randn(2, 4, dim)})
            field = backbone(
                result["action_flow/clean_latent"],
                torch.ones(2),
                condition=torch.randn(2, 1, 5),
            )
            field.square().mean().backward()
        assert all(count > 0 for count in scope.counts.values())
    finally:
        scope.restore()
    with torch.no_grad():
        result, captured, indexes = _block_outputs(
            bank["gripper"],
            lambda: bank["gripper"](torch.randn(2, 4, 7)),
            label="encoder",
            required=True,
        )
    assert captured.shape == (2, 2, 2, 16)
    assert indexes.tolist() == [0, 1]
    assert result.shape == (2, 2, 4)


def test_mot_av_callback_and_original_flow_objective_reach_shared_banks():
    from torch import nn

    from egomimic.pipeline.action_flow_av_norm import install_av_only
    from egomimic.pipeline.stages_action_flow import (
        ActionFlowObjectiveStage,
        ConditionalVelocityStage,
        LatentBridgeStage,
    )
    from egomimic.utils.unite_optim import ReleasedUniteCompositeOptimizer

    class Field(nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = nn.Linear(4, 4)
            self.condition = nn.Linear(5, 4)

        def forward(self, value, time, condition, *, condition_drop_mask=None):
            return (
                self.projection(value)
                + self.condition(condition)[:, None, :]
                + time[:, None, None]
            )

    encoder = RoutedContentEncoderStage(encoders=codecs("encoder"), route_key="route")
    decoder = RoutedContentDecoderStage(decoders=codecs("decoder"), route_key="route")
    install_av_only(decoder, expected_norms_per_decoder=8)
    field = Field()
    owners = nn.ModuleDict(dict(encoder=encoder, decoder=decoder, field=field))
    optimizer = ReleasedUniteCompositeOptimizer(
        named_params=tuple(owners.named_parameters())
    )
    adamw = optimizer.group_manifest["adamw_parameter_names"]
    muon = optimizer.group_manifest["muon_parameter_names"]
    assert any("content_projection.weight" in name for name in adamw)
    assert any("shared.proj_d.weight" in name for name in adamw)
    assert not any("content_projection.weight" in name for name in muon)
    params = [p for group in optimizer.param_groups for p in group["params"]]
    assert len(params) == len({id(p) for p in params}) == len(list(owners.parameters()))
    stages = [
        encoder,
        LatentBridgeStage(samples_per_content=2, condition_dropout_probability=0),
        ConditionalVelocityStage(field, num_inference_steps=3),
        decoder,
        ActionFlowObjectiveStage(),
    ]
    for route, dim in [("gripper", 7), ("sharpa", 69)]:
        optimizer.zero_grad(set_to_none=True)
        batch = {
            "route": route,
            "target": torch.randn(2, 4, dim),
            "condition": torch.randn(2, 5),
            "sampler/noise": torch.randn(2, 2, 4),
        }
        for stage in stages:
            batch = stage(batch)
        loss = batch["loss/action_flow"]
        assert torch.isfinite(loss)
        loss.backward()
        for owner in [
            encoder.encoder[route].shared,
            decoder.decoder[route].shared,
            field,
        ]:
            assert any(
                p.grad is not None
                and torch.isfinite(p.grad).all()
                and p.grad.abs().sum() > 0
                for p in owner.parameters()
            )
        optimizer.step()


def test_existing_alias_safe_ema_verifier_covers_mot_shared_aliases():
    from torch import nn

    from scripts.train.verify_libero_native_action_flow_smoke import (
        validate_joint_ema_reload,
    )

    class Wrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.nets = nn.ModuleDict(
                dict(encoder=codecs("encoder"), decoder=codecs("decoder"))
            )

    module = Wrapper()
    config = dict(
        decay=0.9978,
        use_warmup=False,
        update_after_step=0,
        inv_gamma=1.0,
        power=2 / 3,
        min_decay=0.0,
    )
    payload = dict(
        global_step=8,
        ema_num_updates=8,
        ema_config=config,
        ema_state_dict={
            name: torch.full_like(p, 0.125) for name, p in module.named_parameters()
        },
    )
    result = validate_joint_ema_reload(module, payload, {"callbacks": {"ema": config}})
    assert result["alias_safe_strict_reload"]
    assert result["state_parameter_aliases"] > result["canonical_parameters"]
    for p in module.parameters():
        assert torch.equal(p, torch.full_like(p, 0.125))


@pytest.mark.parametrize("phase", ["normalization", "preflight", "smoke", "full"])
def test_actual_mot_phase_factory_and_profile_are_bound(tmp_path, monkeypatch, phase):
    from pathlib import Path

    import hydra
    from hydra.core.hydra_config import HydraConfig
    from omegaconf import OmegaConf

    from egomimic.benchmarks.libero.native_launch_profiles import (
        SPATIAL_MOT_JOINT_PROFILE,
        joint_phase_argv,
        joint_profile_for_config,
    )

    root = Path(__file__).parents[1]
    for name in [
        "LIBERO_ORIGINAL_SPATIAL_ROOT",
        "LIBERO_SHARPA_SPATIAL_ROOT",
        "LIBERO_DIAGNOSTIC_SEED_BANK",
    ]:
        monkeypatch.setenv(name, "/wrong/unselected")
    args = dict(
        source_root=str(root),
        original_root=str(tmp_path / "original"),
        sharpa_root=str(tmp_path / "sharpa"),
        output_root=str(tmp_path / phase),
        context_path=str(tmp_path / "context.json"),
        split_path=str(tmp_path / "split.json"),
        split_sha256="a" * 64,
        normalization_sha256=None if phase == "normalization" else "b" * 64,
        source_commit="c" * 40,
        diagnostic_seed_bank=str(tmp_path / "seeds.json"),
        wandb_entity="test",
        wandb_project="test",
        wandb_run_id="mot-" + phase,
        profile_name=SPATIAL_MOT_JOINT_PROFILE.name,
    )
    argv = joint_phase_argv(phase=phase, **args)
    with hydra.initialize_config_dir(
        version_base="1.2", config_dir=str(root / "egomimic/hydra_configs")
    ):
        cfg = hydra.compose(
            config_name="train_zarr_cartesian",
            return_hydra_config=True,
            overrides=[*argv[1:], "hydra/launcher=basic"],
        )
        cfg.hydra.runtime.output_dir = str(tmp_path)
        HydraConfig.instance().set_config(cfg)
        c = OmegaConf.to_container(
            OmegaConf.masked_copy(cfg, [k for k in cfg if k != "hydra"]), resolve=True
        )
    assert joint_profile_for_config(c) == SPATIAL_MOT_JOINT_PROFILE
    if phase == "preflight":
        from scripts.train.libero_native_launch_contract import construct_joint_model

        with pytest.raises(ValueError, match="source/config identity"):
            construct_joint_model(c, tmp_path / "unopened.yaml", source_head="d" * 40)
        gpu = copy.deepcopy(c)
        gpu["trainer"]["accelerator"] = "gpu"
        with pytest.raises(ValueError, match="CPU preflight"):
            construct_joint_model(gpu, tmp_path / "unopened.yaml", source_head="c" * 40)

    assert c["callbacks"]["av_only_layer_norm"]["expected_norms_per_decoder"] == 48
    changed = copy.deepcopy(c)
    changed["model"]["pipeline"]["stages"][4]["encoders"]["action_dims"][
        "libero_panda_sharpa"
    ] = 7
    with pytest.raises(ValueError, match="MoT source widths"):
        joint_profile_for_config(changed)
    changed = copy.deepcopy(c)
    changed["model"]["pipeline"]["stages"][7]["decoders"]["condition_dim"] = 276
    with pytest.raises(ValueError, match="context-free"):
        joint_profile_for_config(changed)


def test_checkpoint_owner_guard_allows_only_declared_mot_aliases():
    from types import SimpleNamespace

    from torch import nn

    from scripts.train.verify_action_flow_training_smoke import _joint_checkpoint_owners

    def bank(role):
        return make_mot_codecs(
            action_dims={"libero_panda": 7, "libero_panda_sharpa": 69},
            role=role,
            horizon=4,
            tokens=2,
            latent_dim=4,
            width=16,
            depth=2,
            heads=2,
        )

    enc, dec = bank("encoder"), bank("decoder")
    stages = [None] * 9
    stages[4] = RoutedContentEncoderStage(encoders=enc)
    stages[7] = RoutedContentDecoderStage(decoders=dec, route_key="embodiment")
    stages[6] = SimpleNamespace(field=nn.Linear(4, 4))
    assert len(_joint_checkpoint_owners(stages)) == 5
    enc["libero_panda_sharpa"].blocks[0].action = enc["libero_panda"].blocks[0].action
    with pytest.raises(RuntimeError, match="escape shared latent"):
        _joint_checkpoint_owners(stages)


def test_phase_cli_rejects_wrong_experiment_before_model_or_composition(
    tmp_path, monkeypatch
):
    import hashlib
    import json
    import sys

    from scripts.train import libero_native_launch_contract as adapter

    path = tmp_path / "inputs.json"
    path.write_text(
        json.dumps({"profile_name": "action_flow_libero_spatial_sharpa_mot_s42"})
    )
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("SLURM_STEP_ID", "0")
    monkeypatch.setenv(
        "AF_EXPERIMENT",
        "sim/libero/action_flow_libero_spatial_sharpa_gripper_keypoints_s42",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "adapter",
            "--action",
            "joint-compose",
            "--repo",
            str(tmp_path),
            "--output",
            str(tmp_path),
            "--phase",
            "preflight",
            "--phase-inputs",
            str(path),
            "--expected-phase-inputs-sha256",
            hashlib.sha256(path.read_bytes()).hexdigest(),
        ],
    )
    with pytest.raises(ValueError, match="experiment/environment mismatch"):
        adapter.main()
