"""Regressions for the actual Phoenix 12935484/12935488 verifier failures."""

from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from tests.test_verify_action_flow_training_smoke import MODULE, _checkpoint_payload
from tools.validate_action_flow_config import (
    GRAPH_METHOD,
    STOPGRAD_METHOD,
    compose_experiment,
)

# Exact six-key schedules printed by the failed final verifiers. The jobs'
# training steps exited 0; both experiments intentionally train jointly at 0.
OBSERVED_SCHEDULES = (
    (
        "pusht/action_flow_bc_usocket_latent_fm_sg_recon1_s42",
        STOPGRAD_METHOD,
        {
            "joint_objective_begins_at_global_step": 0,
            "reconstruction_only_optimizer_steps": 0,
            "joint_flow_weight": 1.0,
            "joint_reconstruction_weight": 1.0,
            "joint_action_velocity_weight": 1.0,
            "schema_version": 1,
        },
    ),
    (
        "pusht/action_flow_bc_usocket_graph_section_s42",
        GRAPH_METHOD,
        {
            "joint_objective_begins_at_global_step": 0,
            "reconstruction_only_optimizer_steps": 0,
            "joint_flow_weight": 1.0,
            "joint_reconstruction_weight": 0.0,
            "joint_action_velocity_weight": 1.0,
            "schema_version": 1,
        },
    ),
)


@pytest.mark.parametrize("experiment,method,schedule", OBSERVED_SCHEDULES)
@pytest.mark.parametrize("restored_warmup", (0, 1))
def test_observed_joint_schedule_and_strict_reload(
    tmp_path, monkeypatch, experiment, method, schedule, restored_warmup
):
    config = compose_experiment(experiment, overrides=["hydra/launcher=basic"])
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    payload = _checkpoint_payload()
    payload["action_flow_loss_schedule"] = schedule
    immutable = checkpoint_dir / "epoch-0-step-2.ckpt"
    torch.save(payload, immutable)
    (checkpoint_dir / "last.ckpt").symlink_to(immutable.name)

    class Wrapper:
        reconstruction_only_warmup_steps = restored_warmup
        encoder_e, field_v, decoder_g = object(), object(), object()
        model = SimpleNamespace(pipeline=SimpleNamespace(stages=[]))
        nets = SimpleNamespace(named_parameters=lambda **kwargs: [])

        @classmethod
        def load_from_checkpoint(cls, *args, **kwargs):
            assert kwargs["strict"] is True
            return cls()

        def parameters(self):
            return [SimpleNamespace(numel=lambda: MODULE.EXPECTED_PARAMETER_COUNT)]

    monkeypatch.setattr(MODULE, "ActionFlowModelWrapper", Wrapper)
    monkeypatch.setattr(MODULE, "_validate_dimensions_and_modules", lambda *args: None)
    monkeypatch.setattr(MODULE, "_validate_gradient_route_manifest", lambda *args: {})
    kwargs = dict(
        reconstruction_weight=schedule["joint_reconstruction_weight"],
        flow_weight=1.0,
        method=method,
        config=config,
    )
    if restored_warmup:
        with pytest.raises(MODULE.SmokeVerificationError, match="strict reload lost"):
            MODULE._validate_checkpoint(tmp_path, **kwargs)
    else:
        result = MODULE._validate_checkpoint(tmp_path, **kwargs)
        assert result["loss_schedule"] == schedule
        assert result["strict_checkpoint_reload"] == "passed"


@pytest.mark.parametrize("key", tuple(OBSERVED_SCHEDULES[0][2]))
def test_every_schedule_field_remains_checked(key):
    experiment, _, schedule = OBSERVED_SCHEDULES[0]
    changed = {**schedule, key: schedule[key] + 1}
    with pytest.raises(MODULE.SmokeVerificationError, match="unexpected.*schedule"):
        MODULE._validate_checkpoint_loss_schedule(
            changed,
            compose_experiment(experiment, overrides=["hydra/launcher=basic"]),
            reconstruction_weight=1,
            flow_weight=1,
        )


@pytest.mark.parametrize("flow_weight", (1.0, 0.01))
def test_actual_warmup_smoke_still_requires_configured_step_one(flow_weight):
    config = compose_experiment(
        "pusht/action_flow_bc_usocket_recon10_warmup10k_s42",
        overrides=["hydra/launcher=basic"],
    )
    # Exercise both verifier weights without reviving the retired flow001 leaf.
    OmegaConf.update(config, "model.flow_weight", flow_weight)
    # The maintained launcher reduces this experiment's full 10k warmup to one
    # update for the real two-update warmup smoke.
    OmegaConf.update(config, "model.reconstruction_only_warmup_steps", 1)
    schedule = {
        "joint_objective_begins_at_global_step": 1,
        "reconstruction_only_optimizer_steps": 1,
        "joint_flow_weight": flow_weight,
        "joint_reconstruction_weight": 10.0,
        "joint_action_velocity_weight": 1.0,
        "schema_version": 1,
    }
    assert (
        MODULE._validate_checkpoint_loss_schedule(
            schedule, config, reconstruction_weight=10, flow_weight=flow_weight
        )
        == 1
    )
    schedule["joint_objective_begins_at_global_step"] = 0
    schedule["reconstruction_only_optimizer_steps"] = 0
    with pytest.raises(MODULE.SmokeVerificationError, match="unexpected.*schedule"):
        MODULE._validate_checkpoint_loss_schedule(
            schedule, config, reconstruction_weight=10, flow_weight=flow_weight
        )


def test_schedule_cannot_be_validated_without_resolved_config():
    with pytest.raises(MODULE.SmokeVerificationError, match="exact config"):
        MODULE._validate_checkpoint_loss_schedule(
            OBSERVED_SCHEDULES[0][2], None, reconstruction_weight=1, flow_weight=1
        )


def test_joint_checkpoint_owners_reject_codec_and_parameter_aliases():
    from egomimic.pipeline.stages_action_flow import (
        RoutedContentDecoderStage,
        RoutedContentEncoderStage,
    )

    def stages_with(encoders, decoders, field):
        stages = [None] * 8
        stages[4] = (
            RoutedContentEncoderStage(encoders, route_key="embodiment")
            if encoders
            else SimpleNamespace(encoder={})
        )
        stages[6] = SimpleNamespace(field=field)
        stages[7] = RoutedContentDecoderStage(decoders, route_key="embodiment")
        return stages

    names = ("libero_panda", "libero_panda_sharpa")
    encoders = {name: torch.nn.Linear(2, 2) for name in names}
    decoders = {name: torch.nn.Linear(2, 2) for name in names}
    field = torch.nn.Linear(2, 2)
    assert (
        len(MODULE._joint_checkpoint_owners(stages_with(encoders, decoders, field)))
        == 5
    )
    decoders[names[0]].weight = encoders[names[0]].weight
    with pytest.raises(MODULE.SmokeVerificationError, match="share parameters"):
        MODULE._joint_checkpoint_owners(stages_with(encoders, decoders, field))
    decoders[names[0]] = encoders[names[0]]
    with pytest.raises(MODULE.SmokeVerificationError, match="aliased"):
        MODULE._joint_checkpoint_owners(stages_with(encoders, decoders, field))
    with pytest.raises(MODULE.SmokeVerificationError, match="coverage"):
        MODULE._joint_checkpoint_owners(stages_with({}, decoders, field))


def test_joint_smoke_uses_unique_immutable_step_eight_without_rolling_last(tmp_path):
    checkpoint = tmp_path / "epoch-0000-step-000000008.ckpt"
    with pytest.raises(MODULE.SmokeVerificationError, match="one immutable"):
        MODULE._joint_smoke_checkpoint_path(tmp_path)
    checkpoint.write_bytes(b"path-contract-only")
    assert MODULE._joint_smoke_checkpoint_path(tmp_path) == checkpoint
    duplicate = tmp_path / "epoch-0001-step-000000008.ckpt"
    duplicate.write_bytes(b"path-contract-only")
    with pytest.raises(MODULE.SmokeVerificationError, match="one immutable"):
        MODULE._joint_smoke_checkpoint_path(tmp_path)
    duplicate.unlink()
    (tmp_path / "last.ckpt").symlink_to(checkpoint.name)
    with pytest.raises(MODULE.SmokeVerificationError, match="rolling"):
        MODULE._joint_smoke_checkpoint_path(tmp_path)


def test_joint_history_requires_both_source_rows_at_step_seven():
    from tests.test_verify_action_flow_training_smoke import _history_row

    row = _history_row()
    for key, value in list(row.items()):
        if MODULE.SOURCE_LABEL in key:
            for label in ("libero_panda", "libero_panda_sharpa"):
                row[key.replace(MODULE.SOURCE_LABEL, label)] = value
    row["Train/ActionFlow/Compute/FieldForwardCallsPerStep"] = 2.0
    row["Train/ActionFlow/Compute/FieldSampleEquivalentsPerStep"] = 28.0
    for prefix in (
        "GradientCosine",
        "GradientCosineDefined",
        "GradientIntersectionParameterCount",
    ):
        row[f"Train/ActionFlow/{prefix}/FM__Reconstruction"] = 0.0
    row["Optimizer/LR/AdamW"] = row["Optimizer/LR/Muon"] = 2.5e-8
    options = dict(
        method=MODULE.STOPGRAD_UNITE_METHOD,
        source_labels=("libero_panda", "libero_panda_sharpa"),
        minimum_step=7,
    )
    assert MODULE._validate_history({7: row}, **options)["valid_step"] == 7
    with pytest.raises(MODULE.SmokeVerificationError, match="gradient telemetry"):
        MODULE._validate_history({1: row}, **options)
    del row["Train/MSE/libero_panda_sharpa"]
    with pytest.raises(MODULE.SmokeVerificationError, match="gradient telemetry"):
        MODULE._validate_history({7: row}, **options)


def test_joint_ema_reload_updates_every_alias_and_rejects_wrong_updates():
    from scripts.train.verify_libero_native_action_flow_smoke import (
        validate_joint_ema_reload,
    )

    model = torch.nn.Module()
    model.model = torch.nn.Sequential(torch.nn.Linear(3, 2))
    model.nets = model.model
    # The native normalizer rebuilds its ParameterDict during strict reload.
    from egomimic.models.oat.model.common.dict_of_tensor_mixin import DictOfTensorMixin

    model.model.add_module(
        "normalizer",
        DictOfTensorMixin(
            torch.nn.ParameterDict(
                {"scale": torch.nn.Parameter(torch.ones(2), requires_grad=False)}
            )
        ),
    )
    settings = dict(
        decay=0.9978,
        use_warmup=False,
        update_after_step=0,
        inv_gamma=1.0,
        power=0.75,
        min_decay=0.0,
    )
    cfg = dict(callbacks=dict(ema=settings))
    ema = {
        name: torch.full_like(parameter, 7)
        for name, parameter in model.named_parameters()
    }
    payload = dict(
        global_step=8, ema_num_updates=8, ema_config=settings, ema_state_dict=ema
    )
    result = validate_joint_ema_reload(model, payload, cfg)
    assert result["state_parameter_aliases"] > result["canonical_parameters"]
    assert all(
        torch.equal(value, torch.full_like(value, 7))
        for value in model.state_dict().values()
    )
    payload["ema_num_updates"] = 7
    with pytest.raises(ValueError, match="update count"):
        validate_joint_ema_reload(model, payload, cfg)
    payload["ema_num_updates"] = 8
    ema[next(iter(ema))].fill_(float("nan"))
    with pytest.raises(ValueError, match="EMA tensor differs"):
        validate_joint_ema_reload(model, payload, cfg)


def test_joint_smoke_cli_dispatch_is_explicit_and_requires_all_hashes(
    tmp_path, monkeypatch
):
    from scripts.train import verify_libero_native_action_flow_smoke as native

    experiment = "libero/action_flow_libero_spatial_sharpa_gripper_keypoints_s42"
    args = MODULE._parser().parse_args(
        [str(tmp_path), "--experiment", experiment, "--expected-head", "a" * 40]
    )
    assert args.experiment == experiment
    # Missing identities must fail before source inspection, checkpoint loads or API reads.
    with pytest.raises(ValueError, match="every explicit expected SHA"):
        MODULE.verify_smoke(tmp_path, experiment=experiment, expected_head="a" * 40)
    captured = {}

    def routed(**kwargs):
        captured.update(kwargs)
        return {"dispatch_only": True}

    monkeypatch.setattr(native, "verify_joint_smoke", routed)
    assert MODULE.verify_smoke(
        tmp_path,
        experiment=experiment,
        expected_head="a" * 40,
        expected_config_sha256="b" * 64,
        expected_split_sha256="c" * 64,
        expected_normalization_sha256="d" * 64,
        expected_preflight_sha256="e" * 64,
    ) == {"dispatch_only": True}
    assert captured["expected_preflight_sha256"] == "e" * 64
    assert captured["run_dir"] == tmp_path


@pytest.mark.parametrize("transport", [None, "slurm-steps", "trainer-pidfd"])
def test_joint_shared_child_rejects_unbound_signal_transport(transport):
    import os
    import subprocess
    from pathlib import Path

    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("AF_", "ICE_"))
    }
    env["AF_EXPERIMENT"] = (
        "libero/action_flow_libero_spatial_sharpa_gripper_keypoints_s42"
    )
    if transport is not None:
        env["AF_CHECKPOINT_TRANSPORT"] = transport
    child = (
        Path(__file__).resolve().parents[1] / "scripts/train/action_flow_run_child.sh"
    )
    result = subprocess.run(
        ["bash", str(child)], env=env, capture_output=True, text=True
    )
    assert result.returncode == 2
    assert (
        "requires audited trainer-pidfd"
        if transport != "trainer-pidfd"
        else "PIDfd binding missing"
    ) in result.stderr
