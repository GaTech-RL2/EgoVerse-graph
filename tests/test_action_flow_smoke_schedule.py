"""Regressions for the actual Phoenix 12935484/12935488 verifier failures."""

from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from test_verify_action_flow_training_smoke import MODULE, _checkpoint_payload
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
    config = compose_experiment(experiment)
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
            compose_experiment(experiment),
            reconstruction_weight=1,
            flow_weight=1,
        )


@pytest.mark.parametrize("flow_weight", (1.0, 0.01))
def test_actual_warmup_smoke_still_requires_configured_step_one(flow_weight):
    suffix = "" if flow_weight == 1.0 else "_flow001"
    config = compose_experiment(
        f"pusht/action_flow_bc_usocket_recon10_warmup10k{suffix}_s42"
    )
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
