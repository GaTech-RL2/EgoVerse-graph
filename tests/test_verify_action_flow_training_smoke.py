from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf, open_dict

from tools.validate_action_flow_config import compose_experiment

SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "train"
    / "verify_action_flow_training_smoke.py"
)
SPEC = importlib.util.spec_from_file_location(
    "verify_action_flow_training_smoke", SCRIPT
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

HEAD = "a" * 40


def _resolved_smoke_config(tmp_path: Path, *, reconstruction_weight: float = 1.0):
    suffix = {1.0: "1", 10.0: "10", 100.0: "100"}[reconstruction_weight]
    experiment = f"pusht/action_flow_bc_usocket_recon{suffix}_s42"
    cfg = compose_experiment(experiment)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    normalization = tmp_path / "normalization.pt"
    normalization.write_bytes(b"train-only-normalization")
    normalization_hash = hashlib.sha256(normalization.read_bytes()).hexdigest()

    with open_dict(cfg):
        cfg.trainer.max_steps = 2
        cfg.trainer.val_check_interval = 1
        cfg.trainer.limit_val_batches = 1
        cfg.trainer.log_every_n_steps = 1
        cfg.callbacks.model_checkpoint.every_n_train_steps = 1
        cfg.model.gradient_telemetry_cadence = 2
        cfg.norm_stats.precomputed_norm_path = str(normalization)
        cfg.run_provenance.source_commit = HEAD
        cfg.run_provenance.normalization_sha256 = normalization_hash
        cfg.evaluator.artifact_root = str(run_dir / "energy")
        cfg.evaluator.action_flow_diagnostics.artifact_root = str(
            run_dir / "diagnostics"
        )
        cfg.hydra.runtime.cwd = str(MODULE.REPOSITORY_ROOT)
        cfg.hydra.runtime.output_dir = str(run_dir)
    HydraConfig.instance().set_config(cfg)
    selected = OmegaConf.masked_copy(cfg, [key for key in cfg.keys() if key != "hydra"])
    resolved = OmegaConf.create(OmegaConf.to_container(selected, resolve=True))
    config_path = run_dir / ".hydra/config.yaml"
    config_path.parent.mkdir()
    OmegaConf.save(resolved, config_path)
    return experiment, run_dir, config_path, normalization_hash


def test_config_gate_accepts_only_exact_two_step_contract(tmp_path, monkeypatch):
    experiment, run_dir, config_path, normalization_hash = _resolved_smoke_config(
        tmp_path
    )
    monkeypatch.setattr(MODULE, "_git_head", lambda: HEAD)

    _, identities = MODULE._validate_config(
        config_path=config_path,
        experiment=experiment,
        run_dir=run_dir,
        expected_head=HEAD,
        expected_config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
        expected_split_sha256=None,
        expected_normalization_sha256=normalization_hash,
    )

    assert identities["repo_head"] == HEAD
    assert identities["normalization_sha256"] == normalization_hash
    assert (
        identities["config_sha256"]
        == hashlib.sha256(config_path.read_bytes()).hexdigest()
    )


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        ("model.flow_samples_per_content", 1, "flow_samples_per_content"),
        ("model.gradient_telemetry_cadence", 1, "gradient_telemetry_cadence"),
        ("trainer.devices", 2, "trainer.devices"),
        ("callbacks.model_checkpoint.every_n_train_steps", 2, "every_n_train_steps"),
    ],
)
def test_config_gate_rejects_smoke_shortcuts(
    tmp_path, monkeypatch, path, value, message
):
    experiment, run_dir, config_path, _ = _resolved_smoke_config(tmp_path)
    cfg = OmegaConf.load(config_path)
    OmegaConf.update(cfg, path, value)
    OmegaConf.save(cfg, config_path)
    monkeypatch.setattr(MODULE, "_git_head", lambda: HEAD)

    with pytest.raises(MODULE.SmokeVerificationError, match=message):
        MODULE._validate_config(
            config_path=config_path,
            experiment=experiment,
            run_dir=run_dir,
            expected_head=HEAD,
            expected_config_sha256=None,
            expected_split_sha256=None,
            expected_normalization_sha256=None,
        )


def _history_row():
    row = {}
    for name in (
        "TotalLoss",
        "FlowMatchingLoss",
        "ReconstructionLoss",
        "ReconstructionL1",
        "ActionVelocityLoss",
    ):
        train_value = 3.75 if name == "TotalLoss" else 1.25
        valid_value = 4.5 if name == "TotalLoss" else 1.5
        row[f"Train/ActionFlow/{name}_step"] = train_value
        row[f"Train/ActionFlow/{name}/{MODULE.SOURCE_LABEL}_step"] = train_value
        row[f"Valid/ActionFlow/{name}"] = valid_value
    for label in ("FM", "Reconstruction", "ActionVelocity"):
        row[f"Train/ActionFlow/GradientNorm/{label}"] = 0.5
        row[f"Train/ActionFlow/GradientParameterCount/{label}"] = 100.0
    for pair in (
        "FM__Reconstruction",
        "FM__ActionVelocity",
        "Reconstruction__ActionVelocity",
    ):
        row[f"Train/ActionFlow/GradientCosine/{pair}"] = 0.25
        row[f"Train/ActionFlow/GradientCosineDefined/{pair}"] = 1.0
        row[f"Train/ActionFlow/GradientIntersectionParameterCount/{pair}"] = 50.0
    row["Train/MSE"] = 1.25
    row[f"Train/MSE/{MODULE.SOURCE_LABEL}"] = 1.25
    row["Train/ActionFlow/Compute/FieldForwardCallsPerStep"] = 1.0
    row["Train/ActionFlow/Compute/FieldSampleEquivalentsPerStep"] = 14.0
    row["Train/ActionFlow/Compute/DecoderJVPCallsPerStep"] = 1.0
    row["Train/ActionFlow/Compute/PeakAllocatedBytes"] = 1024.0
    row["Train/ActionFlow/Schedule/ReconstructionOnly"] = 0.0
    row["Train/ActionFlow/Schedule/EffectiveFlowWeight"] = 1.0
    row["Train/ActionFlow/Schedule/EffectiveActionVelocityWeight"] = 1.0
    for base in (
        "Valid/MSE",
        "Valid/Native_MSE",
        "Valid/EnergyScore@32",
        "Valid/EnergyScoreAccuracy@32",
        "Valid/EnergyScoreDiversity@32",
    ):
        row[base] = 0.75
        row[f"{base}/{MODULE.SOURCE_LABEL}"] = 0.75
    for name in (
        "Valid/ActionFlow/CleanReconstructionMSE",
        "Valid/ActionFlow/CleanReconstructionNativeMSE",
        "Valid/ActionFlow/Latent/clean/RMS",
        "Valid/ActionFlow/Latent/generated/EffectiveRank",
        "Valid/ActionFlow/DecoderJacobian/clean/spectral_norm_mean",
        "Valid/ActionFlow/DecodedFullNoise/token_radius_mean",
        "Valid/ActionFlow/DenoisingTrajectory/LatentMSE/t1000",
        "Valid/ActionFlow/DenoisingTrajectory/DecodedMSE/t0000",
        "Valid/ActionFlow/DenoisingTrajectory/DecodedNativeMSE/t0000",
        "Valid/ActionFlow/Alignment/FinalLatentCosine/t1000",
        "Valid/ActionFlow/Alignment/CKA/encoder_00__field_00/t0500",
        "Valid/ActionFlow/Alignment/CKNNA/encoder_00__field_00/t0500",
    ):
        row[name] = 0.5
        if "NativeMSE" in name or "decoded_native_mse" in name:
            row[f"{name}/{MODULE.SOURCE_LABEL}"] = 0.5
    return row


def test_history_gate_requires_components_gradients_and_scheduled_validation():
    result = MODULE._validate_history({0: {}, 2: _history_row()})

    assert result["train_step"] == 2
    assert result["valid_step"] == 2
    assert result["train"]["Train/ActionFlow/FlowMatchingLoss"] == 1.25


def test_history_gate_accepts_float32_flow_weight_telemetry():
    row = _history_row()
    row["Train/ActionFlow/Schedule/EffectiveFlowWeight"] = float(
        torch.tensor(0.01)
    )
    for suffix in ("", f"/{MODULE.SOURCE_LABEL}"):
        row[f"Train/ActionFlow/TotalLoss{suffix}_step"] = 2.5125
    row["Valid/ActionFlow/TotalLoss"] = 3.015

    result = MODULE._validate_history(
        {2: row}, reconstruction_weight=1.0, flow_weight=0.01
    )

    assert result["train_step"] == 2


def test_history_gate_rejects_missing_gradient_telemetry():
    row = _history_row()
    del row["Train/ActionFlow/GradientCosine/FM__ActionVelocity"]

    with pytest.raises(MODULE.SmokeVerificationError, match="gradient telemetry"):
        MODULE._validate_history({2: row})


def _write_artifacts(tmp_path: Path):
    split_hash = "b" * 64
    norm_hash = "d" * 64
    content_hash = "e" * 64
    aggregate_hash = "f" * 64
    energy_root = tmp_path / "energy"
    diagnostic_root = tmp_path / "diagnostics"
    config_path = tmp_path / ".hydra/config.yaml"
    config_path.parent.mkdir()
    config_path.write_text("name: test\n")
    content_path = tmp_path / "content.json"
    content_path.write_text("{}\n")
    leaf = Path("epoch-0-step-2/rank-0-batch-0.pt")
    energy_path = energy_root / leaf
    diagnostic_path = diagnostic_root / leaf
    energy_path.parent.mkdir(parents=True)
    diagnostic_path.parent.mkdir(parents=True)
    seed_hash = "c" * 64
    targets = torch.zeros(16, 16, 4)
    condition_ids = [
        {
            "batch_position": index,
            "episode_hash": f"episode-{index:04d}",
            "frame_index": index,
            "normalized_target_sha256": MODULE._target_tensor_sha256(targets[index]),
        }
        for index in range(16)
    ]
    energy_view = {
        "definition": "first_deterministic_validation_batch_per_ddp_rank",
        "split_manifest_sha256": split_hash,
        "per_rank_batch_size": 16,
        "world_size": 1,
    }
    distance = MODULE.usocket_energy_distance_metadata(
        MODULE.USOCKET_ENERGY_DISTANCE_CONFIG
    )
    wandb = {"entity": "rl2-group", "project": "test", "run_id": "smoke"}
    energy_identity = {
        "schema": "energy-score-validation-artifact/v2",
        "metric": {
            "name": "EnergyScore@32",
            "sample_count": 32,
            "deterministic_seed": 420042,
        },
        "source_commit": HEAD,
        "normalization_sha256": norm_hash,
        "split_manifest_sha256": split_hash,
        "resolved_config_path": str(config_path.resolve()),
        "resolved_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "wandb": wandb,
        "dataset_content": {
            "manifest_path": str(content_path.resolve()),
            "manifest_sha256": content_hash,
            "aggregate_sha256": aggregate_hash,
        },
        "distance": distance,
        "seed_bank_sha256": seed_hash,
        "validation_view": energy_view,
        "validation_conditions": {MODULE.SOURCE_LABEL: condition_ids},
        "checkpoint_binding": {
            "global_step": 2,
            "checkpoint_sha256": None,
            "sha256_status": MODULE.ENERGY_CHECKPOINT_STATUS,
        },
    }
    torch.save(
        {
            "schema_version": 2,
            "metric": "EnergyScore@32",
            "sample_count": 32,
            "seed_bank": list(range(32)),
            "seed_bank_sha256": seed_hash,
            "deterministic_seed": 420042,
            "distance": distance,
            "global_step": 2,
            "validation_view": energy_view,
            "identity": energy_identity,
            "identity_sha256": MODULE._canonical_json_sha256(energy_identity),
            "domains": {
                MODULE.SOURCE_LABEL: {
                    "predictions": torch.zeros(32, 16, 16, 4),
                    "targets": targets,
                    "native_predictions": torch.zeros(32, 16, 16, 3),
                    "native_targets": torch.zeros(16, 16, 3),
                    "condition_ids": condition_ids,
                }
            },
        },
        energy_path,
    )
    diagnostic_provenance = {
        "source_commit": HEAD,
        "normalization_sha256": norm_hash,
        "split_manifest_sha256": split_hash,
        "dataset_content": {
            "manifest_sha256": content_hash,
            "aggregate_sha256": aggregate_hash,
        },
    }
    identity = {
        "schema": "action-flow-validation-diagnostics/v1",
        "validation_view": energy_view,
        "provenance": diagnostic_provenance,
        "native_error": MODULE.USOCKET_NATIVE_ERROR_CONFIG,
    }
    identity_hash = hashlib.sha256(
        json.dumps(
            identity,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode()
    ).hexdigest()
    torch.save(
        {
            "schema_version": 1,
            "metric": "ActionFlowValidationDiagnostics",
            "global_step": 2,
            "identity": identity,
            "identity_sha256": identity_hash,
            "sampler": {"name": "reverse_euler"},
            "validation_view": energy_view,
            "provenance": diagnostic_provenance,
            "statistics": {"native_action_error": MODULE.USOCKET_NATIVE_ERROR_CONFIG},
            "sources": {
                MODULE.SOURCE_LABEL: {
                    "computed": {
                        "clean_reconstruction_native_mse_by_condition": torch.ones(16),
                        "trajectory_decoded_native_mse_by_condition": torch.ones(
                            17, 16
                        ),
                        "fixed_level_metrics": {
                            label: {
                                metric: torch.ones(16)
                                for metric in (
                                    "state_decoded_native_mse",
                                    "predicted_clean_decoded_native_mse",
                                    "final_decoded_native_mse",
                                )
                            }
                            for label in MODULE.NATIVE_LEVELS
                        },
                    }
                }
            },
        },
        diagnostic_path,
    )
    diagnostic_hash = hashlib.sha256(diagnostic_path.read_bytes()).hexdigest()
    Path(f"{diagnostic_path}.sha256").write_text(
        json.dumps(
            {
                "artifact": diagnostic_path.name,
                "identity_sha256": identity_hash,
                "sha256": diagnostic_hash,
            }
        )
    )
    config = OmegaConf.create(
        {
            "evaluator": {
                "artifact_root": str(energy_root),
                "seed_bank_sha256": seed_hash,
                "energy_score_validation_view": energy_view,
                "action_flow_diagnostics": {
                    "artifact_root": str(diagnostic_root),
                    "validation_view": energy_view,
                },
            },
            "logger": {
                "wandb": {
                    "entity": wandb["entity"],
                    "project": wandb["project"],
                    "id": wandb["run_id"],
                }
            },
        }
    )
    identities = {
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "content_manifest_path": str(content_path.resolve()),
        "content_manifest_sha256": content_hash,
        "dataset_content_aggregate_sha256": aggregate_hash,
        "normalization_sha256": norm_hash,
        "repo_head": HEAD,
        "split_manifest_sha256": split_hash,
    }
    checkpoint = {
        "checkpoint_sha256": "1" * 64,
        "file_size_bytes": 100,
        "global_step": 2,
    }
    return config, identities, checkpoint, diagnostic_path


def test_artifact_gate_verifies_energy_and_diagnostic_immutability(tmp_path):
    config, identities, checkpoint, diagnostic_path = _write_artifacts(tmp_path)

    result = MODULE._validate_artifacts(
        config=config,
        run_dir=tmp_path,
        identities=identities,
        checkpoint=checkpoint,
    )

    assert result["energy_score"]["sha256"]
    assert result["action_flow_diagnostics"]["path"] == str(diagnostic_path)


def test_artifact_gate_selects_exact_slurm_attempt_and_checks_execution(tmp_path, monkeypatch):
    root = tmp_path / "artifacts"
    for restart in (0, 1):
        artifact = root / f"job-5714540-restart-{restart}" / "epoch-0-step-2/rank-0-batch-0.pt"
        artifact.parent.mkdir(parents=True)
        torch.save({
            "global_step": 2,
            "execution": {"slurm_job_id": "5714540", "slurm_restart_count": restart},
        }, artifact)
    monkeypatch.setenv("SLURM_JOB_ID", "5714540")
    monkeypatch.setenv("SLURM_RESTART_COUNT", "1")
    selected, payload = MODULE._step_two_artifact(root, label="test")
    assert "job-5714540-restart-1" in str(selected)
    payload["execution"]["slurm_restart_count"] = 0
    torch.save(payload, selected)
    with pytest.raises(MODULE.SmokeVerificationError, match="execution identity mismatch"):
        MODULE._step_two_artifact(root, label="test")
    monkeypatch.delenv("SLURM_JOB_ID")
    with pytest.raises(MODULE.SmokeVerificationError, match="expected one step-2"):
        MODULE._step_two_artifact(root, label="test")


def test_artifact_gate_rejects_tampered_diagnostic_sidecar(tmp_path):
    config, identities, checkpoint, diagnostic_path = _write_artifacts(tmp_path)
    sidecar = Path(f"{diagnostic_path}.sha256")
    payload = json.loads(sidecar.read_text())
    payload["sha256"] = "0" * 64
    sidecar.write_text(json.dumps(payload))

    with pytest.raises(MODULE.SmokeVerificationError, match="sidecar mismatch"):
        MODULE._validate_artifacts(
            config=config,
            run_dir=tmp_path,
            identities=identities,
            checkpoint=checkpoint,
        )


def test_artifact_gate_accepts_unique_origin_from_separate_validation_job(tmp_path, monkeypatch):
    root = tmp_path / "artifacts"
    artifact = root / "job-5714540-restart-1/epoch-0-step-2/rank-0-batch-0.pt"
    artifact.parent.mkdir(parents=True)
    torch.save({
        "global_step": 2,
        "execution": {"slurm_job_id": "5714540", "slurm_restart_count": 1},
    }, artifact)
    monkeypatch.setenv("SLURM_JOB_ID", "5719999")
    monkeypatch.setenv("SLURM_RESTART_COUNT", "0")
    selected, payload = MODULE._step_two_artifact(root, label="test")
    assert selected == artifact
    assert payload["execution"]["slurm_job_id"] == "5714540"


def test_preflight_gate_binds_source_experiment_split_and_normalization(tmp_path):
    experiment = "pusht/action_flow_bc_usocket_recon1_s42"
    split_hash = "b" * 64
    normalization_hash = "c" * 64
    content_hash = "d" * 64
    aggregate_hash = "e" * 64
    destination = tmp_path / "provenance/restart-0/PREFLIGHT_RESULT.json"
    destination.parent.mkdir(parents=True)
    destination.write_text(
        json.dumps(
            {
                "status": "PASS",
                "source": {"head": HEAD},
                "experiment": experiment,
                "split_manifest_sha256": split_hash,
                "normalization_sha256": normalization_hash,
                "content_manifest_sha256": content_hash,
                "dataset_content_aggregate_sha256": aggregate_hash,
            },
            sort_keys=True,
        )
    )
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()

    result = MODULE._validate_preflight(
        run_dir=tmp_path,
        expected_sha256=digest,
        expected_head=HEAD,
        experiment=experiment,
        split_sha256=split_hash,
        normalization_sha256=normalization_hash,
        content_manifest_sha256=content_hash,
        dataset_content_aggregate_sha256=aggregate_hash,
    )

    assert result == {"path": str(destination), "sha256": digest, "copy_count": 1}


def test_cli_accepts_launcher_compatibility_names():
    args = MODULE._parser().parse_args(
        [
            "/tmp/run",
            "--expected-head",
            HEAD,
            "--expected-experiment",
            "pusht/action_flow_bc_usocket_recon10_s42",
            "--expected-reconstruction-weight",
            "10",
            "--expected-preflight-sha256",
            "d" * 64,
        ]
    )

    assert args.experiment.endswith("recon10_s42")
    assert args.expected_reconstruction_weight == 10.0


def test_gpu_probe_gate_requires_real_single_h100_or_h200_bf16(tmp_path):
    path = tmp_path / "provenance/restart-0/gpu_probe.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "status": "PASSED",
                "world_size": 1,
                "gpu_name": "NVIDIA H200",
                "bf16_supported": True,
                "bf16_forward_backward": {
                    "dtype": "torch.bfloat16",
                    "gradient_dtype": "torch.bfloat16",
                    "finite": True,
                },
                "nccl": {"world_size": 1, "all_reduce": 1.0, "destroyed": True},
            }
        )
    )

    records = MODULE._validate_gpu_probes(tmp_path)

    assert records[0]["gpu_name"] == "NVIDIA H200"


def test_gpu_probe_gate_rejects_non_target_gpu(tmp_path):
    path = tmp_path / "provenance/restart-0/gpu_probe.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "status": "PASSED",
                "world_size": 1,
                "gpu_name": "NVIDIA A100-SXM4-80GB",
                "bf16_supported": True,
                "bf16_forward_backward": {
                    "dtype": "torch.bfloat16",
                    "gradient_dtype": "torch.bfloat16",
                    "finite": True,
                },
                "nccl": {"world_size": 1, "all_reduce": 1.0, "destroyed": True},
            }
        )
    )

    with pytest.raises(MODULE.SmokeVerificationError, match="H100 or H200"):
        MODULE._validate_gpu_probes(tmp_path)


def _checkpoint_payload():
    return {
        "global_step": 2,
        "state_dict": {"model.weight": torch.ones(1)},
        "optimizer_states": [
            {
                "state": {0: {"exp_avg": torch.zeros(1)}},
                "param_groups": [{"lr": 3.0e-5, "params": [0]}],
            }
        ],
        "lr_schedulers": [{"last_epoch": 2, "_last_lr": [3.0e-6]}],
        "loops": {"fit_loop": {"completed": 2}},
    }


def test_checkpoint_gate_strictly_reloads_exact_action_flow_wrapper(
    tmp_path, monkeypatch
):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    payload = _checkpoint_payload()
    immutable = checkpoint_dir / "epoch-epoch=0-step-step=2.ckpt"
    torch.save(payload, immutable)
    (checkpoint_dir / "last.ckpt").symlink_to(immutable.name)

    class _Parameter:
        @staticmethod
        def numel():
            return MODULE.EXPECTED_PARAMETER_COUNT

    class _Wrapper:
        encoder_e = object()
        field_v = object()
        decoder_g = object()

        class _Nets:
            @staticmethod
            def named_parameters(**_kwargs):
                return []

        nets = _Nets()

        @classmethod
        def load_from_checkpoint(cls, *_args, **_kwargs):
            return cls()

        @staticmethod
        def parameters():
            return [_Parameter()]

    monkeypatch.setattr(MODULE, "ActionFlowModelWrapper", _Wrapper)
    monkeypatch.setattr(
        MODULE,
        "_validate_gradient_route_manifest",
        lambda *_args: {"manifest_sha256": "f" * 64},
    )

    result = MODULE._validate_checkpoint(
        tmp_path, reconstruction_weight=1.0, flow_weight=1.0
    )

    assert result["global_step"] == 2
    assert result["parameter_count"] == MODULE.EXPECTED_PARAMETER_COUNT
    assert result["strict_checkpoint_reload"] == "passed"
    assert result["file_size_bytes"] > 0


def _gradient_manifest(named_parameters):
    route_names = {
        "FM": [name for name, _ in named_parameters[:3]],
        "Reconstruction": [named_parameters[1][0], named_parameters[3][0]],
        "ActionVelocity": [name for name, _ in named_parameters],
    }
    routes = {}
    route_hashes = {}
    for label, names in route_names.items():
        parameters = dict(named_parameters)
        route = [
            {
                "dtype": str(parameters[name].dtype),
                "name": name,
                "numel": parameters[name].numel(),
                "shape": list(parameters[name].shape),
            }
            for name in names
        ]
        routes[label] = route
        route_hashes[label] = MODULE._canonical_json_sha256(route)
    intersections = {
        "FM__Reconstruction": [named_parameters[1][0]],
        "FM__ActionVelocity": [name for name, _ in named_parameters[:3]],
        "Reconstruction__ActionVelocity": [
            named_parameters[1][0],
            named_parameters[3][0],
        ],
    }
    core = {
        "routes": routes,
        "route_sha256": route_hashes,
        "intersections": intersections,
        "schema_version": 1,
    }
    return {**core, "manifest_sha256": MODULE._canonical_json_sha256(core)}


def test_gradient_route_manifest_binds_exact_named_shared_paths():
    named = (
        ("nets.pipeline.stages.0.observation", torch.nn.Parameter(torch.ones(2))),
        ("nets.pipeline.stages.3.encoder", torch.nn.Parameter(torch.ones(3))),
        ("nets.pipeline.stages.5.field", torch.nn.Parameter(torch.ones(4))),
        ("nets.pipeline.stages.6.decoder", torch.nn.Parameter(torch.ones(5))),
    )
    manifest = _gradient_manifest(named)

    result = MODULE._validate_gradient_route_manifest(manifest, named)

    assert result["manifest_sha256"] == manifest["manifest_sha256"]
    assert result["route_parameter_counts"] == {
        "FM": 9,
        "Reconstruction": 8,
        "ActionVelocity": 14,
    }
    assert result["shared_path_parameter_counts"]["FM__Reconstruction"] == 3


def test_gradient_route_manifest_rejects_tampered_named_route():
    named = (
        ("nets.pipeline.stages.0.observation", torch.nn.Parameter(torch.ones(2))),
        ("nets.pipeline.stages.3.encoder", torch.nn.Parameter(torch.ones(3))),
        ("nets.pipeline.stages.5.field", torch.nn.Parameter(torch.ones(4))),
        ("nets.pipeline.stages.6.decoder", torch.nn.Parameter(torch.ones(5))),
    )
    manifest = _gradient_manifest(named)
    manifest["routes"]["FM"][0]["numel"] = 99

    with pytest.raises(MODULE.SmokeVerificationError, match="size mismatch"):
        MODULE._validate_gradient_route_manifest(manifest, named)


def test_checkpoint_gate_rejects_missing_scheduler(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    payload = _checkpoint_payload()
    payload["lr_schedulers"] = []
    immutable = checkpoint_dir / "epoch-epoch=0-step-step=2.ckpt"
    torch.save(payload, immutable)
    (checkpoint_dir / "last.ckpt").symlink_to(immutable.name)

    with pytest.raises(MODULE.SmokeVerificationError, match="scheduler state"):
        MODULE._validate_checkpoint(
            tmp_path, reconstruction_weight=1.0, flow_weight=1.0
        )
