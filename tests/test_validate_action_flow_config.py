from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "egomimic" / "hydra_configs"
TOOL = ROOT / "tools" / "validate_action_flow_config.py"
SPEC = importlib.util.spec_from_file_location("action_flow_preflight", TOOL)
assert SPEC is not None and SPEC.loader is not None
preflight = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(preflight)


def test_real_action_flow_config_instantiates_and_passes_full_preflight():
    report, _ = preflight.validate_experiment(
        "pusht/action_flow_bc_usocket_recon1_s42",
        config_root=CONFIG_ROOT,
    )

    assert report["status"] == "PASS"
    assert report["config_name"] == "action_flow_bc_usocket_recon1_s42"
    assert len(report["resolved_config_sha256"]) == 64
    assert report["dimensions"] == {
        "action": [16, 4],
        "condition": 67,
        "image_feature": 64,
        "latent": [16, 8],
        "normalized_state": 3,
    }
    expected_counts = {
        "decoder_g": 10_744,
        "encoder_e": 10_748,
        "field_v": 39_506_641,
        "observation_encoder": 11_197_088,
        "pipeline_total": 50_725_221,
    }
    assert set(report["parameters"]) == set(expected_counts)
    for name, expected in expected_counts.items():
        manifest = report["parameters"][name]
        assert manifest["total"] == manifest["trainable"] == expected
        assert len(manifest["manifest_sha256"]) == 64
        assert sum(entry["numel"] for entry in manifest["entries"]) == expected
        assert len({entry["name"] for entry in manifest["entries"]}) == len(
            manifest["entries"]
        )
    assert report["topology"]["shared_field_instance"] is True
    assert report["topology"]["shared_decoder_instance"] is True
    assert report["topology"]["inference_order"] == [
        "FusedObsEncoder",
        "GaussianLatentNoise",
        "ConditionalVelocityStage",
        "ContentDecoderStage",
    ]
    assert report["launch"]["effective_global_batch"] == 32
    assert report["split"]["zero_id_overlap"] is True
    assert report["split"]["zero_resolved_path_overlap"] is True
    content_manifest_path = ROOT / preflight.CANONICAL_CONTENT_MANIFEST_RELATIVE_PATH
    content_manifest = json.loads(content_manifest_path.read_text())
    assert report["split"]["content_manifest_path"] == (
        preflight.CANONICAL_CONTENT_MANIFEST_RELATIVE_PATH.as_posix()
    )
    assert (
        report["split"]["content_manifest_sha256"]
        == hashlib.sha256(content_manifest_path.read_bytes()).hexdigest()
    )
    assert (
        report["split"]["dataset_content_aggregate_sha256"]
        == content_manifest["aggregate_sha256"]
    )


def test_codec98k_config_changes_only_the_typed_reconstruction_capacity():
    report, _ = preflight.validate_experiment(
        "pusht/action_flow_bc_usocket_latent_fm_sg_recon1_codec98k_s42",
        config_root=CONFIG_ROOT,
    )

    assert report["status"] == "PASS"
    assert report["config_name"] == (
        "action_flow_bc_usocket_latent_fm_sg_recon1_codec98k_s42"
    )
    assert report["parameters"]["encoder_e"]["total"] == 48_980
    assert report["parameters"]["decoder_g"]["total"] == 48_976
    assert report["parameters"]["pipeline_total"]["total"] == 50_801_685


def test_option_a_200m_muon_config_has_exact_capacity_optimizer_and_schedule():
    report, _ = preflight.validate_experiment(
        "pusht/action_flow_bc_usocket_latent_fm_sg_recon1_200m_muon_lr1e5_s42",
        config_root=CONFIG_ROOT,
    )

    assert report["status"] == "PASS"
    assert report["parameters"]["encoder_e"]["total"] == 1_010_420
    assert report["parameters"]["decoder_g"]["total"] == 1_010_416
    assert report["parameters"]["field_v"]["total"] == 186_536_913
    assert report["parameters"]["pipeline_total"]["total"] == 199_754_837
    optimization = report["optimization"]
    assert optimization["optimizer"]["target"].endswith(
        "ReleasedUniteCompositeOptimizer"
    )
    assert optimization["optimizer"]["lr"] == pytest.approx(1.0e-5)
    assert optimization["scheduler"]["eta_min"] == pytest.approx(1.0e-6)
    assert optimization["parameter_groups"]["complete"] is True
    assert optimization["parameter_groups"]["disjoint"] is True


def test_resolved_hash_is_stable_and_uses_runtime_sentinels():
    first = preflight.compose_experiment(
        "pusht/action_flow_bc_usocket_recon1_s42", config_root=CONFIG_ROOT
    )
    second = preflight.compose_experiment(
        "pusht/action_flow_bc_usocket_recon1_s42", config_root=CONFIG_ROOT
    )
    first_payload, first_hash = preflight.resolved_config_payload(first)
    second_payload, second_hash = preflight.resolved_config_payload(second)

    assert first_hash == second_hash
    assert first_payload == second_payload
    assert first_payload["paths"]["output_dir"] == "<RUNTIME_OUTPUT_DIR>"
    assert first_payload["logger"]["wandb"]["id"] == "<RUNTIME_RUN_ID>"


def test_pair_diff_is_only_the_declared_reconstruction_setting():
    report = preflight.validate_pair(
        "pusht/action_flow_bc_usocket_recon1_s42",
        "pusht/action_flow_bc_usocket_recon10_s42",
        config_root=CONFIG_ROOT,
    )

    assert report["status"] == "PASS"
    assert report["comparison"] == {
        "differing_paths": sorted(preflight.ALLOWED_PAIR_DIFFERENCES),
        "only_declared_reconstruction_differences": True,
        "status": "PASS",
    }
    assert {
        item["objective"]["reconstruction_weight"] for item in report["experiments"]
    } == {1.0, 10.0}


def test_recon100_diff_is_only_the_declared_reconstruction_setting():
    report = preflight.validate_pair(
        "pusht/action_flow_bc_usocket_recon1_s42",
        "pusht/action_flow_bc_usocket_recon100_s42",
        config_root=CONFIG_ROOT,
    )

    assert report["status"] == "PASS"
    assert report["comparison"]["only_declared_reconstruction_differences"] is True
    assert {
        item["objective"]["reconstruction_weight"] for item in report["experiments"]
    } == {1.0, 100.0}


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        ("model.pipeline.stages.5.field.time_scale", 1.0, "time scale"),
        ("model.pipeline.stages.4.samples_per_content", 13, "bridge samples"),
        ("callbacks.model_checkpoint.every_n_train_steps", 20_000, "checkpoint"),
        ("launch_params.gpus_per_node", 2, "GPUs per node"),
        (
            "evaluator.action_flow_diagnostics.enabled",
            False,
            "diagnostics enabled",
        ),
        (
            "evaluator.action_flow_diagnostics.native_error.native_theta_index",
            1,
            "unsupported USocket native error contract",
        ),
        (
            "run_provenance.content_manifest_path",
            "egomimic/hydra_configs/data/pusht/manifests/not-canonical.json",
            "canonical dataset-content manifest path",
        ),
        (
            "run_provenance.content_manifest_sha256",
            "0" * 64,
            "canonical dataset-content manifest hash",
        ),
        (
            "run_provenance.dataset_content_aggregate_sha256",
            "0" * 64,
            "canonical dataset aggregate content hash",
        ),
        (
            "evaluator.energy_score_provenance.source_commit",
            "0" * 40,
            "EnergyScore source commit provenance",
        ),
        (
            "evaluator.energy_score_provenance.normalization_sha256",
            "0" * 64,
            "EnergyScore normalization provenance",
        ),
        (
            "evaluator.energy_score_provenance.split_manifest_sha256",
            "0" * 64,
            "EnergyScore split provenance",
        ),
        (
            "evaluator.energy_score_validation_view.split_manifest_sha256",
            "0" * 64,
            "EnergyScore validation-view split provenance",
        ),
        (
            "evaluator.energy_score_provenance.dataset_content.manifest_path",
            "not-the-run-manifest.json",
            "evaluator dataset-content manifest path provenance",
        ),
        (
            "evaluator.energy_score_provenance.dataset_content.manifest_sha256",
            "0" * 64,
            "evaluator dataset-content manifest hash provenance",
        ),
        (
            "evaluator.energy_score_provenance.dataset_content.aggregate_sha256",
            "0" * 64,
            "evaluator dataset aggregate content hash provenance",
        ),
        (
            "evaluator.action_flow_diagnostics.provenance.source_commit",
            "0" * 40,
            "diagnostic source commit provenance",
        ),
        (
            "evaluator.action_flow_diagnostics.provenance.normalization_sha256",
            "0" * 64,
            "diagnostic normalization provenance",
        ),
        (
            "evaluator.action_flow_diagnostics.provenance.split_manifest_sha256",
            "0" * 64,
            "diagnostic split provenance",
        ),
        (
            "evaluator.action_flow_diagnostics.validation_view.split_manifest_sha256",
            "0" * 64,
            "diagnostic validation-view split provenance",
        ),
        (
            "evaluator.action_flow_diagnostics.provenance.dataset_content.manifest_sha256",
            "0" * 64,
            "diagnostic dataset-content manifest provenance",
        ),
        (
            "evaluator.action_flow_diagnostics.provenance.dataset_content.aggregate_sha256",
            "0" * 64,
            "diagnostic dataset aggregate content provenance",
        ),
    ],
)
def test_preflight_fails_closed_before_training(path, value, message):
    config = preflight.compose_experiment(
        "pusht/action_flow_bc_usocket_recon1_s42", config_root=CONFIG_ROOT
    )
    from omegaconf import OmegaConf

    OmegaConf.update(config, path, value)
    with pytest.raises(preflight.PreflightError, match=message):
        preflight.validate_config(
            config,
            experiment="pusht/action_flow_bc_usocket_recon1_s42",
            config_root=CONFIG_ROOT,
        )


@pytest.mark.parametrize(
    "path",
    [
        "evaluator.energy_score_distance",
        "run_provenance.energy_score_contract.distance",
        "evaluator.energy_score_provenance.distance_contract",
    ],
)
def test_every_energy_distance_surface_requires_the_typed_usocket_contract(path):
    config = preflight.compose_experiment(
        "pusht/action_flow_bc_usocket_recon1_s42", config_root=CONFIG_ROOT
    )
    from omegaconf import OmegaConf

    invalid = copy.deepcopy(preflight.USOCKET_ENERGY_DISTANCE_CONFIG)
    invalid["rotation_scale_radians"] = 1.0
    OmegaConf.update(config, path, invalid, merge=False)

    with pytest.raises(
        preflight.PreflightError,
        match="unsupported USocket EnergyScore distance contract",
    ):
        preflight.validate_config(
            config,
            experiment="pusht/action_flow_bc_usocket_recon1_s42",
            config_root=CONFIG_ROOT,
        )


def test_native_error_surface_rejects_undeclared_contract_keys():
    config = preflight.compose_experiment(
        "pusht/action_flow_bc_usocket_recon1_s42", config_root=CONFIG_ROOT
    )
    from omegaconf import OmegaConf

    invalid = copy.deepcopy(preflight.USOCKET_NATIVE_ERROR_CONFIG)
    invalid["undeclared"] = True
    OmegaConf.update(
        config,
        "evaluator.action_flow_diagnostics.native_error",
        invalid,
        merge=False,
    )

    with pytest.raises(preflight.PreflightError, match="native error keys differ"):
        preflight.validate_config(
            config,
            experiment="pusht/action_flow_bc_usocket_recon1_s42",
            config_root=CONFIG_ROOT,
        )


@pytest.mark.parametrize(
    "path",
    [
        "evaluator.energy_score_provenance.dataset_content",
        "evaluator.action_flow_diagnostics.provenance.dataset_content",
    ],
)
def test_dataset_content_provenance_rejects_undeclared_keys(path):
    config = preflight.compose_experiment(
        "pusht/action_flow_bc_usocket_recon1_s42", config_root=CONFIG_ROOT
    )
    from omegaconf import OmegaConf

    payload = OmegaConf.to_container(OmegaConf.select(config, path), resolve=True)
    assert isinstance(payload, dict)
    payload["undeclared"] = True
    OmegaConf.update(config, path, payload, merge=False)

    with pytest.raises(preflight.PreflightError, match="provenance keys"):
        preflight.validate_config(
            config,
            experiment="pusht/action_flow_bc_usocket_recon1_s42",
            config_root=CONFIG_ROOT,
        )


def test_cli_emits_deterministic_machine_readable_failure(tmp_path, monkeypatch):
    output = tmp_path / "preflight.json"

    def fail(*_args, **_kwargs):
        raise preflight.PreflightError("deliberate contract failure")

    monkeypatch.setattr(preflight, "validate_experiment", fail)
    status = preflight.main(
        [
            "--experiment",
            "pusht/action_flow_bc_usocket_recon1_s42",
            "--output",
            str(output),
        ]
    )

    assert status == 1
    assert json.loads(output.read_text()) == {
        "error": {
            "message": "deliberate contract failure",
            "type": "PreflightError",
        },
        "schema_version": 1,
        "status": "FAIL",
    }
    assert output.read_text().endswith("\n")


def test_direct_script_cli_resolves_repository_package(tmp_path):
    output = tmp_path / "preflight.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "--experiment",
            "pusht/action_flow_bc_usocket_recon1_s42",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr or completed.stdout
    payload = json.loads(output.read_text())
    assert payload["status"] == "PASS"
    assert payload["parameters"]["field_v"]["total"] == 39_506_641
