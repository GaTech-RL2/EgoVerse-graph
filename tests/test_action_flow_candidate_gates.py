"""Method-specific gates: real configs plus tiny, honest telemetry receipts."""

import hashlib
import re
import subprocess

import pytest
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf, open_dict

from tools.validate_action_flow_config import (
    CANDIDATE_METHODS,
    GRAPH_METHOD,
    LIKELIHOOD_METHOD,
    STOPGRAD_METHOD,
    PreflightError,
    compose_experiment,
    validate_experiment,
    validate_method_contract,
)
from test_verify_action_flow_training_smoke import MODULE, HEAD, _history_row


@pytest.mark.parametrize("experiment", CANDIDATE_METHODS)
def test_real_candidate_config_and_parameter_manifest(experiment):
    report, _ = validate_experiment(experiment)
    assert report["status"] == "PASS"
    assert report["action_flow_method"] == CANDIDATE_METHODS[experiment]
    assert report["topology"]["shared_field_instance"]
    assert report["topology"]["shared_decoder_instance"]
    assert len(report["topology"]["train_order"]) == 8
    assert (
        sum(
            v["total"] for k, v in report["parameters"].items() if k != "pipeline_total"
        )
        == report["parameters"]["pipeline_total"]["total"]
    )


@pytest.mark.parametrize("experiment", CANDIDATE_METHODS)
def test_candidate_two_optimizer_update_config_gate(experiment, tmp_path, monkeypatch):
    cfg = compose_experiment(experiment)
    normalization = tmp_path / "norm_stats.json"
    normalization.write_text("{}")
    norm_hash = hashlib.sha256(normalization.read_bytes()).hexdigest()
    with open_dict(cfg):
        cfg.trainer.max_steps = 2
        cfg.trainer.val_check_interval = 1
        cfg.trainer.limit_val_batches = 1
        cfg.callbacks.model_checkpoint.every_n_train_steps = 1
        cfg.model.gradient_telemetry_cadence = 2
        cfg.norm_stats.precomputed_norm_path = str(normalization)
        cfg.run_provenance.source_commit = HEAD
        cfg.run_provenance.normalization_sha256 = norm_hash
        cfg.evaluator.artifact_root = str(tmp_path / "energy")
        if CANDIDATE_METHODS[experiment] != LIKELIHOOD_METHOD:
            cfg.evaluator.action_flow_diagnostics.artifact_root = str(
                tmp_path / "diagnostics"
            )
        cfg.hydra.runtime.cwd = str(MODULE.REPOSITORY_ROOT)
        cfg.hydra.runtime.output_dir = str(tmp_path)
    HydraConfig.instance().set_config(cfg)
    selected = OmegaConf.masked_copy(cfg, [key for key in cfg if key != "hydra"])
    path = tmp_path / "config.yaml"
    OmegaConf.save(
        OmegaConf.create(OmegaConf.to_container(selected, resolve=True)), path
    )
    monkeypatch.setattr(MODULE, "_git_head", lambda: HEAD)
    MODULE._validate_config(
        config_path=path,
        experiment=experiment,
        run_dir=tmp_path,
        expected_head=HEAD,
        expected_config_sha256=None,
        expected_split_sha256=None,
        expected_normalization_sha256=norm_hash,
    )


@pytest.mark.parametrize(
    "experiment,path,value",
    [
        (
            "pusht/action_flow_bc_usocket_latent_fm_sg_recon1_s42",
            "model.pipeline.stages.5.flow_clean_gradient_mode",
            "attached",
        ),
        (
            "pusht/action_flow_bc_usocket_graph_section_s42",
            "model.reconstruction_weight",
            1.0,
        ),
        (
            "pusht/action_flow_bc_usocket_bridge_likelihood_s42",
            "run_provenance.inference.sampler",
            "reverse_euler",
        ),
        (
            "pusht/action_flow_bc_usocket_bridge_likelihood_s42",
            "model.pipeline.stages.7.tau",
            0.01,
        ),
    ],
)
def test_method_contract_rejects_scientific_identity_drift(experiment, path, value):
    cfg = compose_experiment(experiment)
    OmegaConf.update(cfg, path, value)
    with pytest.raises(PreflightError):
        validate_method_contract(cfg, experiment)


def test_stopgrad_telemetry_requires_empty_fm_reconstruction_intersection():
    row = _history_row()
    for key in list(row):
        if key.endswith("/FM__Reconstruction"):
            row[key] = 0.0
    row["Train/ActionFlow/Compute/FieldForwardCallsPerStep"] = 2.0
    row["Train/ActionFlow/Compute/FieldSampleEquivalentsPerStep"] = 28.0
    MODULE._validate_history({1: row}, method=STOPGRAD_METHOD)
    row["Train/ActionFlow/GradientIntersectionParameterCount/FM__Reconstruction"] = 1.0
    with pytest.raises(MODULE.SmokeVerificationError, match="reachability"):
        MODULE._validate_history({1: row}, method=STOPGRAD_METHOD)


def test_graph_telemetry_does_not_invent_reconstruction_gradients():
    row = {
        key: value
        for key, value in _history_row().items()
        if not ("/Gradient" in key and "Reconstruction" in key)
        and "/Alignment/CK" not in key
    }
    row["Train/ActionFlow/TotalLoss_step"] = 2.5
    row[f"Train/ActionFlow/TotalLoss/{MODULE.SOURCE_LABEL}_step"] = 2.5
    row["Valid/ActionFlow/TotalLoss"] = 3.0
    MODULE._validate_history({1: row}, method=GRAPH_METHOD, reconstruction_weight=0.0)


def test_likelihood_history_requires_real_nll_and_common_validation_only():
    row = {
        key: value
        for key, value in _history_row().items()
        if "/ActionFlow/" not in key or "/Compute/" in key
    }
    for component in ("InteriorBridgeNLL", "BoundaryNLL", "TotalLoss"):
        value = 2.5 if component == "TotalLoss" else 1.25
        row[f"Train/ActionFlow/{component}_step"] = value
        row[f"Train/ActionFlow/{component}/{MODULE.SOURCE_LABEL}_step"] = value
        row[f"Valid/ActionFlow/{component}"] = value
    for label in ("InteriorBridgeNLL", "BoundaryNLL"):
        row[f"Train/ActionFlow/GradientNorm/{label}"] = 0.5
        row[f"Train/ActionFlow/GradientParameterCount/{label}"] = 100.0
    pair = "InteriorBridgeNLL__BoundaryNLL"
    row[f"Train/ActionFlow/GradientCosine/{pair}"] = 0.25
    row[f"Train/ActionFlow/GradientCosineDefined/{pair}"] = 1.0
    row[f"Train/ActionFlow/GradientIntersectionParameterCount/{pair}"] = 50.0
    row["Train/ActionFlow/Compute/FieldSampleEquivalentsPerStep"] = 15.0
    row["Train/ActionFlow/Compute/DecoderJVPCallsPerStep"] = 0.0
    MODULE._validate_history({1: row}, method=LIKELIHOOD_METHOD)
    del row["Valid/EnergyScore@32"]
    with pytest.raises(MODULE.SmokeVerificationError):
        MODULE._validate_history({1: row}, method=LIKELIHOOD_METHOD)


@pytest.mark.parametrize("method", [STOPGRAD_METHOD, GRAPH_METHOD, LIKELIHOOD_METHOD])
def test_checkpoint_gradient_receipt_uses_exact_method_routes(method):
    named = [
        (f"nets.pipeline.stages.{stage}.weight", torch.nn.Parameter(torch.ones(2)))
        for stage in (0, 3, 5, 6)
    ]
    groups = {
        "FM": [0, 2] if method == STOPGRAD_METHOD else [0, 1, 2],
        "Reconstruction": [1, 3],
        "ActionVelocity": [0, 1, 2, 3],
    }
    if method == GRAPH_METHOD:
        del groups["Reconstruction"]
    elif method == LIKELIHOOD_METHOD:
        groups = {"InteriorBridgeNLL": [0, 1, 2], "BoundaryNLL": [0, 1, 2, 3]}
    routes = {
        label: [
            {"name": named[i][0], "shape": [2], "numel": 2, "dtype": "torch.float32"}
            for i in indices
        ]
        for label, indices in groups.items()
    }
    labels = list(groups)
    core = {
        "schema_version": 1,
        "routes": routes,
        "route_sha256": {
            k: MODULE._canonical_json_sha256(v) for k, v in routes.items()
        },
        "intersections": {
            f"{left}__{right}": [
                named[i][0] for i in groups[left] if i in groups[right]
            ]
            for index, left in enumerate(labels)
            for right in labels[index + 1 :]
        },
    }
    manifest = {**core, "manifest_sha256": MODULE._canonical_json_sha256(core)}
    MODULE._validate_gradient_route_manifest(manifest, named, method=method)


@pytest.mark.parametrize(
    "requested,selected,accepted",
    [
        ("gpu-h100,gpu-h200", "gpu-h100", True),
        ("gpu-h100,gpu-h200", "gpu-h200", True),
        ("ice-gpu", "ice-gpu", True),
        ("gpu-h100,gpu-h200", "gpu", False),
        ("gpu-h100", "gpu-h200", False),
        ("gpu-h100,", "gpu-h100", False),
    ],
)
def test_portable_launcher_accepts_only_explicit_partition_members(
    requested, selected, accepted
):
    source = (
        MODULE.REPOSITORY_ROOT / "scripts/train/launch_action_flow_usocket.sbatch"
    ).read_text()
    function = re.search(r"partition_allowed\(\) \{.*?\n\}", source, re.S).group()
    result = subprocess.run(
        [
            "bash",
            "-c",
            function + '\npartition_allowed "$1" "$2"',
            "test",
            requested,
            selected,
        ]
    )
    assert (result.returncode == 0) is accepted
