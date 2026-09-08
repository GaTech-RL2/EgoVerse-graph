from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "train" / "launch_action_flow_usocket.sbatch"
DATASET_VALIDATOR = ROOT / "scripts" / "ice" / "validate_planar_dataset.py"


def _source() -> str:
    return LAUNCHER.read_text()


def test_launcher_has_valid_shell_syntax():
    completed = subprocess.run(
        ["bash", "-n", str(LAUNCHER)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "#SBATCH --kill-on-invalid-dep=yes" in _source()


def test_launcher_uses_the_maintained_dataset_validator_success_token():
    validator_source = DATASET_VALIDATOR.read_text()
    match = re.search(r'"status":\s*"([A-Z_]+)"', validator_source)
    assert match is not None
    assert f'dataset_report["status"] == "{match.group(1)}"' in _source()


def test_launcher_is_one_portable_fail_closed_contract():
    source = _source()
    assert "AF_EXPECTED_HEAD" in source
    assert "AF_EXPECTED_LAUNCHER_SHA256" in source
    assert "AF_EXPECTED_SPLIT_MANIFEST_SHA256" in source
    assert "AF_EXPECTED_CONTENT_MANIFEST_SHA256" in source
    assert "AF_EXPECTED_DATASET_CONTENT_AGGREGATE_SHA256" in source
    assert "AF_EXPECTED_NORM_SHA256" in source
    assert "AF_PREFLIGHT_RESULT" in source
    assert "AF_EXPECTED_PREFLIGHT_SHA256" in source
    assert "validate_action_flow_config.py" in source
    assert "capture_runtime_lock.py" in source
    assert "validate_slurm_job_contract.py" in source
    assert "check_checkpoint_storage.py" in source
    assert "--gres=gpu:1 --constraint='H100|H200'" in source
    assert '--allowed-gpu-name "NVIDIA H100 80GB HBM3"' in source
    assert '--allowed-gpu-name "NVIDIA H200"' in source
    assert "trainer.devices=1" in source
    assert "trainer.strategy=auto" in source
    assert "#SBATCH --requeue" in source
    assert "#SBATCH --signal=B:USR1@600" in source
    assert "/coc/" not in source
    assert "/storage/ice" not in source
    assert "/storage/project" not in source
    assert 'absolute_path AF_EXTRA_PYTHONPATH "$AF_EXTRA_PYTHONPATH"' in source
    assert (
        'PYTHONPATH="$AF_REPO${AF_EXTRA_PYTHONPATH:+:$AF_EXTRA_PYTHONPATH}"'
        in source
    )


def test_launcher_accepts_only_the_approved_sweep_and_pins_training_semantics():
    source = _source()
    assert "pusht/action_flow_bc_usocket_recon1_s42" in source
    assert "pusht/action_flow_bc_usocket_recon10_s42" in source
    assert "pusht/action_flow_bc_usocket_recon100_s42" in source
    assert "AF_EXPECTED_CONFIG_NAME=action_flow_bc_usocket_recon1_s42" in source
    assert "AF_EXPECTED_CONFIG_NAME=action_flow_bc_usocket_recon10_s42" in source
    assert "AF_EXPECTED_CONFIG_NAME=action_flow_bc_usocket_recon100_s42" in source
    assert "expected_name = {" not in source
    assert "MAX_STEPS=240000" in source
    assert "VALIDATE_EVERY=10000" in source
    assert "CHECKPOINT_EVERY=40000" in source
    assert "TELEMETRY_EVERY=100" in source
    assert (
        "data.train_dataloader_params.pushshapes_sim_u_socket.batch_size=32" in source
    )
    assert (
        "data.valid_dataloader_params.pushshapes_sim_u_socket.batch_size=16" in source
    )
    assert "ckpt_path=null" in source
    assert "norm_stats.precomputed_norm_path=$AF_NORM_STATS_PATH" in source
    assert "++run_provenance.source_commit=$AF_EXPECTED_HEAD" in source
    assert "++run_provenance.content_manifest_sha256=" in source
    assert "++run_provenance.dataset_content_aggregate_sha256=" in source
    assert "++run_provenance.normalization_sha256=$AF_EXPECTED_NORM_SHA256" in source
    assert (
        "++run_provenance.preflight_result_sha256=$AF_EXPECTED_PREFLIGHT_SHA256"
        in source
    )
    assert '"runtime_lock_sha256": digest(runtime_lock_path)' in source
    assert 'current["installed_distributions"]["canonical_sha256"]' in source
    assert "[run-preflight] PASS kind={run_kind} parameters={count}" in source
    assert "assert count == 50_725_221" in source


def test_smoke_runs_optimizer_validation_checkpoint_and_verifier():
    source = _source()
    assert "MAX_STEPS=2" in source
    assert "VALIDATE_EVERY=1" in source
    assert "LIMIT_VAL_BATCHES=1" in source
    assert "CHECKPOINT_EVERY=1" in source
    assert "TELEMETRY_EVERY=2" in source
    assert "verify_action_flow_training_smoke.py" in source
    assert "--planned-checkpoint-count 14" in source
    assert '--expected-constraint "$AF_EXPECTED_GPU_CONSTRAINT"' in source
    assert "--expected-reconstruction-weight" in source
    assert "--expected-config-sha256" in source
    assert (
        '--expected-config-sha256 "$(sha256 '
        '"$AF_OUTPUT_DIR/.hydra/config.yaml")"' in source
    )
    assert "--expected-normalization-sha256" in source
    assert "--expected-content-manifest-sha256" in source
    assert "--expected-dataset-content-aggregate-sha256" in source
    assert "--expected-preflight-sha256" in source
    assert "callbacks.model_checkpoint.save_last=link" in source
    assert "single-gpu,$AF_RUN_KIND" in source


def test_full_and_smoke_share_the_requeue_and_strict_checkpoint_path():
    source = _source()
    assert "ice_requeue_runner.py" in source
    assert "validate_lightning_checkpoint.py" in source
    assert "--checkpoint-validator" in source
    assert "--requeue-owner runner" in source
    assert "--confirm-child-requeue-disabled" in source
    assert "runtime.slurm_requeue_owner=runner" in source
    assert "runtime.slurm_save_signal=SIGUSR2" in source


def test_full_mode_requires_the_exact_passed_smoke_gate():
    source = _source()
    assert "required AF_SMOKE_RESULT" in source
    assert "required AF_EXPECTED_SMOKE_SHA256" in source
    assert 'payload["status"] == "PASS"' in source
    assert 'payload["experiment"] == experiment' in source
    assert 'identities["repo_head"] == head' in source
    assert 'identities["split_manifest_sha256"] == split_sha' in source
    assert 'identities["normalization_sha256"] == norm_sha' in source
    assert 'identities["content_manifest_sha256"] == content_sha' in source
    assert 'payload["preflight"]["sha256"] == preflight_sha' in source
    assert "++run_provenance.smoke_result_sha256=$AF_EXPECTED_SMOKE_SHA256" in source
    assert 'cp -- "$AF_SMOKE_RESULT" "$ATTEMPT/SMOKE_RESULT.json"' in source


def test_second_preflight_can_reuse_the_first_train_only_normalization():
    source = _source()
    assert 'if test -z "${AF_NORM_STATS_PATH:-}"; then' in source
    assert 'test -s "$EFFECTIVE_NORM_FILE"' in source
    assert 'normalization SHA-256 mismatch' in source


@pytest.mark.parametrize(
    ("kind", "profile", "constraint", "success", "gpu_name"),
    [
        ("full", "h100-h200", "H100|H200", True, "NVIDIA H200"),
        ("smoke", "h100-h200", "H100|H200", True, "NVIDIA H100"),
        ("smoke", "smoke-bf16", "A40", True, "NVIDIA A40"),
        ("smoke", "smoke-bf16", "A100-40GB|A100-80GB", True, "NVIDIA A100"),
        ("smoke", "smoke-bf16", "L40S", True, "NVIDIA L40S"),
        ("full", "smoke-bf16", "H100|H200", False, ""),
        ("full", "h100-h200", "A40", False, ""),
        ("smoke", "h100-h200", "A40", False, ""),
        ("smoke", "smoke-bf16", "V100", False, ""),
    ],
)
def test_actual_launcher_profile_boundary(kind, profile, constraint, success, gpu_name):
    source = _source()
    begin = source.index('case "$AF_RUN_KIND" in')
    end = source.index('if test "$AF_LAUNCH_MODE" = run;', begin)
    script = "set -eu\ndie() { echo \"$*\" >&2; exit 64; }\n"
    script += source[begin:end] + "\ndeclare -p GPU_NAME_ARGS\n"
    result = subprocess.run(
        ["bash", "-c", script],
        env={**os.environ, "AF_RUN_KIND": kind, "AF_GPU_PROFILE": profile,
             "AF_EXPECTED_GPU_CONSTRAINT": constraint},
        capture_output=True, text=True,
    )
    assert (result.returncode == 0) is success, result.stderr
    if success:
        assert gpu_name in result.stdout


def test_preflight_reuses_hashed_dataset_evidence_and_removes_logger_group():
    source = _source()
    assert "AF_CACHED_DATASET_VALIDATION" in source
    assert "AF_EXPECTED_CACHED_DATASET_VALIDATION_SHA256" in source
    assert 'payload["status"] == "DATASET_VALIDATED"' in source
    assert "'~logger'" in source
    assert "logger=null" not in source
    assert "++logger.wandb.offline=false" in source
    assert "++logger.wandb.name=$AF_WANDB_NAME" in source
    assert "++logger.wandb.resume=allow" in source
    for key in ("entity", "project", "group", "id", "tags"):
        assert f"++logger.wandb.{key}=" in source
