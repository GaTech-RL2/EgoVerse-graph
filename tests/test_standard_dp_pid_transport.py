"""Exercise the operational driver against the maintained Lightning save callback."""

import importlib.util
from pathlib import Path

import pytest
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "dp_pid_transport_driver", ROOT / "scripts/train/standard_dp_launch.py"
)
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


@pytest.fixture
def environment(monkeypatch, tmp_path):
    fields = {
        "DP_COTRAIN_STANDARD": "usocket_chain_manual4919_af_obs_multiplier_261m_v1",
        "DP_REPO": str(ROOT),
        "DP_EXPECTED_HEAD": "a" * 40,
        "DP_WANDB_ID": "test-run",
        "DP_NORM_SHA256": "b" * 64,
        "DP_SPLIT_SHA256": "c" * 64,
        "DP_CONTENT_MANIFEST": str(tmp_path / "content.json"),
        "DP_CONTENT_SHA256": "d" * 64,
        "DP_CONTENT_AGGREGATE_SHA256": "e" * 64,
    }
    for key, value in fields.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("DP_REQUEUE_OWNER", raising=False)
    return tmp_path


@pytest.mark.parametrize("phase", ["smoke", "full"])
def test_runner_uses_native_non_requeue_save_callback(environment, monkeypatch, phase):
    from egomimic.trainHydra import _instantiate_slurm_callbacks, _slurm_auto_requeue
    from egomimic.utils.slurm_requeue import SaveOnlySignalCheckpoint

    monkeypatch.setenv("DP_REQUEUE_OWNER", "runner")
    monkeypatch.setenv("ICE_REQUEUE_OWNER", "runner")
    monkeypatch.setenv("ICE_CHILD_REQUEUE_DISABLED", "1")
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    args = driver.arguments(phase, environment, environment / "norm")
    runtime = [x[2:] for x in args if x.startswith("++runtime.")]
    cfg = OmegaConf.from_dotlist(["paths.output_dir=" + str(environment)] + runtime)
    assert _slurm_auto_requeue(cfg) is False
    callbacks = _instantiate_slurm_callbacks(cfg)
    assert len(callbacks) == 1
    assert isinstance(callbacks[0], SaveOnlySignalCheckpoint)
    assert callbacks[0].checkpoint_dir == environment / "checkpoints"


def test_default_and_normalization_have_no_external_owner(environment, monkeypatch):
    assert not any(
        x.startswith("++runtime.")
        for x in driver.arguments("full", environment, environment / "norm")
    )
    monkeypatch.setenv("DP_REQUEUE_OWNER", "runner")
    assert not any(
        x.startswith("++runtime.")
        for x in driver.arguments("normalize", environment, environment / "norm")
    )


def test_unknown_owner_rejects_before_training(environment, monkeypatch):
    monkeypatch.setenv("DP_REQUEUE_OWNER", "child")
    with pytest.raises(ValueError, match="DP_REQUEUE_OWNER"):
        driver.arguments("full", environment, environment / "norm")


def test_full_restart_preserves_output_and_tracking(environment, monkeypatch):
    output = environment / "full"
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(parents=True)
    checkpoint = checkpoints / "step-1.ckpt"
    checkpoint.write_bytes(b"native-runner-selected")
    fields = {
        "DP_REQUEUE_OWNER": "runner",
        "ICE_REQUEUE_OWNER": "runner",
        "ICE_CHILD_REQUEUE_DISABLED": "1",
        "ICE_RESTART_COUNT": "1",
        "ICE_RESUME_CHECKPOINT": str(checkpoint),
        "ICE_RESUME_CHECKPOINT_SHA256": "a" * 64,
        "ICE_RESUME_GLOBAL_STEP": "1",
    }
    for key, value in fields.items():
        monkeypatch.setenv(key, value)
    driver.verify_output_directory(output, "full")
    args = driver.arguments("full", output, environment / "norm")
    assert "++logger.wandb.resume=must" in args
    assert "trainer.max_steps=80000" in args
    assert "ckpt_path=null" in args
    assert checkpoint.read_bytes() == b"native-runner-selected"


def test_full_output_requires_same_run_resume_authority(environment, monkeypatch):
    output = environment / "full"
    (output / "checkpoints").mkdir(parents=True)
    with pytest.raises(FileExistsError):
        driver.verify_output_directory(output, "full")
    monkeypatch.setenv("ICE_RESTART_COUNT", "1")
    with pytest.raises(ValueError, match="native resume authority"):
        driver.verify_output_directory(output, "full")


def test_foreign_checkpoint_cannot_unlock_existing_output(environment, monkeypatch):
    output = environment / "full"
    (output / "checkpoints").mkdir(parents=True)
    checkpoint = environment / "foreign.ckpt"
    checkpoint.write_bytes(b"other-run")
    for key, value in {
        "DP_REQUEUE_OWNER": "runner",
        "ICE_REQUEUE_OWNER": "runner",
        "ICE_CHILD_REQUEUE_DISABLED": "1",
        "ICE_RESTART_COUNT": "1",
        "ICE_RESUME_CHECKPOINT": str(checkpoint),
        "ICE_RESUME_CHECKPOINT_SHA256": "a" * 64,
        "ICE_RESUME_GLOBAL_STEP": "1",
    }.items():
        monkeypatch.setenv(key, value)
    with pytest.raises(ValueError):
        driver.verify_output_directory(output, "full")


@pytest.mark.parametrize("case", ["valid", "missing_python3", "wrong_python3"])
def test_pid_runner_pins_validator_venv(monkeypatch, tmp_path, case):
    import hashlib
    import os
    import runpy
    import sys

    shim = ROOT / "scripts/ice/ice_requeue_trainer_pid.py"
    native = ROOT / "scripts/ice/ice_requeue_runner.py"
    adapter = ROOT / "scripts/ice/trainer_pid_signal.py"
    runtime = tmp_path / "venv/bin"
    runtime.mkdir(parents=True)
    python = runtime / "python"
    python.write_text("pinned")
    python.chmod(0o700)
    if case == "valid":
        (runtime / "python3").symlink_to(python)
    elif case == "wrong_python3":
        (runtime / "python3").write_text("foreign")
    for key, value in {
        "ICE_CANONICAL_REQUEUE_RUNNER": str(native),
        "ICE_TRAINER_REPO": str(ROOT),
        "ICE_NATIVE_RUNNER_SHA256": hashlib.sha256(native.read_bytes()).hexdigest(),
        "ICE_TRAINER_SIGNAL_ADAPTER_SHA256": hashlib.sha256(
            adapter.read_bytes()
        ).hexdigest(),
        "ICE_TRAINER_PYTHON": str(python),
        "PATH": "/foreign/login/bin",
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(sys, "argv", [str(shim), "--", "child"])
    captured = []
    original = runpy.run_path
    monkeypatch.setattr(runpy, "run_path", lambda *args, **kw: captured.append(args))
    if case == "valid":
        original(str(shim))
        assert os.environ["PATH"] == str(runtime) + ":/foreign/login/bin"
        assert captured == [(str(native),)]
        assert sys.argv[1:3] == ["--scancel", str(adapter)]
    else:
        with pytest.raises(AssertionError):
            original(str(shim))
        assert not captured
