"""Real Hydra/OmegaConf restoration, no model/data/cluster execution."""

import hashlib
import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest
from hydra import __version__ as hydra_version
from hydra.conf import HydraConf
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

from egomimic.benchmarks.libero.native_launch_profiles import PROFILES
from egomimic.benchmarks.libero.saved_hydra_context import load_saved_native_config


def setup(tmp_path, suite):
    repo = tmp_path / "source"
    repo.mkdir()
    run = tmp_path / "run"
    (run / ".hydra").mkdir(parents=True)
    p = PROFILES[suite]
    receipt = tmp_path / "receipt.json"
    logical_sha256 = "a" * 64
    receipt.write_text(
        json.dumps(
            dict(
                suite=suite,
                replay_path="/replay",
                dataset_logical_sha256=logical_sha256,
            )
        )
    )
    proof = tmp_path / "physical.json"
    proof.write_text("{}")

    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    binding = dict(
        source_commit="head",
        data_root="/replay",
        dataset_receipt_path=str(receipt),
        dataset_receipt_sha256=sha(receipt),
        physical_proof_path=str(proof),
        physical_proof_sha256=sha(proof),
    )
    config = dict(
        name=p.name,
        benchmark=dict(suite=suite, dataset="${oc.env:" + p.replay_environment + "}"),
        run_provenance=dict(
            source_commit="head",
            dataset_sha256=logical_sha256,
            dataset_content_aggregate_sha256=logical_sha256,
        ),
        norm_stats=dict(native_saved_state_binding=binding),
        data={
            group: {
                "libero_panda": {
                    "resolver": dict(
                        suite="${benchmark.suite}",
                        folder_path="${oc.env:" + p.replay_environment + "}",
                    )
                }
            }
            for group in ("train_datasets", "valid_datasets")
        },
        evaluator=dict(
            energy_seed_bank_path="${hydra:runtime.cwd}/egomimic/hydra_configs/evaluator/energy_score_seed_bank_k32_v1.json"
        ),
    )
    OmegaConf.save(OmegaConf.create(config), run / ".hydra/config.yaml")
    hydra = OmegaConf.structured(HydraConf)
    hydra.runtime.cwd = str(repo)
    hydra.runtime.output_dir = str(run)
    hydra.runtime.version = hydra_version
    saved = OmegaConf.to_container(hydra, resolve=False)
    saved["env"] = {}
    OmegaConf.save(OmegaConf.create({"hydra": saved}), run / ".hydra/hydra.yaml")
    return repo, run, p


@pytest.mark.parametrize("suite", list(PROFILES))
def test_saved_context_preserves_all_fields_and_missing_env(
    tmp_path, monkeypatch, suite
):
    repo, run, p = setup(tmp_path, suite)
    monkeypatch.delenv(p.replay_environment, raising=False)
    with patch("subprocess.check_output", return_value="head\n"):
        cfg = load_saved_native_config(run, repo)
    assert os.environ[p.replay_environment] == "/replay"
    assert cfg.benchmark.dataset == "/replay"
    assert OmegaConf.get_type(HydraConfig.get()) is HydraConf
    assert dict(HydraConfig.get().env) == {}
    saved = OmegaConf.to_container(
        OmegaConf.load(run / ".hydra/hydra.yaml").hydra, resolve=False, enum_to_str=True
    )
    actual = OmegaConf.to_container(HydraConfig.get(), resolve=False, enum_to_str=True)
    assert all(actual[k] == v for k, v in saved.items())


@pytest.mark.parametrize(
    "fault",
    ["conflict", "unknown_env", "cwd", "output", "source", "receipt", "resolver"],
)
def test_context_rejects_identity_drift(tmp_path, monkeypatch, fault):
    repo, run, p = setup(tmp_path, "libero_spatial")
    monkeypatch.delenv(p.replay_environment, raising=False)
    cfg = OmegaConf.load(run / ".hydra/config.yaml")
    saved = OmegaConf.load(run / ".hydra/hydra.yaml")
    if fault == "conflict":
        monkeypatch.setenv(p.replay_environment, "/wrong")
    if fault == "unknown_env":
        cfg.extra = "${oc.env:UNEXPECTED}"
    if fault == "cwd":
        saved.hydra.runtime.cwd = "/wrong"
    if fault == "output":
        saved.hydra.runtime.output_dir = "/wrong"
    if fault == "source":
        cfg.run_provenance.source_commit = "wrong"
    if fault == "receipt":
        Path(cfg.norm_stats.native_saved_state_binding.dataset_receipt_path).write_text(
            "{}"
        )
    if fault == "resolver":
        cfg.data.valid_datasets.libero_panda.resolver.folder_path = "/wrong"
    OmegaConf.save(cfg, run / ".hydra/config.yaml")
    OmegaConf.save(saved, run / ".hydra/hydra.yaml")
    with (
        patch("subprocess.check_output", return_value="head\n"),
        pytest.raises(ValueError),
    ):
        load_saved_native_config(run, repo)


@pytest.mark.parametrize("fault", [None, "context", "source", "resolver"])
def test_joint_saved_context_uses_complete_context_without_historical_binding(
    tmp_path, monkeypatch, fault
):
    import hydra

    from egomimic.benchmarks.libero.native_launch_profiles import joint_phase_argv

    repo = Path(__file__).resolve().parents[1]
    run = tmp_path / "joint"
    (run / ".hydra").mkdir(parents=True)
    context = tmp_path / "context.json"
    split = tmp_path / "split.json"
    context.write_text("{}")
    split.write_text("{}")

    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    args = dict(
        source_root=str(repo),
        original_root=str(tmp_path / "original"),
        sharpa_root=str(tmp_path / "sharpa"),
        output_root=str(run),
        context_path=str(context),
        split_path=str(split),
        split_sha256=sha(split),
        normalization_sha256=sha(context),
        source_commit="c" * 40,
        diagnostic_seed_bank=str(tmp_path / "seeds.json"),
        wandb_entity="test",
        wandb_project="test",
        wandb_run_id="joint",
    )
    with hydra.initialize_config_dir(
        version_base="1.2", config_dir=str(repo / "egomimic/hydra_configs")
    ):
        cfg = hydra.compose(
            config_name="train_zarr_cartesian",
            return_hydra_config=True,
            overrides=joint_phase_argv(phase="smoke", **args)[1:]
            + ["hydra/launcher=basic"],
        )
        cfg.hydra.runtime.cwd = str(repo)
        cfg.hydra.runtime.output_dir = str(run)
        HydraConfig.instance().set_config(cfg)
        payload = OmegaConf.to_container(
            OmegaConf.masked_copy(cfg, [k for k in cfg if k != "hydra"]), resolve=True
        )
        saved = OmegaConf.to_container(cfg.hydra, resolve=False)
    if fault == "context":
        context.write_text("changed")
    if fault == "source":
        payload["run_provenance"]["source_commit"] = "wrong"
    if fault == "resolver":
        payload["data"]["valid_datasets"]["libero_panda_sharpa"]["resolver"][
            "folder_path"
        ] = "/wrong"
    OmegaConf.save(OmegaConf.create(payload), run / ".hydra/config.yaml")
    OmegaConf.save(OmegaConf.create({"hydra": saved}), run / ".hydra/hydra.yaml")
    with patch("subprocess.check_output", return_value="c" * 40 + "\n"):
        if fault:
            with pytest.raises(ValueError):
                load_saved_native_config(run, repo)
        else:
            loaded = load_saved_native_config(run, repo)
            assert OmegaConf.to_container(loaded, resolve=True) == payload


@pytest.mark.parametrize("fault", [None, "rolling", "wrong_callback", "wrong_path"])
def test_joint_checkpoint_distinguishes_configured_terminal_from_rolling_alias(
    tmp_path, fault
):
    from scripts.train.verify_action_flow_training_smoke import (
        _joint_smoke_checkpoint_path,
    )

    cadence = tmp_path / "epoch-0000-step-000000008.ckpt"
    terminal = tmp_path / "last.ckpt"
    cadence.write_bytes(b"cadence")
    terminal.write_bytes(b"terminal")
    cfg = OmegaConf.create(
        {
            "callbacks": {
                "model_checkpoint": {"save_last": False},
                "ema": {
                    "_target_": "egomimic.pl_utils.oat_matched_training.ActionFlowFixedEMACallback",
                    "final_checkpoint_path": str(terminal),
                },
            }
        }
    )
    if fault == "rolling":
        cfg.callbacks.model_checkpoint.save_last = True
    if fault == "wrong_callback":
        cfg.callbacks.ema._target_ = "unknown.Callback"
    if fault == "wrong_path":
        cfg.callbacks.ema.final_checkpoint_path = "/wrong"
    if fault:
        with pytest.raises(Exception, match="rolling last"):
            _joint_smoke_checkpoint_path(tmp_path, cfg)
    else:
        assert _joint_smoke_checkpoint_path(tmp_path, cfg) == cadence
        assert terminal.read_bytes() == b"terminal"
    with pytest.raises(Exception, match="rolling last"):
        _joint_smoke_checkpoint_path(tmp_path)


def test_separate_verifier_finalizes_through_same_loaded_helpers(tmp_path, monkeypatch):
    from egomimic.benchmarks.libero import saved_hydra_context as loader
    from scripts.train import libero_native_launch_contract as producer
    from scripts.train import verify_action_flow_training_smoke as shared

    training, verifier, run = [tmp_path / p for p in ("training", "verifier", "run")]
    for p in (training, verifier, run):
        p.mkdir()
    preflight = tmp_path / "preflight.json"
    preflight.write_text("{}")
    runtime = tmp_path / "runtime.json"
    runtime.write_text("{}")
    monkeypatch.setattr(shared, "REPOSITORY_ROOT", verifier)
    monkeypatch.setattr(shared, "VERIFIER_PROVENANCE", None)
    monkeypatch.setattr(
        shared,
        "_git_head",
        lambda repository_root=None: "a" * 40
        if repository_root == verifier
        else "b" * 40,
    )
    monkeypatch.setattr(shared.subprocess, "check_output", lambda *a, **k: "")
    monkeypatch.setattr(shared.subprocess, "run", lambda *a, **k: None)
    monkeypatch.setenv("SLURM_STEP_ID", "0")
    monkeypatch.setenv("AF_PREFLIGHT_RESULT", str(preflight))
    calls = []

    def load(root, repo):
        assert root == run and repo == training
        return OmegaConf.create({"original": True})

    def evidence(cfg, root, **kwargs):
        assert isinstance(cfg, dict) and cfg["original"] and root == run
        json.dumps(cfg, allow_nan=False)
        assert kwargs["shared"] is shared
        calls.append("finalize")
        return {"actual": True}

    monkeypatch.setattr(loader, "load_saved_native_config", load)
    monkeypatch.setattr(producer, "joint_smoke_evidence", evidence)
    monkeypatch.setattr(producer, "write", lambda path, value: calls.append("write"))
    monkeypatch.setattr(
        shared,
        "verify_smoke",
        lambda *a, **k: calls.append("verify") or {"run_dir": str(run)},
    )
    assert (
        shared.main(
            [
                str(run),
                "--experiment",
                "sim/libero/action_flow_libero_spatial_sharpa_gripper_keypoints_s42",
                "--expected-head",
                "b" * 40,
                "--training-repo",
                str(training),
                "--expected-verifier-head",
                "a" * 40,
                "--finalize-runtime-lock",
                str(runtime),
                "--expected-preflight-sha256",
                hashlib.sha256(preflight.read_bytes()).hexdigest(),
            ]
        )
        == 0
    )
    assert calls == ["finalize", "write", "verify"]


def test_joint_artifacts_bind_cadence_epoch_and_preserve_postfit(tmp_path, monkeypatch):
    import torch

    from scripts.train import verify_action_flow_training_smoke as shared

    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    run = tmp_path / "run"
    checkpoint = run / "checkpoints/epoch-0000-step-000000008.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"filename-only-selection")
    root = run / "artifacts"
    for epoch in (0, 1):
        path = root / f"epoch-{epoch}-step-8/rank-0-batch-0.pt"
        path.parent.mkdir(parents=True)
        torch.save({"global_step": 8, "epoch": epoch}, path)
    epoch = shared._joint_smoke_artifact_epoch(run, None)
    assert epoch == 0
    selected, payload = shared._step_artifact(
        root, label="joint", expected_step=8, expected_epoch=epoch
    )
    assert selected.parent.name == "epoch-0-step-8" and payload["epoch"] == 0
    assert (root / "epoch-1-step-8/rank-0-batch-0.pt").is_file()
    with pytest.raises(shared.SmokeVerificationError, match="expected one step-8"):
        shared._step_artifact(root, label="joint", expected_step=8)
    torch.save({"global_step": 8, "epoch": 1}, selected)
    with pytest.raises(shared.SmokeVerificationError, match="epoch differs"):
        shared._step_artifact(root, label="joint", expected_step=8, expected_epoch=0)
