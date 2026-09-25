from pathlib import Path

import pytest
from lightning import LightningModule
from omegaconf import OmegaConf

from egomimic.pl_utils.pl_model import ModelWrapper
from egomimic.utils.logging_utils import configure_runner_wandb


def _cfg(tmp_path: Path, *, run_id="stable-run", checkpoint=None):
    return OmegaConf.create(
        {
            "paths": {"output_dir": str(tmp_path)},
            "ckpt_path": checkpoint,
            "logger": {
                "wandb": {
                    "_target_": "lightning.pytorch.loggers.wandb.WandbLogger",
                    "id": run_id,
                    "entity": "rl2-group",
                    "project": "abc-yam",
                }
            },
        }
    )


def test_explicit_checkpoint_preserves_fresh_wandb_identity(tmp_path):
    checkpoint = tmp_path / "source.ckpt"
    checkpoint.touch()
    cfg = _cfg(tmp_path, checkpoint=str(checkpoint))
    cfg.logger.wandb.resume = "never"
    (tmp_path / "wandb-run-id.txt").write_text("old-run\n")

    configure_runner_wandb(cfg, {"SLURM_RESTART_COUNT": "0"})

    assert cfg.logger.wandb.id == "stable-run"
    assert cfg.logger.wandb.resume == "never"


def test_lightning_restart_recovers_sidecar_and_last_checkpoint(tmp_path):
    checkpoint = tmp_path / "checkpoints" / "last.ckpt"
    checkpoint.parent.mkdir()
    checkpoint.touch()
    (tmp_path / "wandb-run-id.txt").write_text("original-run\n")
    cfg = _cfg(tmp_path, run_id="${now:%Y%m%d}")

    configure_runner_wandb(cfg, {"SLURM_RESTART_COUNT": "2"})

    assert cfg.ckpt_path == str(checkpoint)
    assert cfg.logger.wandb.id == "original-run"
    assert cfg.logger.wandb.resume == "must"


def test_resume_rejects_conflicting_wandb_identities(tmp_path):
    checkpoint = tmp_path / "source.ckpt"
    checkpoint.touch()
    (tmp_path / "wandb-run-id.txt").write_text("sidecar-run\n")
    cfg = _cfg(tmp_path, run_id="configured-run", checkpoint=str(checkpoint))

    with pytest.raises(ValueError, match="conflicting W&B run identities"):
        configure_runner_wandb(cfg, {
            "SLURM_RESTART_COUNT": "1", "ICE_RESUME_CHECKPOINT": str(checkpoint)
        })


def test_offline_evaluation_keeps_its_separate_wandb_run(tmp_path):
    cfg = _cfg(tmp_path, run_id="validation-run", checkpoint="source.ckpt")
    cfg.mode = "eval"
    configure_runner_wandb(cfg, {"SLURM_RESTART_COUNT": "0"})
    assert cfg.logger.wandb.id == "validation-run"
    assert "resume" not in cfg.logger.wandb


def test_lightning_resume_recovers_actual_wandb_artifact_filename(tmp_path):
    artifact = tmp_path / "wandb/run-20260922_120000-original-run/run-original-run.wandb"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"wandb")
    cfg = _cfg(tmp_path, run_id="${now:%Y%m%d}", checkpoint="source.ckpt")
    configure_runner_wandb(cfg, {
        "SLURM_RESTART_COUNT": "1", "ICE_RESUME_CHECKPOINT": "source.ckpt"
    })
    assert cfg.logger.wandb.id == "original-run"
    assert cfg.logger.wandb.resume == "must"


def test_model_wrapper_disables_dataloader_suffix_by_default(monkeypatch):
    observed = []

    def fake_log(self, *args, **kwargs):
        observed.append(kwargs)

    monkeypatch.setattr(LightningModule, "log", fake_log)
    wrapper = ModelWrapper.__new__(ModelWrapper)
    LightningModule.__init__(wrapper)

    wrapper.log("metric", 1.0)
    wrapper.log("explicit", 1.0, add_dataloader_idx=True)

    assert observed[0]["add_dataloader_idx"] is False
    assert observed[1]["add_dataloader_idx"] is True


def test_model_wrapper_log_dict_disables_dataloader_suffix(monkeypatch):
    observed = {}

    def fake_log_dict(self, *args, **kwargs):
        observed.update(kwargs)

    monkeypatch.setattr(LightningModule, "log_dict", fake_log_dict)
    wrapper = ModelWrapper.__new__(ModelWrapper)
    LightningModule.__init__(wrapper)

    wrapper.log_dict({"metric": 1.0})

    assert observed["add_dataloader_idx"] is False
