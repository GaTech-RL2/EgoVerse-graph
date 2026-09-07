from types import SimpleNamespace
import json

from omegaconf import OmegaConf
from lightning_utilities.core.rank_zero import rank_zero_only
import pytest

from egomimic.utils.logging_utils import configure_runner_wandb, log_hyperparameters


def test_logger_keeps_resolved_manifest_provenance(monkeypatch):
    monkeypatch.setattr(rank_zero_only, "rank", 0, raising=False)
    recorded = []
    logger = SimpleNamespace(log_hyperparams=recorded.append)
    trainer = SimpleNamespace(logger=logger, loggers=[logger])
    cfg = OmegaConf.create({
        "model": {}, "data": {}, "trainer": {}, "seed": 42,
        "run_provenance": {"source_commit": "exact-source", "split_seed": "${seed}"},
    })
    log_hyperparameters({"cfg": cfg, "model": SimpleNamespace(parameters=lambda: []), "trainer": trainer})
    assert recorded[0]["run_provenance"] == {"source_commit": "exact-source", "split_seed": 42}


def wandb_config():
    return OmegaConf.create({"logger": {"wandb": {
        "_target_": "lightning.pytorch.loggers.wandb.WandbLogger",
        "id": "exact-run", "entity": "team", "project": "project", "resume": "allow",
    }}})


def test_actual_automatic_restart_requires_matching_run_and_must_resume():
    cfg = wandb_config()
    env = {"ICE_REQUEUE_OWNER": "runner", "SLURM_RESTART_COUNT": "1",
           "ICE_RESUME_CHECKPOINT": "/run/step-40000.ckpt",
           "ICE_RESUME_CHECKPOINT_METADATA_JSON": json.dumps({
               "wandb_run_id": "exact-run", "run_id": "exact-run",
               "wandb_entity": "team", "wandb_project": "project",
           })}
    configure_runner_wandb(cfg, env)
    assert cfg.logger.wandb.resume == "must"
    cfg.logger.wandb.id = "accidental-new-run"
    with pytest.raises(ValueError, match="identity"):
        configure_runner_wandb(cfg, env)


def test_fresh_start_rejects_collision_but_initial_pretrained_new_run_is_not_autorequeue():
    cfg = wandb_config()
    configure_runner_wandb(cfg, {"ICE_REQUEUE_OWNER": "runner", "SLURM_RESTART_COUNT": "0"})
    assert cfg.logger.wandb.resume == "never"
    cfg = wandb_config()
    configure_runner_wandb(cfg, {"ICE_REQUEUE_OWNER": "runner", "SLURM_RESTART_COUNT": "0",
                                "ICE_RESUME_CHECKPOINT": "/pretrained/different-run.ckpt",
                                "ICE_RESUME_CHECKPOINT_METADATA_JSON": '{"run_id":"pretrained"}'})
    assert cfg.logger.wandb.resume == "allow"


def test_debug_unchanged_and_automatic_restart_cannot_hide_missing_identity():
    debug = OmegaConf.create({"logger": None})
    configure_runner_wandb(debug, {"ICE_REQUEUE_OWNER": "runner", "SLURM_RESTART_COUNT": "1"})
    assert debug.logger is None
    cfg = wandb_config()
    with pytest.raises(ValueError, match="runner-selected checkpoint"):
        configure_runner_wandb(cfg, {"ICE_REQUEUE_OWNER": "runner", "SLURM_RESTART_COUNT": "1"})
