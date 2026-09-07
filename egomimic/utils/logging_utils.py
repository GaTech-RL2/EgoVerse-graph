import json
import os
from typing import Any, Dict

from lightning_utilities.core.rank_zero import rank_zero_only
from omegaconf import DictConfig, OmegaConf, open_dict

from egomimic.utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


def configure_runner_wandb(cfg: DictConfig, environ=None) -> None:
    """Bind automatic Slurm restart logging to the runner's verified identity.

    An initial checkpoint can seed a new experiment; only an actual automatic
    restart requires the previous W&B run. Debug and non-W&B loggers are inert.
    """
    env = os.environ if environ is None else environ
    if env.get("ICE_REQUEUE_OWNER") != "runner":
        return
    logger_cfg = cfg.get("logger")
    if not OmegaConf.is_dict(logger_cfg):
        return
    wandb_configs = [
        value for value in logger_cfg.values()
        if OmegaConf.is_dict(value) and str(value.get("_target_", "")) in {
            "lightning.pytorch.loggers.wandb.WandbLogger",
            "pytorch_lightning.loggers.wandb.WandbLogger",
        }
    ]
    if not wandb_configs:
        return
    restart_count = int(env.get("SLURM_RESTART_COUNT", "0"))
    if restart_count < 0:
        raise ValueError("SLURM_RESTART_COUNT must be nonnegative")
    checkpoint = env.get("ICE_RESUME_CHECKPOINT") or cfg.get("ckpt_path")
    if restart_count == 0 and checkpoint:
        return  # Intentional initial checkpoint/new-run semantics stay configured.
    metadata = {}
    if restart_count > 0:
        if not env.get("ICE_RESUME_CHECKPOINT"):
            raise ValueError("automatic W&B resume requires the runner-selected checkpoint")
        metadata = json.loads(env.get("ICE_RESUME_CHECKPOINT_METADATA_JSON", "{}"))
        if not isinstance(metadata, dict):
            raise ValueError("runner checkpoint metadata must be a JSON object")
    for logger in wandb_configs:
        raw_id = OmegaConf.to_container(logger, resolve=False).get("id")
        run_id = logger.get("id")
        if not isinstance(run_id, str) or not run_id.strip() or "${now:" in str(raw_id):
            raise ValueError("runner-owned W&B logging requires an explicit stable logger id")
        if restart_count > 0:
            recorded_ids = [metadata[key] for key in ("wandb_run_id", "run_id") if metadata.get(key)]
            if not recorded_ids or any(value != run_id for value in recorded_ids):
                raise ValueError("W&B logger id does not match the runner checkpoint run identity")
            for field in ("entity", "project"):
                recorded = metadata.get(f"wandb_{field}")
                if recorded is not None and logger.get(field) != recorded:
                    raise ValueError(f"W&B logger {field} does not match the runner checkpoint")
        with open_dict(logger):
            logger.resume = "must" if restart_count > 0 else "never"


@rank_zero_only
def log_hyperparameters(object_dict: Dict[str, Any]) -> None:
    """Controls which config parts are saved by Lightning loggers.

    Additionally saves:
        - Number of model parameters

    :param object_dict: A dictionary containing the following objects:
        - `"cfg"`: A DictConfig object containing the main config.
        - `"model"`: The Lightning model.
        - `"trainer"`: The Lightning trainer.
    """
    hparams = {}

    cfg = OmegaConf.to_container(object_dict["cfg"])
    model = object_dict["model"]
    trainer = object_dict["trainer"]

    if not trainer.logger:
        log.warning("Logger not found! Skipping hyperparameter logging...")
        return

    hparams["model"] = cfg["model"]

    # save number of model parameters
    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params/non_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    hparams["data"] = cfg["data"]
    hparams["trainer"] = cfg["trainer"]

    hparams["callbacks"] = cfg.get("callbacks")
    hparams["extras"] = cfg.get("extras")

    hparams["task_name"] = cfg.get("task_name")
    hparams["tags"] = cfg.get("tags")
    hparams["ckpt_path"] = cfg.get("ckpt_path")
    hparams["seed"] = cfg.get("seed")
    # Preserve the source/dataset/launch identity supplied by the launcher.
    # Without this, W&B health checks cannot compare the run to its manifest.
    if cfg.get("run_provenance") is not None:
        hparams["run_provenance"] = OmegaConf.to_container(
            object_dict["cfg"].run_provenance, resolve=True
        )

    # send hparams to all loggers
    for logger in trainer.loggers:
        logger.log_hyperparams(hparams)
