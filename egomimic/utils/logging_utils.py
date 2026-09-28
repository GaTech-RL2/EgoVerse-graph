import json
import os
from pathlib import Path
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
    # Offline validation uses a separate W&B run, not the training identity.
    if cfg.get("mode", "train") == "eval":
        return
    logger_cfg = cfg.get("logger")
    if not OmegaConf.is_dict(logger_cfg):
        return
    wandb_configs = [
        value
        for value in logger_cfg.values()
        if OmegaConf.is_dict(value)
        and str(value.get("_target_", ""))
        in {
            "lightning.pytorch.loggers.wandb.WandbLogger",
            "pytorch_lightning.loggers.wandb.WandbLogger",
        }
    ]
    if not wandb_configs:
        return
    if env.get("ICE_REQUEUE_OWNER") != "runner":
        _configure_lightning_wandb(cfg, wandb_configs, env)
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
            raise ValueError(
                "automatic W&B resume requires the runner-selected checkpoint"
            )
        metadata = json.loads(env.get("ICE_RESUME_CHECKPOINT_METADATA_JSON", "{}"))
        if not isinstance(metadata, dict):
            raise ValueError("runner checkpoint metadata must be a JSON object")
    for logger in wandb_configs:
        raw_id = OmegaConf.to_container(logger, resolve=False).get("id")
        run_id = logger.get("id")
        if not isinstance(run_id, str) or not run_id.strip() or "${now:" in str(raw_id):
            raise ValueError(
                "runner-owned W&B logging requires an explicit stable logger id"
            )
        if restart_count > 0:
            recorded_ids = [
                metadata[key] for key in ("wandb_run_id", "run_id") if metadata.get(key)
            ]
            if not recorded_ids or any(value != run_id for value in recorded_ids):
                raise ValueError(
                    "W&B logger id does not match the runner checkpoint run identity"
                )
            for field in ("entity", "project"):
                recorded = metadata.get(f"wandb_{field}")
                if recorded is not None and logger.get(field) != recorded:
                    raise ValueError(
                        f"W&B logger {field} does not match the runner checkpoint"
                    )
        with open_dict(logger):
            logger.resume = "must" if restart_count > 0 else "never"


def _local_wandb_run_id(output_dir: Path) -> str | None:
    """Recover the most substantial local W&B artifact's run ID."""

    artifacts = sorted(
        output_dir.glob("wandb/run-*/run-*.wandb"),
        key=lambda path: path.stat().st_size,
        reverse=True,
    )
    if not artifacts:
        return None
    stem = artifacts[0].name
    if not stem.startswith("run-") or not stem.endswith(".wandb"):
        return None
    return stem[len("run-") : -len(".wandb")] or None


def _configure_lightning_wandb(cfg, wandb_configs, env) -> None:
    """Fail closed when Lightning-owned jobs resume W&B without the old ID."""

    restart_count = int(env.get("SLURM_RESTART_COUNT", "0"))
    if restart_count < 0:
        raise ValueError("SLURM_RESTART_COUNT must be nonnegative")
    output_dir = Path(str(OmegaConf.select(cfg, "paths.output_dir")))
    identity_file = output_dir / "wandb-run-id.txt"
    # A user-provided ckpt_path initializes a new experiment's weights. It is
    # not evidence that its W&B run should be resumed. Only the runner's
    # explicit requeue checkpoint or a Slurm restart carries W&B identity.
    automatic_checkpoint = env.get("ICE_RESUME_CHECKPOINT")
    automatic_resume = bool(automatic_checkpoint) or restart_count > 0
    checkpoint = automatic_checkpoint
    inferred_checkpoint = output_dir / "checkpoints" / "last.ckpt"
    local_id = _local_wandb_run_id(output_dir)
    sidecar_id = identity_file.read_text().strip() if identity_file.is_file() else None
    if not checkpoint and automatic_resume:
        if inferred_checkpoint.is_file():
            checkpoint = str(inferred_checkpoint)
            with open_dict(cfg):
                cfg.ckpt_path = checkpoint
        else:
            raise ValueError(
                "automatic W&B resume requires a previous last.ckpt; "
                f"none exists at {inferred_checkpoint}"
            )

    # Fresh experiments may initialize model weights with cfg.ckpt_path, but
    # retain their new W&B identity and configured resume='never'.
    if not automatic_resume:
        return

    for logger in wandb_configs:
        raw_id = OmegaConf.to_container(logger, resolve=False).get("id")
        configured_id = None
        if "${now:" not in str(raw_id):
            resolved_id = logger.get("id")
            if isinstance(resolved_id, str) and resolved_id.strip():
                configured_id = resolved_id
        explicit_ids = [
            value
            for value in (env.get("ICE_WANDB_RUN_ID"), sidecar_id, configured_id)
            if value
        ]
        if len(set(explicit_ids)) > 1:
            raise ValueError("conflicting W&B run identities for checkpoint resume")
        run_id = explicit_ids[0] if explicit_ids else local_id
        if checkpoint and not run_id:
            raise ValueError(
                "W&B resume requires the original run ID; refusing to create a new run"
            )
        if checkpoint:
            with open_dict(logger):
                logger.id = run_id
                logger.resume = "must"
                if not logger.get("name"):
                    logger.name = run_id


def persist_wandb_run_identity(cfg: DictConfig, loggers, trainer) -> None:
    """Persist the instantiated W&B ID for a future Lightning requeue."""

    if (
        cfg.get("mode", "train") == "eval"
        or not loggers
        or not getattr(trainer, "is_global_zero", True)
    ):
        return
    wandb_logger = next(
        (item for item in loggers if item.__class__.__name__ == "WandbLogger"),
        None,
    )
    run_id = getattr(getattr(wandb_logger, "experiment", None), "id", None)
    if not run_id:
        return
    identity_file = (
        Path(str(OmegaConf.select(cfg, "paths.output_dir"))) / "wandb-run-id.txt"
    )
    identity_file.parent.mkdir(parents=True, exist_ok=True)
    identity_file.write_text(f"{run_id}\n")


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
