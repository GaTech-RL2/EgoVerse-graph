import hashlib
import os
import re
import signal
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import hydra
import lightning as L
import torch
from fsspec.implementations.local import LocalFileSystem
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.fabric.plugins.io.torch_io import TorchCheckpointIO
from lightning.fabric.utilities.cloud_io import _load as pl_load
from lightning.fabric.utilities.cloud_io import get_filesystem
from lightning.pytorch.loggers import Logger
from lightning.pytorch.plugins.environments import SLURMEnvironment
from omegaconf import DictConfig, OmegaConf, open_dict
from tabulate import tabulate

from egomimic.eval.checkpoint_loading import strict_load_pipeline_checkpoint
from egomimic.eval.eval import (
    Eval,
    validate_validation_loop,
    validation_trainer_overrides,
)
from egomimic.pipeline.algo import PipelineAlgo
from egomimic.pipeline.inference_config import export_configured_inference_artifact
from egomimic.pl_utils.data_context import ContextDataModule
from egomimic.pl_utils.pl_model import ModelWrapper
from egomimic.rldb.zarr.utils import set_global_seed
from egomimic.utils.ema_callback import EMACallback
from egomimic.utils.env import load_env
from egomimic.utils.instantiators import instantiate_callbacks, instantiate_loggers
from egomimic.utils.logging_utils import configure_runner_wandb, log_hyperparameters
from egomimic.utils.pylogger import RankedLogger
from egomimic.utils.slurm_requeue import SaveOnlySignalCheckpoint
from egomimic.utils.utils import extras, task_wrapper

log = RankedLogger(__name__, rank_zero_only=True)

_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _slurm_auto_requeue(cfg: DictConfig) -> bool:
    """Resolve and cross-check the configured Slurm requeue owner."""

    configured_owner = str(
        OmegaConf.select(
            cfg,
            "runtime.slurm_requeue_owner",
            default="lightning",
        )
    )
    runner_owner = os.environ.get("ICE_REQUEUE_OWNER")
    child_requeue_disabled = os.environ.get("ICE_CHILD_REQUEUE_DISABLED")
    if (
        configured_owner == "none"
        and runner_owner is None
        and child_requeue_disabled is None
    ):
        return False
    if configured_owner == "lightning" and (
        (runner_owner is None and child_requeue_disabled is None)
        or (runner_owner == "child" and child_requeue_disabled == "0")
    ):
        return True
    if (
        configured_owner == "runner"
        and runner_owner == "runner"
        and child_requeue_disabled == "1"
    ):
        return False
    raise RuntimeError(
        "inconsistent Slurm requeue ownership: "
        f"runtime.slurm_requeue_owner={configured_owner!r}, "
        f"ICE_REQUEUE_OWNER={runner_owner!r}, "
        f"ICE_CHILD_REQUEUE_DISABLED={child_requeue_disabled!r}; "
        "use none with no runner variables, lightning with no runner variables "
        "(or child/0), or runner with runner/1"
    )


def _slurm_environment(cfg: DictConfig) -> SLURMEnvironment:
    """Build the one Slurm environment plugin with an explicit requeue owner."""

    return SLURMEnvironment(
        auto_requeue=_slurm_auto_requeue(cfg),
        requeue_signal=signal.SIGUSR1,
    )


def _instantiate_slurm_callbacks(cfg: DictConfig) -> List[Callback]:
    """Add the save-only signal callback when an external runner owns requeue."""

    owner = str(
        OmegaConf.select(cfg, "runtime.slurm_requeue_owner", default="lightning")
    )
    if (
        not os.environ.get("SLURM_JOB_ID")
        or _slurm_auto_requeue(cfg)
        or owner == "none"
    ):
        return []
    save_signal = str(
        OmegaConf.select(cfg, "runtime.slurm_save_signal", default="SIGUSR2")
    )
    if save_signal != "SIGUSR2":
        raise RuntimeError(
            f"runtime.slurm_save_signal must be 'SIGUSR2'; got {save_signal!r}"
        )
    checkpoint_dir = OmegaConf.select(
        cfg,
        "runtime.slurm_signal_checkpoint_dir",
        default=None,
    )
    if not checkpoint_dir:
        raise RuntimeError(
            "runtime.slurm_signal_checkpoint_dir is required for runner-owned requeue"
        )
    return [
        SaveOnlySignalCheckpoint(
            checkpoint_dir=str(checkpoint_dir),
            save_signal=signal.SIGUSR2,
        )
    ]


def _instantiate_trainer_plugins(cfg: DictConfig) -> List[Any]:
    """Instantiate each trainer plugin exactly once."""

    plugins: List[Any] = []
    if cfg.get("mmap_checkpoint", True):
        plugins.append(MmapCheckpointIO())
    if os.environ.get("SLURM_JOB_ID"):
        slurm_environment = _slurm_environment(cfg)
        plugins.append(slurm_environment)
        print(
            "SLURM ENVIRONMENT ENABLED "
            f"(Lightning auto_requeue={slurm_environment.auto_requeue})"
        )
    return plugins


def _build_model_config_tree(cfg: DictConfig) -> DictConfig:
    model_cfg = OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True))
    config_tree = {"model": model_cfg}
    if cfg.get("run_provenance") is not None:
        run_provenance = OmegaConf.to_container(
            cfg.run_provenance,
            resolve=True,
        )
        wandb_run_id = OmegaConf.select(cfg, "logger.wandb.id", default=None)
        if wandb_run_id is not None:
            run_provenance["run_id"] = str(wandb_run_id)
        config_tree["run_provenance"] = run_provenance
    return OmegaConf.create(config_tree)


def _validate_run_config(cfg: DictConfig) -> str:
    if cfg.get("model") is None:
        raise ValueError("Select a complete Pipeline model config")
    if OmegaConf.select(cfg, "model.pipeline", default=None) is None:
        raise ValueError("Model config must define model.pipeline")
    if cfg.get("data") is None:
        raise ValueError("Select a data config")
    mode = cfg.get("mode")
    if mode not in {"train", "eval"}:
        raise ValueError("Config mode must be 'train' or 'eval'")
    if mode == "eval" and cfg.get("evaluator") is None:
        raise ValueError("Evaluation mode requires an evaluator config")
    return mode


def _load_eval_checkpoint(model, checkpoint: dict, cfg: DictConfig):
    """Strictly restore a configured Pipeline for standalone evaluation."""
    algo = getattr(model, "model", None)
    if not isinstance(algo, PipelineAlgo):
        raise TypeError("Evaluation model must wrap PipelineAlgo")
    settings = OmegaConf.select(cfg, "eval_checkpoint", default=None)
    use_ema = settings.get("use_ema", False) if settings is not None else False
    if not isinstance(use_ema, bool):
        raise TypeError("eval_checkpoint.use_ema must be a boolean")
    strict_load_pipeline_checkpoint(
        algo,
        checkpoint,
        use_ema=use_ema,
    )
    return model


def _callbacks_for_mode(callbacks: List[Callback], mode: str) -> List[Callback]:
    """Keep training-time EMA swapping out of standalone evaluation."""
    if mode != "eval":
        return callbacks
    return [callback for callback in callbacks if not isinstance(callback, EMACallback)]


def _resolve_model_wrapper_class(cfg: DictConfig) -> type[ModelWrapper]:
    """Require the single generic Lightning wrapper for every pipeline."""
    target = OmegaConf.select(cfg, "model._target_", default=None)
    if target is None:
        return ModelWrapper
    wrapper_class = hydra.utils.get_class(str(target))
    if wrapper_class is not ModelWrapper:
        raise TypeError(
            "cfg.model._target_ must resolve exactly to ModelWrapper; "
            "configure model-specific framework behavior under "
            "model.training_behavior instead of subclassing Lightning; "
            f"got {target!r}"
        )
    return wrapper_class


def _instantiate_model_wrapper(cfg: DictConfig) -> LightningModule:
    """Construct the configured wrapper with every wrapper-level control."""
    wrapper_class = _resolve_model_wrapper_class(cfg)
    return wrapper_class(
        config_tree=_build_model_config_tree(cfg),
        scheduler_interval=cfg.model.get("scheduler_interval", "step"),
        scheduler_frequency=cfg.model.get("scheduler_frequency", 1),
        train_log_on_step=cfg.model.get("train_log_on_step", False),
        enable_grad_norm=bool(cfg.model.get("enable_grad_norm", True)),
    )


def _log_dataset_frame_counts(train_datasets: dict, valid_entries) -> None:
    """Frame counts per split. ``valid_entries`` is the datamodule's
    ``iter_valid_datasets()`` -- (group, source, dataset) over EVERY val group,
    so a run with more than one group reports all of them rather than the one
    the flat alias happens to point at."""
    rows = []
    for name, ds in train_datasets.items():
        rows.append(("train", name, len(ds)))
    if train_datasets:
        rows.append(
            ("TOTAL", "(train)", sum(len(ds) for ds in train_datasets.values()))
        )
    valid_lengths = []
    for group, source, ds in valid_entries:
        rows.append((f"valid[{group}]", source, len(ds)))
        valid_lengths.append(len(ds))
    if valid_lengths:
        rows.append(("TOTAL", "(valid)", sum(valid_lengths)))
    table = tabulate(
        rows,
        headers=["Split", "Dataset", "Frames"],
        tablefmt="rounded_outline",
        intfmt=",",
    )
    log.info("Dataset frame counts:\n" + table)


class MmapCheckpointIO(TorchCheckpointIO):
    """``TorchCheckpointIO`` that memory-maps tensor storages instead of reading them.

    Every DDP rank loads the checkpoint independently and with no rank guard
    (``checkpoint_connector.resume_start``), so a plain read holds N private
    copies of the file in host RAM at once -- N x (weights + optimizer moments),
    and the moments are 2x the weights for AdamW. Mapping the file instead lets
    ranks on a node share page-cache pages, so one physical copy backs them all.

    ``pl_load`` does not forward ``mmap``, hence the reimplementation. Falls back
    to the normal read for checkpoints that predate torch's zipfile format (they
    cannot be mapped).
    """

    def load_checkpoint(
        self,
        path: str,
        map_location: Optional[Any] = lambda storage, loc: storage,
        weights_only: Optional[bool] = None,
    ) -> Dict[str, Any]:
        fs = get_filesystem(path)
        if not fs.exists(path):
            raise FileNotFoundError(f"Checkpoint file not found: {path}")
        if not isinstance(fs, LocalFileSystem):
            # mmap needs a real local file; let pl_load handle fsspec/URL paths.
            return pl_load(path, map_location=map_location, weights_only=weights_only)
        try:
            return torch.load(
                path, map_location=map_location, weights_only=weights_only, mmap=True
            )
        except (RuntimeError, ValueError, NotImplementedError) as e:
            log.warning(
                f"mmap load of {path} failed ({e}); falling back to a full read"
            )
            return pl_load(path, map_location=map_location, weights_only=weights_only)


def _checkpoint_file_identity(path: Path) -> tuple[int, int, int, int, int]:
    stat = path.stat()
    return (
        stat.st_dev,
        stat.st_ino,
        stat.st_size,
        stat.st_mtime_ns,
        stat.st_ctime_ns,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_runner_resume_checkpoint() -> tuple[str | None, int | None]:
    """Validate the exact checkpoint selected by the external ICE runner."""

    variable_names = (
        "ICE_RESUME_CHECKPOINT",
        "ICE_RESUME_CHECKPOINT_SHA256",
        "ICE_RESUME_GLOBAL_STEP",
    )
    missing = [name for name in variable_names if name not in os.environ]
    if missing:
        raise RuntimeError(
            "runner-owned requeue requires the runner resume environment: "
            + ", ".join(missing)
        )

    checkpoint_text = os.environ["ICE_RESUME_CHECKPOINT"]
    expected_sha256 = os.environ["ICE_RESUME_CHECKPOINT_SHA256"]
    expected_step_text = os.environ["ICE_RESUME_GLOBAL_STEP"]
    values = (checkpoint_text, expected_sha256, expected_step_text)
    if not any(values):
        restart_text = os.environ.get("SLURM_RESTART_COUNT", "0")
        if re.fullmatch(r"0|[1-9][0-9]*", restart_text) is None:
            raise RuntimeError(
                f"SLURM_RESTART_COUNT must be a nonnegative integer; got {restart_text!r}"
            )
        if int(restart_text) > 0:
            raise RuntimeError(
                "runner-owned Slurm restart has no ICE_RESUME_CHECKPOINT"
            )
        return None, None
    if not all(values):
        raise RuntimeError(
            "runner resume checkpoint, SHA-256, and global step must be supplied together"
        )
    if _SHA256_RE.fullmatch(expected_sha256) is None:
        raise RuntimeError(
            "ICE_RESUME_CHECKPOINT_SHA256 must be 64 lowercase hex digits"
        )
    if re.fullmatch(r"0|[1-9][0-9]*", expected_step_text) is None:
        raise RuntimeError("ICE_RESUME_GLOBAL_STEP must be a nonnegative integer")

    checkpoint_path = Path(checkpoint_text)
    if not checkpoint_path.is_absolute():
        raise RuntimeError("ICE_RESUME_CHECKPOINT must be an absolute path")
    try:
        checkpoint_path = checkpoint_path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"ICE_RESUME_CHECKPOINT does not exist: {checkpoint_text}"
        ) from exc
    if not checkpoint_path.is_file():
        raise RuntimeError(f"ICE_RESUME_CHECKPOINT is not a file: {checkpoint_path}")

    identity_before = _checkpoint_file_identity(checkpoint_path)
    actual_sha256 = _sha256(checkpoint_path)
    identity_after_hash = _checkpoint_file_identity(checkpoint_path)
    if identity_after_hash != identity_before:
        raise RuntimeError("ICE resume checkpoint changed while hashing")
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            "ICE_RESUME_CHECKPOINT_SHA256 mismatch: "
            f"expected={expected_sha256} actual={actual_sha256}"
        )

    checkpoint = MmapCheckpointIO().load_checkpoint(
        str(checkpoint_path),
        map_location="cpu",
        weights_only=False,
    )
    if _checkpoint_file_identity(checkpoint_path) != identity_before:
        raise RuntimeError("ICE resume checkpoint changed while validating global_step")
    if not isinstance(checkpoint, dict):
        raise RuntimeError("ICE resume checkpoint must contain a mapping")
    actual_step = checkpoint.get("global_step")
    if (
        not isinstance(actual_step, int)
        or isinstance(actual_step, bool)
        or actual_step < 0
    ):
        raise RuntimeError(
            "ICE resume checkpoint global_step must be a nonnegative integer"
        )
    expected_step = int(expected_step_text)
    if actual_step != expected_step:
        raise RuntimeError(
            "ICE_RESUME_GLOBAL_STEP mismatch: "
            f"expected={expected_step} actual={actual_step}"
        )
    return str(checkpoint_path), actual_step


def _resolve_training_checkpoint(
    cfg: DictConfig, trainer: Trainer | None = None
) -> str | None:
    """Resolve resume authority without mixing runner and Lightning semantics."""

    owner = str(
        OmegaConf.select(cfg, "runtime.slurm_requeue_owner", default="lightning")
    )
    _slurm_auto_requeue(cfg)

    if owner == "none" and os.environ.get("SLURM_RESTART_COUNT", "0") != "0":
        raise RuntimeError("non-requeue job cannot resume after a Slurm restart")

    if owner == "runner":
        selected_path, selected_step = _validate_runner_resume_checkpoint()
        configured_path = cfg.get("ckpt_path")
        if configured_path:
            configured_resolved = Path(str(configured_path)).expanduser().resolve()
            if selected_path is None or configured_resolved != Path(selected_path):
                raise RuntimeError(
                    "cfg.ckpt_path disagrees with runner-selected ICE_RESUME_CHECKPOINT"
                )
        with open_dict(cfg):
            cfg.ckpt_path = selected_path
        if selected_path is None:
            log.info("ICE runner selected a clean start with no resume checkpoint")
        else:
            log.info(
                "ICE runner resume checkpoint validated: "
                f"path={selected_path} global_step={selected_step}"
            )
        return selected_path

    configured_path = cfg.get("ckpt_path")
    if configured_path in {"last", "best", "hpc"}:
        if configured_path != "last":
            raise ValueError(
                "Preflight requires an explicit checkpoint path; only 'last' has a deterministic default location"
            )
        configured_path = os.path.join(
            trainer.default_root_dir
            if trainer is not None
            else str(cfg.trainer.default_root_dir),
            "checkpoints",
            "last.ckpt",
        )
        if not Path(configured_path).is_file():
            raise FileNotFoundError(
                f"Requested last checkpoint does not exist: {configured_path}"
            )
        with open_dict(cfg):
            cfg.ckpt_path = configured_path
    if (
        os.environ.get("SLURM_JOB_ID")
        and os.environ.get("SLURM_RESTART_COUNT", "0") != "0"
        and not configured_path
    ):
        configured_path = os.path.join(
            trainer.default_root_dir
            if trainer is not None
            else str(cfg.trainer.default_root_dir),
            "checkpoints",
            "last.ckpt",
        )
        with open_dict(cfg):
            cfg.ckpt_path = configured_path
        log.info("Detected Lightning-owned Slurm requeue — using 'last.ckpt'")
    elif (
        os.environ.get("SLURM_JOB_ID")
        and os.environ.get("SLURM_RESTART_COUNT", "0") != "0"
    ):
        log.info("Detected Lightning-owned Slurm requeue — using configured checkpoint")
    return configured_path


@task_wrapper
def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """
    mode = _validate_run_config(cfg)
    if mode == "train":
        configure_runner_wandb(cfg)

    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed") is not None:
        L.seed_everything(cfg.seed, workers=True)

        set_global_seed(cfg.seed)
    else:
        raise ValueError("Seed must be provided in cfg for reproducibility!")

    load_env()

    # Resume authority must be resolved before preparing normalization. A
    # requeued job restores the same context as its weights and optimizer.
    if mode == "train" and not cfg.get("norm_stats_only", False):
        _resolve_training_checkpoint(cfg)

    eval_obj: Eval | None = (
        hydra.utils.instantiate(cfg.evaluator)
        if cfg.get("evaluator") is not None
        else None
    )
    if eval_obj is not None:
        requirements = eval_obj.data_requirements()
        overrides = validation_trainer_overrides(eval_obj)
        with open_dict(cfg):
            for key, value in overrides.items():
                cfg.trainer[key] = value
        validate_validation_loop(requirements, cfg.trainer, mode=mode)
    checkpoint = None
    if cfg.get("ckpt_path"):
        if cfg.ckpt_path in {"last", "best", "hpc"}:
            raise ValueError(
                "Standalone evaluation requires an explicit checkpoint path"
            )
        checkpoint = MmapCheckpointIO().load_checkpoint(
            str(cfg.ckpt_path), map_location="cpu", weights_only=False
        )
    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(
        cfg.data, _recursive_=False
    )
    if not isinstance(datamodule, ContextDataModule):
        raise TypeError(
            "Configured DataModule must declare prepare_context(), "
            "configure_evaluation() and frame_counts(); see the data-context contract"
        )
    context = datamodule.prepare_context(
        mode="normalization" if cfg.get("norm_stats_only", False) else mode,
        normalization=cfg.get("norm_stats"),
        normalizer=cfg.get("normalizer"),
        restored_state=None if checkpoint is None else checkpoint.get("data_context"),
    )
    from egomimic.pipeline.inference_config import validate_model_data_context

    validate_model_data_context(cfg, context)
    if cfg.get("norm_stats_only", False):
        return {}, {
            "cfg": cfg,
            "data_context": context,
            "norm_stats": context.normalizer,
        }
    if eval_obj is not None:
        datamodule.configure_evaluation(requirements)
    if checkpoint is not None:
        from egomimic.pipeline.checkpoint_binding import validate_checkpoint_binding

        validate_checkpoint_binding(checkpoint, cfg, context)
    if mode == "train":
        exported = export_configured_inference_artifact(cfg, data_context=context)
        if exported is not None:
            artifact_path, artifact = exported
            message = (
                f"Inference config artifact ({artifact['status']}): {artifact_path}"
            )
            if artifact["status"] == "ready":
                log.info(message)
            else:
                log.warning(f"{message}; {artifact['reason']}")
    log.info(f"Instantiating model <{cfg.model._target_}>")
    from egomimic.pipeline.construction import checkpoint_construction

    with checkpoint_construction(enabled=checkpoint is not None):
        model: LightningModule = _instantiate_model_wrapper(cfg)
        context.bind(model.model, eval_obj)
    model.data_context = context
    log.info(
        "Dataset frames:\n"
        + tabulate(datamodule.frame_counts(), headers=["Split", "Source", "Frames"])
    )

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))
    if mode == "train":
        callbacks.extend(_instantiate_slurm_callbacks(cfg))

    callbacks = _callbacks_for_mode(callbacks, mode)
    # Evaluator loop requirements were applied before expensive construction.
    if mode == "eval":
        log.info("Eval mode: disabling logger")
        with open_dict(cfg):
            cfg.logger = None

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    plugins = _instantiate_trainer_plugins(cfg)
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=logger, plugins=plugins or None
    )

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    if mode == "train":
        if cfg.get("evaluator") is not None:
            eval_obj.trainer = trainer
            eval_obj.model = model.model
            model.evaluator = eval_obj
        log.info("Starting training!")
        if (
            cfg.get("val_at_start", False)
            and not cfg.get("ckpt_path")
            and os.environ.get("SLURM_RESTART_COUNT", "0") == "0"
        ):
            trainer.validate(model=model, datamodule=datamodule)
        trainer.fit(
            model=model,
            datamodule=datamodule,
            ckpt_path=cfg.get("ckpt_path"),
            weights_only=False,
        )
    elif mode == "eval":
        eval_obj.trainer = trainer
        eval_obj.model = model.model
        model.evaluator = eval_obj

        ckpt_path = cfg.get("ckpt_path")
        if ckpt_path:
            _load_eval_checkpoint(model, checkpoint, cfg)
            log.info(f"Loaded weights from {ckpt_path}")
        log.info("Starting evaluation!")
        trainer.validate(model=model, datamodule=datamodule)
    else:
        raise ValueError(f"Invalid mode: {mode}")

    return trainer.callback_metrics, object_dict


@hydra.main(
    version_base="1.3",
    config_path="./hydra_configs",
    config_name="train_zarr_cartesian.yaml",
)
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    print(OmegaConf.to_yaml(cfg))

    train(cfg)


if __name__ == "__main__":
    main()
