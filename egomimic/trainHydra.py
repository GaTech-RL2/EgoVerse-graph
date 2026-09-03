import copy
import os
import signal
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
from egomimic.eval.eval import Eval
from egomimic.pipeline.algo import PipelineAlgo
from egomimic.pl_utils.pl_model import ModelWrapper
from egomimic.rldb.zarr.utils import set_global_seed
from egomimic.rldb.zarr.zarr_dataset_multi import MultiDataset
from egomimic.utils.aws.aws_data_utils import load_env
from egomimic.utils.ema_callback import EMACallback
from egomimic.utils.instantiators import instantiate_callbacks, instantiate_loggers
from egomimic.utils.logging_utils import log_hyperparameters
from egomimic.utils.pylogger import RankedLogger
from egomimic.utils.utils import extras, task_wrapper

OmegaConf.register_new_resolver("eval", eval)

log = RankedLogger(__name__, rank_zero_only=True)


def _build_model_config_tree(cfg: DictConfig) -> DictConfig:
    model_cfg = OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True))
    return OmegaConf.create({"model": model_cfg})


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


def _log_dataset_frame_counts(train_datasets: dict, valid_datasets: dict) -> None:
    rows = []
    for name, ds in train_datasets.items():
        rows.append(("train", name, len(ds)))
    if train_datasets:
        rows.append(
            ("TOTAL", "(train)", sum(len(ds) for ds in train_datasets.values()))
        )
    for name, ds in valid_datasets.items():
        rows.append(("valid", name, len(ds)))
    if valid_datasets:
        rows.append(
            ("TOTAL", "(valid)", sum(len(ds) for ds in valid_datasets.values()))
        )
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

    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

        set_global_seed(cfg.seed)
    else:
        raise ValueError("Seed must be provided in cfg for reproducibility!")

    load_env()

    train_datasets = {}
    for dataset_name in cfg.data.train_datasets:
        train_datasets[dataset_name] = hydra.utils.instantiate(
            cfg.data.train_datasets[dataset_name]
        )

    valid_datasets = {}
    for dataset_name in cfg.data.valid_datasets:
        valid_datasets[dataset_name] = hydra.utils.instantiate(
            cfg.data.valid_datasets[dataset_name]
        )

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    assert "MultiDataModuleWrapper" in cfg.data._target_, (
        "cfg.data._target_ must be 'MultiDataModuleWrapper'"
    )
    datamodule: LightningDataModule = hydra.utils.instantiate(
        cfg.data, train_datasets=train_datasets, valid_datasets=valid_datasets
    )

    # Stats-only MultiDataset (no graph of its own; explicitly populated from
    # datamodule.train_datasets). MultiDataset now owns NormStats's role too.
    norm_stats = MultiDataset(
        state={},
        norm_mode=OmegaConf.select(cfg, "norm_stats.norm_mode", default="quantile"),
    )
    norm_stats.populate_from_datasets(datamodule.train_datasets)

    for dataset_name, dataset in datamodule.train_datasets.items():
        log.info(f"Inferring shapes for dataset <{dataset_name}>")
        norm_stats.infer_shapes_from_batch(dataset[0])
        instantiate_copy = copy.deepcopy(cfg.data.train_datasets[dataset_name])
        keymap_cfg = instantiate_copy.resolver.key_map
        km = OmegaConf.to_container(keymap_cfg, resolve=False)  # plain dict

        # this remove annotation and image keys from the keymap
        km["norm_mode"] = True

        instantiate_copy.resolver.key_map = km
        norm_dataset = hydra.utils.instantiate(instantiate_copy)
        # infer_norm_from_dataset: load from precomputed JSON/dir if set, else compute (no disk write).
        norm_stats.infer_norm_from_dataset(
            norm_dataset,
            dataset_name,
            sample_frac=OmegaConf.select(cfg, "norm_stats.sample_frac", default=1.0),
            num_workers=OmegaConf.select(cfg, "norm_stats.num_workers", default=4),
            precomputed_norm_path=OmegaConf.select(
                cfg, "norm_stats.precomputed_norm_path", default=None
            ),
        )
        # Cache norm stats if save_cache_dir is set
        save_cache_dir = OmegaConf.select(
            cfg, "norm_stats.save_cache_dir", default=None
        )
        if save_cache_dir:
            norm_stats.cache_stats(save_cache_dir=save_cache_dir)

    # Wire each training/valid MultiDataset to the stats-only ``norm_stats``
    # by reference. Bounds-check + normalize run at the MultiDataset level in
    # ``__getitem__`` — not as per-leaf transforms — which avoids the shared
    # transform_list aliasing trap.
    for ds in datamodule.train_datasets.values():
        ds.set_norm_stats_from(norm_stats)
    for ds in datamodule.valid_datasets.values():
        ds.set_norm_stats_from(norm_stats)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = ModelWrapper(
        config_tree=_build_model_config_tree(cfg),
        scheduler_interval=cfg.model.get("scheduler_interval", "step"),
    )

    _log_dataset_frame_counts(datamodule.train_datasets, datamodule.valid_datasets)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    callbacks = _callbacks_for_mode(callbacks, mode)
    # In eval mode, apply trainer overrides from the eval object and disable logger
    if mode == "eval":
        eval_obj: Eval = hydra.utils.instantiate(cfg.evaluator)
        log.info(
            "Eval mode: applying trainer overrides from eval config, disabling logger"
        )
        with open_dict(cfg):
            for k, v in eval_obj.override_dict.items():
                cfg.trainer[k] = v
            cfg.trainer.devices = 1
            cfg.trainer.num_nodes = 1
            cfg.trainer.num_sanity_val_steps = 0
            cfg.logger = None

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    plugins = []
    if cfg.get("mmap_checkpoint", True):
        plugins.append(MmapCheckpointIO())
    if os.environ.get("SLURM_JOB_ID"):
        # requeue_signal is a single signal, not a list -- SignalConnector passes
        # it straight to signal.getsignal(), which raises TypeError on a list.
        plugins.append(SLURMEnvironment(requeue_signal=signal.SIGUSR1))
        print("SLURM REQUEUE ENABLED")
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

    if (
        os.environ.get("SLURM_JOB_ID")
        and os.environ.get("SLURM_RESTART_COUNT", "0") != "0"
    ):
        last_ckpt_path = os.path.join(
            trainer.default_root_dir, "checkpoints", "last.ckpt"
        )
        log.info("Detected SLURM requeue — resuming from 'last.ckpt'")
        cfg.ckpt_path = last_ckpt_path

    if mode == "train":
        if cfg.get("evaluator") is not None:
            eval_obj: Eval = hydra.utils.instantiate(cfg.evaluator)
            eval_obj.trainer = trainer
            eval_obj.model = model.model
            if hasattr(eval_obj, "bind_data_context"):
                eval_obj.bind_data_context(normalizer=norm_stats)
            model.evaluator = eval_obj
        log.info("Starting training!")
        trainer.fit(
            model=model,
            datamodule=datamodule,
            ckpt_path=cfg.get("ckpt_path"),
            weights_only=False,
        )
    elif mode == "eval":
        eval_obj.trainer = trainer
        eval_obj.model = model.model
        if hasattr(eval_obj, "bind_data_context"):
            eval_obj.bind_data_context(normalizer=norm_stats)
        model.evaluator = eval_obj

        ckpt_path = cfg.get("ckpt_path")
        if ckpt_path:
            checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
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
