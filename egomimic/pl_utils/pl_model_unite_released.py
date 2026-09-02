"""Lightning wrapper for the released UNITE register policy."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict

import hydra
import torch

import egomimic.utils.tensor_utils as TensorUtils
from egomimic.pl_utils.pl_model import ModelWrapper


class ReleasedUniteModelWrapper(ModelWrapper):
    """Optimize the joint released objective with topology-stable checkpoints."""

    def __init__(self, share_encoder_denoiser: bool | None = None, **kwargs):
        # These old experimental controls may still appear in a saved config.
        # The released register policy always uses one joint update.
        kwargs.pop("unite_flow_updates_per_reconstruction", None)
        kwargs.pop("unite_gradient_telemetry_every_n_steps", None)
        kwargs.pop("train_metrics_on_step", None)
        kwargs.pop("train_metrics_on_epoch", None)
        super().__init__(**kwargs)
        config_tree = getattr(self.hparams, "config_tree", None)
        cfg = self._as_config(config_tree) if config_tree is not None else None
        configured = (
            None if cfg is None else cfg.model.get("share_encoder_denoiser", None)
        )
        configured = None if configured is None else bool(configured)
        checkpointed = (
            None if share_encoder_denoiser is None else bool(share_encoder_denoiser)
        )
        if (
            configured is not None
            and checkpointed is not None
            and configured != checkpointed
        ):
            raise ValueError(
                "Checkpoint share_encoder_denoiser disagrees with config_tree"
            )
        self.share_encoder_denoiser = (
            configured if configured is not None else checkpointed
        )

    def on_save_checkpoint(self, checkpoint):
        super().on_save_checkpoint(checkpoint)
        checkpoint.setdefault("hyper_parameters", {})[
            "share_encoder_denoiser"
        ] = self.share_encoder_denoiser

    @staticmethod
    def _mean_loss_terms(losses, suffix: str) -> torch.Tensor:
        terms = [
            value
            for key, value in losses.items()
            if key.endswith(suffix) and torch.is_tensor(value) and value.ndim == 0
        ]
        if not terms:
            raise RuntimeError(f"Missing UNITE loss terms ending in {suffix!r}")
        return torch.stack(terms).mean()

    def training_step(self, batch, batch_idx):
        del batch_idx
        self.train()
        batch = self.model.process_batch_for_training(batch)
        predictions = self.model.forward_training(batch)
        source_losses = self.model.compute_losses(predictions, batch)
        losses = OrderedDict(
            (key, value.mean() if torch.is_tensor(value) else value)
            for key, value in source_losses.items()
        )
        reconstruction = self._mean_loss_terms(losses, "_loss_unite_reconstruction")
        flow = self._mean_loss_terms(losses, "_loss_unite_latent")
        total = reconstruction + flow
        losses["action_loss"] = total

        self.log(
            "Train/UNITE/ReconstructionLoss",
            reconstruction,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "Train/UNITE/FlowLoss",
            flow,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        info = {"losses": TensorUtils.detach(losses)}
        for key, value in self.model.log_info(info).items():
            self.log(
                "Train/" + key,
                value,
                on_step=True,
                on_epoch=True,
                sync_dist=True,
            )
        return total

    def on_train_batch_end(self, outputs, batch, batch_idx):
        optimizer = self.optimizers(use_pl_optimizer=False)
        if not hasattr(optimizer, "adamw") or not hasattr(optimizer, "muon"):
            raise RuntimeError("Released UNITE optimizer groups are unavailable")
        self.log(
            "Optimizer/LR/AdamW",
            optimizer.adamw.param_groups[0]["lr"],
            on_step=True,
            on_epoch=False,
        )
        self.log(
            "Optimizer/LR/Muon",
            optimizer.muon.param_groups[0]["lr"],
            on_step=True,
            on_epoch=False,
        )
        return super().on_train_batch_end(outputs, batch, batch_idx)

    def configure_optimizers(self) -> Dict[str, Any]:
        named_parameters = tuple(
            self.nets.named_parameters(prefix="nets", remove_duplicate=True)
        )
        config_tree = getattr(self.hparams, "config_tree", None)
        if config_tree is not None:
            cfg = self._as_config(config_tree)
            optimizer = hydra.utils.instantiate(
                cfg.model.optimizer,
                named_params=named_parameters,
            )
            if callable(optimizer):
                optimizer = optimizer()
            scheduler_cfg = cfg.model.get("scheduler")
            scheduler = (
                hydra.utils.instantiate(scheduler_cfg, optimizer=optimizer)
                if scheduler_cfg is not None
                else None
            )
            if callable(scheduler):
                scheduler = scheduler()
        else:
            optimizer = self.hparams.optimizer(named_params=named_parameters)
            scheduler = (
                self.hparams.scheduler(optimizer=optimizer)
                if self.hparams.scheduler is not None
                else None
            )

        if scheduler is None:
            return {"optimizer": optimizer}
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": self.hparams.scheduler_interval,
                "frequency": self.hparams.scheduler_frequency,
            },
        }
