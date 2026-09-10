"""Composable training behavior for the single Pipeline Lightning wrapper."""

from __future__ import annotations

import weakref
from typing import TYPE_CHECKING, Any

import torch.nn as nn

if TYPE_CHECKING:
    from omegaconf import DictConfig

    from egomimic.pl_utils.pl_model import ModelWrapper


class TrainingBehavior:
    """Optional policy for training mechanics around a generic pipeline.

    Pipeline stages own model semantics and tensor transformations. A behavior
    only adapts their standardized outputs to framework concerns such as
    logging, checkpoint metadata, diagnostics, or optimizer parameter binding.
    It is deliberately not an ``nn.Module`` or ``LightningModule``.
    """

    def __init__(self) -> None:
        self._context_ref: weakref.ReferenceType[ModelWrapper] | None = None

    def __setattr__(self, name: str, value: Any) -> None:
        if isinstance(value, (nn.Module, nn.Parameter)):
            raise TypeError(
                "TrainingBehavior cannot own modules or parameters; register "
                f"{name!r} on the pipeline instead"
            )
        super().__setattr__(name, value)

    def bind(self, context: ModelWrapper) -> None:
        """Bind an explicit, non-owning framework context.

        Behaviors are policy objects rather than module containers.  A weak
        context reference prevents the wrapper/behavior ownership cycle while
        keeping every framework dependency visible as ``self.context``.
        """

        current = None if self._context_ref is None else self._context_ref()
        if current is not None and current is not context:
            raise RuntimeError("TrainingBehavior instances cannot be shared")
        self._context_ref = weakref.ref(context)
        self.on_bind()

    @property
    def context(self) -> ModelWrapper:
        context = None if self._context_ref is None else self._context_ref()
        if context is None:
            raise RuntimeError("TrainingBehavior is not bound to a ModelWrapper")
        return context

    def __getstate__(self) -> dict[str, Any]:
        """Exclude the runtime-only weak reference from serialized state."""

        state = dict(self.__dict__)
        state["_context_ref"] = None
        return state

    def on_bind(self) -> None:
        return None

    def training_step(self, batch, batch_idx):
        return self.context._default_training_step(batch, batch_idx)

    def on_after_backward(self) -> None:
        return self.context._default_on_after_backward()

    def on_before_optimizer_step(self, optimizer) -> None:
        return self.context._default_on_before_optimizer_step(optimizer)

    def on_validation_start(self) -> None:
        return self.context._default_on_validation_start()

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        return self.context._default_validation_step(batch, batch_idx, dataloader_idx)

    def on_validation_epoch_end(self) -> None:
        return None

    def on_validation_end(self) -> None:
        return self.context._default_on_validation_end()

    def configure_optimizers(self):
        return self.context._default_configure_optimizers()

    def optimizer_instantiation_kwargs(self, cfg: DictConfig) -> dict[str, Any]:
        return self.context._default_optimizer_instantiation_kwargs(cfg)

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        return None

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        return None

    def forward_eval(self, batch):
        return self.context._default_forward_eval(batch)
