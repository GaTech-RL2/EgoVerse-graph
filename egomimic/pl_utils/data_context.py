"""Declared boundary between data adapters and shared model orchestration."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from egomimic.eval.eval import EvaluationDataRequirements


@runtime_checkable
class ContextDataModule(Protocol):
    def prepare_context(
        self, *, mode, normalization, normalizer, restored_state=None
    ): ...

    def configure_evaluation(self, requirements: EvaluationDataRequirements): ...

    def frame_counts(self) -> list[tuple[str, str, int]]: ...


@dataclass
class DataContext:
    """A model-facing context; its adapter owns serialization and restoration."""

    normalizer: Any
    sample_schema: dict
    validation_groups: tuple[str, ...]
    state: dict

    def bind(self, model, evaluator=None):
        model.bind_data_context(normalizer=self.normalizer)
        if evaluator is not None:
            evaluator.bind_data_context(normalizer=self.normalizer)

    def snapshot(self):
        return deepcopy(self.state)
