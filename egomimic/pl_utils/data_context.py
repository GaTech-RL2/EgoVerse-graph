"""Declared boundary between data adapters and shared model orchestration."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from egomimic.eval.eval import EvaluationDataRequirements


def serializable_state(value):
    """Canonical JSON values for an adapter-owned snapshot, without interpreting it."""
    if isinstance(value, Mapping):
        result = {str(key): serializable_state(item) for key, item in value.items()}
        if len(result) != len(value):
            raise ValueError("Data snapshot keys collide after serialization")
        return result
    if isinstance(value, (list, tuple)):
        return [serializable_state(item) for item in value]
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        return serializable_state(value.tolist())
    return value


def state_fingerprint(state):
    return hashlib.sha256(
        json.dumps(
            serializable_state(state),
            sort_keys=True,
            allow_nan=False,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


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

    def fingerprint(self):
        """Bind the entire immutable adapter state, including preprocessing semantics."""
        return state_fingerprint(self.state)
