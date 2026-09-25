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


def require_state(actual, required, path="data_context"):
    """Validate model-declared constraints without interpreting adapter semantics.

    Mappings are required subsets; sequences and scalar values are exact. A
    missing value is never equivalent to an adapter default or a wildcard.
    """
    if isinstance(required, Mapping):
        if not isinstance(actual, Mapping):
            raise ValueError(f"Model data requirement {path} needs a mapping")
        for key, expected in required.items():
            if key not in actual:
                raise ValueError(f"Model data requirement {path}.{key} is missing")
            require_state(actual[key], expected, f"{path}.{key}")
    elif actual != required:
        raise ValueError(
            f"Model data requirement {path} expected {required!r}, got {actual!r}"
        )


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

    def validate_requirements(self, requirements):
        """Check a configured model's required subset of this adapter's state."""
        if not isinstance(requirements, Mapping):
            raise TypeError("Model data requirements must be a mapping")
        require_state(serializable_state(self.state), serializable_state(requirements))
