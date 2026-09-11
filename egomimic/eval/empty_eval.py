"""Minimal evaluator for training runs that intentionally skip policy sampling."""

from __future__ import annotations

from typing import Any


class NoOpEvaluator:
    """Satisfy the evaluator lifecycle without running predictions or metrics."""

    def __init__(self, **_: Any) -> None:
        self.model = None

    def on_validation_start(self) -> None:
        pass

    def on_validation_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        pass

    def on_validation_end(self) -> None:
        pass
