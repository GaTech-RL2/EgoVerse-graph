"""Diagnostic capability boundary shared by orchestration and evaluators."""

from collections.abc import Mapping
from typing import Any


class DiagnosticProvider:
    """Stateless configured provider; owns no optimizer or model parameters."""

    capability: str

    def run(
        self, model: Any, batch: Mapping, *, already_processed: bool, **kwargs: Any
    ) -> Mapping:
        raise NotImplementedError
