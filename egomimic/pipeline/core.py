"""Small dependency-aware stage runner used by Pipeline policies."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch.nn as nn


def sum_losses(batch: dict):
    """Sum every scalar written under the ``loss/`` namespace."""
    losses = [value for key, value in batch.items() if key.startswith("loss/")]
    if not losses:
        raise RuntimeError("no loss/* keys in batch -- no loss stage ran")
    return sum(losses[1:], start=losses[0])


class Stage(nn.Module):
    """A ``dict -> dict`` module with declarative read/write contracts."""

    reads: Sequence[str] = ()
    writes: Sequence[str] = ()
    reads_by_mode: Mapping[str, Sequence[str]] = {}
    writes_by_mode: Mapping[str, Sequence[str]] = {}

    def contract(self, mode: str = "train") -> tuple[tuple[str, ...], tuple[str, ...]]:
        if mode not in {"train", "rollout"}:
            raise ValueError(f"Stage contract mode must be train|rollout, got {mode!r}")
        reads = self.reads_by_mode.get(mode, self.reads)
        writes = self.writes_by_mode.get(mode, self.writes)
        return tuple(reads or ()), tuple(writes or ())

    def forward(self, batch: dict) -> dict:  # pragma: no cover - interface
        raise NotImplementedError


def _read_is_available(read: str, available: set[str]) -> bool:
    if read.endswith("*"):
        return any(key.startswith(read[:-1]) for key in available)
    if read in available:
        return True
    return any(
        written.endswith("*") and read.startswith(written[:-1]) for written in available
    )


class Pipeline(Stage):
    """An ordered, registered list of stages."""

    def __init__(self, stages: Sequence[Stage]):
        super().__init__()
        self.stages = nn.ModuleList(stages)

    def forward(self, batch: dict) -> dict:
        for stage in self.stages:
            batch = stage(batch)
        return batch

    def plan(
        self, seed_keys: Sequence[str], mode: str = "train"
    ) -> tuple[list[Stage], list[tuple[Stage, list[str]]]]:
        """Resolve runnable stages without executing the graph."""
        if mode not in {"train", "rollout"}:
            raise ValueError(f"Pipeline.plan mode must be train|rollout, got {mode!r}")

        available = set(seed_keys)
        runnable: list[Stage] = []
        excluded: list[tuple[Stage, list[str]]] = []
        for stage in self.stages:
            if mode == "rollout" and getattr(stage, "train_only", False):
                excluded.append((stage, ["<train-only>"]))
                continue
            reads, writes = stage.contract(mode)
            missing = [key for key in reads if not _read_is_available(key, available)]
            if missing:
                excluded.append((stage, missing))
                continue
            runnable.append(stage)
            available.update(writes)
        return runnable, excluded

    def explain(self, seed_keys: Sequence[str] = (), mode: str = "train") -> str:
        """Return a compact, human-readable dependency plan."""
        if mode not in {"train", "rollout"}:
            raise ValueError(
                f"Pipeline.explain mode must be train|rollout, got {mode!r}"
            )

        available = set(seed_keys)
        lines = []
        for stage in self.stages:
            reads, writes = stage.contract(mode)
            if mode == "rollout" and getattr(stage, "train_only", False):
                reason = " (EXCLUDED: train-only)"
            else:
                missing = [
                    key for key in reads if not _read_is_available(key, available)
                ]
                reason = f" (EXCLUDED: missing {','.join(missing)})" if missing else ""
                if not missing:
                    available.update(writes)
            lines.append(
                f"{type(stage).__name__:28s} reads={list(reads)} "
                f"writes={list(writes)}{reason}"
            )
        return "\n".join(lines)
