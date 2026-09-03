"""Small data-boundary stages for configured pipelines."""

from __future__ import annotations

from egomimic.pipeline.core import Stage


class ActionTargetBuilder(Stage):
    """Move normalized loader actions into the pipeline target namespace."""

    train_only = True
    reads = ("actions",)
    writes = ("target",)

    def forward(self, batch: dict) -> dict:
        if "actions" not in batch:
            raise ValueError("ActionTargetBuilder requires normalized actions")
        batch["target"] = batch.pop("actions")
        return batch
