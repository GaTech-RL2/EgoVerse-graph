"""Generic multi-source adapter for dependency-aware pipelines."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable, Mapping

import torch
import torch.nn as nn

from egomimic.pipeline.core import Pipeline, Stage, sum_losses


class PipelineAlgo:
    """Expose a :class:`Pipeline` to training and evaluation callers.

    The outer mapping comes from a multi-source loader. Its keys are opaque and
    are preserved only to align inputs with results. Every value is one flat
    pipeline batch whose keys are interpreted exclusively by configured stages.
    """

    def __init__(
        self,
        stages: Iterable[Stage],
        device=None,
        stage_ids=None,
        initialization=None,
        trainability=None,
        loss_pipeline: Pipeline | None = None,
        training_passes=None,
    ):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.nets = nn.ModuleDict(
            {"pipeline": Pipeline(list(stages), stage_ids=stage_ids)}
        )
        if loss_pipeline is not None:
            self.nets["loss_pipeline"] = loss_pipeline
        self.training_passes = dict(training_passes or {})
        for name, spec in self.training_passes.items():
            if not isinstance(name, str) or not name or "/" in name:
                raise ValueError("Training pass names must be nonempty path components")
            if not spec.get("outputs") or not spec.get("stage_ids"):
                raise ValueError(
                    "Training passes require explicit stage_ids and outputs"
                )
            for stage_id in spec["stage_ids"]:
                self.pipeline.stage_by_id(stage_id)
        from egomimic.pipeline.initialization import (
            configure_trainability,
            initialize_weights,
        )

        self.initialization_receipts = initialize_weights(self.pipeline, initialization)
        configure_trainability(self.pipeline, trainability)
        self.nets.to(self.device)

    @property
    def pipeline(self) -> Pipeline:
        return self.nets["pipeline"]

    def bind_data_context(self, *, normalizer):
        self.pipeline.bind_data_context(normalizer=normalizer)
        if "loss_pipeline" in self.nets:
            self.nets["loss_pipeline"].bind_data_context(normalizer=normalizer)
        self.nets.to(self.device)

    def _move_value(self, value):
        if torch.is_tensor(value):
            if value.dtype == torch.float64:
                return value.to(device=self.device, dtype=torch.float32)
            return value.to(self.device)
        if isinstance(value, Mapping):
            return {key: self._move_value(item) for key, item in value.items()}
        if isinstance(value, tuple):
            return tuple(self._move_value(item) for item in value)
        if isinstance(value, list):
            return [self._move_value(item) for item in value]
        return value

    @staticmethod
    def _validate_groups(batch: Mapping) -> None:
        if not isinstance(batch, Mapping):
            raise TypeError("PipelineAlgo input must be a mapping of flat batches")
        for source, value in batch.items():
            if not isinstance(value, Mapping):
                raise TypeError(
                    "PipelineAlgo source values must be flat mappings; "
                    f"source {source!r} has {type(value).__name__}"
                )
            if any(not isinstance(key, str) for key in value):
                raise TypeError(
                    f"PipelineAlgo source {source!r} contains a non-string batch key"
                )

    def process_batch_for_training(self, batch: Mapping) -> OrderedDict:
        """Move a loader-produced mapping to the configured device unchanged."""
        self._validate_groups(batch)
        return OrderedDict(
            (source, self._move_value(value)) for source, value in batch.items()
        )

    def _execute(self, batch: Mapping, *, mode: str) -> OrderedDict:
        self._validate_groups(batch)
        return OrderedDict(
            (source, self.pipeline.execute(dict(value), mode=mode))
            for source, value in batch.items()
        )

    def forward_training(self, batch: Mapping) -> OrderedDict:
        results = self._execute(batch, mode="train")
        for name, spec in self.training_passes.items():
            for source, value in batch.items():
                extra = self.pipeline.execute_subset(
                    dict(value), spec["stage_ids"], mode="train"
                )
                for output_name, key in spec["outputs"].items():
                    destination = f"pass/{name}/{output_name}"
                    if destination in results[source]:
                        raise ValueError(
                            f"Training pass output collides at {destination}"
                        )
                    results[source][destination] = extra[key]
        return results

    @torch.inference_mode()
    def forward_eval(self, batch: Mapping) -> OrderedDict:
        return self._execute(batch, mode="inference")

    def compute_losses(self, predictions: Mapping, batch: Mapping) -> OrderedDict:
        self._validate_groups(batch)
        if tuple(predictions) != tuple(batch):
            raise ValueError("Pipeline results do not align with input sources")
        if not predictions:
            raise RuntimeError("PipelineAlgo received no source batches")

        per_source = []
        diagnostics = OrderedDict()
        for index, result in enumerate(predictions.values()):
            if not isinstance(result, Mapping):
                raise TypeError("Each pipeline result must be a mapping")
            total = sum_losses(result)
            per_source.append(total)
            diagnostics[f"source_{index}_loss"] = total
            for key, value in result.items():
                if not (key.startswith("loss/") or key.startswith("log/")):
                    continue
                if not torch.is_tensor(value):
                    value = torch.tensor(float(value), device=total.device)
                if value.ndim == 0:
                    diagnostics[f"source_{index}_{key.replace('/', '_')}"] = value

        source_losses = torch.stack(per_source)
        total = source_losses.mean()
        if "loss_pipeline" in self.nets:
            # Source names remain opaque. Only the configured loss graph knows
            # which feature keys to compare across those sources.
            names = [str(source) for source in predictions]
            if len(set(names)) != len(names) or any("/" in name for name in names):
                raise ValueError(
                    "Loss graph source names must be unique path components"
                )
            combined = {
                f"source/{source}/{key}": value
                for source, result in predictions.items()
                for key, value in result.items()
            }
            combined.update(
                (f"input/{source}/{key}", value)
                for source, values in batch.items()
                for key, value in values.items()
            )
            combined["source_losses"] = source_losses
            combined["source_count"] = len(per_source)
            result = self.nets["loss_pipeline"].execute(combined, mode="train")
            total = sum_losses(result)
            diagnostics.update(
                (key.replace("/", "_"), value)
                for key, value in result.items()
                if key.startswith("log/") and torch.is_tensor(value) and value.ndim == 0
            )
        losses = OrderedDict(loss=total)
        losses.update(diagnostics)
        return losses

    def log_info(self, info: dict) -> OrderedDict:
        losses = info["losses"]
        logged = OrderedDict(Loss=losses["loss"].item())
        logged.update(
            (key, value.item())
            for key, value in losses.items()
            if torch.is_tensor(value) and value.ndim == 0
        )
        return logged
