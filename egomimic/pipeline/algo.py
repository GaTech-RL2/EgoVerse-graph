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

    def __init__(self, stages: Iterable[Stage], device=None, homogeneous_training=False):
        self.homogeneous_training = bool(homogeneous_training)
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.nets = nn.ModuleDict({"pipeline": Pipeline(list(stages))})
        self.nets.to(self.device)

    @property
    def pipeline(self) -> Pipeline:
        return self.nets["pipeline"]

    def bind_data_context(self, *, normalizer):
        self.pipeline.bind_data_context(normalizer=normalizer)
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
        moved = OrderedDict(
            (source, self._move_value(value)) for source, value in batch.items()
        )
        if self.homogeneous_training and self.pipeline.training and len(moved) > 1:
            fused = self._fuse_equal_batches(list(moved.values()))
            self.last_training_batch_layout = {"sources": list(moved), "fused": fused is not None}
            if fused is not None:
                return OrderedDict(homogeneous=fused)
        return moved

    @staticmethod
    def _fuse_equal_batches(batches):
        """Equal source sizes preserve mean-of-source loss under concatenation.

        This is opt-in for source-agnostic stages. Incompatible layouts retain
        the ordinary source loop; inference always keeps its source mapping.
        """
        keys = list(batches[0])
        if any(set(b) != set(keys) for b in batches):
            return None
        sizes = {v.shape[0] for b in batches for v in b.values()
                 if torch.is_tensor(v) and v.ndim}
        if len(sizes) != 1:
            return None
        size = next(iter(sizes), 0)
        result = {}
        for key in keys:
            values = [b[key] for b in batches]
            first = values[0]
            if torch.is_tensor(first):
                if first.ndim == 0 or any(not torch.is_tensor(v) or v.shape != first.shape
                    or v.dtype != first.dtype or v.device != first.device for v in values):
                    return None
                result[key] = torch.cat(values, dim=0)
            elif isinstance(first, (list, tuple)) and all(isinstance(v, type(first)) and len(v) == size for v in values):
                result[key] = [item for value in values for item in value]
            else:
                return None
        return result

    def _execute(self, batch: Mapping, *, mode: str) -> OrderedDict:
        self._validate_groups(batch)
        return OrderedDict(
            (source, self.pipeline.execute(dict(value), mode=mode))
            for source, value in batch.items()
        )

    def forward_training(self, batch: Mapping) -> OrderedDict:
        return self._execute(batch, mode="train")

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

        losses = OrderedDict(loss=torch.stack(per_source).mean())
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
