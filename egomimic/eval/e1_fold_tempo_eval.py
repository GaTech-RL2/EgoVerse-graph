"""E1 tempo evaluation for graph policies, preserving the campaign metric schema."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from egomimic.eval.bimanual_cartesian_eval import BimanualCartesianEval
from egomimic.eval.e1_metrics import (  # re-export the campaign's public helpers
    ARM_BLOCKS,
    PAIRED_COLS,
    XYZ_COLS,
    E1TempoAccumulator,
    _mse_cols,
    arm_travel,
    cumulative_arc_length,
    gt_spans,
    match_spans,
    tokenize_span,
)

_STATE_KEYS = (
    "_sums",
    "_arm_sums",
    "_prog_frames",
    "_paired",
    "_n_paired",
    "_am",
    "_am_n",
    "_am_arms",
    "_am_degenerate",
    "_amg",
    "_amg_n",
    "_n",
    "_n_partial",
    "_per_chunk",
)


def _merge_state(left, right):
    if isinstance(left, dict):
        return {key: _merge_state(left[key], right[key]) for key in left}
    return left + right


class E1FoldTempoEval(BimanualCartesianEval):
    """Score native predictions and raw time targets, separately per validation group.

    Graph stages predict normalized ``pred_action``. Both that tensor and the
    carried time target are unnormalized exactly once through the bound data
    context. Missing targets are errors so a misconfigured campaign cannot
    produce an apparently successful empty score file.
    """

    def __init__(
        self,
        *,
        variant: str,
        time_key="actions_time",
        D=0.40,
        M=100,
        dt=1 / 30,
        h_match_frames=40,
        results_path=None,
        prog_horizon_m=0.19,
        progress_smooth_hz=None,
        velocity_norm="path",
        arc_match_points=100,
        span_floor_m=0.01,
        arcmatch_gt_span_m=None,
        arc_match_distance=None,
        **kwargs,
    ):
        if arc_match_distance is not None:
            raise ValueError("E1 matching uses D and arcmatch_gt_span_m")
        super().__init__(**kwargs)
        self.time_key = str(time_key)
        self.results_path = Path(results_path) if results_path else None
        self.metric_options = dict(
            variant=variant,
            D=D,
            M=M,
            dt=dt,
            h_match_frames=h_match_frames,
            prog_horizon_m=prog_horizon_m,
            progress_smooth_hz=progress_smooth_hz,
            velocity_norm=velocity_norm,
            arc_match_points=arc_match_points,
            span_floor_m=span_floor_m,
            arcmatch_gt_span_m=arcmatch_gt_span_m,
        )
        E1TempoAccumulator(**self.metric_options)  # validate before a run starts
        if D <= 0 or M < 2 or dt <= 0 or h_match_frames < 1 or arc_match_points < 2:
            raise ValueError(
                "E1 distances/time must be positive and waypoint counts >= 2"
            )
        self.on_validation_start()

    def on_validation_start(self):
        self._accumulators = {}
        self.last_results = None

    @torch.inference_mode()
    def on_validation_step(self, batch, batch_idx, dataloader_idx=0):
        del batch_idx, dataloader_idx
        predictions = self._forward_deterministic(batch)
        metrics = {}
        for source, values in batch.items():
            embodiment, label = self._embodiment(values)
            prediction = predictions[source]["pred_action"]
            native = self._native(prediction, embodiment)
            target = self.normalizer.unnormalize(
                {self.time_key: values[self.time_key]}, embodiment
            )[self.time_key]
            pred = native.detach().double().cpu().numpy()
            gt = target.detach().double().cpu().numpy()
            if pred.ndim != 3 or gt.ndim != 3 or gt.shape[-1] != 14:
                raise ValueError(
                    "E1 expects batched predictions and (B,T,14) time targets"
                )
            if len(pred) != len(gt) or gt.shape[1] < 2 or pred.shape[1] < 2:
                raise ValueError(
                    "E1 requires aligned nonempty batches with at least two timesteps"
                )
            if not np.isfinite(pred).all() or not np.isfinite(gt).all():
                raise ValueError("E1 predictions and ground truth must be finite")
            key = (self._validation_group or "valid", label)
            accumulator = self._accumulators.setdefault(
                key, E1TempoAccumulator(**self.metric_options)
            )
            metrics.update(
                {
                    key: value.to(prediction.device)
                    for key, value in accumulator.update(pred, gt, label).items()
                }
            )
        return self._namespaced(metrics)

    def _combined_accumulators(self):
        local = {
            key: {field: copy.deepcopy(getattr(acc, field)) for field in _STATE_KEYS}
            for key, acc in self._accumulators.items()
        }
        states = [local]
        if dist.is_available() and dist.is_initialized():
            states = [None] * dist.get_world_size()
            dist.all_gather_object(states, local)
        merged = {}
        for state in states:
            for key, fields in state.items():
                acc = merged.setdefault(key, E1TempoAccumulator(**self.metric_options))
                for field, value in fields.items():
                    setattr(acc, field, _merge_state(getattr(acc, field), value))
        return merged

    def on_validation_end(self):
        accumulators = self._combined_accumulators()
        if not accumulators:
            raise RuntimeError("E1 evaluation received no prediction/target batches")
        if len(accumulators) == 1:
            result = next(iter(accumulators.values())).summary()
        else:
            result = {"schema_version": 2, "groups": {}}
            for (group, label), accumulator in accumulators.items():
                result["groups"].setdefault(group, {})[label] = accumulator.summary()
        self.last_results = result
        if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
            return result
        if self.trainer is not None and not getattr(
            self.trainer, "is_global_zero", True
        ):
            return result
        print("E1_TEMPO_RESULT " + json.dumps(result), flush=True)
        if self.results_path is not None:
            self.results_path.parent.mkdir(parents=True, exist_ok=True)
            self.results_path.write_text(json.dumps(result, indent=2) + "\n")
        return result
