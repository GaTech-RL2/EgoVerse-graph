"""Bridge the PI campaign's native metrics/videos to graph validation groups."""

import copy

import torch

from egomimic.campaigns.pi05.eval_metrics import PIEvalVideo
from egomimic.campaigns.pi05.eval_train_viz import TrainVizEvalVideo
from egomimic.eval.eval import Eval
from egomimic.pipeline.stages_pi05 import PI05Stage


class PI05GraphEval(Eval):
    def __init__(self, train_viz_limit=50, **kwargs):
        self.options = kwargs
        self.train_viz_limit = train_viz_limit
        self.override_dict = {"limit_val_batches": kwargs.get("limit_val_batches", 400)}
        self.trainer = None
        self.model = None
        self.normalizer = None
        self.group = "valid"
        self.cores = {}

    def bind_data_context(self, *, normalizer):
        self.normalizer = normalizer

    def set_validation_group(self, group_name):
        self.group = group_name or "valid"

    def on_validation_start(self):
        self.cores = {}

    def _stage(self):
        graph = getattr(self.model, "model", self.model)
        stages = [
            stage for stage in graph.pipeline.modules() if isinstance(stage, PI05Stage)
        ]
        if len(stages) != 1:
            raise ValueError("PI05GraphEval requires exactly one PI05Stage")
        if stages[0].normalizer is not self.normalizer:
            raise RuntimeError("PI evaluator and policy must share the same normalizer")
        return stages[0]

    @torch.no_grad()
    def on_validation_step(self, batch, batch_idx, dataloader_idx=0):
        if self.group == "train_viz" and batch_idx >= self.train_viz_limit:
            return {}
        stage = self._stage()
        if self.group not in self.cores:
            core = PIEvalVideo(**copy.deepcopy(self.options))
            if self.group == "train_viz":
                core = TrainVizEvalVideo(core, limit_val_batches=self.train_viz_limit)
            elif self.group != "valid":
                raise ValueError(f"Unsupported PI validation group: {self.group}")
            core.trainer, core.model = self.trainer, stage.backend
            core.bind_data_context(normalizer=self.normalizer)
            core.on_validation_start()
            self.cores[self.group] = core
        core = self.cores[self.group]
        prepared = {}
        with torch.inference_mode(False), torch.no_grad():
            for source, values in batch.items():
                embodiment, row = stage.prepare(values)
                if embodiment in prepared:
                    raise ValueError(f"Duplicate PI embodiment in source {source!r}")
                prepared.update(row)
            results = core.on_validation_step(prepared, batch_idx, dataloader_idx)
        self.trainer.lightning_module.log_dict(
            results, sync_dist=True, add_dataloader_idx=False
        )
        return results

    def on_validation_end(self):
        for core in self.cores.values():
            core.on_validation_end()
