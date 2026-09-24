"""Configured deterministic sequence prediction and exact-shape regression."""

import torch
from torch.nn import functional as F

from egomimic.pipeline.core import Stage


class SequenceRegressionStage(Stage):
    reads = ("condition", "target")
    writes = ("pred_action", "loss/regression")
    reads_by_mode = {"inference": ("condition",)}
    writes_by_mode = {"inference": ("pred_action",)}

    def __init__(self, model, action_horizon, action_dim, loss="smooth_l1", beta=0.05):
        super().__init__()
        self.model = model
        self.shape = (int(action_horizon), int(action_dim))
        if min(self.shape) < 1 or loss not in {"smooth_l1", "mse"} or beta <= 0:
            raise ValueError(
                "Declare positive sequence dimensions and a supported loss"
            )
        self.loss, self.beta = loss, float(beta)

    def execute(self, batch, *, mode):
        prediction = self.model(batch["condition"])
        if prediction.ndim != 3 or tuple(prediction.shape[1:]) != self.shape:
            raise ValueError(
                f"Regression head must emit (B, {self.shape}), got {prediction.shape}"
            )
        batch["pred_action"] = prediction
        if mode == "train":
            target = batch["target"].to(prediction)
            if target.shape != prediction.shape:
                raise ValueError(
                    "Regression target and prediction must have identical shapes"
                )
            if not torch.isfinite(target).all():
                raise ValueError("Regression targets must be finite")
            batch["loss/regression"] = (
                F.smooth_l1_loss(prediction, target, beta=self.beta)
                if self.loss == "smooth_l1"
                else F.mse_loss(prediction, target)
            )
        return batch
