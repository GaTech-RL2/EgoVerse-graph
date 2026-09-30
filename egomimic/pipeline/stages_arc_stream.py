"""Training diagnostics for the wide ARC shape/velocity token layout."""

from __future__ import annotations

import torch.nn.functional as F

from egomimic.pipeline.core import Stage


class ArcStreamVelocityLossStage(Stage):
    """Keep the standard joint FM loss and log its shape/velocity components.

    The joint loss is the arithmetic mean of the two stream MSEs. Since the ARC
    shape and per-waypoint velocity streams have equal widths, this is exactly
    the same objective as MSE over the full wide token.
    """

    train_only = True
    reads = ("flow/predicted_velocity", "flow/velocity_target")
    writes = ("loss/flow_velocity", "log/*")

    def __init__(self, stream_dim: int = 14):
        super().__init__()
        self.stream_dim = int(stream_dim)
        if self.stream_dim <= 0:
            raise ValueError("stream_dim must be positive")

    def forward(self, batch: dict) -> dict:
        prediction = batch["flow/predicted_velocity"]
        target = batch["flow/velocity_target"]
        if prediction.shape != target.shape:
            raise ValueError(
                "ARC stream flow loss shape mismatch: "
                f"prediction={tuple(prediction.shape)} target={tuple(target.shape)}"
            )
        if prediction.ndim != 3 or int(prediction.shape[-1]) != 2 * self.stream_dim:
            raise ValueError(
                "ARC stream flow tensors must have shape "
                f"(B, M, {2 * self.stream_dim}), got {tuple(prediction.shape)}"
            )

        shape_loss = F.mse_loss(
            prediction[..., : self.stream_dim], target[..., : self.stream_dim]
        )
        velocity_loss = F.mse_loss(
            prediction[..., self.stream_dim :], target[..., self.stream_dim :]
        )
        loss = 0.5 * (shape_loss + velocity_loss)
        batch["loss/flow_velocity"] = loss
        batch["log/flow_velocity"] = loss.detach()
        batch["log/flow_shape_velocity_mse"] = shape_loss.detach()
        batch["log/flow_waypoint_velocity_mse"] = velocity_loss.detach()
        return batch
