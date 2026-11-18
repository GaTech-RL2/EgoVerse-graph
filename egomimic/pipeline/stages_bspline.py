"""Decode diffusion-predicted BSP parameters into the common action interface."""

import numpy as np
import torch

from egomimic.models.bspline import BSplineCodec
from egomimic.pipeline.core import Stage


class BSplineDecodeStage(Stage):
    inference_only = True
    reads = ("pred_action",)
    writes = ("pred_action",)

    def __init__(self, codec=None, horizon=32):
        super().__init__()
        self.codec = BSplineCodec(**(codec or {}))
        if type(horizon) is not int or horizon < 1:
            raise ValueError("horizon must be a positive integer")
        self.horizon = horizon
        self.register_buffer("parameter_scale", torch.ones(8))
        self.register_buffer("parameter_offset", torch.zeros(8))
        self.register_buffer("action_scale", torch.ones(7))
        self.register_buffer("action_offset", torch.zeros(7))

    def bind_data_context(self, *, normalizer):
        from egomimic.rldb.zarr.libero_dataset import EMBODIMENT

        context = normalizer.context.get("bspline", {})
        if any(context.get(k) != v for k, v in self.codec.context().items()):
            raise ValueError("BSP dataset and decoder representation differ")
        stats = normalizer.norm_stats[EMBODIMENT]
        for prefix, key in (("parameter", "bspline"), ("action", "actions")):
            for field in ("scale", "offset"):
                value = torch.as_tensor(stats[key][field])
                target = getattr(self, f"{prefix}_{field}")
                if value.shape != target.shape or not torch.isfinite(value).all():
                    raise ValueError("Invalid BSP normalizer shape or values")
                if field == "scale" and not torch.all(value > 0):
                    raise ValueError("BSP normalizer scales must be positive")
                target.copy_(value)

    def forward(self, batch):
        prediction = batch["pred_action"]
        if prediction.ndim != 3 or prediction.shape[1:] != (self.codec.num_knots, 8):
            raise ValueError(
                "Expected a batch of 16x8 (or configured knot count) BSP parameters"
            )
        parameters = (prediction.float() - self.parameter_offset) / self.parameter_scale
        decoded = np.stack(
            [
                self.codec.decode(row, horizon=self.horizon)
                for row in parameters.detach().cpu().numpy()
            ]
        )
        native = torch.as_tensor(decoded, device=prediction.device, dtype=torch.float32)
        batch["pred_action"] = native * self.action_scale + self.action_offset
        return batch
