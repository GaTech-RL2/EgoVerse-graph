"""LIBERO ARC graph nodes; the shared diffusion stages predict ARC supports."""

import numpy as np
import torch

from egomimic.pipeline.core import Stage
from egomimic.rldb.zarr.libero_arc_timed import make_libero_arc_codec


class LiberoArcStage(Stage):
    reads = ("actions",)
    writes = ("target",)
    reads_by_mode = {"inference": ("pred_arc",)}
    writes_by_mode = {"inference": ("pred_action",)}

    def __init__(
        self,
        codec=None,
        reconstruction=False,
        operation="encode",
        arc_mode="joint_dur",
        **codec_kwargs,
    ):
        super().__init__()
        self.codec = (
            make_libero_arc_codec(arc_mode, **codec_kwargs) if codec is None else codec
        )
        self.arc_mode = getattr(self.codec, "mode", "joint_dur")
        self.reconstruction = reconstruction
        if operation not in {"encode", "decode"}:
            raise ValueError("operation must be encode or decode")
        self.train_only = operation == "encode" and not reconstruction
        self.inference_only = operation == "decode" or reconstruction
        self.register_buffer("action_scale", torch.ones(7))
        self.register_buffer("action_offset", torch.zeros(7))
        if reconstruction:
            self.reads_by_mode = {"inference": ("actions",)}

    def bind_data_context(self, *, normalizer):
        from egomimic.rldb.zarr.libero_dataset import EMBODIMENT

        stats = normalizer.norm_stats[EMBODIMENT]["actions"]
        self.action_scale.copy_(torch.as_tensor(stats["scale"]))
        self.action_offset.copy_(torch.as_tensor(stats["offset"]))
        self.normalizer_state = normalizer.to_state()
        self.data_context = normalizer.tokenizer_context()

    def _token_scale(self, tensor):
        # Fixed physical units, recorded in config; no separately fitted split.
        scale = tensor.new_ones(11 if self.arc_mode == "joint_dur" else 12)
        scale[:3] = self.codec.translation_scale * self.codec.horizon
        if self.arc_mode == "joint_dur":
            scale[10] = self.codec.dt * self.codec.horizon
        elif self.arc_mode == "dur":
            scale[[3, 10]] = self.codec.dt * self.codec.horizon
        else:
            scale[3] = self.codec.translation_scale / self.codec.dt
            scale[10] = self.codec.rotation_scale / self.codec.dt
        return scale

    def execute(self, batch, *, mode):
        if mode == "train" or self.reconstruction:
            native = (batch["actions"] - self.action_offset) / self.action_scale
            values = np.stack(
                [
                    self.codec.encode(row)
                    for row in native.detach().float().cpu().numpy()
                ]
            )
            tokens = torch.as_tensor(values, device=native.device, dtype=native.dtype)
            batch["target"] = tokens / self._token_scale(tokens)
        if mode == "inference":
            tokens = batch["target"] if self.reconstruction else batch["pred_arc"]
            tokens = tokens * self._token_scale(tokens)
            decoded = np.stack(
                [
                    self.codec.decode(row)
                    for row in tokens.detach().float().cpu().numpy()
                ]
            )
            actions = torch.as_tensor(decoded, device=tokens.device, dtype=tokens.dtype)
            batch["pred_action"] = actions * self.action_scale + self.action_offset
        return batch

    def forward(self, batch):
        return self.execute(batch, mode="train")


class ArcPredictionName(Stage):
    inference_only = True
    reads = ("pred_action",)
    writes = ("pred_arc",)

    def forward(self, batch):
        batch["pred_arc"] = batch.pop("pred_action")
        return batch
