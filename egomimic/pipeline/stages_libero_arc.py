"""LIBERO ARC graph nodes; the shared diffusion stages predict ARC supports."""

from collections import OrderedDict

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
        velocity_norm_bound=1.0,
        encode_cache_size=0,
        **codec_kwargs,
    ):
        super().__init__()
        self.codec = (
            make_libero_arc_codec(arc_mode, **codec_kwargs) if codec is None else codec
        )
        self.arc_mode = getattr(self.codec, "mode", "joint_dur")
        self.encode_cache_size = int(encode_cache_size)
        if self.encode_cache_size < 0:
            raise ValueError("ARC encode cache size must be nonnegative")
        # Derived targets only: no tensors, gradients, RNG or checkpoint state.
        self._encode_cache = OrderedDict()
        self._encode_cache_signature = None
        # The config records the rate scale so checkpoints retain their units.
        # Legacy checkpoints omit this argument and retain their original scale.
        self.velocity_norm_bound = float(velocity_norm_bound)
        if not np.isfinite(self.velocity_norm_bound) or self.velocity_norm_bound <= 0:
            raise ValueError("Velocity normalization bound must be finite and positive")
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
            scale[3] = (
                self.velocity_norm_bound * self.codec.translation_scale / self.codec.dt
            )
            scale[10] = (
                self.velocity_norm_bound * self.codec.rotation_scale / self.codec.dt
            )
        return scale

    def execute(self, batch, *, mode):
        if mode == "train" or self.reconstruction:
            native = (batch["actions"] - self.action_offset) / self.action_scale
            values = self._encode(native.detach().float().cpu().numpy())
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

    def _encode(self, actions):
        if not self.encode_cache_size:
            return np.stack([self.codec.encode(row) for row in actions])
        if actions.ndim != 3 or actions.shape[1:] != (self.codec.horizon, 7):
            raise ValueError("Expected a batch of finite LIBERO action windows")
        signature = tuple(
            getattr(self.codec, key, None)
            for key in (
                "mode",
                "horizon",
                "num_waypoints",
                "dt",
                "translation_scale",
                "rotation_scale",
                "rotation_radius",
                "gripper_radius",
                "max_translation",
                "max_rotation_degrees",
            )
        )
        if signature != self._encode_cache_signature:
            self._encode_cache.clear()
            self._encode_cache_signature = signature
        values = []
        for row in actions:
            # Exact native float32 bytes preserve even sub-quantization changes.
            key = row.tobytes()
            if key in self._encode_cache:
                value = self._encode_cache[key]
                self._encode_cache.move_to_end(key)
            else:
                value = self.codec.encode(row).copy()
                self._encode_cache[key] = value
                if len(self._encode_cache) > self.encode_cache_size:
                    self._encode_cache.popitem(last=False)
            values.append(value)
        return np.stack(values)

    def forward(self, batch):
        return self.execute(batch, mode="train")


class ArcPredictionName(Stage):
    inference_only = True
    reads = ("pred_action",)
    writes = ("pred_arc",)

    def forward(self, batch):
        batch["pred_arc"] = batch.pop("pred_action")
        return batch
