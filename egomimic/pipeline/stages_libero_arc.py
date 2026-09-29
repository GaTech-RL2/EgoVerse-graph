"""LIBERO ARC graph nodes; the shared diffusion stages predict ARC supports."""

from collections import OrderedDict
import hashlib
import json

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
        encode_inference=False,
        token_affine=None,
        **codec_kwargs,
    ):
        super().__init__()
        self.codec = (
            make_libero_arc_codec(arc_mode, **codec_kwargs) if codec is None else codec
        )
        self.arc_mode = getattr(self.codec, "mode", "joint_dur")
        self.token_affine_context = None
        if token_affine is not None:
            if self.arc_mode != "global_basis":
                raise ValueError(
                    "Coefficient affine scaling requires a global basis codec"
                )
            expected = {
                "kind": "bounded_train_std_v1",
                "basis": self.codec.basis,
                "geometry": self.codec.geometry,
                "clock": self.codec.clock,
                "geometry_fit_grid": self.codec.geometry_fit_grid,
                "fit_split": "train_only",
            }
            if any(token_affine.get(key) != value for key, value in expected.items()):
                raise ValueError("Coefficient statistics and codec identity differ")
            center = np.asarray(token_affine["center"], dtype=np.float32)
            divisor = np.asarray(token_affine["scale"], dtype=np.float32)
            shape = (self.codec.num_waypoints, 12)
            if (
                center.shape != shape
                or divisor.shape != shape
                or not np.isfinite(center).all()
                or not np.isfinite(divisor).all()
                or np.any(divisor <= 0)
            ):
                raise ValueError("Invalid coefficient affine arrays")
            self.register_buffer("token_center", torch.from_numpy(center.copy()))
            self.register_buffer("token_divisor", torch.from_numpy(divisor.copy()))
            context = {
                **expected,
                "training_windows_sha256": token_affine["training_windows_sha256"],
                "center": center.tolist(),
                "scale": divisor.tolist(),
            }
            self.token_affine_context = {
                **expected,
                "sha256": hashlib.sha256(
                    json.dumps(context, sort_keys=True).encode()
                ).hexdigest(),
                "training_windows_sha256": context["training_windows_sha256"],
            }
        else:
            # No new state_dict entries or numeric operations for old/uniform runs.
            self.token_center = self.token_divisor = None
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
        if encode_inference and (operation != "encode" or reconstruction):
            raise ValueError("encode_inference requires an encode-only stage")
        self.operation, self.encode_inference = operation, encode_inference
        self.train_only = (
            operation == "encode" and not reconstruction and not encode_inference
        )
        self.inference_only = operation == "decode" or reconstruction
        self.register_buffer("action_scale", torch.ones(7))
        self.register_buffer("action_offset", torch.zeros(7))
        if reconstruction:
            self.reads_by_mode = {"inference": ("actions",)}
        elif encode_inference:
            self.reads_by_mode = {"inference": ("actions",)}
            self.writes_by_mode = {"inference": ("target",)}

    def representation_context(self):
        """Identify the physical representation a learned tokenizer consumes."""
        fields = (
            "num_waypoints",
            "horizon",
            "dt",
            "translation_scale",
            "rotation_scale",
            "rotation_radius",
            "gripper_radius",
            "max_translation",
            "max_rotation_degrees",
        )
        context = {
            "kind": "libero_arc",
            "mode": self.arc_mode,
            **{key: getattr(self.codec, key) for key in fields},
            "velocity_norm_bound": self.velocity_norm_bound,
            "token_scale": self._token_scale(torch.ones(1)).tolist(),
        }
        if self.arc_mode == "global_basis":
            context.update(
                {
                    key: getattr(self.codec, key)
                    for key in (
                        "basis",
                        "geometry",
                        "clock",
                        "clock_fit",
                        "scalars",
                    )
                }
            )
            if self.codec.geometry_fit_grid != "uniform":
                context["geometry_fit_grid"] = self.codec.geometry_fit_grid
        if self.token_affine_context is not None:
            context["token_affine"] = self.token_affine_context
        return context

    def bind_data_context(self, *, normalizer):
        from egomimic.rldb.zarr.libero_dataset import EMBODIMENT

        stats = normalizer.norm_stats[EMBODIMENT]["actions"]
        self.action_scale.copy_(torch.as_tensor(stats["scale"]))
        self.action_offset.copy_(torch.as_tensor(stats["offset"]))
        self.normalizer_state = normalizer.to_state()
        self.data_context = normalizer.tokenizer_context()

    def _token_scale(self, tensor):
        if hasattr(self.codec, "token_scale"):
            return tensor.new_tensor(self.codec.token_scale())
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
        if mode == "train" or self.reconstruction or self.encode_inference:
            native = (batch["actions"] - self.action_offset) / self.action_scale
            values = self._encode(native.detach().float().cpu().numpy())
            tokens = torch.as_tensor(values, device=native.device, dtype=native.dtype)
            batch["target"] = tokens / self._token_scale(tokens)
            if self.token_center is not None:
                batch["target"] = (
                    batch["target"].float() - self.token_center
                ) / self.token_divisor
        if mode == "inference" and (self.operation == "decode" or self.reconstruction):
            tokens = batch["target"] if self.reconstruction else batch["pred_arc"]
            if self.token_center is not None:
                tokens = tokens.float() * self.token_divisor + self.token_center
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
                "basis",
                "geometry",
                "clock",
                "clock_fit",
                "geometry_fit_grid",
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
