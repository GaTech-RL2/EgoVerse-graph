"""Three-modality continuous/discrete-time world model (independent of Action Flow).

Inputs are normalized prepared tokens. Every stream has value[B,N,C],
time[B,N], valid[B,N] (padding only), observed[B,N]. Image/track streams
also have reference xy[B,N,2], camera[B,N]; tracks have persistent point_id.
Targets/confidence are used by the objective, never the backbone.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from egomimic.models.attention_execution import (
    apply_temporal_rope,
    execute_flash_attention,
    execute_packed_flash_attention,
    load_flash_attention_kernels,
    prepare_attention_layout,
)
from egomimic.models.compilation import compile_region

MODALITIES = ("image", "track", "proprio")


def masked_mse(pred, target, weight, *, reduction="token_mean", check_target=True):
    if reduction not in ("token_mean", "mean_per_example"):
        raise ValueError("Unknown masked MSE reduction")
    active = weight > 0
    if check_target and not torch.isfinite(torch.where(active[..., None], target, 0)).all():
        raise ValueError("Non-finite supervised target")
    safe_target = torch.where(active[..., None], target, torch.zeros_like(target))
    error = (pred.float() - safe_target.float()).square().mean(-1)
    if reduction == "mean_per_example":
        # Match the mean of independent batch-one losses, including empty
        # examples' differentiable zero. Padding and clip length add no weight.
        numerator = (error * weight.float()).flatten(1).sum(1)
        denominator = weight.float().flatten(1).sum(1)
        return (numerator / torch.where(denominator > 0, denominator, 1.0)).mean()
    total_weight = weight.float().sum()
    # Confidence weights can sum to less than one. Preserve a weighted mean;
    # use one only for an empty objective so its value and gradients stay zero.
    denominator = torch.where(total_weight > 0, total_weight, 1.0)
    return (error * weight.float()).sum() / denominator


def noise_coefficients(level):
    angle = level.float() * (math.pi / 2)
    return angle.cos()[..., None], angle.sin()[..., None]


def temporal_rope(x, times):
    # x[B,H,N,D]; rotation evaluated in FP32 even under BF16 autocast.
    half = x.shape[-1] // 2
    freq = torch.exp(-math.log(10000) * torch.arange(half, device=x.device) / half)
    phase = times.float()[:, None, :, None] * freq
    left, right = x.float().chunk(2, dim=-1)
    return torch.cat(
        (
            left * phase.cos() - right * phase.sin(),
            left * phase.sin() + right * phase.cos(),
        ),
        -1,
    ).to(x.dtype)


class MoTBlock(nn.Module):
    """Private modality projections/FFNs, one joint attention operation."""

    def __init__(
        self,
        width,
        heads,
        mlp_ratio=4,
        attention_query_chunk_size=0,
        modalities=MODALITIES,
        attention_backend="sdpa",
        attention_dtype="native",
        image_adaln=False,
    ):
        super().__init__()
        if width % heads or (width // heads) % 2:
            raise ValueError("RoPE requires an even attention head width")
        self.heads = heads
        self.head_dim = width // heads
        if type(attention_query_chunk_size) is not int or attention_query_chunk_size < 0:
            raise ValueError("attention_query_chunk_size must be a nonnegative integer")
        self.attention_query_chunk_size = attention_query_chunk_size
        if attention_backend not in ("sdpa", "efficient", "flash3", "flash4"):
            raise ValueError("attention_backend must be sdpa, efficient, flash3 or flash4")
        if attention_dtype not in ("native", "bfloat16"):
            raise ValueError("attention_dtype must be native or bfloat16")
        if attention_backend in ("flash3", "flash4") and attention_dtype != "bfloat16":
            raise ValueError(f"{attention_backend} requires explicit bfloat16 attention")
        self.attention_backend = attention_backend
        self.attention_dtype = attention_dtype
        if type(image_adaln) is not bool:
            raise ValueError("image_adaln must be bool")
        self.image_adaln = image_adaln
        self.norm1 = nn.ModuleList(
            nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
            if image_adaln and i == 0 else nn.LayerNorm(width)
            for i, _ in enumerate(modalities)
        )
        self.qkv = nn.ModuleList(nn.Linear(width, 3 * width) for _ in modalities)
        self.out = nn.ModuleList(nn.Linear(width, width) for _ in modalities)
        self.norm2 = nn.ModuleList(
            nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
            if image_adaln and i == 0 else nn.LayerNorm(width)
            for i, _ in enumerate(modalities)
        )
        self.ffn = nn.ModuleList(
            nn.Sequential(
                nn.Linear(width, width * mlp_ratio),
                nn.GELU(),
                nn.Linear(width * mlp_ratio, width),
            )
            for _ in modalities
        )
        if image_adaln:
            self.adaln_image = nn.Sequential(nn.SiLU(), nn.Linear(width, 6 * width))
            nn.init.zeros_(self.adaln_image[-1].weight)
            nn.init.zeros_(self.adaln_image[-1].bias)
        self.configure_compilation()

    def configure_compilation(self, mode="none", dynamic=False, backend="inductor",
                              recompile_limit=None, accumulated_recompile_limit=None):
        names = ("_project_inputs", "_finish_outputs")
        if self.image_adaln:
            names += ("_image_modulation",)
        for name in names:
            object.__setattr__(self, name + "_executor", compile_region(
                getattr(self, name), mode=mode, dynamic=dynamic, backend=backend,
                recompile_limit=recompile_limit, accumulated_recompile_limit=accumulated_recompile_limit
            ))

    def _image_modulation(self, condition):
        # Project once per physical-frame/role group, not once per patch.
        return self.adaln_image(condition).chunk(6, dim=-1)

    @staticmethod
    def _frame_values(values, frame_indices):
        return values.index_select(0, frame_indices.reshape(-1)).reshape(
            *frame_indices.shape, values.shape[-1]
        )

    def _project_inputs(self, streams, times, rope, image_modulation=None, frame_indices=None):
        qs, ks, vs = [], [], []
        for i, x in enumerate(streams):
            b, n, d = x.shape
            normalized = self.norm1[i](x)
            if i == 0 and self.image_adaln:
                shift, scale = (self._frame_values(value, frame_indices) for value in image_modulation[:2])
                normalized = normalized * (1 + scale) + shift
            q, k, v = (
                self.qkv[i](normalized)
                .reshape(b, n, 3, self.heads, d // self.heads)
                .permute(2, 0, 3, 1, 4)
            )
            qs.append(temporal_rope(q, times[i]) if rope is None else apply_temporal_rope(q, rope[i]))
            ks.append(temporal_rope(k, times[i]) if rope is None else apply_temporal_rope(k, rope[i]))
            vs.append(v)
        return torch.cat(qs, 2), torch.cat(ks, 2), torch.cat(vs, 2)

    def forward(self, streams, times, valid, *, attention_layout=None, rope=None, image_condition=None):
        sizes = [s.shape[1] for s in streams]
        modulation = None
        frame_indices = None
        if self.image_adaln:
            if image_condition is None:
                raise ValueError("Image AdaLN requires frame conditioning")
            condition, frame_indices = image_condition
            modulation = self._image_modulation_executor(condition)
            query, key, value = self._project_inputs_executor(streams, times, rope, modulation, frame_indices)
        else:
            query, key, value = self._project_inputs_executor(streams, times, rope)
        mask = (torch.cat(valid, 1)[:, None, None, :] if attention_layout is None
                else attention_layout.key_mask)
        chunk = self.attention_query_chunk_size
        if chunk and query.shape[2] > chunk:
            # Split only queries. Every row retains all keys/values and the same
            # softmax denominator; this is full attention, not a local window.
            # Recompute each chunk in backward rather than retaining all chunks'
            # attention intermediates and losing the intended memory bound.
            parts = []
            for q in query.split(chunk, dim=2):
                if torch.is_grad_enabled() and any(t.requires_grad for t in (q, key, value)):
                    part = checkpoint(self._apply_attention, q, key, value, mask, use_reentrant=False)
                else:
                    part = self._apply_attention(q, key, value, mask)
                parts.append(part)
            result = torch.cat(parts, dim=2)
        else:
            # Keep the historical four-argument call for native SDPA overrides.
            result = (self._apply_attention(query, key, value, mask)
                      if attention_layout is None or self.attention_backend not in ("flash3", "flash4")
                      else self._apply_attention(
                          query, key, value, mask, attention_layout=attention_layout))
        if self.image_adaln:
            return self._finish_outputs_executor(streams, result, valid, sizes, modulation, frame_indices)
        return self._finish_outputs_executor(streams, result, valid, sizes)

    def _finish_outputs(self, streams, result, valid, sizes, image_modulation=None, frame_indices=None):
        result = result.transpose(1, 2).flatten(2).split(sizes, 1)
        outputs = []
        for i, (x, y) in enumerate(zip(streams, result)):
            if i == 0 and self.image_adaln:
                gate_attention, shift, scale, gate_mlp = (
                    self._frame_values(value, frame_indices) for value in image_modulation[2:]
                )
                x = x + gate_attention * self.out[i](y)
                x = x + gate_mlp * self.ffn[i](self.norm2[i](x) * (1 + scale) + shift)
            else:
                x = x + self.out[i](y)
                x = x + self.ffn[i](self.norm2[i](x))
            outputs.append(x.masked_fill(~valid[i][..., None], 0))
        return outputs

    def forward_packed(self, stream, valid, *, attention_layout, rope, image_condition=None):
        """Run expert zero on packed image tokens, preserving example boundaries.

        Pure tensor regions reuse the same modules and compilation policy as
        padded execution. Only optional attention dispatch remains outside them.
        """
        if self.attention_backend not in ("flash3", "flash4"):
            raise ValueError("Packed token execution requires flash3 or flash4")
        if self.attention_dtype != "bfloat16":
            raise ValueError("Packed token execution requires explicit bfloat16 attention")
        if self.attention_query_chunk_size:
            raise ValueError("Packed token execution requires unchunked attention")
        if stream.ndim != 3 or stream.shape[0] != 1 or valid.shape != stream.shape[:2]:
            raise ValueError("Packed expert input must be [1,total_tokens,width]")
        modulation = None
        frame_indices = None
        if self.image_adaln:
            if image_condition is None:
                raise ValueError("Image AdaLN requires frame conditioning")
            condition, frame_indices = image_condition
            modulation = self._image_modulation_executor(condition)
            query, key, value = self._project_inputs_executor([stream], None, (rope,), modulation, frame_indices)
        else:
            query, key, value = self._project_inputs_executor([stream], None, (rope,))
        original_dtype = query.dtype
        kernels = load_flash_attention_kernels(self.attention_backend, query.device)
        with torch.autocast(device_type=query.device.type, enabled=False):
            query, key, value = (
                x.squeeze(0).transpose(0, 1).to(torch.bfloat16)
                for x in (query, key, value)
            )
            result = execute_packed_flash_attention(
                query, key, value, attention_layout, kernels
            ).to(original_dtype)
        result = result.transpose(0, 1).unsqueeze(0)
        if self.image_adaln:
            return self._finish_outputs_executor(
                [stream], result, [valid], [stream.shape[1]], modulation, frame_indices
            )[0]
        return self._finish_outputs_executor(
            [stream], result, [valid], [stream.shape[1]]
        )[0]

    def _apply_attention(self, query, key, value, mask, *, attention_layout=None):
        # Preserve historical default arithmetic and autocast behavior exactly.
        if self.attention_backend == "sdpa" and self.attention_dtype == "native":
            return self._attend(query, key, value, mask)
        original_dtype = query.dtype
        with torch.autocast(device_type=query.device.type, enabled=False):
            if self.attention_dtype == "bfloat16":
                query, key, value = (x.to(torch.bfloat16) for x in (query, key, value))
            if self.attention_backend in ("flash3", "flash4"):
                kernels = load_flash_attention_kernels(self.attention_backend, query.device)
                layout = attention_layout or prepare_attention_layout(
                    mask[:, 0, 0], query_tokens=query.shape[2]
                )
                result = execute_flash_attention(query, key, value, layout, kernels)
            elif self.attention_backend == "efficient":
                if query.device.type != "cuda":
                    raise RuntimeError("efficient attention requires CUDA; no math fallback")
                from torch.nn.attention import SDPBackend, sdpa_kernel
                with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
                    result = self._attend(query, key, value, mask)
            else:
                result = self._attend(query, key, value, mask)
        return result.to(original_dtype)

    @staticmethod
    def _flash4_attend(query, key, value, mask):
        kernels = load_flash_attention_kernels("flash4", query.device)
        layout = prepare_attention_layout(mask[:, 0, 0], query_tokens=query.shape[2])
        return execute_flash_attention(query, key, value, layout, kernels)

    @staticmethod
    def _attend(query, key, value, mask):
        return F.scaled_dot_product_attention(
            query, key, value, attn_mask=mask, dropout_p=0.0
        )


class ImageResidualAdapter(nn.Module):
    """Embodiment-specific image residual; starts as an identity operation."""

    def __init__(self, width, bottleneck):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.down = nn.Linear(width, bottleneck)
        self.up = nn.Linear(bottleneck, width)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return x + self.up(F.gelu(self.down(self.norm(x))))


class MoTWorldModel(nn.Module):
    def __init__(
        self,
        pose_dims: Mapping[str, int],
        image_dim=768,
        width=384,
        depth=8,
        heads=6,
        mlp_ratio=4,
        max_cameras=3,
        max_tracks=1120,
        max_tokens=8192,
        time_mode="continuous",
        image_adapter_dim=48,
        image_prediction_type="v",
        attention_query_chunk_size=0,
        activation_checkpointing=False,
        attention_backend="sdpa",
        attention_dtype="native",
        strict_fp32=False,
    ):
        super().__init__()
        if type(activation_checkpointing) is not bool:
            raise ValueError("activation_checkpointing must be bool")
        self.activation_checkpointing = activation_checkpointing
        if type(strict_fp32) is not bool:
            raise ValueError("strict_fp32 must be a boolean")
        if strict_fp32 and attention_dtype != "native":
            raise ValueError("Explicit BF16 attention conflicts with strict_fp32; select the mixed-attention execution profile")
        if image_prediction_type not in ("v", "x0"):
            raise ValueError("image_prediction_type must be v or x0")
        self.image_prediction_type = image_prediction_type
        if time_mode not in ("continuous", "discrete"):
            raise ValueError("Unknown time mode")
        if not pose_dims or any(int(d) < 1 for d in pose_dims.values()):
            raise ValueError(
                "Explicit positive per-embodiment pose dimensions required"
            )
        self.pose_dims = {str(k): int(v) for k, v in pose_dims.items()}
        if (
            isinstance(image_adapter_dim, bool)
            or not isinstance(image_adapter_dim, int)
            or image_adapter_dim < 0
        ):
            raise ValueError("image_adapter_dim must be a nonnegative integer")
        self.image_adapter_dim = image_adapter_dim
        self.image_dim, self.max_tokens, self.time_mode = (
            image_dim,
            max_tokens,
            time_mode,
        )
        self.image_in = nn.Linear(image_dim, width)
        self.track_in = nn.Linear(3, width)
        self.pose_in = nn.ModuleDict(
            {k: nn.Linear(d, width) for k, d in self.pose_dims.items()}
        )
        self.pose_out = nn.ModuleDict(
            {k: nn.Linear(width, d) for k, d in self.pose_dims.items()}
        )
        self.image_out = nn.Linear(width, image_dim)
        self.track_out = nn.Linear(width, 3)
        self.xy = nn.Linear(2, width)
        self.camera = nn.Embedding(max_cameras, width)
        self.point = nn.Embedding(max_tracks, width)
        self.modality = nn.Parameter(torch.randn(3, width) * 0.02)
        self.query = nn.Parameter(torch.randn(width) * 0.02)
        self.embodiment = nn.Embedding(len(pose_dims), width)
        self.embodiment_ids = {k: i for i, k in enumerate(self.pose_dims)}
        # Noise level and observed/query status are distinct from physical time.
        self.noise_status = nn.Sequential(
            nn.Linear(2, width), nn.SiLU(), nn.Linear(width, width)
        )
        self.blocks = nn.ModuleList(
            MoTBlock(width, heads, mlp_ratio, attention_query_chunk_size,
                     attention_backend=attention_backend,
                     attention_dtype=attention_dtype)
            for _ in range(depth)
        )
        # One private image adapter per embodiment per block. Zero disables the
        # extension explicitly for adapter-free ablations/historical checkpoints.
        self.image_adapters = (
            nn.ModuleList(
                nn.ModuleDict(
                    {
                        k: ImageResidualAdapter(width, image_adapter_dim)
                        for k in self.pose_dims
                    }
                )
                for _ in range(depth)
            )
            if image_adapter_dim
            else nn.ModuleList()
        )
        self.norm = nn.ModuleList(nn.LayerNorm(width) for _ in MODALITIES)

    def validate(self, batch, embodiment):
        if embodiment not in self.pose_dims:
            raise ValueError(f"Unknown embodiment {embodiment!r}")
        total = 0
        b = None
        for name, dim in zip(
            MODALITIES, (self.image_dim, 3, self.pose_dims[embodiment])
        ):
            s = batch[name]
            value = s["value"]
            if value.ndim != 3 or value.shape[-1] != dim:
                raise ValueError(f"{name}: invalid value shape")
            if b is not None and b != value.shape[0]:
                raise ValueError("Mixed batch sizes")
            b = value.shape[0]
            total += value.shape[1]
            for key in ("time", "valid", "observed"):
                if s[key].shape != value.shape[:2]:
                    raise ValueError(f"{name}: {key} shape mismatch")
            if s["valid"].dtype != torch.bool or s["observed"].dtype != torch.bool:
                raise ValueError("Masks must be boolean")
            if "target_valid" in s and (
                s["target_valid"].shape != s["valid"].shape
                or s["target_valid"].dtype != torch.bool
            ):
                raise ValueError("Target validity must be a boolean token mask")
            if name != "proprio":
                if s["xy"].shape != (*value.shape[:2], 2):
                    raise ValueError("Reference image coordinates must match tokens")
                if not torch.isfinite(s["xy"][s["valid"]]).all():
                    raise ValueError("Non-finite reference image coordinates")
                for key, limit in (("camera", self.camera.num_embeddings),):
                    if s[key].shape != s["valid"].shape or s[key].dtype != torch.long:
                        raise ValueError(f"{key} must contain integer token IDs")
                    active_ids = s[key][s["valid"]]
                    if ((active_ids < 0) | (active_ids >= limit)).any():
                        raise ValueError(f"{key} ID out of range")
            if name == "track":
                ids = s["point_id"]
                if ids.shape != s["valid"].shape or ids.dtype != torch.long:
                    raise ValueError("Persistent point IDs must match tokens")
                if (
                    (ids[s["valid"]] < 0)
                    | (ids[s["valid"]] >= self.point.num_embeddings)
                ).any():
                    raise ValueError("Persistent point ID out of range")
            if (s["observed"] & ~s["valid"]).any():
                raise ValueError("Padding cannot be observed")
            if not torch.isfinite(s["time"][s["valid"]]).all():
                raise ValueError("Non-finite timestamp")
            if not torch.isfinite(value[s["observed"]]).all():
                raise ValueError("Non-finite observed value")
            if self.time_mode == "discrete":
                if "index" not in s or s["index"].shape != s["time"].shape:
                    raise ValueError("Discrete mode requires explicit grid indices")
                if not torch.isfinite(s["index"][s["valid"]]).all() or not torch.equal(
                    s["index"][s["valid"]], s["index"][s["valid"]].round()
                ):
                    raise ValueError("Discrete indices must be finite integers")
        if total > self.max_tokens:
            raise ValueError(f"Token budget exceeded: {total} > {self.max_tokens}")
        valid = torch.cat([batch[n]["valid"] for n in MODALITIES], 1)
        if not valid.any(1).all():
            raise ValueError("Every sample needs at least one valid token")

    def forward(self, batch, embodiment, visual_values, levels):
        self.validate(batch, embodiment)
        embeds, times, valid = [], [], []
        for i, name in enumerate(MODALITIES):
            s = batch[name]
            observed = s["observed"] & s["valid"]
            if name == "proprio":
                # Never embed unknown target poses, even during teacher-forced training.
                value = torch.where(observed[..., None], s["value"], 0)
                x = self.pose_in[embodiment](value)
                x = torch.where(observed[..., None], x, self.query)
                level = torch.zeros_like(s["time"])
            else:
                value = torch.where(s["valid"][..., None], visual_values[name], 0)
                x = (self.image_in if name == "image" else self.track_in)(value)
                x = x + self.xy(torch.where(s["valid"][..., None], s["xy"], 0))
                x = x + self.camera(s["camera"].masked_fill(~s["valid"], 0))
                if name == "track":
                    x = x + self.point(s["point_id"].masked_fill(~s["valid"], 0))
                level = levels[name]
            condition = torch.stack((level, observed.to(level.dtype)), -1)
            x = x + self.noise_status(condition.to(x.dtype)) + self.modality[i]
            x = x + self.embodiment.weight[self.embodiment_ids[embodiment]]
            embeds.append(x.masked_fill(~s["valid"][..., None], 0))
            clock = s["time"] if self.time_mode == "continuous" else s["index"]
            times.append(clock.masked_fill(~s["valid"], 0))
            valid.append(s["valid"])
        for index, block in enumerate(self.blocks):
            if self.activation_checkpointing and torch.is_grad_enabled():
                embeds = checkpoint(block, embeds, times, valid, use_reentrant=False)
            else:
                embeds = block(embeds, times, valid)
            if self.image_adapter_dim:
                embeds[0] = self.image_adapters[index][embodiment](
                    embeds[0]
                ).masked_fill(~valid[0][..., None], 0)
        embeds = [norm(x) for norm, x in zip(self.norm, embeds)]
        return {
            "image": self.image_out(embeds[0]),
            "track": self.track_out(embeds[1]),
            "proprio": self.pose_out[embodiment](embeds[2]),
        }

    @staticmethod
    def frame_noise_levels(stream):
        # One noise level per timestamp per example; all patches/cameras agree.
        result = torch.zeros_like(stream["time"])
        for b in range(result.shape[0]):
            _, inverse = torch.unique(
                stream["time"][b], sorted=True, return_inverse=True
            )
            draws = torch.rand(
                int(inverse.max()) + 1, device=result.device, dtype=result.dtype
            ).clamp(0.001, 0.999)
            result[b] = draws[inverse]
        return result.masked_fill(stream["observed"] | ~stream["valid"], 0)

    def _complete_training_tracks(self, batch, embodiment, policy, steps):
        """Explicit approximate completion; never invent zero clean geometry.

        Available recorded visuals condition a detached auxiliary sample. Only
        missing tracks are retained, without becoming supervised targets. The
        main denoising pass receives neither completion nor availability flags.
        """
        if policy not in ("error", "conditional_sample"):
            raise ValueError("Unknown missing_track_policy")
        if type(steps) is not int or steps < 1:
            raise ValueError("imputation_steps must be a positive integer")
        available = {}
        for name in ("image", "track"):
            s = batch[name]
            confidence = s.get("confidence", torch.ones_like(s["time"]))
            if confidence.shape != s["time"].shape or (
                not torch.isfinite(confidence).all()
                or (confidence < 0).any()
                or (confidence > 1).any()
            ):
                raise ValueError("Confidence must match tokens and be finite in [0,1]")
            known = s["valid"] & s.get("target_valid", s["valid"]) & (confidence > 0)
            if (s["observed"] & ~known).any():
                raise ValueError(
                    "Observed visual values require valid positive-confidence targets"
                )
            if not torch.isfinite(s["value"][known]).all():
                raise ValueError("Non-finite visual target")
            available[name] = known
        if (batch["image"]["valid"] & ~available["image"]).any():
            raise ValueError(
                "Missing image targets require explicit preprocessing; unsupported"
            )
        track = batch["track"]
        missing = track["valid"] & ~available["track"]
        if not missing.any():
            # No forward or random draw: retain exact complete-data behavior.
            return track["value"], missing
        if policy == "error":
            raise ValueError(
                "Missing track targets: zero-fill is forbidden. Supply valid tracks "
                "or explicitly select missing_track_policy='conditional_sample'."
            )
        teacher = {n: dict(batch[n]) for n in MODALITIES}
        for name in ("image", "track"):
            teacher[name]["observed"] = available[name]
            teacher[name]["value"] = torch.where(
                available[name][..., None], batch[name]["value"], 0
            )
        # Even an auxiliary sampler must never receive withheld recorded poses.
        teacher["proprio"]["value"] = torch.where(
            batch["proprio"]["observed"][..., None], batch["proprio"]["value"], 0
        )
        with torch.no_grad():
            completion = self.sample(teacher, embodiment, steps=steps)["track"].detach()
        if not torch.isfinite(completion[missing]).all():
            raise ValueError("Non-finite conditional track completion")
        return torch.where(missing[..., None], completion, track["value"]), missing

    def training_losses(
        self,
        batch,
        embodiment,
        pose_weight=1.0,
        image_weight=1.0,
        track_weight=1.0,
        missing_track_policy="error",
        imputation_steps=20,
    ):
        self.validate(batch, embodiment)
        track_values, imputed = self._complete_training_tracks(
            batch, embodiment, missing_track_policy, imputation_steps
        )
        noisy, levels, targets, weights = {}, {}, {}, {}
        for name in ("image", "track"):
            s = batch[name]
            confidence = s.get("confidence", torch.ones_like(s["time"]))
            if confidence.shape != s["time"].shape:
                raise ValueError("Confidence must match token shape")
            if (
                not torch.isfinite(confidence).all()
                or (confidence < 0).any()
                or (confidence > 1).any()
            ):
                raise ValueError("Confidence must be finite and between zero and one")
            supervised = (
                s.get("target_valid", s["valid"]) & s["valid"] & (confidence > 0)
            )
            if not torch.isfinite(s["value"][supervised | s["observed"]]).all():
                raise ValueError("Non-finite visual target")
            if name == "track":
                value = torch.where(s["valid"][..., None], track_values, 0)
            else:
                value = torch.where(
                    (supervised | s["observed"])[..., None], s["value"], 0
                )
            levels[name] = self.frame_noise_levels(s)
            alpha, sigma = noise_coefficients(levels[name])
            eps = torch.randn_like(value)
            noisy[name] = alpha * value + sigma * eps
            targets[name] = (
                value
                if name == "image" and self.image_prediction_type == "x0"
                else alpha * eps - sigma * value
            )
            weights[name] = (supervised & ~s["observed"]).float() * confidence
        prediction = self(batch, embodiment, noisy, levels)
        losses = {
            "image": masked_mse(
                prediction["image"], targets["image"], weights["image"]
            ),
            "track": masked_mse(
                prediction["track"], targets["track"], weights["track"]
            ),
        }
        # Same-trajectory x0 estimates. No detach: pose loss reaches both visual heads.
        clean = {}
        for name in ("image", "track"):
            alpha, sigma = noise_coefficients(levels[name])
            x0 = (
                prediction[name]
                if name == "image" and self.image_prediction_type == "x0"
                else alpha * noisy[name] - sigma * prediction[name]
            )
            clean[name] = torch.where(
                batch[name]["observed"][..., None], batch[name]["value"], x0
            )
        pose = batch["proprio"]
        if pose_weight > 0:
            readout = self(
                batch,
                embodiment,
                clean,
                {k: torch.zeros_like(v) for k, v in levels.items()},
            )
            pose_mask = (
                pose.get("target_valid", pose["valid"])
                & pose["valid"]
                & ~pose["observed"]
            )
            losses["pose"] = masked_mse(
                readout["proprio"], pose["value"], pose_mask.float()
            )
        else:
            losses["pose"] = prediction["proprio"].sum() * 0
        losses["total"] = (
            image_weight * losses["image"]
            + track_weight * losses["track"]
            + pose_weight * losses["pose"]
        )
        losses["imputed_track_tokens"] = imputed.sum().to(losses["total"].dtype)
        losses["imputation_forwards"] = losses["total"].new_tensor(
            imputation_steps + 1 if imputed.any() else 0
        )
        return losses

    def _refinement_state(self, batch, generator):
        """Validate explicit clean predictions and build a private sampling batch.

        Optional batch["refinement"][image|track] contains strength[B,N] and
        prediction[B,N,C]. Strength is a diffusion noise level, not a linear
        blend percentage. No withheld batch value supplies a starting estimate.
        """
        controls = batch.get("refinement", {})
        if not isinstance(controls, Mapping) or set(controls) - {"image", "track"}:
            raise ValueError("Refinement controls support only image and track")
        sampling_batch = {n: dict(batch[n]) for n in MODALITIES}
        starts, anchors = {}, {}
        for name in ("image", "track"):
            s = batch[name]
            valid, observed = s["valid"], s["observed"]
            control = controls.get(name)
            strength = torch.ones_like(s["time"]).masked_fill(observed | ~valid, 0)
            anchor = torch.where(observed[..., None], s["value"], 0)
            if control is not None:
                if (
                    not isinstance(control, Mapping)
                    or "strength" not in control
                    or set(control) - {"strength", "prediction"}
                ):
                    raise ValueError(
                        "Refinement requires strength and optional prediction"
                    )
                supplied = control["strength"]
                if (
                    not isinstance(supplied, torch.Tensor)
                    or supplied.shape != valid.shape
                    or supplied.device != s["value"].device
                    or not supplied.is_floating_point()
                ):
                    raise ValueError(
                        "Refinement strength must be a floating token-shaped tensor"
                    )
                if (
                    not torch.isfinite(supplied[valid]).all()
                    or (supplied[valid] < 0).any()
                    or (supplied[valid] > 1).any()
                ):
                    raise ValueError("Refinement strength must be finite in [0,1]")
                if (supplied[observed] != 0).any():
                    raise ValueError("Observed values cannot be revised")
                # Match training's one noise level per modality/timestamp.
                for b in range(valid.shape[0]):
                    for time in torch.unique(s["time"][b][valid[b] & ~observed[b]]):
                        selected = valid[b] & ~observed[b] & (s["time"][b] == time)
                        if torch.unique(supplied[b][selected]).numel() != 1:
                            raise ValueError(
                                "Refinement strength must agree within each frame"
                            )
                strength = supplied.float().masked_fill(~valid, 0)
                needs_prediction = valid & ~observed & (strength < 1)
                previous = control.get("prediction")
                if previous is not None:
                    if (
                        not isinstance(previous, torch.Tensor)
                        or previous.shape != s["value"].shape
                        or previous.device != s["value"].device
                        or previous.dtype != s["value"].dtype
                    ):
                        raise ValueError(
                            "Refinement prediction must match value shape/device/dtype"
                        )
                    if not torch.isfinite(previous[needs_prediction]).all():
                        raise ValueError("Non-finite retained/refined prediction")
                    anchor = torch.where(needs_prediction[..., None], previous, anchor)
                elif needs_prediction.any():
                    raise ValueError(
                        "Retaining or refining requires an explicit prediction"
                    )
            fixed = valid & (strength == 0)
            sampling_batch[name]["observed"] = fixed
            # Fixed generated predictions act as clean anchors only in this call.
            sampling_batch[name]["value"] = anchor
            starts[name], anchors[name] = strength, anchor
        current = {}
        # Validate all controls before consuming the caller's random generator.
        for name in ("image", "track"):
            s = batch[name]
            noise = torch.randn(
                s["value"].shape,
                device=s["value"].device,
                dtype=s["value"].dtype,
                generator=generator,
            )
            alpha, sigma = noise_coefficients(starts[name])
            initial = alpha * anchors[name] + sigma * noise
            # Explicit endpoints preserve old fresh-noise sampling and anchors exactly.
            current[name] = torch.where(
                (starts[name] == 1)[..., None],
                noise,
                torch.where((starts[name] == 0)[..., None], anchors[name], initial),
            ).masked_fill(~s["valid"][..., None], 0)
        return sampling_batch, current, starts, anchors

    @torch.no_grad()
    def sample(self, batch, embodiment, steps=20, generator=None):
        """Deterministic DDIM; randomness only in initial Gaussian draws.

        Unknown values/target masks/confidence are never read by the sampler.
        Optional batch["refinement"] supplies explicit clean previous predictions
        and per-frame start levels. Each level descends start -> 0 together.
        Physical query times stay fixed. Omitted controls retain fresh sampling.
        """
        self.validate(batch, embodiment)
        if steps < 1:
            raise ValueError("At least one denoising step is required")
        sampling_batch, current, starts, anchors = self._refinement_state(
            batch, generator
        )
        for j in range(steps, 0, -1):
            levels = {n: start * (j / steps) for n, start in starts.items()}
            prediction = self(sampling_batch, embodiment, current, levels)
            for name in current:
                alpha, sigma = noise_coefficients(levels[name])
                next_level = starts[name] * ((j - 1) / steps)
                an, sn = noise_coefficients(next_level)
                if name == "image" and self.image_prediction_type == "x0":
                    # Predict structured clean features directly. The bounded
                    # ratio avoids forming epsilon at tiny positive strengths:
                    # division first can overflow, followed by zero * infinity.
                    x0 = prediction[name]
                    denominator = torch.where(
                        levels[name][..., None] > 0, sigma, torch.ones_like(sigma)
                    )
                    candidate = an * x0 + (sn / denominator) * (
                        current[name] - alpha * x0
                    )
                    candidate = candidate.masked_fill(
                        ~sampling_batch[name]["valid"][..., None], 0
                    )
                else:
                    x0 = alpha * current[name] - sigma * prediction[name]
                    eps = sigma * current[name] + alpha * prediction[name]
                    candidate = an * x0 + sn * eps
                current[name] = torch.where(
                    sampling_batch[name]["observed"][..., None],
                    anchors[name],
                    candidate,
                )
        zeros = {n: torch.zeros_like(batch[n]["time"]) for n in current}
        pose = self(sampling_batch, embodiment, current, zeros)["proprio"]
        current["proprio"] = torch.where(
            batch["proprio"]["observed"][..., None], batch["proprio"]["value"], pose
        )
        return {
            n: x.masked_fill(~batch[n]["valid"][..., None], 0)
            for n, x in current.items()
        }
