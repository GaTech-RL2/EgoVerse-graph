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

MODALITIES = ("image", "track", "proprio")


def masked_mse(pred, target, weight):
    active = weight > 0
    if not torch.isfinite(target[active]).all():
        raise ValueError("Non-finite supervised target")
    safe_target = torch.where(active[..., None], target, torch.zeros_like(target))
    error = (pred.float() - safe_target.float()).square().mean(-1)
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

    def __init__(self, width, heads, mlp_ratio=4):
        super().__init__()
        if width % heads or (width // heads) % 2:
            raise ValueError("RoPE requires an even attention head width")
        self.heads = heads
        self.norm1 = nn.ModuleList(nn.LayerNorm(width) for _ in MODALITIES)
        self.qkv = nn.ModuleList(nn.Linear(width, 3 * width) for _ in MODALITIES)
        self.out = nn.ModuleList(nn.Linear(width, width) for _ in MODALITIES)
        self.norm2 = nn.ModuleList(nn.LayerNorm(width) for _ in MODALITIES)
        self.ffn = nn.ModuleList(
            nn.Sequential(
                nn.Linear(width, width * mlp_ratio),
                nn.GELU(),
                nn.Linear(width * mlp_ratio, width),
            )
            for _ in MODALITIES
        )

    def forward(self, streams, times, valid):
        sizes = [s.shape[1] for s in streams]
        qs, ks, vs = [], [], []
        for i, x in enumerate(streams):
            b, n, d = x.shape
            q, k, v = (
                self.qkv[i](self.norm1[i](x))
                .reshape(b, n, 3, self.heads, d // self.heads)
                .permute(2, 0, 3, 1, 4)
            )
            qs.append(temporal_rope(q, times[i]))
            ks.append(temporal_rope(k, times[i]))
            vs.append(v)
        mask = torch.cat(valid, 1)[:, None, None, :]
        result = F.scaled_dot_product_attention(
            torch.cat(qs, 2),
            torch.cat(ks, 2),
            torch.cat(vs, 2),
            attn_mask=mask,
            dropout_p=0.0,
        )
        result = result.transpose(1, 2).flatten(2).split(sizes, 1)
        outputs = []
        for i, (x, y) in enumerate(zip(streams, result)):
            x = x + self.out[i](y)
            x = x + self.ffn[i](self.norm2[i](x))
            outputs.append(x.masked_fill(~valid[i][..., None], 0))
        return outputs


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
    ):
        super().__init__()
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
            MoTBlock(width, heads, mlp_ratio) for _ in range(depth)
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

    def training_losses(
        self, batch, embodiment, pose_weight=1.0, image_weight=1.0, track_weight=1.0
    ):
        self.validate(batch, embodiment)
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
            supervised = s.get("target_valid", s["valid"]) & s["valid"]
            if not torch.isfinite(s["value"][supervised | s["observed"]]).all():
                raise ValueError("Non-finite visual target")
            value = torch.where((supervised | s["observed"])[..., None], s["value"], 0)
            levels[name] = self.frame_noise_levels(s)
            alpha, sigma = noise_coefficients(levels[name])
            eps = torch.randn_like(value)
            noisy[name] = alpha * value + sigma * eps
            targets[name] = alpha * eps - sigma * value
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
            x0 = alpha * noisy[name] - sigma * prediction[name]
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
                x0 = alpha * current[name] - sigma * prediction[name]
                eps = sigma * current[name] + alpha * prediction[name]
                next_level = starts[name] * ((j - 1) / steps)
                an, sn = noise_coefficients(next_level)
                current[name] = torch.where(
                    sampling_batch[name]["observed"][..., None],
                    anchors[name],
                    an * x0 + sn * eps,
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
