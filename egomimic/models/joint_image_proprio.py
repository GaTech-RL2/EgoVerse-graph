"""Joint state denoising built around the unchanged native image model.

Both variants run one active-stream layer loop using the historical image
embedding, compiled tensor regions and decoder. An optional narrow proprio
expert participates in the same shared attention operation. Optional command
tokens use another narrow expert in that same loop, without phase-B regression.
"""

import math

import torch
from torch import nn

from egomimic.models.attention_execution import (
    apply_temporal_rope, prepare_attention_layout, load_flash_attention_kernels,
    execute_packed_flash_attention,
)
from egomimic.models.compilation import compile_region
from egomimic.models.image_proprio import ImageProprioModel
from egomimic.models.pixel_diffusion import (
    LinearDDIMImageSampler,
    LinearGaussianImageNoising,
)
from egomimic.models.world_model import MoTBlock

ACTIVE_STREAMS = (("image",), ("image", "proprio"),
                  ("image", "action"), ("image", "proprio", "action"))
SUPPORTED_STREAMS = ACTIVE_STREAMS + (("image", "proprio", "camera"),
                                     ("image", "proprio", "action", "camera"))


def paired_state_groups(world, name="proprio"):
    """Bind one pose to each image timestamp, with identical observation roles."""
    image, pose = world["image"], world[name]
    image_groups = torch.zeros_like(image["valid"], dtype=torch.long)
    pose_groups = torch.zeros_like(pose["valid"], dtype=torch.long)
    representatives, offset = [], 0
    if image["valid"].shape[0] != pose["valid"].shape[0]:
        raise ValueError("Image/proprio batch sizes differ")
    for b in range(image["valid"].shape[0]):
        iv, pv = image["valid"][b], pose["valid"][b]
        positions = iv.nonzero(as_tuple=True)[0]
        times, imap = torch.unique(image["time"][b, iv], return_inverse=True)
        ptimes, pmap = torch.unique(pose["time"][b, pv], return_inverse=True)
        if not torch.equal(times, ptimes) or pmap.numel() != times.numel():
            raise ValueError("Image/proprio require one pose per valid timestamp")
        observed = torch.empty_like(times, dtype=torch.bool)
        observed[pmap] = pose["observed"][b, pv]
        if not torch.equal(image["observed"][b, iv], observed[imap]):
            raise ValueError("Image/proprio observation masks must be tied")
        first = torch.full_like(times, positions.numel(), dtype=torch.long)
        first.scatter_reduce_(
            0, imap, torch.arange(positions.numel(), device=positions.device),
            reduce="amin",
        )
        if "index" in image and "index" in pose:
            indices = image["index"][b, positions[first]]
            if not torch.equal(image["index"][b, iv], indices[imap]) or not torch.equal(
                pose["index"][b, pv], indices[pmap]
            ):
                raise ValueError("Image/proprio discrete indices must be tied")
        representatives.append(b * iv.numel() + positions[first])
        image_groups[b, iv], pose_groups[b, pv] = imap + offset, pmap + offset
        offset += times.numel()
    if offset == 0:
        raise ValueError("Paired state requires valid timestamps")
    return image_groups, pose_groups, torch.cat(representatives)


class CoupledImageProprioNoising:
    """Native image corruption followed by tied-strength proprio corruption.

Image-only corruption delegates exactly to the historical noiser and consumes
no vector draws. Vector corruption forks the RNG so it cannot advance the
subsequent image sequence. Commands draw independent strengths and noise after
the optional proprio draw; inactive streams consume no random numbers.
    """

    def __init__(self, maximum=1.0, active_streams=("image", "proprio")):
        self.streams = tuple(active_streams)
        if self.streams not in SUPPORTED_STREAMS:
            raise ValueError("Expected image with optional proprio/action streams")
        self.image_noising = LinearGaussianImageNoising(maximum)

    def __call__(self, world):
        image = self.image_noising(world["image"])
        result = {
            "values": {"image": image["image_values"]},
            "levels": {"image": image["noise_levels"]},
            "targets": {"image": image["target"]},
            "masks": {"image": image["mask"]},
            "metadata": {"image": image["metadata"]},
        }
        clean = world["image"]["value"]
        devices = [clean.device.index] if clean.is_cuda else []
        with torch.random.fork_rng(devices=devices):
            for name in self.streams[1:]:
                if name not in world:
                    continue
                if name == "proprio":
                    self._add_proprio(world, image, result, name)
                else:
                    self._add_action(world, result, name)
        return result

    def _add_proprio(self, world, image, result, name="proprio"):
        image_groups, pose_groups, representatives = paired_state_groups(world, name)
        grouped = image["noise_levels"].reshape(-1).index_select(0, representatives)
        if ((image["noise_levels"] != grouped[image_groups]) & world["image"]["valid"]).any():
            raise ValueError("Image noise strengths must agree per timestamp")
        pose = world[name]
        known = pose["valid"] & pose.get("target_valid", pose["valid"])
        clean = torch.where((known | pose["observed"])[..., None], pose["value"], 0)
        if not torch.isfinite(clean).all():
            raise ValueError("Nonfinite measured proprioception target")
        mask = known & ~pose["observed"]
        levels = grouped[pose_groups].masked_fill(~pose["valid"], 0)
        # Missing tracked state is an unsupervised pure-noise query. Its cache
        # placeholder is never a clean observation or a denoising target.
        known = pose.get("target_valid", pose["valid"])
        levels = levels.masked_fill(pose["valid"] & ~known & ~pose["observed"], 1)
        sigma = levels[..., None].to(clean.dtype)
        pose_noise = torch.randn_like(clean)
        noisy = ((1 - sigma) * clean + sigma * pose_noise).masked_fill(
            ~pose["valid"][..., None], 0
        )
        result["values"][name] = noisy
        result["levels"][name] = levels
        result["targets"][name] = clean
        result["masks"][name] = mask
        result["metadata"][name] = {
            "timestamps": pose["time"], "valid": pose["valid"], "observed": pose["observed"]
        }

    def _add_action(self, world, result, name="action"):
        stream = world[name]
        valid, observed = stream["valid"], stream["observed"]
        known = valid & stream.get("target_valid", valid)
        clean = torch.where((known | observed)[..., None], stream["value"], 0)
        if not torch.isfinite(clean).all():
            raise ValueError("Nonfinite recorded action command")
        levels = self.image_noising.frame_noise_levels(stream, self.image_noising.maximum)
        # Unsupervised command queries stay pure noise and never leak a clean label.
        levels = levels.masked_fill(valid & ~known & ~observed, 1)
        sigma = levels[..., None].to(clean.dtype)
        noisy = (1 - sigma) * clean + sigma * torch.randn_like(clean)
        result["values"][name] = noisy.masked_fill(~valid[..., None], 0)
        result["levels"][name] = levels
        result["targets"][name] = clean
        result["masks"][name] = known & ~observed
        result["metadata"][name] = {
            "timestamps": stream["time"], "valid": valid, "observed": observed,
        }


class _ProprioBlock(nn.Module):
    """One private residual bank projected into the native image attention space."""

    def __init__(self, image_block, width, mlp_ratio, compilation):
        super().__init__()
        self.heads, self.head_dim = image_block.heads, image_block.head_dim
        attention_width = self.heads * self.head_dim
        self.norm1 = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.qkv = nn.Linear(width, 3 * attention_width)
        self.out = nn.Linear(attention_width, width)
        self.ffn = nn.Sequential(
            nn.Linear(width, width * mlp_ratio), nn.GELU(),
            nn.Linear(width * mlp_ratio, width),
        )
        self.adaln = nn.Sequential(nn.SiLU(), nn.Linear(width, 6 * width))
        nn.init.zeros_(self.adaln[-1].weight)
        nn.init.zeros_(self.adaln[-1].bias)
        for name in ("modulation", "project", "finish"):
            object.__setattr__(self, name + "_execute", compile_region(
                getattr(self, name), **compilation
            ))

    def modulation(self, condition):
        return self.adaln(condition).chunk(6, -1)

    def project(self, x, rope, mods, groups):
        shift, scale = (MoTBlock._frame_values(v, groups) for v in mods[:2])
        normalized = self.norm1(x) * (1 + scale) + shift
        q, k, v = self.qkv(normalized).reshape(
            *x.shape[:2], 3, self.heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        return apply_temporal_rope(q, rope), apply_temporal_rope(k, rope), v

    def finish(self, x, attended, valid, mods, groups):
        gate, shift, scale, mlp_gate = (
            MoTBlock._frame_values(v, groups) for v in mods[2:]
        )
        x = x + gate * self.out(attended.transpose(1, 2).flatten(2))
        x = x + mlp_gate * self.ffn(self.norm2(x) * (1 + scale) + shift)
        return x.masked_fill(~valid[..., None], 0)


class _ProprioExpert(nn.Module):
    def __init__(self, image_model, pose_dims, width, mlp_ratio):
        super().__init__()
        self.inputs = nn.ModuleDict({k: nn.Linear(d, width) for k, d in pose_dims.items()})
        self.outputs = nn.ModuleDict({k: nn.Linear(width, d) for k, d in pose_dims.items()})
        self.noise = nn.Sequential(nn.Linear(256, width), nn.SiLU(), nn.Linear(width, width))
        self.norm = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.final_adaln = nn.Sequential(nn.SiLU(), nn.Linear(width, 2 * width))
        compilation = dict(mode=image_model.compile_mode, dynamic=image_model.compile_dynamic,
                           backend=image_model.compile_backend,
                           recompile_limit=image_model.compile_recompile_limit,
                           accumulated_recompile_limit=image_model.compile_accumulated_recompile_limit)
        self.blocks = nn.ModuleList(
            _ProprioBlock(block, width, mlp_ratio, compilation) for block in image_model.blocks
        )
        for layer in self.outputs.values():
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        nn.init.zeros_(self.final_adaln[-1].weight)
        nn.init.zeros_(self.final_adaln[-1].bias)
        for layer in (self.noise[0], self.noise[2]):
            nn.init.normal_(layer.weight, std=0.02)
            nn.init.zeros_(layer.bias)
        for name in ("embed_noise", "decode"):
            object.__setattr__(self, name + "_execute", compile_region(
                getattr(self, name), **compilation
            ))

    def embed_noise(self, levels):
        frequency = torch.exp(-math.log(10000) * torch.arange(
            128, device=levels.device, dtype=torch.float32
        ) / 128)
        angles = (1000 * levels.float())[:, None] * frequency[None]
        return self.noise(torch.cat((angles.cos(), angles.sin()), -1))

    def decode(self, x, valid, condition, groups, embodiment):
        shift, scale = self.final_adaln(condition).chunk(2, -1)
        normalized = self.norm(x) * (1 + MoTBlock._frame_values(scale, groups))
        normalized = normalized + MoTBlock._frame_values(shift, groups)
        return self.outputs[embodiment](normalized).masked_fill(~valid[..., None], 0)


class JointImageProprioModel(nn.Module):
    """Native image core and optional state/control experts with AdaLN language.

    Cached per-example language features add no tokens or attention edges. Each
    expert fuses the shared projected feature with its own frame noise vector.
    """

    def __init__(self, image_model, pose_dims, proprio_width=256, mlp_ratio=4,
                 active_streams=("image", "proprio"), action_dims=None, action_width=256,
                 language_feature_dim=0, language_width=128, camera_dims=None, camera_width=256, camera_max_count=3, separate_proprio_experts=False):
        super().__init__()
        self.streams = tuple(active_streams)
        if self.streams not in SUPPORTED_STREAMS:
            raise ValueError("Expected image with optional proprio/action streams")
        if not isinstance(image_model, ImageProprioModel) or image_model.has_proprio_branch:
            raise ValueError("Provide the native image-only ImageProprioModel")
        if (image_model.objective != "visual_only" or image_model.noise_conditioning != "adaln_zero"
                or image_model.use_phase_embedding or image_model.use_modality_embedding
                or image_model.use_embodiment_embedding or image_model.activation_checkpointing):
            raise ValueError("Joint state requires native AdaLN image-only execution without extra conditioning")
        if any(type(x) is not int or x < 1 for x in (proprio_width, action_width, mlp_ratio)):
            raise ValueError("Proprio width and MLP ratio must be positive integers")
        if type(language_feature_dim) is not int or language_feature_dim not in (0, 1024):
            raise ValueError("language_feature_dim must be 0 (disabled) or 1024")
        if type(language_width) is not int or language_width != 128:
            raise ValueError("language_width must be 128")
        if type(camera_max_count) is not int or camera_max_count < 1:
            raise ValueError("camera_max_count must be positive integer")
        if type(separate_proprio_experts) is not bool:
            raise ValueError("separate_proprio_experts must be boolean")
        self.separate_proprio_experts = separate_proprio_experts
        self.camera_max_count = camera_max_count
        self.language_feature_dim = language_feature_dim
        self.language_width = language_width
        self.image_model = image_model
        self.pose_dims = ImageProprioModel._dimensions(pose_dims, "pose_dims")
        self.action_dims = (ImageProprioModel._dimensions(action_dims, "action_dims")
                            if "action" in self.streams else None)
        self.camera_dims = (ImageProprioModel._dimensions(camera_dims, "camera_dims")
                            if "camera" in self.streams else None)
        self.expert_widths = (image_model.width,)
        if len(self.streams) > 1:
            # The native image model was initialized first. Vector initialization
            # cannot perturb its weights or the caller's following CPU RNG draws.
            # Construction is CPU/meta only, before Lightning moves the module.
            if any(p.device.type not in ("cpu", "meta") for p in image_model.parameters()):
                raise ValueError("Construct the joint model on CPU/meta before device placement")
            with torch.random.fork_rng(devices=[]):
                for name in self.streams[1:]:
                    dims = self._stream_dimensions(name)
                    width = {"proprio": proprio_width, "action": action_width, "camera": camera_width}[name]
                    if name == "proprio" and self.separate_proprio_experts:
                        self.proprio_banks = nn.ModuleDict({
                            emb: _ProprioExpert(image_model, {emb: dim}, width, mlp_ratio)
                            for emb, dim in dims.items()
                        })
                    else:
                        setattr(self, name, _ProprioExpert(image_model, dims, width, mlp_ratio))
                    self.expert_widths += (width,)
        if "camera" in self.streams:
            with torch.random.fork_rng(devices=[]):
                self.camera_identity = nn.Embedding(camera_max_count, camera_width)
        if self.language_feature_dim:
            # Preserve the complete native initialization and subsequent RNG
            # sequence. The disabled path constructs no additional parameters.
            if any(p.device.type not in ("cpu", "meta") for p in image_model.parameters()):
                raise ValueError("Construct language conditioning on CPU/meta before device placement")
            with torch.random.fork_rng(devices=[]):
                self.language_projection = nn.Linear(language_feature_dim, language_width)
                self.language_fusions = nn.ModuleDict({
                    name: nn.Sequential(nn.Linear(width + language_width, width),
                                        nn.SiLU(), nn.Linear(width, width))
                    for name, width in zip(self.streams, self.expert_widths)
                    for name in ([f"proprio__{emb}" for emb in self.pose_dims]
                                 if name == "proprio" and self.separate_proprio_experts else [name])
                })
                for module in (self.language_projection, self.language_fusions):
                    for layer in module.modules():
                        if isinstance(layer, nn.Linear):
                            nn.init.normal_(layer.weight, std=0.02)
                            nn.init.zeros_(layer.bias)

    def _expert(self, name, embodiment):
        if name == "proprio" and self.separate_proprio_experts:
            return self.proprio_banks[embodiment]
        return getattr(self, name)

    def _stream_dimensions(self, name):
        return {"proprio": self.pose_dims, "action": self.action_dims,
                "camera": self.camera_dims}[name]

    def active_streams(self, embodiment):
        """Private interfaces can be absent, e.g. humans have no command expert."""
        return tuple(name for name in self.streams if name == "image" or
                     embodiment in self._stream_dimensions(name))

    def image_parameter_state(self):
        return self.image_model.state_dict()

    def expert_parameter_counts(self):
        result = {"image": sum(p.numel() for p in self.image_model.parameters())}
        for name in self.streams[1:]:
            result[name] = sum(p.numel() for p in (
                self.proprio_banks.parameters() if name == "proprio" and self.separate_proprio_experts
                else getattr(self, name).parameters()))
        if "camera" in self.streams:
            result["camera"] += sum(p.numel() for p in self.camera_identity.parameters())
        if self.language_feature_dim:
            for name in self.streams:
                keys = ([f"proprio__{emb}" for emb in self.pose_dims]
                        if name == "proprio" and self.separate_proprio_experts else [name])
                result[name] += sum(p.numel() for key in keys for p in self.language_fusions[key].parameters())
            result["language_shared"] = sum(p.numel() for p in self.language_projection.parameters())
        return result

    def _validate_language_features(self, world, language_features):
        if not self.language_feature_dim:
            if language_features is not None:
                raise ValueError("language_features supplied to a language-disabled model")
            return
        image = world["image"]["value"]
        if (not isinstance(language_features, torch.Tensor)
                or language_features.shape != (image.shape[0], self.language_feature_dim)
                or not language_features.is_floating_point()):
            raise ValueError("language_features must be a floating B x 1024 tensor")
        if language_features.device != image.device:
            raise ValueError("language_features must be on the same device as the world")
        features = language_features.to(self.language_projection.weight.dtype)
        if not torch.isfinite(features).all():
            raise ValueError("language_features must be finite")
        return features

    def _language_condition(self, name, condition, projected, representatives, token_count, embodiment=None):
        if projected is None:
            return condition
        # Representatives index the original B x tokens layout, not the
        # expert-major packed sequence. Each frame receives its own sample's
        # instruction even when examples have different valid frame counts.
        sample = torch.div(representatives, token_count, rounding_mode="floor")
        features = projected.index_select(0, sample)
        key = (f"proprio__{embodiment}" if name == "proprio" and self.separate_proprio_experts else name)
        return condition + self.language_fusions[key](torch.cat((condition, features), -1))

    def validate(self, world, embodiment):
        streams = self.active_streams(embodiment)
        if not set(streams).issubset(world):
            raise ValueError("Required embodiment streams absent")
        if "action" in self.streams and embodiment not in self.action_dims and "action" in world:
            raise ValueError("Embodiment has no declared action interface")
        self.image_model.validate({"image": world["image"]}, embodiment)
        for name in streams[1:]:
            pose = world[name]
            dims = self._stream_dimensions(name)
            value, valid, observed = pose["value"], pose["valid"], pose["observed"]
            if name == "camera":
                camera = pose.get("camera")
                if (camera is None or camera.shape != valid.shape or camera.dtype != torch.long
                        or ((camera < 0) | (camera >= self.camera_max_count))[valid].any()):
                    raise ValueError("Camera pose stream requires valid explicit camera identities")
            if embodiment not in dims or value.ndim != 3 or value.shape[-1] != dims[embodiment]:
                raise ValueError(f"Invalid {name} embodiment/value shape")
            if not value.is_floating_point() or valid.dtype != torch.bool or observed.dtype != torch.bool:
                raise ValueError(f"Invalid {name} value/mask dtype")
            if any(pose[k].shape != value.shape[:2] for k in ("valid", "observed", "time")):
                raise ValueError(f"{name} metadata shape mismatch")
            if (observed & ~valid).any() or not torch.isfinite(pose["time"].masked_fill(~valid, 0)).all():
                raise ValueError(f"Invalid {name} observation/timestamp")
            if not torch.isfinite(torch.where(observed[..., None], value, 0)).all():
                raise ValueError(f"Nonfinite observed {name}")
            if self.image_model.time_mode == "discrete":
                if "index" not in pose or pose["index"].shape != valid.shape:
                    raise ValueError(f"Discrete {name} requires grid indices")
                indices = pose["index"].masked_fill(~valid, 0)
                if not torch.isfinite(indices).all() or (indices != indices.round()).any():
                    raise ValueError("Discrete indices must be finite integers")
        if sum(world[n]["value"].shape[1] for n in streams) > self.image_model.max_tokens:
            raise ValueError("Joint token budget exceeded")
        return paired_state_groups(world) if "proprio" in streams else None

    def _prepare_execution(self, world, streams):
        """One layout policy for any active expert count; no cross-example keys."""
        native = self.image_model
        valid = [world[name]["valid"] for name in streams]
        times = [native._clock(world[name], v) for name, v in zip(streams, valid)]
        layout, rope = native._prepare_execution(times, valid, world["image"]["xy"])
        packed = native.token_execution == "packed_valid" and not bool(torch.cat(valid, 1).all())
        if not packed:
            return valid, times, layout, rope, None
        if any(b.attention_backend not in ("flash3", "flash4") or
               b.attention_dtype != "bfloat16" or b.attention_query_chunk_size
               for b in native.blocks):
            raise ValueError("Ragged packed_valid requires unchunked BF16 flash3 or flash4")
        layout = prepare_attention_layout(torch.cat(valid, 1), query_valid=torch.cat(valid, 1))
        indices = [v.reshape(-1).nonzero(as_tuple=True)[0] for v in valid]
        # Experts store their own packed tokens. Attention consumes example-major
        # order, then its inverse restores expert-major order before private FFNs.
        ids = torch.arange(valid[0].shape[0], device=valid[0].device)
        sample_ids = torch.cat([ids[:, None].expand_as(v)[v] for v in valid])
        order = torch.argsort(sample_ids, stable=True)
        packed_times = [t.reshape(-1).index_select(0, i)[None] for t, i in zip(times, indices)]
        xy = world["image"]["xy"].flatten(0, 1).index_select(0, indices[0])[None]
        rope = tuple(native._prepare_rope(t, xy if j == 0 else None)
                     for j, t in enumerate(packed_times))
        return valid, packed_times, layout, rope, (indices, order, torch.argsort(order))

    @staticmethod
    def _pack(value, indices):
        return value.flatten(0, 1).index_select(0, indices).unsqueeze(0)

    def _attention(self, block, qkv, valid, layout, packing):
        if packing is not None:
            _, order, inverse = packing
            dtype = qkv[0].dtype
            kernels = load_flash_attention_kernels(block.attention_backend, qkv[0].device)
            with torch.autocast(device_type=qkv[0].device.type, enabled=False):
                qkv = [x.squeeze(0).transpose(0, 1).index_select(0, order).to(torch.bfloat16)
                       for x in qkv]
                attended = execute_packed_flash_attention(*qkv, layout, kernels).to(dtype)
            return attended.index_select(0, inverse).transpose(0, 1).unsqueeze(0)
        mask = torch.cat(valid, 1)[:, None, None, :] if layout is None else layout.key_mask
        if block.attention_query_chunk_size:
            raise ValueError("Shared MoT execution requires unchunked attention")
        return (block._apply_attention(*qkv, mask) if layout is None else
                block._apply_attention(*qkv, mask, attention_layout=layout))

    def forward(self, world, embodiment, values, levels, language_features=None):
        streams = self.active_streams(embodiment)
        groups = self.validate(world, embodiment)
        features = self._validate_language_features(world, language_features)
        projected = self.language_projection(features) if features is not None else None
        native, image = self.image_model, world["image"]
        valid, times, layout, rope, packing = self._prepare_execution(world, streams)
        frame_map = native._prepare_noise_frame_map(image)
        condition, image_groups = native._image_noise_condition(image, levels["image"], frame_map)
        condition = self._language_condition("image", condition, projected,
                                             frame_map.representatives, image["valid"].shape[1])
        conditions = {"image": (condition, image_groups)}
        if "proprio" in streams:
            paired_image, pose_groups, representatives = groups
            grouped = levels["image"].reshape(-1).index_select(0, representatives)
            for name, index in (("image", paired_image), ("proprio", pose_groups)):
                tied = world[name]["valid"]
                if name == "proprio":
                    tied = tied & (world[name].get("target_valid", tied) | world[name]["observed"])
                if ((levels[name] != grouped[index]) & tied).any():
                    raise ValueError("Image/proprio noise strengths must be tied")
            condition = self._language_condition("proprio", self._expert("proprio", embodiment).embed_noise_execute(grouped),
                                                 projected, representatives, image["valid"].shape[1], embodiment)
            conditions["proprio"] = (condition, pose_groups)
        for name in (n for n in streams if n in ("action", "camera")):
            stream = world[name]
            mapping = native._prepare_noise_frame_map(stream)
            grouped = levels[name].reshape(-1).index_select(0, mapping.representatives)
            reconstructed = grouped[mapping.token_groups]
            if ((levels[name] != reconstructed) & stream["valid"]).any():
                raise ValueError("Action strengths must agree within each command timestamp")
            condition = self._language_condition(name, self._expert(name, embodiment).embed_noise_execute(grouped),
                                                 projected, mapping.representatives, stream["valid"].shape[1])
            conditions[name] = (condition, mapping.token_groups)
        tokens, execution_valid = [], []
        for i, name in enumerate(streams):
            stream, level = world[name], levels[name]
            if values[name].shape != stream["value"].shape or level.shape != stream["valid"].shape:
                raise ValueError("Noised state shape mismatch")
            if ((~torch.isfinite(level) | (level < 0) | (level > 1)) & stream["valid"]).any():
                raise ValueError("Noise strengths must be finite in [0,1]")
            if (level.masked_select(stream["observed"]) != 0).any():
                raise ValueError("Observed state must be clean")
            value = torch.where(stream["observed"][..., None], stream["value"], values[name])
            v, metadata = valid[i], stream
            if packing is not None:
                index = packing[0][i]
                value, level = self._pack(value, index), self._pack(level, index)
                v = v.new_ones(value.shape[:2])
                metadata = {k: self._pack(stream[k], index) for k in ("xy", "observed", "camera") if k in stream}
                c, g = conditions[name]
                conditions[name] = (c, self._pack(g, index))
            token = (native._embed_image_execute(metadata, value, level, v, "A", False)
                     if name == "image" else self._expert(name, embodiment).inputs[embodiment](value.masked_fill(~v[..., None], 0)))
            if name == "camera":
                token = token + self.camera_identity(metadata["camera"].masked_fill(~v, 0))
            tokens.append(token.masked_fill(~v[..., None], 0))
            execution_valid.append(v)
        for depth, ib in enumerate(native.blocks):
            projected, modulations = [], []
            for i, name in enumerate(streams):
                c, g = conditions[name]
                if name == "image":
                    mods = ib._image_modulation_executor(c)
                    qkv = ib._project_inputs_executor([tokens[i]], [times[i]], (rope[i],), mods, g)
                else:
                    pb = self._expert(name, embodiment).blocks[depth]
                    mods = pb.modulation_execute(c)
                    qkv = pb.project_execute(tokens[i], rope[i], mods, g)
                projected.append(qkv)
                modulations.append(mods)
            qkv = [torch.cat(parts, 2) for parts in zip(*projected)]
            attended = self._attention(ib, qkv, execution_valid, layout, packing)
            pieces = attended.split([x.shape[1] for x in tokens], dim=2)
            for i, name in enumerate(streams):
                x, v, g = tokens[i], execution_valid[i], conditions[name][1]
                tokens[i] = (ib._finish_outputs_executor([x], pieces[i], [v], [x.shape[1]], modulations[i], g)[0]
                             if name == "image" else self._expert(name, embodiment).blocks[depth].finish_execute(x, pieces[i], v, modulations[i], g))
        result = {}
        for i, name in enumerate(streams):
            value = (native._decode_image_execute(tokens[i], execution_valid[i], conditions[name])
                     if name == "image" else self._expert(name, embodiment).decode_execute(tokens[i], execution_valid[i], *conditions[name], embodiment))
            if packing is not None:
                batch, count = valid[i].shape
                value = value.new_zeros(batch * count, value.shape[-1]).index_copy(
                    0, packing[0][i], value.squeeze(0)).reshape(batch, count, -1)
            result[name] = value
        return result


class JointImageProprioSampler:
    """Native linear-DDIM image integration; joint states use the same schedule."""

    def __init__(self, state_steps=20, action_steps=None):
        LinearDDIMImageSampler._check_steps(state_steps)
        self.state_steps = state_steps
        if action_steps is not None:
            if action_steps != state_steps:
                raise ValueError("Shared native sampler requires equal state/action integration steps")
            self.action_steps = action_steps

    @torch.no_grad()
    def __call__(self, model, world, embodiment, language_features=None):
        streams = model.active_streams(embodiment)
        model.validate(world, embodiment)
        model._validate_language_features(world, language_features)
        if "action" in streams and getattr(self, "action_steps", None) != self.state_steps:
            raise ValueError("Action sampling needs an explicit matching action-step contract")
        current = {}
        s = world["image"]
        noise = torch.randn(s["value"].shape, device=s["value"].device, dtype=s["value"].dtype)
        current["image"] = torch.where(s["observed"][..., None], s["value"], noise).masked_fill(~s["valid"][..., None], 0)
        devices = [s["value"].device.index] if s["value"].is_cuda else []
        with torch.random.fork_rng(devices=devices):
            for name in streams[1:]:
                s = world[name]
                noise = torch.randn_like(s["value"])
                current[name] = torch.where(s["observed"][..., None], s["value"], noise).masked_fill(~s["valid"][..., None], 0)
        for j in range(self.state_steps, 0, -1):
            level, next_level = j / self.state_steps, (j - 1) / self.state_steps
            levels = {name: torch.full_like(world[name]["time"], level, dtype=torch.float32).masked_fill(
                world[name]["observed"] | ~world[name]["valid"], 0
            ) for name in streams}
            language = {} if language_features is None else {"language_features": language_features}
            prediction = model(world, embodiment, current, levels, **language)
            for name in streams:
                s = world[name]
                x0 = prediction[name].to(current[name].dtype)
                candidate = (1 - next_level) * x0 + next_level / level * (current[name] - (1 - level) * x0)
                current[name] = torch.where(s["observed"][..., None], s["value"], candidate).masked_fill(~s["valid"][..., None], 0)
        return current
