"""Joint pixel/state/control diffusion with three private MoT parameter banks.

Image and measured state share availability and corruption strength. Commands
have an independent corruption draw. Physical time never denotes noise strength.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from egomimic.models.attention_execution import (
    apply_temporal_rope,
    execute_packed_flash_attention,
    load_flash_attention_kernels,
    prepare_attention_layout,
    prepare_spatiotemporal_rope,
)
from egomimic.models.compilation import compile_region
from egomimic.models.image_proprio import ImageProprioModel
from egomimic.models.pixel_diffusion import LinearGaussianImageNoising
from egomimic.models.world_model import MoTBlock

STREAMS = ("image", "proprio", "action")


def state_groups(world):
    """Validate exact timestamp alignment and coupled state observation masks."""
    image, pose = world["image"], world["proprio"]
    image_groups = torch.zeros_like(image["valid"], dtype=torch.long)
    pose_groups = torch.zeros_like(pose["valid"], dtype=torch.long)
    observed_groups, representatives = [], []
    offset = 0
    for b in range(image["time"].shape[0]):
        iv, pv = image["valid"][b], pose["valid"][b]
        positions = iv.nonzero(as_tuple=True)[0]
        times, imap = torch.unique(image["time"][b, iv], return_inverse=True)
        ptimes, pmap = torch.unique(pose["time"][b, pv], return_inverse=True)
        if not torch.equal(times, ptimes) or pmap.numel() != times.numel():
            raise ValueError("Image/proprio require the same valid timestamps")
        observed = torch.empty_like(times, dtype=torch.bool)
        observed[pmap] = pose["observed"][b, pv]
        if not torch.equal(image["observed"][b, iv], observed[imap]):
            raise ValueError("Image/proprio observation masks must be tied")
        first = torch.full(
            (times.numel(),),
            positions.numel(),
            device=positions.device,
            dtype=torch.long,
        )
        first.scatter_reduce_(
            0,
            imap,
            torch.arange(positions.numel(), device=positions.device),
            reduce="amin",
        )
        representatives.append(b * iv.numel() + positions[first])
        image_groups[b, iv], pose_groups[b, pv] = imap + offset, pmap + offset
        observed_groups.append(observed)
        offset += times.numel()
    return (
        image_groups,
        pose_groups,
        torch.cat(observed_groups),
        torch.cat(representatives),
    )


class CoupledStateActionNoising:
    """Uniform per-state-time strengths; independent per-command-time strengths."""

    def __init__(self, maximum=1.0):
        if not math.isfinite(maximum) or not 0 < maximum <= 1:
            raise ValueError("maximum must be in (0,1]")
        self.maximum = maximum

    def __call__(self, world):
        image_groups, pose_groups, observed, _ = state_groups(world)
        levels = {
            k: torch.zeros_like(world[k]["time"], dtype=torch.float32) for k in STREAMS
        }
        strengths = (
            torch.rand(observed.shape, device=observed.device) * self.maximum
        ).masked_fill(observed, 0)
        for name, groups in (("image", image_groups), ("proprio", pose_groups)):
            levels[name] = strengths[groups].masked_fill(~world[name]["valid"], 0)
        levels["action"] = LinearGaussianImageNoising.frame_noise_levels(
            world["action"], self.maximum
        )
        result = {k: {} for k in ("values", "levels", "targets", "masks", "metadata")}
        for name in STREAMS:
            stream = world[name]
            valid, observed = stream["valid"], stream["observed"]
            known = valid & stream.get("target_valid", valid)
            # An unlabelled command can remain a pure-noise attention token but
            # must never receive an invented command label or a clean GT input.
            clean = torch.where((known | observed)[..., None], stream["value"], 0)
            if not torch.isfinite(clean).all():
                raise ValueError(f"Nonfinite {name} training value")
            if name != "action" and (valid & ~known).any():
                raise ValueError(
                    "Coupled state training requires recorded state targets"
                )
            strength = levels[name].masked_fill(valid & ~known & ~observed, 1)
            sigma = strength[..., None].to(clean.dtype)
            noisy = (1 - sigma) * clean + sigma * torch.randn_like(clean)
            result["values"][name] = noisy.masked_fill(~valid[..., None], 0)
            result["levels"][name] = strength
            result["targets"][name] = clean
            result["masks"][name] = known & ~observed
            result["metadata"][name] = {
                "timestamps": stream["time"],
                "valid": valid,
                "observed": observed,
            }
        return result


class JointMoTBlock(MoTBlock):
    """Three AdaLN-Zero experts and one shared full-attention operation."""

    def __init__(
        self,
        width,
        heads,
        mlp_ratio,
        attention_backend,
        attention_dtype,
        compile_mode,
        compile_dynamic,
        compile_backend,
    ):
        super().__init__(
            width,
            heads,
            mlp_ratio,
            modalities=STREAMS,
            attention_backend=attention_backend,
            attention_dtype=attention_dtype,
        )
        self.norm1 = nn.ModuleList(
            nn.LayerNorm(width, elementwise_affine=False, eps=1e-6) for _ in STREAMS
        )
        self.norm2 = nn.ModuleList(
            nn.LayerNorm(width, elementwise_affine=False, eps=1e-6) for _ in STREAMS
        )
        self.adaln = nn.ModuleList(
            nn.Sequential(nn.SiLU(), nn.Linear(width, 6 * width)) for _ in STREAMS
        )
        for mod in self.adaln:
            nn.init.zeros_(mod[-1].weight)
            nn.init.zeros_(mod[-1].bias)
        for name in ("project", "finish"):
            object.__setattr__(
                self,
                name + "_execute",
                compile_region(
                    getattr(self, name),
                    mode=compile_mode,
                    dynamic=compile_dynamic,
                    backend=compile_backend,
                ),
            )

    def project(self, streams, conditions, rope):
        projected, mods = [], []
        for i, x in enumerate(streams):
            condition, indices = conditions[i]
            grouped = self.adaln[i](condition).chunk(6, dim=-1)
            mod = [self._frame_values(value, indices) for value in grouped[:2]]
            normalized = self.norm1[i](x) * (1 + mod[1]) + mod[0]
            q, k, v = (
                self.qkv[i](normalized)
                .reshape(*x.shape[:2], 3, self.heads, self.head_dim)
                .permute(2, 0, 3, 1, 4)
            )
            projected.append(
                (apply_temporal_rope(q, rope[i]), apply_temporal_rope(k, rope[i]), v)
            )
            mods.append((grouped, indices))
        return tuple(torch.cat([p[j] for p in projected], 2) for j in range(3)), mods

    def finish(self, streams, attended, mods, valid):
        parts = (
            attended.transpose(1, 2).flatten(2).split([s.shape[1] for s in streams], 1)
        )
        outputs = []
        for i, (x, y, grouped) in enumerate(zip(streams, parts, mods)):
            mod, indices = grouped
            gate, shift, scale, mlp_gate = [
                self._frame_values(value, indices) for value in mod[2:]
            ]
            x = x + gate * self.out[i](y)
            x = x + mlp_gate * self.ffn[i](self.norm2[i](x) * (1 + scale) + shift)
            outputs.append(x.masked_fill(~valid[i][..., None], 0))
        return outputs

    def forward(self, streams, conditions, rope, valid, layout, packing=None):
        (q, k, v), mods = self.project_execute(streams, conditions, rope)
        if packing is None:
            mask = torch.cat(valid, 1)[:, None, None, :]
            attended = self._apply_attention(q, k, v, mask, attention_layout=layout)
        else:
            order, inverse, counts = packing
            q, k, v = (t.index_select(2, order) for t in (q, k, v))
            if self.attention_backend in ("flash3", "flash4"):
                dtype = q.dtype
                kernels = load_flash_attention_kernels(self.attention_backend, q.device)
                with torch.autocast(device_type=q.device.type, enabled=False):
                    packed = [
                        t.squeeze(0).transpose(0, 1).to(torch.bfloat16)
                        for t in (q, k, v)
                    ]
                    attended = execute_packed_flash_attention(*packed, layout, kernels)
                attended = attended.to(dtype).transpose(0, 1).unsqueeze(0)
            else:
                # Reference execution uses the identical example boundaries;
                # avoids allocating a quadratic cross-example block mask.
                split = [t.split(counts, 2) for t in (q, k, v)]
                attended = torch.cat(
                    [F.scaled_dot_product_attention(*item) for item in zip(*split)], 2
                )
            attended = attended.index_select(2, inverse)
        return self.finish_execute(streams, attended, mods, valid)


class JointStateActionModel(nn.Module):
    """Fresh jointly trained image/proprio/action MoT; no phase/readout query."""

    def __init__(
        self,
        pose_dims,
        action_dims,
        image_dim=768,
        width=768,
        depth=8,
        heads=6,
        mlp_ratio=4,
        max_cameras=2,
        max_tokens=12928,
        time_mode="continuous",
        rope_axis_dims=(64, 32, 32),
        spatial_grid_size=(10, 10),
        attention_backend="sdpa",
        attention_dtype="native",
        token_execution="packed_valid",
        compile_mode="none",
        compile_dynamic=False,
        compile_backend="inductor",
    ):
        super().__init__()
        if set(pose_dims) != set(action_dims) or not pose_dims:
            raise ValueError("Declare matching state/action embodiment dimensions")
        if any(
            type(d) is not int or d < 1
            for d in (*pose_dims.values(), *action_dims.values())
        ):
            raise ValueError("State/action dimensions must be positive integers")
        if image_dim > width:
            raise ValueError("Raw RGB patch compression must be explicitly redesigned")
        if time_mode not in ("continuous", "discrete") or token_execution not in (
            "padded",
            "packed_valid",
        ):
            raise ValueError("Unknown time mode or token execution")
        self.pose_dims, self.action_dims = dict(pose_dims), dict(action_dims)
        self.image_dim, self.width, self.max_tokens = image_dim, width, max_tokens
        self.time_mode, self.token_execution = time_mode, token_execution
        self.rope_axis_dims, self.spatial_grid_size = rope_axis_dims, spatial_grid_size
        self.inputs = nn.ModuleDict(
            {
                "image": nn.Linear(image_dim, width),
                "proprio": nn.ModuleDict(
                    {k: nn.Linear(d, width) for k, d in pose_dims.items()}
                ),
                "action": nn.ModuleDict(
                    {k: nn.Linear(d, width) for k, d in action_dims.items()}
                ),
            }
        )
        self.outputs = nn.ModuleDict(
            {
                "image": nn.Linear(width, image_dim),
                "proprio": nn.ModuleDict(
                    {k: nn.Linear(width, d) for k, d in pose_dims.items()}
                ),
                "action": nn.ModuleDict(
                    {k: nn.Linear(width, d) for k, d in action_dims.items()}
                ),
            }
        )
        self.camera = nn.Embedding(max_cameras, width)
        self.noise_embeddings = nn.ModuleList(
            nn.Sequential(nn.Linear(256, width), nn.SiLU(), nn.Linear(width, width))
            for _ in STREAMS
        )
        for embedding in self.noise_embeddings:
            for layer in (embedding[0], embedding[2]):
                nn.init.normal_(layer.weight, std=0.02)
                nn.init.zeros_(layer.bias)
        self.blocks = nn.ModuleList(
            JointMoTBlock(
                width,
                heads,
                mlp_ratio,
                attention_backend,
                attention_dtype,
                compile_mode,
                compile_dynamic,
                compile_backend,
            )
            for _ in range(depth)
        )
        self.norms = nn.ModuleList(
            nn.LayerNorm(width, elementwise_affine=False, eps=1e-6) for _ in STREAMS
        )
        self.final_adaln = nn.ModuleList(
            nn.Sequential(nn.SiLU(), nn.Linear(width, 2 * width)) for _ in STREAMS
        )
        for module in self.final_adaln:
            nn.init.zeros_(module[-1].weight)
            nn.init.zeros_(module[-1].bias)
        for module in self.outputs.modules():
            if isinstance(module, nn.Linear):
                nn.init.zeros_(module.weight)
                nn.init.zeros_(module.bias)

    def validate(self, world, embodiment):
        if set(world) != set(STREAMS) or embodiment not in self.pose_dims:
            raise ValueError("Joint model requires exactly image, proprio and action")
        dimensions = (
            self.image_dim,
            self.pose_dims[embodiment],
            self.action_dims[embodiment],
        )
        batch = world["image"]["value"].shape[0]
        for name, dim in zip(STREAMS, dimensions):
            s = world[name]
            if (
                s["value"].ndim != 3
                or s["value"].shape[0] != batch
                or s["value"].shape[-1] != dim
            ):
                raise ValueError(f"Invalid {name} dimensions")
            shape = s["value"].shape[:2]
            for key in ("valid", "observed"):
                if s[key].shape != shape or s[key].dtype != torch.bool:
                    raise ValueError(f"Invalid {name} {key} mask")
            if (s["observed"] & ~s["valid"]).any():
                raise ValueError("Padding cannot be observed")
            if (
                s["time"].shape != shape
                or not torch.isfinite(s["time"][s["valid"]]).all()
            ):
                raise ValueError("Invalid physical timestamps")
            if not torch.isfinite(
                torch.where(s["observed"][..., None], s["value"], 0)
            ).all():
                raise ValueError("Nonfinite observed context")
            if self.time_mode == "discrete" and (
                "index" not in s
                or s["index"].shape != shape
                or not torch.isfinite(s["index"][s["valid"]]).all()
                or (s["index"][s["valid"]] != s["index"][s["valid"]].round()).any()
            ):
                raise ValueError("Discrete mode requires integer grid indices")
        if sum(s["value"].shape[1] for s in world.values()) > self.max_tokens:
            raise ValueError("Joint token budget exceeded")
        if not world["image"]["observed"].any(1).all():
            raise ValueError("Each example requires an observed state")
        return state_groups(world)

    def forward(self, world, embodiment, values, levels):
        image_groups, pose_groups, _, representatives = self.validate(world, embodiment)
        valid = [world[k]["valid"] for k in STREAMS]
        layout = prepare_attention_layout(
            torch.cat(valid, 1), query_valid=torch.cat(valid, 1), all_valid=False
        )
        packed = self.token_execution == "packed_valid"
        packing = None
        if packed:
            ids = torch.arange(valid[0].shape[0], device=valid[0].device)
            sample_ids = torch.cat([ids[:, None].expand_as(v)[v] for v in valid])
            order = torch.argsort(sample_ids, stable=True)
            inverse = torch.argsort(order)
            counts = torch.cat(valid, 1).sum(1).cpu().tolist()
            packing = order, inverse, counts
        streams, conditions, rope, active = [], [], [], []
        for i, name in enumerate(STREAMS):
            s, v = world[name], valid[i]
            if values[name].shape != s["value"].shape or levels[name].shape != v.shape:
                raise ValueError("Noised value/level shape mismatch")
            level = levels[name]
            if ((~torch.isfinite(level) | (level < 0) | (level > 1)) & v).any():
                raise ValueError("Noise strengths must be finite in [0,1]")
            if (level.masked_select(s["observed"]) != 0).any():
                raise ValueError("Observed state/control must be clean")
            frame_map = ImageProprioModel._prepare_noise_frame_map(s)
            grouped_levels = level.reshape(-1).index_select(
                0, frame_map.representatives
            )
            indices = frame_map.token_groups
            if ((grouped_levels[indices] != level) & v).any():
                raise ValueError(
                    "Noise strength must agree within each stream timestamp"
                )
            value = torch.where(s["observed"][..., None], s["value"], values[name])
            time = s["time"] if self.time_mode == "continuous" else s["index"]
            xy = s.get("xy") if name == "image" else None
            if packed:
                value, level, time = value[v][None], level[v][None], time[v][None]
                indices = indices[v][None]
                xy = None if xy is None else xy[v][None]
                mask = torch.ones_like(time, dtype=torch.bool)
            else:
                value = value.masked_fill(~v[..., None], 0)
                level, time = level.masked_fill(~v, 0), time.masked_fill(~v, 0)
                xy = None if xy is None else xy.masked_fill(~v[..., None], 0)
                mask = v
            projection = (
                self.inputs[name] if name == "image" else self.inputs[name][embodiment]
            )
            x = projection(value)
            if name == "image":
                camera = (
                    s["camera"][v][None] if packed else s["camera"].masked_fill(~v, 0)
                )
                x = x + self.camera(camera)
            frequency = torch.exp(
                -math.log(10000) * torch.arange(128, device=level.device) / 128
            )
            phase = grouped_levels.float()[:, None] * 1000 * frequency
            conditions.append(
                (
                    self.noise_embeddings[i](torch.cat((phase.cos(), phase.sin()), -1)),
                    indices,
                )
            )
            rope.append(
                prepare_spatiotemporal_rope(
                    time,
                    xy,
                    self.blocks[0].head_dim,
                    axis_dims=self.rope_axis_dims,
                    spatial_grid_size=self.spatial_grid_size,
                )
            )
            streams.append(x.masked_fill(~mask[..., None], 0))
            active.append(mask)
        # Reject direct callers that violate the state corruption contract.
        grouped = levels["image"].reshape(-1).index_select(0, representatives)
        for name, groups in (("image", image_groups), ("proprio", pose_groups)):
            if ((levels[name] != grouped[groups]) & world[name]["valid"]).any():
                raise ValueError("Image/proprio noise strengths must be tied")
        for block in self.blocks:
            streams = block(streams, conditions, rope, active, layout, packing)
        result = {}
        for i, name in enumerate(STREAMS):
            condition, indices = conditions[i]
            shift, scale = [
                JointMoTBlock._frame_values(value, indices)
                for value in self.final_adaln[i](condition).chunk(2, -1)
            ]
            x = self.norms[i](streams[i]) * (1 + scale) + shift
            output = (
                self.outputs[name]
                if name == "image"
                else self.outputs[name][embodiment]
            )
            prediction = output(x)
            if packed:
                prediction = prediction.new_zeros(
                    *world[name]["value"].shape
                ).masked_scatter(
                    valid[i][..., None].expand_as(world[name]["value"]),
                    prediction.reshape(-1),
                )
            result[name] = prediction.masked_fill(~valid[i][..., None], 0)
        return result


class JointStateActionSampler:
    """Replaceable joint sampler; clamps state observations and known controls."""

    def __init__(self, state_steps=20, action_steps=20):
        if any(type(n) is not int or n < 1 for n in (state_steps, action_steps)):
            raise ValueError("Sampler step counts must be positive integers")
        self.state_steps, self.action_steps = state_steps, action_steps

    @torch.no_grad()
    def __call__(self, model, world, embodiment):
        model.validate(world, embodiment)
        current = {}
        for k in STREAMS:
            s = world[k]
            current[k] = torch.where(
                s["observed"][..., None], s["value"], torch.randn_like(s["value"])
            ).masked_fill(~s["valid"][..., None], 0)
        for j in range(max(self.state_steps, self.action_steps)):
            scalar = {
                k: max(
                    1 - j / (self.action_steps if k == "action" else self.state_steps),
                    0,
                )
                for k in STREAMS
            }
            levels = {
                k: torch.full_like(
                    world[k]["time"], scalar[k], dtype=torch.float32
                ).masked_fill(world[k]["observed"] | ~world[k]["valid"], 0)
                for k in STREAMS
            }
            x0 = model(world, embodiment, current, levels)
            for k in STREAMS:
                if scalar[k] == 0:
                    continue
                s = world[k]
                next_level = max(
                    1
                    - (j + 1)
                    / (self.action_steps if k == "action" else self.state_steps),
                    0,
                )
                predicted = x0[k].to(current[k].dtype)
                candidate = (1 - next_level) * predicted + next_level / scalar[k] * (
                    current[k] - (1 - scalar[k]) * predicted
                )
                current[k] = torch.where(
                    s["observed"][..., None], s["value"], candidate
                ).masked_fill(~s["valid"][..., None], 0)
        return current
