"""Minimal two-stream image-patch/proprio models; no point-track dependency.

``visual_proprio`` predicts clean image-patch values under linear Gaussian
corruption, then reads out withheld proprioceptive states in a separate phase.
``direct_action`` regresses recorded action commands from clean observed image
patches and observed proprioception. Action labels are never model inputs.
``visual_only`` predicts image patches, without state/action heads, withheld
proprio queries, or a readout phase. With phase_a_condition_on_proprio=False it
constructs only an image expert and takes only images. The same flag keeps
phase A image-only when later adding a proprio expert for phase B readout.
``visual_state_action`` adds typed state and command queries in phase B, sharing
the existing proprio branch. Both readouts use MSE; action labels stay loss-only.

Prepared streams contain value[B,N,D], time[B,N], valid[B,N], observed[B,N].
Image tokens additionally carry xy[B,N,2] and camera[B,N]. Physical time and
noise level are separate conditioning variables. No image observed/query flag
is embedded. Target availability is used only by the loss. Image values may be
prepared features or flattened RGB patches; the model adds no image encoder.
Camera identity conditioning is enabled by default to distinguish multiple views.
Modality and embodiment identity embeddings are disabled by default; private
modality weights and the dimension-specific proprio adapters remain in use.
Channel compression requires an explicit constructor opt-in. That opt-in does
not add a wide denoising head or remove the resulting low-rank output constraint.
Attention execution is independently selectable: native SDPA remains the default,
while explicit BF16 attention changes only Q/K/V and attention output arithmetic.
The caller owns trainer precision and the TF32 policy for the remaining model.
Learned phase offsets are optional; A/B labels always select control flow.
The optional spatial-temporal RoPE mode replaces additive XY with axial query/key
rotations in centered patch units. Camera conditioning remains separate. State
and command tokens share image temporal frequencies and have zero spatial phases.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from egomimic.models.attention_execution import (
    AttentionLayout,
    load_flash_attention_kernels,
    prepare_attention_layout,
    prepare_spatiotemporal_rope,
    prepare_temporal_rope,
    resolve_spatiotemporal_rope_geometry,
)
from egomimic.models.compilation import compile_region
from egomimic.models.pixel_diffusion import (
    LinearDDIMImageSampler,
    LinearGaussianImageNoising,
    PlainPixelMSE,
)
from egomimic.models.world_model import MoTBlock

MODALITIES = ("image", "proprio")


@dataclass(frozen=True)
class _PackedImageExecution:
    layout: AttentionLayout
    image: Mapping[str, torch.Tensor]
    valid: torch.Tensor
    rope: tuple[torch.Tensor, torch.Tensor]


@dataclass(frozen=True)
class _NoiseFrameMap:
    representatives: torch.Tensor
    token_groups: torch.Tensor


@dataclass(frozen=True)
class _ConditionedImageExecution:
    attention: object
    noise_frames: _NoiseFrameMap


class ImageProprioModel(nn.Module):
    def __init__(
        self,
        pose_dims: Mapping[str, int] | None = None,
        action_dims: Mapping[str, int] | None = None,
        objective="visual_proprio",
        image_dim=768,
        width=768,
        depth=8,
        heads=6,
        mlp_ratio=4,
        max_cameras=3,
        max_tokens=76816,
        time_mode="continuous",
        attention_query_chunk_size=0,
        activation_checkpointing=False,
        pose_conditioning="low_noise",
        pose_noise_max=0.2,
        allow_patch_compression=False,
        use_camera_embedding=True,
        use_modality_embedding=False,
        use_embodiment_embedding=False,
        attention_backend="sdpa",
        attention_dtype="native",
        strict_fp32=False,
        phase_a_condition_on_proprio=True,
        loss_reduction="token_mean",
        compile_mode="none",
        compile_dynamic=False,
        compile_backend="inductor",
        token_execution="padded",
        use_phase_embedding=True,
        noise_conditioning="additive",
        position_encoding="linear_xy_temporal_rope",
        rope_axis_dims=None,
        spatial_grid_size=None,
    ):
        super().__init__()
        if type(strict_fp32) is not bool:
            raise ValueError("strict_fp32 must be a boolean")
        if strict_fp32 and attention_dtype != "native":
            raise ValueError(
                "Explicit BF16 attention conflicts with strict_fp32; "
                "select the mixed-attention execution profile"
            )
        if objective not in (
            "visual_only",
            "visual_proprio",
            "visual_state_action",
            "direct_action",
        ):
            raise ValueError(
                "objective must be visual_only, visual_proprio, visual_state_action or direct_action"
            )
        if type(phase_a_condition_on_proprio) is not bool:
            raise ValueError("phase_a_condition_on_proprio must be bool")
        if objective == "direct_action" and not phase_a_condition_on_proprio:
            raise ValueError("Direct action BC requires its observed proprio context")
        self.phase_a_condition_on_proprio = phase_a_condition_on_proprio
        self.has_proprio_branch = (
            objective != "visual_only" or phase_a_condition_on_proprio
        )
        self.modalities = MODALITIES if self.has_proprio_branch else ("image",)
        if not self.has_proprio_branch:
            if pose_dims not in (None, {}):
                raise ValueError("Image-only visual_only must omit pose_dims")
            if use_embodiment_embedding:
                raise ValueError("Image-only visual_only has no embodiment embedding")
        if type(image_dim) is not int or image_dim < 1:
            raise ValueError("image_dim must be a positive integer")
        if type(allow_patch_compression) is not bool:
            raise ValueError("allow_patch_compression must be bool")
        for name, value in (
            ("use_camera_embedding", use_camera_embedding),
            ("use_modality_embedding", use_modality_embedding),
            ("use_embodiment_embedding", use_embodiment_embedding),
            ("use_phase_embedding", use_phase_embedding),
        ):
            if type(value) is not bool:
                raise ValueError(f"{name} must be bool")
        if type(width) is not int or width < 1:
            raise ValueError("width must be a positive integer")
        if width < image_dim and not allow_patch_compression:
            raise ValueError(
                "Hidden width must be at least image_dim unless "
                "allow_patch_compression=True is explicitly selected"
            )
        if type(depth) is not int or depth < 1:
            raise ValueError("depth must be a positive integer")
        if type(heads) is not int or heads < 1:
            raise ValueError("heads must be a positive integer")
        if type(mlp_ratio) is not int or mlp_ratio < 1:
            raise ValueError("mlp_ratio must be a positive integer")
        if type(max_tokens) is not int or max_tokens < 1:
            raise ValueError("max_tokens must be a positive integer")
        if type(activation_checkpointing) is not bool:
            raise ValueError("activation_checkpointing must be bool")
        if time_mode not in ("continuous", "discrete"):
            raise ValueError("time_mode must be continuous or discrete")
        if pose_conditioning not in ("clean", "low_noise"):
            raise ValueError("pose_conditioning must be clean or low_noise")
        if not math.isfinite(pose_noise_max) or not 0 < pose_noise_max <= 0.2:
            raise ValueError("pose_noise_max must be in (0, 0.2]")
        self.pose_dims = (
            self._dimensions(pose_dims, "pose_dims") if self.has_proprio_branch else {}
        )
        self.action_dims = (
            self._dimensions(action_dims, "action_dims")
            if objective in ("direct_action", "visual_state_action")
            else {}
        )
        if (
            objective in ("direct_action", "visual_state_action")
            and self.action_dims.keys() != self.pose_dims.keys()
        ):
            raise ValueError("action_dims and pose_dims must name the same embodiments")
        if (
            objective not in ("direct_action", "visual_state_action")
            and action_dims is not None
        ):
            raise ValueError("This objective has no action head; omit action_dims")
        self.objective = objective
        if noise_conditioning not in ("additive", "adaln_zero"):
            raise ValueError("noise_conditioning must be additive or adaln_zero")
        if noise_conditioning == "adaln_zero" and objective == "direct_action":
            raise ValueError("Direct action BC has no diffusion noise conditioning")
        self.noise_conditioning = noise_conditioning
        if position_encoding not in ("linear_xy_temporal_rope", "spatiotemporal_rope"):
            raise ValueError(
                "position_encoding must be linear_xy_temporal_rope or spatiotemporal_rope"
            )
        self.position_encoding = position_encoding
        if position_encoding == "spatiotemporal_rope":
            if width % heads:
                raise ValueError("Model width must be divisible by attention heads")
            self.rope_axis_dims, self.spatial_grid_size = (
                resolve_spatiotemporal_rope_geometry(
                    width // heads, rope_axis_dims, spatial_grid_size
                )
            )
        else:
            if rope_axis_dims is not None or spatial_grid_size is not None:
                raise ValueError(
                    "Spatial RoPE geometry requires position_encoding='spatiotemporal_rope'"
                )
            self.rope_axis_dims = self.spatial_grid_size = None
        self.image_dim, self.width = image_dim, width
        self.allow_patch_compression = allow_patch_compression
        self.use_camera_embedding = use_camera_embedding
        self.use_modality_embedding = use_modality_embedding
        self.use_embodiment_embedding = use_embodiment_embedding
        self.use_phase_embedding = use_phase_embedding
        self.max_tokens, self.time_mode = max_tokens, time_mode
        self.activation_checkpointing = activation_checkpointing
        self.attention_backend = attention_backend
        self.attention_dtype = attention_dtype
        self.strict_fp32 = strict_fp32
        if loss_reduction not in ("token_mean", "mean_per_example"):
            raise ValueError("loss_reduction must be token_mean or mean_per_example")
        self.loss_reduction = loss_reduction
        self.compile_mode, self.compile_dynamic = compile_mode, compile_dynamic
        self.compile_backend = compile_backend
        if token_execution not in ("padded", "packed_valid"):
            raise ValueError("token_execution must be padded or packed_valid")
        self.token_execution = token_execution
        self.pose_conditioning, self.pose_noise_max = pose_conditioning, pose_noise_max
        self.image_in = nn.Linear(image_dim, width)
        if self.has_proprio_branch:
            self.pose_in = nn.ModuleDict(
                {k: nn.Linear(d, width) for k, d in self.pose_dims.items()}
            )
        if position_encoding == "linear_xy_temporal_rope":
            self.xy = nn.Linear(2, width)
        if use_camera_embedding:
            self.camera = nn.Embedding(max_cameras, width)
        if use_modality_embedding:
            self.modality = nn.Parameter(
                torch.randn(len(self.modalities), width) * 0.02
            )
        if use_embodiment_embedding:
            self.embodiment = nn.Embedding(len(self.pose_dims), width)
            self.embodiment_ids = {k: i for i, k in enumerate(self.pose_dims)}
        roles = (
            1
            if objective == "visual_only"
            else 3
            if objective == "visual_state_action"
            else 2
        )
        if self.has_proprio_branch:
            self.proprio_role = nn.Embedding(roles, width)
        if objective != "visual_only":
            self.query = nn.Parameter(torch.randn(width) * 0.02)
        if objective == "visual_state_action":
            self.action_query = nn.Parameter(torch.randn(width) * 0.02)
        if objective != "direct_action":
            self.image_out = nn.Linear(width, image_dim)
            if objective in ("visual_proprio", "visual_state_action"):
                self.pose_out = nn.ModuleDict(
                    {k: nn.Linear(width, d) for k, d in self.pose_dims.items()}
                )
            self.noise_level = nn.Sequential(
                nn.Linear(256 if noise_conditioning == "adaln_zero" else 1, width),
                nn.SiLU(),
                nn.Linear(width, width),
            )
            if use_phase_embedding:
                self.phase = nn.Embedding(1 if objective == "visual_only" else 2, width)
        if objective in ("direct_action", "visual_state_action"):
            self.action_out = nn.ModuleDict(
                {k: nn.Linear(width, d) for k, d in self.action_dims.items()}
            )
        self.blocks = nn.ModuleList(
            MoTBlock(
                width,
                heads,
                mlp_ratio,
                attention_query_chunk_size,
                modalities=self.modalities,
                attention_backend=attention_backend,
                attention_dtype=attention_dtype,
                image_adaln=noise_conditioning == "adaln_zero",
            )
            for _ in range(depth)
        )
        self.norm = nn.ModuleList(
            [
                nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
                if noise_conditioning == "adaln_zero"
                else nn.LayerNorm(width)
            ]
        )
        if self.has_proprio_branch:
            self.norm.append(
                nn.Identity() if objective == "visual_only" else nn.LayerNorm(width)
            )
        if noise_conditioning == "adaln_zero":
            self.final_adaln = nn.Sequential(nn.SiLU(), nn.Linear(width, 2 * width))
            nn.init.zeros_(self.final_adaln[-1].weight)
            nn.init.zeros_(self.final_adaln[-1].bias)
            nn.init.zeros_(self.image_out.weight)
            nn.init.zeros_(self.image_out.bias)
            for layer in (self.noise_level[0], self.noise_level[2]):
                nn.init.normal_(layer.weight, std=0.02)
                nn.init.zeros_(layer.bias)
        # Compile pure tensor regions, keeping validation, packing, and optional
        # attention-library dispatch outside graph capture. Never register an
        # OptimizedModule: historical checkpoint parameter names stay intact.
        for block in self.blocks:
            block.configure_compilation(
                mode=compile_mode, dynamic=compile_dynamic, backend=compile_backend
            )
        compiled_names = ("_embed_image", "_decode_image")
        if noise_conditioning == "adaln_zero":
            compiled_names += ("_embed_noise_frames",)
        for name in compiled_names:
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

    @staticmethod
    def _dimensions(dims, name):
        if (
            not isinstance(dims, Mapping)
            or not dims
            or any(
                not isinstance(k, str) or not k or type(d) is not int or d < 1
                for k, d in dims.items()
            )
        ):
            raise ValueError(
                f"{name} requires positive integer dimensions by embodiment"
            )
        return dict(dims)

    def validate(self, batch, embodiment):
        if self.has_proprio_branch and embodiment not in self.pose_dims:
            raise ValueError(f"Unknown embodiment {embodiment!r}")
        if not self.has_proprio_branch and set(batch) != {"image"}:
            raise ValueError("Image-only visual_only accepts exactly the image stream")
        if "track" in batch:
            raise ValueError("Image-patch/proprio models do not accept a track stream")
        dimensions = {"image": self.image_dim}
        if self.has_proprio_branch:
            dimensions["proprio"] = self.pose_dims[embodiment]
        if self.objective in ("direct_action", "visual_state_action"):
            dimensions["action"] = self.action_dims[embodiment]
        total, batch_size = 0, None
        checks = []
        for name, dim in dimensions.items():
            stream = batch[name]
            value = stream["value"]
            if (
                value.ndim != 3
                or value.shape[-1] != dim
                or not value.is_floating_point()
            ):
                raise ValueError(f"{name}: invalid value shape or dtype")
            if batch_size is not None and batch_size != value.shape[0]:
                raise ValueError("Mixed batch sizes")
            batch_size = value.shape[0]
            total += value.shape[1]
            for key in ("time", "valid", "observed"):
                if stream[key].shape != value.shape[:2]:
                    raise ValueError(f"{name}: {key} shape mismatch")
            valid, observed = stream["valid"], stream["observed"]
            if valid.dtype != torch.bool or observed.dtype != torch.bool:
                raise ValueError("Masks must be boolean")
            checks.append(((observed & ~valid).any(), "Padding cannot be observed"))
            if "target_valid" in stream and (
                stream["target_valid"].shape != valid.shape
                or stream["target_valid"].dtype != torch.bool
            ):
                raise ValueError("Target validity must be a boolean token mask")
            checks.append(
                (
                    ~torch.isfinite(stream["time"].masked_fill(~valid, 0)).all(),
                    "Non-finite timestamp",
                )
            )
            checks.append(
                (
                    ~torch.isfinite(torch.where(observed[..., None], value, 0)).all(),
                    "Non-finite observed value",
                )
            )
            if name == "action":
                checks.append(
                    (observed.any(), "Action labels must never be marked observed")
                )
            if self.time_mode == "discrete":
                if "index" not in stream or stream["index"].shape != valid.shape:
                    raise ValueError("Discrete mode requires explicit grid indices")
                indices = stream["index"].masked_fill(~valid, 0)
                checks.append(
                    (
                        (~torch.isfinite(indices) | (indices != indices.round())).any(),
                        "Discrete indices must be finite integers",
                    )
                )
            if name == "image":
                if stream["xy"].shape != (*valid.shape, 2):
                    raise ValueError("Reference image coordinates must match tokens")
                checks.append(
                    (
                        ~torch.isfinite(
                            torch.where(valid[..., None], stream["xy"], 0)
                        ).all(),
                        "Non-finite reference image coordinates",
                    )
                )
                if self.use_camera_embedding:
                    ids = stream["camera"]
                    if ids.shape != valid.shape or ids.dtype != torch.long:
                        raise ValueError("Camera IDs must be integer token IDs")
                    checks.append(
                        (
                            (
                                ((ids < 0) | (ids >= self.camera.num_embeddings))
                                & valid
                            ).any(),
                            "Camera ID out of range",
                        )
                    )
        if total > self.max_tokens:
            raise ValueError(f"Token budget exceeded: {total} > {self.max_tokens}")
        # A query without any clean observation is not a conditional BC example.
        conditioning = MODALITIES if self.phase_a_condition_on_proprio else ("image",)
        context = torch.cat([batch[n]["observed"] for n in conditioning], 1)
        checks.append(
            (
                ~context.any(1).all(),
                "Every sample requires an observed image or proprio token",
            )
        )
        # One device-to-host synchronization for all value checks. Boolean
        # indexing/nonzero used here previously introduced additional syncs.
        for failed, (_, message) in zip(
            torch.stack([item[0] for item in checks]).tolist(), checks
        ):
            if failed:
                raise ValueError(message)

    def _clock(self, stream, valid):
        key = "time" if self.time_mode == "continuous" else "index"
        return stream[key].masked_fill(~valid, 0)

    def forward(
        self,
        batch,
        embodiment,
        image_values=None,
        levels=None,
        phase="A",
        *,
        _validated=False,
        _execution=None,
    ):
        """Forward one phase. Withheld labels never enter an embedding.

        Phase A excludes withheld proprio queries from joint attention entirely.
        Phase B adds those queries, whose values are learned constants. In the
        direct-action variant only observed context and action queries are used.
        """
        if not _validated:
            self.validate(batch, embodiment)
            if self.objective != "direct_action":
                image = batch["image"]
                if image_values is None or levels is None:
                    raise ValueError(
                        "Visual phases require image_values and noise levels"
                    )
                if (
                    image_values.shape != image["value"].shape
                    or levels.shape != image["valid"].shape
                ):
                    raise ValueError("Visual values/levels must match image tokens")
                bad_levels = (
                    ~torch.isfinite(levels) | (levels < 0) | (levels > 1)
                ) & image["valid"]
                if bad_levels.any():
                    raise ValueError("Noise levels must be finite in [0,1]")
        return self._forward_validated(
            batch, embodiment, image_values, levels, phase, execution=_execution
        )

    def _prepare_rope(self, times, xy=None):
        if self.position_encoding == "spatiotemporal_rope":
            return prepare_spatiotemporal_rope(
                times,
                xy,
                self.blocks[0].head_dim,
                axis_dims=self.rope_axis_dims,
                spatial_grid_size=self.spatial_grid_size,
            )
        return prepare_temporal_rope(times, self.blocks[0].head_dim)

    def _prepare_execution(self, times, valid, image_xy=None):
        joint_valid = torch.cat(valid, 1)
        layout = None
        fused_backends = {block.attention_backend for block in self.blocks} & {
            "flash3",
            "flash4",
        }
        if fused_backends:
            # Respect explicit block execution overrides used by the offline
            # weights-preservation proof; model config identity stays recorded.
            for backend in fused_backends:
                load_flash_attention_kernels(backend, joint_valid.device)
            layout = prepare_attention_layout(joint_valid, query_valid=joint_valid)
        if self.position_encoding == "spatiotemporal_rope" and image_xy is None:
            raise ValueError(
                "Image spatial-temporal attention requires patch coordinates"
            )
        safe_xy = (
            None if image_xy is None else torch.where(valid[0][..., None], image_xy, 0)
        )
        rope = tuple(
            self._prepare_rope(t, safe_xy if index == 0 else None)
            for index, t in enumerate(times)
        )
        return layout, rope

    @staticmethod
    def _prepare_noise_frame_map(image):
        """Map same-example timestamp/observed-role tokens to one noise condition.

        Camera and patch identities are deliberately excluded. Grouping is eager,
        reused by all blocks and cached with attention metadata during sampling.
        Noise values are checked on each call and never cached across sampler steps.
        """
        valid = image["valid"]
        groups = torch.zeros_like(valid, dtype=torch.long)
        representatives, offset = [], 0
        for batch_index in range(valid.shape[0]):
            positions = valid[batch_index].nonzero(as_tuple=False).flatten()
            if positions.numel() == 0:
                continue
            keys = torch.stack(
                (
                    image["time"][batch_index, positions].double(),
                    image["observed"][batch_index, positions].double(),
                ),
                dim=1,
            )
            unique, inverse = torch.unique(keys, dim=0, return_inverse=True)
            first = torch.full(
                (unique.shape[0],),
                positions.numel(),
                device=positions.device,
                dtype=torch.long,
            )
            first.scatter_reduce_(
                0,
                inverse,
                torch.arange(positions.numel(), device=positions.device),
                reduce="amin",
            )
            representatives.append(batch_index * valid.shape[1] + positions[first])
            groups[batch_index, positions] = inverse + offset
            offset += unique.shape[0]
        indices = (
            torch.cat(representatives)
            if representatives
            else torch.zeros(1, device=valid.device, dtype=torch.long)
        )
        return _NoiseFrameMap(indices, groups)

    def _embed_noise_frames(self, levels):
        # Embedding convention only: tau remains the original linear-corruption
        # strength in [0,1]. Multiplying by1000 does not discretize/change it.
        frequency = torch.exp(
            -math.log(10000)
            * torch.arange(128, device=levels.device, dtype=torch.float32)
            / 128
        )
        angles = (1000 * levels.float())[:, None] * frequency[None]
        return self.noise_level(torch.cat((angles.cos(), angles.sin()), dim=-1))

    def _image_noise_condition(self, image, levels, frame_map):
        effective = levels.masked_fill(image["observed"] | ~image["valid"], 0)
        grouped = effective.reshape(-1).index_select(0, frame_map.representatives)
        reconstructed = grouped.index_select(
            0, frame_map.token_groups.reshape(-1)
        ).reshape_as(effective)
        if ((effective != reconstructed) & image["valid"]).any():
            raise ValueError(
                "Image noise levels must agree within each timestamp and observed/query group"
            )
        return self._embed_noise_frames_execute(grouped), frame_map.token_groups

    def _prepare_image_execution(self, image, valid):
        execution = self._prepare_image_attention_execution(image, valid)
        if self.noise_conditioning == "adaln_zero":
            return _ConditionedImageExecution(
                execution, self._prepare_noise_frame_map(image)
            )
        return execution

    def _prepare_image_attention_execution(self, image, valid):
        """Prepare reusable image-only metadata before any learned projection."""
        times = self._clock(image, valid)
        if self.token_execution == "padded":
            return self._prepare_execution([times], [valid], image["xy"])
        layout = prepare_attention_layout(valid, query_valid=valid)
        backends = {block.attention_backend for block in self.blocks}
        for backend in backends & {"flash3", "flash4"}:
            load_flash_attention_kernels(backend, valid.device)
        if layout.kind == "dense":
            # Equal, fully valid batches already have no padding overhead and
            # retain the existing dense kernel and tensor-region shapes.
            return (
                layout if backends & {"flash3", "flash4"} else None,
                (self._prepare_rope(times, image["xy"]),),
            )
        if not backends <= {"flash3", "flash4"}:
            raise ValueError("Ragged packed_valid execution requires flash3 or flash4")
        if any(block.attention_query_chunk_size for block in self.blocks):
            raise ValueError("Packed token execution requires unchunked attention")
        indices = layout.key_indices
        metadata_keys = (
            ("xy", "observed", "camera")
            if self.use_camera_embedding
            else ("xy", "observed")
        )
        packed_image = {
            key: image[key].flatten(0, 1).index_select(0, indices).unsqueeze(0)
            for key in metadata_keys
        }
        packed_times = times.reshape(-1).index_select(0, indices).unsqueeze(0)
        packed_valid = valid.new_ones((1, indices.numel()))
        return _PackedImageExecution(
            layout,
            packed_image,
            packed_valid,
            self._prepare_rope(packed_times, packed_image["xy"]),
        )

    def _forward_packed_image(
        self, values, levels, embodiment, execution, image_condition=None
    ):
        indices = execution.layout.key_indices
        packed_values = values.flatten(0, 1).index_select(0, indices).unsqueeze(0)
        packed_levels = levels.reshape(-1).index_select(0, indices).unsqueeze(0)
        if image_condition is not None:
            condition, frame_groups = image_condition
            image_condition = (
                condition,
                frame_groups.reshape(-1).index_select(0, indices).unsqueeze(0),
            )
        stream = self._embed_image_execute(
            execution.image, packed_values, packed_levels, execution.valid, "A", False
        )
        if self.use_modality_embedding:
            stream = stream + self.modality[0]
        if self.use_embodiment_embedding:
            stream = stream + self.embodiment.weight[self.embodiment_ids[embodiment]]
        for block in self.blocks:
            if self.activation_checkpointing and torch.is_grad_enabled():
                stream = checkpoint(
                    block.forward_packed,
                    stream,
                    execution.valid,
                    attention_layout=execution.layout,
                    rope=execution.rope,
                    image_condition=image_condition,
                    use_reentrant=False,
                )
            else:
                stream = block.forward_packed(
                    stream,
                    execution.valid,
                    attention_layout=execution.layout,
                    rope=execution.rope,
                    image_condition=image_condition,
                )
        predicted = self._decode_image_execute(
            stream, execution.valid, image_condition
        ).squeeze(0)
        batch, tokens = execution.layout.batch_size, execution.layout.key_tokens
        return (
            predicted.new_zeros(batch * tokens, self.image_dim)
            .index_copy(0, indices, predicted)
            .reshape(batch, tokens, self.image_dim)
        )

    def _embed_image(self, image, values, levels, image_valid, phase, direct):
        image_x = self.image_in(torch.where(image_valid[..., None], values, 0))
        if self.position_encoding == "linear_xy_temporal_rope":
            image_x = image_x + self.xy(
                torch.where(image_valid[..., None], image["xy"], 0)
            )
        if self.use_camera_embedding:
            image_x = image_x + self.camera(
                image["camera"].masked_fill(~image_valid, 0)
            )
        if not direct and self.noise_conditioning == "additive":
            effective_levels = levels.masked_fill(image["observed"] | ~image_valid, 0)
            image_x = image_x + self.noise_level(
                effective_levels[..., None].to(image_x.dtype)
            )
        if not direct and self.use_phase_embedding:
            image_x = image_x + self.phase.weight[int(phase == "B")]
        return image_x

    def _decode_image(self, stream, valid, image_condition=None):
        normalized = self.norm[0](stream)
        if self.noise_conditioning == "adaln_zero":
            condition, frame_groups = image_condition
            shift, scale = self.final_adaln(condition).chunk(2, dim=-1)
            normalized = normalized * (
                1 + MoTBlock._frame_values(scale, frame_groups)
            ) + MoTBlock._frame_values(shift, frame_groups)
        return self.image_out(normalized).masked_fill(~valid[..., None], 0)

    def _forward_validated(
        self,
        batch,
        embodiment,
        image_values=None,
        levels=None,
        phase="A",
        execution=None,
    ):
        if phase not in ("A", "B"):
            raise ValueError("phase must be A or B")
        if self.objective == "visual_only" and phase != "A":
            raise ValueError("visual_only has no proprio readout phase")
        direct = self.objective == "direct_action"
        combined = self.objective == "visual_state_action"
        include_pose = self.has_proprio_branch and (
            direct or phase == "B" or self.phase_a_condition_on_proprio
        )
        image = batch["image"]
        image_valid = image["valid"] & (image["observed"] if direct else True)
        if direct:
            if image_values is not None or levels is not None:
                raise ValueError("Direct action BC takes only clean observed features")
            values = image["value"]
        else:
            values = torch.where(
                image["observed"][..., None], image["value"], image_values
            )
        frame_map = None
        if (
            self.noise_conditioning == "adaln_zero"
            and phase == "A"
            and not include_pose
            and execution is None
        ):
            execution = self._prepare_image_execution(image, image_valid)
        if isinstance(execution, _ConditionedImageExecution):
            frame_map = execution.noise_frames
            execution = execution.attention
        image_condition = None
        if self.noise_conditioning == "adaln_zero":
            frame_map = frame_map or self._prepare_noise_frame_map(image)
            image_condition = self._image_noise_condition(image, levels, frame_map)
        # Phase B retains the existing multi-expert route. Packing here changes
        # execution only; no token, camera, or per-example boundary is removed.
        if self.token_execution == "packed_valid" and phase == "A" and not include_pose:
            execution = (
                self._prepare_image_execution(image, image_valid)
                if execution is None
                else execution
            )
            if isinstance(execution, _PackedImageExecution):
                return {
                    "image": self._forward_packed_image(
                        values, levels, embodiment, execution, image_condition
                    )
                }
        image_x = self._embed_image_execute(
            image, values, levels, image_valid, phase, direct
        )
        if include_pose:
            pose = batch["proprio"]
            pose_valid = pose["valid"] & (
                pose["observed"] if direct or phase == "A" else True
            )
            observed_pose = pose["observed"] & pose_valid
            pose_value = torch.where(observed_pose[..., None], pose["value"], 0)
            pose_x = self.pose_in[embodiment](pose_value)
            if self.objective == "visual_only":
                pose_x = pose_x + self.proprio_role.weight[0]
            else:
                pose_x = torch.where(observed_pose[..., None], pose_x, self.query)
                pose_x = pose_x + self.proprio_role((~observed_pose).long())
            pose_time = self._clock(pose, pose_valid)
            action_start = pose_x.shape[1]
        if direct or (combined and phase == "B"):
            action = batch["action"]
            # The shape and query metadata are needed, but action values are not.
            query = (self.action_query if combined else self.query).expand(
                *action["valid"].shape, -1
            )
            query = query + self.proprio_role.weight[2 if combined else 1]
            pose_x = torch.cat((pose_x, query), 1)
            pose_valid = torch.cat((pose_valid, action["valid"]), 1)
            pose_time = torch.cat((pose_time, self._clock(action, action["valid"])), 1)
        if not direct and self.use_phase_embedding:
            phase_embedding = self.phase.weight[int(phase == "B")]
            if include_pose:
                pose_x = pose_x + phase_embedding
        valid = [image_valid]
        times = [self._clock(image, image_valid)]
        streams = [image_x]
        if include_pose:
            valid.append(pose_valid)
            times.append(pose_time)
            streams.append(pose_x)
        if self.use_modality_embedding:
            streams = [x + self.modality[i] for i, x in enumerate(streams)]
        if self.use_embodiment_embedding:
            embodiment_x = self.embodiment.weight[self.embodiment_ids[embodiment]]
            streams = [x + embodiment_x for x in streams]
        streams = [
            x.masked_fill(~valid[i][..., None], 0) for i, x in enumerate(streams)
        ]
        attention_layout, rope = (
            execution
            if execution is not None
            else self._prepare_execution(times, valid, image["xy"])
        )
        for block in self.blocks:
            if self.activation_checkpointing and torch.is_grad_enabled():
                streams = checkpoint(
                    block,
                    streams,
                    times,
                    valid,
                    attention_layout=attention_layout,
                    rope=rope,
                    image_condition=image_condition,
                    use_reentrant=False,
                )
            else:
                streams = block(
                    streams,
                    times,
                    valid,
                    attention_layout=attention_layout,
                    rope=rope,
                    image_condition=image_condition,
                )
        image_result = (
            None
            if direct
            else self._decode_image_execute(streams[0], image_valid, image_condition)
        )
        if include_pose:
            streams[1] = self.norm[1](streams[1])
        if direct:
            action = self.action_out[embodiment](streams[1][:, action_start:])
            return {
                "action": action.masked_fill(~batch["action"]["valid"][..., None], 0)
            }
        result = {"image": image_result}
        if self.objective == "visual_only" or not include_pose:
            return result
        result.update(
            {
                "proprio": self.pose_out[embodiment](
                    streams[1][:, :action_start]
                ).masked_fill(~pose_valid[:, :action_start, None], 0),
            }
        )
        if combined and phase == "B":
            result["action"] = self.action_out[embodiment](
                streams[1][:, action_start:]
            ).masked_fill(~batch["action"]["valid"][..., None], 0)
        return result

    @staticmethod
    def frame_noise_levels(stream, maximum=1.0):
        """One scalar per physical timestamp; observed tokens stay at zero."""
        return LinearGaussianImageNoising.frame_noise_levels(stream, maximum)

    def _noisy_images(self, image, clean, maximum=1.0):
        corruption = LinearGaussianImageNoising(maximum)(image, clean=clean)
        return corruption["image_values"], corruption["noise_levels"]

    def _loss(self, prediction, target, mask, *, check_target=True):
        return PlainPixelMSE(self.loss_reduction)(
            prediction, target, mask, check_target=check_target
        )

    def training_losses(
        self, batch, embodiment, pose_weight=1.0, image_weight=1.0, action_weight=1.0
    ):
        self.validate(batch, embodiment)
        for name, weight in (
            ("pose_weight", pose_weight),
            ("image_weight", image_weight),
            ("action_weight", action_weight),
        ):
            if not math.isfinite(weight) or weight < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.objective == "direct_action":
            prediction = self(batch, embodiment, _validated=True)["action"]
            stream = batch["action"]
            mask = stream["valid"] & stream.get("target_valid", stream["valid"])
            loss = self._loss(prediction, stream["value"], mask)
            return {"action": loss, "total": loss}
        image = batch["image"]
        clean, image_mask = LinearGaussianImageNoising.prepare_target(image)
        noisy, levels = self._noisy_images(image, clean)
        prediction = self(batch, embodiment, noisy, levels, phase="A", _validated=True)
        losses = {
            "image": self._loss(
                prediction["image"],
                clean,
                image_mask,
                check_target=False,
            )
        }
        if self.objective == "visual_only":
            losses["total"] = image_weight * losses["image"]
            return losses
        pose = batch["proprio"]
        pose_mask = (
            pose["valid"] & ~pose["observed"] & pose.get("target_valid", pose["valid"])
        )
        combined = self.objective == "visual_state_action"
        action_mask = None
        if combined:
            action = batch["action"]
            action_mask = action["valid"] & action.get("target_valid", action["valid"])
        if (pose_weight > 0 and pose_mask.any()) or (
            combined and action_weight > 0 and action_mask.any()
        ):
            if self.pose_conditioning == "low_noise":
                # A separate low-noise prediction aligns the readout with this
                # trajectory. Do not train pose readout on arbitrary high-noise
                # x0 predictions. No detach: state loss reaches the image head.
                low_noisy, low_levels = self._noisy_images(
                    image, clean, self.pose_noise_max
                )
                estimated = self(
                    batch, embodiment, low_noisy, low_levels, phase="A", _validated=True
                )["image"]
                pose_images = torch.where(
                    image["observed"][..., None], clean, estimated
                )
            else:
                pose_images = clean
            readout = self(
                batch,
                embodiment,
                pose_images,
                torch.zeros_like(levels),
                phase="B",
                _validated=True,
            )
            losses["pose"] = (
                self._loss(readout["proprio"], pose["value"], pose_mask)
                if pose_weight > 0
                else readout["proprio"].sum() * 0
            )
            if combined:
                losses["action"] = (
                    self._loss(readout["action"], action["value"], action_mask)
                    if action_weight > 0
                    else readout["action"].sum() * 0
                )
        else:
            losses["pose"] = prediction.get("proprio", prediction["image"]).sum() * 0
            if combined:
                losses["action"] = losses["image"] * 0
        losses["total"] = image_weight * losses["image"] + pose_weight * losses["pose"]
        if combined:
            losses["total"] = losses["total"] + action_weight * losses["action"]
        return losses

    @torch.no_grad()
    def sample(self, batch, embodiment, steps=20, generator=None):
        """Sample images from noise, then read states; or one direct BC pass.

        For linear corruption, epsilon=(z_s-(1-s)*x0)/s. Implement the DDIM
        update using the bounded next_s/s ratio, never division at s=0. All
        recorded observations are clamped exactly throughout sampling.
        """
        self.validate(batch, embodiment)
        if self.objective == "direct_action":
            return self(batch, embodiment, _validated=True)
        current = LinearDDIMImageSampler(steps)(
            self, batch, embodiment, generator=generator, validated=True
        )
        if self.objective == "visual_only":
            return {"image": current}
        readout = self(
            batch,
            embodiment,
            current,
            torch.zeros_like(batch["image"]["time"], dtype=torch.float32),
            phase="B",
            _validated=True,
        )
        pose = readout["proprio"]
        pose = torch.where(
            batch["proprio"]["observed"][..., None], batch["proprio"]["value"], pose
        )
        result = {
            "image": current,
            "proprio": pose.masked_fill(~batch["proprio"]["valid"][..., None], 0),
        }
        if self.objective == "visual_state_action":
            result["action"] = readout["action"]
        return result
