"""EgoVerse RDT (Robotics Diffusion Transformer) on the graph runtime.

Ported from EgoVerse ``egomimic.algo.rdt.RDTModel`` (GaTech-RL2/EgoVerse
dd87495a). RDT is HPT with the policy network swapped: no Perceiver pooling.
Stems hand their full token sequences to a shared DiT (``RDTBackbone``) that
cross-attends language and image tokens in alternating blocks.

``RDTConditionStage`` is ``RDTModel.forward_features``: it encodes every
camera, the proprioception and the prompt into one ``RDTConditions``.
``RDTFlowDenoiserStage`` is the shared Euler flow sampler run through the
embodiment's ``RDTDenoiser``. ``rdt_nets``, the DINOv3/Qwen stems and the
per-sample image augmentations are copied from EgoVerse unchanged.

Not ported: multi-embodiment heads and RDT's long-range image memory
(``trunk.memory`` / ``short_dropout``); EgoVerse's reference configs leave both
off.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
import torch.nn.functional as F
from torch import nn

from egomimic.models.image_augs import PerSampleAugs
from egomimic.models.rdt_nets import (
    RDTBackbone,
    RDTConditions,
    RDTDenoiser,
    mlp_gelu,
    sincos_1d,
    sincos_2d,
)
from egomimic.models.stems.rdt_encoders import PretrainedWeights
from egomimic.pipeline.core import Stage
from egomimic.pipeline.stages_flow import FlowDenoiserStage
from egomimic.pipeline.stages_hpt import _as_stem_dtype, _module_name

INIT_STD = 0.02


def _init_weights(module):
    """HPT's init pass (xavier Linear, unit LayerNorm), as EgoVerse applies it."""
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.bias, 0)
        nn.init.constant_(module.weight, 1.0)


def _apply_skipping_pretrained(root: nn.Module, fn) -> None:
    """``root.apply(fn)`` that never enters a pretrained encoder."""
    if isinstance(root, PretrainedWeights):
        return
    for child in root.children():
        _apply_skipping_pretrained(child, fn)
    fn(root)


class RDTConditionStage(Stage):
    """Encode cameras, proprioception and prompt into ``RDTConditions``.

    Images are float RGB in [0, 1], ``B,3,H,W`` or ``B,T,3,H,W`` with
    ``T == image_history`` (current frame last). ``image_masks`` optionally
    maps a camera key to a ``B,T`` real-frame mask. Proprioception arrives
    normalized, ``B,D`` or ``B,K,D``; each step becomes one state token.

    Without ``language_stem`` every DiT block cross-attends the image tokens.
    ``cond_drop`` is RDT's per-stream ``cond_mask_prob`` (training only): a
    dropped state becomes a learned null token, a dropped prompt becomes
    ``""``, and ``img`` drops whole cameras without ever blinding a sample.
    ``lang_pad_to`` right-pads the masked language tokens to a multiple, so
    ``torch.compile`` sees fixed lengths; masked keys get zero weight.
    """

    writes = ("condition",)

    def __init__(
        self,
        backbone: RDTBackbone,
        image_encoders: Mapping[str, nn.Module],
        image_stems: Mapping[str, nn.Module],
        state_stems: Mapping[str, nn.Module],
        language_stem: nn.Module | None = None,
        prompt_key: str = "observations.annotation",
        image_masks: Mapping[str, str] | None = None,
        cond_drop: Mapping[str, float] | None = None,
        image_history: int = 1,
        image_history_dropout: float = 0.0,
        image_history_null_token: bool = True,
        lang_pad_to: int | None = None,
        train_image_augs=None,
        eval_image_augs=None,
        control_frequency_hz: float = 30.0,
        frequency_key: str | None = None,
    ):
        super().__init__()
        self.camera_keys = tuple(str(key) for key in image_encoders)
        self.state_keys = tuple(str(key) for key in state_stems)
        if not self.camera_keys:
            raise ValueError("RDT needs at least one camera")
        if set(image_stems) != set(self.camera_keys):
            raise ValueError(
                "image_encoders and image_stems must name the same cameras"
            )
        if language_stem is not None and not hasattr(
            language_stem, "forward_with_mask"
        ):
            raise TypeError("RDT language stem must expose forward_with_mask")
        self.hidden_dim = int(backbone.hidden_dim)
        if self.hidden_dim % 4:
            raise ValueError("RDT sin-cos positions need a width divisible by 4")
        self.backbone = backbone
        self.encoders = nn.ModuleDict(
            {_module_name(k): image_encoders[k] for k in self.camera_keys}
        )
        self.image_stems = nn.ModuleDict(
            {_module_name(k): image_stems[k] for k in self.camera_keys}
        )
        self.state_stems = nn.ModuleDict(
            {_module_name(k): state_stems[k] for k in self.state_keys}
        )
        self.language_stem = language_stem
        self.prompt_key = str(prompt_key)
        self.image_masks = {str(k): str(v) for k, v in dict(image_masks or {}).items()}
        if not set(self.image_masks) <= set(self.camera_keys):
            raise ValueError("image_masks names a camera that is not configured")
        self.cond_drop = {
            "state": 0.0,
            "lang": 0.0,
            "img": 0.0,
            **dict(cond_drop or {}),
        }
        if set(self.cond_drop) != {"state", "lang", "img"} or not all(
            0.0 <= float(p) <= 1.0 for p in self.cond_drop.values()
        ):
            raise ValueError("cond_drop takes state/lang/img probabilities in [0, 1]")
        self.image_history = int(image_history)
        self.image_history_dropout = float(image_history_dropout)
        self.image_history_null_token = (
            bool(image_history_null_token) and self.image_history > 1
        )
        self.lang_pad_to = None if lang_pad_to is None else int(lang_pad_to)
        self.train_image_augs = (
            None if train_image_augs is None else PerSampleAugs(train_image_augs)
        )
        self.eval_image_augs = (
            None if eval_image_augs is None else PerSampleAugs(eval_image_augs)
        )
        self.control_frequency_hz = float(control_frequency_hz)
        if not (
            math.isfinite(self.control_frequency_hz) and self.control_frequency_hz > 0
        ):
            raise ValueError("control_frequency_hz must be finite and positive")
        self.frequency_key = None if frequency_key is None else str(frequency_key)

        D = self.hidden_dim
        # RDT's mlp2x_gelu language adaptor; image / state adaptors are the stems.
        self.lang_adaptor = None if language_stem is None else mlp_gelu(D, D, 2)
        for module in (
            *self.image_stems.values(),
            *self.state_stems.values(),
            self.language_stem,
            self.lang_adaptor,
        ):
            if module is not None:
                _apply_skipping_pretrained(module, _init_weights)
        self.camera_embed = nn.ParameterDict(
            {
                _module_name(k): nn.Parameter(torch.zeros(1, 1, 1, D))
                for k in self.camera_keys
            }
        )
        self.frame_embed = nn.Parameter(torch.zeros(1, self.image_history, 1, D))
        # Only what can be used: an unused parameter crashes plain DDP.
        if self.cond_drop["state"] > 0 and self.state_keys:
            self.null_state = nn.Parameter(torch.randn(1, 1, D) * INIT_STD)
        if self.cond_drop["img"] > 0 and len(self.camera_keys) > 1:
            self.null_img = nn.Parameter(torch.randn(1, 1, D) * INIT_STD)
        if self.image_history_null_token:
            self.null_frame = nn.Parameter(torch.randn(D) * INIT_STD)

        self.reads = (
            *self.camera_keys,
            *self.state_keys,
            *((self.prompt_key,) if language_stem is not None else ()),
            *self.image_masks.values(),
            *((self.frequency_key,) if self.frequency_key else ()),
        )

    def _drop(self, prob: float, batch_size: int, device):
        """(B,) bool: which samples lose a condition; None outside training."""
        if not (self.training and prob > 0):
            return None
        return torch.rand(batch_size, device=device) < prob

    def _augment(self, clip):
        """(B, T, 3, H, W): one aug draw per sample, absent (all-zero) frames stay zero."""
        augs = self.train_image_augs if self.training else self.eval_image_augs
        if augs is None:
            return clip
        n, t = clip.shape[:2]
        flat = clip.flatten(0, 1)
        absent = (flat == 0).flatten(1).all(1)
        out = torch.where(absent.view(-1, 1, 1, 1), 0.0, augs(flat, frames=t))
        return out.reshape(clip.shape)

    def _image_tokens(self, key, x, frame_mask=None):
        """(B, T, 3, H, W) frames, current last -> (B, T * patches, D)."""
        if x.ndim == 4:
            x = x[:, None]
        B, T = x.shape[:2]
        if x.ndim != 5 or x.shape[2] != 3 or T != self.image_history:
            raise ValueError(
                f"'{key}' must be B,{self.image_history},3,H,W (image_history), got {tuple(x.shape)}"
            )
        if not x.is_floating_point():
            raise TypeError(f"'{key}' must be float RGB in [0, 1], not integer pixels")
        x = self._augment(x)
        drop = self._drop(self.image_history_dropout, B, x.device) if T > 1 else None
        if drop is not None and not self.image_history_null_token:
            x = torch.where(drop.view(B, 1, 1, 1, 1), x[:, -1:].expand_as(x), x)
        name = _module_name(key)
        encoder, stem = self.encoders[name], self.image_stems[name]
        tokens = stem(encoder(_as_stem_dtype(x, encoder)))
        tokens = tokens.reshape(B, T, -1, tokens.shape[-1])
        if self.image_history_null_token:
            null = torch.zeros(B, T, dtype=torch.bool, device=x.device)
            if frame_mask is not None:
                null = frame_mask.reshape(B, T).to(x.device) < 0.5
            if drop is not None:
                null = null | drop[:, None]
            null[:, -1] = False
            # torch.where even when nothing is nulled keeps the token in the
            # graph, so plain DDP never sees an unused parameter.
            tokens = torch.where(
                null[:, :, None, None], self.null_frame.to(tokens), tokens
            )
        # DINOv3 is RoPE-only: its tokens carry no absolute position.
        grid = getattr(encoder, "grid_size", None)
        if grid is not None and grid[0] * grid[1] == tokens.shape[2]:
            pos = sincos_2d(tokens.shape[-1], grid)
        else:
            pos = torch.from_numpy(sincos_1d(tokens.shape[-1], range(tokens.shape[2])))
        tokens = tokens + pos.to(tokens) + self.frame_embed + self.camera_embed[name]
        return tokens.reshape(B, -1, tokens.shape[-1])

    def _drop_cameras(self, img: list) -> list:
        """Per-camera dropout that never blinds a sample completely."""
        if len(img) < 2 or not (self.training and self.cond_drop["img"] > 0):
            return img
        B, device = len(img[0]), img[0].device
        drop = torch.rand(B, len(img), device=device) < self.cond_drop["img"]
        keep = torch.randint(len(img), (B,), device=device)
        blind = drop.all(dim=1)
        drop[blind, keep[blind]] = False
        return [
            torch.where(drop[:, i, None, None], self.null_img.to(tokens), tokens)
            for i, tokens in enumerate(img)
        ]

    def forward(self, batch):
        img = [
            self._image_tokens(
                key,
                batch[key],
                batch[self.image_masks[key]] if key in self.image_masks else None,
            )
            for key in self.camera_keys
        ]
        img = torch.cat(self._drop_cameras(img), dim=1)
        B = len(img)
        if self.frequency_key:
            freq = torch.as_tensor(
                batch[self.frequency_key], device=img.device, dtype=img.dtype
            )
            freq = freq.reshape(-1).expand(B)
        else:
            freq = img.new_full((B,), self.control_frequency_hz)
        cond = RDTConditions(img=img, freq=freq, backbone=self.backbone)

        if self.state_keys:
            state = []
            for key in self.state_keys:
                stem = self.state_stems[_module_name(key)]
                tokens = stem(_as_stem_dtype(batch[key], stem))
                state.append(tokens.reshape(len(tokens), -1, tokens.shape[-1]))
            cond.state = torch.cat(state, dim=1)
            drop = self._drop(self.cond_drop["state"], B, img.device)
            if drop is not None:
                cond.state = torch.where(
                    drop[:, None, None], self.null_state.to(cond.state), cond.state
                )

        if self.language_stem is None:
            batch["condition"] = cond
            return batch
        prompts = batch[self.prompt_key]
        if isinstance(prompts, str) or len(prompts) != B:
            raise ValueError(f"'{self.prompt_key}' must hold one prompt per sample")
        drop = self._drop(self.cond_drop["lang"], B, img.device)
        if drop is not None:
            prompts = ["" if d else p for p, d in zip(prompts, drop.tolist())]
        tokens, mask = self.language_stem.forward_with_mask(list(prompts))
        if not mask.any(dim=1).all():
            raise ValueError(f"'{self.prompt_key}': a prompt encoded to zero tokens")
        cond.lang = self.lang_adaptor(_as_stem_dtype(tokens, self.lang_adaptor))
        cond.lang_mask = mask.bool()
        if self.lang_pad_to:
            pad = -cond.lang.shape[1] % self.lang_pad_to
            cond.lang = F.pad(cond.lang, (0, 0, 0, pad))
            cond.lang_mask = F.pad(cond.lang_mask, (0, pad), value=False)
        batch["condition"] = cond
        return batch


class RDTFlowDenoiserStage(FlowDenoiserStage):
    """The shared Euler flow sampler with structured ``RDTConditions``.

    Only condition validation differs from ``FlowDenoiserStage``: RDT's
    condition is a dataclass, not one tensor. Training and sampling call the
    EgoVerse ``RDTDenoiser`` unchanged.
    """

    def __init__(self, model: RDTDenoiser, **kwargs):
        super().__init__(model=model, **kwargs)
        if not isinstance(model, RDTDenoiser):
            raise TypeError("RDTFlowDenoiserStage needs an RDTDenoiser")
        if model.act_seq != self.action_horizon:
            raise ValueError("RDTDenoiser act_seq must equal action_horizon")

    def _validate_condition(self, condition):
        if not isinstance(condition, RDTConditions):
            raise TypeError("RDTFlowDenoiserStage needs RDTConditions")
        if condition.img.shape[-1] != self.condition_input_dim:
            raise ValueError("RDT condition width does not match condition_input_dim")
