"""RDT pretrained stems extracted from EgoVerse dd87495a.

Class/function bodies are unchanged from models/hpt_nets.py. Only the imports
are localized to the current graph repository's PolicyStem role home.
"""

import json
import os
from collections import OrderedDict
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import nn

from egomimic.models.stems.hpt_stems import PolicyStem


class PretrainedWeights:
    """Mixin marking a module whose weights come from a pretrained checkpoint.

    Subclasses list the attribute names holding the pretrained submodules in
    ``_hpt_pretrained_attrs``. Two things follow:

    * ``finalize_modules``'s init pass skips those subtrees entirely.
    * if the subclass can produce a reference state dict for the checkpoint
      (``pretrained_reference_state_dict``), the weights are hashed and
      compared with that reference once the model is fully built.
    """

    _hpt_pretrained = True
    _hpt_pretrained_attrs: tuple = ()

    def pretrained_submodules(self) -> List[nn.Module]:
        """The submodules whose weights must never be re-initialised."""
        mods = []
        for name in self._hpt_pretrained_attrs:
            mod = getattr(self, name, None)
            if isinstance(mod, nn.Module):
                mods.append(mod)
        return mods

    def pretrained_state_dict(self) -> dict:
        """Flat ``{"<attr>.<param>": tensor}`` view of the pretrained weights."""
        state = {}
        for name in self._hpt_pretrained_attrs:
            mod = getattr(self, name, None)
            if isinstance(mod, nn.Module):
                for key, value in mod.state_dict().items():
                    state[f"{name}.{key}"] = value
        return state

    def pretrained_reference_state_dict(self) -> Optional[dict]:
        """The checkpoint's own weights, read back independently of this model.

        ``None`` (the default) means "cannot be verified offline"; the checker
        then skips this module instead of failing, so only the stems that own
        an HF snapshot are hash-checked. ``ResNet`` is protected from the init
        pass but not verified: its torchvision weights are not a file this can
        re-read.
        """
        return None

    def pretrained_hash_dtype(self) -> Optional[torch.dtype]:
        """dtype both sides are cast to before hashing (``None`` = as stored)."""
        return None


def _snapshot_state_dict(
    snapshot_dir: str, prefix: str = "", dtype: Optional[torch.dtype] = None
) -> Optional[dict]:
    """Read a local HF snapshot's safetensors shards into a flat state dict.

    ``None`` if the snapshot has no safetensors (the caller then skips the
    check rather than hitting the network).
    """
    from safetensors.torch import load_file

    index = os.path.join(snapshot_dir, "model.safetensors.index.json")
    if os.path.isfile(index):
        with open(index, encoding="utf-8") as handle:
            shards = sorted(set(json.load(handle)["weight_map"].values()))
        files = [os.path.join(snapshot_dir, shard) for shard in shards]
    else:
        single = os.path.join(snapshot_dir, "model.safetensors")
        if not os.path.isfile(single):
            return None
        files = [single]

    state = {}
    for path in files:
        for key, tensor in load_file(path, device="cpu").items():
            state[f"{prefix}{key}"] = tensor if dtype is None else tensor.to(dtype)
    return state


def _local_snapshot_if_offline(model_name: str) -> str:
    """Resolve a HF repo id to its cached snapshot directory when the process
    runs offline (``HF_HUB_OFFLINE`` / ``TRANSFORMERS_OFFLINE``); a local path
    or an online process is returned unchanged."""
    if os.path.isdir(model_name):
        return model_name
    # huggingface_hub parses these with _is_true, i.e. 1 / ON / YES / TRUE in
    # any case, and reads them once at import; match the values but read the
    # live environment, since a launcher or a test may set them after import.
    from huggingface_hub.constants import ENV_VARS_TRUE_VALUES

    offline = any(
        (os.environ.get(var) or "").upper() in ENV_VARS_TRUE_VALUES
        for var in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    )
    if not offline:
        return model_name
    from huggingface_hub import snapshot_download

    return snapshot_download(model_name, local_files_only=True)


class DINOv3Stem(PretrainedWeights, PolicyStem):
    """Pretrained DINOv3 ViT (timm), a drop-in replacement for ``ResNet``:
    ``(B, T, N, 3, H, W)`` images in [0, 1] in, ``(B, patches, output_dim)`` out.
    The stem owns the resize and the checkpoint's mean / std, so the config
    must NOT normalize these images in its augs.

    Loaded through timm rather than transformers: the ``timm/*_dinov3`` hub
    repos are not gated (``facebook/dinov3-*`` is), and DINOv3 in transformers
    needs >= 4.56, above our pin.

    Args:
        model_name: timm model id, e.g. ``vit_base_patch16_dinov3.lvd1689m``
            (86M) or ``vit_large_patch16_dinov3.lvd1689m`` (300M).
        output_dim: width of the projected tokens.
        freeze_backbone: freeze the tower, keep it in eval mode and run it
            under ``no_grad``.
        image_size: ``S`` or ``[H, W]``, multiples of the patch size. DINOv3 is
            RoPE-based, so non-square inputs need no position interpolation.
        pretrained: ``False`` builds a randomly initialised tower (tests).
        tower_kwargs: timm architecture overrides (tests).
        pool: ``"patch"`` (every patch token) or ``"cls"`` (one token per image).
        grad_checkpointing: recompute the tower's blocks in backward, for a
            fine-tuned tower over many frames.
    """

    DEFAULT_MODEL = "vit_base_patch16_dinov3.lvd1689m"
    _hpt_pretrained_attrs = ("tower",)

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        output_dim: int = 10,
        freeze_backbone: bool = True,
        image_size=256,
        pretrained: bool = True,
        tower_kwargs: Optional[dict] = None,
        pool: str = "patch",
        grad_checkpointing: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        import timm

        if pool not in ("patch", "cls"):
            raise ValueError(f"DINOv3Stem pool must be 'patch' or 'cls', got {pool!r}")
        self.pool = pool

        self.model_name = model_name
        self.pretrained = pretrained
        self.tower = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0, **dict(tower_kwargs or {})
        )
        self.tower.set_grad_checkpointing(grad_checkpointing)
        self.hidden_size = int(self.tower.embed_dim)
        self.num_prefix_tokens = int(self.tower.num_prefix_tokens)
        size = [image_size] * 2 if isinstance(image_size, int) else list(image_size)
        patch = self.tower.patch_embed.patch_size[0]
        if any(s % patch for s in size):
            raise ValueError(
                f"DINOv3Stem image_size {size} is not a multiple of {patch}"
            )
        self.image_size = tuple(int(s) for s in size)
        self.grid_size = tuple(s // patch for s in self.image_size)
        cfg = self.tower.pretrained_cfg
        self.register_buffer(
            "image_mean", torch.tensor(cfg["mean"]).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "image_std", torch.tensor(cfg["std"]).view(1, 3, 1, 1), persistent=False
        )

        self.out_dim = output_dim
        self.proj = nn.Linear(self.hidden_size, output_dim)

        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for param in self.tower.parameters():
                param.requires_grad = False

    def train(self, mode: bool = True):
        """Keep a frozen tower in eval mode regardless of the outer flag."""
        super().train(mode)
        if self.freeze_backbone:
            self.tower.eval()
        return self

    def backbone_parameters(self):
        """Pretrained-tower parameters (``proj`` stays in the main group)."""
        return list(self.tower.parameters())

    def pretrained_reference_state_dict(self) -> Optional[dict]:
        """The cached timm checkpoint, keyed ``tower.*``. ``None`` for a random
        tower, or when the file is not in the local HF cache."""
        if not self.pretrained:
            return None
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        try:
            path = hf_hub_download(
                self.tower.pretrained_cfg["hf_hub_id"],
                "model.safetensors",
                local_files_only=True,
            )
        except Exception:  # not cached, or not an HF-hosted timm model
            return None
        return {f"tower.{k}": v for k, v in load_file(path).items()}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, *_, H, W = x.shape
        x = x.reshape(-1, 3, H, W)
        x = F.interpolate(
            x,
            size=self.image_size,
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        x = (x - self.image_mean) / self.image_std
        with torch.set_grad_enabled(
            not self.freeze_backbone and torch.is_grad_enabled()
        ):
            feat = self.tower.forward_features(x)
        if self.pool == "cls":
            feat = feat[:, :1]
        else:
            feat = feat[:, self.num_prefix_tokens :]
        return self.proj(feat.reshape(B, -1, self.hidden_size))


class _Qwen3BaseEncoder(PretrainedWeights, PolicyStem):
    """Shared base for Qwen3-Embedding stems used by HPT.

    Owns the tokenizer + transformer encoder so HPT.process_batch_for_training
    only needs to pass a list of raw prompt strings. Subclasses pick whether
    the feature passed to cross-attention is a pooled (B, 1, D) summary or the
    full per-token (B, L, D) sequence.

    Args:
        model_name: HF identifier for the Qwen3-Embedding checkpoint.
        max_length: tokenizer truncation length.
        freeze: if True (default), freeze the encoder weights and run in
            eval mode under no_grad. If False, the encoder is trainable.
        dtype: weight dtype for the HF model (fp16 by default to keep VRAM
            cost down when frozen).
        normalize_pooled: only used by the pooled subclass; L2-normalizes the
            sentence embedding (Qwen3 official recipe).
        cache_text_features: cache the frozen per-token hidden states per
            unique prompt string (default True). Ignored when ``freeze`` is
            False - a trainable encoder must stay online.
        text_cache_max_entries: cache capacity in ENTRIES; LRU eviction beyond
            it. An entry is the prompt's un-padded (L, D) hidden states, so
            its size follows the prompt: 4-25 KB for a 2-12 token prompt at
            1024-dim fp16, up to max_length * D * 2 bytes (262 KB at
            max_length 128). The 8192 default is therefore ~100-200 MB of
            device memory on real annotations and 2 GB in the worst case.
    """

    DEFAULT_MODEL = "Qwen/Qwen3-Embedding-0.6B"
    _hpt_pretrained_attrs = ("encoder",)

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        max_length: int = 128,
        freeze: bool = True,
        dtype: str = "float16",
        output_dim: int | None = None,
        cache_text_features: bool = True,
        text_cache_max_entries: int = 8192,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        from transformers import AutoModel, AutoTokenizer

        self.model_name = model_name
        self.max_length = max_length
        self.freeze_encoder = freeze
        torch_dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        # transformers >= 4.57 probes the Hub while loading a tokenizer by repo
        # id even under HF_HUB_OFFLINE=1 (tokenization_utils_base
        # ._patch_mistral_regex -> model_info), which raises on compute nodes
        # with no internet. Loading from the cached snapshot DIRECTORY skips
        # that probe, so resolve the repo id to its local snapshot when offline.
        load_path = _local_snapshot_if_offline(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(load_path, padding_side="left")
        self.encoder = AutoModel.from_pretrained(load_path, torch_dtype=torch_dtype)
        # Remembered so the weight-hash check can read the same snapshot back
        # and compare in the dtype the encoder was loaded in.
        self._snapshot_dir = load_path
        self._load_dtype = torch_dtype
        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad = False
            self.encoder.eval()
        self.hidden_size = int(self.encoder.config.hidden_size)
        # Project Qwen's hidden_size (typ. 1024) down to the cross-attn
        # modality_embed_dim. Must match the trunk's embed_dim so the post-stem
        # token tensors concat with other modalities' tokens in
        # ``HPTModel.preprocess_tokens``.
        self.output_dim = output_dim if output_dim is not None else self.hidden_size
        self.proj = (
            nn.Linear(self.hidden_size, self.output_dim)
            if self.output_dim != self.hidden_size
            else nn.Identity()
        )
        self._init_text_feature_cache(cache_text_features, text_cache_max_entries)

    # ----------------------------------------------------------------
    # frozen text-feature cache
    # ----------------------------------------------------------------

    def _init_text_feature_cache(
        self, cache_text_features: bool = True, text_cache_max_entries: int = 8192
    ) -> None:
        """Set up the (empty) prompt -> per-token-features cache."""
        self.cache_text_features = bool(cache_text_features)
        self.text_cache_max_entries = int(text_cache_max_entries)
        self._text_feature_cache = OrderedDict()
        self._text_cache_hits = 0
        self._text_cache_misses = 0

    def clear_text_feature_cache(self) -> None:
        """Drop every cached prompt (the hit/miss counters are cumulative)."""
        self._text_feature_cache.clear()

    def text_feature_cache_stats(self) -> dict:
        """``{"entries": n, "hits": h, "misses": m}``; hits/misses count prompt
        occurrences, duplicates inside one batch included."""
        return {
            "entries": len(self._text_feature_cache),
            "hits": self._text_cache_hits,
            "misses": self._text_cache_misses,
        }

    def _text_cache_enabled(self) -> bool:
        # A trainable encoder must stay online: its outputs change every step
        # and the loss needs the graph. Only a frozen encoder is cacheable.
        return bool(
            self.cache_text_features
            and self.freeze_encoder
            and self.text_cache_max_entries > 0
        )

    def _encode_online(self, prompts):
        """Tokenize + run the encoder, returning ``(last_hidden_state,
        attention_mask)`` left-padded (the tokenizer is ``padding_side="left"``)."""
        tokens = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)
        if self.freeze_encoder:
            with torch.no_grad():
                out = self.encoder(**tokens)
        else:
            out = self.encoder(**tokens)
        return out.last_hidden_state, tokens["attention_mask"]

    def _encode(self, prompts):
        """``(last_hidden_state, attention_mask)`` for ``prompts``, served from
        the per-prompt cache whenever the encoder is frozen.

        The annotation vocabulary is tiny next to the number of training steps,
        so a frozen encoder recomputes the same 600M-parameter forward over and
        over. Misses go through ``_encode_online`` in ONE sub-batch of unique
        strings; each prompt's un-padded ``(L_i, D)`` rows are stored (the mask
        is implied by ``L_i``) and the batch is re-assembled with the same LEFT
        padding the tokenizer produces, so callers cannot tell the paths apart.

        One deliberate difference: padded positions come back as ZEROS here,
        whereas the online path leaves the encoder's output for the pad tokens
        there. Every consumer masks those positions out
        (``forward_with_mask``; last-token pooling reads the final, always-real
        position under left padding), so this only removes junk.
        """
        if not self._text_cache_enabled():
            return self._encode_online(prompts)

        prompts = list(prompts)
        device = self.device
        misses = []
        for prompt in prompts:
            entry = self._text_feature_cache.get(prompt)
            # Entries computed on another device are stale (the module moved):
            # drop them rather than silently copy across devices.
            if entry is not None and entry.device != device:
                del self._text_feature_cache[prompt]
                entry = None
            if entry is None:
                self._text_cache_misses += 1
                if prompt not in misses:
                    misses.append(prompt)
            else:
                self._text_cache_hits += 1
                self._text_feature_cache.move_to_end(prompt)

        fresh = {}
        if misses:
            hidden, mask = self._encode_online(misses)
            mask = mask.bool()
            for i, prompt in enumerate(misses):
                # un-padded per-token states; detached, no graph is kept
                fresh[prompt] = hidden[i][mask[i]].detach()
                self._text_feature_cache[prompt] = fresh[prompt]
                self._text_feature_cache.move_to_end(prompt)
            while len(self._text_feature_cache) > self.text_cache_max_entries:
                self._text_feature_cache.popitem(last=False)  # LRU victim

        # Read this batch's rows from what was just encoded first: a batch with
        # more unique prompts than text_cache_max_entries evicts its own
        # earliest entries in the trim above, and a KeyError here would be a
        # crash rather than a slow path.
        rows = [
            fresh[p] if p in fresh else self._text_feature_cache[p] for p in prompts
        ]
        longest = max(row.shape[0] for row in rows)
        hidden = rows[0].new_zeros((len(rows), longest, rows[0].shape[-1]))
        mask = torch.zeros((len(rows), longest), dtype=torch.long, device=device)
        for i, row in enumerate(rows):  # LEFT padding, as padding_side="left"
            length = row.shape[0]
            hidden[i, longest - length :] = row
            mask[i, longest - length :] = 1
        return hidden, mask

    def _apply(self, *args, **kwargs):
        """``.to()`` / ``.cuda()`` / ``.float()`` invalidate the cache: the
        stored tensors belong to the old device / dtype."""
        out = super()._apply(*args, **kwargs)
        if getattr(self, "_text_feature_cache", None):
            self.clear_text_feature_cache()
        return out

    def pretrained_reference_state_dict(self) -> Optional[dict]:
        """The snapshot's own safetensors, cast exactly the way
        ``from_pretrained(torch_dtype=...)`` casts them."""
        if not os.path.isdir(self._snapshot_dir):
            return None
        return _snapshot_state_dict(
            self._snapshot_dir, prefix="encoder.", dtype=self._load_dtype
        )

    def pretrained_hash_dtype(self) -> Optional[torch.dtype]:
        """Hash in the load dtype. ``HPT.__init__`` upcasts the whole model with
        ``nets.float()``; fp16 -> fp32 -> fp16 is exact, so casting the live
        weights back down compares like for like without a second model load."""
        return self._load_dtype

    def train(self, mode: bool = True):
        """Keep the frozen encoder in eval mode regardless of outer train flag."""
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()
        return self


class QwenPerTokenEncoder(_Qwen3BaseEncoder):
    """Qwen3-Embedding stem returning per-token hidden states (B, L, hidden_size).

    Padding positions are zeroed before cross-attention so they don't pull
    information from the learnable stem tokens.
    """

    def forward_with_mask(self, prompts):
        """Returns ``(features, attention_mask)``; the mask is what padded
        positions must be excluded by, since zeroing them before the biased
        ``proj`` turns them into the learned constant ``proj.bias``."""
        hidden, mask = self._encode(prompts)
        feat = hidden.float() * mask.unsqueeze(-1).float()
        return self.proj(feat), mask.bool()  # (B, L, output_dim), (B, L)

    def forward(self, prompts):
        return self.forward_with_mask(prompts)[0]

    def compute_latent(self, prompts):
        feat, mask = self.forward_with_mask(prompts)  # (B, L, output_dim), (B, L)
        stem_tokens = self.tokens.repeat(feat.shape[0], 1, 1)
        return self.cross_attention(stem_tokens, feat, mask=mask)
