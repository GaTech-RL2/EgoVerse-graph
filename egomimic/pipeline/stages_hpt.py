"""HPT input-side stages: per-modality stems feeding a shared trunk.

Upstream HPT is one monolithic ``Algo`` that owns stems, trunk and head and
walks them internally. Decomposing it into stages is what makes it a first
class citizen here: each piece declares what it reads and writes, so
``tools/config_graph.py`` can lint the boundaries and a config's stage list
shows the actual dataflow instead of a single opaque node.

The split follows HPT's own structure:

    observations --HPTStemStage--> hpt/tokens --HPTTrunkStage--> condition

``condition`` is deliberately the same key the DP path writes, so any head
already in the graph -- the diffusion stages, the flow stages, the planar
sampler -- consumes an HPT representation without changing a line.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn as nn

from egomimic.pipeline.core import Stage, resolve_homogeneous_scalar


def _as_stem_dtype(value: torch.Tensor, stem: nn.Module) -> torch.Tensor:
    """Cast a floating input to the stem's own parameter dtype.

    Zarr stamps poses as float64 and ``torch.from_numpy`` preserves that, so a
    Double tensor can reach a stem's Linear unchanged -- autocast only handles
    float32/float16/bfloat16, never Double, and the matmul then raises
    "mat1 and mat2 must have the same dtype".

    A stage should not assume what dtype the loader or the runner handed it, so
    the cast happens here rather than relying on an upstream normalisation.
    Integer inputs are left alone; float32 in is a no-op.
    """
    if not torch.is_tensor(value) or not value.is_floating_point():
        return value
    try:
        target = next(stem.parameters()).dtype
    except StopIteration:  # a parameterless stem has nothing to match
        return value
    return value if value.dtype == target else value.to(target)


def _module_name(batch_key: str) -> str:
    """Sanitize a batch key for use as an ``nn.ModuleDict`` key.

    Batch keys are dotted paths ("observations.state.ee_pose") and ModuleDict
    rejects "." because it would break ``state_dict`` path resolution. The
    mapping is kept explicit so checkpoint keys stay readable.
    """
    return batch_key.replace(".", "__")


def _cross_attn_spec(stem: nn.Module):
    """Return a stem's ``specs.cross_attn`` block, or None if it has none.

    Hydra hands back a DictConfig (attribute access) but a hand-built stem may
    carry a plain dict, so accept either rather than failing on the one the
    caller happened to use.
    """
    specs = getattr(stem, "specs", None)
    if specs is None:
        return None
    cross_attn = (
        specs.get("cross_attn")
        if isinstance(specs, Mapping)
        else getattr(specs, "cross_attn", None)
    )
    if cross_attn is None:
        return None
    if isinstance(cross_attn, Mapping):
        from omegaconf import OmegaConf

        # init_cross_attn uses attribute access.
        return OmegaConf.create(dict(cross_attn))
    return cross_attn


class HPTStemStage(Stage):
    """Encode each observation modality into a fixed set of latent tokens.

    ``stems`` maps a batch key to a :class:`PolicyStem`. Every stem compresses
    its modality to ``crossattn_latent`` tokens of ``modality_embed_dim`` via
    cross-attention, and the stage concatenates them along the token axis. That
    is what lets modalities of wildly different shapes -- a 14-D pose and a
    3x480x640 image -- meet in one sequence the trunk can attend over.

    Stems are applied in sorted key order so the token layout is deterministic
    across processes; DDP would otherwise diverge on dict ordering.

    ``domain_stems`` optionally adds per-embodiment stems on top of the shared
    set, mirroring HPT's ``stem_specs`` / ``shared_stem_specs`` split. The
    branch is chosen from ``selector_key`` in the batch, the same mechanism
    ``KeyedFeatureProjection`` uses -- the runner itself still treats sources as
    opaque.
    """

    writes = ("hpt/tokens",)

    def __init__(
        self,
        stems: Mapping[str, nn.Module],
        domain_stems: Mapping[str, Mapping[str, nn.Module]] | None = None,
        selector_key: str = "embodiment",
        selector_aliases: Mapping | None = None,
    ):
        super().__init__()
        shared = {str(key): stem for key, stem in dict(stems or {}).items()}
        if not shared and not domain_stems:
            raise ValueError("HPTStemStage needs at least one stem")
        self.shared_keys = tuple(sorted(shared))
        self.stems = nn.ModuleDict(
            {_module_name(key): stem for key, stem in shared.items()}
        )

        self.selector_key = str(selector_key)
        self.selector_aliases = {
            str(key): str(value) for key, value in dict(selector_aliases or {}).items()
        }
        self.domain_keys: dict[str, tuple[str, ...]] = {}
        domain_modules = {}
        for domain, per_domain in dict(domain_stems or {}).items():
            entries = {str(key): stem for key, stem in dict(per_domain or {}).items()}
            if not entries:
                continue
            self.domain_keys[str(domain)] = tuple(sorted(entries))
            domain_modules[_module_name(str(domain))] = nn.ModuleDict(
                {_module_name(key): stem for key, stem in entries.items()}
            )
        self.domain_stems = nn.ModuleDict(domain_modules)

        for stem in list(self.stems.values()) + [
            inner for module in self.domain_stems.values() for inner in module.values()
        ]:
            spec = _cross_attn_spec(stem)
            if spec is not None and hasattr(stem, "init_cross_attn"):
                stem.init_cross_attn(spec)

        reads = list(self.shared_keys)
        for keys in self.domain_keys.values():
            reads.extend(keys)
        if self.domain_stems:
            reads.append(self.selector_key)
        self.reads = tuple(dict.fromkeys(reads))

    def _domain(self, batch: dict) -> str | None:
        if not self.domain_stems:
            return None
        raw = resolve_homogeneous_scalar(
            batch[self.selector_key], label=self.selector_key
        )
        name = self.selector_aliases.get(str(raw), str(raw))
        if _module_name(name) not in self.domain_stems:
            raise KeyError(
                f"HPTStemStage has no stems for domain {name!r}; "
                f"configured: {sorted(self.domain_keys)}"
            )
        return name

    def forward(self, batch: dict) -> dict:
        domain = self._domain(batch)
        tokens = []
        for key in self.shared_keys:
            stem = self.stems[_module_name(key)]
            tokens.append(stem.compute_latent(_as_stem_dtype(batch[key], stem)))
        if domain is not None:
            module = self.domain_stems[_module_name(domain)]
            for key in self.domain_keys[domain]:
                stem = module[_module_name(key)]
                tokens.append(stem.compute_latent(_as_stem_dtype(batch[key], stem)))
        if not tokens:
            raise RuntimeError("HPTStemStage produced no tokens")
        widths = {int(t.shape[-1]) for t in tokens}
        if len(widths) != 1:
            raise ValueError(
                f"HPTStemStage stems disagree on embed dim: {sorted(widths)}. "
                "Every stem's modality_embed_dim must match the trunk's."
            )
        batch["hpt/tokens"] = torch.cat(tokens, dim=1)
        return batch


class HPTTrunkStage(Stage):
    """Attend over the stem tokens and pool them into one conditioning vector.

    Writes ``condition``, the key the DP and flow heads already read, so an
    HPT representation drops into any head the graph has without touching it.

    ``token_postprocessing`` decides how the token sequence collapses:

      "action_token"  prepend a learned token and read it back out, the way a
                      CLS token works. This is HPT's default.
      "mean"          average over tokens.
      "last"          take the final token.

    A learned position embedding is added before the trunk when
    ``use_position_embedding`` is set; ``use_domain_embedding`` adds a
    per-domain learned offset so a cotrained trunk can tell its sources apart.
    The domain is resolved from the batch, never from the runner.
    """

    reads = ("hpt/tokens",)
    writes = ("condition",)

    _POOLING = ("action_token", "mean", "last")

    def __init__(
        self,
        trunk: nn.Module,
        embed_dim: int,
        token_postprocessing: str = "action_token",
        use_position_embedding: bool = True,
        max_tokens: int = 512,
        domains: tuple[str, ...] | list[str] | None = None,
        use_domain_embedding: bool = False,
        selector_key: str = "embodiment",
        selector_aliases: Mapping | None = None,
    ):
        super().__init__()
        if token_postprocessing not in self._POOLING:
            raise ValueError(
                f"token_postprocessing must be one of {self._POOLING}, "
                f"got {token_postprocessing!r}"
            )
        self.trunk = trunk
        self.embed_dim = int(embed_dim)
        self.token_postprocessing = token_postprocessing
        self.max_tokens = int(max_tokens)

        self.action_token = (
            nn.Parameter(torch.randn(1, 1, self.embed_dim) * 0.02)
            if token_postprocessing == "action_token"
            else None
        )

        if use_position_embedding:
            from egomimic.models.cores.hpt_utils import get_sinusoid_encoding_table

            # Registered as a buffer: it is fixed, but it must follow the
            # module across .to(device) and land in the checkpoint.
            self.register_buffer(
                "position_embedding",
                get_sinusoid_encoding_table(0, self.max_tokens, self.embed_dim),
                persistent=False,
            )
        else:
            self.position_embedding = None

        self.domain_list = [str(d) for d in (domains or [])]
        self.use_domain_embedding = bool(use_domain_embedding)
        if self.use_domain_embedding:
            if not self.domain_list:
                raise ValueError("use_domain_embedding needs a non-empty domains list")
            self.domain_embedding = nn.Parameter(
                torch.randn(len(self.domain_list), 1, self.embed_dim) * 0.02
            )
        else:
            self.domain_embedding = None

        self.selector_key = str(selector_key)
        self.selector_aliases = {
            str(key): str(value) for key, value in dict(selector_aliases or {}).items()
        }
        self.reads = (
            ("hpt/tokens", self.selector_key)
            if self.use_domain_embedding
            else ("hpt/tokens",)
        )

    def _add_domain_embedding(self, tokens: torch.Tensor, batch: dict) -> torch.Tensor:
        raw = resolve_homogeneous_scalar(
            batch[self.selector_key], label=self.selector_key
        )
        name = self.selector_aliases.get(str(raw), str(raw))
        if name not in self.domain_list:
            raise KeyError(
                f"HPTTrunkStage has no domain embedding for {name!r}; "
                f"configured: {self.domain_list}"
            )
        index = self.domain_list.index(name)
        return tokens + self.domain_embedding[index]

    def forward(self, batch: dict) -> dict:
        tokens = batch["hpt/tokens"]
        if tokens.ndim != 3:
            raise ValueError(
                f"HPTTrunkStage expects (B, tokens, {self.embed_dim}), "
                f"got {tuple(tokens.shape)}"
            )
        if int(tokens.shape[-1]) != self.embed_dim:
            raise ValueError(
                f"HPTTrunkStage embed_dim is {self.embed_dim} but the stems "
                f"produced width {int(tokens.shape[-1])}"
            )

        if self.domain_embedding is not None:
            tokens = self._add_domain_embedding(tokens, batch)
        if self.action_token is not None:
            tokens = torch.cat(
                (self.action_token.expand(len(tokens), -1, -1), tokens), dim=1
            )
        if self.position_embedding is not None:
            length = int(tokens.shape[1])
            if length > self.max_tokens:
                raise ValueError(
                    f"HPTTrunkStage got {length} tokens but max_tokens is "
                    f"{self.max_tokens}; raise max_tokens to fit the stems"
                )
            tokens = tokens + self.position_embedding[:, :length].to(tokens.dtype)

        out = self.trunk(tokens)
        # SimpleTransformer returns (tokens, per_block_outputs); a plain module
        # may return just the tokens.
        if isinstance(out, tuple):
            out = out[0]

        if self.token_postprocessing == "action_token":
            condition = out[:, 0]
        elif self.token_postprocessing == "mean":
            condition = out.mean(dim=1)
        else:
            condition = out[:, -1]
        batch["condition"] = condition
        return batch
