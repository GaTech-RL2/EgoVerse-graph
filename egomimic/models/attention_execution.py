"""Reusable sequence layout and explicit optional fused-attention kernels.

Layout construction is an eager boundary: inspect masks once, then reuse the
same gather indices and sequence boundaries across layers and sampling steps.
No backend or layout changes model parameters or permits cross-example keys.
"""

import importlib
import math
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable

import torch


@dataclass(frozen=True)
class AttentionLayout:
    kind: str
    batch_size: int
    query_tokens: int
    key_tokens: int
    key_mask: torch.Tensor
    query_indices: torch.Tensor | None = None
    key_indices: torch.Tensor | None = None
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0


def prepare_attention_layout(valid, *, query_valid=None, query_tokens=None, all_valid=None):
    """Prepare dense or packed attention without assuming prefix-only padding.

    ``query_valid=None`` retains every query, including for query chunks or
    cross attention. Supply the full self-attention validity mask to pack Q as
    well as K/V. ``all_valid=True`` is a trusted, already-validated caller
    contract and avoids reading CUDA mask values entirely.
    """
    if valid.ndim != 2 or valid.dtype != torch.bool:
        raise ValueError("Attention validity must be boolean [B,N]")
    batch, key_tokens = valid.shape
    if batch < 1 or key_tokens < 1:
        raise ValueError("Attention requires nonempty batch and key dimensions")
    if query_valid is not None:
        if (query_valid.ndim != 2 or query_valid.dtype != torch.bool
                or query_valid.shape[0] != batch or query_valid.device != valid.device):
            raise ValueError("Query validity must be boolean [B,Q] on the key device")
        if query_tokens is not None and query_tokens != query_valid.shape[1]:
            raise ValueError("Query token count differs from query validity")
        query_tokens = query_valid.shape[1]
    query_tokens = key_tokens if query_tokens is None else query_tokens
    if type(query_tokens) is not int or query_tokens < 1:
        raise ValueError("Query token count must be positive")
    if all_valid is not None and type(all_valid) is not bool:
        raise ValueError("all_valid must be a boolean or None")
    if all_valid is None:
        all_valid = bool(valid.all()) and (
            query_valid is None or bool(query_valid.all())
        )
    # Clone masks so later batch mutations cannot alter this prepared layout.
    key_mask = valid[:, None, None, :].clone()
    if all_valid:
        return AttentionLayout("dense", batch, query_tokens, key_tokens, key_mask)
    key_counts = valid.sum(-1, dtype=torch.int32)
    query_counts = (
        query_valid.sum(-1, dtype=torch.int32) if query_valid is not None
        else torch.full((batch,), query_tokens, device=valid.device, dtype=torch.int32)
    )
    # This one eager transfer also verifies the optional kernel preconditions.
    counts = torch.stack((query_counts, key_counts)).cpu().tolist()
    if min(counts[1]) < 1:
        raise ValueError("Fused attention requires at least one valid key per example")
    if min(counts[0]) < 1:
        raise ValueError("Packed attention requires at least one valid query per example")
    zero = key_counts.new_zeros(1)
    return AttentionLayout(
        "varlen", batch, query_tokens, key_tokens, key_mask,
        query_indices=(query_valid.reshape(-1).nonzero(as_tuple=True)[0]
                       if query_valid is not None else None),
        key_indices=valid.reshape(-1).nonzero(as_tuple=True)[0],
        cu_seqlens_q=torch.cat((zero, query_counts.cumsum(0, dtype=torch.int32))),
        cu_seqlens_k=torch.cat((zero, key_counts.cumsum(0, dtype=torch.int32))),
        max_seqlen_q=max(counts[0]), max_seqlen_k=max(counts[1]),
    )


def prepare_temporal_rope(times, head_dim):
    """Return reusable FP32 (cos, sin) for times[B,N] and even head width."""
    if times.ndim != 2 or type(head_dim) is not int or head_dim < 2 or head_dim % 2:
        raise ValueError("RoPE requires [B,N] times and a positive even head width")
    half = head_dim // 2
    freq = torch.exp(-math.log(10000) * torch.arange(
        half, device=times.device, dtype=torch.float32
    ) / half)
    phase = times.float()[:, None, :, None] * freq
    return phase.cos(), phase.sin()


def resolve_spatiotemporal_rope_geometry(head_dim, axis_dims, spatial_grid_size):
    """Validate time/x/y channel allocation and explicit image patch grid H/W."""
    if type(head_dim) is not int or head_dim < 6 or head_dim % 2:
        raise ValueError("Spatial-temporal RoPE requires an even head width of at least 6")
    if axis_dims is None:
        if head_dim % 8:
            raise ValueError("Automatic RoPE axis dimensions require head width divisible by 8")
        axis_dims = (head_dim // 2, head_dim // 4, head_dim // 4)
    if (not isinstance(axis_dims, Sequence) or isinstance(axis_dims, (str, bytes)) or len(axis_dims) != 3
            or any(type(value) is not int or value < 2 or value % 2 for value in axis_dims)
            or sum(axis_dims) != head_dim):
        raise ValueError("rope_axis_dims must be positive-even time/x/y dimensions summing to head width")
    if (not isinstance(spatial_grid_size, Sequence) or isinstance(spatial_grid_size, (str, bytes)) or len(spatial_grid_size) != 2
            or any(type(value) is not int or value < 1 for value in spatial_grid_size)):
        raise ValueError("spatial_grid_size must explicitly name positive integer patch-grid height/width")
    return tuple(axis_dims), tuple(spatial_grid_size)


def prepare_spatiotemporal_rope(times, xy, head_dim, *, axis_dims=None, spatial_grid_size=None):
    """FP32 axial rotations for time plus normalized image-patch center XY.

    Normalized centers scale by(grid_width/2,grid_height/2), producing centered
    patch units: a10-wide grid has x=-4.5,...,4.5 with adjacent spacing1.
    ``xy=None`` gives nonimage tokens zero spatial phases while preserving the
    same temporal frequencies as image tokens in joint attention.

    The existing rotation pairs the first and second halves of each head.
    Therefore each half orders its coordinates as[time,x,y], and this helper
    concatenates axis frequency vectors in that order. A64/32/32 allocation
    means32 time,16 x and16 y *pairs*, not a contiguous-slice rotation.
    """
    axes, grid = resolve_spatiotemporal_rope_geometry(head_dim, axis_dims, spatial_grid_size)
    if times.ndim != 2 or times.dtype == torch.bool or times.is_complex():
        raise ValueError("Spatial-temporal RoPE requires real-valued times[B,N]")
    if xy is not None and (xy.shape != (*times.shape, 2) or xy.device != times.device
                           or not xy.is_floating_point()):
        raise ValueError("Spatial RoPE coordinates must be floating-point[B,N,2] on the time device")
    positions = [times.float()]
    positions.extend((torch.zeros_like(times, dtype=torch.float32),) * 2 if xy is None else
                     (xy[..., 0].float() * (grid[1] / 2), xy[..., 1].float() * (grid[0] / 2)))
    phases = []
    for coordinate, dimension in zip(positions, axes):
        pairs = dimension // 2
        frequency = torch.exp(-math.log(10000) * torch.arange(
            pairs, device=times.device, dtype=torch.float32
        ) / pairs)
        phases.append(coordinate[:, None, :, None] * frequency)
    phase = torch.cat(phases, dim=-1)
    return phase.cos(), phase.sin()


def apply_temporal_rope(x, rope):
    cos, sin = rope
    left, right = x.float().chunk(2, dim=-1)
    return torch.cat((left * cos - right * sin, left * sin + right * cos), -1).to(x.dtype)


@dataclass(frozen=True)
class FlashAttentionKernels:
    backend: str
    dense: Callable
    varlen: Callable


@lru_cache(maxsize=16)
def load_flash_attention_kernels(backend, device):
    """Resolve a declared CUDA backend once; missing kernels never fall back."""
    device = torch.device(device)
    if backend not in ("flash3", "flash4"):
        raise ValueError("Expected flash3 or flash4 backend")
    if device.type != "cuda":
        raise RuntimeError(f"{backend} attention requires a CUDA Hopper/Blackwell GPU")
    if torch.cuda.get_device_capability(device)[0] < 9:
        raise RuntimeError(f"{backend} requires Hopper or newer CUDA hardware")
    if backend == "flash4":
        from packaging.version import Version
        if Version(torch.__version__.split("+")[0]) < Version("2.9"):
            raise RuntimeError("flash4 requires the separately verified Torch>=2.9 runtime")
    module_names = (("flash_attn.cute",) if backend == "flash4"
                    else ("flash_attn_3.flash_attn_interface", "flash_attn_interface"))
    try:
        for index, module_name in enumerate(module_names):
            try:
                module = importlib.import_module(module_name)
                break
            except ModuleNotFoundError as error:
                # Older official FA3 builds expose a top-level alias. Only
                # missing module/package names permit that compatibility path;
                # an installed backend with a missing dependency must fail.
                if (index == len(module_names) - 1
                        or error.name not in {module_name, module_name.split(".")[0]}):
                    raise
        dense, varlen = module.flash_attn_func, module.flash_attn_varlen_func
    except (ImportError, AttributeError) as error:
        raise RuntimeError(
            f"Install the verified optional {backend} runtime ({module_name}); no backend fallback"
        ) from error
    if not callable(dense) or not callable(varlen):
        raise RuntimeError(f"Unsupported {backend} dense/varlen API")
    return FlashAttentionKernels(backend, dense, varlen)


def _output_tensor(result, expected):
    # FA4 b35 returns (out, optional_lse); FA3 returns out when LSE is disabled.
    if isinstance(result, tuple):
        if len(result) != 2:
            raise RuntimeError("Unsupported fused-attention result tuple")
        result = result[0]
    if (not torch.is_tensor(result) or result.shape != expected.shape
            or result.dtype != expected.dtype or result.device != expected.device):
        raise RuntimeError("Unsupported fused-attention output shape/dtype/device")
    return result


def execute_flash_attention(query, key, value, layout, kernels):
    """Execute prepared full attention; arguments use [B,H,N,D] layout."""
    if any(x.dtype != torch.bfloat16 for x in (query, key, value)):
        raise ValueError("Fused attention requires bfloat16 Q/K/V")
    b, h, nq, d = query.shape
    if (b != layout.batch_size or nq != layout.query_tokens
            or key.shape != (b, h, layout.key_tokens, d) or value.shape != key.shape
            or key.device != query.device or value.device != query.device
            or layout.key_mask.device != query.device):
        raise ValueError("Prepared attention layout differs from Q/K/V shape/device")
    q, k, v = (x.transpose(1, 2).contiguous() for x in (query, key, value))
    if layout.kind == "dense":
        out = _output_tensor(kernels.dense(q, k, v, causal=False), q)
        return out.transpose(1, 2)
    if layout.kind != "varlen":
        raise ValueError("Unknown attention layout kind")
    q = q.reshape(-1, h, d)
    if layout.query_indices is not None:
        q = q.index_select(0, layout.query_indices)
    k = k.reshape(-1, h, d).index_select(0, layout.key_indices)
    v = v.reshape(-1, h, d).index_select(0, layout.key_indices)
    out = _output_tensor(kernels.varlen(
        q, k, v, cu_seqlens_q=layout.cu_seqlens_q, cu_seqlens_k=layout.cu_seqlens_k,
        max_seqlen_q=layout.max_seqlen_q, max_seqlen_k=layout.max_seqlen_k,
        causal=False,
    ), q)
    if layout.query_indices is not None:
        out = out.new_zeros(b * nq, h, d).index_copy(0, layout.query_indices, out)
    return out.reshape(b, nq, h, d).transpose(1, 2)


def execute_packed_flash_attention(query, key, value, layout, kernels):
    """Attend already-packed [total_tokens,H,D] tensors without repadding.

    The layout retains the original examples' boundaries. It must never be
    rebuilt as one sequence merely because tensor regions use a leading 1.
    """
    if layout.kind != "varlen":
        raise ValueError("Direct packed attention requires a varlen layout")
    if any(x.ndim != 3 or x.dtype != torch.bfloat16 for x in (query, key, value)):
        raise ValueError("Packed fused attention requires bfloat16 [tokens,H,D] Q/K/V")
    query_count = (layout.query_indices.numel() if layout.query_indices is not None
                   else layout.batch_size * layout.query_tokens)
    if (query.shape[0] != query_count
            or key.shape != (layout.key_indices.numel(), *query.shape[1:])
            or value.shape != key.shape or key.device != query.device
            or value.device != query.device or layout.key_mask.device != query.device):
        raise ValueError("Prepared packed attention layout differs from Q/K/V shape/device")
    return _output_tensor(kernels.varlen(
        query.contiguous(), key.contiguous(), value.contiguous(),
        cu_seqlens_q=layout.cu_seqlens_q, cu_seqlens_k=layout.cu_seqlens_k,
        max_seqlen_q=layout.max_seqlen_q, max_seqlen_k=layout.max_seqlen_k,
        causal=False,
    ), query)
