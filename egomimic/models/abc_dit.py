"""ABC-DiT, ported from amazon-far/abc (``abc_minimal/dit.py``, d0832d1).

The upstream policy is one module that owns vision, conditioning and the
denoiser. Here it is split along the graph's existing ``condition`` seam:

    images, state, task --ABCObservationStage--> condition (B, 2 + cams * queries, H)
    condition, noisy action, time --FlowDenoiserStage(ABCDiT)--> velocity

Token 0 of ``condition`` is the embedded proprioceptive state and token 1 the
projected CLIP task embedding; both feed AdaLN with the flow time. The
remaining tokens are the per-camera attention-pooled DINOv3 features the DiT
blocks cross-attend to. The DINOv3 module names follow upstream so Meta's
``.pth`` loads unchanged; initialization matches upstream's from-scratch
finetune (PyTorch defaults for the DiT, pretrained DINOv3 vision, frozen
pretrained CLIP ViT-B/32 text).

Deliberately not ported: the CLIP vision backbone and the RTC action-prefix
paths.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# DINOv3 ViT-B/16 vision encoder.


def _rope_rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


class DinoRope(nn.Module):
    """RoPE over the 2D patch grid (base=100, separate coord normalization)."""

    def __init__(self, embed_dim, num_heads, base=100.0, rescale_coords=2.0):
        super().__init__()
        d_head = embed_dim // num_heads
        self.d_head = d_head
        self.rescale_coords = rescale_coords
        self.register_buffer("periods", torch.empty(d_head // 4), persistent=True)
        with torch.no_grad():
            self.periods.copy_(
                base
                ** (2 * torch.arange(d_head // 4, dtype=torch.float32) / (d_head // 2))
            )

    def forward(self, H, W):
        dev = self.periods.device
        coords_h = torch.arange(0.5, H, device=dev, dtype=torch.float32) / H
        coords_w = torch.arange(0.5, W, device=dev, dtype=torch.float32) / W
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"), dim=-1)
        coords = 2.0 * coords.flatten(0, 1) - 1.0
        if self.training and self.rescale_coords is not None:
            r = np.log(self.rescale_coords)
            coords = coords * torch.empty(1, device=dev).uniform_(-r, r).exp()
        angles = 2 * math.pi * coords[:, :, None] / self.periods[None, None, :]
        angles = angles.flatten(1, 2).tile(2)
        return torch.sin(angles), torch.cos(angles)


class LinearKMaskedBias(nn.Linear):
    """qkv Linear whose k-third of the bias is masked to zero (DINOv3 quirk)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer("bias_mask", torch.full_like(self.bias, math.nan))

    def forward(self, x):
        return F.linear(x, self.weight, self.bias * self.bias_mask.to(self.bias.dtype))


class DinoAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.qkv = LinearKMaskedBias(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x, rope=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = [t.transpose(1, 2) for t in torch.unbind(qkv, 2)]
        if rope is not None:
            sin, cos = rope
            n_prefix = N - sin.shape[-2]  # cls + storage tokens are not rotated
            q_dt, k_dt = q.dtype, k.dtype
            q, k = q.to(sin.dtype), k.to(sin.dtype)
            q = torch.cat(
                [
                    q[:, :, :n_prefix],
                    q[:, :, n_prefix:] * cos
                    + _rope_rotate_half(q[:, :, n_prefix:]) * sin,
                ],
                dim=-2,
            )
            k = torch.cat(
                [
                    k[:, :, :n_prefix],
                    k[:, :, n_prefix:] * cos
                    + _rope_rotate_half(k[:, :, n_prefix:]) * sin,
                ],
                dim=-2,
            )
            q, k = q.to(q_dt), k.to(k_dt)
        x = F.scaled_dot_product_attention(q, k, v)
        return self.proj(x.transpose(1, 2).reshape(B, N, C))


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x * self.gamma


class DinoMlp(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class DinoBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-5)
        self.attn = DinoAttention(dim, num_heads)
        self.ls1 = LayerScale(dim)
        self.norm2 = nn.LayerNorm(dim, eps=1e-5)
        self.mlp = DinoMlp(dim, int(dim * ffn_ratio))
        self.ls2 = LayerScale(dim)

    def forward(self, x, rope=None):
        x = x + self.ls1(self.attn(self.norm1(x), rope=rope))
        return x + self.ls2(self.mlp(self.norm2(x)))


class DinoPatchEmbed(nn.Module):
    def __init__(self, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        x = self.proj(x)  # (B, D, H/16, W/16)
        return x.flatten(2).transpose(1, 2), x.shape[2], x.shape[3]


class DinoVisionTransformer(nn.Module):
    N_STORAGE_TOKENS = 4

    def __init__(self, embed_dim, depth, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_embed = DinoPatchEmbed(embed_dim=embed_dim)
        self.cls_token = nn.Parameter(torch.empty(1, 1, embed_dim))
        self.storage_tokens = nn.Parameter(
            torch.empty(1, self.N_STORAGE_TOKENS, embed_dim)
        )
        self.mask_token = nn.Parameter(torch.empty(1, embed_dim))
        self.rope_embed = DinoRope(embed_dim, num_heads)
        self.blocks = nn.ModuleList(
            DinoBlock(embed_dim, num_heads) for _ in range(depth)
        )
        self.norm = nn.LayerNorm(embed_dim, eps=1e-5)
        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.storage_tokens, std=0.02)
        nn.init.zeros_(self.mask_token)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                if isinstance(m, LinearKMaskedBias):
                    o = m.out_features
                    m.bias_mask.fill_(1)
                    m.bias_mask[o // 3 : 2 * o // 3].fill_(0)
            elif isinstance(m, nn.LayerNorm):
                m.reset_parameters()
            elif isinstance(m, LayerScale):
                nn.init.constant_(m.gamma, 1e-5)
            elif isinstance(m, DinoPatchEmbed):
                m.proj.reset_parameters()

    def encode_image_tokens(self, images):
        """Returns (B, 1 + patches, D): CLS + patch tokens, storage tokens dropped."""
        x, H, W = self.patch_embed(images)
        B = x.shape[0]
        cls_token = self.cls_token + 0 * self.mask_token  # upstream quirk, kept
        x = torch.cat(
            [cls_token.expand(B, -1, -1), self.storage_tokens.expand(B, -1, -1), x],
            dim=1,
        )
        rope = self.rope_embed(H, W)
        for blk in self.blocks:
            x = blk(x, rope=rope)
        x = self.norm(x)
        return torch.cat([x[:, :1], x[:, 1 + self.N_STORAGE_TOKENS :]], dim=1)


def load_dinov3_weights(vit: DinoVisionTransformer, path: str) -> None:
    """Load Meta's ``dinov3_vitb16_pretrain_lvd1689m.pth`` into ``vit``.

    Upstream loads with strict=False and only prints the miss count, which lets a
    wrong file silently train a random encoder. Fail instead; only the
    deterministic buffers may be absent.
    """
    state = torch.load(path, map_location="cpu", weights_only=False)
    state = state.get("model", state)
    missing, _ = vit.load_state_dict(state, strict=False)
    missing = [
        k for k in missing if not k.endswith(("bias_mask", "rope_embed.periods"))
    ]
    if missing:
        raise RuntimeError(
            f"{path} is not a DINOv3 ViT-B/16 checkpoint; missing {missing[:8]}"
        )


# ABC-DiT policy pieces.


def modulate(x, shift, scale):
    if shift.ndim == 2:
        shift = shift.unsqueeze(1)
        scale = scale.unsqueeze(1)
    return x * (1 + scale) + shift


def gate_residual(gate, residual):
    if gate.ndim == 2:
        gate = gate.unsqueeze(1)
    return gate * residual


def get_1d_sincos_pos_embed(embed_dim, length):
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    out = np.einsum("m,d->md", np.arange(length, dtype=np.float64), omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        half = frequency_embedding_size // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, dtype=torch.float32) / half
        )
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, t):
        args = t.reshape(-1)[:, None].float() * self.freqs[None]
        t_freq = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.mlp(t_freq.to(self.mlp[0].weight.dtype)).reshape(*t.shape, -1)


class DiTAttention(nn.Module):
    """Self-attention over action tokens (timm-equivalent, qkv_bias=True)."""

    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        x = F.scaled_dot_product_attention(q, k, v)
        return self.proj(x.transpose(1, 2).reshape(B, N, C))


class DiTMlp(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class DiTBlock(nn.Module):
    """AdaLN DiT block with vision cross-attention (9-way modulation)."""

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = DiTAttention(hidden_size, num_heads)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = DiTMlp(hidden_size, int(hidden_size * mlp_ratio))
        self.norm_xattn = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm_xattn_kv = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6
        )
        self.cross_attn = nn.MultiheadAttention(
            hidden_size, num_heads, batch_first=True
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 9 * hidden_size, bias=True)
        )

    def forward(self, x, c, vision_tokens):
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_xattn,
            scale_xattn,
            gate_xattn,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.adaLN_modulation(c).chunk(9, dim=-1)
        x = x + gate_residual(
            gate_msa, self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        )
        x_normed = modulate(self.norm_xattn(x), shift_xattn, scale_xattn)
        kv = self.norm_xattn_kv(vision_tokens)
        xattn_out, _ = self.cross_attn(x_normed, kv, kv, need_weights=False)
        x = x + gate_residual(gate_xattn, xattn_out)
        return x + gate_residual(
            gate_mlp, self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        )


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, action_dim):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, action_dim, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        return self.linear(modulate(self.norm_final(x), shift, scale))


class PoolMlp(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, in_dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class AttentionPoolBlock(nn.Module):
    """Learnable queries cross-attend to ViT tokens (per camera)."""

    def __init__(self, embed_dim, num_heads, mlp_ratio=4):
        super().__init__()
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.ln_2 = nn.LayerNorm(embed_dim)
        self.mlp = PoolMlp(embed_dim, int(mlp_ratio * embed_dim))

    def forward(self, x, queries):
        x_kv = self.ln_1(x)
        out, _ = self.attention(self.ln_1(queries), x_kv, x_kv, need_weights=False)
        return self.mlp(self.ln_2(out)) + out


class ABCDiT(nn.Module):
    """ABC-DiT velocity network with the ``FlowDenoiserStage`` model signature.

    ``forward(x_t, time, condition)`` where ``condition`` is the
    ``(B, 2 + V, hidden_size)`` token block written by ``ABCObservationStage``.
    Tokens 0 (state), 1 (task) and the flow time drive AdaLN; tokens 2: are
    the cross-attention keys. Same flow convention as upstream and ``stages_flow``:
    x_t = t * noise + (1 - t) * action, velocity = noise - action.
    """

    def __init__(
        self,
        action_dim: int,
        action_horizon: int,
        hidden_size: int = 1536,
        depth: int = 32,
        num_heads: int = 24,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        H = hidden_size
        self.y_embedder = nn.Linear(action_dim, H)
        self.t_embedder = TimestepEmbedder(H)
        pos = get_1d_sincos_pos_embed(H, action_horizon)
        self.register_buffer(
            "pos_embed", torch.from_numpy(pos).float().unsqueeze(0), persistent=False
        )
        self.blocks = nn.ModuleList(
            DiTBlock(H, num_heads, mlp_ratio) for _ in range(depth)
        )
        self.final_layer = FinalLayer(H, action_dim)
        # cond = [state, task, timestep] -> hidden (vision goes via cross-attn)
        self.cond_proj = nn.Sequential(
            nn.Linear(3 * H, H), nn.SiLU(), nn.Linear(H, H), nn.LayerNorm(H)
        )

    def forward(self, x_t, time, condition):
        state_token, task_token = condition[:, 0], condition[:, 1]
        vision_tokens = condition[:, 2:]
        c = self.cond_proj(
            torch.cat([state_token, task_token, self.t_embedder(time)], dim=-1)
        )
        z = self.y_embedder(x_t) + self.pos_embed[:, : x_t.shape[1]]
        for block in self.blocks:
            z = block(z, c, vision_tokens)
        return self.final_layer(z, c)


class CLIPTaskEncoder(nn.Module):
    """Frozen OpenAI CLIP ViT-B/32 text tower -> L2-normalized 512-d task vectors.

    Upstream loads the OpenAI ``ViT-B-32.pt`` text weights; ``checkpoint`` is a
    local ``openai/clip-vit-base-patch32`` Hugging Face directory holding the
    same weights. Embeddings are cached per prompt string.
    """

    def __init__(self, checkpoint: str):
        super().__init__()
        from transformers import CLIPTextModelWithProjection, CLIPTokenizer

        self.tokenizer = CLIPTokenizer.from_pretrained(checkpoint)
        self.model = CLIPTextModelWithProjection.from_pretrained(checkpoint).eval()
        self.model.requires_grad_(False)
        self._cache: dict[str, torch.Tensor] = {}

    def train(self, mode: bool = True):
        super().train(mode)
        self.model.eval()
        return self

    @torch.no_grad()
    def forward(self, texts: list[str]) -> torch.Tensor:
        device = self.model.text_projection.weight.device
        fresh = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if fresh:
            tokens = self.tokenizer(fresh, padding=True, return_tensors="pt").to(device)
            features = self.model(**tokens).text_embeds
            features = features / features.norm(dim=-1, keepdim=True)
            self._cache.update(zip(fresh, features.float().cpu()))
        return torch.stack([self._cache[t] for t in texts]).to(device)


class ABCVisionStateEncoder(nn.Module):
    """DINOv3 + per-camera attention pooling + state embedding -> condition tokens.

    ``images`` are ``(B, 3, 224, 224)`` ImageNet-normalized tensors in
    ``num_cameras`` order and ``tasks`` one prompt string per sample; returns
    ``(B, 2 + num_cameras * pool_queries, H)``.
    """

    def __init__(
        self,
        state_dim: int,
        num_cameras: int,
        hidden_size: int = 1536,
        vit_embed_dim: int = 768,
        vit_depth: int = 12,
        vit_num_heads: int = 12,
        pool_queries: int = 12,
        pool_num_heads: int = 8,
        pool_mlp_ratio: int = 4,
        backbone_checkpoint: str | None = None,
        backbone_bf16: bool = True,
        text_checkpoint: str | None = None,
        task_embed_dim: int = 512,
    ):
        super().__init__()
        self.dinov3_model = DinoVisionTransformer(
            vit_embed_dim, vit_depth, vit_num_heads
        )
        if backbone_checkpoint:
            load_dinov3_weights(self.dinov3_model, backbone_checkpoint)
        self.backbone_bf16 = bool(backbone_bf16)
        self.apool_queries = nn.Parameter(
            torch.randn(num_cameras, 1, pool_queries, vit_embed_dim) * 0.02
        )
        self.apool = nn.ModuleList(
            AttentionPoolBlock(vit_embed_dim, pool_num_heads, pool_mlp_ratio)
            for _ in range(num_cameras)
        )
        self.vision_tokens_proj = nn.Linear(vit_embed_dim, hidden_size)
        self.vision_camera_embed = nn.Embedding(num_cameras, hidden_size)
        self.x_embedder = nn.Linear(state_dim, hidden_size)
        self.text_encoder = (
            CLIPTaskEncoder(text_checkpoint) if text_checkpoint else None
        )
        self.task_to_hidden = nn.Linear(task_embed_dim, hidden_size)

    def _encode(self, images):
        if self.backbone_bf16 and images.is_cuda:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                return self.dinov3_model.encode_image_tokens(images).float()
        return self.dinov3_model.encode_image_tokens(images)

    def forward(
        self,
        images: list[torch.Tensor],
        state: torch.Tensor,
        tasks: list[str] | torch.Tensor,
    ) -> torch.Tensor:
        """``tasks`` is prompt strings, or precomputed (B, task_embed_dim) vectors."""
        pooled = []
        for index, image in enumerate(images):
            tokens = self._encode(image)
            queries = self.apool_queries[index].expand(tokens.shape[0], -1, -1)
            pooled.append(self.apool[index](tokens, queries))
        vision = self.vision_tokens_proj(torch.stack(pooled, dim=1))  # (B, Nc, K, H)
        vision = vision + self.vision_camera_embed.weight[None, :, None, :]
        vision = vision.flatten(1, 2)
        task_vec = tasks if torch.is_tensor(tasks) else self.text_encoder(list(tasks))
        task = self.task_to_hidden(task_vec.to(state.dtype))
        return torch.cat(
            [self.x_embedder(state)[:, None], task[:, None], vision], dim=1
        )
