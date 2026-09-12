"""HPT trunk primitives and the tensor helpers this fork had to re-add."""

from functools import partial

import numpy as np
import pytest
import torch

from egomimic.models.cores.hpt_transformer import (
    CrossAttention,
    MultiheadAttention,
    SimpleTransformer,
)
from egomimic.models.cores.hpt_utils import EinOpsRearrange, get_sinusoid_encoding_table

_DIM = 64
_HEADS = 4


def _trunk(num_blocks: int = 2, embed_dim: int = _DIM, **kwargs) -> SimpleTransformer:
    return SimpleTransformer(
        attn_target=partial(
            MultiheadAttention,
            embed_dim=embed_dim,
            num_heads=_HEADS,
            batch_first=True,
        ),
        embed_dim=embed_dim,
        num_blocks=num_blocks,
        **kwargs,
    )


# -- trunk ------------------------------------------------------------------


def test_trunk_preserves_token_shape_and_returns_per_block_outputs():
    tokens = torch.randn(2, 10, _DIM)
    out, block_outputs = _trunk(num_blocks=3)(tokens)
    assert out.shape == tokens.shape
    # The second return value is what a hierarchical head would consume; the
    # graph stage only forwards the final tokens, so pin the contract.
    assert len(block_outputs) == 3
    assert all(b.shape == tokens.shape for b in block_outputs)


def test_trunk_accepts_any_sequence_length():
    trunk = _trunk()
    for length in (1, 7, 64):
        out, _ = trunk(torch.randn(2, length, _DIM))
        assert out.shape == (2, length, _DIM)


def test_trunk_actually_mixes_across_tokens():
    # Self-attention must couple positions; without it a per-token MLP would
    # leave an isolated perturbation local.
    torch.manual_seed(0)
    trunk = _trunk().eval()
    a = torch.randn(1, 8, _DIM)
    b = a.clone()
    b[0, 0] += 5.0
    with torch.no_grad():
        out_a, _ = trunk(a)
        out_b, _ = trunk(b)
    moved = (out_a - out_b).abs().mean(dim=-1)[0]
    assert moved[0] > 0
    assert moved[1:].max() > 0, "perturbing one token left the others unchanged"


def test_trunk_gradient_checkpointing_matches_the_plain_path():
    torch.manual_seed(0)
    trunk = _trunk(num_blocks=4).eval()
    tokens = torch.randn(2, 6, _DIM)
    with torch.no_grad():
        plain, _ = trunk(tokens)
        checkpointed, _ = trunk(tokens, use_checkpoint=True)
    torch.testing.assert_close(plain, checkpointed)


def test_trunk_rejects_an_unknown_drop_path_type():
    with pytest.raises(ValueError, match="Unknown drop_path_type"):
        _trunk(drop_path_type="sometimes")


def test_trunk_attention_mask_changes_the_output():
    torch.manual_seed(0)
    trunk = _trunk().eval()
    tokens = torch.randn(1, 5, _DIM)
    causal = torch.triu(torch.ones(5, 5, dtype=torch.bool), diagonal=1)
    with torch.no_grad():
        unmasked, _ = trunk(tokens)
        masked, _ = trunk(tokens, attn_mask=causal)
    assert not torch.allclose(unmasked, masked)


# -- cross attention (used by the stems) ------------------------------------


def test_cross_attention_maps_queries_to_the_query_dim():
    attn = CrossAttention(query_dim=_DIM, heads=_HEADS, dim_head=16)
    out = attn(torch.randn(2, 8, _DIM), torch.randn(2, 12, _DIM))
    assert out.shape == (2, 8, _DIM)


def test_cross_attention_output_depends_on_the_context():
    torch.manual_seed(0)
    attn = CrossAttention(query_dim=_DIM, heads=_HEADS, dim_head=16).eval()
    query = torch.randn(1, 4, _DIM)
    with torch.no_grad():
        a = attn(query, torch.randn(1, 6, _DIM))
        b = attn(query, torch.randn(1, 6, _DIM))
    assert not torch.allclose(a, b)


def test_cross_attention_latent_count_sets_the_output_length():
    # This is how the stems compress a variable-length modality to a fixed
    # number of latents before the trunk sees them.
    attn = CrossAttention(query_dim=_DIM, heads=_HEADS, dim_head=16)
    for latents in (1, 16):
        out = attn(torch.randn(2, latents, _DIM), torch.randn(2, 9, _DIM))
        assert out.shape == (2, latents, _DIM)


# -- tensor helpers ---------------------------------------------------------


def test_sinusoid_table_shape_broadcasts_onto_a_token_sequence():
    table = get_sinusoid_encoding_table(0, 10, _DIM)
    assert table.shape == (1, 10, _DIM)
    assert torch.isfinite(table).all()


def test_sinusoid_table_is_bounded_and_position_dependent():
    table = get_sinusoid_encoding_table(0, 16, _DIM)[0]
    assert table.abs().max() <= 1.0
    assert not torch.allclose(table[0], table[1])


def test_sinusoid_table_respects_a_nonzero_start():
    full = get_sinusoid_encoding_table(0, 20, _DIM)[0]
    offset = get_sinusoid_encoding_table(5, 20, _DIM)[0]
    assert offset.shape == (15, _DIM)
    torch.testing.assert_close(offset, full[5:])


def test_sinusoid_table_handles_an_odd_hidden_size():
    # The cosine half is one column short when d_hid is odd; that slice is why.
    table = get_sinusoid_encoding_table(0, 4, 65)
    assert table.shape == (1, 4, 65)
    assert torch.isfinite(table).all()


@pytest.mark.parametrize("bad", [(5, 5), (7, 3)])
def test_sinusoid_table_rejects_an_empty_range(bad):
    with pytest.raises(ValueError, match="position_end"):
        get_sinusoid_encoding_table(bad[0], bad[1], _DIM)


def test_sinusoid_table_rejects_a_nonpositive_hidden_size():
    with pytest.raises(ValueError, match="d_hid"):
        get_sinusoid_encoding_table(0, 4, 0)


def test_einops_rearrange_module_transposes_as_declared():
    module = EinOpsRearrange("b t d -> b d t")
    assert module(torch.zeros(2, 10, _DIM)).shape == (2, _DIM, 10)


def test_einops_rearrange_module_composes_inside_a_sequential():
    model = torch.nn.Sequential(EinOpsRearrange("b t d -> b (t d)"))
    assert model(torch.zeros(2, 3, 4)).shape == (2, 12)


def test_einops_rearrange_module_rejects_a_non_tensor():
    with pytest.raises(TypeError, match="expects a tensor"):
        EinOpsRearrange("b t d -> b d t")(np.zeros((2, 3, 4)))
