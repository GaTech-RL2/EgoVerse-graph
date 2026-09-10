"""latent_norm_affine=False pins the tokenizer's output LayerNorm to unit scale."""
import pytest
import torch

from egomimic.pipeline.stages_unite_separate import (
    build_configurable_unite_generative_encoder,
)
from tests.test_unite_split_tokenizer import C, U, _backbone_config


def _build(per_embodiment: bool, affine: bool, shared: bool = False):
    return build_configurable_unite_generative_encoder(
        backbone_config=_backbone_config(4),
        share_encoder_denoiser=shared,
        action_dims={U: 4, C: 6},
        condition_input_dim=14,
        latent_dim=16,
        num_latent_tokens=4,
        condition_dim=12,
        denoiser_hidden_dim=32,
        gradient_checkpointing=False,
        in_context_start=1,
        in_context_len=8,
        per_embodiment_tokenizer=per_embodiment,
        latent_norm_affine=affine,
    )


@pytest.mark.parametrize("per_embodiment", [False, True])
def test_pinned_tokenization_norm_has_no_parameters_and_unit_scale(per_embodiment):
    torch.manual_seed(0)
    enc = _build(per_embodiment, affine=False)
    for domain in (U, C):
        norm = enc._tokenization_norm(domain)
        assert not norm.elementwise_affine
        assert sum(p.numel() for p in norm.parameters()) == 0
    # the denoiser-side norm keeps its affine parameters
    assert enc._denoising_norm().elementwise_affine
    assert enc.latent_norm_affine is False
    # the backbone's output projection is zero-initialised, so give it weight
    with torch.no_grad():
        for p in enc._tokenization_backbone(U).parameters():
            p.normal_(0.0, 0.05)
    actions = torch.randn(3, 5, 4)
    registers = torch.randn(3, 4, 16)
    latent = enc.tokenize(actions, U, registers)
    assert latent.shape == (3, 4, 16)
    assert torch.allclose(latent.mean(-1), torch.zeros(3, 4), atol=1e-4)
    assert torch.allclose(latent.var(-1, unbiased=False), torch.ones(3, 4), atol=2e-2)


def test_default_keeps_affine_and_parameter_count_differs():
    free = _build(False, affine=True)
    pinned = _build(False, affine=False)
    n_free = sum(p.numel() for p in free.parameters())
    n_pinned = sum(p.numel() for p in pinned.parameters())
    assert n_free - n_pinned == 2 * 16  # gain + bias of one LayerNorm(16)
    assert not hasattr(free, "latent_norm_affine") or free.latent_norm_affine is not False


def test_pinned_norm_rejects_tied_encoder():
    with pytest.raises(ValueError):
        _build(False, affine=False, shared=True)
