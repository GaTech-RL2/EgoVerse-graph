"""Cotrain topology B: one tokenizer per embodiment, one shared denoiser."""

import numpy as np
import pytest
import torch

from egomimic.pipeline.pushshapes import PaddedPlanarCommon5NativeDecoder
from egomimic.pipeline.stages_unite_separate import (
    PerEmbodimentTokenizerUniteGenerativeEncoder,
    SeparateUniteGenerativeEncoder,
    build_configurable_unite_generative_encoder,
)
from egomimic.rldb.embodiment.pushshapes import get_planar_paper_padded_transform_list
from egomimic.rldb.zarr.action_chunk_transforms import PadActionWidth

from tests.test_unite_register import _backbone_config

U = "pushshapes_sim_u_socket"
C = "pushshapes_sim_chain_gripper"


def _encoder(per_embodiment: bool, shared: bool = False):
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
    )


def test_split_tokenizer_topology_and_parameter_ownership():
    enc = _encoder(True)
    assert isinstance(enc, PerEmbodimentTokenizerUniteGenerativeEncoder)
    assert not isinstance(enc, SeparateUniteGenerativeEncoder)
    assert set(enc.tokenization_modules) == {U, C}
    assert enc.tokenization_modules[U] is not enc.tokenization_modules[C]
    assert enc.tokenization_module is not enc.denoising_module
    assert not hasattr(enc, "output_norm")
    # parameter budget: A (shared tokenizer body) + one extra tokenizer DiT
    a = _encoder(False)
    tok = sum(p.numel() for p in a.tokenization_module.parameters())
    n_a = sum(p.numel() for p in a.parameters())
    n_b = sum(p.numel() for p in enc.parameters())
    extra_heads = sum(
        p.numel()
        for n, p in enc.named_parameters()
        if n.startswith(("tokenization_output_norms.", "tokenization_condition_projections."))
    ) - sum(
        p.numel()
        for n, p in a.named_parameters()
        if n.startswith(("output_norm.", "tokenization_condition_projection."))
    )
    assert n_b == n_a + tok + extra_heads
    names = dict(enc.named_parameters())
    assert not any(n.startswith("tokenization_module.") for n in names)
    assert len({id(p) for p in names.values()}) == len(names)


def test_split_tokenizer_routes_each_embodiment_to_its_own_tokenizer():
    torch.manual_seed(0)
    enc = _encoder(True).eval()
    reg = torch.randn(3, 4, 16)
    cond = torch.randn(3, 14)
    z_u = enc.tokenize(torch.randn(3, 16, 4), U, reg)
    z_c = enc.tokenize(torch.randn(3, 16, 6), C, reg)
    assert z_u.shape == z_c.shape == (3, 4, 16)
    # zeroing the chain tokenizer must not change the U-Socket latent
    with torch.no_grad():
        for p in enc.tokenization_modules[C].parameters():
            p.zero_()
    torch.manual_seed(1)
    z_u2 = enc.tokenize(torch.randn(3, 16, 4) * 0 + 1, U, reg)
    torch.manual_seed(1)
    z_u3 = enc.tokenize(torch.randn(3, 16, 4) * 0 + 1, U, reg)
    assert torch.allclose(z_u2, z_u3)
    with pytest.raises(ValueError):
        enc.tokenize(torch.randn(3, 16, 6), U, reg)
    out = enc.denoise(torch.randn(3, 4, 16), torch.rand(3), cond, C)
    assert out.shape == (3, 4, 16)


def test_split_tokenizer_gradients_stay_per_embodiment():
    enc = _encoder(True).train()
    reg = torch.randn(2, 4, 16)
    enc.tokenize(torch.randn(2, 16, 4), U, reg).sum().backward()
    assert all(p.grad is None for p in enc.tokenization_modules[C].parameters())
    assert any(p.grad is not None for p in enc.tokenization_modules[U].parameters())
    assert all(p.grad is None for p in enc.denoising_module.parameters())


def test_split_tokenizer_requires_unshared_flag():
    with pytest.raises(ValueError):
        _encoder(True, shared=True)


def test_split_tokenizer_state_dict_round_trip():
    a = _encoder(True)
    b = _encoder(True)
    b.load_state_dict(a.state_dict(), strict=True)
    for (n1, p1), (n2, p2) in zip(a.named_parameters(), b.named_parameters()):
        assert n1 == n2 and torch.equal(p1, p2)


def test_padded_planar_transform_and_decoder_round_trip_shape():
    transforms = get_planar_paper_padded_transform_list(action_horizon=16, action_target_offset=1)
    batch = {"actions": np.random.rand(17, 3).astype(np.float32)}
    for t in transforms:
        batch = t.transform(batch)
    assert batch["actions"].shape == (16, 6)
    assert np.all(batch["actions"][:, 4:] == 0)
    dec = PaddedPlanarCommon5NativeDecoder(action_horizon=16, native_action_dim=3, padded_dim=6)
    native = dec.decode(torch.from_numpy(batch["actions"]))
    assert native.shape == (1, 16, 3)
    with pytest.raises(ValueError):
        PadActionWidth(["actions"], width=2).transform({"actions": np.zeros((4, 3))})


def _policy(per_embodiment: bool, compile_backbones: bool):
    from egomimic.models.unite_action_decoder import UniteActionDecoder
    from egomimic.pipeline.stages_unite_released import ReleasedRecipeUniteLatentPolicy

    decoders = {
        d: UniteActionDecoder(latent_dim=16, action_dim=a, num_latent_tokens=4, action_horizon=16,
                              hidden_dim=32, depth=2, num_heads=4, gradient_checkpointing=False)
        for d, a in ((U, 4), (C, 6))
    }
    return ReleasedRecipeUniteLatentPolicy(
        generative_encoder=_encoder(per_embodiment), decoders=decoders,
        flow_steps_per_reconstruction=14, flow_mini_batch=14, compile_backbones=compile_backbones,
    )


@pytest.mark.parametrize("per_embodiment", [False, True])
def test_compile_flag_wraps_backbones_without_renaming_parameters(per_embodiment):
    torch.manual_seed(0)
    plain = _policy(per_embodiment, False)
    torch.manual_seed(0)
    compiled = _policy(per_embodiment, True)
    assert compiled.compile_backbones is True and plain.compile_backbones is False
    targets = compiled.compiled_modules()
    assert len(targets) == (3 if per_embodiment else 2) + 2  # tokenizer(s) + denoiser + two decoders
    assert all(getattr(m, "_compiled_call_impl", None) is not None for m in targets)
    assert all(getattr(m, "_compiled_call_impl", None) is None for m in plain.compiled_modules())
    assert list(compiled.state_dict().keys()) == list(plain.state_dict().keys())
    plain.load_state_dict(compiled.state_dict(), strict=True)
    for (n1, p1), (n2, p2) in zip(plain.named_parameters(), compiled.named_parameters()):
        assert n1 == n2 and torch.equal(p1, p2)
