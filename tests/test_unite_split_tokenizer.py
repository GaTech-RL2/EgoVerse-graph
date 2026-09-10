"""Cotrain topology B: one tokenizer per embodiment, one shared denoiser."""
from pathlib import Path

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

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
ROOT = Path(__file__).parents[1]
CONFIG_DIR = ROOT / "egomimic/hydra_configs"


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


def test_row_b_config_composes_and_flags_per_embodiment_tokenizer():
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(
            config_name="train_zarr_cartesian",
            overrides=[
                "+experiment=pusht/unite_cotrain_usocket_chain_val01_h16",
                "model=bf/ct_unite_register_split_tok_nt8_h384_s42",
            ],
        )
    ge = cfg.model.pipeline.stages[4].generative_encoder
    assert bool(ge.per_embodiment_tokenizer) is True
    assert bool(cfg.model.share_encoder_denoiser) is False
    assert set(ge.action_dims) == {U, C}
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg_a = compose(
            config_name="train_zarr_cartesian",
            overrides=[
                "+experiment=pusht/unite_cotrain_usocket_chain_val01_h16",
                "model=bf/ct_unite_register_separate_nt8_h384_s42",
            ],
        )
    ge_a = OmegaConf.to_container(cfg_a.model.pipeline.stages[4].generative_encoder, resolve=True)
    ge_b = OmegaConf.to_container(ge, resolve=True)
    ge_b.pop("per_embodiment_tokenizer")
    assert ge_a == ge_b  # B differs from A by exactly one flag


def test_dp_cotrain_row_composes_with_six_wide_head_and_padded_decoder():
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(
            config_name="train_zarr_cartesian",
            overrides=["+experiment=pusht/planar_v2_cotrain_dp_paper_points6"],
        )
    assert set(cfg.data.train_datasets) == {U, C}
    assert int(cfg.model.pipeline.stages[3].policy.model.input_dim) == 6
    assert cfg.evaluator.native_decoders[U]._target_.endswith("PaddedPlanarCommon5NativeDecoder")
    assert int(cfg.planar.observation_horizon) == 2 and int(cfg.planar.action_target_offset) == 1
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
