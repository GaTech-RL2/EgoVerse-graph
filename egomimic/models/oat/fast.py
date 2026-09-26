"""Construct the released FAST policy and load its self-contained BPE artifact."""

import torch

from egomimic.models.oat.factory import (
    _preserve_normalizer_devices,
    libero_shape_meta,
    make_obs_encoder,
)
from egomimic.models.oat.policy.fastpolicy import FASTPolicy
from egomimic.models.oat.tokenizer.fast.tokenizer_wrapper import FASTTok


def load_tokenizer(checkpoint):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("format") != "egoverse_fast_v1":
        raise ValueError("Expected a native fitted FAST tokenizer artifact")
    tokenizer = FASTTok(**payload["fast_tokenizer_config"])
    if not tokenizer._training_data_context or (
        tokenizer._training_data_context != payload["benchmark_data_context"]
    ):
        raise ValueError("FAST artifact lacks matching training data provenance")
    return _preserve_normalizer_devices(tokenizer).eval().requires_grad_(False)


def make_policy(
    tokenizer_checkpoint=None,
    *,
    tokenizer_config=None,
    shape_meta=None,
    n_obs_steps=2,
    n_action_steps=16,
    embed_dim=256,
    n_layers=4,
    n_heads=4,
    dropout=0.1,
    max_seq_len=128,
    temperature=1.0,
    topk=10,
):
    if (tokenizer_checkpoint is None) == (tokenizer_config is None):
        raise ValueError(
            "Supply exactly one FAST tokenizer artifact or embedded config"
        )
    tokenizer = (
        load_tokenizer(tokenizer_checkpoint)
        if tokenizer_config is None
        else _preserve_normalizer_devices(FASTTok(**tokenizer_config))
    )
    shape_meta = shape_meta or libero_shape_meta(
        action_dim=tokenizer.fast_tok.action_dim
    )
    return FASTPolicy(
        shape_meta=shape_meta,
        obs_encoder=make_obs_encoder(shape_meta),
        action_tokenizer=tokenizer,
        horizon=tokenizer.fast_tok.time_horizon,
        n_obs_steps=n_obs_steps,
        n_action_steps=n_action_steps,
        embed_dim=embed_dim,
        n_layers=n_layers,
        n_heads=n_heads,
        dropout=dropout,
        max_seq_len=max_seq_len,
        temperature=temperature,
        topk=topk,
    )
