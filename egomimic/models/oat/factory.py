"""Native constructors for the pinned OAT networks (no upstream runtime)."""

from __future__ import annotations

import torch

from egomimic.models.oat.model.common.normalizer import (
    LinearNormalizer,
    SingleFieldLinearNormalizer,
)
from egomimic.models.oat.tokenizer.oat.decoder.single_pass_decoder import (
    SinglePassDecoder,
)
from egomimic.models.oat.tokenizer.oat.encoder.register_encoder import RegisterEncoder
from egomimic.models.oat.tokenizer.oat.quantizer.fsq import FSQ
from egomimic.models.oat.tokenizer.oat.tokenizer import OATTok


def _capture_normalizer_devices(module, *_args):
    # Upstream normalizers replace their ParameterDict from checkpoint tensors.
    # A CPU-mapped checkpoint otherwise moves a CUDA normalizer back to CPU.
    module._normalizer_load_devices = [
        (child, next(child.parameters()).device)
        for child in module.modules()
        if isinstance(child, LinearNormalizer) and len(child.params_dict)
    ]


def _restore_normalizer_devices(module, _incompatible_keys):
    for child, device in module._normalizer_load_devices:
        child.to(device)
    del module._normalizer_load_devices


def _preserve_normalizer_devices(network):
    # Hook the parent: DictOfTensorMixin overrides _load_from_state_dict without
    # invoking Module's pre-hooks. This also covers Lightning GPU resume.
    network.register_load_state_dict_pre_hook(_capture_normalizer_devices)
    network.register_load_state_dict_post_hook(_restore_normalizer_devices)
    return network


def identity_normalizer(keys):
    normalizer = LinearNormalizer()
    for key in keys:
        normalizer[key] = SingleFieldLinearNormalizer.create_identity()
    return normalizer


def make_tokenizer(
    action_dim=7,
    horizon=32,
    num_registers=8,
    levels=(8, 5, 5, 5),
    emb_dim=256,
    head_dim=64,
    encoder_depth=2,
    decoder_depth=4,
    dropout=0.1,
    token_dropout_mode="pow2",
    use_causal_decoder=True,
):
    """The released LIBERO OAT configuration; inputs are graph-normalized."""
    if horizon < 1 or action_dim < 1 or num_registers < 1:
        raise ValueError("horizon, action_dim and num_registers must be positive")
    if not levels or any(int(level) != level or level < 2 for level in levels):
        raise ValueError("FSQ levels must be integers of at least two")
    if emb_dim % head_dim:
        raise ValueError("emb_dim must be divisible by head_dim")
    if token_dropout_mode == "pow2" and num_registers & (num_registers - 1):
        raise ValueError("pow2 dropout requires a power-of-two register count")
    tokenizer = OATTok(
        encoder=RegisterEncoder(
            action_dim,
            horizon,
            emb_dim,
            head_dim,
            encoder_depth,
            dropout,
            len(levels),
            num_registers,
        ),
        decoder=SinglePassDecoder(
            sample_dim=action_dim,
            sample_horizon=horizon,
            emb_dim=emb_dim,
            head_dim=head_dim,
            depth=decoder_depth,
            pdropout=dropout,
            token_dropout_mode=token_dropout_mode,
            use_causal_decoder=use_causal_decoder,
            latent_dim=len(levels),
            latent_horizon=num_registers,
        ),
        quantizer=FSQ(levels=list(levels)),
    )
    # MultiDataset owns the real affine transform, once, for both methods.
    tokenizer.set_normalizer(identity_normalizer(["action"]))
    return _preserve_normalizer_devices(tokenizer)


def libero_shape_meta(image_size=128):
    return {
        "action": {"shape": [7]},
        "obs": {
            "agentview_rgb": {"shape": [image_size, image_size, 3], "type": "rgb"},
            "robot0_eye_in_hand_rgb": {
                "shape": [image_size, image_size, 3],
                "type": "rgb",
            },
            "robot0_eef_pos": {"shape": [3], "type": "state"},
            "robot0_eef_quat": {"shape": [4], "type": "state"},
            "robot0_gripper_qpos": {"shape": [2], "type": "state"},
            "task_uid": {"shape": [1], "type": "state"},
        },
    }


def make_obs_encoder(shape_meta=None, crop_shape=(76, 76)):
    from omegaconf import OmegaConf

    from egomimic.models.oat.perception.fused_obs_encoder import FusedObservationEncoder

    shape_meta = libero_shape_meta() if shape_meta is None else shape_meta
    encoder = FusedObservationEncoder(
        shape_meta,
        vision_encoder=OmegaConf.create(
            {
                "_target_": "egomimic.models.oat.perception.robomimic_vision_encoder.RobomimicRgbEncoder",
                "crop_shape": list(crop_shape),
            }
        ),
        state_encoder=OmegaConf.create(
            {
                "_target_": "egomimic.models.oat.perception.state_encoder.ProjectionStateEncoder",
                "out_dim": None,
            }
        ),
    )
    encoder.set_normalizer(identity_normalizer(shape_meta["obs"]))
    return _preserve_normalizer_devices(encoder)


def load_tokenizer(checkpoint, *, use_ema=True):
    """Extract a trained tokenizer from a native graph checkpoint, strictly."""
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    from egomimic.eval.checkpoint_loading import strict_load_pipeline_checkpoint

    if checkpoint is None:
        raise ValueError(
            "Set benchmark.tokenizer_checkpoint to a trained OAT tokenizer"
        )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(payload["hyper_parameters"]["config_tree"])
    stage_cfg = cfg.model.pipeline.stages
    if (
        len(stage_cfg) != 1
        or stage_cfg[0]._target_ != "egomimic.pipeline.stages_oat.OATTokenizerStage"
    ):
        raise ValueError("Expected a graph OAT tokenizer-training checkpoint")
    algo = instantiate(cfg.model.pipeline, device="cpu")
    strict_load_pipeline_checkpoint(algo, payload, use_ema=use_ema)
    tokenizer = algo.pipeline.stages[0].tokenizer
    tokenizer._native_config = OmegaConf.to_container(
        stage_cfg[0].tokenizer, resolve=True
    )
    tokenizer.eval().requires_grad_(False)
    # The dataset fingerprint and affine state accompany the learned weights.
    tokenizer._training_data_context = payload.get("benchmark_data_context")
    if tokenizer._training_data_context is None:
        raise ValueError("Tokenizer checkpoint is missing benchmark_data_context")
    return tokenizer


def make_policy(
    tokenizer_checkpoint,
    *,
    use_ema=True,
    shape_meta=None,
    n_obs_steps=2,
    n_action_steps=16,
    embed_dim=256,
    n_layers=4,
    n_heads=4,
    dropout=0.1,
    temperature=1.0,
    topk=10,
):
    from egomimic.models.oat.policy.oatpolicy import OATPolicy

    shape_meta = libero_shape_meta() if shape_meta is None else shape_meta
    return OATPolicy(
        shape_meta,
        make_obs_encoder(shape_meta),
        load_tokenizer(tokenizer_checkpoint, use_ema=use_ema),
        n_action_steps=n_action_steps,
        n_obs_steps=n_obs_steps,
        embed_dim=embed_dim,
        n_layers=n_layers,
        n_heads=n_heads,
        dropout=dropout,
        temperature=temperature,
        topk=topk,
    )


def make_policy_from_config(
    tokenizer_config,
    *,
    shape_meta=None,
    n_obs_steps=2,
    n_action_steps=16,
    embed_dim=256,
    n_layers=4,
    n_heads=4,
    dropout=0.1,
    temperature=1.0,
    topk=10,
):
    """Restore architecture before strictly loading a self-contained checkpoint."""
    from hydra.utils import instantiate

    from egomimic.models.oat.policy.oatpolicy import OATPolicy

    shape_meta = libero_shape_meta() if shape_meta is None else shape_meta
    tokenizer = instantiate(tokenizer_config)
    tokenizer._native_config = dict(tokenizer_config)
    return OATPolicy(
        shape_meta,
        make_obs_encoder(shape_meta),
        tokenizer,
        n_action_steps,
        n_obs_steps,
        embed_dim,
        n_layers,
        n_heads,
        dropout,
        temperature,
        topk,
    )
