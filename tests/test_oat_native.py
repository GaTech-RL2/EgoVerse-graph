"""Numerical parity with the pinned upstream, plus native graph contracts."""

import importlib
import json
import os
import sys
from pathlib import Path

import pytest
import torch

from egomimic.models.oat.factory import make_tokenizer
from egomimic.models.oat.model.autoregressive.transformer_cache import (
    AutoregressiveModel,
)
from egomimic.models.oat.tokenizer.oat.encoder.register_encoder import (
    create_causal_last_mask,
)
from egomimic.models.oat.tokenizer.oat.quantizer.fsq import FSQ
from egomimic.pipeline.algo import PipelineAlgo
from egomimic.pipeline.stages_oat import OATPolicyStage, OATTokenizerStage

SMALL = dict(
    horizon=8,
    emb_dim=32,
    head_dim=8,
    encoder_depth=1,
    decoder_depth=1,
    num_registers=4,
    levels=(3, 4),
    dropout=0.0,
)


@pytest.fixture(scope="module")
def reference():
    root = os.environ.get("OAT_REFERENCE_ROOT")
    if not root:
        pytest.skip(
            "Set OAT_REFERENCE_ROOT to the pinned OAT checkout for source parity"
        )
    root = Path(root)
    import hashlib

    manifest = json.loads(
        (Path(__file__).parents[1] / "egomimic/models/oat/UPSTREAM.json").read_text()
    )
    for relative, expected in manifest["files"].items():
        assert (
            hashlib.sha256((root / "oat" / relative).read_bytes()).hexdigest()
            == expected
        )
    sys.path.insert(0, str(root))
    return lambda name: importlib.import_module("oat." + name)


def reference_tokenizer(reference, config=None):
    config = SMALL if config is None else config
    enc = reference("tokenizer.oat.encoder.register_encoder").RegisterEncoder(
        7,
        config["horizon"],
        config["emb_dim"],
        config["head_dim"],
        config["encoder_depth"],
        config["dropout"],
        len(config["levels"]),
        config["num_registers"],
    )
    dec = reference("tokenizer.oat.decoder.single_pass_decoder").SinglePassDecoder(
        sample_dim=7,
        sample_horizon=config["horizon"],
        emb_dim=config["emb_dim"],
        head_dim=config["head_dim"],
        depth=config["decoder_depth"],
        pdropout=config["dropout"],
        token_dropout_mode="pow2",
        use_causal_decoder=True,
        latent_dim=len(config["levels"]),
        latent_horizon=config["num_registers"],
    )
    quant = reference("tokenizer.oat.quantizer.fsq").FSQ(list(config["levels"]))
    tok = reference("tokenizer.oat.tokenizer").OATTok(enc, dec, quant)
    norm_module = reference("model.common.normalizer")
    norm = norm_module.LinearNormalizer()
    norm["action"] = norm_module.SingleFieldLinearNormalizer.create_identity()
    tok.set_normalizer(norm)
    return tok


def test_released_tokenizer_defaults_match_source_for_all_four_prefixes(reference):
    config = dict(
        horizon=32,
        emb_dim=256,
        head_dim=64,
        encoder_depth=2,
        decoder_depth=4,
        dropout=0.1,
        levels=(8, 5, 5, 5),
        num_registers=8,
    )
    native = make_tokenizer()
    original = reference_tokenizer(reference, config)
    original.load_state_dict(native.state_dict(), strict=True)
    actions = torch.randn(2, 32, 7)
    for tokenizer in (native, original):
        tokenizer.eval()
    with torch.no_grad():
        tokens = native.tokenize(actions)
        assert tokens.shape == (2, 8)
        assert torch.equal(tokens, original.tokenize(actions))
        for keep in (1, 2, 4, 8):
            torch.testing.assert_close(
                native.detokenize(tokens[:, :keep]),
                original.detokenize(tokens[:, :keep]),
                rtol=0,
                atol=0,
            )


def test_task_catalog_is_exact_pinned_libero_source(reference):
    import ast

    from egomimic.benchmarks.libero.catalog import TASKS

    source = Path(os.environ["OAT_REFERENCE_ROOT"]) / (
        "third_party/LIBERO/libero/libero/benchmark/libero_suite_task_map.py"
    )
    tree = ast.parse(source.read_text())
    assignment = next(node for node in tree.body if isinstance(node, ast.Assign))
    assert list(ast.literal_eval(assignment.value).items()) == list(TASKS.items())


def test_reference_tokenizer_loss_gradients_codes_and_prefixes(reference):
    torch.manual_seed(41)
    native = make_tokenizer(**SMALL)
    original = reference_tokenizer(reference)
    original.load_state_dict(native.state_dict(), strict=True)
    actions = torch.randn(3, 8, 7)
    torch.manual_seed(53)
    loss = native({"action": actions})
    loss.backward()
    torch.manual_seed(53)
    reference_loss = original({"action": actions})
    reference_loss.backward()
    torch.testing.assert_close(loss, reference_loss, rtol=0, atol=0)
    for (name, p), (other_name, q) in zip(
        native.named_parameters(), original.named_parameters()
    ):
        assert name == other_name
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)
    native.eval()
    original.eval()
    tokens = native.tokenize(actions)
    assert torch.equal(tokens, original.tokenize(actions))
    for keep in (1, 2, 4):
        torch.testing.assert_close(
            native.detokenize(tokens[:, :keep]),
            original.detokenize(tokens[:, :keep]),
            rtol=0,
            atol=0,
        )
    torch.testing.assert_close(
        native.autoencode(actions, [1, 2, 4]),
        original.autoencode(actions, [1, 2, 4]),
        rtol=0,
        atol=0,
    )


def test_reference_autoregression_logits_gradients_and_cached_sampling(reference):
    kwargs = dict(
        vocab_size=13,
        max_seq_len=5,
        max_cond_len=2,
        cond_dim=6,
        n_layer=2,
        n_head=2,
        n_emb=16,
        p_drop_emb=0,
        p_drop_attn=0,
    )
    native = AutoregressiveModel(**kwargs).eval()
    original = (
        reference("model.autoregressive.transformer_cache")
        .AutoregressiveModel(**kwargs)
        .eval()
    )
    original.load_state_dict(native.state_dict())
    tokens, cond = torch.randint(0, 13, (2, 4)), torch.randn(2, 2, 6)
    torch.testing.assert_close(
        native(tokens, cond), original(tokens, cond), rtol=0, atol=0
    )
    for temperature in (0, 1):
        torch.manual_seed(81)
        actual = native.generate(
            tokens[:, :1], cond, 4, temperature=temperature, top_k=3
        )
        torch.manual_seed(81)
        expected = original.generate(
            tokens[:, :1], cond, 4, temperature=temperature, top_k=3
        )
        assert torch.equal(actual, expected)
    # Cached greedy generation must equal repeated full-prefix inference.
    expected = tokens[:, :1]
    for _ in range(4):
        expected = torch.cat(
            (expected, native(expected, cond)[:, -1].argmax(-1, keepdim=True)), dim=1
        )
    assert torch.equal(native.generate(tokens[:, :1], cond, 4, temperature=0), expected)


def test_fsq_full_codebook_and_straight_through_gradients():
    fsq = FSQ([8, 5, 5, 5])
    ids = torch.arange(1000)
    assert torch.equal(fsq.codes_to_indices(fsq.indices_to_embedding(ids)).long(), ids)
    values = torch.randn(2, 8, 4, requires_grad=True)
    quantized, indices = fsq(values)
    assert indices.dtype == torch.long and indices.min() >= 0 and indices.max() < 1000
    quantized.sum().backward()
    assert torch.isfinite(values.grad).all() and values.grad.abs().sum() > 0


def test_registers_are_causal_and_actions_cannot_read_registers():
    mask = create_causal_last_mask(8, 4, "cpu")
    assert mask[:8, :8].all() and not mask[:8, 8:].any()
    assert mask[8:, :8].all()
    assert torch.equal(mask[8:, 8:], torch.ones(4, 4, dtype=torch.bool).tril())
    tok = make_tokenizer(**SMALL).eval()
    x = torch.randn(2, 8, 7)
    before = tok.encoder(x)
    with torch.no_grad():
        tok.encoder.registers[2:] += 3
    torch.testing.assert_close(tok.encoder(x)[:, :2], before[:, :2])


def test_dropped_suffix_cannot_change_prefix_reconstruction():
    tok = make_tokenizer(**SMALL).eval()
    tokens = tok.tokenize(torch.randn(2, 8, 7))
    latent = tok.quantizer.indices_to_embedding(tokens)
    changed = latent.clone()
    changed[:, 2:] += 100
    torch.testing.assert_close(tok.decode(latent, [2, 2]), tok.decode(changed, [2, 2]))


class TinyObservation(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(3, 6)

    def modalities(self):
        return ["state"]

    def output_feature_dim(self):
        return 6

    def forward(self, obs):
        return self.projection(obs["state"])


def small_policy(tokenizer=None):
    from egomimic.models.oat.policy.oatpolicy import OATPolicy

    return OATPolicy(
        {"action": {"shape": [7]}, "obs": {"state": {"shape": [3], "type": "state"}}},
        TinyObservation(),
        tokenizer or make_tokenizer(**SMALL),
        n_action_steps=4,
        n_obs_steps=2,
        embed_dim=16,
        n_layers=1,
        n_heads=2,
        dropout=0.1,
        topk=1,
    )


def test_policy_graph_freezes_tokenizer_and_infers_without_actions():
    policy = small_policy()
    stage = OATPolicyStage(policy)
    algo = PipelineAlgo([stage], device="cpu")
    algo.nets.train()
    assert policy.model.training and policy.obs_encoder.training
    assert all(not module.training for module in policy.action_tokenizer.modules())
    batch = {
        "libero_panda": {"state": torch.randn(2, 2, 3), "actions": torch.randn(2, 8, 7)}
    }
    outputs = algo.forward_training(batch)
    algo.compute_losses(outputs, batch)["loss"].backward()
    assert policy.obs_encoder.projection.weight.grad is not None
    assert all(
        parameter.grad is None for parameter in policy.action_tokenizer.parameters()
    )
    algo.nets.eval()
    inference = algo.forward_eval(
        {"libero_panda": {"state": batch["libero_panda"]["state"]}}
    )
    assert inference["libero_panda"]["pred_action"].shape == (2, 8, 7)


def test_tokenizer_graph_checkpoint_roundtrip():
    algo = PipelineAlgo([OATTokenizerStage(make_tokenizer(**SMALL))], device="cpu")
    data = {"libero_panda": {"actions": torch.randn(2, 8, 7)}}
    output = algo.forward_training(data)
    algo.compute_losses(output, data)["loss"].backward()
    restored = PipelineAlgo([OATTokenizerStage(make_tokenizer(**SMALL))], device="cpu")
    restored.nets.load_state_dict(algo.nets.state_dict(), strict=True)
    algo.nets.eval()
    restored.nets.eval()
    torch.testing.assert_close(
        algo.forward_eval(data)["libero_panda"]["pred_action"],
        restored.forward_eval(data)["libero_panda"]["pred_action"],
    )


def test_reference_real_vision_encoder_and_policy_loss(reference):
    from omegaconf import OmegaConf

    from egomimic.models.oat.factory import libero_shape_meta, make_obs_encoder
    from egomimic.models.oat.policy.oatpolicy import OATPolicy

    meta = libero_shape_meta()
    native_encoder = make_obs_encoder(meta)
    original_encoder = reference(
        "perception.fused_obs_encoder"
    ).FusedObservationEncoder(
        meta,
        vision_encoder=OmegaConf.create(
            {
                "_target_": "oat.perception.robomimic_vision_encoder.RobomimicRgbEncoder",
                "crop_shape": [76, 76],
            }
        ),
        state_encoder=OmegaConf.create(
            {
                "_target_": "oat.perception.state_encoder.ProjectionStateEncoder",
                "out_dim": None,
            }
        ),
    )
    original_encoder.set_normalizer(native_encoder.vision_encoder.normalizer)
    original_encoder.load_state_dict(native_encoder.state_dict(), strict=True)
    obs = {key: torch.randn(1, 2, *spec["shape"]) for key, spec in meta["obs"].items()}
    for mode in (False, True):
        native_encoder.train(mode)
        original_encoder.train(mode)
        torch.manual_seed(8)
        actual = native_encoder(obs)
        torch.manual_seed(8)
        expected = original_encoder(obs)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    native = OATPolicy(
        meta, native_encoder, make_tokenizer(**SMALL), 4, 2, 16, 1, 2, 0.1
    )
    original = reference("policy.oatpolicy").OATPolicy(
        meta, original_encoder, reference_tokenizer(reference), 4, 2, 16, 1, 2, 0.1
    )
    original.load_state_dict(native.state_dict(), strict=True)
    stage = OATPolicyStage(native).train()
    original.train()
    original.action_tokenizer.eval()
    actions = torch.randn(1, 8, 7)
    torch.manual_seed(81)
    actual = stage.execute({**obs, "actions": actions}, mode="train")[
        "loss/oat_token_ce"
    ]
    torch.manual_seed(81)
    expected = original({"obs": obs, "action": actions})
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_ema_decay_matches_upstream_counter(reference):
    from egomimic.pl_utils.oat_training import OATEMACallback

    callback = OATEMACallback(use_warmup=True, power=0.75)
    source = reference("model.diffusion.ema_model").EMAModel(
        torch.nn.Linear(2, 2), power=0.75
    )
    for update in (1, 2, 3, 101, 10001):
        assert callback._schedule(update) == source.get_decay(update - 1)
