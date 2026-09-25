"""Change-loop terms on the released UNITE policy (2026-09-25).

Contrastive flow matching, dispersive loss and the flow-only latent BatchNorm
are default-off kwargs of ``ReleasedRecipeUniteLatentPolicy``; the wrapper
optimises every optional ``loss/*`` term the objective writes (including the
pre-existing decoded-action term, which it used to drop).
"""
from collections import OrderedDict

import math

import pytest
import torch

from egomimic.eval.checkpoint_loading import strict_load_pipeline_checkpoint
from egomimic.models.unite_action_decoder import UniteActionDecoder
from egomimic.pipeline.stages_unite_released import (
    ReleasedRecipeUniteLatentPolicy,
    ReleasedRecipeUniteObjective,
)
from egomimic.pipeline.stages_unite_separate import (
    build_configurable_unite_generative_encoder,
)
from tests.test_unite_split_tokenizer import C, U, _backbone_config
from tests.test_unite_training import _wrapper

EMB = {U: 19, C: 20}
DIMS = {U: 4, C: 6}
DEFAULTS = dict(
    contrastive_flow_weight=0.0,
    dispersive_weight=0.0,
    dispersive_tau=0.5,
    dispersive_block=3,
    flow_latent_batchnorm=False,
)


def _policy(seed: int = 0, **kwargs) -> ReleasedRecipeUniteLatentPolicy:
    torch.manual_seed(seed)
    encoder = build_configurable_unite_generative_encoder(
        backbone_config=_backbone_config(4),
        share_encoder_denoiser=False,
        action_dims=dict(DIMS),
        condition_input_dim=14,
        latent_dim=16,
        num_latent_tokens=4,
        condition_dim=12,
        denoiser_hidden_dim=32,
        gradient_checkpointing=False,
        in_context_start=1,
        in_context_len=8,
    )
    decoders = {
        domain: UniteActionDecoder(
            latent_dim=16,
            action_dim=dim,
            num_latent_tokens=4,
            action_horizon=16,
            hidden_dim=32,
            depth=2,
            num_heads=4,
            gradient_checkpointing=False,
        )
        for domain, dim in DIMS.items()
    }
    policy = ReleasedRecipeUniteLatentPolicy(
        generative_encoder=encoder,
        decoders=decoders,
        flow_steps_per_reconstruction=14,
        flow_mini_batch=14,
        **kwargs,
    )
    # zero-initialised output projections would make several gradients vanish
    with torch.no_grad():
        for name, parameter in policy.named_parameters():
            if parameter.abs().sum() == 0 and "norm" not in name:
                parameter.normal_(0.0, 0.02)
    return policy


def _batch(domain: str, rows: int = 3, seed: int = 1) -> dict:
    generator = torch.Generator().manual_seed(seed)
    return {
        "target": torch.randn(rows, 16, DIMS[domain], generator=generator),
        "condition": torch.randn(rows, 14, generator=generator),
        "sampler/noise": torch.randn(rows, 4, 16, generator=generator),
        "embodiment": torch.full((rows,), EMB[domain]),
    }


def _step(policy, objective=None, *, seed: int = 7, domains=(U, C)):
    """Run every source through policy+objective and the wrapper reduction."""
    objective = objective or ReleasedRecipeUniteObjective()
    policy.train()
    policy.zero_grad(set_to_none=True)
    torch.manual_seed(seed)
    predictions = OrderedDict(
        (domain, objective(policy(_batch(domain, seed=index + 1))))
        for index, domain in enumerate(domains)
    )
    components, _ = _wrapper()._weighted_components(predictions)
    components["TotalLoss"].backward()
    grads = OrderedDict(
        (name, None if p.grad is None else p.grad.detach().clone())
        for name, p in policy.named_parameters()
    )
    return predictions, components, grads


def _assert_identical(first, second):
    (_, c1, g1), (_, c2, g2) = first, second
    assert list(c1) == list(c2)
    for name in c1:
        assert torch.equal(c1[name], c2[name]), name
    assert list(g1) == list(g2)
    for name in g1:
        if g1[name] is None or g2[name] is None:
            assert g1[name] is None and g2[name] is None, name
        else:
            assert torch.equal(g1[name], g2[name]), name


def test_default_kwargs_are_byte_identical_to_absent_kwargs():
    absent = _policy()
    explicit = _policy(**DEFAULTS)
    assert list(absent.state_dict()) == list(explicit.state_dict())
    for key, value in absent.state_dict().items():
        assert torch.equal(value, explicit.state_dict()[key])
    first, second = _step(absent), _step(explicit)
    _assert_identical(first, second)
    predictions, components, _ = first
    # the wrapper's TotalLoss is still exactly reconstruction + flow
    assert list(components) == [
        "TotalLoss", "ReconstructionLoss", "FlowLoss", "ReconstructionL1"
    ]
    for result in predictions.values():
        assert not any(
            key in result
            for key in (
                "loss/unite_decoded_action",
                "loss/unite_contrastive_flow",
                "loss/unite_dispersive",
                "unite/contrastive_flow_loss",
                "unite/dispersive_loss",
            )
        )


def test_zero_weights_and_disabled_batchnorm_add_no_modules_or_buffers():
    base = _policy()
    assert not hasattr(base, "flow_latent_norms")
    bn = _policy(flow_latent_batchnorm=True)
    count = lambda m: sum(p.numel() for p in m.parameters())  # noqa: E731
    assert count(base) == count(bn)
    extra = set(bn.state_dict()) - set(base.state_dict())
    assert extra == {
        f"flow_latent_norms.{d}.{b}"
        for d in (U, C)
        for b in ("running_mean", "running_var", "num_batches_tracked")
    }


def test_decoded_action_term_is_now_optimised():
    policy = _policy(decoded_action_samples_per_reconstruction=2)
    objective = ReleasedRecipeUniteObjective(decoded_action_weight=0.1)
    predictions, components, grads = _step(policy, objective)
    assert "DecodedActionLoss" in components
    decoded = components["DecodedActionLoss"]
    assert float(decoded) > 0.0 and math.isfinite(float(decoded))
    total = (
        components["ReconstructionLoss"] + components["FlowLoss"] + decoded
    )
    assert torch.equal(components["TotalLoss"], total)
    # decoder weights receive gradient from the decoded path (the tokenizer does not)
    assert any(
        g is not None and g.abs().sum() > 0
        for n, g in grads.items()
        if n.startswith("action_decoder.")
    )


def test_contrastive_flow_term_sign_scale_and_formula():
    weight = 0.05
    policy = _policy(contrastive_flow_weight=weight)
    predictions, components, _ = _step(policy)
    term = components["ContrastiveFlowLoss"]
    distance = components["ContrastiveFlowDistance"]
    assert float(term) < 0.0 and float(distance) > 0.0
    assert torch.allclose(term, -weight * distance)
    assert torch.equal(
        components["TotalLoss"],
        components["ReconstructionLoss"] + components["FlowLoss"] + term,
    )
    # the distance to a wrong sample's velocity dominates the flow loss at init
    assert float(distance) > 0.5 * float(components["FlowLoss"])


def test_contrastive_negative_is_the_rolled_row_of_the_same_repeat(monkeypatch):
    policy = _policy(contrastive_flow_weight=0.05).train()
    seen = {}

    def fake_denoise(latent, time, condition, embodiment):
        seen["latent"] = latent
        return torch.zeros_like(latent)

    monkeypatch.setattr(policy.generative_encoder, "denoise", fake_denoise)
    monkeypatch.setattr(policy, "_sample_flow_time", lambda n, device: torch.zeros(n))
    clean = torch.randn(3, 4, 16)
    torch.manual_seed(0)
    flow, contrast, dispersive = policy._released_flow_terms(
        clean, torch.randn(3, 14), U
    )
    assert dispersive is None
    # t = 0: z_t = noise, prediction 0 -> v_hat = -noise; v = z - noise
    noise = seen["latent"]
    v_hat = -noise
    target = clean.repeat(14, 1, 1) - noise
    negative = target.reshape(14, 3, 4, 16).roll(1, dims=1).reshape_as(target)
    expected = (v_hat - negative).square().mean() * 14
    assert torch.allclose(contrast, expected)
    assert torch.allclose(flow, (clean.repeat(14, 1, 1) - 0).square().mean() * 14)


def test_dispersive_term_matches_formula_and_reaches_the_denoiser():
    policy = _policy(dispersive_weight=0.5, dispersive_tau=0.5, dispersive_block=1)
    # formula: identical rows -> log mean exp(0) = 0; distant rows -> log(1/B)
    same = torch.ones(4, 3, 5)
    assert float(policy._dispersive_term(same)) == pytest.approx(0.0, abs=1e-7)
    far = torch.eye(4).reshape(4, 1, 4) * 1.0e3
    assert float(policy._dispersive_term(far)) == pytest.approx(math.log(1 / 4))
    h = torch.randn(5, 2, 3)
    d = torch.cdist(h.reshape(5, -1), h.reshape(5, -1)).square() / 6
    assert torch.allclose(
        policy._dispersive_term(h), torch.log(torch.exp(-d / 0.5).mean()), atol=1e-6
    )

    predictions, components, grads = _step(policy)
    raw = components["DispersiveRaw"]
    assert float(raw) <= 0.0
    assert torch.allclose(components["DispersiveLoss"], 0.5 * raw)
    # the term alone differentiates the hooked and earlier denoiser blocks
    policy.zero_grad(set_to_none=True)
    torch.manual_seed(3)
    out = policy.train()(_batch(U))
    out["unite/dispersive_loss"].backward()
    blocks = policy.generative_encoder.denoising_module.blocks
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in blocks[0].parameters())
    assert all(p.grad is None for p in policy.action_decoder.parameters())


def test_dispersive_block_must_exist():
    with pytest.raises(ValueError, match="dispersive_block"):
        _policy(dispersive_weight=0.5, dispersive_block=7)


def test_flow_latent_batchnorm_round_trip_and_running_stats():
    policy = _policy(flow_latent_batchnorm=True).train()
    latent = torch.randn(6, 4, 16) * 3.0 + 1.5
    normalized, stats = policy._normalize_flow_latent(latent, U)
    assert torch.allclose(normalized.mean(dim=(0, 1)), torch.zeros(16), atol=1e-5)
    assert torch.allclose(
        normalized.var(dim=(0, 1), unbiased=False), torch.ones(16), atol=1e-3
    )
    restored = policy._denormalize_flow_latent(normalized, stats)
    assert torch.allclose(restored, latent, atol=1e-5)
    norm = policy.flow_latent_norms[U]
    assert int(norm.num_batches_tracked) == 1
    assert int(policy.flow_latent_norms[C].num_batches_tracked) == 0
    policy.eval()
    normalized_eval, stats_eval = policy._normalize_flow_latent(latent, U)
    assert torch.allclose(
        policy._denormalize_flow_latent(normalized_eval, stats_eval), latent, atol=1e-5
    )
    assert torch.allclose(stats_eval[0].flatten(), norm.running_mean)


def test_flow_latent_batchnorm_trains_and_inverts_before_decoding(monkeypatch):
    policy = _policy(
        flow_latent_batchnorm=True, decoded_action_samples_per_reconstruction=2
    )
    objective = ReleasedRecipeUniteObjective(decoded_action_weight=0.1)
    predictions, components, _ = _step(policy, objective)
    assert "FlowLatentBatchStd" in components
    assert math.isfinite(float(components["TotalLoss"]))
    for domain in (U, C):
        assert int(policy.flow_latent_norms[domain].num_batches_tracked) == 1
    # rollout: the sampled endpoint is mapped back with the running statistics
    policy.eval()
    with torch.no_grad():
        policy.flow_latent_norms[U].running_mean.fill_(2.0)
        policy.flow_latent_norms[U].running_var.fill_(9.0)
    monkeypatch.setattr(
        policy, "sample", lambda noise, condition, embodiment: torch.zeros_like(noise)
    )
    decoded = {}
    monkeypatch.setattr(
        policy, "_decode", lambda latent, embodiment: decoded.setdefault("z", latent)
    )
    out = policy(_batch(U))
    assert torch.allclose(out["sampler/endpoint"], torch.full((3, 4, 16), 2.0))
    assert torch.allclose(decoded["z"], torch.full((3, 4, 16), 2.0))


def test_ema_overlay_keeps_the_batchnorm_buffers():
    policy = _policy(flow_latent_batchnorm=True)
    _step(policy)
    online = {f"nets.{k}": v.clone() for k, v in policy.state_dict().items()}
    ema = {
        f"nets.{k}": torch.zeros_like(v) for k, v in policy.named_parameters()
    }

    class _Algo:
        def __init__(self, nets):
            self.nets = nets

    fresh = _Algo(_policy(seed=5, flow_latent_batchnorm=True))
    strict_load_pipeline_checkpoint(
        fresh, {"state_dict": online, "ema_state_dict": ema}, use_ema=True
    )
    state = fresh.nets.state_dict()
    for domain in (U, C):
        for buffer in ("running_mean", "running_var", "num_batches_tracked"):
            key = f"flow_latent_norms.{domain}.{buffer}"
            assert torch.equal(state[key], online[f"nets.{key}"])
    assert int(state[f"flow_latent_norms.{U}.num_batches_tracked"]) == 1
    assert all(float(p.abs().sum()) == 0.0 for p in fresh.nets.parameters())


def test_new_terms_refuse_unite_av():
    with pytest.raises(ValueError, match="UNITE-AV"):
        _policy(contrastive_flow_weight=0.05, action_velocity_samples_per_reconstruction=2)
