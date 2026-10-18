"""Packed attention isolation, regional compilation, and optional CUDA parity."""

import copy
import os

import pytest
import torch
from torch.nn import functional as F

from egomimic.models.attention_execution import (
    FlashAttentionKernels,
    apply_temporal_rope,
    execute_flash_attention,
    load_flash_attention_kernels,
    prepare_attention_layout,
    prepare_temporal_rope,
)
from egomimic.models.compilation import compile_region
from egomimic.models.world_model import MoTBlock, temporal_rope


def _dense(q, k, v, *, causal):
    assert causal is False
    return (
        F.scaled_dot_product_attention(
            q.float().transpose(1, 2),
            k.float().transpose(1, 2),
            v.float().transpose(1, 2),
        )
        .transpose(1, 2)
        .to(q.dtype)
    )


def _varlen(q, k, v, *, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, causal):
    assert cu_seqlens_q.dtype == cu_seqlens_k.dtype == torch.int32
    qs, ks = cu_seqlens_q.tolist(), cu_seqlens_k.tolist()
    assert max_seqlen_q == max(b - a for a, b in zip(qs, qs[1:]))
    assert max_seqlen_k == max(b - a for a, b in zip(ks, ks[1:]))
    return torch.cat(
        [
            _dense(q[qa:qb][None], k[ka:kb][None], v[ka:kb][None], causal=causal)[0]
            for qa, qb, ka, kb in zip(qs, qs[1:], ks, ks[1:])
        ]
    )


def _kernels(backend="flash4"):
    def dense(*args, **kwargs):
        out = _dense(*args, **kwargs)
        return (out, None) if backend == "flash4" else out

    def varlen(*args, **kwargs):
        out = _varlen(*args, **kwargs)
        return (out, None) if backend == "flash4" else out

    return FlashAttentionKernels(backend, dense, varlen)


@pytest.mark.parametrize("backend", ["flash3", "flash4"])
@pytest.mark.parametrize("packing", ["dense", "keys", "queries_and_keys"])
def test_batched_outputs_gradients_and_example_isolation(backend, packing):
    torch.manual_seed(43)
    b, h, nq, nk, d = 3, 2, 5, 7, 8
    q, k, v = [
        x.requires_grad_()
        for x in (
            torch.randn(b, h, nq, d).bfloat16(),
            torch.randn(b, h, nk, d).bfloat16(),
            torch.randn(b, h, nk, d).bfloat16(),
        )
    ]
    valid = torch.ones(b, nk, dtype=torch.bool)
    qvalid = torch.ones(b, nq, dtype=torch.bool)
    if packing != "dense":
        valid[0, [1, 4, 6]] = False
        valid[1, [0, 3]] = False
    if packing == "queries_and_keys":
        qvalid[0, [1, 3]] = False
        qvalid[2, 4] = False
    layout = prepare_attention_layout(valid, query_valid=qvalid, query_tokens=nq)
    out = execute_flash_attention(q, k, v, layout, _kernels(backend))
    expected = (
        F.scaled_dot_product_attention(
            q.float(), k.float(), v.float(), attn_mask=valid[:, None, None]
        )
        .to(q.dtype)
        .masked_fill(~qvalid[:, None, :, None], 0)
    )
    torch.testing.assert_close(out, expected, atol=0.008, rtol=0.02)
    actual_grads = torch.autograd.grad(
        out.float().square().sum(), (q, k, v), retain_graph=True
    )
    reference_grads = torch.autograd.grad(expected.float().square().sum(), (q, k, v))
    for actual, reference in zip(actual_grads, reference_grads):
        torch.testing.assert_close(actual, reference, atol=0.015, rtol=0.03)
        assert torch.isfinite(actual).all()
    assert actual_grads[1].transpose(1, 2)[~valid].count_nonzero() == 0
    assert actual_grads[0].transpose(1, 2)[~qvalid].count_nonzero() == 0
    changed = [x.detach().clone() for x in (q, k, v)]
    for x in changed:
        x[1:] *= 100
    isolated = execute_flash_attention(*changed, layout, _kernels(backend))
    assert torch.equal(out[0], isolated[0])


def test_key_only_layout_retains_all_queries_with_different_lengths():
    q = torch.randn(2, 2, 3, 8).bfloat16()
    k, v = (torch.randn(2, 2, 7, 8).bfloat16() for _ in range(2))
    valid = torch.tensor([[True, False, True, False, True, False, True], [True] * 7])
    layout = prepare_attention_layout(valid, query_tokens=3)
    assert layout.query_indices is None
    out = execute_flash_attention(q, k, v, layout, _kernels())
    expected = F.scaled_dot_product_attention(
        q.float(), k.float(), v.float(), attn_mask=valid[:, None, None]
    ).to(q.dtype)
    torch.testing.assert_close(out, expected, atol=0.008, rtol=0.02)


def test_verified_dense_layout_does_not_read_mask_values(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Fully valid path must not inspect CUDA mask values")

    valid = torch.ones(4, 11, dtype=torch.bool)
    monkeypatch.setattr(torch.Tensor, "all", forbidden)
    monkeypatch.setattr(torch.Tensor, "nonzero", forbidden)
    monkeypatch.setattr(torch.Tensor, "item", forbidden)
    layout = prepare_attention_layout(valid, query_valid=valid, all_valid=True)
    assert layout.kind == "dense" and layout.key_indices is None


def test_layout_owns_mask_and_handles_interior_holes():
    valid = torch.tensor([[True, False, True], [False, True, True]])
    layout = prepare_attention_layout(valid, query_valid=valid)
    valid[:] = False
    assert layout.key_mask.sum() == 4
    assert layout.key_indices.tolist() == [0, 2, 4, 5]
    assert layout.cu_seqlens_k.tolist() == [0, 2, 4]


@pytest.mark.parametrize(
    "valid,qvalid,match",
    [
        (torch.zeros(2, 3, dtype=torch.bool), None, "valid key"),
        (
            torch.ones(2, 3, dtype=torch.bool),
            torch.zeros(2, 3, dtype=torch.bool),
            "valid query",
        ),
        (torch.ones(2, 3), None, "boolean"),
    ],
)
def test_bad_layout_fails_closed(valid, qvalid, match):
    with pytest.raises(ValueError, match=match):
        prepare_attention_layout(valid, query_valid=qvalid)


def test_precomputed_rope_matches_historical_values_and_gradients():
    x = torch.randn(2, 3, 7, 8, requires_grad=True)
    times = torch.randn(2, 7)
    rope = prepare_temporal_rope(times, 8)
    assert all(value.dtype == torch.float32 for value in rope)
    actual, expected = apply_temporal_rope(x, rope), temporal_rope(x, times)
    assert torch.equal(actual, expected)
    assert torch.equal(
        torch.autograd.grad(actual.square().sum(), x)[0],
        torch.autograd.grad(expected.square().sum(), x)[0],
    )


def test_mot_reuses_layout_and_rope_without_mask_discovery(monkeypatch):
    block = MoTBlock(
        16,
        2,
        modalities=("image", "proprio"),
        attention_backend="flash4",
        attention_dtype="bfloat16",
    )
    streams = [
        torch.randn(2, 5, 16, requires_grad=True),
        torch.randn(2, 3, 16, requires_grad=True),
    ]
    times = [torch.randn(2, 5), torch.randn(2, 3)]
    valid = [
        torch.tensor([[1, 0, 1, 1, 1], [1, 1, 1, 1, 1]], dtype=torch.bool),
        torch.tensor([[0, 1, 0], [1, 0, 1]], dtype=torch.bool),
    ]
    joined = torch.cat(valid, 1)
    layout = prepare_attention_layout(joined, query_valid=joined)
    ropes = tuple(prepare_temporal_rope(t, 8) for t in times)
    reference = copy.deepcopy(block)
    reference.attention_backend = "sdpa"
    monkeypatch.setattr(
        "egomimic.models.world_model.load_flash_attention_kernels",
        lambda *args: _kernels(),
    )
    monkeypatch.setattr(
        torch.Tensor, "nonzero", lambda *args, **kwargs: pytest.fail("repeated packing")
    )
    actual = block(streams, times, valid, attention_layout=layout, rope=ropes)
    expected = reference(streams, times, valid)
    for a, e, mask in zip(actual, expected, valid):
        torch.testing.assert_close(a, e, atol=0.01, rtol=0.02)
        assert a[~mask].count_nonzero() == 0
    sum(x.square().sum() for x in actual).backward()
    for x, mask in zip(streams, valid):
        assert torch.isfinite(x.grad).all() and x.grad[~mask].count_nonzero() == 0


def test_query_chunks_preserve_generic_batched_nonprefix_mask_semantics(monkeypatch):
    torch.manual_seed(53)
    full = MoTBlock(
        16,
        2,
        modalities=("image",),
        attention_backend="flash4",
        attention_dtype="bfloat16",
    )
    chunked = copy.deepcopy(full)
    chunked.attention_query_chunk_size = 3
    monkeypatch.setattr(
        "egomimic.models.world_model.load_flash_attention_kernels",
        lambda *args: _kernels(),
    )
    values = torch.randn(2, 7, 16, requires_grad=True)
    times = torch.randn(2, 7)
    valid = torch.tensor(
        [[1, 0, 1, 1, 0, 1, 1], [1, 1, 1, 0, 1, 1, 0]], dtype=torch.bool
    )
    layout = prepare_attention_layout(valid, query_valid=valid)
    (a,) = full([values], [times], [valid], attention_layout=layout)
    (b,) = chunked([values], [times], [valid], attention_layout=layout)
    torch.testing.assert_close(a, b, rtol=0.02, atol=0.008)
    (ga,) = torch.autograd.grad(a.square().sum(), values)
    (gb,) = torch.autograd.grad(b.square().sum(), values)
    torch.testing.assert_close(ga, gb, rtol=0.03, atol=0.015)


@pytest.mark.parametrize("backend", ["flash3", "flash4"])
def test_missing_cuda_never_falls_back(backend):
    with pytest.raises(RuntimeError, match="CUDA"):
        load_flash_attention_kernels(backend, torch.device("cpu"))


@pytest.mark.parametrize("legacy", [False, True])
def test_flash3_import_uses_official_namespace_or_legacy_alias(monkeypatch, legacy):
    from types import SimpleNamespace

    calls = []

    def load(name):
        calls.append(name)
        if legacy and name == "flash_attn_3.flash_attn_interface":
            raise ModuleNotFoundError("legacy install", name="flash_attn_3")
        return SimpleNamespace(flash_attn_func=_dense, flash_attn_varlen_func=_varlen)

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (9, 0))
    monkeypatch.setattr(
        "egomimic.models.attention_execution.importlib.import_module", load
    )
    load_flash_attention_kernels.cache_clear()
    try:
        result = load_flash_attention_kernels("flash3", torch.device("cuda:0"))
        assert result.backend == "flash3"
        expected = ["flash_attn_3.flash_attn_interface"]
        assert calls == expected + (["flash_attn_interface"] if legacy else [])
    finally:
        load_flash_attention_kernels.cache_clear()


def test_installed_backend_missing_dependency_does_not_try_alias(monkeypatch):
    calls = []

    def load(name):
        calls.append(name)
        raise ModuleNotFoundError(
            "broken installed extension", name="flash_attn_3_cuda"
        )

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (9, 0))
    monkeypatch.setattr(
        "egomimic.models.attention_execution.importlib.import_module", load
    )
    load_flash_attention_kernels.cache_clear()
    try:
        with pytest.raises(RuntimeError, match="no backend fallback"):
            load_flash_attention_kernels("flash3", torch.device("cuda:0"))
        assert calls == ["flash_attn_3.flash_attn_interface"]
    finally:
        load_flash_attention_kernels.cache_clear()


def test_regional_compile_preserves_checkpoint_keys_and_backward():
    torch.manual_seed(19)
    eager = MoTBlock(16, 2, modalities=("image",))
    compiled = copy.deepcopy(eager)
    compiled.configure_compilation(mode="default", dynamic=True, backend="eager")
    assert list(eager.state_dict()) == list(compiled.state_dict())
    for n in (5, 8):
        x = torch.randn(2, n, 16, requires_grad=True)
        times = torch.randn(2, n)
        valid = torch.ones(2, n, dtype=torch.bool)
        rope = (prepare_temporal_rope(times, 8),)
        (a,) = eager([x], [times], [valid], rope=rope)
        (b,) = compiled([x], [times], [valid], rope=rope)
        assert torch.equal(a, b)
        assert torch.equal(
            torch.autograd.grad(a.square().sum(), x)[0],
            torch.autograd.grad(b.square().sum(), x)[0],
        )
    compiled.load_state_dict(eager.state_dict(), strict=True)


def test_compile_fails_instead_of_falling_back(monkeypatch):
    import torch._dynamo

    with torch._dynamo.config.patch(suppress_errors=True):
        with pytest.raises(RuntimeError, match="no eager fallback"):
            compile_region(lambda x: x + 1, mode="default")
    with pytest.raises(ValueError, match="compile mode"):
        compile_region(lambda x: x, mode="invalid")


def test_compiled_backward_autocast_policy_covers_lazy_invocation_and_restores(
    monkeypatch,
):
    from contextlib import contextmanager
    from types import SimpleNamespace

    import egomimic.models.compilation as compilation

    config = SimpleNamespace(backward_pass_autocast="same_as_forward")
    calls = []

    @contextmanager
    def patch(**settings):
        previous = config.backward_pass_autocast
        config.backward_pass_autocast = settings["backward_pass_autocast"]
        try:
            yield
        finally:
            config.backward_pass_autocast = previous

    config.patch = patch
    monkeypatch.setattr(compilation, "_aot_config", config)

    def lazy_compile(function, **options):
        assert config.backward_pass_autocast == "same_as_forward"
        assert options["fullgraph"]

        def lazy_call(value):
            calls.append(config.backward_pass_autocast)
            assert config.backward_pass_autocast == "off"
            if value == "fail":
                raise RuntimeError("tracing failed")
            return function(value)

        return lazy_call

    monkeypatch.setattr(torch, "compile", lazy_compile)
    execute = compile_region(lambda value: value + 1, mode="default")
    assert execute(2) == 3
    assert config.backward_pass_autocast == "same_as_forward"
    with pytest.raises(RuntimeError, match="tracing failed"):
        execute("fail")
    assert config.backward_pass_autocast == "same_as_forward"
    assert calls == ["off", "off"]


def test_missing_aot_autocast_policy_fails_closed_only_for_compiled_mixed_mode(
    monkeypatch,
):
    from types import SimpleNamespace

    import egomimic.models.compilation as compilation

    monkeypatch.setattr(compilation, "_aot_config", SimpleNamespace())
    monkeypatch.setattr(torch, "compile", lambda function, **_: function)

    def function(value):
        return value + 1

    execute = compile_region(function, mode="default")
    assert execute(torch.ones(1)).item() == 2
    with torch.autocast("cpu", dtype=torch.bfloat16):
        with pytest.raises(RuntimeError, match="no unsafe mixed-precision fallback"):
            execute(torch.ones(1))
        assert compile_region(function, mode="none") is function
        assert function(torch.ones(1)).item() == 2


def test_aot_eager_bf16_forward_backward_matches_standard_amp():
    from torch._functorch import config

    if not hasattr(config, "backward_pass_autocast"):
        pytest.skip("This PyTorch lacks the required compiled AMP backward policy")
    torch.manual_seed(19)
    eager = MoTBlock(32, 4, modalities=("image",), attention_dtype="bfloat16")
    compiled = copy.deepcopy(eager)
    compiled.configure_compilation(mode="default", backend="aot_eager")
    times = torch.randn(2, 7)
    valid = torch.ones(2, 7, dtype=torch.bool)
    rope = (prepare_temporal_rope(times, 8),)
    x = torch.randn(2, 7, 32, requires_grad=True)
    copied = x.detach().clone().requires_grad_()
    before = config.backward_pass_autocast
    with torch.autocast("cpu", dtype=torch.bfloat16):
        (expected,) = eager([x], [times], [valid], rope=rope)
        (actual,) = compiled([copied], [times], [valid], rope=rope)
        expected_loss = expected.float().square().mean()
        actual_loss = actual.float().square().mean()
    assert config.backward_pass_autocast == before
    expected_loss.backward()
    actual_loss.backward()
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.01)
    for a, b in [(copied.grad, x.grad)] + [
        (a.grad, b.grad) for a, b in zip(compiled.parameters(), eager.parameters())
    ]:
        assert a.dtype == b.dtype == torch.float32
        assert torch.isfinite(a).all()
        torch.testing.assert_close(a, b, rtol=0.02, atol=0.002)
        assert (a - b).norm() / b.norm().clamp_min(1e-8) < 0.02


@pytest.mark.skipif(
    os.environ.get("EGOMIMIC_RUN_FUSED_ATTENTION_TESTS") != "1",
    reason="Explicit opt-in real CUDA attention proof",
)
@pytest.mark.parametrize("backend", ["flash3", "flash4"])
@pytest.mark.parametrize("dense", [True, False])
def test_real_cuda_batched_attention_matches_independent_gradients(backend, dense):
    if not torch.cuda.is_available():
        pytest.fail("GPU proof requested without CUDA")
    device = torch.device("cuda", torch.cuda.current_device())
    kernels = load_flash_attention_kernels(backend, device)
    torch.manual_seed(71)
    q, k, v = [
        torch.randn(
            3, 6, 97, 128, device=device, dtype=torch.bfloat16, requires_grad=True
        )
        for _ in range(3)
    ]
    valid = torch.ones(3, 97, device=device, dtype=torch.bool)
    if not dense:
        valid[0, 1::3] = False
        valid[1, 1::2] = False
    layout = prepare_attention_layout(valid, query_valid=valid)
    actual = execute_flash_attention(q, k, v, layout, kernels)
    reference = F.scaled_dot_product_attention(
        q.float(), k.float(), v.float(), attn_mask=valid[:, None, None]
    ).masked_fill(~valid[:, None, :, None], 0)
    torch.testing.assert_close(actual.float(), reference, rtol=0.03, atol=0.02)
    ga = torch.autograd.grad(
        actual.float().square().sum(), (q, k, v), retain_graph=True
    )
    gr = torch.autograd.grad(reference.square().sum(), (q, k, v))
    for a, r in zip(ga, gr):
        assert torch.isfinite(a).all()
        assert (a.float() - r.float()).norm() / r.float().norm().clamp_min(1e-8) < 0.04
    for i in range(3):
        one = prepare_attention_layout(valid[i : i + 1], query_valid=valid[i : i + 1])
        independent = execute_flash_attention(
            q[i : i + 1], k[i : i + 1], v[i : i + 1], one, kernels
        )
        torch.testing.assert_close(actual[i : i + 1], independent, rtol=0.03, atol=0.02)
        independent_grads = torch.autograd.grad(
            independent.float().square().sum(), (q, k, v)
        )
        for full, single in zip(ga, independent_grads):
            relative = (full[i].float() - single[i].float()).norm()
            assert relative / full[i].float().norm().clamp_min(1e-8) < 0.04
