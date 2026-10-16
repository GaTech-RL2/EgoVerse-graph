"""Opt-in, read-only checkpoint equivalence/timing test for exact endpoint reuse.

Set ACTION_FLOW_JACOBIAN_PROFILE_REQUEST to a JSON request outside the checkout.
Ordinary CPU suites skip the GPU benchmark. No optimizer or policy sampler runs.
"""

from __future__ import annotations

import hashlib
import json
import os
import statistics
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

import egomimic.eval.action_flow_diagnostic_forward as owner


def _digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def load_profile_components(request, device="cpu"):
    """Strict component restoration; also used by the scheduled CPU preflight."""
    assert _digest(request["checkpoint"]) == request["checkpoint_sha256"]
    cfg = OmegaConf.load(request["config"])
    state = torch.load(
        request["checkpoint"], map_location="cpu", mmap=True, weights_only=False
    )
    assert state["global_step"] == request["checkpoint_step"]
    assert state["hyper_parameters"]["config_tree"]["model"] == OmegaConf.to_container(
        cfg.model, resolve=True
    )
    result = {}
    for name, spec in request["components"].items():
        module = instantiate(OmegaConf.select(cfg, spec["config_key"]))
        prefix = spec["state_prefix"]
        weights = {
            k[len(prefix) :]: v
            for k, v in state["state_dict"].items()
            if k.startswith(prefix)
        }
        assert weights, f"Checkpoint component prefix absent: {prefix}"
        module.load_state_dict(weights, strict=True)
        module.eval().float().to(device)
        for parameter in module.parameters():
            parameter.requires_grad_(False)
        result[name] = module
    assert int(cfg.model.num_latent_tokens) == request["latent_shape"][0]
    assert int(cfg.model.latent_dim) == request["latent_shape"][1]
    assert int(cfg.model.action_horizon) == request["action_shape"][0]
    assert int(cfg.model.action_dim) == request["action_shape"][1]
    return result


@pytest.mark.skipif(
    not os.environ.get("ACTION_FLOW_JACOBIAN_PROFILE_REQUEST"),
    reason="Opt-in checkpoint GPU benchmark",
)
def test_checkpoint_endpoint_reuse_equivalence_and_timing():
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    request_path = Path(os.environ["ACTION_FLOW_JACOBIAN_PROFILE_REQUEST"])
    request = json.loads(request_path.read_text())
    request_sha = _digest(request_path)
    out = Path(request["output_root"])
    out.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    modules = load_profile_components(request, "cuda")
    encoder, decoder = modules["encoder"], modules["decoder"]
    assert _digest(request["targets"]) == request["targets_sha256"]
    with np.load(request["targets"], allow_pickle=False) as artifact:
        target = (
            torch.from_numpy(artifact["target"][request["target_indices"]].copy())
            .float()
            .cuda()
        )
    assert target.shape[1:] == tuple(request["action_shape"])
    torch.manual_seed(request["seed"])
    with torch.no_grad():
        clean = encoder(target)
    assert clean.shape[1:] == tuple(request["latent_shape"])
    generator = torch.Generator(device=clean.device).manual_seed(request["seed"])
    noise = torch.randn(
        clean.shape, device=clean.device, dtype=clean.dtype, generator=generator
    )
    fixed = torch.stack(
        [(1.0 - level) * clean + level * noise for level in request["noise_levels"]]
    )
    count = request["jacobian_samples"]
    assert count <= len(target)
    method = "forward_math_chunk4"
    native = owner._decoder_singular_values
    parameters = {k: v.detach().cpu().clone() for k, v in decoder.state_dict().items()}
    rng_cpu, rng_cuda = torch.get_rng_state(), torch.cuda.get_rng_state()
    flags = [p.requires_grad for p in decoder.parameters()]

    def measure(kind):
        with patch.object(owner, "_decoder_singular_values", wraps=native) as calls:
            clean_s = owner._decoder_singular_values(
                decoder, clean, count, jacobian_method=method
            )
            noise_s = owner._decoder_singular_values(
                decoder, noise, count, jacobian_method=method
            )
            if kind == "baseline":
                fixed_s = torch.stack(
                    [
                        owner._decoder_singular_values(
                            decoder, value, count, jacobian_method=method
                        )
                        for value in fixed
                    ]
                )
            else:
                fixed_s = owner._fixed_decoder_singular_values(
                    decoder,
                    fixed,
                    clean,
                    noise,
                    clean_s,
                    noise_s,
                    count,
                    jacobian_method=method,
                )
            assert calls.call_count == (
                2 + len(fixed) if kind == "baseline" else len(fixed)
            )
        return {
            "clean": clean_s.cpu(),
            "noise": noise_s.cpu(),
            "fixed": fixed_s.cpu(),
        }, calls.call_count

    records, reference = [], None
    # Alternate order to reduce first-case/allocator bias; completed cases are reusable.
    order = [
        kind
        for repeat in range(request["repeats"])
        for kind in (
            ("baseline", "optimized") if repeat % 2 == 0 else ("optimized", "baseline")
        )
    ]
    for index, kind in enumerate(order):
        artifact = out / f"case-{index:02d}-{kind}.pt"
        if artifact.exists():
            payload = torch.load(artifact, map_location="cpu", weights_only=False)
            assert payload["request_sha256"] == request_sha
        else:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            spectra, calls = measure(kind)
            torch.cuda.synchronize()
            payload = {
                "request_sha256": request_sha,
                "kind": kind,
                "seconds": time.perf_counter() - started,
                "full_jacobians": calls * count,
                "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
                "spectra": spectra,
            }
            temporary = artifact.with_suffix(".partial")
            torch.save(payload, temporary)
            temporary.replace(artifact)
        if reference is None:
            reference = payload["spectra"]
        maximum = 0.0
        bit_exact = True
        for key, actual in payload["spectra"].items():
            torch.testing.assert_close(actual, reference[key], rtol=1e-5, atol=1e-6)
            maximum = max(maximum, float((actual - reference[key]).abs().max()))
            bit_exact = bit_exact and torch.equal(actual, reference[key])
        record = {k: v for k, v in payload.items() if k != "spectra"}
        record.update(
            case=index, max_spectrum_absolute_difference=maximum, bit_exact=bit_exact
        )
        records.append(record)
        print("JACOBIAN_PROFILE_CASE", json.dumps(record), flush=True)
        (out / "progress.json").write_text(
            json.dumps({"status": "RUNNING", "cases": records}, indent=2)
        )

    assert torch.equal(rng_cpu, torch.get_rng_state())
    assert torch.equal(rng_cuda, torch.cuda.get_rng_state())
    assert flags == [p.requires_grad for p in decoder.parameters()]
    assert all(
        torch.equal(v.cpu(), parameters[k]) for k, v in decoder.state_dict().items()
    )
    assert all(p.grad is None for p in decoder.parameters())
    baseline = statistics.median(
        r["seconds"] for r in records if r["kind"] == "baseline"
    )
    optimized = statistics.median(
        r["seconds"] for r in records if r["kind"] == "optimized"
    )
    result = {
        "status": "PASS",
        "request_sha256": request_sha,
        "checkpoint_step": request["checkpoint_step"],
        "weights": "raw_strict_components",
        "precision": "FP32_math_SDPA_noTF32",
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "cases": records,
        "baseline_median_seconds": baseline,
        "optimized_median_seconds": optimized,
        "speedup": baseline / optimized,
        "parameter_state_unchanged": True,
        "rng_unchanged": True,
        "all_singular_values_checked": True,
    }
    (out / "RESULT.json").write_text(json.dumps(result, indent=2))
    print("JACOBIAN_PROFILE_PASS", json.dumps(result), flush=True)
