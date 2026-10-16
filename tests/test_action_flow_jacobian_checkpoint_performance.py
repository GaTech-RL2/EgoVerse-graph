"""Opt-in, read-only checkpoint equivalence/timing test for exact endpoint reuse.

Set ACTION_FLOW_JACOBIAN_PROFILE_REQUEST to a JSON request outside the checkout.
Ordinary CPU suites skip the GPU benchmark. No optimizer or policy sampler runs.
"""

from __future__ import annotations

import copy
import gc
import hashlib
import json
import os
import statistics
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

import egomimic.eval.action_flow_diagnostic_forward as owner


def prepare_training_profile(request):
    """Restore native data/model state on CPU; capture one real training batch."""
    from torch.utils.data import default_collate

    from egomimic.rldb.zarr.zarr_dataset_multi import MultiDataset

    assert _digest(request["split"]) == request["split_sha256"]
    split = json.loads(Path(request["split"]).read_text())
    for domain in ("yam", "human"):
        for part in ("train", "valid"):
            os.environ[f"{domain.upper()}_{part.upper()}_IDS_SHA256"] = split[domain][
                "summary"
            ][f"{part}_ids_sha256"]
    assert _digest(request["checkpoint"]) == request["checkpoint_sha256"]
    payload = torch.load(
        request["checkpoint"], map_location="cpu", mmap=True, weights_only=False
    )
    assert payload["global_step"] == request["checkpoint_step"]
    cfg = OmegaConf.load(request["config"])
    assert payload["hyper_parameters"]["config_tree"][
        "model"
    ] == OmegaConf.to_container(cfg.model, resolve=True)
    assert [int(cfg.model.num_latent_tokens), int(cfg.model.latent_dim)] == request[
        "latent_shape"
    ]
    assert (
        int(cfg.data.train_dataloader_params[request["source_key"]].batch_size)
        == request["batch_size"]
    )
    state = copy.deepcopy(payload["data_context"]["normalizer_state"])
    for field in ("key_types", "zarr_keys", "shapes", "norm_stats"):
        state[field] = {int(k): v for k, v in state[field].items()}
    normalizer = MultiDataset.from_state(state)
    fixture_path = Path(request["fixture"])
    if fixture_path.exists():
        if request.get("fixture_sha256"):
            assert _digest(fixture_path) == request["fixture_sha256"]
        fixture = torch.load(fixture_path, weights_only=False)
        assert fixture["checkpoint"] == request.get(
            "fixture_checkpoint", request["checkpoint"]
        )
        assert fixture["source_key"] == request["source_key"]
    else:
        dataset = instantiate(cfg.data.train_datasets[request["source_key"]])
        dataset.set_norm_stats_from(normalizer)
        indices = np.linspace(
            0, len(dataset) - 1, request["batch_size"], dtype=int
        ).tolist()
        flat = default_collate([dataset[index] for index in indices])
        fixture = {
            "checkpoint": request["checkpoint"],
            "source_key": request["source_key"],
            "indices": indices,
            "batch": flat,
        }
        torch.save(fixture, fixture_path)
    algo = instantiate(cfg.model.pipeline, device="cpu")
    algo.bind_data_context(normalizer=normalizer)
    weights = {
        k.removeprefix("nets."): v
        for k, v in payload["state_dict"].items()
        if k.startswith("nets.")
    }
    algo.nets.load_state_dict(weights, strict=True)
    return cfg, payload, algo, fixture


def _cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: _cpu_tree(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_cpu_tree(v) for v in value)
    return copy.deepcopy(value)


def _compare_training_tree(actual, expected):
    """Compare every tensor; retain exact mismatch counts and largest errors."""
    mismatches, tensor_count = [], 0

    def visit(a, b, path):
        nonlocal tensor_count
        if torch.is_tensor(a):
            tensor_count += 1
            assert (
                torch.is_tensor(b) and a.shape == b.shape and a.dtype == b.dtype
            ), path
            if not torch.equal(a, b):
                diff = (a.double() - b.double()).abs()
                mismatches.append(
                    {
                        "path": path,
                        "max_abs": float(diff.max()),
                        "different_elements": int(torch.count_nonzero(diff)),
                        "relative_rms": float(
                            torch.linalg.vector_norm(diff)
                            / torch.linalg.vector_norm(b.double()).clamp_min(1e-30)
                        ),
                    }
                )
        elif isinstance(a, dict):
            assert a.keys() == b.keys(), path
            for k in a:
                visit(a[k], b[k], path + "/" + str(k))
        elif isinstance(a, (tuple, list)):
            assert len(a) == len(b), path
            for k, (x, y) in enumerate(zip(a, b)):
                visit(x, y, path + "/" + str(k))
        elif torch.is_tensor(b) or a != b:
            mismatches.append(
                {"path": path, "actual": str(a), "expected": str(b), "max_abs": 0.0}
            )

    visit(actual, expected, "")
    return {
        "bit_exact": not mismatches,
        "tensor_count": tensor_count,
        "mismatch_count": len(mismatches),
        "largest_errors": sorted(mismatches, key=lambda x: x["max_abs"], reverse=True)[
            :12
        ],
    }


@pytest.mark.skipif(
    not os.environ.get("ACTION_FLOW_TRAINING_PROFILE_REQUEST"),
    reason="Opt-in native checkpoint training test",
)
def test_checkpoint_training_execution_equivalence_and_timing():
    """Test existing execution options without changing/adopting model weights."""
    from egomimic.utils.libero_dit_half import LiberoDiTHalf

    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    request = json.loads(
        Path(os.environ["ACTION_FLOW_TRAINING_PROFILE_REQUEST"]).read_text()
    )
    out = Path(request["output_root"])
    out.mkdir(parents=True, exist_ok=True)
    cfg, payload, initial, fixture = prepare_training_profile(request)
    assert str(cfg.trainer.precision) == "bf16-mixed"
    baseline, records = [], []
    initial_state = _cpu_tree(initial.nets.state_dict())
    del initial
    for case_index, variant in enumerate(request["variants"]):
        algo = instantiate(cfg.model.pipeline, device="cpu")
        algo.nets.load_state_dict(initial_state, strict=True)
        holder = torch.nn.Module()
        holder.nets = algo.nets
        holder.cuda().train()
        algo.device = torch.device("cuda")
        optimizer = instantiate(
            cfg.model.optimizer, named_params=tuple(holder.named_parameters())
        )()
        scheduler = instantiate(cfg.model.scheduler, optimizer=optimizer)()
        optimizer.load_state_dict(copy.deepcopy(payload["optimizer_states"][0]))
        scheduler.load_state_dict(copy.deepcopy(payload["lr_schedulers"][0]))
        ema = instantiate(cfg.callbacks.ema)
        ema.on_load_checkpoint(None, holder, payload)
        ema.on_fit_start(SimpleNamespace(global_step=payload["global_step"]), holder)
        policy = LiberoDiTHalf(source_key=request["source_key"])
        policy.on_fit_start(None, SimpleNamespace(model=algo))
        encoder, field, decoder = (
            algo.pipeline.stages[3].encoder,
            algo.pipeline.stages[5],
            algo.pipeline.stages[6].decoder,
        )
        if variant == "no_dit_recomputation":
            encoder.backbone.gradient_checkpointing = False
            field.field.backbone.gradient_checkpointing = False
        elif variant == "no_reconstruction_recomputation":
            decoder.gradient_checkpointing = False
        elif variant == "shared_field_prediction":
            field.fm_field_execution = "shared"
        else:
            assert variant == "baseline"
        batch = algo.process_batch_for_training(
            {request["source_key"]: fixture["batch"]}
        )
        timings, comparisons = [], []
        failure = None
        try:
            for step in range(request["steps"]):
                torch.manual_seed(request["seed"] + step)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                started = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    prediction = algo.forward_training(batch)
                    loss = algo.compute_losses(prediction, batch)["loss"]
                assert bool(torch.isfinite(loss))
                loss.backward()
                torch.cuda.synchronize()
                backward_done = time.perf_counter()
                gradients = _cpu_tree(
                    {name: p.grad for name, p in holder.named_parameters()}
                )
                components = _cpu_tree(
                    {
                        k: v
                        for k, v in prediction[request["source_key"]].items()
                        if k.startswith("log/action_flow_") or k == "loss/action_flow"
                    }
                )
                # Copying proof tensors is deliberately outside measured GPU work.
                optimizer_start = time.perf_counter()
                clip = float(cfg.trainer.get("gradient_clip_val", 0) or 0)
                if clip:
                    assert cfg.trainer.get("gradient_clip_algorithm", "norm") == "norm"
                    torch.nn.utils.clip_grad_norm_(holder.parameters(), clip)
                optimizer.step()
                scheduler.step()
                ema._update(holder)
                torch.cuda.synchronize()
                ended = time.perf_counter()
                timings.append(
                    {
                        "step": step,
                        "forward_backward_seconds": backward_done - started,
                        "optimizer_scheduler_ema_seconds": ended - optimizer_start,
                        "seconds": backward_done - started + ended - optimizer_start,
                        "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
                    }
                )
                actual = {
                    "gradients": gradients,
                    "components": components,
                    "state": _cpu_tree(holder.state_dict()),
                    "optimizer": _cpu_tree(optimizer.state_dict()),
                    "scheduler": _cpu_tree(scheduler.state_dict()),
                    "ema": _cpu_tree(ema._shadow),
                    "rng_cpu": torch.get_rng_state(),
                    "rng_cuda": torch.cuda.get_rng_state(),
                }
                if case_index == 0:
                    baseline.append(actual)
                else:
                    comparisons.append(
                        {
                            section: _compare_training_tree(
                                value, baseline[step][section]
                            )
                            for section, value in actual.items()
                        }
                    )
                print(
                    "TRAINING_PROFILE_STEP",
                    variant,
                    json.dumps(timings[-1]),
                    flush=True,
                )
                del prediction, loss, actual, gradients, components
        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            failure = {"type": type(exc).__name__, "message": str(exc)}
            if case_index == 0:
                raise
        finally:
            policy.on_fit_end(None, None)
        record = {
            "variant": variant,
            "timings": timings,
            "comparisons": comparisons,
            "failure": failure,
            "bit_exact": failure is None
            and all(all(s["bit_exact"] for s in step.values()) for step in comparisons),
            "median_seconds": statistics.median(t["seconds"] for t in timings[1:])
            if len(timings) > 1
            else None,
        }
        records.append(record)
        (out / "TRAINING_PROGRESS.json").write_text(json.dumps(records, indent=2))
        del (
            algo,
            holder,
            optimizer,
            scheduler,
            ema,
            batch,
            policy,
            encoder,
            field,
            decoder,
        )
        gc.collect()
        torch.cuda.empty_cache()
    result = {
        "status": "COMPLETE",
        "checkpoint_step": request["checkpoint_step"],
        "batch_size": request["batch_size"],
        "latent_shape": request["latent_shape"],
        "precision": "bf16-mixed_native_FP32_decoder_JVP",
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "records": records,
        "scope": "native full pipeline forward/backward/optimizer/scheduler/EMA; frozen real training batch; excludes data-loader, telemetry and proof-copy overhead; no weights adopted",
    }
    (out / "TRAINING_RESULT.json").write_text(json.dumps(result, indent=2))


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
