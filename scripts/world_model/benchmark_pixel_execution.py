"""Bounded single-GPU benchmark of exact image-only pixel execution candidates.

Inputs are synthetic and device-resident. This measures neither a data loader nor
NCCL, and does not replace real-data train/validation/checkpoint smoke tests.
Every candidate runs in a fresh process; missing kernels, OOMs, correctness
failures and compilation failures are reported, never replaced by a fallback.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import signal
import statistics
import subprocess
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PARAMETERS = 88_178_688
BACKENDS = ("flash4", "flash3", "sdpa")
SCENARIOS = ("fixed16", "mixed", "worst128")
COMPILE_MODES = ("none", "default", "reduce-overhead", "max-autotune")
TOKEN_EXECUTIONS = ("padded", "packed_valid")
MIXED_FRAMES = (2, 16, 64, 128)


class CorrectnessError(RuntimeError):
    pass


class UnsupportedBackend(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise CorrectnessError(message)


def write_once(path, value):
    with Path(path).open("x") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def frame_schedule(scenario, batch_size):
    """A complete mixed cycle gives every batch size the same frame histogram."""
    if scenario == "fixed16":
        return [(16,) * batch_size]
    if scenario == "worst128":
        return [(128,) * batch_size]
    if scenario != "mixed":
        raise ValueError(f"Unknown scenario: {scenario}")
    cycle = len(MIXED_FRAMES) // math.gcd(batch_size, len(MIXED_FRAMES))
    return [tuple(MIXED_FRAMES[(step * batch_size + row) % len(MIXED_FRAMES)]
                  for row in range(batch_size)) for step in range(cycle)]


def complete_cycles(steps, schedule):
    return math.ceil(steps / len(schedule)) * len(schedule)


def model_arguments(backend, scenario, *, compile_mode="none", compile_dynamic=False,
                    token_execution="padded"):
    if backend not in BACKENDS or scenario not in SCENARIOS or token_execution not in TOKEN_EXECUTIONS:
        raise ValueError("Unknown execution candidate")
    return dict(
        objective="visual_only", phase_a_condition_on_proprio=False,
        pose_dims=None, action_dims=None, image_dim=768, width=768,
        depth=8, heads=6, mlp_ratio=4, max_cameras=2,
        max_tokens=3200 if scenario == "fixed16" else 25600,
        time_mode="discrete" if scenario == "fixed16" else "continuous",
        use_camera_embedding=True, use_modality_embedding=False,
        use_embodiment_embedding=False, allow_patch_compression=False,
        use_phase_embedding=False, noise_conditioning="adaln_zero",
        position_encoding="spatiotemporal_rope", rope_axis_dims=(64, 32, 32),
        spatial_grid_size=(10, 10),
        attention_backend=backend, attention_dtype="bfloat16", strict_fp32=False,
        attention_query_chunk_size=0, activation_checkpointing=False,
        loss_reduction="mean_per_example", compile_mode=compile_mode,
        compile_dynamic=compile_dynamic, compile_backend="inductor", token_execution=token_execution,
    )


def make_batch(frames, *, seed, device, image_size=160, patch_size=16, cameras=2,
               frame_seconds=0.05):
    """True RGB patch geometry; only batch padding is invalid."""
    if not frames or any(type(length) is not int or length < 2 for length in frames):
        raise ValueError("Every synthetic episode needs at least two frames")
    if image_size % patch_size or image_size < patch_size:
        raise ValueError("Image size must be divisible by patch size")
    side = image_size // patch_size
    patches, dimension = side * side, patch_size * patch_size * 3
    per_frame, tokens = cameras * patches, max(frames) * cameras * patches
    generator = torch.Generator().manual_seed(seed)
    value = torch.zeros(len(frames), tokens, dimension, dtype=torch.float32)
    valid = torch.zeros(len(frames), tokens, dtype=torch.bool)
    observed = torch.zeros_like(valid)
    clock = torch.zeros(len(frames), tokens, dtype=torch.float32)
    index = torch.zeros_like(clock)
    camera = torch.arange(cameras).repeat_interleave(patches).repeat(max(frames))
    axis = (torch.arange(side, dtype=torch.float32) + 0.5) * 2 / side - 1
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    xy = torch.stack((x.flatten(), y.flatten()), -1).repeat(max(frames) * cameras, 1)
    for row, length in enumerate(frames):
        size = length * per_frame
        value[row, :size] = torch.rand(size, dimension, generator=generator) * 2 - 1
        valid[row, :size] = True
        observed[row, :max(1, length // 2) * per_frame] = True
        indices = torch.arange(length, dtype=torch.float32).repeat_interleave(per_frame)
        index[row, :size] = indices
        clock[row, :size] = indices * frame_seconds
    return {"image": {key: tensor.to(device) for key, tensor in dict(
        value=value, valid=valid, observed=observed, time=clock, index=index,
        camera=camera[None].expand(len(frames), -1).clone(),
        xy=xy[None].expand(len(frames), -1, -1).clone(),
    ).items()}}


def single_example(batch, row):
    indices = batch["image"]["valid"][row].nonzero(as_tuple=True)[0]
    return {"image": {key: value[row:row + 1].index_select(1, indices)
                      for key, value in batch["image"].items()}}


def fixed_corruption(batch, seed):
    image = batch["image"]
    generator = torch.Generator(device=image["value"].device).manual_seed(seed)
    noise = torch.randn(image["value"].shape, dtype=torch.float32,
                        device=image["value"].device, generator=generator)
    levels = (0.15 + 0.7 * torch.remainder(image["index"], 7) / 6).masked_fill(
        image["observed"] | ~image["valid"], 0
    )
    noisy = ((1 - levels[..., None]) * image["value"] + levels[..., None] * noise)
    return noisy.masked_fill(~image["valid"][..., None], 0), levels


def example_mean_loss(prediction, batch):
    image = batch["image"]
    mask = image["valid"] & ~image["observed"]
    counts = mask.sum(1)
    require(bool((counts > 0).all()), "Every example needs a query target")
    error = (prediction.float() - image["value"].float()).square().mean(-1)
    return (error.masked_fill(~mask, 0).sum(1) / counts).mean()


@contextmanager
def execution_context(backend, device):
    # The SDPA candidate explicitly enables only its fused Flash kernel.
    # A backend that cannot handle the mask/shape raises instead of using math.
    with torch.autocast(device.type, dtype=torch.bfloat16):
        if backend == "sdpa":
            from torch.nn.attention import SDPBackend, sdpa_kernel
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                yield
        else:
            yield


def relative_error(actual, expected):
    error = (actual.float() - expected.float()).double()
    return dict(squared_error=float(error.square().sum()),
                squared_reference=float(expected.double().square().sum()),
                max_abs=float(error.abs().max()))


@contextmanager
def nontrivial_conditioning_probe(model, seed):
    """Exercise attention and noise conditioning beyond DiT's zero initialization.

    Only correctness probes use these deterministic nonzero weights. Restore
    every changed parameter and RNG before timing actual zero-initialized
    training; otherwise zero gates/head would make parity checks vacuous.
    """
    if getattr(model, "noise_conditioning", "additive") != "adaln_zero":
        yield {"nonzero_adaln_probe": False}
        return
    selected = {name: parameter for name, parameter in model.named_parameters()
                if name.startswith(("image_out.", "final_adaln."))
                or ".adaln_image." in name}
    require(selected and any("adaln_image" in name for name in selected),
            "AdaLN correctness probe could not find modulation parameters")
    original = {name: parameter.detach().clone() for name, parameter in selected.items()}
    device = next(model.parameters()).device
    devices = [device.index] if device.type == "cuda" else []
    try:
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            # Seed only the probed device; do not disturb other CUDA generators.
            torch.random.default_generator.manual_seed(seed)
            if devices:
                torch.cuda.default_generators[device.index].manual_seed(seed)
            for parameter in selected.values():
                parameter.normal_(mean=0.0, std=0.02)
        yield {"nonzero_adaln_probe": True, "seed": seed,
               "restores_training_initialization": True}
    finally:
        with torch.no_grad():
            for name, parameter in selected.items():
                parameter.copy_(original[name])
        model.zero_grad(set_to_none=True)


def check_outputs(model, batch, *, backend, seed, tolerance, gradients=False):
    """Compare a batch to independent, unpadded B1 calls at fixed noise levels."""
    device = batch["image"]["value"].device
    noisy, levels = fixed_corruption(batch, seed)
    model.zero_grad(set_to_none=True)
    with execution_context(backend, device):
        prediction = model(batch, "libero_gripper", noisy, levels, phase="A")["image"]
        loss = example_mean_loss(prediction, batch)
    require(loss.dtype == torch.float32 and torch.isfinite(loss).item(), "Nonfinite/non-FP32 probe loss")
    actual = prediction.detach()
    batched_gradients = None
    if gradients:
        loss.backward()
        require(all(p.grad is not None for p in model.parameters()), "Unused trainable parameter")
        batched_gradients = {name: p.grad.detach().cpu().clone() for name, p in model.named_parameters()}
    del prediction, loss
    model.zero_grad(set_to_none=True)
    errors = []
    batch_size = actual.shape[0]
    for row in range(batch_size):
        selected = batch["image"]["valid"][row].nonzero(as_tuple=True)[0]
        independent = single_example(batch, row)
        with execution_context(backend, device):
            predicted = model(independent, "libero_gripper",
                              noisy[row:row + 1].index_select(1, selected),
                              levels[row:row + 1].index_select(1, selected), phase="A")["image"]
            individual_loss = example_mean_loss(predicted, independent) / batch_size
        errors.append(relative_error(actual[row:row + 1].index_select(1, selected), predicted.detach()))
        if gradients:
            individual_loss.backward()
        del predicted, individual_loss
    output_relative = math.sqrt(sum(v["squared_error"] for v in errors) /
                                max(sum(v["squared_reference"] for v in errors), 1e-30))
    require(math.isfinite(output_relative) and output_relative <= tolerance,
            f"Batched output differs from independent B1: relative L2={output_relative}")
    report = {"examples": batch_size, "output_relative_l2": output_relative,
              "output_max_abs": max(v["max_abs"] for v in errors)}
    if gradients:
        error_sum, reference_sum = 0.0, 0.0
        for name, parameter in model.named_parameters():
            require(parameter.grad is not None and parameter.grad.dtype == torch.float32,
                    "Missing/non-FP32 reference gradient")
            measured = relative_error(batched_gradients[name], parameter.grad.detach().cpu())
            error_sum += measured["squared_error"]
            reference_sum += measured["squared_reference"]
        gradient_relative = math.sqrt(error_sum / max(reference_sum, 1e-30))
        require(math.isfinite(gradient_relative) and gradient_relative <= tolerance,
                f"Batched gradient differs from mean independent B1: relative L2={gradient_relative}")
        report["gradient_relative_l2"] = gradient_relative
    model.zero_grad(set_to_none=True)
    # Perturb every other example, preserving row zero and its noise exactly.
    if batch_size > 1:
        changed = {"image": {key: value.clone() for key, value in batch["image"].items()}}
        changed["image"]["value"][1:] *= -0.7
        perturbed = noisy.clone()
        perturbed[1:] *= -0.7
        with torch.no_grad(), execution_context(backend, device):
            # Inductor may select different kernels for grad-enabled and
            # no-grad execution. Compare intervention/control under the same
            # mode and graph; separately bound the mode's BF16 rounding drift.
            control = model(batch, "libero_gripper", noisy, levels, phase="A")["image"][0].clone()
            independent = model(changed, "libero_gripper", perturbed, levels, phase="A")["image"][0]
        mode_error = relative_error(control, actual[0])
        mode_relative = math.sqrt(mode_error["squared_error"] / max(mode_error["squared_reference"], 1e-30))
        require(math.isfinite(mode_relative) and mode_relative <= tolerance,
                f"Grad/no-grad output differs beyond BF16 tolerance: relative L2={mode_relative}")
        report["grad_vs_no_grad_relative_l2"] = mode_relative
        error = relative_error(independent, control)
        leak = math.sqrt(error["squared_error"] / max(error["squared_reference"], 1e-30))
        require(math.isfinite(leak) and leak <= 1e-6,
                f"Cross-example attention leakage detected: same-mode relative L2={leak}")
        report["cross_example_relative_l2"] = leak
    else:
        report["cross_example_check"] = "not_applicable_batch_one"
    return report


def check_eager_compiled(eager, compiled, batch, *, backend, seed, tolerance):
    """Require state-identical eager/compiled AMP forward and FP32 backward parity."""
    require(eager.token_execution == compiled.token_execution,
            "Eager/compiled token execution differs")
    reference_state, candidate_state = eager.state_dict(), compiled.state_dict()
    require(reference_state.keys() == candidate_state.keys(), "Eager/compiled state keys differ")
    require(all(torch.equal(value, candidate_state[name]) for name, value in reference_state.items()),
            "Eager/compiled initial state differs")
    require(all(p.dtype == torch.float32 for model in (eager, compiled) for p in model.parameters()),
            "Eager/compiled stored parameters must be FP32")
    noisy, levels = fixed_corruption(batch, seed)
    device = batch["image"]["value"].device

    def forward_backward(model):
        model.zero_grad(set_to_none=True)
        with execution_context(backend, device):
            prediction = model(batch, "libero_gripper", noisy, levels, phase="A")["image"]
            loss = example_mean_loss(prediction, batch)
        require(loss.dtype == torch.float32 and torch.isfinite(loss).item(),
                "Eager/compiled loss must be finite FP32")
        # Match standard mixed precision: backward is outside forward autocast.
        loss.backward()
        require(all(p.grad is not None and p.grad.dtype == torch.float32
                    and torch.isfinite(p.grad).all() for p in model.parameters()),
                "Eager/compiled gradient must be present and finite FP32")
        return prediction.detach().cpu(), float(loss.detach())

    try:
        reference_output, reference_loss = forward_backward(eager)
        reference_gradients = {name: p.grad.detach().cpu().clone() for name, p in eager.named_parameters()}
        eager.zero_grad(set_to_none=True)
        output, loss = forward_backward(compiled)
        measured = relative_error(output, reference_output)
        output_relative = math.sqrt(measured["squared_error"] / max(measured["squared_reference"], 1e-30))
        loss_relative = abs(loss - reference_loss) / max(abs(reference_loss), 1e-30)
        require(math.isfinite(output_relative) and output_relative <= tolerance,
                f"Compiled output differs from eager: relative L2={output_relative}")
        require(math.isfinite(loss_relative) and loss_relative <= tolerance,
                f"Compiled loss differs from eager: relative error={loss_relative}")
        errors = [relative_error(p.grad.detach().cpu(), reference_gradients[name])
                  for name, p in compiled.named_parameters()]
        gradient_relative = math.sqrt(sum(value["squared_error"] for value in errors) /
                                      max(sum(value["squared_reference"] for value in errors), 1e-30))
        require(math.isfinite(gradient_relative) and gradient_relative <= tolerance,
                f"Compiled gradient differs from eager: relative L2={gradient_relative}")
        return {"output_relative_l2": output_relative, "output_max_abs": measured["max_abs"],
                "loss_relative_error": loss_relative, "eager_loss": reference_loss,
                "compiled_loss": loss, "gradient_relative_l2": gradient_relative,
                "parameters_and_gradients_dtype": "float32", "backward_autocast": "off",
                "same_state_verified": True, "backend": backend,
                "token_execution": compiled.token_execution}
    finally:
        eager.zero_grad(set_to_none=True)
        compiled.zero_grad(set_to_none=True)


def compiler_counters():
    from torch._dynamo.utils import counters
    return {group: {str(k): int(v) for k, v in values.items()} for group, values in counters.items()}


def counter_total(counters, group, key=None):
    values = counters.get(group, {})
    return sum(values.values()) if key is None else values.get(key, 0)


def backend_inventory(backend):
    module_names = {"flash4": ("flash_attn.cute.interface",),
                    "flash3": ("flash_attn_3.flash_attn_interface", "flash_attn_interface"),
                    "sdpa": ("torch",)}[backend]
    try:
        for index, module_name in enumerate(module_names):
            try:
                module = importlib.import_module(module_name)
                break
            except ModuleNotFoundError as error:
                # Match the model's official FA3 package/legacy alias support;
                # an installed backend with a missing dependency must fail.
                if (index == len(module_names) - 1
                        or error.name not in {module_name, module_name.split(".")[0]}):
                    raise
    except ImportError as error:
        raise UnsupportedBackend(f"Required backend import failed: {module_name}: {error}") from error
    module_path = Path(module.__file__)
    versions = {}
    for name in ("torch", "flash-attn-4", "flash-attn-3", "flash-attn", "nvidia-cutlass-dsl"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"module": module_name, "path": str(module_path), "sha256": digest(module_path),
            "versions": versions, "sdpa_kernel": "FLASH_ATTENTION_ONLY" if backend == "sdpa" else None}


def timed_step(model, batch, optimizer, backend, device):
    optimizer.zero_grad(set_to_none=True)
    with execution_context(backend, device):
        losses = model.training_losses(batch, "libero_gripper", pose_weight=0.0)
    require(losses["total"].dtype == losses["image"].dtype == torch.float32, "MSE must remain FP32")
    losses["total"].backward()
    optimizer.step()
    return losses["total"].detach()


def run_candidate(request, destination):
    from egomimic.models.image_proprio import ImageProprioModel

    if not torch.cuda.is_available():
        raise UnsupportedBackend("A scheduled H100 CUDA device is required; no CPU benchmark fallback")
    device = torch.device("cuda", request["device"])
    torch.cuda.set_device(device)
    properties = torch.cuda.get_device_properties(device)
    if "H100" not in properties.name:
        raise UnsupportedBackend(f"This benchmark binds H100 hardware, got {properties.name}")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(request["seed"])
    inventory = backend_inventory(request["backend"])
    arguments = model_arguments(request["backend"], request["scenario"],
                                compile_mode=request["compile_mode"], compile_dynamic=request["compile_dynamic"],
                                token_execution=request.get("token_execution", "padded"))
    model = ImageProprioModel(**arguments).to(device).train()
    require(sum(p.numel() for p in model.parameters()) == PARAMETERS, "Exact architecture parameter count changed")
    require(all(p.dtype == torch.float32 for p in model.parameters()), "Stored parameters must be FP32")
    before_counters = compiler_counters()
    schedule = frame_schedule(request["scenario"], request["batch_size"])
    batches = [make_batch(frames, seed=request["seed"] + index, device=device,
                          frame_seconds=0.2 if request["scenario"] == "fixed16" else 0.05)
               for index, frames in enumerate(schedule)]
    with nontrivial_conditioning_probe(model, request["seed"] + 1000) as probe_record:
        probe = make_batch((2, 3) if request["batch_size"] > 1 else (2,), seed=request["seed"], device=device)
        correctness = {"bounded_gradient_probe": check_outputs(
            model, probe, backend=request["backend"], seed=request["seed"],
            tolerance=request["tolerance"], gradients=True)}
        if request["compile_mode"] != "none":
            # This is a separate eager model, not another invocation of the same
            # compiled callable. Keep synthetic noise RNG unchanged by construction.
            with torch.random.fork_rng(devices=[device.index]):
                eager = ImageProprioModel(**{**arguments, "compile_mode": "none"}).to(device).train()
            eager.load_state_dict(model.state_dict(), strict=True)
            correctness["compiled_vs_eager"] = check_eager_compiled(
                eager, model, probe, backend=request["backend"], seed=request["seed"],
                tolerance=request["tolerance"])
            del eager
        del probe
        correctness["conditioning_probe"] = probe_record
        correctness["full_shape_outputs"] = []
        with torch.no_grad():
            for batch in batches:
                correctness["full_shape_outputs"].append(check_outputs(
                    model, batch, backend=request["backend"], seed=request["seed"],
                    tolerance=request["tolerance"], gradients=False))
    torch.cuda.empty_cache()  # Release unused correctness-probe allocator blocks.
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    warmup = complete_cycles(request["warmup"], schedule)
    steps = complete_cycles(request["steps"], schedule)
    warm_start = time.perf_counter()
    for step in range(warmup):
        timed_step(model, batches[step % len(batches)], optimizer, request["backend"], device)
    torch.cuda.synchronize(device)
    warmup_seconds = time.perf_counter() - warm_start
    warm_counters = compiler_counters()
    if request["compile_mode"] != "none":
        require(counter_total(warm_counters, "stats", "unique_graphs") > counter_total(before_counters, "stats", "unique_graphs"),
                "Requested compilation captured no tensor regions")
        require(counter_total(warm_counters, "graph_break") == counter_total(before_counters, "graph_break"),
                "A compiled tensor region broke its fullgraph contract")
    torch.cuda.reset_peak_memory_stats(device)
    walls, cuda_times, loss_values = [], [], []
    valid_tokens = 0
    for step in range(steps):
        batch = batches[step % len(batches)]
        torch.cuda.synchronize(device)
        start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start = time.perf_counter()
        start_event.record()
        loss = timed_step(model, batch, optimizer, request["backend"], device)
        end_event.record()
        torch.cuda.synchronize(device)
        walls.append(time.perf_counter() - start)
        cuda_times.append(start_event.elapsed_time(end_event) / 1000)
        loss_values.append(float(loss))
        valid_tokens += sum(schedule[step % len(schedule)]) * 200
    after_counters = compiler_counters()
    require(all(math.isfinite(value) for value in loss_values), "Nonfinite measured training loss")
    require(counter_total(after_counters, "stats", "unique_graphs") == counter_total(warm_counters, "stats", "unique_graphs"),
            "New compiled graphs appeared during steady-state timing")
    require(all(p.grad is not None and p.grad.dtype == torch.float32 and torch.isfinite(p.grad).all()
                for p in model.parameters()), "Missing, nonfinite or non-FP32 measured gradient")
    require(all(state[key].dtype == torch.float32 and torch.isfinite(state[key]).all()
                for state in optimizer.state.values() for key in ("exp_avg", "exp_avg_sq")),
            "Nonfinite or non-FP32 AdamW moments")
    require(all(p.dtype == torch.float32 and torch.isfinite(p).all() for p in model.parameters()),
            "Nonfinite or non-FP32 optimized parameters")
    result = {
        "status": "PASS", "candidate": request, "parameters_total_trainable": PARAMETERS,
        "model_arguments": arguments, "backend_inventory": inventory,
        "gpu": {"name": properties.name, "uuid": str(properties.uuid), "total_memory_bytes": properties.total_memory},
        "correctness": correctness, "frame_schedule": schedule,
        "warmup_steps": warmup, "timed_steps": steps, "warmup_seconds": warmup_seconds,
        "wall_seconds": walls, "cuda_event_seconds": cuda_times,
        "median_wall_seconds": statistics.median(walls), "median_cuda_seconds": statistics.median(cuda_times),
        "examples_per_second": steps * request["batch_size"] / sum(walls),
        "valid_tokens_per_second": valid_tokens / sum(walls), "valid_tokens_measured": valid_tokens,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "resident_input_bytes": sum(value.numel() * value.element_size()
                                    for batch in batches for value in batch["image"].values()),
        "losses": loss_values, "compiler_counters_before": before_counters,
        "compiler_counters_after_warmup": warm_counters, "compiler_counters_after_timing": after_counters,
        "compile_scope": "declared_fullgraph_tensor_regions; attention_adapter_and_validation_eager",
        "compiled_backward_autocast": "off" if request["compile_mode"] != "none" else None,
        "measurement_scope": "single_gpu_device_resident_synthetic_model; no_loader_H2D_NCCL_or_checkpoint_IO",
    }
    if request["profile"]:
        trace = destination / "trace.json"
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                                    record_shapes=True, profile_memory=True) as profiler:
            with torch.profiler.record_function("synthetic_device_resident_optimizer_step"):
                timed_step(model, batches[0], optimizer, request["backend"], device)
            torch.cuda.synchronize(device)
        profiler.export_chrome_trace(str(trace))
        result["profile"] = {"path": str(trace), "sha256": digest(trace)}
    return result


def worker(request_path):
    request_path = Path(request_path)
    request = json.loads(request_path.read_text())
    try:
        result = run_candidate(request, request_path.parent)
    except torch.cuda.OutOfMemoryError as error:
        result = {"status": "OOM", "error": str(error), "candidate": request}
    except UnsupportedBackend as error:
        result = {"status": "UNSUPPORTED", "error": str(error), "candidate": request}
    except CorrectnessError as error:
        result = {"status": "CORRECTNESS_FAILED", "error": str(error), "candidate": request}
    except Exception as error:
        # Backend/API/compiler failures are never retried with another kernel.
        unsupported = (isinstance(error, RuntimeError) and any(
            phrase in str(error) for phrase in ("No available kernel", "No viable backend")
        )) or (isinstance(error, ValueError) and
               str(error) == "Ragged packed_valid execution requires flash3 or flash4")
        result = {"status": "UNSUPPORTED" if unsupported else "ERROR", "error": f"{type(error).__name__}: {error}",
                  "traceback": traceback.format_exc(), "candidate": request}
    write_once(request_path.parent / "result.json", result)


def csv_choices(value, choices):
    values = value.split(",")
    if not values or len(values) != len(set(values)) or set(values) - set(choices):
        raise argparse.ArgumentTypeError(f"Choose distinct values from {choices}")
    return values


def run_worker_process(command, folder, timeout):
    """Bound compiler subprocesses as well as the GPU-owning worker on timeout."""
    with (folder / "stdout.log").open("x") as out, (folder / "stderr.log").open("x") as err:
        process = subprocess.Popen(command, stdout=out, stderr=err, start_new_session=True)
        try:
            return process.wait(timeout=timeout), False
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass  # The bounded child exited between timeout and kill.
            process.wait()
            return process.returncode, True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--backends", type=lambda s: csv_choices(s, BACKENDS), default=list(BACKENDS))
    parser.add_argument("--scenarios", type=lambda s: csv_choices(s, SCENARIOS), default=list(SCENARIOS))
    parser.add_argument("--batch-sizes", type=lambda s: [int(v) for v in csv_choices(s, ("1", "2", "4", "8"))], default=[1, 2, 4, 8])
    parser.add_argument("--compile-mode", choices=COMPILE_MODES, default="none")
    parser.add_argument("--compile-dynamic", action="store_true")
    parser.add_argument("--token-execution", choices=TOKEN_EXECUTIONS, default="padded",
                        help="Keep padded tensors, or pack valid image tokens across the image-only expert")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--tolerance", type=float, default=0.03)
    parser.add_argument("--worker-timeout", type=int, default=1800)
    parser.add_argument("--python-bootstrap", type=Path,
                        help="Run every child through this reviewed runtime bootstrap (.pth hooks included)")
    parser.add_argument("--profile-candidate", help="Exact candidate ID such as flash4-mixed-b4-none")
    parser.add_argument("--dry-run", action="store_true", help="Print the immutable candidate plan without CUDA work")
    parser.add_argument("--worker-request", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_request:
        worker(args.worker_request)
        return
    if min(args.warmup, args.steps, args.worker_timeout) < 1 or args.device < 0 or not 0 < args.tolerance < 1:
        parser.error("Positive bounded warmup/steps/timeout/tolerance and nonnegative device required")
    candidates = []
    for backend in args.backends:
        for scenario in args.scenarios:
            for batch_size in args.batch_sizes:
                identity = f"{backend}-{scenario}-b{batch_size}-{args.compile_mode}"
                if args.token_execution != "padded":
                    identity += f"-{args.token_execution}"
                candidates.append(dict(id=identity, backend=backend, scenario=scenario,
                                       batch_size=batch_size, compile_mode=args.compile_mode,
                                       compile_dynamic=args.compile_dynamic, token_execution=args.token_execution,
                                       seed=args.seed,
                                       device=args.device, warmup=args.warmup, steps=args.steps,
                                       tolerance=args.tolerance, profile=args.profile_candidate == identity))
    if args.profile_candidate and not any(c["profile"] for c in candidates):
        parser.error("--profile-candidate must identify an enabled candidate")
    if args.dry_run:
        print(json.dumps({"parameters_total_trainable": PARAMETERS, "candidates": candidates}, indent=2))
        return
    if args.output is None:
        parser.error("--output is required")
    bootstrap = None
    if args.python_bootstrap:
        bootstrap_path = args.python_bootstrap.resolve(strict=True)
        bootstrap = {"path": str(bootstrap_path), "sha256": digest(bootstrap_path)}
    args.output.mkdir(parents=True, exist_ok=False)
    source_files = [Path(__file__), ROOT / "egomimic/models/image_proprio.py",
                    ROOT / "egomimic/models/pixel_diffusion.py",
                    ROOT / "egomimic/models/world_model.py", ROOT / "egomimic/models/attention_execution.py",
                    ROOT / "egomimic/models/compilation.py"]
    source = {str(path.relative_to(ROOT)): digest(path) for path in source_files if path.exists()}
    write_once(args.output / "plan.json", {"schema": "pixel-execution-benchmark-v1", "candidates": candidates,
               "source_sha256": source, "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
               "python_bootstrap": bootstrap, "python_executable": sys.executable,
               "scope": "synthetic_single_gpu_execution_only; not_training_readiness_or_policy_quality"})
    results = []
    for candidate in candidates:
        folder = args.output / candidate["id"]
        folder.mkdir()
        request_path = folder / "request.json"
        write_once(request_path, candidate)
        command = [sys.executable]
        if args.python_bootstrap:
            command.append(str(args.python_bootstrap.resolve()))
        command += [str(Path(__file__).resolve()), "--worker-request", str(request_path.resolve())]
        returncode, timed_out = run_worker_process(command, folder, args.worker_timeout)
        if timed_out:
            result = {"status": "TIMEOUT", "seconds": args.worker_timeout, "candidate": candidate}
        elif returncode != 0 or not (folder / "result.json").exists():
            result = {"status": "PROCESS_ERROR", "returncode": returncode, "candidate": candidate}
        else:
            result = json.loads((folder / "result.json").read_text())
        results.append(result)
        print(json.dumps({"candidate": candidate["id"], "status": result["status"]}), flush=True)
    write_once(args.output / "report.json", {"schema": "pixel-execution-benchmark-v1", "source_sha256": source,
               "status": "COMPLETED", "results": results,
               "scope": "Only PASS candidates have measured and checked synthetic throughput; real-data DDP smoke still required"})


if __name__ == "__main__":
    main()
