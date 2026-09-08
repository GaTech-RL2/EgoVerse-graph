"""Model-side fixed-bank forward pass for Action Flow validation diagnostics."""

from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn as nn


def resolve_action_flow_topology(model: Any) -> tuple[Any, Any, Any]:
    """Find one encoder, field, and decoder from declarative stage contracts."""

    pipeline = getattr(model, "pipeline", None)
    stages = tuple(getattr(pipeline, "stages", ()))
    if not stages:
        raise RuntimeError("Action Flow diagnostics require a staged pipeline")

    def writes(stage, mode: str) -> tuple[str, ...]:
        contract = getattr(stage, "contract", None)
        if not callable(contract):
            return ()
        _, values = contract(mode)
        return tuple(values)

    encoder_stages = tuple(
        stage
        for stage in stages
        if isinstance(getattr(stage, "encoder", None), nn.Module)
        and any(key.endswith("/clean_latent") for key in writes(stage, "train"))
    )
    field_stages = tuple(
        stage
        for stage in stages
        if isinstance(getattr(stage, "field", None), nn.Module)
        and any(key.endswith("/predicted_velocity") for key in writes(stage, "train"))
        and any(key.endswith("/generated_latent") for key in writes(stage, "inference"))
    )
    decoder_stages = tuple(
        stage
        for stage in stages
        if isinstance(getattr(stage, "decoder", None), nn.Module)
        and any(key.endswith("/reconstruction") for key in writes(stage, "train"))
    )
    counts = (len(encoder_stages), len(field_stages), len(decoder_stages))
    if counts != (1, 1, 1):
        raise RuntimeError(
            "Action Flow graph must have exactly one clean-latent encoder, "
            "conditional field, and content decoder; "
            f"found {counts}"
        )
    return encoder_stages[0], field_stages[0], decoder_stages[0]


def clone_inference_tensors(value: Any) -> Any:
    """Clone inference tensors into ordinary tensors for AD-sensitive paths."""

    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, Mapping):
        return type(value)(
            (key, clone_inference_tensors(item)) for key, item in value.items()
        )
    if isinstance(value, tuple):
        return tuple(clone_inference_tensors(item) for item in value)
    if isinstance(value, list):
        return [clone_inference_tensors(item) for item in value]
    return value


def cuda_devices(value: Any) -> list[int]:
    """Return the CUDA device indices found recursively in a value."""

    devices = set()

    def visit(item: Any) -> None:
        if torch.is_tensor(item):
            if item.device.type == "cuda" and item.device.index is not None:
                devices.add(int(item.device.index))
        elif isinstance(item, Mapping):
            for child in item.values():
                visit(child)
        elif isinstance(item, (tuple, list)):
            for child in item:
                visit(child)

    visit(value)
    return sorted(devices)


def _block_outputs(
    module: nn.Module,
    call,
    *,
    label: str,
    required: bool,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    blocks = getattr(module, "blocks", None)
    if not isinstance(blocks, nn.ModuleList) or not blocks:
        if required:
            raise RuntimeError(
                f"Action Flow {label} activation capture needs a non-empty "
                "ModuleList named blocks"
            )
        return call(), None, None

    captured: list[torch.Tensor] = []

    canonicalize = getattr(module, "canonicalize_diagnostic_block_output", None)

    def capture(block_index: int):
        def hook(_module, _inputs, output):
            if not torch.is_tensor(output):
                raise TypeError(f"Action Flow {label} block output must be a tensor")
            if callable(canonicalize):
                output = canonicalize(output, block_index)
            captured.append(output)

        return hook

    handles = [
        block.register_forward_hook(capture(block_index))
        for block_index, block in enumerate(blocks)
    ]
    try:
        result = call()
    finally:
        for handle in handles:
            handle.remove()
    if len(captured) != len(blocks):
        raise RuntimeError(
            f"Action Flow {label} activation capture expected {len(blocks)} "
            f"outputs, got {len(captured)}"
        )
    shape = tuple(captured[0].shape)
    if any(tuple(value.shape) != shape for value in captured[1:]):
        raise RuntimeError(f"Action Flow {label} block shapes do not align")
    return (
        result,
        torch.stack(captured),
        torch.arange(len(blocks), device=captured[0].device, dtype=torch.int64),
    )


def _cap_batch_rows(batch: Mapping, count: int, limit: int) -> dict:
    def cap(value: Any) -> Any:
        if torch.is_tensor(value) and value.ndim and int(value.shape[0]) == count:
            return value[:limit]
        if isinstance(value, Mapping):
            return type(value)((key, cap(item)) for key, item in value.items())
        if isinstance(value, tuple):
            return tuple(cap(item) for item in value)
        if isinstance(value, list):
            return [cap(item) for item in value]
        return value

    return dict(cap(batch))


def _decoder_singular_values(
    decoder: nn.Module, values: torch.Tensor, sample_count: int
) -> torch.Tensor:
    selected = values[:sample_count].detach().clone()
    singular_values = []
    for value in selected:

        def decode_one(item):
            return decoder(item.unsqueeze(0)).squeeze(0)

        with torch.enable_grad():
            jacobian = torch.func.jacrev(decode_one)(value)
            matrix = jacobian.float().reshape(jacobian.numel() // value.numel(), -1)
            singular_values.append(torch.linalg.svdvals(matrix).detach())
    return torch.stack(singular_values)


def _fixed_endpoint(
    *,
    state: torch.Tensor,
    raw_level: float,
    steps: int,
    condition: torch.Tensor,
    predict,
) -> tuple[torch.Tensor, int]:
    scaled = raw_level * steps
    rounded = round(scaled)
    evaluations = (
        int(rounded)
        if math.isclose(scaled, rounded, rel_tol=0.0, abs_tol=1.0e-7)
        else int(math.ceil(scaled))
    )
    evaluations = max(0, min(steps, evaluations))
    current_time = raw_level
    result = state
    batch_size = int(state.shape[0])
    drop_mask = torch.zeros(batch_size, dtype=torch.bool, device=state.device)
    for grid_index in range(evaluations, 0, -1):
        next_time = (grid_index - 1) / steps
        delta = current_time - next_time
        if delta <= 0.0:
            raise RuntimeError("Action Flow fixed integration grid is not decreasing")
        time_value = torch.full(
            (batch_size,),
            current_time,
            dtype=torch.float32,
            device=state.device,
        )
        result = result - delta * predict(result, time_value, condition, drop_mask)
        current_time = next_time
    return result, evaluations


def _diagnostic_source(
    model: Any,
    source: str,
    source_batch: Mapping,
    *,
    raw_noise_levels: Sequence[float],
    noise_seed: int,
    max_samples: int | None,
    jacobian_samples: int,
    capture_activations: bool,
) -> OrderedDict[str, Any]:
    encoder_stage, field_stage, decoder_stage = resolve_action_flow_topology(model)
    stages = tuple(model.pipeline.stages)
    encoder_index = stages.index(encoder_stage)

    prepared = dict(source_batch)
    for stage in stages[:encoder_index]:
        prepared = stage.execute(prepared, mode="train")
        if not isinstance(prepared, dict):
            raise TypeError("Action Flow preprocessing stage must return a dict")

    target_key = getattr(encoder_stage, "input_key", "target")
    condition_key = getattr(field_stage, "inference_condition_key", "condition")
    target = prepared.get(target_key)
    condition = prepared.get(condition_key)
    if not torch.is_tensor(target) or target.ndim < 2:
        raise TypeError(f"Action Flow diagnostic {target_key!r} must be batched")
    batch_size = int(target.shape[0])
    if not torch.is_tensor(condition) or condition.ndim < 2:
        raise TypeError(f"Action Flow diagnostic {condition_key!r} must be batched")
    if int(condition.shape[0]) != batch_size:
        raise ValueError("Action Flow target and condition batches do not align")
    limit = batch_size if max_samples is None else min(batch_size, max_samples)
    prepared = _cap_batch_rows(prepared, batch_size, limit)
    target = prepared[target_key]
    condition = prepared[condition_key]
    batch_size = limit

    encoder = encoder_stage.encoder
    field = field_stage.field
    decoder = decoder_stage.decoder
    clean, encoder_activations, encoder_indices = _block_outputs(
        encoder,
        lambda: encoder(target),
        label="encoder",
        required=capture_activations,
    )
    if not torch.is_tensor(clean) or clean.ndim != 3:
        raise ValueError("Action Flow clean latent must have shape (B, H, D)")
    if int(clean.shape[0]) != batch_size:
        raise ValueError("Action Flow encoder changed the batch size")

    generator = torch.Generator(device=clean.device)
    generator.manual_seed(noise_seed)
    noise = torch.randn(
        clean.shape,
        device=clean.device,
        dtype=clean.dtype,
        generator=generator,
    )
    levels = torch.tensor(
        tuple(float(level) for level in raw_noise_levels),
        dtype=torch.float32,
        device=clean.device,
    )
    if levels.ndim != 1 or not levels.numel():
        raise ValueError("raw_noise_levels must be a non-empty sequence")
    if not bool(torch.isfinite(levels).all()) or not bool(
        ((levels >= 0.0) & (levels <= 1.0)).all()
    ):
        raise ValueError("raw_noise_levels must contain finite values in [0, 1]")

    predict = getattr(field_stage, "_predict", None)
    if not callable(predict):
        raise RuntimeError("Action Flow field stage lacks its prediction boundary")
    target_velocity = noise - clean
    fixed_states = []
    predicted_velocities = []
    velocity_residuals = []
    predicted_clean = []
    field_activations = []
    field_indices = None
    drop_mask = torch.zeros(batch_size, dtype=torch.bool, device=clean.device)
    for level_tensor in levels:
        raw_level = float(level_tensor)
        level = torch.full(
            (batch_size,),
            raw_level,
            dtype=torch.float32,
            device=clean.device,
        )
        level_view = level.to(clean.dtype).reshape(
            batch_size, *([1] * (clean.ndim - 1))
        )
        state = (1.0 - level_view) * clean + level_view * noise
        prediction, activations, indices = _block_outputs(
            field,
            lambda state=state, level=level: predict(
                state, level, condition, drop_mask
            ),
            label="field",
            required=capture_activations,
        )
        fixed_states.append(state)
        predicted_velocities.append(prediction)
        velocity_residuals.append(prediction - target_velocity)
        predicted_clean.append(state - level_view * prediction)
        if activations is not None:
            field_activations.append(activations)
            field_indices = indices

    fixed_states_tensor = torch.stack(fixed_states)
    predicted_velocity_tensor = torch.stack(predicted_velocities)
    velocity_residual_tensor = torch.stack(velocity_residuals)
    predicted_clean_tensor = torch.stack(predicted_clean)

    steps = int(getattr(field_stage, "num_inference_steps", 0))
    if steps <= 0:
        raise ValueError("Action Flow inference step count must be positive")
    fixed_final = []
    fixed_evaluations = []
    for level, state in zip(levels.tolist(), fixed_states_tensor, strict=True):
        endpoint, evaluations = _fixed_endpoint(
            state=state,
            raw_level=float(level),
            steps=steps,
            condition=condition,
            predict=predict,
        )
        fixed_final.append(endpoint)
        fixed_evaluations.append(evaluations)
    fixed_final_tensor = torch.stack(fixed_final)

    trajectory = [noise]
    generated = noise
    step_size = 1.0 / steps
    for index in range(steps):
        time_value = torch.full(
            (batch_size,),
            1.0 - index * step_size,
            dtype=torch.float32,
            device=clean.device,
        )
        generated = generated - step_size * predict(
            generated, time_value, condition, drop_mask
        )
        trajectory.append(generated)
    trajectory_tensor = torch.stack(trajectory)

    decoded_reconstruction = decoder(clean)
    decoded_noise = decoder(noise)
    decoded_generated = decoder(generated)

    def decode_leading(value: torch.Tensor) -> torch.Tensor:
        if value.ndim < 3:
            raise ValueError("Action Flow diagnostic decoder input must be batched")
        outer_shape = value.shape[:-3]
        if not outer_shape:
            return decoder(value)
        grouped = value.reshape(-1, *value.shape[-3:])
        # Preserve the logical validation batch for every trajectory/noise level.
        # Flattening outer axes into B can select a different BF16 GEMM kernel and
        # make identical endpoint tensors decode differently by batch geometry.
        decoded = torch.stack([decoder(group) for group in grouped])
        return decoded.reshape(*outer_shape, *decoded.shape[1:])

    decoded_fixed_states = decode_leading(fixed_states_tensor)
    decoded_predicted_clean = decode_leading(predicted_clean_tensor)
    decoded_fixed_final = decode_leading(fixed_final_tensor)
    decoded_trajectory = decode_leading(trajectory_tensor)

    jacobian_count = min(batch_size, jacobian_samples)
    clean_singular = _decoder_singular_values(decoder, clean, jacobian_count)
    noise_singular = _decoder_singular_values(decoder, noise, jacobian_count)
    fixed_singular = torch.stack(
        [
            _decoder_singular_values(decoder, value, jacobian_count)
            for value in fixed_states_tensor
        ]
    )

    result: OrderedDict[str, Any] = OrderedDict(
        schema="action-flow-validation-diagnostics/v1",
        source=source,
        noise_seed=noise_seed,
        returned_samples=batch_size,
        jacobian_samples=jacobian_count,
        num_inference_steps=steps,
        integration_semantics=(
            "reverse_euler_partial_to_lower_canonical_grid_then_uniform_to_zero"
        ),
        noise_levels=levels,
        fixed_level_field_evaluations=torch.tensor(
            fixed_evaluations, device=clean.device, dtype=torch.int64
        ),
        target=target,
        condition=condition,
        **{
            "latent/clean": clean,
            "latent/noise": noise,
            "latent/generated": generated,
            "latent/fixed_states": fixed_states_tensor,
            "latent/predicted_clean": predicted_clean_tensor,
            "latent/fixed_final": fixed_final_tensor,
            "latent/trajectory": trajectory_tensor,
            "field/predicted_velocity": predicted_velocity_tensor,
            "field/velocity_residual": velocity_residual_tensor,
            "decoded/reconstruction": decoded_reconstruction,
            "decoded/noise": decoded_noise,
            "decoded/generated": decoded_generated,
            "decoded/fixed_states": decoded_fixed_states,
            "decoded/predicted_clean": decoded_predicted_clean,
            "decoded/fixed_final": decoded_fixed_final,
            "decoded/trajectory": decoded_trajectory,
            "decoder_jacobian/clean_singular_values": clean_singular,
            "decoder_jacobian/noise_singular_values": noise_singular,
            "decoder_jacobian/fixed_singular_values": fixed_singular,
        },
    )
    if encoder_activations is not None:
        result["activation/encoder_blocks"] = encoder_activations
        result["activation/encoder_block_indices"] = encoder_indices
        result["activation/field_blocks"] = torch.stack(field_activations)
        result["activation/field_block_indices"] = field_indices
    result["provenance/encoder_class"] = type(encoder).__name__
    result["provenance/field_class"] = type(field).__name__
    result["provenance/decoder_class"] = type(decoder).__name__
    return OrderedDict(
        (key, value.detach() if torch.is_tensor(value) else value)
        for key, value in result.items()
    )


def collect_action_flow_diagnostics(
    model: Any,
    batch: Mapping,
    *,
    raw_noise_levels: Sequence[float],
    noise_seed: int,
    max_samples: int | None,
    jacobian_samples: int,
    capture_activations: bool,
    already_processed: bool,
) -> OrderedDict[str, OrderedDict[str, Any]]:
    """Run the fixed-bank diagnostic forward and return detached tensors."""

    with torch.inference_mode(False):
        ordinary = clone_inference_tensors(batch)
        devices = cuda_devices(ordinary)
        model_device = getattr(model, "device", None)
        if (
            isinstance(model_device, torch.device)
            and model_device.type == "cuda"
            and model_device.index is not None
            and int(model_device.index) not in devices
        ):
            devices.append(int(model_device.index))
        with torch.random.fork_rng(devices=sorted(devices)):
            torch.manual_seed(noise_seed)
            processed = (
                ordinary
                if already_processed
                else model.process_batch_for_training(ordinary)
            )
            output = OrderedDict()
            for source, source_batch in processed.items():
                with torch.no_grad():
                    output[source] = _diagnostic_source(
                        model,
                        source,
                        source_batch,
                        raw_noise_levels=raw_noise_levels,
                        noise_seed=noise_seed,
                        max_samples=max_samples,
                        jacobian_samples=jacobian_samples,
                        capture_activations=capture_activations,
                    )
    return output
