#!/usr/bin/env python3
"""Explicit, offline visual-only -> state/action readout weights migration.

This writes a plain tensor dictionary for PipelineAlgo.initialization. It never
resumes an optimizer or edits a training configuration. Model arguments must be
resolved JSON mappings, not Hydra experiment names. For a Lightning checkpoint,
declare its exact state dictionary/config paths and namespace; nothing is guessed.
Loading a pickle-based Lightning checkpoint additionally requires the explicit
trusted flag, after its SHA256 has been checked. Tensor-only sources use the safe
weights-only loader, with their architecture explicitly supplied by the caller.
"""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
import sys
from collections.abc import Mapping
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from egomimic.models.image_proprio import ImageProprioModel

MODEL_TARGET = "egomimic.models.image_proprio.ImageProprioModel"
EXPANDED_ROWS = {"proprio_role.weight": (1, 3), "phase.weight": (1, 2)}


def sha256_file(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def resolve_model_config(config):
    """Bind the exact class signature; reject unknown Hydra/model arguments."""
    if not isinstance(config, Mapping):
        raise TypeError("Model configuration must be an argument mapping")
    config = copy.deepcopy(dict(config))
    if config.pop("_target_", MODEL_TARGET) != MODEL_TARGET:
        raise ValueError("Only ImageProprioModel is supported")
    signature = inspect.signature(ImageProprioModel)
    bound = signature.bind(**config)
    bound.apply_defaults()
    # Reject unresolved interpolation and non-JSON objects before model creation.
    resolved = json.loads(json.dumps(dict(bound.arguments), allow_nan=False))
    if "${" in json.dumps(resolved):
        raise ValueError("Model configuration must have resolved interpolations")
    return resolved


def validate_transition(source_config, target_config):
    source = resolve_model_config(source_config)
    target = resolve_model_config(target_config)
    if source["objective"] != "visual_only":
        raise ValueError("Source objective must be visual_only")
    if target["objective"] != "visual_state_action":
        raise ValueError("Target objective must be visual_state_action")
    if source["action_dims"] is not None:
        raise ValueError("Visual-only source must not declare action dimensions")
    if not isinstance(target["action_dims"], Mapping):
        raise ValueError("Target requires explicit action dimensions")
    image_only = not source["phase_a_condition_on_proprio"]
    target_poses = ImageProprioModel._dimensions(target["pose_dims"], "pose_dims")
    if image_only:
        if source["pose_dims"] not in (None, {}):
            raise ValueError("Image-only source must omit pose_dims")
    elif list(source["pose_dims"].items()) != list(target_poses.items()):
        # Embodiment order affects embedding IDs and deterministic initialization.
        raise ValueError("Source and target pose_dims and order must match")
    if list(target["action_dims"]) != list(target_poses):
        raise ValueError("Action dimensions must use the same embodiment order")
    allowed = {"objective", "action_dims", "max_tokens"}
    if image_only:
        allowed.add("pose_dims")
    changed = [
        key for key in source if key not in allowed and source[key] != target[key]
    ]
    if changed:
        raise ValueError(f"Source/target model configuration mismatch: {changed}")
    if target["max_tokens"] < source["max_tokens"]:
        raise ValueError("Target token capacity must not decrease")
    return source, target


def _select(payload, path):
    for key in path:
        # JSON path components can be object keys or integer sequence indices.
        payload = payload[key]
    return payload


def _plain_config(value):
    from omegaconf import OmegaConf

    return (
        OmegaConf.to_container(value, resolve=True)
        if OmegaConf.is_config(value)
        else value
    )


def _validate_state(state, expected):
    if set(state) != set(expected):
        raise ValueError(
            "Source namespace mismatch: "
            f"missing={sorted(set(expected) - set(state))}, "
            f"unexpected={sorted(set(state) - set(expected))}"
        )
    for name, value in state.items():
        if (
            not isinstance(value, torch.Tensor)
            or value.shape != expected[name].shape
            or value.dtype != expected[name].dtype
        ):
            raise ValueError(f"Source tensor shape/dtype mismatch: {name}")
        if not torch.isfinite(value).all():
            raise ValueError(f"Source tensor is nonfinite: {name}")


def transfer_shared_weights(source, target):
    """Copy every source tensor; initialize only this exact readout expansion."""
    if source.use_phase_embedding != target.use_phase_embedding:
        raise ValueError("Source/target use_phase_embedding must match")
    if source.noise_conditioning != target.noise_conditioning:
        raise ValueError("Source/target noise_conditioning must match")
    for field in ("position_encoding", "rope_axis_dims", "spatial_grid_size"):
        if getattr(source, field) != getattr(target, field):
            raise ValueError(f"Source/target {field} must match")
    source_state, target_state = source.state_dict(), target.state_dict()
    added = {"query", "action_query", "norm.1.weight", "norm.1.bias"}
    for head in ("pose_out", "action_out"):
        for embodiment in target.pose_dims:
            added.update(f"{head}.{embodiment}.{field}" for field in ("weight", "bias"))
    expanded = {
        name: shape for name, shape in EXPANDED_ROWS.items() if name in source_state
    }
    if not source.has_proprio_branch:
        added.add("proprio_role.weight")
        for embodiment in target.pose_dims:
            added.update(
                f"pose_in.{embodiment}.{field}" for field in ("weight", "bias")
            )
        for index, block in enumerate(target.blocks):
            for branch in ("norm1", "qkv", "out", "norm2", "ffn"):
                added.update(
                    f"blocks.{index}.{branch}.1.{name}"
                    for name, _ in getattr(block, branch)[1].named_parameters()
                )
        if source.use_modality_embedding:
            expanded["modality"] = (1, 2)
    if (
        set(source_state) - set(target_state)
        or set(target_state) - set(source_state) != added
    ):
        raise ValueError("Unexpected parameter namespace in readout transition")
    for name, value in source_state.items():
        expected = target_state[name]
        if name in expanded:
            old_rows, new_rows = expanded[name]
            if value.shape != (old_rows, source.width) or expected.shape != (
                new_rows,
                source.width,
            ):
                raise ValueError(f"Unexpected expanded embedding shape: {name}")
            if value.dtype != expected.dtype:
                raise ValueError(f"Shared tensor dtype mismatch: {name}")
            expected[:old_rows].copy_(value)
        else:
            if value.shape != expected.shape or value.dtype != expected.dtype:
                raise ValueError(f"Shared tensor shape/dtype mismatch: {name}")
            expected.copy_(value)
    target.load_state_dict(target_state, strict=True)
    return {
        "copied_tensors": sorted(set(source_state) - set(expanded)),
        "expanded_embeddings": {
            key: {"copied_rows": [0], "new_rows": list(range(1, rows[1]))}
            for key, rows in expanded.items()
        },
        "new_tensors": sorted(added),
        "new_parameter_initialization": "ImageProprioModel constructor at declared seed",
    }


def _probe_batch(source, target, embodiment, completion):
    """Small deterministic input with observed/query images and withheld labels."""
    image_observed = torch.tensor([[True, False, False, True if completion else False]])
    pose_observed = (
        torch.tensor([[False, True]]) if completion else torch.tensor([[True, False]])
    )

    def stream(tokens, dimensions, observed):
        return {
            "value": torch.linspace(-0.8, 0.9, tokens * dimensions).reshape(
                1, tokens, dimensions
            ),
            "time": torch.linspace(-0.6, 0.7, tokens).reshape(1, tokens),
            "index": torch.arange(tokens).reshape(1, tokens),
            "valid": torch.ones(1, tokens, dtype=torch.bool),
            "observed": observed,
        }

    batch = {
        "image": stream(4, source.image_dim, image_observed),
        "proprio": stream(2, target.pose_dims[embodiment], pose_observed),
        "action": stream(
            2, target.action_dims[embodiment], torch.zeros(1, 2, dtype=torch.bool)
        ),
    }
    batch["image"]["xy"] = torch.tensor(
        [[[-0.5, -0.5], [0.5, -0.5], [-0.5, 0.5], [0.5, 0.5]]]
    )
    cameras = source.camera.num_embeddings if source.use_camera_embedding else 1
    batch["image"]["camera"] = torch.arange(4).remainder(cameras).reshape(1, 4)
    # Neither missing state labels nor command labels may influence phase A.
    batch["proprio"]["value"][~pose_observed] = float("nan")
    batch["action"]["value"].fill_(float("nan"))
    values = torch.linspace(0.7, -0.8, 4 * source.image_dim).reshape(
        1, 4, source.image_dim
    )
    levels = torch.tensor([[0.0, 0.25, 0.75, 0.5]])
    return batch, values, levels


@torch.no_grad()
def verify_phase_a(source, target, cpu_attention_override=None):
    """Require bit-exact CPU output; an optional native override is disclosed."""
    if cpu_attention_override not in (None, "native"):
        raise ValueError("cpu_attention_override must be None or native")
    blocks = [block for model in (source, target) for block in model.blocks]
    execution = [(block.attention_backend, block.attention_dtype) for block in blocks]
    if cpu_attention_override is None and any(
        backend != "sdpa" for backend, _ in execution
    ):
        raise ValueError(
            "GPU attention needs an explicit cpu_attention_override='native' "
            "for this CPU weights-preservation proof"
        )
    source.eval()
    target.eval()
    checks = []
    try:
        if cpu_attention_override == "native":
            for block in blocks:
                block.attention_backend, block.attention_dtype = "sdpa", "native"
        for embodiment in target.pose_dims:
            for completion in (False, True):
                batch, values, levels = _probe_batch(
                    source, target, embodiment, completion
                )
                source_batch = (
                    batch if source.has_proprio_branch else {"image": batch["image"]}
                )
                original = source(source_batch, embodiment, values, levels, phase="A")[
                    "image"
                ]
                migrated = target(batch, embodiment, values, levels, phase="A")["image"]
                if not torch.isfinite(original).all() or not torch.equal(
                    original, migrated
                ):
                    difference = (original - migrated).abs().max().item()
                    raise ValueError(
                        f"Phase A preservation failed: {embodiment}, max_abs={difference}"
                    )
                checks.append(
                    {
                        "embodiment": embodiment,
                        "mask": "completion" if completion else "forecast",
                        "max_abs_error": 0.0,
                    }
                )
    finally:
        for block, (backend, dtype) in zip(blocks, execution):
            block.attention_backend, block.attention_dtype = backend, dtype
    return {
        "device": "cpu",
        "dtype": "float32",
        "bit_exact": True,
        "checks": checks,
        "cpu_attention_override": cpu_attention_override,
        "declared_attention": sorted(
            {f"{backend}/{dtype}" for backend, dtype in execution}
        ),
        "gpu_kernel_verified": False,
    }


def initialize_readout(
    *,
    source_checkpoint,
    source_sha256,
    source_config,
    target_config,
    output_path,
    receipt_path,
    seed,
    source_prefix="",
    state_dict_path=(),
    source_config_path=(),
    trusted_lightning_checkpoint=False,
    stage_id="image_proprio",
    module_path="model",
    cpu_attention_override=None,
):
    """Create a fresh stage-two tensor artifact, never an optimizer resume.

    Full checkpoints must bind their model configuration using an explicit path.
    A bare tensor dictionary has no architecture metadata; in that case the
    receipt explicitly records that the caller supplied the model declaration.
    """
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("Seed must be an integer in [0, 2**63)")
    source_config, target_config = validate_transition(source_config, target_config)
    source_path = Path(source_checkpoint).resolve()
    output_path, receipt_path = (
        Path(output_path).resolve(),
        Path(receipt_path).resolve(),
    )
    if output_path == receipt_path or source_path in (output_path, receipt_path):
        raise ValueError("Input, output and receipt paths must be distinct")
    if output_path.exists() or receipt_path.exists():
        raise FileExistsError(
            "Initialization artifacts are immutable; choose new paths"
        )
    if trusted_lightning_checkpoint and not source_config_path:
        raise ValueError(
            "Trusted Lightning input requires a source_config_path binding"
        )
    # Hash and load the same open file. A replacement at this path cannot swap
    # the verified source underneath torch.load; also reject in-place mutation.
    with source_path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
        if digest != source_sha256:
            raise ValueError("Source checkpoint SHA256 mismatch")
        handle.seek(0)
        # No automatic unsafe-pickle fallback.
        payload = torch.load(
            handle, map_location="cpu", weights_only=not trusted_lightning_checkpoint
        )
        handle.seek(0)
        if hashlib.file_digest(handle, "sha256").hexdigest() != digest:
            raise ValueError("Source checkpoint changed while loading")
    if source_config_path:
        recorded = resolve_model_config(
            _plain_config(_select(payload, source_config_path))
        )
        if recorded != source_config or list(
            (recorded["pose_dims"] or {}).items()
        ) != list((source_config["pose_dims"] or {}).items()):
            raise ValueError(
                "Checkpoint model configuration does not match declared source"
            )
    selected = _select(payload, state_dict_path)
    if not isinstance(selected, Mapping) or any(
        not isinstance(key, str) for key in selected
    ):
        raise TypeError("State dictionary path must select a string-keyed mapping")
    if not source_config_path and (state_dict_path or source_prefix):
        raise ValueError(
            "Nested/namespaced checkpoint input requires source_config_path"
        )
    state = {
        key[len(source_prefix) :]: value
        for key, value in selected.items()
        if key.startswith(source_prefix)
    }
    # fork_rng prevents a utility call from changing the caller's initialization RNG.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        source = ImageProprioModel(**source_config).cpu().float()
        _validate_state(state, source.state_dict())
        source.load_state_dict(state, strict=True)
        torch.random.default_generator.manual_seed(seed)
        target = ImageProprioModel(**target_config).cpu().float()
        transfer = transfer_shared_weights(source, target)
        preservation = verify_phase_a(source, target, cpu_attention_override)
    weights = {
        key: value.detach().cpu().clone() for key, value in target.state_dict().items()
    }
    for path in (output_path, receipt_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("xb") as handle:
        torch.save(weights, handle)
    output_sha = sha256_file(output_path)
    receipt = {
        "schema": "pixel-readout-initialization-v1",
        "status": "PASS",
        "source": {
            "path": str(source_path),
            "sha256": digest,
            "state_dict_path": list(state_dict_path),
            "source_prefix": source_prefix,
            "model_config_path": list(source_config_path),
            "config_binding": "checkpoint"
            if source_config_path
            else "caller_declared_tensor_state",
            "trusted_pickle": bool(trusted_lightning_checkpoint),
        },
        "source_model": source_config,
        "target_model": target_config,
        "seed": seed,
        "transfer": transfer,
        "phase_a_preservation": preservation,
        "output": {
            "path": str(output_path),
            "sha256": output_sha,
            "tensor_count": len(weights),
            "format": "plain_tensor_state_dict",
        },
        "implementation": {
            "utility_sha256": sha256_file(__file__),
            "model_sha256": sha256_file(inspect.getfile(ImageProprioModel)),
            "torch_version": str(torch.__version__),
        },
        "optimizer_state_transferred": False,
        "training_step_transferred": False,
        "initialization": [
            {
                "stage_id": stage_id,
                "module_path": module_path,
                "source": str(output_path),
                "sha256": output_sha,
            }
        ],
    }
    with receipt_path.open("x") as handle:
        json.dump(receipt, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument(
        "--source-config", required=True, help="Resolved model arguments JSON"
    )
    parser.add_argument(
        "--target-config", required=True, help="Resolved model arguments JSON"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--source-prefix", default="")
    parser.add_argument(
        "--state-dict-path",
        type=json.loads,
        default=[],
        help='JSON path, e.g. ["state_dict"]',
    )
    parser.add_argument(
        "--source-config-path",
        type=json.loads,
        default=[],
        help="JSON path into checkpoint for exact model arguments",
    )
    parser.add_argument("--trusted-lightning-checkpoint", action="store_true")
    parser.add_argument("--stage-id", default="image_proprio")
    parser.add_argument("--module-path", default="model")
    parser.add_argument(
        "--cpu-attention-override",
        choices=["native"],
        help="Explicit native SDPA CPU weight-preservation proof; does not verify GPU kernels",
    )
    args = parser.parse_args()
    receipt = initialize_readout(
        source_checkpoint=args.source_checkpoint,
        source_sha256=args.source_sha256,
        source_config=json.loads(Path(args.source_config).read_text()),
        target_config=json.loads(Path(args.target_config).read_text()),
        output_path=args.output,
        receipt_path=args.receipt,
        seed=args.seed,
        source_prefix=args.source_prefix,
        state_dict_path=args.state_dict_path,
        source_config_path=args.source_config_path,
        trusted_lightning_checkpoint=args.trusted_lightning_checkpoint,
        stage_id=args.stage_id,
        module_path=args.module_path,
        cpu_attention_override=args.cpu_attention_override,
    )
    print(
        json.dumps({"status": receipt["status"], "output": receipt["output"]}, indent=2)
    )


if __name__ == "__main__":
    main()
