"""Strict, hash-verified weights initialization at declared stage boundaries."""

import hashlib
from collections.abc import Mapping
from pathlib import Path

import torch
from torch import nn

from egomimic.pipeline.construction import restoring_parameters


def stage_module(pipeline, stage_id, module_path=""):
    module = pipeline.stage_by_id(stage_id)
    for part in module_path.split(".") if module_path else ():
        if not part or part.startswith("_"):
            raise ValueError("Initialization paths must name public registered modules")
        module = getattr(module, part)
    if not isinstance(module, nn.Module):
        raise TypeError("Initialization target is not a registered module")
    return module


def initialize_weights(pipeline, specifications):
    """Preflight every source and namespace before changing any model tensor.

    A mapping selects exact modules and exact source prefixes. It cannot ignore
    unknown keys or infer legacy architectures from shapes. This only loads
    weights, never optimizer state or a training step.
    """
    if restoring_parameters():
        return []
    pending, receipts, occupied = [], [], set()
    for spec in specifications or ():
        module = stage_module(pipeline, spec["stage_id"], spec.get("module_path", ""))
        targets = {id(value) for value in (*module.parameters(), *module.buffers())}
        if occupied & targets:
            raise ValueError(
                "Initialization specifications overlap the same parameters or buffers"
            )
        occupied.update(targets)
        source = spec["source"]
        if isinstance(source, Mapping):
            from huggingface_hub import hf_hub_download

            revision = source["revision"]
            if len(revision) != 40 or any(
                c not in "0123456789abcdef" for c in revision
            ):
                raise ValueError("Hugging Face initialization requires a pinned commit")
            source = hf_hub_download(
                repo_id=source["repo_id"],
                filename=source["filename"],
                revision=revision,
            )
        path = Path(source)
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        if digest != spec["sha256"]:
            raise ValueError(f"Initialization source hash mismatch: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        for key in spec.get("state_dict_path", []):
            payload = payload[key]
        if not isinstance(payload, Mapping):
            raise TypeError("Initialization must select a tensor state dictionary")
        prefix = spec.get("source_prefix", "")
        state = {
            key[len(prefix) :]: value
            for key, value in payload.items()
            if key.startswith(prefix)
        }
        expected = module.state_dict()
        if set(state) != set(expected):
            raise ValueError(
                f"Initialization namespace mismatch: missing={sorted(set(expected)-set(state))}, unexpected={sorted(set(state)-set(expected))}"
            )
        for key, value in state.items():
            if (
                not torch.is_tensor(value)
                or value.shape != expected[key].shape
                or value.dtype != expected[key].dtype
            ):
                raise ValueError(f"Initialization tensor shape/dtype mismatch at {key}")
            if (value.is_floating_point() or value.is_complex()) and not torch.isfinite(
                value
            ).all():
                raise ValueError(f"Nonfinite initialization tensor at {key}")
        pending.append((module, state))
        receipts.append(
            {
                "stage_id": spec["stage_id"],
                "module_path": spec.get("module_path", ""),
                "sha256": digest,
                "source_prefix": prefix,
            }
        )
    for module, state in pending:
        module.load_state_dict(state, strict=True)
    return receipts


def configure_trainability(pipeline, specifications):
    targets = []
    for spec in specifications or ():
        if type(spec["trainable"]) is not bool:
            raise TypeError("Trainability must be declared as a boolean")
        targets.append(
            (
                stage_module(pipeline, spec["stage_id"], spec.get("module_path", "")),
                spec["trainable"],
            )
        )
    for module, trainable in targets:
        module.requires_grad_(trainable)
