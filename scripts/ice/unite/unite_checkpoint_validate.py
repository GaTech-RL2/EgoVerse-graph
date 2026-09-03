#!/usr/bin/env python3
"""Validate one complete, immutable UNITE continuation checkpoint on ICE.

Requeue discovery performs a cheap structural pass. A cutover or resume child
sets ``ICE_STRICT_LOAD=1`` and supplies the expected checkpoint SHA/step; that
path validates every tensor, strictly loads model/optimizer/scheduler state,
and binds the checkpoint to its adjacent Hydra config and W&B identity.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import os
import re
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Iterator

# The runner and preflight invoke this as a dedicated subprocess from inside a
# GPU allocation.  Hide accelerators before importing Torch so every restore is
# deterministically CPU-only without stripping CUDA from the later train child.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
from omegaconf import OmegaConf

SHA256_RE = re.compile(r"[0-9a-f]{64}")
SIGNAL_CHECKPOINT_RE = re.compile(
    r"signal-job=([0-9_]+)-restart=([0-9]{3,})-request=([0-9]{3,})-"
    r"epoch=([0-9]+)-step=([0-9]+)\.ckpt"
)
SAVE_ONLY_PROOF_KEYS = {
    "schema_version",
    "status",
    "checkpoint_path",
    "checkpoint_sha256",
    "checkpoint_size_bytes",
    "epoch",
    "global_step",
    "hook",
    "job_id",
    "restart_count",
    "save_signal",
    "world_size",
    "all_ranks_save_checkpoint_returned",
    "weights_only",
    "publish_atomic_no_overwrite",
    "partial_path",
    "partial_path_absent",
    "reservation_path",
    "reservation_path_absent",
    "application_validation_status",
    "scontrol_invoked",
    "wandb_finalize_invoked",
    "saved_at_unix_ns",
}
SAVE_ONLY_HOOKS = {
    "on_train_batch_end",
    "on_validation_batch_end",
    "on_train_epoch_end",
    "on_validation_epoch_end",
    "on_fit_end",
}


def required(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        raise RuntimeError(f"missing required environment variable {name}")
    return value


def stat_signature(path: Path) -> tuple[int, int, int, int, int]:
    value = path.stat()
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def strict_json_object(path: Path, label: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON number: {value}")

    if not path.is_file():
        raise RuntimeError(f"{label} is missing: {path}")
    before = stat_signature(path)
    raw = path.read_bytes()
    after = stat_signature(path)
    if before != after:
        raise RuntimeError(f"{label} changed while being read: {path}")
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError(f"{label} is not strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be a JSON object")
    return value


def validate_save_only_proof(
    *,
    checkpoint: Path,
    checkpoint_sha256: str,
    checkpoint_size_bytes: int,
    epoch: int,
    global_step: int,
) -> tuple[bool, Path | None, dict[str, Any] | None]:
    match = SIGNAL_CHECKPOINT_RE.fullmatch(checkpoint.name)
    if match is None:
        return False, None, None
    proof_path = Path(str(checkpoint) + ".save-only.json")
    proof = strict_json_object(proof_path, "save-only checkpoint proof")
    if set(proof) != SAVE_ONLY_PROOF_KEYS:
        raise RuntimeError(
            "save-only checkpoint proof keys are not canonical: "
            f"missing={sorted(SAVE_ONLY_PROOF_KEYS - set(proof))} "
            f"unknown={sorted(set(proof) - SAVE_ONLY_PROOF_KEYS)}"
        )

    job_id, restart_text, _request_text, epoch_text, step_text = match.groups()
    expected_world_size_text = required("ICE_EXPECTED_WORLD_SIZE")
    if re.fullmatch(r"[1-9][0-9]*", expected_world_size_text) is None:
        raise RuntimeError("ICE_EXPECTED_WORLD_SIZE must be a positive integer")
    exact_values = {
        "schema_version": 1,
        "status": "FULL_STATE_CHECKPOINT_PUBLISHED_UNVALIDATED",
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_size_bytes": checkpoint_size_bytes,
        "epoch": epoch,
        "global_step": global_step,
        "job_id": job_id,
        "restart_count": int(restart_text),
        "save_signal": "SIGUSR2",
        "world_size": int(expected_world_size_text),
        "all_ranks_save_checkpoint_returned": True,
        "weights_only": False,
        "publish_atomic_no_overwrite": True,
        "partial_path": str(
            checkpoint.with_name(f".{checkpoint.name}.partial")
        ),
        "partial_path_absent": True,
        "reservation_path": str(
            checkpoint.with_name(f".{checkpoint.name}.reserve")
        ),
        "reservation_path_absent": True,
        "application_validation_status": "NOT_RUN_BY_CALLBACK",
        "scontrol_invoked": False,
        "wandb_finalize_invoked": False,
    }
    for key, expected in exact_values.items():
        if proof.get(key) != expected or type(proof.get(key)) is not type(expected):
            raise RuntimeError(
                f"save-only checkpoint proof mismatch for {key}: "
                f"{proof.get(key)!r} != {expected!r}"
            )
    if int(epoch_text) != epoch or int(step_text) != global_step:
        raise RuntimeError("save-only checkpoint filename identity is wrong")
    hook = proof.get("hook")
    if not isinstance(hook, str) or hook not in SAVE_ONLY_HOOKS:
        raise RuntimeError("save-only checkpoint hook is not canonical")
    saved_at = proof.get("saved_at_unix_ns")
    if isinstance(saved_at, bool) or not isinstance(saved_at, int) or saved_at <= 0:
        raise RuntimeError("save-only checkpoint timestamp is invalid")
    for temporary_key in ("partial_path", "reservation_path"):
        temporary_path = Path(str(proof[temporary_key]))
        if not temporary_path.is_absolute():
            raise RuntimeError(
                f"save-only checkpoint {temporary_key} is not absolute"
            )
        if temporary_path.exists():
            raise RuntimeError(
                f"save-only checkpoint temporary artifact remains: {temporary_path}"
            )
    return True, proof_path.resolve(), proof


def iter_tensors(
    value: Any, prefix: str = "root"
) -> Iterator[tuple[str, torch.Tensor]]:
    if isinstance(value, torch.Tensor):
        yield prefix, value
    elif isinstance(value, Mapping):
        for key, child in value.items():
            yield from iter_tensors(child, f"{prefix}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            yield from iter_tensors(child, f"{prefix}[{index}]")


def finite_tensor_scan(value: Any, *, complete: bool) -> tuple[int, int]:
    return finite_tensor_scan_with_sentinels(
        value,
        complete=complete,
        verified_nonfinite_sentinels={},
    )[:2]


def finite_tensor_scan_with_sentinels(
    value: Any,
    *,
    complete: bool,
    verified_nonfinite_sentinels: Mapping[str, str],
) -> tuple[int, int, list[dict[str, str]]]:
    tensor_count = 0
    tensor_numel = 0
    observed_sentinels: list[dict[str, str]] = []
    for name, tensor in iter_tensors(value):
        tensor_count += 1
        tensor_numel += tensor.numel()
        if not tensor.numel():
            continue
        candidate = tensor._values() if tensor.is_sparse else tensor
        finite = (
            torch.isfinite(candidate).all()
            if complete
            else torch.isfinite(candidate.reshape(-1)[0])
        )
        if not bool(finite):
            sentinel_reason = verified_nonfinite_sentinels.get(name)
            if sentinel_reason is not None:
                observed_sentinels.append(
                    {"path": name, "reason": sentinel_reason}
                )
                continue
            scope = "full" if complete else "sampled"
            raise RuntimeError(
                f"checkpoint contains a non-finite {scope} tensor: {name}"
            )
    if tensor_count == 0:
        raise RuntimeError("checkpoint contains no tensors")
    missing_sentinels = sorted(
        set(verified_nonfinite_sentinels)
        - {record["path"] for record in observed_sentinels}
    )
    if missing_sentinels:
        raise RuntimeError(
            "verified non-finite framework sentinel was not observed: "
            f"{missing_sentinels}"
        )
    return tensor_count, tensor_numel, observed_sentinels


def verified_model_checkpoint_sentinels(
    callbacks: Mapping[str, Any], tree: Any
) -> dict[str, str]:
    """Allow only Lightning's exact unranked ModelCheckpoint kth-value sentinel."""

    checkpoint_config = OmegaConf.select(
        tree, "callbacks.model_checkpoint", default=None
    )
    if checkpoint_config is None:
        return {}
    monitor = OmegaConf.select(checkpoint_config, "monitor", default=None)
    save_top_k = int(
        OmegaConf.select(checkpoint_config, "save_top_k", default=1)
    )
    if monitor is not None or save_top_k != -1:
        return {}

    mode = str(OmegaConf.select(checkpoint_config, "mode", default="min"))
    if mode not in {"min", "max"}:
        raise RuntimeError("ModelCheckpoint mode is not canonical")
    every_n_train_steps_value = OmegaConf.select(
        checkpoint_config, "every_n_train_steps", default=None
    )
    if (
        isinstance(every_n_train_steps_value, bool)
        or not isinstance(every_n_train_steps_value, int)
        or every_n_train_steps_value != 20_000
    ):
        raise RuntimeError(
            "unranked ModelCheckpoint does not use the required 20k cadence"
        )
    every_n_epochs_value = OmegaConf.select(
        checkpoint_config, "every_n_epochs", default=None
    )
    if every_n_epochs_value is not None:
        raise RuntimeError(
            "unranked ModelCheckpoint unexpectedly has an epoch cadence"
        )
    train_time_interval = OmegaConf.select(
        checkpoint_config, "train_time_interval", default=None
    )
    if train_time_interval is not None:
        raise RuntimeError(
            "unranked ModelCheckpoint unexpectedly has a wall-clock cadence"
        )

    state_key = (
        "ModelCheckpoint{'monitor': None, "
        f"'mode': '{mode}', "
        "'every_n_train_steps': 20000, "
        "'every_n_epochs': 0, "
        "'train_time_interval': None}"
    )
    checkpoint_state = callbacks.get(state_key)
    if not isinstance(checkpoint_state, Mapping):
        raise RuntimeError(
            "checkpoint is missing the exact unranked ModelCheckpoint state"
        )
    if checkpoint_state.get("monitor") is not None:
        raise RuntimeError("ModelCheckpoint state unexpectedly monitors a metric")
    best_k_models = checkpoint_state.get("best_k_models")
    if not isinstance(best_k_models, Mapping) or best_k_models:
        raise RuntimeError(
            "unranked ModelCheckpoint unexpectedly carries ranked models"
        )
    if checkpoint_state.get("best_model_score") is not None:
        raise RuntimeError(
            "unranked ModelCheckpoint unexpectedly carries a best score"
        )
    if checkpoint_state.get("current_score") is not None:
        raise RuntimeError(
            "unranked ModelCheckpoint unexpectedly carries a current score"
        )

    kth_value = checkpoint_state.get("kth_value")
    if (
        not isinstance(kth_value, torch.Tensor)
        or kth_value.numel() != 1
        or not kth_value.dtype.is_floating_point
    ):
        raise RuntimeError(
            "unranked ModelCheckpoint kth_value sentinel is not one float tensor"
        )
    scalar = float(kth_value.item())
    expected_sign = 1.0 if mode == "min" else -1.0
    if not math.isinf(scalar) or math.copysign(1.0, scalar) != expected_sign:
        raise RuntimeError(
            "unranked ModelCheckpoint kth_value is not its exact infinity sentinel"
        )

    tensor_path = f"root.callbacks.{state_key}.kth_value"
    return {
        tensor_path: (
            "lightning.pytorch.callbacks.ModelCheckpoint unranked "
            f"monitor=None/save_top_k=-1/mode={mode} initialization sentinel"
        )
    }


def finite_positive(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise RuntimeError(f"{label} must be finite and positive: {value!r}")
    return result


def exact_nonnegative_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RuntimeError(f"{label} must be a nonnegative integer: {value!r}")
    return value


def exact_environment_integer(name: str, *, positive: bool = False) -> int:
    value = required(name)
    pattern = r"[1-9][0-9]*" if positive else r"[0-9]+"
    if re.fullmatch(pattern, value) is None:
        qualifier = "positive" if positive else "nonnegative"
        raise RuntimeError(f"{name} must be a {qualifier} base-10 integer")
    return int(value)


def load_adjacent_config(checkpoint: Path) -> tuple[Path, Any, str]:
    # Names are deliberately unconstrained; immutability is established by
    # stable stat identity plus SHA. The run-bundle location remains fixed.
    config_path = checkpoint.parent.parent / ".hydra" / "config.yaml"
    if not config_path.is_file():
        raise RuntimeError(
            f"checkpoint-adjacent Hydra config is missing: {config_path}"
        )
    before = stat_signature(config_path)
    raw = config_path.read_bytes()
    after = stat_signature(config_path)
    if before != after:
        raise RuntimeError("Hydra config changed while being read")
    config_sha = hashlib.sha256(raw).hexdigest()
    expected_config_sha = os.environ.get(
        "ICE_EXPECTED_CHECKPOINT_CONFIG_SHA256", ""
    )
    if expected_config_sha:
        if not SHA256_RE.fullmatch(expected_config_sha):
            raise RuntimeError(
                "invalid ICE_EXPECTED_CHECKPOINT_CONFIG_SHA256"
            )
        if config_sha != expected_config_sha:
            raise RuntimeError(
                "checkpoint-adjacent Hydra config SHA-256 mismatch"
            )
    config = OmegaConf.load(io.StringIO(raw.decode("utf-8")))
    return config_path.resolve(), config, config_sha


def plain(value: Any) -> Any:
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value


def model_without_rebased_schedule(value: Any) -> Any:
    result = copy.deepcopy(plain(value))
    if not isinstance(result, dict):
        raise RuntimeError("model config is not a mapping")
    optimizer = result.get("optimizer")
    scheduler = result.get("scheduler")
    if not isinstance(optimizer, dict) or not isinstance(scheduler, dict):
        raise RuntimeError("model optimizer/scheduler config is missing")
    optimizer.pop("lr", None)
    for key in (
        "warmup_steps",
        "decay_start_1_steps",
        "decay_end_1_steps",
        "decay_start_2_steps",
        "decay_end_2_steps",
        "base_lr_1",
        "base_lr_2",
        "final_lr",
    ):
        scheduler.pop(key, None)
    return result


def validate_optimizer_state(
    state: Any,
) -> tuple[dict[str, list[float]], dict[str, list[float]], int]:
    if not isinstance(state, Mapping) or set(state) != {
        "adamw",
        "muon",
        "group_manifest",
    }:
        raise RuntimeError(
            "invalid released UNITE composite optimizer state"
        )
    manifest = state["group_manifest"]
    if not isinstance(manifest, Mapping):
        raise RuntimeError("optimizer group_manifest is not a mapping")
    adam_names = tuple(manifest.get("adamw_parameter_names", ()))
    muon_names = tuple(manifest.get("muon_parameter_names", ()))
    source_commit = str(manifest.get("muon_source_commit", ""))
    if (
        not adam_names
        or not muon_names
        or set(adam_names).intersection(muon_names)
    ):
        raise RuntimeError(
            "optimizer parameter-name partitions are empty or overlap"
        )
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise RuntimeError("optimizer Muon source commit is missing or invalid")

    lrs: dict[str, list[float]] = {}
    initial_lrs: dict[str, list[float]] = {}
    for name in ("adamw", "muon"):
        child = state[name]
        if not isinstance(child, Mapping):
            raise RuntimeError(f"optimizer {name} state is not a mapping")
        groups = child.get("param_groups")
        slots = child.get("state")
        if not isinstance(groups, list) or len(groups) != 1:
            raise RuntimeError(
                f"optimizer {name} must contain exactly one parameter group"
            )
        if not isinstance(slots, Mapping) or not slots:
            raise RuntimeError(f"optimizer {name} has no per-parameter state")
        group = groups[0]
        if not isinstance(group, Mapping) or not group.get("params"):
            raise RuntimeError(f"optimizer {name} parameter group is empty")
        lrs[name] = [finite_positive(group.get("lr"), f"{name} lr")]
        if "initial_lr" in group:
            initial_lrs[name] = [
                finite_positive(group["initial_lr"], f"{name} initial_lr")
            ]
        else:
            initial_lrs[name] = []
        parameter_ids = set(group["params"])
        if not set(slots).issubset(parameter_ids):
            raise RuntimeError(
                f"optimizer {name} state references unknown parameters"
            )
    return lrs, initial_lrs, len(adam_names) + len(muon_names)


def validate_scheduler_state(
    state: Any, step: int
) -> tuple[list[float], list[float]]:
    if not isinstance(state, Mapping):
        raise RuntimeError("scheduler state is not a mapping")
    if exact_nonnegative_integer(
        state.get("last_epoch"), "scheduler last_epoch"
    ) != step:
        raise RuntimeError("scheduler last_epoch differs from global_step")
    base_lrs = [
        finite_positive(value, "scheduler base_lr")
        for value in state.get("base_lrs", ())
    ]
    last_lrs = [
        finite_positive(value, "scheduler last_lr")
        for value in state.get("_last_lr", ())
    ]
    if len(base_lrs) != 2 or len(last_lrs) != 2:
        raise RuntimeError(
            "released UNITE scheduler must carry two optimizer-group LRs"
        )
    return base_lrs, last_lrs


def close_lr(actual: float, expected: float, label: str) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-12):
        raise RuntimeError(f"{label} differs: {actual!r} != {expected!r}")


def expected_cosine_lr(
    step: int, start_step: int, end_step: int, start_lr: float, final_lr: float
) -> float:
    if step <= start_step:
        return start_lr
    if step >= end_step:
        return final_lr
    progress = (step - start_step) / (end_step - start_step)
    return final_lr + (start_lr - final_lr) * 0.5 * (
        1.0 + math.cos(math.pi * progress)
    )


def validate_rebased_schedule(
    *,
    tree: Any,
    step: int,
    optimizer_lrs: Mapping[str, list[float]],
    optimizer_initial_lrs: Mapping[str, list[float]],
    scheduler_base_lrs: list[float],
    scheduler_last_lrs: list[float],
) -> tuple[float, float, int, int, float]:
    start_lr = finite_positive(required("ICE_EXPECTED_LR_START"), "expected LR start")
    final_lr = finite_positive(required("ICE_EXPECTED_LR_FINAL"), "expected LR final")
    start_step = exact_environment_integer("ICE_EXPECTED_LR_START_STEP")
    end_step = exact_environment_integer("ICE_EXPECTED_LR_END_STEP", positive=True)
    if not 0 < final_lr < start_lr:
        raise RuntimeError("expected rebased LR contract must decay to a positive LR")
    if start_step < 0 or end_step <= start_step or not start_step <= step <= end_step:
        raise RuntimeError("checkpoint step is outside the rebased LR schedule")

    optimizer_config = tree.model.optimizer
    scheduler_config = tree.model.scheduler
    if str(optimizer_config._target_) != (
        "egomimic.utils.unite_optim.ReleasedUniteCompositeOptimizer"
    ):
        raise RuntimeError("embedded optimizer target is not released UNITE")
    if str(scheduler_config._target_) != (
        "egomimic.utils.unite_optim.released_unite_two_stage_scheduler"
    ):
        raise RuntimeError("embedded scheduler target is not released UNITE")
    close_lr(float(optimizer_config.lr), start_lr, "embedded optimizer LR")
    integer_contract = {
        "warmup_steps": 0,
        "decay_start_1_steps": start_step,
        "decay_end_1_steps": end_step,
        "decay_start_2_steps": end_step,
        "decay_end_2_steps": end_step,
    }
    for key, expected in integer_contract.items():
        actual = int(scheduler_config[key])
        if actual != expected:
            raise RuntimeError(
                f"embedded rebased scheduler {key} differs: {actual} != {expected}"
            )
    for key, expected in {
        "base_lr_1": start_lr,
        "base_lr_2": final_lr,
        "final_lr": final_lr,
    }.items():
        close_lr(float(scheduler_config[key]), expected, f"embedded scheduler {key}")

    current_lr = expected_cosine_lr(
        step, start_step, end_step, start_lr, final_lr
    )
    for optimizer_name in ("adamw", "muon"):
        values = optimizer_lrs.get(optimizer_name, [])
        initial_values = optimizer_initial_lrs.get(optimizer_name, [])
        if len(values) != 1 or len(initial_values) != 1:
            raise RuntimeError(
                f"{optimizer_name} must preserve one LR and one initial_lr"
            )
        close_lr(values[0], current_lr, f"{optimizer_name} current LR")
        close_lr(initial_values[0], start_lr, f"{optimizer_name} initial LR")
    if len(scheduler_base_lrs) != 2 or len(scheduler_last_lrs) != 2:
        raise RuntimeError("rebased scheduler must preserve two LR groups")
    for index, value in enumerate(scheduler_base_lrs):
        close_lr(value, start_lr, f"scheduler base LR group {index}")
    for index, value in enumerate(scheduler_last_lrs):
        close_lr(value, current_lr, f"scheduler current LR group {index}")
    return start_lr, final_lr, start_step, end_step, current_lr


def strict_restore(
    path: Path,
    tree: Any,
    optimizer_state: Mapping[str, Any],
    scheduler_state: Mapping[str, Any],
    expected_parameter_count: int,
    ema_state: Mapping[str, torch.Tensor],
    step: int,
) -> tuple[int, int]:
    if torch.cuda.is_available():
        raise RuntimeError("strict-load probe must hide CUDA")
    import hydra

    from egomimic.pl_utils.pl_model import ModelWrapper

    target = str(tree.model._target_)
    wrapper_class = hydra.utils.get_class(target)
    if not isinstance(wrapper_class, type) or not issubclass(
        wrapper_class, ModelWrapper
    ):
        raise RuntimeError(f"invalid wrapper class: {target}")
    wrapper = wrapper_class.load_from_checkpoint(
        str(path), map_location="cpu", weights_only=False, strict=True
    )
    wrapper_state = wrapper.state_dict()
    non_cpu = {
        name: str(value.device)
        for name, value in wrapper_state.items()
        if value.device.type != "cpu"
    }
    if non_cpu:
        raise RuntimeError(
            f"strict load produced non-CPU tensors: {non_cpu}"
        )
    strict_parameter_count = sum(
        parameter.numel() for parameter in wrapper.model.nets.parameters()
    )
    if strict_parameter_count != expected_parameter_count:
        raise RuntimeError((strict_parameter_count, expected_parameter_count))
    if hasattr(wrapper, "ema_optimization_step") and int(
        wrapper.ema_optimization_step.item()
    ) != step:
        raise RuntimeError(
            "strictly loaded EMA step differs from global step"
        )

    named_parameters = dict(wrapper.named_parameters())
    if set(ema_state) != set(named_parameters):
        missing = sorted(set(named_parameters) - set(ema_state))
        unexpected = sorted(set(ema_state) - set(named_parameters))
        raise RuntimeError(
            "EMA parameter identity mismatch: "
            f"missing={missing[:5]} unexpected={unexpected[:5]}"
        )
    for name, ema_value in ema_state.items():
        parameter = named_parameters[name]
        if ema_value.shape != parameter.shape or ema_value.dtype != parameter.dtype:
            raise RuntimeError(f"EMA tensor metadata mismatch: {name}")

    configured = wrapper.configure_optimizers()
    if not isinstance(configured, Mapping) or "optimizer" not in configured:
        raise RuntimeError("wrapper did not construct the expected optimizer")
    configured["optimizer"].load_state_dict(optimizer_state)
    scheduler_entry = configured.get("lr_scheduler")
    if not isinstance(scheduler_entry, Mapping) or "scheduler" not in scheduler_entry:
        raise RuntimeError("wrapper did not construct the expected scheduler")
    scheduler_entry["scheduler"].load_state_dict(scheduler_state)
    return strict_parameter_count, len(named_parameters)


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {Path(sys.argv[0]).name} CHECKPOINT")
    path = Path(sys.argv[1]).resolve()
    if not path.is_file() or path.suffix != ".ckpt":
        raise RuntimeError(f"not a checkpoint file: {path}")
    before = stat_signature(path)
    payload = torch.load(
        path, map_location="cpu", weights_only=False, mmap=True
    )
    after_load = stat_signature(path)
    if before != after_load:
        raise RuntimeError("checkpoint changed while being loaded")
    if not isinstance(payload, Mapping):
        raise RuntimeError("checkpoint payload is not a mapping")

    step = exact_nonnegative_integer(payload.get("global_step"), "global_step")
    minimum_step = exact_environment_integer("ICE_EXPECTED_MIN_STEP")
    maximum_step_text = os.environ.get("ICE_EXPECTED_MAX_STEP", "240000")
    if re.fullmatch(r"[0-9]+", maximum_step_text) is None:
        raise RuntimeError("ICE_EXPECTED_MAX_STEP must be a nonnegative integer")
    maximum_step = int(maximum_step_text)
    if not minimum_step <= step <= maximum_step:
        raise RuntimeError((step, minimum_step, maximum_step))
    exact_step = os.environ.get("ICE_EXPECTED_CHECKPOINT_STEP", "")
    if exact_step:
        if re.fullmatch(r"[0-9]+", exact_step) is None:
            raise RuntimeError("expected checkpoint step is not canonical")
        if step != int(exact_step):
            raise RuntimeError(
                f"checkpoint global_step mismatch: {step} != {exact_step}"
            )
    epoch = exact_nonnegative_integer(payload.get("epoch"), "epoch")

    # Exactly one pre-ICE source checkpoint may seed the runner.  Once this
    # three-field contract is present, every other path discovered through the
    # staged/live globs is required below to be an ICE save-only publication.
    source_candidate_environment = {
        "path": os.environ.get("ICE_AUTHORIZED_SOURCE_CANDIDATE_PATH", ""),
        "sha256": os.environ.get("ICE_AUTHORIZED_SOURCE_CANDIDATE_SHA256", ""),
        "step": os.environ.get("ICE_AUTHORIZED_SOURCE_CANDIDATE_STEP", ""),
    }
    source_candidate_present = {
        key for key, value in source_candidate_environment.items() if value
    }
    if source_candidate_present and source_candidate_present != set(
        source_candidate_environment
    ):
        raise RuntimeError("authorized source candidate identity is partial")
    authorized_source_candidate = False
    authorized_source_candidate_sha = ""
    if source_candidate_present:
        source_candidate_path = Path(source_candidate_environment["path"])
        if not source_candidate_path.is_absolute():
            raise RuntimeError("authorized source candidate path is not absolute")
        authorized_source_candidate_sha = source_candidate_environment["sha256"]
        if SHA256_RE.fullmatch(authorized_source_candidate_sha) is None:
            raise RuntimeError("authorized source candidate SHA-256 is invalid")
        source_candidate_step_text = source_candidate_environment["step"]
        if re.fullmatch(r"[1-9][0-9]*", source_candidate_step_text) is None:
            raise RuntimeError("authorized source candidate step is invalid")
        if path == source_candidate_path.resolve():
            if step != int(source_candidate_step_text):
                raise RuntimeError("authorized source candidate step changed")
            authorized_source_candidate = True

    state = payload.get("state_dict")
    ema_state = payload.get("ema_state_dict")
    optimizer_states = payload.get("optimizer_states")
    scheduler_states = payload.get("lr_schedulers")
    loops = payload.get("loops")
    callbacks = payload.get("callbacks")
    if not isinstance(state, Mapping) or not state:
        raise RuntimeError("missing state_dict")
    if not isinstance(ema_state, Mapping) or not ema_state:
        raise RuntimeError("missing ema_state_dict")
    if not isinstance(optimizer_states, list) or len(optimizer_states) != 1:
        raise RuntimeError("expected one composite optimizer state")
    if not isinstance(scheduler_states, list) or len(scheduler_states) != 1:
        raise RuntimeError("expected one scheduler state")
    if not isinstance(loops, Mapping) or not loops:
        raise RuntimeError("missing Lightning loop state")
    if not isinstance(callbacks, Mapping) or not callbacks:
        raise RuntimeError("missing Lightning callback state")
    if exact_nonnegative_integer(
        payload.get("ema_num_updates"), "ema_num_updates"
    ) != step:
        raise RuntimeError("EMA update count differs from global step")
    if payload.get("ema_validate_with_ema") is not True:
        raise RuntimeError("EMA validation contract changed")
    if not math.isclose(
        float(payload.get("ema_decay")),
        0.9978,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError("EMA decay changed")

    (
        optimizer_lrs,
        optimizer_initial_lrs,
        optimizer_parameter_name_count,
    ) = validate_optimizer_state(optimizer_states[0])
    scheduler_base_lrs, scheduler_last_lrs = validate_scheduler_state(
        scheduler_states[0], step
    )

    hyper = payload.get("hyper_parameters")
    if not isinstance(hyper, Mapping) or not isinstance(
        hyper.get("config_tree"), Mapping
    ):
        raise RuntimeError("missing checkpoint config_tree")
    tree = OmegaConf.create(hyper["config_tree"])
    if OmegaConf.select(tree, "model") is None or OmegaConf.select(
        tree, "run_provenance"
    ) is None:
        raise RuntimeError(
            "checkpoint config_tree must contain model and run_provenance"
        )
    task_id = required("ICE_EXPECTED_TASK_ID")
    topology = required("ICE_EXPECTED_TOPOLOGY")
    if topology not in {"shared", "separate"}:
        raise RuntimeError(f"invalid expected topology: {topology}")
    tokens = exact_environment_integer(
        "ICE_EXPECTED_NUM_LATENT_TOKENS", positive=True
    )
    wandb_id = required("ICE_EXPECTED_WANDB_ID")
    provenance = tree.run_provenance
    if str(provenance.sweep_task_id) != task_id:
        raise RuntimeError("task identity changed")
    if str(provenance.topology) != topology:
        raise RuntimeError("run-provenance topology changed")
    if exact_nonnegative_integer(
        provenance.num_latent_tokens, "run-provenance latent-token count"
    ) != tokens:
        raise RuntimeError("run-provenance token count changed")
    if bool(tree.model.share_encoder_denoiser) != (topology == "shared"):
        raise RuntimeError("topology changed")
    if exact_nonnegative_integer(
        tree.model.num_latent_tokens, "model latent-token count"
    ) != tokens:
        raise RuntimeError("model token count changed")
    if str(provenance.split_manifest_sha256) != required(
        "ICE_EXPECTED_SPLIT_SHA256"
    ):
        raise RuntimeError("split identity changed")
    if str(provenance.get("train_only_normalization_sha256", "")) != required(
        "ICE_EXPECTED_NORM_SHA256"
    ):
        raise RuntimeError("normalization identity changed")

    require_rebased_schedule_text = os.environ.get(
        "ICE_REQUIRE_REBASED_SCHEDULE", "0"
    )
    if require_rebased_schedule_text not in {"0", "1"}:
        raise RuntimeError("ICE_REQUIRE_REBASED_SCHEDULE must be 0 or 1")
    rebased_schedule_verified = require_rebased_schedule_text == "1"

    config_path, config, checkpoint_config_sha = load_adjacent_config(path)
    config_wandb_id = str(
        OmegaConf.select(config, "logger.wandb.id", default="")
    )
    if config_wandb_id != wandb_id:
        raise RuntimeError(
            "checkpoint-adjacent config has the wrong W&B ID"
        )
    config_wandb_entity = str(
        OmegaConf.select(config, "logger.wandb.entity", default="")
    )
    config_wandb_project = str(
        OmegaConf.select(config, "logger.wandb.project", default="")
    )
    config_wandb_group = str(
        OmegaConf.select(config, "logger.wandb.group", default="")
    )
    expected_wandb = {
        "entity": required("ICE_EXPECTED_WANDB_ENTITY"),
        "project": required("ICE_EXPECTED_WANDB_PROJECT"),
        "group": required("ICE_EXPECTED_WANDB_GROUP"),
    }
    actual_wandb = {
        "entity": config_wandb_entity,
        "project": config_wandb_project,
        "group": config_wandb_group,
    }
    if actual_wandb != expected_wandb:
        raise RuntimeError(
            f"checkpoint-adjacent W&B identity changed: {actual_wandb!r}"
        )
    origin_model = plain(config.model)
    embedded_model = plain(tree.model)
    if origin_model != embedded_model:
        if not rebased_schedule_verified or model_without_rebased_schedule(
            origin_model
        ) != model_without_rebased_schedule(embedded_model):
            raise RuntimeError(
                "embedded model differs from its adjacent origin config "
                "outside the authorized LR rebase fields"
            )
    if plain(config.run_provenance) != plain(tree.run_provenance):
        raise RuntimeError(
            "checkpoint run provenance differs from adjacent Hydra config"
        )

    runtime_paths = {
        "owner": "runtime.slurm_requeue_owner",
        "signal": "runtime.slurm_save_signal",
        "checkpoint_dir": "runtime.slurm_signal_checkpoint_dir",
    }
    runtime_values = {
        key: OmegaConf.select(config, config_path, default=None)
        for key, config_path in runtime_paths.items()
    }
    runtime_present = {key for key, value in runtime_values.items() if value is not None}
    if runtime_present and runtime_present != set(runtime_paths):
        raise RuntimeError("checkpoint config has a partial Slurm requeue contract")
    runtime_requeue_contract_verified = bool(runtime_present)
    if runtime_requeue_contract_verified:
        if str(runtime_values["owner"]) != "runner":
            raise RuntimeError("resolved Slurm requeue owner is not runner")
        if str(runtime_values["signal"]) != "SIGUSR2":
            raise RuntimeError("resolved Slurm save-only signal is not SIGUSR2")
        runtime_checkpoint_dir = Path(str(runtime_values["checkpoint_dir"]))
        if not runtime_checkpoint_dir.is_absolute():
            raise RuntimeError("resolved signal checkpoint directory is not absolute")
        if runtime_checkpoint_dir.resolve() != path.parent.resolve():
            raise RuntimeError(
                "checkpoint is outside its resolved save-only checkpoint directory"
            )
    require_runtime_contract = os.environ.get(
        "ICE_REQUIRE_RUNTIME_REQUEUE_CONTRACT", "0"
    )
    if require_runtime_contract not in {"0", "1"}:
        raise RuntimeError(
            "ICE_REQUIRE_RUNTIME_REQUEUE_CONTRACT must be 0 or 1"
        )
    runtime_contract_mandatory = require_runtime_contract == "1" or (
        bool(source_candidate_present) and not authorized_source_candidate
    )
    if runtime_contract_mandatory and not runtime_requeue_contract_verified:
        raise RuntimeError("checkpoint has no resolved runner-owned requeue contract")

    # The adjacent origin Hydra file can retain the pre-rebase schedule, and
    # full files legitimately change location/runtime fields on Skynet -> ICE.
    # Its byte hash is therefore named separately. The runner config_sha256 is
    # computed exclusively from embedded checkpoint semantics and stable run
    # identity, so it must remain identical across relocation.
    config_identity = {
        "schema_version": 1,
        "model": plain(tree.model),
        "run_identity": {
            "task_id": task_id,
            "topology": topology,
            "num_latent_tokens": tokens,
            "split_manifest_sha256": required("ICE_EXPECTED_SPLIT_SHA256"),
            "train_only_normalization_sha256": required(
                "ICE_EXPECTED_NORM_SHA256"
            ),
            "parameter_count": exact_environment_integer(
                "ICE_EXPECTED_PARAMETER_COUNT", positive=True
            ),
        },
        "wandb": {
            "id": wandb_id,
            **actual_wandb,
        },
    }
    config_identity_sha = hashlib.sha256(
        json.dumps(
            config_identity,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    expected_config_identity_sha = os.environ.get(
        "ICE_EXPECTED_CONFIG_IDENTITY_SHA256", ""
    )
    if expected_config_identity_sha:
        if not SHA256_RE.fullmatch(expected_config_identity_sha):
            raise RuntimeError("invalid ICE_EXPECTED_CONFIG_IDENTITY_SHA256")
        if config_identity_sha != expected_config_identity_sha:
            raise RuntimeError("checkpoint config identity SHA-256 mismatch")

    expected_lr_start = None
    expected_lr_final = None
    expected_lr_start_step = None
    expected_lr_end_step = None
    expected_current_lr = None
    if rebased_schedule_verified:
        (
            expected_lr_start,
            expected_lr_final,
            expected_lr_start_step,
            expected_lr_end_step,
            expected_current_lr,
        ) = validate_rebased_schedule(
            tree=tree,
            step=step,
            optimizer_lrs=optimizer_lrs,
            optimizer_initial_lrs=optimizer_initial_lrs,
            scheduler_base_lrs=scheduler_base_lrs,
            scheduler_last_lrs=scheduler_last_lrs,
        )

    strict_load = os.environ.get("ICE_STRICT_LOAD", "0") == "1"
    # A strict continuation gate must cover every serialized tensor, including
    # callback and hyperparameter-owned state, rather than only model weights.
    # Lightning initializes one kth-value infinity sentinel for the exact
    # monitor=None/save_top_k=-1 ModelCheckpoint configuration; verify its full
    # callback/config identity before exempting only that metadata scalar.
    verified_nonfinite_sentinels = (
        verified_model_checkpoint_sentinels(callbacks, config)
        if strict_load
        else {}
    )
    (
        tensor_count,
        tensor_numel,
        observed_nonfinite_framework_sentinels,
    ) = finite_tensor_scan_with_sentinels(
        payload,
        complete=strict_load,
        verified_nonfinite_sentinels=verified_nonfinite_sentinels,
    )
    strict_parameter_count = 0
    strict_ema_parameter_count = 0
    if strict_load:
        strict_parameter_count, strict_ema_parameter_count = strict_restore(
            path,
            tree,
            optimizer_states[0],
            scheduler_states[0],
            exact_environment_integer(
                "ICE_EXPECTED_PARAMETER_COUNT", positive=True
            ),
            ema_state,
            step,
        )

    expected_sha = os.environ.get("ICE_EXPECTED_CHECKPOINT_SHA256", "")
    if authorized_source_candidate:
        if expected_sha and expected_sha != authorized_source_candidate_sha:
            raise RuntimeError(
                "expected checkpoint SHA differs from authorized source candidate"
            )
        expected_sha = authorized_source_candidate_sha
    checkpoint_sha = "not_computed"
    signal_checkpoint = SIGNAL_CHECKPOINT_RE.fullmatch(path.name) is not None
    if expected_sha or signal_checkpoint:
        checkpoint_sha = sha256(path)
    if expected_sha:
        if not SHA256_RE.fullmatch(expected_sha):
            raise RuntimeError("invalid expected checkpoint SHA-256")
        if checkpoint_sha != expected_sha:
            raise RuntimeError("checkpoint SHA-256 mismatch")
    (
        save_only_signal_proof_verified,
        save_only_signal_proof_path,
        save_only_signal_proof,
    ) = validate_save_only_proof(
        checkpoint=path,
        checkpoint_sha256=checkpoint_sha,
        checkpoint_size_bytes=before[2],
        epoch=epoch,
        global_step=step,
    )
    if signal_checkpoint and not runtime_requeue_contract_verified:
        raise RuntimeError(
            "signal checkpoint has no embedded runner-owned requeue contract"
        )
    if runtime_contract_mandatory and not save_only_signal_proof_verified:
        raise RuntimeError(
            "post-cutover resume checkpoint has no save-only SIGUSR2 proof"
        )
    after_validation = stat_signature(path)
    if before != after_validation:
        raise RuntimeError("checkpoint changed while being validated")

    record = {
        "status": "passed",
        "checkpoint": str(path),
        "checkpoint_sha256": checkpoint_sha,
        "size_bytes": after_validation[2],
        "global_step": step,
        "epoch": epoch,
        "task_id": task_id,
        "topology": topology,
        "num_latent_tokens": tokens,
        "wandb_id": wandb_id,
        "wandb_entity": config_wandb_entity,
        "wandb_project": config_wandb_project,
        "wandb_group": config_wandb_group,
        "config_path": str(config_path),
        "checkpoint_config_role": "adjacent_origin_hydra",
        "config_sha256": config_identity_sha,
        "embedded_config_identity_sha256": config_identity_sha,
        "checkpoint_config_sha256": checkpoint_config_sha,
        "config_identity": config_identity,
        "runtime_requeue_contract_verified": runtime_requeue_contract_verified,
        "authorized_source_candidate_verified": authorized_source_candidate,
        "runtime_slurm_requeue_owner": (
            str(runtime_values["owner"]) if runtime_present else None
        ),
        "runtime_slurm_save_signal": (
            str(runtime_values["signal"]) if runtime_present else None
        ),
        "runtime_slurm_signal_checkpoint_dir": (
            str(Path(str(runtime_values["checkpoint_dir"])).resolve())
            if runtime_present
            else None
        ),
        "save_only_signal_proof_verified": save_only_signal_proof_verified,
        "save_only_signal_proof_path": (
            str(save_only_signal_proof_path)
            if save_only_signal_proof_path is not None
            else None
        ),
        "save_only_signal_proof": save_only_signal_proof,
        "tensor_count": tensor_count,
        "tensor_numel": tensor_numel,
        "tensor_finiteness": (
            "complete" if strict_load else "sampled_first_scalar"
        ),
        "verified_nonfinite_framework_sentinels": (
            observed_nonfinite_framework_sentinels
        ),
        "optimizer_state_count": 1,
        "optimizer_parameter_name_count": optimizer_parameter_name_count,
        "optimizer_lrs": optimizer_lrs,
        "optimizer_initial_lrs": optimizer_initial_lrs,
        "lr_scheduler_state_count": 1,
        "scheduler_base_lrs": scheduler_base_lrs,
        "scheduler_last_lrs": scheduler_last_lrs,
        "scheduler_last_epoch": step,
        "rebased_schedule_verified": rebased_schedule_verified,
        "expected_lr_start": expected_lr_start,
        "expected_lr_final": expected_lr_final,
        "expected_lr_start_step": expected_lr_start_step,
        "expected_lr_end_step": expected_lr_end_step,
        "expected_current_lr": expected_current_lr,
        "ema_num_updates": int(payload["ema_num_updates"]),
        "strict_load": strict_load,
        "full_state_verified": strict_load,
        "strict_parameter_count": strict_parameter_count,
        "strict_ema_parameter_count": strict_ema_parameter_count,
        "strict_optimizer_load": strict_load,
        "strict_scheduler_load": strict_load,
    }
    print(json.dumps(record, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
