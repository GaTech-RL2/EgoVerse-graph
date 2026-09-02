"""Verify a short real-data training smoke and emit a durable result record.

W&B 0.26 offline runs persist history in ``run-*.wandb`` and do not always
materialize ``files/wandb-summary.json``. This verifier reads the native W&B
stream so a smoke cannot fail merely because that compatibility file is absent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import hydra
import torch
from omegaconf import OmegaConf
from wandb.proto import wandb_internal_pb2
from wandb.sdk.internal.datastore import DataStore

import egomimic.utils.hydra_resolvers  # noqa: F401 -- project config resolvers
from egomimic.pl_utils.pl_model import ModelWrapper
from egomimic.rldb.embodiment.embodiment import get_embodiment

_HISTORY_METRIC_PREFIXES = (
    "Train/",
    "Timing/",
    "Optimizer/",
    "Valid/",
    "log/",
)
_DEFAULT_REQUIRED_STEP_METRICS = {
    "train_metrics": {"Train/Loss"},
    "timing_metrics": {
        "Timing/Process_Batch_Sec",
        "Timing/Forward_Pass_Sec",
        "Timing/Compute_Losses_Sec",
    },
    "optimizer_metrics": {"Optimizer/param_group_0_lr"},
}
_STEP_METRIC_CATEGORIES = {
    "Train/": "train_metrics",
    "Timing/": "timing_metrics",
    "Optimizer/": "optimizer_metrics",
    # Telemetry is sparse and is checked for at least one finite occurrence,
    # rather than being required on every optimizer step.
    "log/": "telemetry_metrics",
}


def _register_training_config_resolvers() -> None:
    """Mirror the resolvers registered by ``egomimic.trainHydra``."""

    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _item_key(item: Any) -> str | None:
    if item.key:
        return item.key
    if item.nested_key:
        return ".".join(item.nested_key)
    return None


def _history_row(record: Any) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for item in record.history.item:
        key = _item_key(item)
        if key is None:
            continue
        try:
            row[key] = json.loads(item.value_json)
        except (json.JSONDecodeError, TypeError):
            continue
    return row


def read_successful_wandb_exit_code(stream_path: Path) -> int:
    """Return a W&B stream's terminal success code without loading history."""

    store = DataStore()
    store.open_for_scan(str(stream_path))
    exit_codes: list[int] = []
    try:
        while True:
            payload = store.scan_data()
            if payload is None:
                break
            record = wandb_internal_pb2.Record()
            record.ParseFromString(payload)
            if record.WhichOneof("record_type") == "exit":
                exit_codes.append(int(record.exit.exit_code))
    finally:
        store.close()

    assert exit_codes, f"No terminal W&B exit record in {stream_path}"
    assert exit_codes[-1] == 0, (stream_path, exit_codes)
    return exit_codes[-1]


def read_wandb_history(
    stream_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    """Return aggregated train/validation rows and the terminal W&B exit code."""

    store = DataStore()
    store.open_for_scan(str(stream_path))
    history_by_step: dict[int, dict[str, float]] = {}
    exit_codes: list[int] = []
    try:
        while True:
            payload = store.scan_data()
            if payload is None:
                break
            record = wandb_internal_pb2.Record()
            record.ParseFromString(payload)
            record_type = record.WhichOneof("record_type")
            if record_type == "exit":
                exit_codes.append(int(record.exit.exit_code))
                continue
            if record_type != "history":
                continue

            row = _history_row(record)
            step = row.get("trainer/global_step")
            if step is None:
                continue
            step = int(step)
            metrics = history_by_step.setdefault(step, {})
            for key, value in row.items():
                if not key.startswith(_HISTORY_METRIC_PREFIXES):
                    continue
                try:
                    metrics[key] = float(value)
                except (TypeError, ValueError):
                    continue
    finally:
        store.close()

    assert exit_codes, f"No terminal W&B exit record in {stream_path}"
    assert exit_codes[-1] == 0, (stream_path, exit_codes)

    training_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    for step, metrics in sorted(history_by_step.items()):
        train_metrics = {
            key: value
            for key, value in metrics.items()
            if key.startswith("Train/") and not key.endswith("_epoch")
        }
        timing_metrics = {
            key: value
            for key, value in metrics.items()
            if key.startswith("Timing/") and not key.endswith("_epoch")
        }
        optimizer_metrics = {
            key: value
            for key, value in metrics.items()
            if key.startswith("Optimizer/") and not key.endswith("_epoch")
        }
        telemetry_metrics = {
            key: value for key, value in metrics.items() if key.startswith("log/")
        }
        if train_metrics or timing_metrics or optimizer_metrics or telemetry_metrics:
            training_row = {
                "trainer_global_step": step,
                "train_metrics": train_metrics,
                "timing_metrics": timing_metrics,
                "optimizer_metrics": optimizer_metrics,
            }
            # Preserve the historical row shape for ordinary policies while
            # exposing UNITE telemetry when it is actually present.
            if telemetry_metrics:
                training_row["telemetry_metrics"] = telemetry_metrics
            training_rows.append(training_row)
        validation_metrics = {
            key: value for key, value in metrics.items() if key.startswith("Valid/")
        }
        if validation_metrics:
            validation_rows.append(
                {
                    "trainer_global_step": step,
                    "validation_metrics": validation_metrics,
                }
            )
    return training_rows, validation_rows, exit_codes[-1]


def read_wandb_validation(
    stream_path: Path,
) -> tuple[list[dict[str, Any]], int]:
    """Backward-compatible validation-only view of the W&B history."""

    _, validation_rows, exit_code = read_wandb_history(stream_path)
    return validation_rows, exit_code


def _has_required_metrics(
    metrics: dict[str, float], required_embodiments: list[int]
) -> bool:
    """Retain the legacy per-embodiment action-MSE contract."""

    for embodiment in required_embodiments:
        prefix = f"Valid/emb{embodiment}_"
        if not any(
            key.startswith(prefix) and key.endswith("_action_mse") for key in metrics
        ):
            return False
    return True


def _required_metric_contract(
    config: Any,
) -> tuple[list[str] | None, dict[str, set[str]], set[str]]:
    """Resolve configured metric requirements and retain legacy smoke defaults."""

    configured = OmegaConf.select(
        config,
        "run_provenance.required_wandb_metrics",
        default=None,
    )
    required_step_metrics = {
        category: set(metrics)
        for category, metrics in _DEFAULT_REQUIRED_STEP_METRICS.items()
    }
    if configured is None:
        return None, required_step_metrics, set()

    configured_metrics = [str(metric) for metric in configured]
    assert configured_metrics, "required_wandb_metrics must not be empty"
    assert len(configured_metrics) == len(set(configured_metrics)), (
        "required_wandb_metrics contains duplicates",
        configured_metrics,
    )

    required_validation_metrics: set[str] = set()
    for metric in configured_metrics:
        assert metric and metric.startswith(_HISTORY_METRIC_PREFIXES), (
            "Unsupported required W&B metric namespace",
            metric,
        )
        if metric.startswith("Valid/"):
            required_validation_metrics.add(metric)
            continue
        category = next(
            category
            for prefix, category in _STEP_METRIC_CATEGORIES.items()
            if metric.startswith(prefix)
        )
        required_step_metrics.setdefault(category, set()).add(metric)

    return configured_metrics, required_step_metrics, required_validation_metrics


def _select_dense_training_history(
    training_history: list[dict[str, Any]],
    required_step_metrics: dict[str, set[str]],
    expected_steps: int = 2,
) -> tuple[list[dict[str, Any]], list[int]]:
    """Select exact dense optimizer rows; sparse telemetry is checked separately."""

    dense_requirements = {
        category: required
        for category, required in required_step_metrics.items()
        if category != "telemetry_metrics"
    }
    dense_training_history = [
        row
        for row in training_history
        if all(
            required.issubset(row.get(category, {}))
            for category, required in dense_requirements.items()
        )
    ]
    assert len(dense_training_history) == expected_steps, training_history
    training_steps = [row["trainer_global_step"] for row in dense_training_history]
    assert training_steps == list(range(expected_steps)), training_steps
    for row in dense_training_history:
        for category, required in dense_requirements.items():
            required_values = {
                metric: row[category][metric] for metric in sorted(required)
            }
            assert all(math.isfinite(value) for value in required_values.values()), (
                category,
                required_values,
            )
    return dense_training_history, training_steps


def _validate_required_telemetry(
    training_history: list[dict[str, Any]],
    required_step_metrics: dict[str, set[str]],
) -> list[dict[str, Any]]:
    """Require every declared sparse telemetry metric at a finite history row."""

    required = required_step_metrics.get("telemetry_metrics", set())
    records = []
    for metric in sorted(required):
        occurrences = [
            {
                "trainer_global_step": row["trainer_global_step"],
                "value": row.get("telemetry_metrics", {}).get(metric),
            }
            for row in training_history
            if metric in row.get("telemetry_metrics", {})
        ]
        assert occurrences, (metric, training_history)
        assert all(math.isfinite(record["value"]) for record in occurrences), (
            metric,
            occurrences,
        )
        records.append({"metric": metric, "occurrences": occurrences})
    return records


def _select_scheduled_validation(
    validation_history: list[dict[str, Any]],
    required_embodiments: list[int],
    required_validation_metrics: set[str],
    minimum_validation_step: int = 1,
) -> dict[str, Any]:
    """Select validation after the requested number of completed optimizer steps."""

    # Lightning associates validation metrics with the zero-based training step
    # that triggered validation. Completed optimizer steps are therefore +1.
    scheduled_history = [
        row
        for row in validation_history
        if row["trainer_global_step"] + 1 >= minimum_validation_step
    ]
    assert scheduled_history, validation_history
    if required_validation_metrics:
        qualifying_history = [
            row
            for row in scheduled_history
            if required_validation_metrics.issubset(row["validation_metrics"])
        ]
        assert qualifying_history, (
            sorted(required_validation_metrics),
            scheduled_history,
        )
        selected = qualifying_history[-1]
        required_values = {
            metric: selected["validation_metrics"][metric]
            for metric in sorted(required_validation_metrics)
        }
        assert all(math.isfinite(value) for value in required_values.values()), (
            required_values
        )
        return selected

    qualifying_history = [
        row
        for row in scheduled_history
        if _has_required_metrics(row["validation_metrics"], required_embodiments)
    ]
    assert qualifying_history, (required_embodiments, scheduled_history)
    selected = qualifying_history[-1]
    metrics = selected["validation_metrics"]
    for embodiment in required_embodiments:
        prefix = f"Valid/emb{embodiment}_"
        matches = {
            key: value
            for key, value in metrics.items()
            if key.startswith(prefix) and key.endswith("_action_mse")
        }
        assert matches, (embodiment, sorted(metrics))
        assert all(math.isfinite(value) for value in matches.values()), matches
    return selected


def _resolve_model_wrapper_class(config: Any | None) -> type[ModelWrapper]:
    """Resolve the checkpoint's configured Lightning wrapper class."""

    if config is None:
        return ModelWrapper
    target = OmegaConf.select(config, "model._target_", default=None)
    if target is None:
        return ModelWrapper
    wrapper_class = hydra.utils.get_class(str(target))
    if not isinstance(wrapper_class, type) or not issubclass(
        wrapper_class, ModelWrapper
    ):
        raise TypeError(
            "cfg.model._target_ must resolve to a ModelWrapper subclass; "
            f"got {target!r}"
        )
    return wrapper_class


def _strict_load_model_wrapper(
    checkpoint_path: Path,
    config: Any | None = None,
    expected_steps: int = 2,
) -> dict[str, Any]:
    """Reconstruct the checkpointed wrapper on CPU with an exact state load."""

    assert not torch.cuda.is_available(), (
        "strict CPU verification must run with CUDA hidden"
    )
    wrapper_class = _resolve_model_wrapper_class(config)
    wrapper = wrapper_class.load_from_checkpoint(
        str(checkpoint_path),
        map_location="cpu",
        weights_only=False,
        strict=True,
    )
    assert isinstance(wrapper, wrapper_class), type(wrapper)
    state_dict = wrapper.state_dict()
    non_cpu_tensors = {
        key: str(value.device)
        for key, value in state_dict.items()
        if value.device.type != "cpu"
    }
    assert not non_cpu_tensors, non_cpu_tensors
    ema_record = None
    if getattr(wrapper, "_ema_config", None) is not None:
        assert hasattr(wrapper, "ema_model")
        optimization_step = int(wrapper.ema_optimization_step.item())
        assert optimization_step == expected_steps, optimization_step
        expected_decay = wrapper._ema_decay_for_step(optimization_step)
        actual_decay = float(wrapper.ema_decay.item())
        assert math.isclose(actual_decay, expected_decay, abs_tol=1.0e-12), (
            actual_decay,
            expected_decay,
        )
        online = dict(wrapper.model.nets.named_parameters())
        averaged = dict(wrapper.ema_model.nets.named_parameters())
        assert online.keys() == averaged.keys()
        assert all(not parameter.requires_grad for parameter in averaged.values())
        assert all(torch.isfinite(parameter).all() for parameter in averaged.values())
        ema_record = {
            "backend": "model_wrapper",
            "enabled": True,
            "optimization_step": optimization_step,
            "decay": actual_decay,
            "power": float(wrapper._ema_config["power"]),
            "max_value": float(wrapper._ema_config["max_value"]),
            "use_for_validation": bool(wrapper._ema_config["use_for_validation"]),
            "parameter_tree_exact": True,
            "parameters_finite": True,
        }
    record = {
        "status": "passed",
        "model_class": f"{type(wrapper).__module__}.{type(wrapper).__qualname__}",
        "map_location": "cpu",
        "strict": True,
        "state_dict_key_count": len(state_dict),
        "parameter_count": sum(parameter.numel() for parameter in wrapper.parameters()),
        "buffer_count": sum(buffer.numel() for buffer in wrapper.buffers()),
    }
    if ema_record is not None:
        record["ema"] = ema_record
    del wrapper
    return record


def _configured_ema_backends(config: Any) -> tuple[Any | None, Any | None]:
    internal = OmegaConf.select(config, "model.ema", default=None)
    if internal is not None and not bool(internal.get("enabled", False)):
        internal = None
    external = OmegaConf.select(config, "callbacks.ema", default=None)
    assert not (internal is not None and external is not None), (
        "model.ema and callbacks.ema cannot both be enabled"
    )
    return internal, external


def _validate_external_ema_checkpoint(
    checkpoint: dict[str, Any],
    ema_config: Any,
    global_step: int,
) -> dict[str, Any]:
    assert str(ema_config._target_) == "egomimic.utils.ema_callback.EMACallback"
    decay = float(ema_config.decay)
    assert math.isfinite(decay) and 0.0 < decay < 1.0
    validate_with_ema = bool(ema_config.validate_with_ema)
    assert validate_with_ema is True

    ema_state_dict = checkpoint.get("ema_state_dict")
    assert ema_state_dict, "Smoke checkpoint has no EMA state"
    assert math.isclose(float(checkpoint["ema_decay"]), decay, abs_tol=1.0e-12)
    ema_num_updates = int(checkpoint["ema_num_updates"])
    assert ema_num_updates == global_step
    assert bool(checkpoint.get("ema_validate_with_ema")) is validate_with_ema
    ema_tensor_count = len(ema_state_dict)
    ema_parameter_count = sum(value.numel() for value in ema_state_dict.values())
    assert ema_parameter_count > 0
    for name, value in ema_state_dict.items():
        assert torch.is_tensor(value), name
        assert bool(torch.isfinite(value).all()), name
    return {
        "backend": "callback",
        "decay": decay,
        "num_updates": ema_num_updates,
        "validation_uses_ema": validate_with_ema,
        "tensor_count": ema_tensor_count,
        "parameter_count": ema_parameter_count,
    }


def _validate_energy_score_artifacts(
    output_dir: Path,
    config: Any,
    *,
    global_step: int,
    expected_world_size: int,
    required_embodiments: list[int],
) -> list[dict[str, Any]]:
    """Validate both single-domain Paper and per-domain UNITE artifacts."""

    energy = OmegaConf.select(config, "evaluator.energy_score", default=None)
    assert energy is not None and bool(energy.enabled)
    assert int(energy.sample_count) == 32
    assert int(energy.max_batches_per_rank) == 1

    seed_bank_path = Path(str(energy.seed_bank_path)).resolve()
    assert seed_bank_path.is_file(), seed_bank_path
    expected_seed_sha = str(energy.seed_bank_sha256)
    assert _sha256(seed_bank_path) == expected_seed_sha
    expected_seeds = json.loads(seed_bank_path.read_text())["seeds"]
    assert len(expected_seeds) == 32 and len(set(expected_seeds)) == 32

    artifact_root = Path(str(energy.artifact_root)).resolve()
    assert (
        artifact_root
        == (output_dir / "validation_predictions" / "energy_score").resolve()
    )
    candidates = sorted(
        artifact_root.glob(f"epoch-*-step-{global_step}/rank-*-batch-*.pt")
    )
    assert len(candidates) == expected_world_size, candidates

    embodiment_to_domain = {
        int(embodiment): get_embodiment(embodiment).lower()
        for embodiment in required_embodiments
    }
    expected_domains = set(embodiment_to_domain.values())
    configured_action_dims = OmegaConf.select(energy, "action_dims", default=None)
    if configured_action_dims is not None:
        action_dims = {
            str(name).lower(): int(width)
            for name, width in configured_action_dims.items()
        }
        assert set(action_dims) == expected_domains, action_dims
    else:
        assert len(expected_domains) == 1, expected_domains
        action_dim = int(OmegaConf.select(energy, "action_dim"))
        action_dims = {next(iter(expected_domains)): action_dim}

    records = []
    seen_ranks = set()
    seen_embodiments = set()
    for path in candidates:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        assert payload["schema_version"] == 1
        assert payload["metric"] == "EnergyScore@32"
        assert payload["sample_count"] == 32
        assert payload["seed_bank"] == expected_seeds
        assert payload["seed_bank_sha256"] == expected_seed_sha
        assert int(payload["global_step"]) == global_step
        rank = int(payload["rank"])
        assert rank not in seen_ranks
        seen_ranks.add(rank)

        domains = payload["domains"]
        assert isinstance(domains, dict) and set(domains) == expected_domains
        for domain_name, artifact in domains.items():
            emb_id = int(artifact["embodiment_id"])
            assert embodiment_to_domain[emb_id] == domain_name
            seen_embodiments.add(emb_id)
            predictions = artifact["predictions"]
            targets = artifact["targets"]
            assert predictions.ndim == 4 and predictions.shape[0] == 32
            assert targets.ndim == 3 and predictions.shape[1:] == targets.shape
            assert predictions.shape[-1] == action_dims[domain_name]
            assert predictions.numel() and targets.numel()
            assert bool(torch.isfinite(predictions).all())
            assert bool(torch.isfinite(targets).all())
            for key in (
                "accuracy_by_condition",
                "diversity_by_condition",
                "score_by_condition",
            ):
                values = artifact[key]
                assert values.shape == targets.shape[:1]
                assert bool(torch.isfinite(values).all()), (path, domain_name, key)
        records.append(
            {
                "path": str(path),
                "sha256": _sha256(path),
                "rank": rank,
            }
        )
    assert seen_ranks == set(range(expected_world_size)), seen_ranks
    assert seen_embodiments == set(required_embodiments), seen_embodiments
    return records


def _validate_unite_alternating_contract(
    config: Any,
    training_history: list[dict[str, Any]],
    expected_steps: int,
    external_ema_enabled: bool,
) -> dict[str, Any] | None:
    """Validate legacy alternating UNITE without weakening register-sweep gates."""

    stages = list(config.model.robomimic_model.stages)
    stage_names = [str(stage._target_).rsplit(".", 1)[-1] for stage in stages]
    released_register_policy = "ReleasedRecipeUniteLatentPolicy" in stage_names
    flow_updates_per_reconstruction = int(
        config.model.get("unite_flow_updates_per_reconstruction", 0)
    )
    telemetry_cadence = int(
        config.model.get("unite_gradient_telemetry_every_n_steps", 0)
    )

    if released_register_policy:
        assert flow_updates_per_reconstruction == 0
        assert telemetry_cadence == 0
        raise AssertionError(
            "Released UNITE register-sweep rows remain launch-blocked until "
            "joint-update-safe topology telemetry, immutable split/normalization "
            "artifacts, parameter manifests, and the required smoke are implemented"
        )
    if flow_updates_per_reconstruction <= 0:
        return None

    assert telemetry_cadence > 0
    assert expected_steps >= flow_updates_per_reconstruction + 1
    assert stage_names.count("UniteLatentPolicy") == 1
    assert "UniteSharedDenoiser" not in stage_names
    assert "UnitePerEmbodimentActionDecoder" not in stage_names
    policy = stages[stage_names.index("UniteLatentPolicy")]
    timestep_shift_alpha = float(policy.timestep_shift_alpha)
    reconstruction_noise_std = float(policy.reconstruction_noise_std)
    assert math.isclose(timestep_shift_alpha, 0.5, abs_tol=1.0e-12)
    assert math.isclose(reconstruction_noise_std, 0.7, abs_tol=1.0e-12)

    required_schedule = {
        "log/unite_update_is_flow",
        "log/unite_update_is_reconstruction",
        "log/unite_update_cycle_position",
    }
    schedule_history = [
        {
            "trainer_global_step": row["trainer_global_step"],
            "telemetry_metrics": row.get("telemetry_metrics", {}),
        }
        for row in training_history
        if required_schedule.issubset(row.get("telemetry_metrics", {}))
    ]
    assert len(schedule_history) == expected_steps, schedule_history
    for expected_step, row in enumerate(schedule_history):
        assert row["trainer_global_step"] == expected_step, row
        metrics = row["telemetry_metrics"]
        expected_position = expected_step % (flow_updates_per_reconstruction + 1)
        expected_flow = float(expected_position < flow_updates_per_reconstruction)
        assert metrics["log/unite_update_cycle_position"] == float(expected_position), (
            row
        )
        assert metrics["log/unite_update_is_flow"] == expected_flow, row
        assert metrics["log/unite_update_is_reconstruction"] == 1.0 - expected_flow

    required_telemetry = {
        "log/unite_gradient_cosine",
        "log/unite_recon_grad_norm",
        "log/unite_denoise_grad_norm",
    }
    telemetry_history = [
        {
            "trainer_global_step": row["trainer_global_step"],
            "telemetry_metrics": row.get("telemetry_metrics", {}),
        }
        for row in training_history
        if required_telemetry.issubset(row.get("telemetry_metrics", {}))
    ]
    assert telemetry_history, training_history
    for row in telemetry_history:
        metrics = row["telemetry_metrics"]
        assert all(math.isfinite(metrics[key]) for key in required_telemetry), row
        assert metrics["log/unite_recon_grad_norm"] > 0.0, row
        assert metrics["log/unite_denoise_grad_norm"] > 0.0, row
        assert -1.000001 <= metrics["log/unite_gradient_cosine"] <= 1.000001, row

    ema_history = []
    if external_ema_enabled:
        ema_keys = {"log/unite_ema_decay", "log/unite_ema_num_updates"}
        ema_history = [
            {
                "trainer_global_step": row["trainer_global_step"],
                "telemetry_metrics": row.get("telemetry_metrics", {}),
            }
            for row in training_history
            if ema_keys.issubset(row.get("telemetry_metrics", {}))
        ]
        assert ema_history, training_history
        for row in ema_history:
            metrics = row["telemetry_metrics"]
            assert math.isfinite(metrics["log/unite_ema_decay"]), row
            assert math.isfinite(metrics["log/unite_ema_num_updates"]), row
            assert metrics["log/unite_ema_num_updates"] >= 1.0, row

    return {
        "unite_topology": "normal_unite_latent_policy",
        "timestep_shift_alpha": timestep_shift_alpha,
        "reconstruction_noise_std": reconstruction_noise_std,
        "unite_update_schedule": {
            "flow_updates_per_reconstruction": flow_updates_per_reconstruction,
            "cycle_length_optimizer_steps": flow_updates_per_reconstruction + 1,
            "telemetry_cadence_optimizer_steps": telemetry_cadence,
            "history": schedule_history,
        },
        "unite_gradient_telemetry": telemetry_history,
        "unite_ema_history": ema_history,
    }


def _load_training_config(config_path: Path):
    _register_training_config_resolvers()
    return OmegaConf.load(config_path)


def verify_training_smoke(
    output_dir: Path,
    required_embodiments: list[int],
    expected_head: str,
    expected_strategy: str = "ddp",
    expected_world_size: int = 1,
    expected_steps: int = 2,
    expected_val_check_interval: int = 1,
    minimum_validation_step: int = 1,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    config_path = output_dir / ".hydra" / "config.yaml"
    assert config_path.is_file(), config_path
    config = _load_training_config(config_path)

    assert expected_steps > 0
    assert expected_val_check_interval > 0
    assert minimum_validation_step >= 0
    assert int(config.trainer.max_steps) == expected_steps
    assert int(config.trainer.limit_train_batches) == expected_steps
    assert int(config.trainer.val_check_interval) == expected_val_check_interval
    assert int(config.trainer.limit_val_batches) == 1
    assert int(config.trainer.num_sanity_val_steps) == 0
    assert int(config.trainer.log_every_n_steps) == 1
    assert str(config.trainer.precision) == "bf16"
    assert str(config.trainer.strategy) == expected_strategy
    assert int(config.launch_params.gpus_per_node) == expected_world_size
    assert int(config.launch_params.nodes) == 1
    assert int(config.trainer.devices) == expected_world_size
    assert int(config.trainer.num_nodes) == 1
    assert config.model.train_metrics_on_step is True
    assert (
        config.evaluator._target_
        == "egomimic.eval.human_robot_overlay_eval.HumanRobotOverlayEval"
    )

    (
        required_wandb_metrics,
        required_step_metrics,
        required_validation_metrics,
    ) = _required_metric_contract(config)
    internal_ema_config, external_ema_config = _configured_ema_backends(config)

    checkpoint_path = output_dir / "checkpoints" / "last.ckpt"
    assert checkpoint_path.is_file(), checkpoint_path
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    global_step = int(checkpoint["global_step"])
    epoch = int(checkpoint["epoch"])
    assert global_step == expected_steps, global_step
    optimizer_states = checkpoint.get("optimizer_states", [])
    assert optimizer_states, "Smoke checkpoint has no optimizer state"
    optimizer_state_count = len(optimizer_states)
    optimizer_lrs = [
        float(group["lr"])
        for state in optimizer_states
        for group in state.get("param_groups", [])
    ]
    assert optimizer_lrs and all(math.isfinite(value) for value in optimizer_lrs)
    scheduler_states = checkpoint.get("lr_schedulers", [])
    assert len(scheduler_states) == 1, scheduler_states
    scheduler_last_epoch = int(scheduler_states[0]["last_epoch"])
    assert scheduler_last_epoch == global_step, scheduler_states[0]
    hyper_parameters = checkpoint.get("hyper_parameters", {})
    assert hyper_parameters.get("train_metrics_on_step") is True

    model_wrapper_load = _strict_load_model_wrapper(
        checkpoint_path,
        config=config,
        expected_steps=expected_steps,
    )
    ema_record = None
    if internal_ema_config is not None:
        ema_record = model_wrapper_load.get("ema")
        assert isinstance(ema_record, dict) and ema_record.get("enabled") is True
        assert int(ema_record["optimization_step"]) == global_step
        assert math.isclose(
            float(ema_record["power"]), float(internal_ema_config.power)
        )
        assert math.isclose(
            float(ema_record["max_value"]), float(internal_ema_config.max_value)
        )
        assert ema_record["use_for_validation"] is bool(
            internal_ema_config.use_for_validation
        )
    elif external_ema_config is not None:
        ema_record = _validate_external_ema_checkpoint(
            checkpoint,
            external_ema_config,
            global_step,
        )
    del checkpoint, hyper_parameters, optimizer_states, scheduler_states

    streams = [
        *output_dir.glob("wandb/offline-run-*/run-*.wandb"),
        *output_dir.glob("wandb/run-*/run-*.wandb"),
    ]
    assert len(streams) == 1, streams
    training_history, validation_history, wandb_exit_code = read_wandb_history(
        streams[0]
    )

    dense_training_history, training_steps = _select_dense_training_history(
        training_history,
        required_step_metrics,
        expected_steps=expected_steps,
    )
    required_telemetry_history = _validate_required_telemetry(
        training_history,
        required_step_metrics,
    )
    unite_record = _validate_unite_alternating_contract(
        config,
        training_history,
        expected_steps,
        external_ema_enabled=external_ema_config is not None,
    )
    selected = _select_scheduled_validation(
        validation_history,
        required_embodiments,
        required_validation_metrics,
        minimum_validation_step=minimum_validation_step,
    )
    metrics = selected["validation_metrics"]
    assert all(math.isfinite(value) for value in metrics.values()), metrics
    validation_after_optimizer_steps = selected["trainer_global_step"] + 1
    assert validation_after_optimizer_steps <= global_step
    energy_score_artifacts = _validate_energy_score_artifacts(
        output_dir,
        config,
        global_step=validation_after_optimizer_steps,
        expected_world_size=expected_world_size,
        required_embodiments=required_embodiments,
    )

    record = {
        "status": "passed",
        "repo_head": expected_head,
        "output": str(output_dir),
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "global_step": global_step,
        "epoch": epoch,
        "precision": str(config.trainer.precision),
        "world_size": expected_world_size,
        "expected_steps": expected_steps,
        "expected_val_check_interval": expected_val_check_interval,
        "minimum_validation_step": minimum_validation_step,
        "optimizer_state_count": optimizer_state_count,
        "optimizer_lrs": optimizer_lrs,
        "scheduler_last_epoch": scheduler_last_epoch,
        "model_wrapper_load": model_wrapper_load,
        "required_embodiments": required_embodiments,
        "trainer_strategy": str(config.trainer.strategy),
        "required_wandb_metrics": required_wandb_metrics,
        "required_step_metrics": {
            category: sorted(required)
            for category, required in required_step_metrics.items()
        },
        "required_validation_metrics": sorted(required_validation_metrics),
        "required_telemetry_history": required_telemetry_history,
        "wandb_stream": str(streams[0]),
        "wandb_stream_sha256": _sha256(streams[0]),
        "wandb_exit_code": wandb_exit_code,
        "validation_trainer_global_step": selected["trainer_global_step"],
        "validation_after_optimizer_steps": validation_after_optimizer_steps,
        "validation_metrics": metrics,
        "energy_score_artifacts": energy_score_artifacts,
        "training_history": training_history,
        "dense_training_steps": training_steps,
        "validation_history": validation_history,
    }
    if ema_record is not None:
        record["ema"] = ema_record
    if unite_record is not None:
        record.update(unite_record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--required-embodiments", required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--expected-strategy", default="ddp")
    parser.add_argument("--expected-world-size", type=int, default=1)
    parser.add_argument("--expected-steps", type=int, default=2)
    parser.add_argument("--expected-val-check-interval", type=int, default=1)
    parser.add_argument("--minimum-validation-step", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    required_embodiments = [
        int(piece) for piece in args.required_embodiments.split(",") if piece.strip()
    ]
    assert required_embodiments
    record = verify_training_smoke(
        args.output_dir,
        required_embodiments,
        args.expected_head,
        expected_strategy=args.expected_strategy,
        expected_world_size=args.expected_world_size,
        expected_steps=args.expected_steps,
        expected_val_check_interval=args.expected_val_check_interval,
        minimum_validation_step=args.minimum_validation_step,
    )
    if not args.dry_run:
        result_path = args.output_dir.resolve() / "SMOKE_RESULT.json"
        temporary_path = result_path.with_suffix(".json.tmp")
        assert not result_path.exists(), f"Refusing to overwrite {result_path}"
        assert not temporary_path.exists(), f"Stale temporary result: {temporary_path}"
        temporary_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        temporary_path.replace(result_path)
    label = "VERIFY_PASS" if args.dry_run else "PASS"
    print(f"[smoke] {label} {json.dumps(record, sort_keys=True)}")


if __name__ == "__main__":
    main()
