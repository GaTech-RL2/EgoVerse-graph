"""Lightning integration for an Action Flow ``PipelineAlgo`` graph."""

from __future__ import annotations

import hashlib
import json
import time
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from itertools import combinations
from typing import Any

import torch
import torch.nn as nn

from egomimic.pl_utils.action_flow_diagnostic_forward import (
    clone_inference_tensors,
    collect_action_flow_diagnostics,
    cuda_devices,
    resolve_action_flow_topology,
)
from egomimic.pl_utils.pl_model import ModelWrapper


class ActionFlowModelWrapper(ModelWrapper):
    """Optimize the explicit Action Flow total and expose component telemetry.

    The pipeline owns all tensor transformations and model calls. This wrapper
    depends only on namespaced tensor keys, so source identifiers remain opaque
    and inference does not need a data-specific adapter.
    """

    _optimizer_key = "loss/action_flow"
    _metric_specs = (
        ("TotalLoss", "log/action_flow_total"),
        ("FlowMatchingLoss", "log/action_flow_fm"),
        ("ReconstructionLoss", "log/action_flow_reconstruction"),
        ("ReconstructionL1", "log/action_flow_reconstruction_l1"),
        ("ActionVelocityLoss", "log/action_flow_action_velocity"),
    )
    _gradient_components = (
        ("FM", "FlowMatchingLoss"),
        ("Reconstruction", "ReconstructionLoss"),
        ("ActionVelocity", "ActionVelocityLoss"),
    )

    def __init__(
        self,
        *,
        gradient_telemetry_cadence: int | None = None,
        reconstruction_only_warmup_steps: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        config_tree = getattr(self.hparams, "config_tree", None)
        configured_warmup_steps = None
        if config_tree is not None:
            configured_warmup_steps = self._as_config(config_tree).model.get(
                "reconstruction_only_warmup_steps", None
            )
        effective_warmup_steps = (
            reconstruction_only_warmup_steps
            if reconstruction_only_warmup_steps is not None
            else (
                0
                if configured_warmup_steps is None
                else configured_warmup_steps
            )
        )
        if (
            isinstance(effective_warmup_steps, bool)
            or not isinstance(effective_warmup_steps, int)
            or effective_warmup_steps < 0
        ):
            raise ValueError(
                "reconstruction_only_warmup_steps must be a nonnegative integer"
            )
        self.reconstruction_only_warmup_steps = effective_warmup_steps
        self.save_hyperparameters(
            {
                "reconstruction_only_warmup_steps": (
                    effective_warmup_steps
                )
            }
        )
        configured = None
        if config_tree is not None:
            configured = self._as_config(config_tree).model.get(
                "gradient_telemetry_cadence", None
            )
        cadence = (
            gradient_telemetry_cadence
            if gradient_telemetry_cadence is not None
            else (100 if configured is None else configured)
        )
        if isinstance(cadence, bool) or not isinstance(cadence, int) or cadence < 0:
            raise ValueError("gradient_telemetry_cadence must be a nonnegative integer")
        self.gradient_telemetry_cadence = cadence
        self._action_flow_validation_sums: OrderedDict[str, torch.Tensor] = (
            OrderedDict()
        )
        self._action_flow_validation_count = 0
        self._active_validation_batch: Mapping | None = None
        self._gradient_route_manifest: dict[str, Any] | None = None

        configured_samples = None
        if config_tree is not None:
            configured_samples = self._as_config(config_tree).model.get(
                "flow_samples_per_content", None
            )
        if configured_samples is None:
            stages = getattr(getattr(self.model, "pipeline", None), "stages", ())
            configured_samples = next(
                (
                    stage.samples_per_content
                    for stage in stages
                    if hasattr(stage, "samples_per_content")
                ),
                None,
            )
        if configured_samples is not None:
            if (
                isinstance(configured_samples, bool)
                or not isinstance(configured_samples, int)
                or configured_samples <= 0
            ):
                raise ValueError("flow_samples_per_content must be a positive integer")
        self.flow_samples_per_content = configured_samples

    def _objective_weight(self, name: str, *, default: float | None = None) -> float:
        direct = getattr(self.model, name, None)
        if direct is not None:
            return float(direct)
        stages = getattr(getattr(self.model, "pipeline", None), "stages", ())
        weight = next(
            (
                float(getattr(stage, name))
                for stage in stages
                if all(
                    hasattr(stage, name)
                    for name in (
                        "flow_weight",
                        "reconstruction_weight",
                        "action_velocity_weight",
                    )
                )
            ),
            None,
        )
        if weight is None:
            if default is not None:
                return float(default)
            raise RuntimeError(f"Action Flow objective weight {name!r} is missing")
        return weight

    def _objective_reconstruction_weight(self) -> float:
        return self._objective_weight("reconstruction_weight")

    def _apply_reconstruction_only_warmup(
        self,
        predictions: Mapping,
        *,
        optimizer_step: int | None = None,
    ) -> bool:
        """Gate the optimizer scalar while retaining raw component telemetry."""

        step = int(self.global_step) if optimizer_step is None else int(optimizer_step)
        active = step < self.reconstruction_only_warmup_steps
        if not active:
            return False
        reconstruction_weight = self._objective_reconstruction_weight()
        for source, result in predictions.items():
            if not isinstance(result, Mapping):
                raise TypeError(
                    f"Action Flow result for source {source!r} must be a mapping"
                )
            reconstruction = self._finite_scalar(
                result.get("log/action_flow_reconstruction"),
                f"{source!r}/reconstruction warmup",
            )
            scheduled_total = reconstruction_weight * reconstruction
            result["loss/action_flow"] = scheduled_total
            result["log/action_flow_total"] = scheduled_total
        return True

    def _action_flow_topology(self) -> tuple[Any, Any, Any]:
        """Resolve the exact registered encoder, field, and decoder stages."""

        return resolve_action_flow_topology(self.model)

    @property
    def encoder_e(self) -> nn.Module:
        """The exact encoder instance registered by the pipeline."""

        return self._action_flow_topology()[0].encoder

    @property
    def field_v(self) -> nn.Module:
        """The exact conditional field instance registered by the pipeline."""

        return self._action_flow_topology()[1].field

    @property
    def decoder_g(self) -> nn.Module:
        """The exact decoder instance registered by the pipeline."""

        return self._action_flow_topology()[2].decoder

    @staticmethod
    def _finite_scalar(value: Any, label: str) -> torch.Tensor:
        if not torch.is_tensor(value) or value.ndim != 0:
            raise TypeError(f"{label} must be a scalar tensor")
        if not bool(torch.isfinite(value.detach())):
            raise RuntimeError(f"Non-finite Action Flow value {label}")
        return value

    @classmethod
    def _source_values(
        cls, predictions: Mapping
    ) -> tuple[
        OrderedDict[str, tuple[int, OrderedDict[str, torch.Tensor]]],
        OrderedDict[str, torch.Tensor],
        torch.Tensor,
        int,
    ]:
        if not isinstance(predictions, Mapping) or not predictions:
            raise RuntimeError("Action Flow received no source predictions")

        per_source = OrderedDict()
        sums = OrderedDict((name, None) for name, _ in cls._metric_specs)
        optimizer_sum = None
        total_count = 0
        for source, result in predictions.items():
            if not isinstance(source, str) or not source:
                raise TypeError("Action Flow source keys must be non-empty strings")
            if not isinstance(result, Mapping):
                raise TypeError(
                    f"Action Flow result for source {source!r} must be a mapping"
                )
            target = result.get("target")
            if not torch.is_tensor(target) or target.ndim == 0:
                raise TypeError(
                    f"Action Flow source {source!r} target must be a batched tensor"
                )
            count = int(target.shape[0])
            if count <= 0:
                raise RuntimeError(f"Action Flow source {source!r} has no samples")

            optimizer_loss = cls._finite_scalar(
                result.get(cls._optimizer_key), f"{source!r}/{cls._optimizer_key}"
            )
            metrics = OrderedDict(
                (
                    name,
                    cls._finite_scalar(result.get(key), f"{source!r}/{key}"),
                )
                for name, key in cls._metric_specs
            )
            if not bool(
                torch.allclose(
                    optimizer_loss.detach(),
                    metrics["TotalLoss"].detach(),
                    rtol=1.0e-6,
                    atol=1.0e-8,
                )
            ):
                raise RuntimeError(
                    f"Action Flow optimizer loss and total metric disagree for {source!r}"
                )

            per_source[source] = (count, metrics)
            total_count += count
            weighted_optimizer = optimizer_loss * count
            optimizer_sum = (
                weighted_optimizer
                if optimizer_sum is None
                else optimizer_sum + weighted_optimizer
            )
            for name, value in metrics.items():
                weighted = value * count
                sums[name] = weighted if sums[name] is None else sums[name] + weighted

        components = OrderedDict(
            (name, value / total_count) for name, value in sums.items()
        )
        optimizer_loss = optimizer_sum / total_count
        cls._finite_scalar(optimizer_loss, cls._optimizer_key)
        return per_source, components, optimizer_loss, total_count

    @staticmethod
    def _reduce_component_means(
        components: Mapping[str, torch.Tensor], count: int
    ) -> tuple[OrderedDict[str, torch.Tensor], int]:
        if count <= 0:
            raise RuntimeError("Action Flow metric reduction has no samples")
        names = tuple(components)
        first = components[names[0]]
        payload = torch.stack(
            (
                *(components[name].detach().double() * count for name in names),
                torch.tensor(float(count), device=first.device, dtype=torch.float64),
            )
        )
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(payload, op=torch.distributed.ReduceOp.SUM)
        if not bool(torch.isfinite(payload).all()) or float(payload[-1]) <= 0.0:
            raise RuntimeError("Non-finite Action Flow distributed metric reduction")
        means = OrderedDict(
            (name, (payload[index] / payload[-1]).to(first.dtype))
            for index, name in enumerate(names)
        )
        return means, int(payload[-1].item())

    def _log_components(
        self,
        per_source: Mapping[str, tuple[int, Mapping[str, torch.Tensor]]],
        components: Mapping[str, torch.Tensor],
        count: int,
    ) -> None:
        reduced, global_count = self._reduce_component_means(components, count)
        for name, value in reduced.items():
            self.log(
                f"Train/ActionFlow/{name}",
                self._finite_scalar(value, name),
                on_step=True,
                on_epoch=True,
                sync_dist=False,
                batch_size=global_count,
            )
        for source, (source_count, source_values) in per_source.items():
            for name, value in source_values.items():
                self.log(
                    f"Train/ActionFlow/{name}/{source}",
                    value.detach(),
                    on_step=True,
                    on_epoch=True,
                    sync_dist=True,
                    batch_size=source_count,
                )

        # The canonical train-MSE contract is normalized clean-action
        # reconstruction MSE. Keep the generic alias explicit rather than
        # relying on an unrelated diffusion metric name.
        reconstruction = reduced["ReconstructionLoss"]
        self.log(
            "Train/MSE",
            self._finite_scalar(reconstruction, "normalized train MSE"),
            on_step=True,
            on_epoch=True,
            sync_dist=False,
            batch_size=global_count,
        )
        for source, (source_count, source_values) in per_source.items():
            self.log(
                f"Train/MSE/{source}",
                self._finite_scalar(
                    source_values["ReconstructionLoss"],
                    f"{source!r} normalized train MSE",
                ).detach(),
                on_step=True,
                on_epoch=True,
                sync_dist=True,
                batch_size=source_count,
            )

    def _log_extra_metrics(self, predictions: Mapping, reference: torch.Tensor) -> None:
        reserved = {key.removeprefix("log/") for _, key in self._metric_specs}
        for metric, source_values in self._prediction_log_metrics(
            predictions, reference
        ).items():
            if metric in reserved:
                continue
            for source, value in source_values:
                self.log(
                    f"Train/{metric}/{source}",
                    value,
                    sync_dist=True,
                    on_step=False,
                    on_epoch=True,
                )
            self.log(
                f"Train/{metric}",
                torch.stack([value for _, value in source_values]).mean(),
                sync_dist=True,
                on_step=False,
                on_epoch=True,
            )

    @staticmethod
    def _distributed_gradient(gradient: torch.Tensor) -> torch.Tensor:
        value = gradient.detach().float().clone()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.SUM)
            value.div_(torch.distributed.get_world_size())
        if not bool(torch.isfinite(value).all()):
            raise RuntimeError("Non-finite Action Flow component gradient")
        return value

    def _component_gradients(
        self,
        loss: torch.Tensor,
        named: Sequence[tuple[str, nn.Parameter]],
        label: str,
    ) -> OrderedDict[int, torch.Tensor]:
        if not loss.requires_grad:
            raise RuntimeError(
                f"Action Flow {label} must retain its autograd graph for telemetry"
            )
        gradients = torch.autograd.grad(
            loss,
            tuple(parameter for _, parameter in named),
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )
        active = OrderedDict(
            (index, self._distributed_gradient(gradient))
            for index, gradient in enumerate(gradients)
            if gradient is not None
        )
        if not active:
            raise RuntimeError(f"Action Flow {label} reaches no trainable parameters")
        return active

    def _log_gradient_telemetry(self, components: Mapping[str, torch.Tensor]) -> None:
        named = tuple(
            (name, parameter)
            for name, parameter in self.nets.named_parameters(
                prefix="nets", remove_duplicate=True
            )
            if parameter.requires_grad
        )
        if not named:
            raise RuntimeError("Action Flow has no trainable parameters")

        gradients = OrderedDict()
        routes = OrderedDict()
        for label, component_name in self._gradient_components:
            # An exact-section codec still reports its reconstruction error,
            # but that diagnostic is not a trained objective.
            if label == "Reconstruction" and self._objective_reconstruction_weight() == 0:
                continue
            active = self._component_gradients(components[component_name], named, label)
            gradients[label] = active
            routes[label] = [
                {
                    "dtype": str(named[index][1].dtype),
                    "name": named[index][0],
                    "numel": int(named[index][1].numel()),
                    "shape": list(named[index][1].shape),
                }
                for index in active
            ]
            norm = sum(
                (gradient.square().sum() for gradient in active.values()),
                next(iter(active.values())).new_zeros(()),
            ).sqrt()
            self._log_telemetry(f"GradientNorm/{label}", norm)
            self._log_telemetry(
                f"GradientParameterCount/{label}",
                sum(named[index][1].numel() for index in active),
            )

        for (left, left_gradients), (right, right_gradients) in combinations(
            gradients.items(), 2
        ):
            shared = tuple(
                index for index in left_gradients if index in right_gradients
            )
            if not shared:
                # FM-only endpoint detachment deliberately separates FM from
                # clean reconstruction. Do not invent a cosine for that pair.
                if {left, right} != {"FM", "Reconstruction"} or not self._fm_endpoint_detached():
                    raise RuntimeError(
                        f"Action Flow {left} and {right} have no shared gradient path"
                    )
                pair = f"{left}__{right}"
                self._log_telemetry(f"GradientCosine/{pair}", 0.0)
                self._log_telemetry(f"GradientCosineDefined/{pair}", 0.0)
                self._log_telemetry(f"GradientIntersectionParameterCount/{pair}", 0)
                continue
            zero = left_gradients[shared[0]].new_zeros(())
            dot = sum(
                (left_gradients[index] * right_gradients[index]).sum()
                for index in shared
            )
            left_norm = sum(
                (left_gradients[index].square().sum() for index in shared), zero
            ).sqrt()
            right_norm = sum(
                (right_gradients[index].square().sum() for index in shared), zero
            ).sqrt()
            denominator = left_norm * right_norm
            defined = bool(float(denominator) > 0.0)
            cosine = (
                (dot / denominator).clamp(-1.0, 1.0)
                if defined
                else denominator.new_zeros(())
            )
            pair = f"{left}__{right}"
            self._log_telemetry(f"GradientCosine/{pair}", cosine)
            self._log_telemetry(f"GradientCosineDefined/{pair}", float(defined))
            self._log_telemetry(
                f"GradientIntersectionParameterCount/{pair}",
                sum(named[index][1].numel() for index in shared),
            )

        route_hashes = OrderedDict(
            (
                label,
                hashlib.sha256(
                    json.dumps(
                        route,
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=True,
                    ).encode("utf-8")
                ).hexdigest(),
            )
            for label, route in routes.items()
        )
        intersection_names = OrderedDict()
        for left, right in combinations(routes, 2):
            right_names = {entry["name"] for entry in routes[right]}
            intersection_names[f"{left}__{right}"] = [
                entry["name"]
                for entry in routes[left]
                if entry["name"] in right_names
            ]
        manifest_core = {
            "routes": routes,
            "route_sha256": route_hashes,
            "intersections": intersection_names,
            "schema_version": 1,
        }
        self._gradient_route_manifest = {
            **manifest_core,
            "manifest_sha256": hashlib.sha256(
                json.dumps(
                    manifest_core,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                ).encode("utf-8")
            ).hexdigest(),
        }

    def _fm_endpoint_detached(self) -> bool:
        stages = getattr(getattr(self.model, "pipeline", None), "stages", ())
        return any(
            getattr(stage, "flow_clean_gradient_mode", "full") == "all_stopgrad"
            for stage in stages
        )

    def _log_compute_contract(self) -> None:
        if self.flow_samples_per_content is None:
            return
        field_calls = 2 if self._fm_endpoint_detached() else 1
        for name, value in (
            ("Compute/FieldForwardCallsPerStep", field_calls),
            (
                "Compute/FieldSampleEquivalentsPerStep",
                field_calls * self.flow_samples_per_content,
            ),
            ("Compute/DecoderJVPCallsPerStep", 1),
        ):
            self._log_telemetry(name, value)

    def _log_telemetry(self, name: str, value: Any) -> None:
        scalar = torch.as_tensor(value, device=self.device, dtype=torch.float32)
        self._finite_scalar(scalar, name)
        self.log(
            f"Train/ActionFlow/{name}",
            scalar,
            on_step=True,
            on_epoch=False,
            sync_dist=False,
        )

    def training_step(self, batch, batch_idx):
        del batch_idx
        self.train()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        start = time.time()
        batch = self.model.process_batch_for_training(batch)
        processed = time.time()
        predictions = self.model.forward_training(batch)
        forwarded = time.time()
        reconstruction_only = self._apply_reconstruction_only_warmup(predictions)
        per_source, components, optimizer_loss, count = self._source_values(predictions)
        measured = time.time()

        for name, value in (
            ("Timing/Process_Batch_Sec", processed - start),
            ("Timing/Forward_Pass_Sec", forwarded - processed),
            ("Timing/Compute_Losses_Sec", measured - forwarded),
        ):
            self.log(
                name,
                value,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )

        self._log_components(per_source, components, count)
        self._log_extra_metrics(predictions, optimizer_loss)
        self._log_compute_contract()
        self._log_telemetry(
            "Schedule/ReconstructionOnly", float(reconstruction_only)
        )
        self._log_telemetry(
            "Schedule/EffectiveFlowWeight",
            0.0
            if reconstruction_only
            else self._objective_weight("flow_weight", default=1.0),
        )
        self._log_telemetry(
            "Schedule/EffectiveActionVelocityWeight",
            0.0
            if reconstruction_only
            else self._objective_weight("action_velocity_weight", default=1.0),
        )
        next_step = int(self.global_step) + 1
        if (
            self.gradient_telemetry_cadence
            and next_step % self.gradient_telemetry_cadence == 0
        ):
            self._log_gradient_telemetry(components)
        return optimizer_loss

    def on_after_backward(self) -> None:
        if self.device.type == "cuda":
            self._log_telemetry(
                "Compute/PeakAllocatedBytes",
                torch.cuda.max_memory_allocated(self.device),
            )
        super().on_after_backward()

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        checkpoint["action_flow_loss_schedule"] = {
            "joint_objective_begins_at_global_step": (
                self.reconstruction_only_warmup_steps
            ),
            "reconstruction_only_optimizer_steps": (
                self.reconstruction_only_warmup_steps
            ),
            "joint_flow_weight": self._objective_weight(
                "flow_weight", default=1.0
            ),
            "joint_reconstruction_weight": self._objective_weight(
                "reconstruction_weight"
            ),
            "joint_action_velocity_weight": self._objective_weight(
                "action_velocity_weight", default=1.0
            ),
            "schema_version": 1,
        }
        if self._gradient_route_manifest is not None:
            checkpoint["action_flow_gradient_route_manifest"] = (
                self._gradient_route_manifest
            )

    @staticmethod
    def _diagnostics_enabled(evaluator: Any) -> bool:
        if evaluator is None:
            return False
        explicit = getattr(evaluator, "action_flow_diagnostics_enabled", None)
        if explicit is not None:
            return bool(explicit)
        config = getattr(evaluator, "action_flow_diagnostics", None)
        if config is None:
            return False
        if not isinstance(config, Mapping):
            raise TypeError("action_flow_diagnostics must be a mapping or None")
        return bool(config.get("enabled", True))

    def on_validation_start(self) -> None:
        self._action_flow_validation_sums = OrderedDict()
        self._action_flow_validation_count = 0
        self._active_validation_batch = None
        if self.evaluator is None:
            return
        self.model.device = self.device
        if self._diagnostics_enabled(self.evaluator):
            self.evaluator.model = self
        self.evaluator.on_validation_start()

    def _measure_validation_components(self, batch: Mapping, batch_idx: int) -> None:
        seed = 420_042 + int(batch_idx) + int(self.global_rank) * 1_000_003
        with torch.inference_mode(False):
            ordinary_batch = clone_inference_tensors(batch)
            with torch.random.fork_rng(devices=cuda_devices(ordinary_batch)):
                torch.manual_seed(seed)
                if self.device.type == "cuda":
                    torch.cuda.manual_seed(seed)
                with torch.enable_grad():
                    predictions = self.model.forward_training(ordinary_batch)
            _, components, _, count = self._source_values(predictions)
        for name, value in components.items():
            weighted = value.detach().double() * count
            self._action_flow_validation_sums[name] = (
                self._action_flow_validation_sums.get(name, 0.0) + weighted
            )
        self._action_flow_validation_count += count

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        if isinstance(batch, Mapping):
            batch = OrderedDict(
                (source, value) for source, value in batch.items() if value is not None
            )
        if not batch:
            return
        processed = self.model.process_batch_for_training(batch)
        self._measure_validation_components(processed, batch_idx)
        if self.evaluator is not None:
            self._active_validation_batch = processed
            try:
                self.evaluator.on_validation_step(processed, batch_idx, dataloader_idx)
            finally:
                self._active_validation_batch = None

    def on_validation_epoch_end(self) -> None:
        if not self._action_flow_validation_count:
            return
        count = self._action_flow_validation_count
        means = OrderedDict(
            (name, value / count)
            for name, value in self._action_flow_validation_sums.items()
        )
        reduced, _ = self._reduce_component_means(means, count)
        self.log_dict(
            OrderedDict(
                (f"Valid/ActionFlow/{name}", value) for name, value in reduced.items()
            ),
            on_step=False,
            on_epoch=True,
            sync_dist=False,
        )

    def on_validation_end(self) -> None:
        self._active_validation_batch = None
        if self.evaluator is not None:
            self.evaluator.on_validation_end()

    def _already_processed(self, batch: Mapping) -> bool:
        return batch is self._active_validation_batch

    @torch.inference_mode()
    def forward_eval(self, batch: Mapping) -> OrderedDict:
        """Execute inference, moving a caller batch exactly once when needed."""

        processed = (
            batch
            if self._already_processed(batch)
            else self.model.process_batch_for_training(batch)
        )
        return self.model.forward_eval(processed)

    def forward_action_flow_diagnostics(
        self,
        batch: Mapping,
        *,
        raw_noise_levels: Sequence[float],
        noise_seed: int = 420_042,
        max_samples: int | None = None,
        jacobian_samples: int = 2,
        capture_activations: bool = True,
    ) -> OrderedDict[str, OrderedDict[str, Any]]:
        """Return deterministic fixed-bank diagnostics for each opaque source.

        The method is safe under an evaluator decorated with
        :func:`torch.inference_mode`: it explicitly creates ordinary cloned
        tensors before running forward AD and decoder Jacobians.  All returned
        tensors are detached while activation rows retain the full returned
        batch independently of the Jacobian sample cap.
        """

        if isinstance(noise_seed, bool) or not isinstance(noise_seed, int):
            raise TypeError("noise_seed must be an integer")
        if max_samples is not None and (
            isinstance(max_samples, bool)
            or not isinstance(max_samples, int)
            or max_samples <= 0
        ):
            raise ValueError("max_samples must be None or a positive integer")
        if (
            isinstance(jacobian_samples, bool)
            or not isinstance(jacobian_samples, int)
            or jacobian_samples <= 0
        ):
            raise ValueError("jacobian_samples must be a positive integer")
        if isinstance(raw_noise_levels, (str, bytes)):
            raise TypeError("raw_noise_levels must be a numeric sequence")
        try:
            levels = tuple(float(level) for level in raw_noise_levels)
        except (TypeError, ValueError) as error:
            raise TypeError("raw_noise_levels must be a numeric sequence") from error

        return collect_action_flow_diagnostics(
            self.model,
            batch,
            raw_noise_levels=levels,
            noise_seed=noise_seed,
            max_samples=max_samples,
            jacobian_samples=jacobian_samples,
            capture_activations=bool(capture_activations),
            already_processed=self._already_processed(batch),
        )
