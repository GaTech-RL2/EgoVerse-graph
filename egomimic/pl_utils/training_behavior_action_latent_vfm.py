"""Training behavior for action-latent velocity FM."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from typing import Any

import torch

from egomimic.pipeline.stages_action_latent_vfm import LatentVelocityFieldStage
from egomimic.pl_utils.training_behavior import TrainingBehavior
from egomimic.pl_utils.training_metrics import (
    MetricAccumulator,
    component_gradients,
    distributed_gradient,
    finite_scalar,
    reduce_component_means,
)


class ActionLatentVFMTrainingBehavior(TrainingBehavior):
    """Adapt action-latent VFM outputs to generic training mechanics."""

    gradient_telemetry_cadence = 100
    _component_keys = (
        ("ReconstructionLoss", "loss/action_latent_reconstruction"),
        ("FlowLoss", "loss/action_latent_fm"),
        ("MonotonicLoss", "loss/action_latent_monotonic"),
        ("ReconstructionL1", "log/action_latent_reconstruction_l1"),
    )
    _finite_scalar = staticmethod(finite_scalar)
    _distributed_gradient = staticmethod(distributed_gradient)

    def __init__(self, gradient_telemetry_cadence: int | None = None):
        super().__init__()
        self._requested_gradient_telemetry_cadence = gradient_telemetry_cadence

    def on_bind(self) -> None:
        config_tree = getattr(self.context.hparams, "config_tree", None)
        configured_cadence = None
        if config_tree is not None:
            cfg = self.context._as_config(config_tree)
            configured_cadence = cfg.model.get("gradient_telemetry_cadence")
        if self._requested_gradient_telemetry_cadence is None:
            cadence = (
                self.gradient_telemetry_cadence
                if configured_cadence is None
                else configured_cadence
            )
        elif configured_cadence is not None and int(
            self._requested_gradient_telemetry_cadence
        ) != int(configured_cadence):
            raise ValueError(
                "gradient_telemetry_cadence disagrees with the resolved config_tree"
            )
        else:
            cadence = self._requested_gradient_telemetry_cadence
        self.gradient_telemetry_cadence = int(cadence)
        if self.gradient_telemetry_cadence <= 0:
            raise ValueError("gradient_telemetry_cadence must be positive")
        self._validation_metrics = MetricAccumulator()

    def _components(self, predictions: Mapping):
        if not isinstance(predictions, Mapping) or not predictions:
            raise RuntimeError("action-latent VFM received no source predictions")
        sums = OrderedDict((name, None) for name, _ in self._component_keys)
        count = 0
        for source, result in predictions.items():
            target = result.get("target")
            if not torch.is_tensor(target) or target.ndim == 0:
                raise TypeError(f"source {source!r} target must be batched")
            source_count = int(target.shape[0])
            if source_count <= 0:
                raise RuntimeError(f"source {source!r} has no samples")
            count += source_count
            for name, key in self._component_keys:
                value = self._finite_scalar(result.get(key), f"{source!r}/{key}")
                weighted = value * source_count
                sums[name] = weighted if sums[name] is None else sums[name] + weighted
        values = OrderedDict((name, value / count) for name, value in sums.items())
        values["TotalLoss"] = (
            values["ReconstructionLoss"] + values["FlowLoss"] + values["MonotonicLoss"]
        )
        values.move_to_end("TotalLoss", last=False)
        for name, value in values.items():
            self._finite_scalar(value, name)
        return values, count

    def _velocity_stage(self) -> LatentVelocityFieldStage:
        matches = [
            stage
            for stage in self.context.model.pipeline.stages
            if isinstance(stage, LatentVelocityFieldStage)
        ]
        if len(matches) != 1:
            raise RuntimeError(
                "action-latent VFM requires exactly one LatentVelocityFieldStage"
            )
        return matches[0]

    def _log_telemetry(self, name: str, value: Any) -> None:
        value = torch.as_tensor(value, device=self.context.device, dtype=torch.float32)
        self._finite_scalar(value, name)
        self.context.log(name, value, on_step=True, on_epoch=False, sync_dist=False)

    def _measure_gradient_conflict(self, reconstruction, flow):
        named = tuple(
            (name, parameter)
            for name, parameter in self._velocity_stage().denoising_module.named_parameters()
            if parameter.requires_grad
        )
        if not named:
            raise RuntimeError("action-latent VFM denoiser has no trainable parameters")
        identities = [id(parameter) for _, parameter in named]
        if len(set(identities)) != len(identities):
            raise RuntimeError("action-latent VFM denoiser parameters are duplicated")
        parameters, reconstruction_gradients = component_gradients(
            reconstruction,
            named,
            allow_unused=False,
            label="action-latent VFM reconstruction",
        )
        _, flow_gradients = component_gradients(
            flow,
            named,
            allow_unused=False,
            label="action-latent VFM flow",
        )
        pairs = [
            (self._distributed_gradient(left), self._distributed_gradient(right))
            for left, right in zip(reconstruction_gradients, flow_gradients)
        ]
        zero = pairs[0][0].new_zeros(())
        dot = sum((left * right).sum() for left, right in pairs)
        reconstruction_norm = sum(
            (left.square().sum() for left, _ in pairs), zero
        ).sqrt()
        flow_norm = sum((right.square().sum() for _, right in pairs), zero).sqrt()
        if (
            not bool(
                torch.isfinite(torch.stack((dot, reconstruction_norm, flow_norm))).all()
            )
            or float(reconstruction_norm) <= 0.0
            or float(flow_norm) <= 0.0
        ):
            raise RuntimeError(
                "action-latent VFM gradient telemetry is zero or non-finite"
            )
        cosine = (dot / (reconstruction_norm * flow_norm)).clamp(-1.0, 1.0)
        for name, value in (
            ("log/unite_gradient_cosine", cosine),
            ("log/unite_recon_grad_norm", reconstruction_norm),
            ("log/unite_denoise_grad_norm", flow_norm),
            ("log/unite_gradient_parameter_count", sum(p.numel() for p in parameters)),
            ("log/unite_gradient_tensor_count", len(parameters)),
        ):
            self._log_telemetry(name, value)

    def training_step(self, batch, batch_idx):
        del batch_idx
        self.context.train()
        batch = self.context.model.process_batch_for_training(batch)
        predictions = self.context.model.forward_training(batch)
        components, count = self._components(predictions)
        self.context._log_prediction_metrics(predictions, components["TotalLoss"])
        logged, global_count = reduce_component_means(
            components, count, label="action-latent VFM"
        )
        for name, value in logged.items():
            self.context.log(
                f"Train/ActionLatentVFM/{name}",
                value,
                on_step=True,
                on_epoch=True,
                sync_dist=False,
                batch_size=global_count,
            )
            self.context.log(
                f"Train/UNITE/{name}",
                value,
                on_step=True,
                on_epoch=True,
                sync_dist=False,
                batch_size=global_count,
            )
        if (int(self.context.global_step) + 1) % self.gradient_telemetry_cadence == 0:
            self._measure_gradient_conflict(
                components["ReconstructionLoss"], components["FlowLoss"]
            )
        return components["TotalLoss"]

    def on_validation_start(self):
        self._validation_metrics.reset()
        if self.context.evaluator is not None:
            self.context.model.device = self.context.device
            self.context.evaluator.model = self.context
            self.context.evaluator.on_validation_start()

    @torch.no_grad()
    def _measure_validation_components(self, batch, batch_idx):
        devices = (
            [self.context.device.index or torch.cuda.current_device()]
            if self.context.device.type == "cuda"
            else []
        )
        seed = 420_042 + int(batch_idx) + int(self.context.global_rank) * 1_000_003
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            if self.context.device.type == "cuda":
                torch.cuda.manual_seed(seed)
            predictions = self.context.model.forward_training(batch)
        components, count = self._components(predictions)
        self._validation_metrics.add(components, count)
        if self.context.evaluator is not None:
            for source, result in predictions.items():
                target = result["target"]
                source_count = int(target.shape[0])
                native_mse, native_l1 = self.context.evaluator.native_action_errors(
                    result["reconstruction/pred_action"], target, batch[source]
                )
                for name, value in (
                    ("ReconstructionNativeMSE", native_mse),
                    ("ReconstructionNativeL1", native_l1),
                ):
                    value = self._finite_scalar(value, f"{source!r}/{name}")
                    self._validation_metrics.add(
                        {name: value}, source_count, update_count=False
                    )

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        if isinstance(batch, Mapping):
            batch = OrderedDict(
                (key, value) for key, value in batch.items() if value is not None
            )
        if not batch:
            return
        batch = self.context.model.process_batch_for_training(batch)
        self._measure_validation_components(batch, batch_idx)
        if self.context.evaluator is not None:
            self.context._run_evaluator_validation_step(
                batch, batch_idx, dataloader_idx
            )

    def on_validation_epoch_end(self):
        if self._validation_metrics.count:
            metrics, _ = self._validation_metrics.reduce(
                label="action-latent VFM"
            )
            self.context.log_dict(
                OrderedDict(
                    (f"Valid/ActionLatentVFM/{name}", value)
                    for name, value in metrics.items()
                ),
                on_step=False,
                on_epoch=True,
                sync_dist=False,
            )
            self.context.log_dict(
                OrderedDict(
                    (f"Valid/UNITE/{name}", value) for name, value in metrics.items()
                ),
                on_step=False,
                on_epoch=True,
                sync_dist=False,
            )

    def on_validation_end(self):
        if self.context.evaluator is not None:
            self.context.evaluator.on_validation_end()
