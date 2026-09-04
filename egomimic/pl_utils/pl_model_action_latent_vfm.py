"""Lightning metrics and gradient telemetry for action-latent velocity FM."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from typing import Any

import torch

from egomimic.pipeline.stages_action_latent_vfm import (
    ActionLatentDecoderStage,
    ActionLatentEncoderStage,
    LatentVelocityFieldStage,
)
from egomimic.pl_utils.pl_model import ModelWrapper


class ActionLatentVFMModelWrapper(ModelWrapper):
    """Optimize and report noisy reconstruction plus direct velocity FM."""

    gradient_telemetry_cadence = 100
    _component_keys = (
        ("ReconstructionLoss", "loss/action_latent_reconstruction"),
        ("FlowLoss", "loss/action_latent_fm"),
        ("MonotonicLoss", "loss/action_latent_monotonic"),
        ("ReconstructionL1", "log/action_latent_reconstruction_l1"),
    )

    def __init__(self, gradient_telemetry_cadence: int | None = None, **kwargs):
        config_tree = kwargs.get("config_tree")
        configured_cadence = None
        if config_tree is not None:
            cfg = self._as_config(config_tree)
            configured_cadence = cfg.model.get("gradient_telemetry_cadence")
        if gradient_telemetry_cadence is None:
            gradient_telemetry_cadence = (
                self.gradient_telemetry_cadence
                if configured_cadence is None
                else configured_cadence
            )
        elif configured_cadence is not None and int(gradient_telemetry_cadence) != int(
            configured_cadence
        ):
            raise ValueError(
                "gradient_telemetry_cadence disagrees with the resolved config_tree"
            )
        super().__init__(**kwargs)
        self.gradient_telemetry_cadence = int(gradient_telemetry_cadence)
        if self.gradient_telemetry_cadence <= 0:
            raise ValueError("gradient_telemetry_cadence must be positive")
        self.hparams.gradient_telemetry_cadence = self.gradient_telemetry_cadence
        self._validation_sums = OrderedDict()
        self._validation_count = 0

    @torch.inference_mode()
    def forward_unite_diagnostics(
        self,
        batch: Mapping,
        *,
        raw_noise_levels: tuple[float, ...] | list[float],
    ) -> OrderedDict:
        """Expose action-latent trajectories through the shared evaluator API."""

        stages = self.model.pipeline.stages
        encoders = [
            stage for stage in stages if isinstance(stage, ActionLatentEncoderStage)
        ]
        velocities = [
            stage for stage in stages if isinstance(stage, LatentVelocityFieldStage)
        ]
        decoders = [
            stage for stage in stages if isinstance(stage, ActionLatentDecoderStage)
        ]
        if tuple(map(len, (encoders, velocities, decoders))) != (1, 1, 1):
            raise RuntimeError(
                "action-latent diagnostics require one encoder, velocity field, and decoder"
            )
        encoder, velocity, decoder = encoders[0], velocities[0], decoders[0]
        diagnostics = OrderedDict()
        for source, source_batch in batch.items():
            if not isinstance(source_batch, Mapping):
                raise TypeError(f"diagnostic source {source!r} must be a mapping")
            result = dict(source_batch)
            for stage in stages:
                if stage is encoder:
                    break
                result = stage.execute(result, mode="train")
            missing = {"target", "condition", "sampler/noise"} - set(result)
            if missing:
                raise RuntimeError(
                    f"diagnostic prefix for {source!r} is missing {sorted(missing)}"
                )
            clean, encoder_activations = encoder.encoder.forward_with_activations(
                result["target"]
            )
            states, final_predictions, denoising_activations = (
                velocity.diagnostic_rollout(
                    clean_latent=clean,
                    noise=result["sampler/noise"],
                    condition=result["condition"],
                    raw_noise_levels=raw_noise_levels,
                )
            )
            encoder_names = tuple(encoder_activations)
            denoiser_names = tuple(denoising_activations)
            if not encoder_names or not denoiser_names:
                raise RuntimeError("diagnostic activation maps must be non-empty")
            paired_encoder = OrderedDict()
            paired_denoiser = OrderedDict()
            for index, encoder_name in enumerate(encoder_names):
                denominator = max(1, len(encoder_names) - 1)
                denoiser_index = round(index * (len(denoiser_names) - 1) / denominator)
                denoiser_name = denoiser_names[denoiser_index]
                pair_name = f"{encoder_name}_to_{denoiser_name}"
                paired_encoder[pair_name] = encoder_activations[encoder_name]
                paired_denoiser[pair_name] = denoising_activations[denoiser_name]
            diagnostics[source] = {
                "clean_latent": clean,
                "sampler_latents": states,
                "decoded_actions_normalized": torch.stack(
                    [decoder.decoder(state) for state in states], dim=0
                ),
                "noise_level_final_predictions": final_predictions,
                "tokenization_activations": paired_encoder,
                "denoising_activations": paired_denoiser,
            }
        return diagnostics

    @staticmethod
    def _finite_scalar(value: Any, label: str) -> torch.Tensor:
        if not torch.is_tensor(value) or value.ndim != 0:
            raise TypeError(f"{label} must be a scalar tensor")
        if not bool(torch.isfinite(value.detach())):
            raise RuntimeError(f"Non-finite action-latent VFM metric {label}")
        return value

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
            for stage in self.model.pipeline.stages
            if isinstance(stage, LatentVelocityFieldStage)
        ]
        if len(matches) != 1:
            raise RuntimeError(
                "action-latent VFM requires exactly one LatentVelocityFieldStage"
            )
        return matches[0]

    @staticmethod
    def _distributed_gradient(gradient: torch.Tensor) -> torch.Tensor:
        value = gradient.detach().float().clone()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.SUM)
            value.div_(torch.distributed.get_world_size())
        return value

    @classmethod
    def _gradient_norm(cls, gradients: Sequence[torch.Tensor]) -> torch.Tensor:
        values = [cls._distributed_gradient(gradient) for gradient in gradients]
        if not values:
            raise RuntimeError("action-latent VFM telemetry received no gradients")
        norm = sum(
            (value.square().sum() for value in values), values[0].new_zeros(())
        ).sqrt()
        if not bool(torch.isfinite(norm)) or float(norm) <= 0.0:
            raise RuntimeError("action-latent VFM gradient norm is zero or non-finite")
        return norm

    @staticmethod
    def _autograd(loss, named):
        parameters = tuple(parameter for _, parameter in named)
        gradients = torch.autograd.grad(
            loss,
            parameters,
            retain_graph=True,
            create_graph=False,
            allow_unused=False,
        )
        return parameters, gradients

    def _log_telemetry(self, name: str, value: Any) -> None:
        value = torch.as_tensor(value, device=self.device, dtype=torch.float32)
        self._finite_scalar(value, name)
        self.log(name, value, on_step=True, on_epoch=False, sync_dist=False)

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
        parameters, reconstruction_gradients = self._autograd(reconstruction, named)
        _, flow_gradients = self._autograd(flow, named)
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

    @staticmethod
    def _reduce_sums(sums, count):
        if count <= 0:
            raise RuntimeError("action-latent VFM metric reduction has no samples")
        names = tuple(sums)
        first = sums[names[0]]
        payload = torch.stack(
            (
                *(sums[name].detach().double() for name in names),
                torch.tensor(float(count), device=first.device, dtype=torch.float64),
            )
        )
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(payload, op=torch.distributed.ReduceOp.SUM)
        if not bool(torch.isfinite(payload).all()) or float(payload[-1]) <= 0.0:
            raise RuntimeError("non-finite action-latent VFM metric reduction")
        return (
            OrderedDict(
                (name, (payload[index] / payload[-1]).to(first.dtype))
                for index, name in enumerate(names)
            ),
            int(payload[-1].item()),
        )

    @classmethod
    def _distributed_components(cls, components, count):
        return cls._reduce_sums(
            OrderedDict(
                (name, value.detach().double() * count)
                for name, value in components.items()
            ),
            count,
        )

    def forward_eval(self, batch: Mapping):
        return self.model.forward_eval(batch)

    def training_step(self, batch, batch_idx):
        del batch_idx
        self.train()
        batch = self.model.process_batch_for_training(batch)
        predictions = self.model.forward_training(batch)
        components, count = self._components(predictions)
        self._log_prediction_metrics(predictions, components["TotalLoss"])
        logged, global_count = self._distributed_components(components, count)
        for name, value in logged.items():
            self.log(
                f"Train/ActionLatentVFM/{name}",
                value,
                on_step=True,
                on_epoch=True,
                sync_dist=False,
                batch_size=global_count,
            )
            self.log(
                f"Train/UNITE/{name}",
                value,
                on_step=True,
                on_epoch=True,
                sync_dist=False,
                batch_size=global_count,
            )
        if (int(self.global_step) + 1) % self.gradient_telemetry_cadence == 0:
            self._measure_gradient_conflict(
                components["ReconstructionLoss"], components["FlowLoss"]
            )
        return components["TotalLoss"]

    def on_validation_start(self):
        self._validation_sums = OrderedDict()
        self._validation_count = 0
        if self.evaluator is not None:
            self.model.device = self.device
            self.evaluator.model = self
            self.evaluator.on_validation_start()

    @torch.no_grad()
    def _measure_validation_components(self, batch, batch_idx):
        devices = (
            [self.device.index or torch.cuda.current_device()]
            if self.device.type == "cuda"
            else []
        )
        seed = 420_042 + int(batch_idx) + int(self.global_rank) * 1_000_003
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            if self.device.type == "cuda":
                torch.cuda.manual_seed(seed)
            predictions = self.model.forward_training(batch)
        components, count = self._components(predictions)
        for name, value in components.items():
            weighted = value.detach().double() * count
            self._validation_sums[name] = (
                self._validation_sums.get(name, 0.0) + weighted
            )
        if self.evaluator is not None:
            for source, result in predictions.items():
                target = result["target"]
                source_count = int(target.shape[0])
                native_mse, native_l1 = self.evaluator.native_action_errors(
                    result["reconstruction/pred_action"], target, batch[source]
                )
                for name, value in (
                    ("ReconstructionNativeMSE", native_mse),
                    ("ReconstructionNativeL1", native_l1),
                ):
                    value = self._finite_scalar(value, f"{source!r}/{name}")
                    weighted = value.detach().double() * source_count
                    self._validation_sums[name] = (
                        self._validation_sums.get(name, 0.0) + weighted
                    )
        self._validation_count += count

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        if isinstance(batch, Mapping):
            batch = OrderedDict(
                (key, value) for key, value in batch.items() if value is not None
            )
        if not batch:
            return
        batch = self.model.process_batch_for_training(batch)
        self._measure_validation_components(batch, batch_idx)
        if self.evaluator is not None:
            self.evaluator.on_validation_step(batch, batch_idx, dataloader_idx)

    def on_validation_epoch_end(self):
        if self._validation_count:
            metrics, _ = self._reduce_sums(
                self._validation_sums, self._validation_count
            )
            self.log_dict(
                OrderedDict(
                    (f"Valid/ActionLatentVFM/{name}", value)
                    for name, value in metrics.items()
                ),
                on_step=False,
                on_epoch=True,
                sync_dist=False,
            )
            self.log_dict(
                OrderedDict(
                    (f"Valid/UNITE/{name}", value) for name, value in metrics.items()
                ),
                on_step=False,
                on_epoch=True,
                sync_dist=False,
            )

    def on_validation_end(self):
        if self.evaluator is not None:
            self.evaluator.on_validation_end()
