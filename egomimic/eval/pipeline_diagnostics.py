"""Model diagnostic providers kept separate from training behavior.

Providers consume a configured pipeline and return the strict tensor schemas
expected by evaluators.  They own no trainable state and do not participate in
optimization or Lightning lifecycle hooks.
"""

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
from egomimic.pipeline.stages_unite_released import ReleasedRecipeUniteLatentPolicy
from egomimic.eval.action_flow_diagnostic_forward import (
    collect_action_flow_diagnostics,
)


class DiagnosticProvider:
    """Stateless interface for one pipeline diagnostic capability."""

    capability: str

    def run(
        self,
        model: Any,
        batch: Mapping,
        *,
        already_processed: bool,
        **kwargs: Any,
    ) -> Mapping:
        raise NotImplementedError


class ActionFlowDiagnosticProvider(DiagnosticProvider):
    capability = "action_flow"

    def run(
        self,
        model: Any,
        batch: Mapping,
        *,
        already_processed: bool,
        raw_noise_levels: Sequence[float],
        noise_seed: int = 420_042,
        max_samples: int | None = None,
        jacobian_samples: int = 2,
        capture_activations: bool = True,
    ) -> OrderedDict[str, OrderedDict[str, Any]]:
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
            model,
            batch,
            raw_noise_levels=levels,
            noise_seed=noise_seed,
            max_samples=max_samples,
            jacobian_samples=jacobian_samples,
            capture_activations=bool(capture_activations),
            already_processed=already_processed,
        )


class ActionLatentVFMDiagnosticProvider(DiagnosticProvider):
    capability = "unite"

    @torch.inference_mode()
    def run(
        self,
        model: Any,
        batch: Mapping,
        *,
        already_processed: bool,
        raw_noise_levels: tuple[float, ...] | list[float],
    ) -> OrderedDict:
        del already_processed
        stages = model.pipeline.stages
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
                "clean_decoded_action_normalized": decoder.decoder(clean),
                "sampler_latents": states,
                "decoded_actions_normalized": torch.stack(
                    [decoder.decoder(state) for state in states], dim=0
                ),
                "noise_level_final_predictions": final_predictions,
                "tokenization_activations": paired_encoder,
                "denoising_activations": paired_denoiser,
            }
        return diagnostics


class ReleasedUniteDiagnosticProvider(DiagnosticProvider):
    capability = "unite"

    @torch.inference_mode()
    def run(
        self,
        model: Any,
        batch: Mapping,
        *,
        already_processed: bool,
        raw_noise_levels: tuple[float, ...] | list[float],
    ) -> OrderedDict:
        del already_processed
        if not isinstance(batch, Mapping):
            raise TypeError("UNITE diagnostics input must be a source mapping")
        stages = model.pipeline.stages
        policies = [
            stage for stage in stages if isinstance(stage, ReleasedRecipeUniteLatentPolicy)
        ]
        if len(policies) != 1:
            raise RuntimeError(
                "Released UNITE diagnostics require exactly one latent policy; "
                f"found {len(policies)}"
            )
        policy = policies[0]
        diagnostics = OrderedDict()
        for source, source_batch in batch.items():
            if not isinstance(source_batch, Mapping):
                raise TypeError(f"UNITE diagnostic source {source!r} must be a mapping")
            result = dict(source_batch)
            for stage in stages:
                if stage is policy:
                    break
                result = stage.execute(result, mode="inference")
            required = {"sampler/noise", "condition", "target", "embodiment"}
            missing = required - set(result)
            if missing:
                raise RuntimeError(
                    f"Released UNITE diagnostic prefix for {source!r} is missing "
                    f"{sorted(missing)}"
                )
            diagnostics[source] = policy.validation_diagnostics(
                noise=result["sampler/noise"],
                condition=result["condition"],
                target=result["target"],
                embodiment=result["embodiment"],
                raw_noise_levels=raw_noise_levels,
            )
        return diagnostics
