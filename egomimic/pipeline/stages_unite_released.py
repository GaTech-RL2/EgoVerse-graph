"""Released UNITE stages for compact action-register latents."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from egomimic.pipeline.core import Stage


def _mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError(
            "Released UNITE tensor shape mismatch: "
            f"prediction={tuple(prediction.shape)} target={tuple(target.shape)}"
        )
    return (prediction - target).square().mean()


class _PerEmbodimentDecoder(nn.Module):
    """Small decoder router that preserves released checkpoint key names."""

    def __init__(self, decoders: Dict[str, nn.Module]):
        super().__init__()
        configured = {str(domain): decoder for domain, decoder in decoders.items()}
        if not configured or any(
            not isinstance(decoder, nn.Module) for decoder in configured.values()
        ):
            raise ValueError("decoders must contain at least one nn.Module")
        self.decoders = nn.ModuleDict(configured)
        self.domains = tuple(configured)
        latent_dims = {
            int(getattr(decoder, "latent_dim", -1)) for decoder in configured.values()
        }
        if len(latent_dims) != 1 or min(latent_dims) <= 0:
            raise ValueError("All decoders must expose the same positive latent_dim")
        self.latent_dim = latent_dims.pop()
        self.action_dims = {
            domain: int(getattr(decoder, "action_dim", -1))
            for domain, decoder in configured.items()
        }
        if any(action_dim <= 0 for action_dim in self.action_dims.values()):
            raise ValueError("Every decoder must expose a positive action_dim")

    def decoder_for(self, embodiment: str) -> nn.Module:
        embodiment = str(embodiment)
        if embodiment not in self.decoders:
            raise KeyError(
                f"Unknown UNITE embodiment {embodiment!r}; configured={self.domains}"
            )
        return self.decoders[embodiment]


class ReleasedRecipeUniteLatentPolicy(Stage):
    """Joint reconstruction/flow UNITE policy with Dopri5 and CFG."""

    reads = ["sampler/noise", "condition", "target", "embodiment"]
    writes = [
        "unite/clean_latent",
        "unite/reconstructed_action",
        "unite/flow_loss",
        "log/*",
    ]
    reads_by_mode = {
        "inference": ["sampler/noise", "condition", "embodiment"],
    }
    writes_by_mode = {
        "inference": ["sampler/endpoint", "pred_action", "log/*"],
    }

    def __init__(
        self,
        generative_encoder: nn.Module,
        decoders: Dict[str, nn.Module],
        reconstruction_noise_std: float = 0.0,
        timestep_shift_alpha: float = 0.5,
        flow_steps_per_reconstruction: int = 14,
        flow_mini_batch: int = 14,
        train_eps: float = 0.05,
        sample_eps: float = 0.05,
        lognorm_mu: float = 0.0,
        lognorm_sigma: float = 1.0,
        reconstruction_noising_start: float = 0.7,
        reconstruction_noising_probability: float = 0.5,
        condition_dropout_probability: float = 0.1,
        sampling_method: str = "dopri5",
        cfg_scale: float = 4.0,
        cfg_interval: tuple[float, float] = (0.0, 1.0),
        cfg_norm_order: str = "norm_first",
        dopri5_num_steps: int = 50,
        dopri5_atol: float = 1.0e-6,
        dopri5_rtol: float = 1.0e-3,
    ):
        super().__init__()
        self.generative_encoder = generative_encoder
        self.action_decoder = _PerEmbodimentDecoder(decoders)
        self.timestep_shift_alpha = float(timestep_shift_alpha)
        self.flow_steps_per_reconstruction = int(flow_steps_per_reconstruction)
        self.flow_mini_batch = int(flow_mini_batch)
        self.train_eps = float(train_eps)
        self.sample_eps = float(sample_eps)
        self.lognorm_mu = float(lognorm_mu)
        self.lognorm_sigma = float(lognorm_sigma)
        self.reconstruction_noising_start = float(reconstruction_noising_start)
        self.reconstruction_noising_probability = float(
            reconstruction_noising_probability
        )
        self.condition_dropout_probability = float(condition_dropout_probability)
        self.sampling_method = str(sampling_method).lower()
        self.cfg_scale = float(cfg_scale)
        self.cfg_interval = tuple(float(value) for value in cfg_interval)
        self.cfg_norm_order = str(cfg_norm_order)
        self.dopri5_num_steps = int(dopri5_num_steps)
        self.dopri5_atol = float(dopri5_atol)
        self.dopri5_rtol = float(dopri5_rtol)
        self._last_sampler_nfe = 0

        domains = tuple(getattr(self.generative_encoder, "domains", ()))
        if not domains or set(domains) != set(self.action_decoder.domains):
            raise ValueError("UNITE encoder and decoder domains must match")
        latent_dim = int(getattr(self.generative_encoder, "latent_dim", -1))
        if latent_dim != self.action_decoder.latent_dim:
            raise ValueError("UNITE encoder and decoder latent dimensions must match")
        if dict(getattr(self.generative_encoder, "action_dims", {})) != (
            self.action_decoder.action_dims
        ):
            raise ValueError("UNITE encoder and decoder action dimensions must match")
        if float(reconstruction_noise_std) != 0.0:
            raise ValueError(
                "Released register UNITE requires reconstruction_noise_std=0"
            )
        if not torch.isfinite(torch.tensor(self.timestep_shift_alpha)):
            raise ValueError("timestep_shift_alpha must be finite")
        if self.timestep_shift_alpha <= 0.0:
            raise ValueError("timestep_shift_alpha must be positive")
        if self.flow_steps_per_reconstruction <= 0 or self.flow_mini_batch <= 0:
            raise ValueError("UNITE flow sample counts must be positive")
        if not 0.0 <= self.train_eps < 0.5 or not 0.0 <= self.sample_eps < 0.5:
            raise ValueError("UNITE epsilon values must be in [0, 0.5)")
        if self.lognorm_sigma <= 0.0:
            raise ValueError("lognorm_sigma must be positive")
        if not 0.0 <= self.reconstruction_noising_start <= 1.0:
            raise ValueError("reconstruction_noising_start must be in [0, 1]")
        if not 0.0 <= self.reconstruction_noising_probability <= 1.0:
            raise ValueError("reconstruction_noising_probability must be in [0, 1]")
        if not 0.0 <= self.condition_dropout_probability <= 1.0:
            raise ValueError("condition_dropout_probability must be in [0, 1]")
        if self.sampling_method != "dopri5":
            raise ValueError("Released register UNITE supports only Dopri5 sampling")
        if not torch.isfinite(torch.tensor(self.cfg_scale)) or self.cfg_scale < 0.0:
            raise ValueError("cfg_scale must be finite and non-negative")
        if len(self.cfg_interval) != 2 or not (
            0.0 <= self.cfg_interval[0] < self.cfg_interval[1] <= 1.0
        ):
            raise ValueError("cfg_interval must satisfy 0 <= start < end <= 1")
        if self.cfg_norm_order != "norm_first":
            raise ValueError(
                "Released register UNITE requires cfg_norm_order=norm_first"
            )
        if self.dopri5_num_steps < 2:
            raise ValueError("dopri5_num_steps must be at least 2")
        if self.dopri5_atol <= 0.0 or self.dopri5_rtol <= 0.0:
            raise ValueError("dopri5 tolerances must be positive")

    def _resolve_domain(self, embodiment: str | None = None) -> str:
        resolver = getattr(self.generative_encoder, "_resolve_domain", None)
        if resolver is None:
            raise TypeError("UNITE generative encoder lacks domain resolution")
        return resolver(embodiment)

    def _decode(self, latent: torch.Tensor, embodiment: str) -> torch.Tensor:
        return self.action_decoder.decoder_for(embodiment)(latent)

    def _validate_noise(self, noise: torch.Tensor) -> None:
        expected = (
            int(getattr(self.generative_encoder, "num_latent_tokens", -1)),
            int(getattr(self.generative_encoder, "latent_dim", -1)),
        )
        if noise.ndim != 3 or tuple(noise.shape[1:]) != expected:
            raise ValueError(
                f"UNITE sampler/noise must have shape (B, {expected[0]}, "
                f"{expected[1]}), got {tuple(noise.shape)}"
            )

    def shift_time(self, time: torch.Tensor) -> torch.Tensor:
        alpha = torch.as_tensor(
            self.timestep_shift_alpha,
            device=time.device,
            dtype=time.dtype,
        )
        return alpha * time / (1.0 + (alpha - 1.0) * time)

    def _null_condition_like(self, condition: torch.Tensor) -> torch.Tensor:
        return self.generative_encoder.null_condition_like(condition)

    def _condition_with_dropout(
        self, condition: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.training or self.condition_dropout_probability == 0.0:
            return condition, condition.new_zeros(())
        mask_shape = (int(condition.shape[0]),) + (1,) * (condition.ndim - 1)
        mask = (
            torch.rand(int(condition.shape[0]), device=condition.device)
            < self.condition_dropout_probability
        ).reshape(mask_shape)
        dropped = torch.where(mask, self._null_condition_like(condition), condition)
        return dropped, mask.float().mean()

    def _guided_clean_prediction(
        self,
        latent: torch.Tensor,
        time: torch.Tensor,
        condition: torch.Tensor,
        cfg_scale: float,
    ) -> torch.Tensor:
        if float(cfg_scale) <= 1.0:
            return self.generative_encoder.denoise(latent, time, condition)
        doubled_latent = torch.cat((latent, latent), dim=0)
        doubled_time = torch.cat((time, time), dim=0)
        doubled_condition = torch.cat(
            (condition, self._null_condition_like(condition)), dim=0
        )
        conditioned, unconditional = self.generative_encoder.denoise(
            doubled_latent, doubled_time, doubled_condition
        ).chunk(2, dim=0)
        guided = unconditional + float(cfg_scale) * (conditioned - unconditional)
        start, end = self.cfg_interval
        active = ((time < end) & ((start == 0.0) | (time > start))).reshape(
            int(time.shape[0]), *([1] * (latent.ndim - 1))
        )
        return torch.where(active, guided, conditioned)

    def _sample_flow_time(self, batch_size: int, device: torch.device) -> torch.Tensor:
        normal = torch.randn(batch_size, device=device, dtype=torch.float32)
        normal = self.lognorm_mu + self.lognorm_sigma * normal
        return self.shift_time(torch.sigmoid(normal))

    def _noisy_reconstruction_latent(self, clean_latent: torch.Tensor) -> torch.Tensor:
        if not self.training or self.reconstruction_noising_probability == 0.0:
            return clean_latent
        batch_size = int(clean_latent.shape[0])
        time = self.reconstruction_noising_start + (
            1.0 - self.reconstruction_noising_start
        ) * torch.rand(batch_size, device=clean_latent.device, dtype=torch.float32)
        time = time.reshape(batch_size, 1, 1).to(clean_latent)
        noised = time * clean_latent + (1.0 - time) * torch.randn_like(clean_latent)
        mask = (
            torch.rand(batch_size, device=clean_latent.device)
            < self.reconstruction_noising_probability
        ).reshape(batch_size, 1, 1)
        return torch.where(mask, noised, clean_latent)

    @staticmethod
    def _flow_chunks(total: int, size: int) -> tuple[int, ...]:
        full, remainder = divmod(int(total), int(size))
        return (*([int(size)] * full), *((int(remainder),) if remainder else ()))

    def _released_flow_loss(
        self,
        clean_latent: torch.Tensor,
        condition: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        detached_clean = clean_latent.detach()
        batch_size = int(detached_clean.shape[0])
        loss = None
        representative_time = None
        dropout_fraction = detached_clean.new_zeros(())
        for repeats in self._flow_chunks(
            self.flow_steps_per_reconstruction, self.flow_mini_batch
        ):
            repeated_clean = detached_clean.repeat(repeats, 1, 1)
            repeated_condition = condition.repeat(
                repeats, *([1] * (condition.ndim - 1))
            )
            repeated_condition, chunk_dropout = self._condition_with_dropout(
                repeated_condition
            )
            dropout_fraction = dropout_fraction + repeats * chunk_dropout
            time = self._sample_flow_time(batch_size * repeats, detached_clean.device)
            time_view = time.reshape(batch_size * repeats, 1, 1).to(detached_clean)
            noise = torch.randn_like(repeated_clean)
            corrupted = time_view * repeated_clean + (1.0 - time_view) * noise
            prediction = self.generative_encoder.denoise(
                corrupted, time, repeated_condition
            )
            denominator = (1.0 - time_view).clamp_min(self.train_eps)
            chunk_loss = ((repeated_clean - prediction) / denominator).square().mean()
            weighted = chunk_loss * repeats
            loss = weighted if loss is None else loss + weighted
            if representative_time is None:
                representative_time = time[:batch_size]
        if loss is None or representative_time is None:
            raise RuntimeError("Released UNITE flow loop produced no samples")
        return (
            loss,
            representative_time,
            dropout_fraction / self.flow_steps_per_reconstruction,
        )

    def _integrate_dopri5(
        self,
        noise: torch.Tensor,
        condition: torch.Tensor,
        cfg_scale: float,
    ) -> torch.Tensor:
        """Integrate once over the released shifted grid and return its endpoint."""

        try:
            from torchdiffeq import odeint
        except ImportError as exc:
            raise RuntimeError("Released UNITE sampling requires torchdiffeq") from exc

        batch_size = int(noise.shape[0])
        raw_grid = torch.linspace(
            0.0,
            1.0,
            self.dopri5_num_steps,
            device=noise.device,
            dtype=torch.float32,
        )
        grid = self.shift_time(raw_grid)
        nfe = 0

        def velocity(time_scalar: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
            nonlocal nfe
            nfe += 1
            clean_prediction = self._guided_clean_prediction(
                latent,
                time_scalar.expand(batch_size),
                condition,
                cfg_scale,
            )
            denominator = (1.0 - time_scalar).clamp_min(self.sample_eps)
            return (clean_prediction.float() - latent.float()) / denominator.float()

        trajectory = odeint(
            velocity,
            noise.float(),
            grid,
            method="dopri5",
            atol=self.dopri5_atol,
            rtol=self.dopri5_rtol,
        )
        self._last_sampler_nfe = nfe
        return trajectory[-1]

    def sample(
        self,
        noise: torch.Tensor,
        condition: torch.Tensor,
        *,
        sampling_method: str | None = None,
        cfg_scale: float | None = None,
    ) -> torch.Tensor:
        self._validate_noise(noise)
        method = self.sampling_method if sampling_method is None else sampling_method
        if str(method).lower() != "dopri5":
            raise ValueError("Released register UNITE supports only Dopri5 sampling")
        scale = self.cfg_scale if cfg_scale is None else float(cfg_scale)
        return self._integrate_dopri5(noise, condition, scale)

    def _forward_training(self, batch: dict, embodiment: str) -> dict:
        target = batch["target"]
        clean_latent = self.generative_encoder.tokenize(
            target,
            embodiment,
            batch["sampler/noise"],
        )
        reconstructed_action = self._decode(
            self._noisy_reconstruction_latent(clean_latent), embodiment
        )
        flow_loss, sampled_time, dropout_fraction = self._released_flow_loss(
            clean_latent, batch["condition"]
        )
        batch.update(
            {
                "unite/clean_latent": clean_latent,
                "unite/reconstructed_action": reconstructed_action,
                "unite/flow_loss": flow_loss,
                "log/sampler_unroll_steps": 1.0,
                "log/unite_time_mean": sampled_time.detach().mean(),
                "log/unite_condition_dropout_fraction": dropout_fraction.detach(),
                "log/unite_cfg_scale": self.cfg_scale,
                "log/unite_flow_samples_per_reconstruction": float(
                    self.flow_steps_per_reconstruction
                ),
            }
        )
        return batch

    def _forward_rollout(self, batch: dict, embodiment: str) -> dict:
        endpoint = self.sample(batch["sampler/noise"], batch["condition"])
        batch["sampler/endpoint"] = endpoint
        batch["pred_action"] = self._decode(endpoint, embodiment)
        batch["log/sampler_unroll_steps"] = float(self._last_sampler_nfe)
        batch["log/unite_cfg_scale"] = self.cfg_scale
        return batch

    def _execute_mode(self, batch: dict, mode: str) -> dict:
        embodiment = self._resolve_domain(batch["embodiment"])
        noise = batch["sampler/noise"]
        self._validate_noise(noise)
        if int(noise.shape[0]) != int(batch["condition"].shape[0]):
            raise ValueError("UNITE noise and condition batch sizes must match")
        if mode == "train":
            batch = self._forward_training(batch, embodiment)
        elif mode == "inference":
            batch = self._forward_rollout(batch, embodiment)
        else:
            raise ValueError(f"Unsupported UNITE execution mode {mode!r}")
        batch["log/sampler_noise_rms"] = noise.detach().square().mean().sqrt()
        if mode == "inference":
            endpoint = batch["sampler/endpoint"]
            prediction = batch["pred_action"]
            batch["log/sampler_endpoint_rms"] = endpoint.detach().square().mean().sqrt()
            batch["log/sampler_prediction_rms"] = (
                prediction.detach().square().mean().sqrt()
            )
        return batch

    def execute(self, batch: dict, *, mode: str) -> dict:
        return self._execute_mode(batch, mode)

    def forward(self, batch: dict) -> dict:
        """Retain direct-call behavior while graph execution passes mode explicitly."""
        return self._execute_mode(batch, "train" if self.training else "inference")


class ReleasedRecipeUniteObjective(Stage):
    """Action-space translation of the released reconstruction/flow objective."""

    train_only = True
    reads = [
        "target",
        "unite/reconstructed_action",
        "unite/flow_loss",
    ]
    writes = ["loss/*", "log/*"]

    def __init__(self, reconstruction_weight: float = 1.0, flow_weight: float = 1.0):
        super().__init__()
        self.reconstruction_weight = float(reconstruction_weight)
        self.flow_weight = float(flow_weight)
        if self.reconstruction_weight <= 0.0 or self.flow_weight <= 0.0:
            raise ValueError("Released UNITE loss weights must be positive")

    def forward(self, batch: dict) -> dict:
        reconstruction = _mse(batch["unite/reconstructed_action"], batch["target"])
        reconstruction_l1 = (
            (batch["unite/reconstructed_action"] - batch["target"]).abs().mean()
        )
        flow = batch["unite/flow_loss"]
        if flow.ndim != 0 or not bool(torch.isfinite(flow)):
            raise RuntimeError("Released UNITE flow loss must be a finite scalar")
        batch["loss/unite_reconstruction"] = self.reconstruction_weight * reconstruction
        batch["loss/unite_latent"] = self.flow_weight * flow
        batch["log/unite_reconstruction"] = reconstruction.detach()
        batch["log/unite_reconstruction_l1"] = reconstruction_l1.detach()
        batch["log/unite_latent"] = flow.detach()
        return batch
