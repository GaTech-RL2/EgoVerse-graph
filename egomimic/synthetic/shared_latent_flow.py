"""Small shared-latent flow models for analytic synthetic distributions."""

from __future__ import annotations

import torch
from torch import nn


def _mlp(input_dim: int, hidden_dim: int, output_dim: int, depth: int) -> nn.Sequential:
    layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.SiLU()]
    for _ in range(depth - 1):
        layers.extend((nn.Linear(hidden_dim, hidden_dim), nn.SiLU()))
    layers.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*layers)


class SyntheticSharedLatentFlow(nn.Module):
    """Encode 3D targets, transport a 2D Gaussian, and decode back to 3D."""

    def __init__(
        self,
        *,
        latent_dim: int = 2,
        codec_width: int = 32,
        codec_depth: int = 2,
        field_width: int = 128,
        field_depth: int = 4,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.encoder = _mlp(3, codec_width, latent_dim, codec_depth)
        self.decoder = _mlp(latent_dim, codec_width, 3, codec_depth)
        self.field = _mlp(latent_dim + 1, field_width, latent_dim, field_depth)

    def velocity(self, state: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        if time.ndim == 1:
            time = time[:, None]
        return self.field(torch.cat((state, time.to(state)), dim=-1))

    def integrate(self, source: torch.Tensor, steps: int = 32) -> torch.Tensor:
        return self.trajectory(source, steps=steps)[-1]

    def trajectory(self, source: torch.Tensor, steps: int = 32) -> torch.Tensor:
        state = source
        points = [self.decoder(state)]
        dt = 1.0 / steps
        for index in range(steps):
            time = torch.full((len(state), 1), index * dt, device=state.device)
            state = state + dt * self.velocity(state, time)
            points.append(self.decoder(state))
        return torch.stack(points)

    def losses(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        *,
        method: str,
        flow_samples: int = 14,
        reconstruction_noise_min: float = 0.5,
        reconstruction_noise_max: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        if source.shape[-1] != self.latent_dim:
            raise ValueError(
                f"source width {source.shape[-1]} does not match latent_dim {self.latent_dim}"
            )
        clean = self.encoder(target)
        batch = len(source)
        source_many = source[:, None].expand(-1, flow_samples, -1).reshape(-1, self.latent_dim)
        clean_many = clean[:, None].expand(-1, flow_samples, -1).reshape(-1, self.latent_dim)
        time = torch.rand(batch * flow_samples, 1, device=source.device)
        state = (1.0 - time) * source_many + time * clean_many
        flow = (self.velocity(state, time) - (clean_many - source_many)).square().mean()

        if method == "unite":
            reconstruction = self.decoder(clean)
        elif method == "vfm":
            severity = torch.empty(batch, 1, device=source.device).uniform_(
                reconstruction_noise_min, reconstruction_noise_max
            )
            bridge_time = 1.0 - severity
            noisy = severity * source + bridge_time * clean
            predicted_clean = noisy + severity * self.velocity(noisy, bridge_time)
            reconstruction = self.decoder(predicted_clean)
        else:
            raise ValueError(f"unknown method: {method}")
        reconstruction_loss = (reconstruction - target).square().mean()
        return {
            "loss": flow + reconstruction_loss,
            "flow_loss": flow,
            "reconstruction_loss": reconstruction_loss,
        }


class SyntheticDirectFlow(nn.Module):
    """Standard flow matching directly between 3D source and target points."""

    def __init__(
        self,
        *,
        data_dim: int = 3,
        field_width: int = 128,
        field_depth: int = 4,
    ) -> None:
        super().__init__()
        self.latent_dim = int(data_dim)
        self.field = _mlp(
            self.latent_dim + 1,
            field_width,
            self.latent_dim,
            field_depth,
        )

    def velocity(self, state: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        if time.ndim == 1:
            time = time[:, None]
        return self.field(torch.cat((state, time.to(state)), dim=-1))

    def integrate(self, source: torch.Tensor, steps: int = 32) -> torch.Tensor:
        return self.trajectory(source, steps=steps)[-1]

    def trajectory(self, source: torch.Tensor, steps: int = 32) -> torch.Tensor:
        state = source
        points = [state]
        dt = 1.0 / steps
        for index in range(steps):
            time = torch.full((len(state), 1), index * dt, device=state.device)
            state = state + dt * self.velocity(state, time)
            points.append(state)
        return torch.stack(points)

    def losses(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        *,
        flow_samples: int = 1,
    ) -> dict[str, torch.Tensor]:
        expected = (self.latent_dim,)
        if source.shape[-1:] != expected or target.shape[-1:] != expected:
            raise ValueError(
                f"direct flow expects source and target width {self.latent_dim}"
            )
        batch = len(source)
        source_many = source[:, None].expand(-1, flow_samples, -1).reshape(
            -1, self.latent_dim
        )
        target_many = target[:, None].expand(-1, flow_samples, -1).reshape(
            -1, self.latent_dim
        )
        time = torch.rand(batch * flow_samples, 1, device=source.device)
        state = (1.0 - time) * source_many + time * target_many
        flow = (
            self.velocity(state, time) - (target_many - source_many)
        ).square().mean()
        return {"loss": flow, "flow_loss": flow}
