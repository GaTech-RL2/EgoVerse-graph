"""Observation and noise stages shared by Pipeline policy families."""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn

from egomimic.pipeline.core import Stage


class DPStyleObsEncoder(nn.Module):
    """Concatenate low-dimensional state with encoded image observations."""

    def __init__(self, obs_specs: Dict[str, dict], img_encoders: Dict[str, nn.Module]):
        super().__init__()
        if not obs_specs or not img_encoders:
            raise ValueError("DPStyleObsEncoder needs low-dim and image inputs")
        self.obs_specs = {str(key): dict(spec) for key, spec in obs_specs.items()}
        self.img_encoders = nn.ModuleDict(img_encoders)

    def forward_packed(self, *, obs_packed: dict, T_total: int, **kwargs):
        features = []
        for key in sorted(self.obs_specs):
            spec = self.obs_specs[key]
            value = obs_packed[key]
            start, end = spec.get("input_slice", [0, value.shape[-1]])
            value = value[..., int(start) : int(end)]
            expected = int(spec.get("input_dim", value.shape[-1]))
            if value.shape[-1] != expected:
                raise ValueError(
                    f"DPStyleObsEncoder {key}: expected {expected} dims after "
                    f"slice, got {value.shape[-1]}"
                )
            features.append(value)
        for key in sorted(self.img_encoders):
            features.append(self.img_encoders[key](obs_packed[key]))
        if any(int(feature.shape[0]) != int(T_total) for feature in features):
            raise ValueError("DPStyleObsEncoder inputs do not share T_total")
        return torch.cat(features, dim=-1)


class FusedObsEncoder(Stage):
    """Encode an ``N``-observation window into one conditioning vector."""

    reads = ["obs/*", "embodiment", "actions"]
    writes = ["condition", "target"]
    reads_by_mode = {"rollout": ["obs/*", "embodiment"]}
    writes_by_mode = {"rollout": ["condition"]}

    def __init__(
        self,
        encoder: nn.Module,
        n_obs_steps: int = 2,
        required_obs_keys: List[str] | None = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.n_obs_steps = int(n_obs_steps)
        self.rollout_obs_steps = self.n_obs_steps
        if self.n_obs_steps <= 0:
            raise ValueError("n_obs_steps must be positive")
        if required_obs_keys is not None:
            required = tuple(
                key if str(key).startswith("obs/") else f"obs/{key}"
                for key in required_obs_keys
            )
            if not required:
                raise ValueError("required_obs_keys cannot be empty")
            self.reads = [*required, "embodiment", "actions"]
            self.reads_by_mode = {"rollout": [*required, "embodiment"]}

    def forward(self, batch: dict) -> dict:
        obs_packed = {
            key.split("/", 1)[1]: value
            for key, value in batch.items()
            if key.startswith("obs/")
        }
        try:
            reference = next(
                value for value in obs_packed.values() if torch.is_tensor(value)
            )
        except StopIteration as exc:
            raise ValueError("FusedObsEncoder requires tensor observations") from exc

        batch_size, n_obs = int(reference.shape[0]), self.n_obs_steps
        if n_obs == 1:
            for key, value in list(obs_packed.items()):
                if not torch.is_tensor(value):
                    continue
                if int(value.shape[0]) != batch_size:
                    raise ValueError(
                        f"FusedObsEncoder: obs/{key} has batch {value.shape[0]}; "
                        f"expected {batch_size}"
                    )
                if "rollout_t" in batch:
                    if value.ndim < 2 or value.shape[1] != 1:
                        raise ValueError(
                            f"FusedObsEncoder rollout obs/{key} must have an "
                            f"explicit singleton history axis, got {tuple(value.shape)}"
                        )
                    obs_packed[key] = value[:, 0]
        else:
            if reference.ndim < 2 or reference.shape[1] != n_obs:
                raise ValueError(
                    f"FusedObsEncoder got shape {tuple(reference.shape)}; expected "
                    f"(B, {n_obs}, ...)"
                )
            for key, value in list(obs_packed.items()):
                if not torch.is_tensor(value):
                    continue
                if value.shape[:2] != (batch_size, n_obs):
                    raise ValueError(
                        f"FusedObsEncoder: obs/{key} has shape {tuple(value.shape)}; "
                        f"expected leading dimensions {(batch_size, n_obs)}"
                    )
                obs_packed[key] = value.reshape(batch_size * n_obs, *value.shape[2:])

        total = batch_size * n_obs
        device = reference.device
        dtype = reference.dtype if reference.is_floating_point() else torch.float32
        donor = torch.zeros((total, 1), device=device, dtype=dtype)
        cu_seqlens = torch.arange(0, total + 1, n_obs, device=device, dtype=torch.long)
        for module in self.modules():
            if getattr(module, "crop_scope", None) == "episode":
                module._episode_cu = cu_seqlens
        encoded = self.encoder.forward_packed(
            actions_packed=donor,
            obs_packed=obs_packed,
            cu_seqlens=cu_seqlens,
            T_total=total,
            device=device,
            dtype=dtype,
            embodiment_id=str(batch["embodiment"]),
        )
        batch["condition"] = encoded.reshape(batch_size, n_obs * encoded.shape[-1])

        target = batch.pop("actions", None)
        if target is None and "rollout_t" not in batch:
            raise ValueError("FusedObsEncoder: training batch has no actions target")
        if target is not None:
            batch["target"] = target
        return batch


class GaussianLatentNoise(Stage):
    """Create Gaussian latent tokens without reading the action target."""

    reads = ["condition"]
    writes = ["sampler/noise"]

    def __init__(
        self,
        num_tokens: int | None = None,
        latent_dim: int = 128,
        action_horizon: int | None = None,
    ):
        super().__init__()
        if num_tokens is None and action_horizon is None:
            raise ValueError("num_tokens must be configured")
        if num_tokens is not None and action_horizon is not None:
            raise ValueError("Configure num_tokens or legacy action_horizon, not both")
        self.num_tokens = int(action_horizon if num_tokens is None else num_tokens)
        self.latent_dim = int(latent_dim)
        if self.num_tokens <= 0 or self.latent_dim <= 0:
            raise ValueError("num_tokens and latent_dim must be positive")

    @property
    def action_horizon(self) -> int:
        """Compatibility alias for the original latent-sampler configs."""
        return self.num_tokens

    def forward(self, batch: dict) -> dict:
        condition = batch["condition"]
        batch_size = int(condition.shape[0])
        if batch_size <= 0:
            raise ValueError("condition must describe at least one sample")
        device = condition.device
        dtype = (
            torch.get_autocast_dtype(device.type)
            if torch.is_autocast_enabled(device.type)
            else torch.get_default_dtype()
        )
        batch["sampler/noise"] = torch.randn(
            batch_size,
            self.num_tokens,
            self.latent_dim,
            dtype=dtype,
            device=device,
        )
        return batch
