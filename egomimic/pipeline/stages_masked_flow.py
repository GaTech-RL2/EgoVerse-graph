"""Padding-aware flow targets and losses; existing unmasked recipes are unchanged."""

import torch

from egomimic.pipeline.core import Stage
from egomimic.pipeline.stages_flow import FlowNoisingStage, _validate_action_shape


def action_mask(batch, key, target):
    mask = batch[key]
    if mask.dtype != torch.bool or mask.shape != target.shape[:2] or target.ndim != 3:
        raise ValueError("Action validity must be boolean [B,H] for [B,H,D] actions")
    if not bool(mask.any(1).all()):
        raise ValueError("Each action window must contain at least one valid timestep")
    return mask[:, :, None]


class MaskedFlowNoisingStage(FlowNoisingStage):
    """Canonicalize padding before joint-horizon attention can mix it into valid rows.

    Optional configured draw keys allow exact stochastic replay for diagnostics.
    Both keys are required when enabled; shape/finite/range errors fail closed.
    """

    def __init__(
        self, *, mask_key="action_valid", noise_key=None, time_key=None, **kwargs
    ):
        super().__init__(**kwargs)
        if (noise_key is None) != (time_key is None):
            raise ValueError("Explicit noise and time keys must be configured together")
        self.mask_key, self.noise_key, self.time_key = mask_key, noise_key, time_key
        self.reads = ("target", mask_key) + ((noise_key, time_key) if noise_key else ())
        self.writes = (*super().writes, "target")

    def forward(self, batch):
        target = batch["target"].to(self.dtype)
        _validate_action_shape(
            target,
            batch_size=target.shape[0],
            action_horizon=self.action_horizon,
            action_dim=self.action_dim,
            label="Masked flow target",
        )
        mask = action_mask(batch, self.mask_key, target)
        target = torch.where(mask, target, 0.0)
        if not bool(torch.isfinite(target).all()):
            raise ValueError("Valid actions must be finite")
        batch["target"] = target
        if self.noise_key is None:
            return super().forward(batch)
        noise, time = batch[self.noise_key], batch[self.time_key]
        if noise.shape != target.shape or time.shape != target.shape[:1]:
            raise ValueError("Explicit flow draws must match actions and batch size")
        if not noise.is_floating_point() or not time.is_floating_point():
            raise ValueError("Flow draws must be floating point")
        noise, time = noise.to(target), time.to(target)
        if (
            not bool(torch.isfinite(noise).all())
            or not bool(torch.isfinite(time).all())
            or bool(((time < 0.001) | (time > 1)).any())
        ):
            raise ValueError("Flow noise must be finite and time must be in [0.001,1]")
        batch["flow/noisy_action"] = (
            time[:, None, None] * noise + (1 - time[:, None, None]) * target
        )
        batch["flow/velocity_target"] = noise - target
        batch["flow/time"] = time
        return batch


class MaskedFlowVelocityLossStage(Stage):
    train_only = True
    writes = ("loss/flow_velocity", "log/*")

    def __init__(self, mask_key="action_valid"):
        super().__init__()
        self.mask_key = mask_key
        self.reads = ("flow/predicted_velocity", "flow/velocity_target", mask_key)

    def forward(self, batch):
        prediction, target = (
            batch["flow/predicted_velocity"],
            batch["flow/velocity_target"],
        )
        if prediction.shape != target.shape:
            raise ValueError("Masked flow prediction and target shapes differ")
        mask = action_mask(batch, self.mask_key, target)
        error = torch.where(mask, prediction.float() - target.float(), 0.0)
        if not bool(torch.isfinite(error).all()):
            raise ValueError("Valid flow errors must be finite")
        count = mask.sum() * target.shape[-1]
        loss = error.square().sum() / count
        batch["loss/flow_velocity"] = loss
        batch["log/flow_velocity"] = loss.detach()
        batch["log/valid_action_scalars"] = count.detach()
        return batch
