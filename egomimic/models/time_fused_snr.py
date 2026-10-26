"""Parameter-free time-aware strongest-anchor weighting for continuous x0.

This is an explicit fused-SNR-style heuristic, not Diffusion Forcing's EMA.
It changes the training objective only: corruption, model inputs, sampling RNG,
and unweighted evaluation stay with their existing owners.
"""

import math
from collections.abc import Mapping

import torch


class TimeAwareFusedSNRMSE:
    """Weight x0 squared error using actual same-stream noisy observations.

    For x_s=(1-s)*x0+s*epsilon, let u=min(((1-s)/s)^2,cap)/cap.
    Another temporal noise group contributes exp(-|dt|/tau)*u; h is the
    strongest such contribution, or zero without another group. The numerator
    weight is fixed_scale*cap*(u+(1-u)*h). The denominator remains the ORIGINAL
    supervision count, including at zero weight. No batch normalization occurs.

    Grouping is within each sample/stream, by timestamp and actual noise role:
    observed input, supervised query, or unknown pure-noise query. Repeated
    camera/patch tokens cannot become extra witnesses. Same-time observed and
    queried tokens may differ only because they are genuinely distinct roles.
    Unknown queries must be pure noise; image strengths cannot supply their SNR.

    cap=5 and tau_seconds=1 are provisional experiment settings. fixed_scale=1
    means the raw capped objective; any alternative calibration must be frozen
    externally from the training population, never recomputed by this class.
    """

    requires_objective_metrics = True

    def __init__(self, cap=5.0, tau_seconds=1.0, fixed_scale=1.0,
                 reduction="mean_per_example"):
        if reduction not in ("mean_per_example", "token_mean"):
            raise ValueError("Unknown masked MSE reduction")
        for name, value in (("cap", cap), ("tau_seconds", tau_seconds),
                            ("fixed_scale", fixed_scale)):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(name + " must be a positive finite FP32 number")
            # Constructor must also work under the graph's meta-device audit;
            # these pure scalar checks must not instantiate/read a Tensor.
            if not math.isfinite(value) or not 2.0 ** -149 <= value <= torch.finfo(torch.float32).max:
                raise ValueError(name + " must be a positive finite FP32 number")
        maximum = cap * fixed_scale
        if not math.isfinite(maximum) or not 2.0 ** -149 <= maximum <= torch.finfo(torch.float32).max:
            raise ValueError("Scaled weight cap must be positive and finite in FP32")
        self.cap, self.tau_seconds = float(cap), float(tau_seconds)
        self.fixed_scale, self.reduction = float(fixed_scale), reduction

    @staticmethod
    def _tensor(value, shape, device, *, boolean=False, name):
        if (not torch.is_tensor(value) or value.shape != shape
                or value.device != device
                or (value.dtype != torch.bool if boolean else not value.is_floating_point())):
            raise ValueError(name + " has an incompatible shape, device or dtype")
        return value

    @torch.no_grad()
    def _weights_and_support(self, noise_levels, metadata, mask):
        if not torch.is_tensor(mask) or mask.ndim != 2 or mask.dtype != torch.bool:
            raise ValueError("The original BxN boolean supervision mask is required")
        if not isinstance(metadata, Mapping):
            raise ValueError("Actual timestamps/valid/observed metadata is required")
        shape, device = mask.shape, mask.device
        valid = self._tensor(metadata.get("valid"), shape, device,
                             boolean=True, name="valid")
        observed = self._tensor(metadata.get("observed"), shape, device,
                                boolean=True, name="observed")
        times = self._tensor(metadata.get("timestamps"), shape, device,
                             name="timestamps")
        levels = self._tensor(noise_levels, shape, device, name="noise_levels")
        if (observed & ~valid).any() or (mask & (~valid | observed)).any():
            raise ValueError("Supervision/observed roles disagree with valid tokens")
        with torch.autocast(device_type=device.type, enabled=False):
            levels, times = levels.detach().float(), times.detach().float()
            if (not torch.isfinite(levels[valid]).all()
                    or not torch.isfinite(times[valid]).all()
                    or ((levels[valid] < 0) | (levels[valid] > 1)).any()):
                raise ValueError("Valid noise strengths/timestamps must be finite and ranged")
            unknown = valid & ~observed & ~mask
            if (levels[observed] != 0).any() or (levels[unknown] != 1).any():
                raise ValueError("Observed inputs must be clean and unknown queries pure noise")
            weights, support = torch.zeros_like(levels), torch.zeros_like(levels)
            threshold = 1.0 / (1.0 + math.sqrt(self.cap))
            for b in range(shape[0]):
                take = valid[b]
                if not take.any():
                    continue
                unique_times, time_index = torch.unique(times[b, take], return_inverse=True)
                role = torch.where(observed[b, take], 0,
                                   torch.where(mask[b, take], 1, 2))
                groups, inverse = torch.unique(time_index * 3 + role, return_inverse=True)
                count = groups.numel()
                strength = levels[b, take]
                low = torch.full((count,), float("inf"), device=device, dtype=torch.float32)
                high = torch.full((count,), -float("inf"), device=device, dtype=torch.float32)
                low.scatter_reduce_(0, inverse, strength, reduce="amin")
                high.scatter_reduce_(0, inverse, strength, reduce="amax")
                if not torch.equal(low, high):
                    raise ValueError("Repeated tokens in a temporal noise group disagree")
                # Normalize BEFORE square to avoid overflow, and clamp the
                # divisor at the clipping threshold so s=0 never divides by 0.
                # Clipping the ratio also gives exact u(1)=0 for very small caps
                # whose threshold rounds to 1. This equals clipped SNR / cap.
                ratio = ((1.0 - low) / low.clamp_min(threshold)) / math.sqrt(self.cap)
                u = ratio.clamp(0.0, 1.0).square()
                u = torch.where(low == 0, torch.ones_like(u),
                                torch.where(low == 1, torch.zeros_like(u), u))
                group_times = unique_times[torch.div(groups, 3, rounding_mode="floor")]
                distance = (group_times[:, None] - group_times[None, :]).abs()
                candidates = torch.exp(-distance / self.tau_seconds) * u[None, :]
                candidates.fill_diagonal_(0.0)
                h = candidates.amax(dim=1)
                raw_weight = self.cap * (u + (1.0 - u) * h)
                if (not torch.isfinite(raw_weight).all()
                        or (raw_weight < 0).any() or (raw_weight > self.cap).any()):
                    raise ValueError("Nonfinite or out-of-range fused-SNR weight")
                weights[b, take] = raw_weight[inverse] * self.fixed_scale
                support[b, take] = h[inverse]
            if not torch.isfinite(weights).all():
                raise ValueError("Nonfinite scaled fused-SNR weight")
            return weights, support

    def weights(self, noise_levels, metadata, mask):
        """Return detached FP32 token weights without reading predictions/labels."""
        return self._weights_and_support(noise_levels, metadata, mask)[0]

    def _reduce(self, values, mask):
        count = mask.float()
        if self.reduction == "mean_per_example":
            denominator = count.sum(1).clamp_min(1.0)
            return (values.sum(1) / denominator).mean()
        return values.sum() / count.sum().clamp_min(1.0)

    def loss_and_metrics(self, prediction, target, mask, noise_levels=None,
                         metadata=None, *, check_target=True):
        if check_target is not True:
            raise ValueError("Fused-SNR loss always checks all supervised targets")
        if (not torch.is_tensor(prediction) or prediction.ndim != 3
                or not prediction.is_floating_point() or prediction.shape[0] == 0
                or prediction.shape[2] == 0):
            raise ValueError("Prediction must be a nonempty BxNxC floating tensor")
        self._tensor(target, prediction.shape, prediction.device, name="target")
        self._tensor(mask, prediction.shape[:2], prediction.device,
                     boolean=True, name="mask")
        weights, support = self._weights_and_support(noise_levels, metadata, mask)
        with torch.autocast(device_type=prediction.device.type, enabled=False):
            # Sanitize BOTH operands outside the original mask, BEFORE square.
            # Otherwise 0*NaN can poison loss or gradients. Zero SNR weight does
            # not remove an originally supervised error from these checks.
            safe_prediction = torch.where(mask[..., None], prediction.float(), 0.0)
            safe_target = torch.where(mask[..., None], target.float(), 0.0)
            if not torch.isfinite(safe_prediction).all() or not torch.isfinite(safe_target).all():
                raise ValueError("Nonfinite originally supervised prediction or target")
            error = (safe_prediction - safe_target).square().mean(-1)
            if not torch.isfinite(error).all():
                raise ValueError("Nonfinite originally supervised squared error")
            objective = self._reduce(error * weights, mask)
            plain = self._reduce(error, mask)
            if not torch.isfinite(objective) or not torch.isfinite(plain):
                raise ValueError("Nonfinite reduced fused objective or unweighted MSE")
            metrics = {
                "mse": plain.detach(),
                "fused_weight_mean": self._reduce(weights * mask, mask).detach(),
                "fused_raw_weight_mean": self._reduce(weights * mask, mask).detach() / self.fixed_scale,
                "fused_support_mean": self._reduce(support * mask, mask).detach(),
                "fused_zero_weight_fraction": self._reduce(((weights == 0) & mask).float(), mask).detach(),
            }
        return objective, metrics

    def __call__(self, prediction, target, mask, noise_levels=None,
                 metadata=None, *, check_target=True):
        return self.loss_and_metrics(prediction, target, mask, noise_levels,
                                     metadata, check_target=check_target)[0]
