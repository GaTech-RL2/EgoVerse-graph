"""Replaceable pixel noising, objective, and image sampling components.

These objects own no trainable parameters. The graph registers the denoiser
once and passes it to the sampler; archived model convenience methods delegate
to the same implementations. Diffusion strength and physical time remain
separate inputs, including at the loss boundary for future objective choices.
"""

import math

import torch

from egomimic.models.world_model import masked_mse


class LinearGaussianImageNoising:
    """Linear corruption with one uniform strength per queried timestamp.

    All cameras and patches at a physical timestamp share a strength. Gaussian
    noise remains independent per channel. Draw order is timestamp strengths for
    each example, then one full-shaped Gaussian draw, including observed/padded
    positions whose noise is subsequently discarded.
    """

    def __init__(self, maximum=1.0):
        self._check_maximum(maximum)
        self.maximum = maximum

    @staticmethod
    def _check_maximum(maximum):
        if not math.isfinite(maximum) or not 0 <= maximum <= 1:
            raise ValueError("maximum noise strength must be finite in [0,1]")

    @staticmethod
    def prepare_target(image):
        """Return recorded clean targets and the queried supervision mask."""
        known = image["valid"] & image.get("target_valid", image["valid"])
        if (image["valid"] & ~known).any():
            raise ValueError(
                "Visual training requires recorded features for every valid image token"
            )
        clean = torch.where(known[..., None], image["value"], 0)
        if not torch.isfinite(clean).all():
            raise ValueError("Non-finite image target")
        return clean, known & ~image["observed"]

    @staticmethod
    def frame_noise_levels(stream, maximum=1.0):
        LinearGaussianImageNoising._check_maximum(maximum)
        levels = torch.zeros_like(stream["time"], dtype=torch.float32)
        for b in range(levels.shape[0]):
            selected = stream["valid"][b] & ~stream["observed"][b]
            if selected.any():
                times, inverse = torch.unique(
                    stream["time"][b][selected], return_inverse=True
                )
                draws = torch.rand(times.numel(), device=levels.device) * maximum
                levels[b, selected] = draws[inverse]
        return levels

    def __call__(self, image, *, clean=None, maximum=None):
        maximum = self.maximum if maximum is None else maximum
        self._check_maximum(maximum)
        if clean is None:
            clean, mask = self.prepare_target(image)
        else:
            # Internal legacy low-noise passes provide an already checked target.
            if clean.shape != image["value"].shape:
                raise ValueError("Clean image target must match image token shape")
            mask = (
                image["valid"]
                & image.get("target_valid", image["valid"])
                & ~image["observed"]
            )
        levels = self.frame_noise_levels(image, maximum)
        strength = levels[..., None].to(clean.dtype)
        noisy = (1 - strength) * clean + strength * torch.randn_like(clean)
        noisy = noisy.masked_fill(~image["valid"][..., None], 0)
        return {
            "image_values": noisy,
            "noise_levels": levels,
            "target": clean,
            "mask": mask,
            "metadata": {
                "timestamps": image["time"],
                "valid": image["valid"],
                "observed": image["observed"],
            },
        }


class PlainPixelMSE:
    """FP32 plain x0 MSE; noise/time inputs are explicit and unweighted.

    The configurable loss boundary deliberately accepts noise levels and metadata
    without using them. A later objective can consume these inputs without changing
    the noising stage or denoiser. No SNR weighting is applied here.
    """

    def __init__(self, reduction="mean_per_example"):
        if reduction not in ("token_mean", "mean_per_example"):
            raise ValueError("Unknown masked MSE reduction")
        self.reduction = reduction

    def __call__(
        self,
        prediction,
        target,
        mask,
        noise_levels=None,
        metadata=None,
        *,
        check_target=True,
    ):
        return masked_mse(
            prediction,
            target,
            mask.float(),
            reduction=self.reduction,
            check_target=check_target,
        )


class LinearDDIMImageSampler:
    """Integrate x0 predictions, then return images without state readout.

    The denoiser is the graph's existing registered model, not a copied submodule.
    Its image-only attention layout and FP32 RoPE tables are reused across steps.
    With standard BF16 mixed precision, recorded patches/state are FP32 and every
    prediction is promoted to that state dtype before integration arithmetic.
    """

    def __init__(self, steps=20):
        self._check_steps(steps)
        self.steps = steps

    @staticmethod
    def _check_steps(steps):
        if type(steps) is not int or steps < 1:
            raise ValueError("steps must be a positive integer")

    @torch.no_grad()
    def __call__(
        self,
        denoiser,
        batch,
        embodiment,
        *,
        steps=None,
        generator=None,
        validated=False,
    ):
        steps = self.steps if steps is None else steps
        if not validated:
            denoiser.validate(batch, embodiment)
        self._check_steps(steps)
        image = batch["image"]
        valid, observed = image["valid"], image["observed"]
        anchor = torch.where(observed[..., None], image["value"], 0)
        noise = torch.randn(
            image["value"].shape,
            device=anchor.device,
            dtype=anchor.dtype,
            generator=generator,
        )
        current = torch.where(observed[..., None], anchor, noise).masked_fill(
            ~valid[..., None], 0
        )
        execution = None
        if not denoiser.has_proprio_branch or (
            (
                denoiser.token_execution == "packed_valid"
                or denoiser.noise_conditioning == "adaln_zero"
            )
            and not denoiser.phase_a_condition_on_proprio
        ):
            execution = denoiser._prepare_image_execution(image, valid)
        for j in range(steps, 0, -1):
            level, next_level = j / steps, (j - 1) / steps
            levels = torch.full_like(
                image["time"], level, dtype=torch.float32
            ).masked_fill(observed | ~valid, 0)
            x0 = denoiser(
                batch,
                embodiment,
                current,
                levels,
                phase="A",
                _validated=True,
                _execution=execution,
            )["image"].to(current.dtype)
            candidate = (1 - next_level) * x0 + (next_level / level) * (
                current - (1 - level) * x0
            )
            current = torch.where(observed[..., None], anchor, candidate).masked_fill(
                ~valid[..., None], 0
            )
        return current
