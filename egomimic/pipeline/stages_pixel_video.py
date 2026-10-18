"""Replaceable image-only diffusion stages with one registered denoiser."""

import math

from egomimic.pipeline.core import Stage, resolve_homogeneous_scalar


def _embodiment(batch):
    return str(resolve_homogeneous_scalar(batch["embodiment"], label="embodiment"))


class PixelVideoNoisingStage(Stage):
    """Draw corruption independently of the denoiser and training objective."""

    train_only = True
    reads = ("world",)
    writes = (
        "pixel_video/noisy_image",
        "pixel_video/noise_levels",
        "pixel_video/target",
        "pixel_video/loss_mask",
        "pixel_video/noise_metadata",
    )

    def __init__(self, noising):
        super().__init__()
        self.noising = noising

    def execute(self, batch, *, mode):
        result = self.noising(batch["world"]["image"])
        for destination, source in zip(
            self.writes, ("image_values", "noise_levels", "target", "mask", "metadata")
        ):
            batch[destination] = result[source]
        return batch


class PixelVideoDenoiserStage(Stage):
    """Own the only denoiser parameters and expose them to the sampler."""

    reads_by_mode = {
        "train": (
            "world",
            "embodiment",
            "pixel_video/noisy_image",
            "pixel_video/noise_levels",
        ),
        "inference": (),
    }
    writes_by_mode = {
        "train": ("pixel_video/prediction", "pixel_video/denoiser"),
        "inference": ("pixel_video/denoiser",),
    }

    def __init__(self, model):
        super().__init__()
        if (
            model.objective not in ("visual_only", "visual_state_action")
            or model.phase_a_condition_on_proprio
        ):
            raise ValueError("Pixel video denoising requires an image-only Phase A")
        self.model = model

    def execute(self, batch, *, mode):
        # A transient reference shares the sole registered model with any
        # configured sampler/readout branch, without a second parameter copy.
        batch["pixel_video/denoiser"] = self.model
        if mode == "train":
            batch["pixel_video/prediction"] = self.model(
                batch["world"],
                _embodiment(batch),
                batch["pixel_video/noisy_image"],
                batch["pixel_video/noise_levels"],
                phase="A",
            )["image"]
        return batch


class PixelVideoLossStage(Stage):
    """Configure the objective separately, retaining temporal/noise metadata."""

    train_only = True
    reads = (
        "pixel_video/prediction",
        "pixel_video/target",
        "pixel_video/loss_mask",
        "pixel_video/noise_levels",
        "pixel_video/noise_metadata",
    )
    writes = ("loss/pixel_video",)

    def __init__(self, loss, weight=1.0):
        super().__init__()
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("Pixel loss weight must be finite and nonnegative")
        self.loss, self.weight = loss, weight

    def execute(self, batch, *, mode):
        loss = self.loss(
            batch["pixel_video/prediction"],
            batch["pixel_video/target"],
            batch["pixel_video/loss_mask"],
            noise_levels=batch["pixel_video/noise_levels"],
            metadata=batch["pixel_video/noise_metadata"],
            check_target=False,
        )
        batch["loss/pixel_video"] = self.weight * loss
        batch["log/image_proprio_image"] = loss.detach()
        batch["log/image_proprio_total"] = batch["loss/pixel_video"].detach()
        metadata = batch["pixel_video/noise_metadata"]
        batch["log/image_proprio_image_tokens"] = metadata["valid"].sum().float()
        batch["log/image_proprio_image_query_tokens"] = (
            (metadata["valid"] & ~metadata["observed"]).sum().float()
        )
        return batch


class PixelVideoSamplerStage(Stage):
    """Run a replaceable inference integrator against the shared denoiser."""

    inference_only = True
    reads = ("world", "embodiment", "pixel_video/denoiser")

    def __init__(self, sampler, prediction_key="image_proprio_prediction"):
        super().__init__()
        if not isinstance(prediction_key, str) or not prediction_key:
            raise ValueError("Sampler prediction key must be a nonempty string")
        self.sampler = sampler
        self.prediction_key = prediction_key
        self.writes = (prediction_key,)

    def execute(self, batch, *, mode):
        image = self.sampler(
            batch["pixel_video/denoiser"], batch["world"], _embodiment(batch)
        )
        batch[self.prediction_key] = {"image": image}
        return batch
