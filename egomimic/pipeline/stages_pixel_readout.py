"""Explicit state/command readout graph sharing one pixel diffusion model.

The initial readout phase freezes every copied image parameter. Legacy models
with learned phase offsets freeze that entire embedding too; new configurations
omit it. State/control adapters, queries, expert and heads remain trainable.
"""

import math

import torch

from egomimic.pipeline.core import Stage
from egomimic.pipeline.stages_pixel_video import (
    PixelVideoDenoiserStage,
    PixelVideoLossStage,
    PixelVideoNoisingStage,
    _embodiment,
)


def configure_readout_trainability(model, training_mode):
    """Partition every parameter explicitly; never freeze only an embedding row."""
    if training_mode not in ("frozen_image", "joint"):
        raise ValueError("Readout training_mode must be frozen_image or joint")
    if model.objective != "visual_state_action" or model.phase_a_condition_on_proprio:
        raise ValueError("Readout requires visual_state_action with image-only Phase A")
    image_modules = [
        model.image_in,
        model.image_out,
        model.noise_level,
        model.norm[0],
    ]
    if hasattr(model, "xy"):
        image_modules.append(model.xy)
    if model.noise_conditioning == "adaln_zero":
        image_modules.append(model.final_adaln)
    if model.use_phase_embedding:
        image_modules.append(model.phase)
    if model.use_camera_embedding:
        image_modules.append(model.camera)
    if model.use_embodiment_embedding:
        image_modules.append(model.embodiment)
    readout_modules = [
        model.pose_in,
        model.pose_out,
        model.action_out,
        model.proprio_role,
        model.norm[1],
    ]
    for block in model.blocks:
        if model.noise_conditioning == "adaln_zero":
            image_modules.append(block.adaln_image)
        for name in ("norm1", "qkv", "out", "norm2", "ffn"):
            branches = getattr(block, name)
            if len(branches) != 2:
                raise ValueError(
                    "Readout requires exactly image and shared proprio experts"
                )
            image_modules.append(branches[0])
            readout_modules.append(branches[1])
    image_ids = {id(p) for module in image_modules for p in module.parameters()}
    if model.use_modality_embedding:
        image_ids.add(id(model.modality))
    readout_ids = {id(p) for module in readout_modules for p in module.parameters()}
    readout_ids.update((id(model.query), id(model.action_query)))
    parameters = dict(model.named_parameters())
    if image_ids & readout_ids or image_ids | readout_ids != {
        id(p) for p in parameters.values()
    }:
        raise ValueError(
            "Readout trainability partition has unknown or shared parameters"
        )
    for parameter in parameters.values():
        parameter.requires_grad_(
            training_mode == "joint" or id(parameter) in readout_ids
        )
    return {
        "training_mode": training_mode,
        "frozen_parameters": tuple(
            name for name, p in parameters.items() if not p.requires_grad
        ),
        "trainable_parameters": tuple(
            name for name, p in parameters.items() if p.requires_grad
        ),
        "phase_embedding_present": model.use_phase_embedding,
        "phase_embedding_frozen": model.use_phase_embedding
        and training_mode == "frozen_image",
        "noise_conditioning": model.noise_conditioning,
    }


class PixelReadoutDenoiserStage(PixelVideoDenoiserStage):
    """Legacy combined model owner and high-noise training prediction stage."""

    def __init__(self, model, training_mode="frozen_image"):
        super().__init__(model)
        self.trainability = configure_readout_trainability(model, training_mode)


class PixelReadoutModelStage(Stage):
    """Register/freeze the sole model without choosing any forward branch."""

    writes = ("pixel_video/denoiser",)

    def __init__(self, model, training_mode="frozen_image"):
        super().__init__()
        self.model = model
        self.trainability = configure_readout_trainability(model, training_mode)

    def execute(self, batch, *, mode):
        batch[self.writes[0]] = self.model
        return batch


class PixelReadoutVideoDenoiserStage(Stage):
    """Optional high-noise image prediction, independent of model ownership."""

    train_only = True
    reads = (
        "world",
        "embodiment",
        "pixel_video/denoiser",
        "pixel_video/noisy_image",
        "pixel_video/noise_levels",
    )
    writes = ("pixel_video/prediction",)

    def execute(self, batch, *, mode):
        batch[self.writes[0]] = batch["pixel_video/denoiser"](
            batch["world"],
            _embodiment(batch),
            batch["pixel_video/noisy_image"],
            batch["pixel_video/noise_levels"],
            phase="A",
        )["image"]
        return batch


class PixelReadoutImageLossStage(PixelVideoLossStage):
    """Image objective in a multi-loss graph; generic Train/loss is the total."""

    def execute(self, batch, *, mode):
        batch = super().execute(batch, mode=mode)
        # The base image-only node labels its sole loss as total. Here the two
        # readout objectives are separate nodes, so that partial total is invalid.
        batch.pop("log/image_proprio_total", None)
        return batch


class PixelReadoutNoisingStage(PixelVideoNoisingStage):
    """Independent low-noise corruption for paired-trajectory training."""

    writes = (
        "pixel_readout/noisy_image",
        "pixel_readout/noise_levels",
        "pixel_readout/target",
        "pixel_readout/loss_mask",
        "pixel_readout/noise_metadata",
    )


class PixelReadoutImageDenoiserStage(Stage):
    """Estimate aligned images using the same phase-A weights, without detaching."""

    train_only = True
    reads = (
        "world",
        "embodiment",
        "pixel_video/denoiser",
        "pixel_readout/noisy_image",
        "pixel_readout/noise_levels",
        "pixel_readout/target",
    )
    writes = ("pixel_readout/image_prediction",)

    def __init__(self, prediction_key="pixel_readout/image_prediction"):
        super().__init__()
        if not isinstance(prediction_key, str) or not prediction_key:
            raise ValueError("Image prediction key must be a nonempty string")
        self.writes = (prediction_key,)

    def execute(self, batch, *, mode):
        image = batch["world"]["image"]
        prediction = batch["pixel_video/denoiser"](
            batch["world"],
            _embodiment(batch),
            batch["pixel_readout/noisy_image"],
            batch["pixel_readout/noise_levels"],
            phase="A",
        )["image"]
        batch[self.writes[0]] = torch.where(
            image["observed"][..., None], batch["pixel_readout/target"], prediction
        ).masked_fill(~image["valid"][..., None], 0)
        return batch


class PixelReadoutSampledImageStage(Stage):
    """Exchangeable sampled-image provider with explicit stopped gradients.

    The built-in sampler does not backpropagate through generation. Frozen-image
    training can use this provider; joint training would need separate image
    objectives or the Phase-B path for gradients to the image parameters.
    """

    reads = ("world", "embodiment", "pixel_video/denoiser")

    def __init__(
        self,
        sampler,
        modes=("inference",),
        prediction_key="pixel_readout/image_prediction",
        gradient_policy="stop",
    ):
        super().__init__()
        if isinstance(modes, str):
            raise ValueError("Provider modes must be a sequence of train/inference")
        modes = tuple(modes)
        if (
            not modes
            or len(set(modes)) != len(modes)
            or not set(modes)
            <= {
                "train",
                "inference",
            }
        ):
            raise ValueError("Provider modes must select train and/or inference once")
        if not isinstance(prediction_key, str) or not prediction_key:
            raise ValueError("Image prediction key must be a nonempty string")
        if gradient_policy != "stop":
            raise ValueError(
                "Sampled image provider only supports gradient_policy=stop"
            )
        self.sampler = sampler
        self.modes, self.gradient_policy = modes, gradient_policy
        self.train_only = "inference" not in modes
        self.inference_only = "train" not in modes
        self.writes = (prediction_key,)

    def execute(self, batch, *, mode):
        with torch.no_grad():
            image = self.sampler(
                batch["pixel_video/denoiser"], batch["world"], _embodiment(batch)
            )
        if not isinstance(image, torch.Tensor):
            raise TypeError("Image providers must produce a tensor")
        batch[self.writes[0]] = image.detach()
        return batch


class PixelStateActionReadoutStage(Stage):
    """Phase B uses observed states and typed state/command queries, never labels."""

    reads_by_mode = {
        "train": (
            "world",
            "embodiment",
            "pixel_video/denoiser",
            "pixel_readout/image_prediction",
        ),
        "inference": (
            "world",
            "embodiment",
            "pixel_video/denoiser",
            "pixel_video/sampled_prediction",
        ),
    }
    writes_by_mode = {
        "train": ("pixel_readout/prediction",),
        "inference": ("image_proprio_prediction",),
    }

    def __init__(self, image_key=None):
        super().__init__()
        if image_key is not None:
            if not isinstance(image_key, str) or not image_key:
                raise ValueError("Readout image key must be a nonempty string")
            self.reads_by_mode = {
                mode: ("world", "embodiment", "pixel_video/denoiser", image_key)
                for mode in ("train", "inference")
            }
        self.image_key = image_key

    def execute(self, batch, *, mode):
        world = batch["world"]
        image = (
            batch[self.image_key]
            if self.image_key is not None
            else batch["pixel_readout/image_prediction"]
            if mode == "train"
            else batch["pixel_video/sampled_prediction"]["image"]
        )
        prediction = batch["pixel_video/denoiser"](
            world,
            _embodiment(batch),
            image,
            torch.zeros_like(world["image"]["time"], dtype=torch.float32),
            phase="B",
        )
        if mode == "train":
            batch["pixel_readout/prediction"] = prediction
        else:
            pose = world["proprio"]
            observed = pose["valid"] & pose["observed"]
            state = torch.where(
                observed[..., None], pose["value"], prediction["proprio"]
            )
            batch["image_proprio_prediction"] = {
                "image": image,
                "proprio": state.masked_fill(~pose["valid"][..., None], 0),
                "action": prediction["action"],
            }
        return batch


class PixelReadoutMSEStage(Stage):
    """One explicit supervised objective for measured state or actual commands."""

    train_only = True
    reads = ("world", "pixel_readout/prediction")

    def __init__(self, loss, stream, weight=1.0):
        super().__init__()
        if stream not in ("proprio", "action"):
            raise ValueError("Readout MSE stream must be proprio or action")
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("Readout loss weight must be finite and nonnegative")
        self.loss, self.stream, self.weight = loss, stream, weight
        name = "measured_state" if stream == "proprio" else "control_command"
        self.writes = (f"loss/{name}",)
        self.log_key = f"log/{name}_mse"

    def execute(self, batch, *, mode):
        stream = batch["world"][self.stream]
        mask = stream["valid"] & stream.get("target_valid", stream["valid"])
        if self.stream == "proprio":
            mask = mask & ~stream["observed"]
        loss = self.loss(
            batch["pixel_readout/prediction"][self.stream],
            stream["value"],
            mask,
            metadata={
                "timestamps": stream["time"],
                "valid": stream["valid"],
                "observed": stream["observed"],
            },
        )
        batch[self.writes[0]] = self.weight * loss
        batch[self.log_key] = loss.detach()
        return batch
