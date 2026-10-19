"""Independent graph nodes for coupled-state / independent-command diffusion."""

import math

from egomimic.pipeline.core import Stage
from egomimic.pipeline.stages_pixel_video import _embodiment


class JointNoisingStage(Stage):
    train_only = True
    reads = ("world",)
    writes = tuple(
        "joint/" + key for key in ("values", "levels", "targets", "masks", "metadata")
    )

    def __init__(self, noising):
        super().__init__()
        self.noising = noising

    def execute(self, batch, *, mode):
        for key, value in self.noising(batch["world"]).items():
            batch["joint/" + key] = value
        return batch


class JointDenoiserStage(Stage):
    reads_by_mode = {
        "train": ("world", "embodiment", "joint/values", "joint/levels"),
        "inference": (),
    }
    writes_by_mode = {
        "train": ("joint/prediction", "joint/model"),
        "inference": ("joint/model",),
    }

    def __init__(self, model):
        super().__init__()
        self.model = model

    def execute(self, batch, *, mode):
        batch["joint/model"] = self.model
        if mode == "train":
            batch["joint/prediction"] = self.model(
                batch["world"],
                _embodiment(batch),
                batch["joint/values"],
                batch["joint/levels"],
            )
        return batch


class JointLossStage(Stage):
    train_only = True
    reads = (
        "joint/prediction",
        "joint/targets",
        "joint/masks",
        "joint/levels",
        "joint/metadata",
    )

    def __init__(self, stream, loss, weight=1.0):
        super().__init__()
        if (
            stream not in ("image", "proprio", "action")
            or not math.isfinite(weight)
            or weight < 0
        ):
            raise ValueError(
                "Declare a joint stream and finite nonnegative loss weight"
            )
        self.stream, self.loss, self.weight = stream, loss, weight
        self.writes = ("loss/" + stream,)

    def execute(self, batch, *, mode):
        k = self.stream
        loss = self.loss(
            batch["joint/prediction"][k],
            batch["joint/targets"][k],
            batch["joint/masks"][k],
            noise_levels=batch["joint/levels"][k],
            metadata=batch["joint/metadata"][k],
        )
        batch[self.writes[0]] = loss * self.weight
        batch["log/joint_" + k + "_mse"] = loss.detach()
        return batch


class JointSamplerStage(Stage):
    inference_only = True
    reads = ("world", "embodiment", "joint/model")
    writes = ("image_proprio_prediction",)

    def __init__(self, sampler):
        super().__init__()
        self.sampler = sampler

    def execute(self, batch, *, mode):
        batch[self.writes[0]] = self.sampler(
            batch["joint/model"], batch["world"], _embodiment(batch)
        )
        return batch
