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

    def __init__(self, model, language_key=None):
        super().__init__()
        self.model = model
        if language_key is not None and (not isinstance(language_key, str) or not language_key):
            raise ValueError("language_key must be a nonempty batch key or None")
        self.language_key = language_key
        if language_key is not None:
            self.reads_by_mode = {
                mode: (*reads, language_key) for mode, reads in self.reads_by_mode.items()
            }

    def execute(self, batch, *, mode):
        batch["joint/model"] = self.model
        if mode == "train":
            batch["joint/prediction"] = self.model(
                batch["world"],
                _embodiment(batch),
                batch["joint/values"],
                batch["joint/levels"],
                **({"language_features": batch[self.language_key]} if self.language_key is not None else {}),
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

    def __init__(self, stream, loss, weight=1.0, optional=False, log_objective=False):
        super().__init__()
        if (
            stream not in ("image", "proprio", "action", "camera")
            or not math.isfinite(weight)
            or weight < 0
        ):
            raise ValueError(
                "Declare a joint stream and finite nonnegative loss weight"
            )
        self.stream, self.loss, self.weight = stream, loss, weight
        self.optional = optional
        if type(log_objective) is not bool:
            raise ValueError("log_objective must be an explicit boolean")
        if log_objective and not callable(getattr(loss, "loss_and_metrics", None)):
            raise ValueError("Objective logging requires loss_and_metrics")
        if getattr(loss, "requires_objective_metrics", False) and not log_objective:
            raise ValueError("Weighted objective requires separate unweighted metrics")
        self.log_objective = log_objective
        self.writes = ("loss/" + stream,)

    def execute(self, batch, *, mode):
        k = self.stream
        if k not in batch["joint/prediction"]:
            if not self.optional:
                raise ValueError(f"Required stream {k} absent")
            # Graph bookkeeping only: no fabricated human action tokens/targets.
            loss = batch["joint/prediction"]["image"].sum() * 0
            batch[self.writes[0]] = loss
            return batch
        evaluate = self.loss.loss_and_metrics if self.log_objective else self.loss
        result = evaluate(
            batch["joint/prediction"][k],
            batch["joint/targets"][k],
            batch["joint/masks"][k],
            noise_levels=batch["joint/levels"][k],
            metadata=batch["joint/metadata"][k],
        )
        if self.log_objective:
            loss, metrics = result
            if "mse" not in metrics:
                raise ValueError("Objective logging requires an unweighted MSE metric")
            batch["log/joint_" + k + "_objective"] = loss.detach()
            for name, value in metrics.items():
                batch["log/joint_" + k + "_" + name] = value.detach()
        else:
            loss = result
            batch["log/joint_" + k + "_mse"] = loss.detach()
        batch[self.writes[0]] = loss * self.weight
        return batch


class JointSamplerStage(Stage):
    inference_only = True
    reads = ("world", "embodiment", "joint/model")
    writes = ("image_proprio_prediction",)

    def __init__(self, sampler, language_key=None):
        super().__init__()
        self.sampler = sampler
        if language_key is not None and (not isinstance(language_key, str) or not language_key):
            raise ValueError("language_key must be a nonempty batch key or None")
        self.language_key = language_key
        if language_key is not None:
            self.reads = (*self.reads, language_key)

    def execute(self, batch, *, mode):
        batch[self.writes[0]] = self.sampler(
            batch["joint/model"], batch["world"], _embodiment(batch),
            **({"language_features": batch[self.language_key]} if self.language_key is not None else {}),
        )
        return batch
