"""Pipeline adapter for the independent image/track/proprio world model."""

from egomimic.pipeline.core import Stage, resolve_homogeneous_scalar


class WorldModelStage(Stage):
    reads_by_mode = {
        "train": ("world", "embodiment", "optimizer_step"),
        "inference": ("world", "embodiment"),
    }
    writes_by_mode = {"train": ("loss/world",), "inference": ("world_prediction",)}

    def __init__(
        self,
        model,
        warmup_steps=10000,
        pose_ramp_steps=10000,
        image_weight=1.0,
        track_weight=1.0,
        pose_weight=1.0,
        sampler_steps=20,
    ):
        super().__init__()
        if (
            min(warmup_steps, pose_ramp_steps) < 0
            or min(image_weight, track_weight, pose_weight) < 0
        ):
            raise ValueError("Schedules and weights must be nonnegative")
        self.model = model
        self.warmup_steps, self.pose_ramp_steps = warmup_steps, pose_ramp_steps
        self.image_weight, self.track_weight, self.pose_weight = (
            image_weight,
            track_weight,
            pose_weight,
        )
        self.sampler_steps = sampler_steps

    def pose_scale(self, step):
        if step < self.warmup_steps:
            return 0.0
        if self.pose_ramp_steps == 0:
            return self.pose_weight
        return self.pose_weight * min(
            1.0, (step - self.warmup_steps) / self.pose_ramp_steps
        )

    def execute(self, batch, *, mode):
        embodiment = str(
            resolve_homogeneous_scalar(batch["embodiment"], label="embodiment")
        )
        if mode == "train":
            scale = self.pose_scale(int(batch["optimizer_step"]))
            losses = self.model.training_losses(
                batch["world"], embodiment, scale, self.image_weight, self.track_weight
            )
            batch["loss/world"] = losses["total"]
            batch.update({f"log/world_{k}": v.detach() for k, v in losses.items()})
            batch["log/pose_weight"] = losses["total"].new_tensor(scale)
        else:
            batch["world_prediction"] = self.model.sample(
                batch["world"], embodiment, steps=self.sampler_steps
            )
        return batch
