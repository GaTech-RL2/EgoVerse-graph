"""Two-modality visual-state and direct-action experiments."""

from egomimic.pipeline.core import Stage, resolve_homogeneous_scalar


class ImageProprioStage(Stage):
    loss_key = "loss/image_proprio"
    prediction_key = "image_proprio_prediction"
    log_prefix = "image_proprio"
    reads_by_mode = {
        "train": ("world", "embodiment", "optimizer_step"),
        "inference": ("world", "embodiment"),
    }
    writes_by_mode = {
        "train": ("loss/image_proprio",),
        "inference": ("image_proprio_prediction",),
    }

    def __init__(
        self,
        model,
        warmup_steps=100,
        pose_ramp_steps=100,
        image_weight=1.0,
        pose_weight=1.0,
        sampler_steps=20,
        action_weight=1.0,
    ):
        super().__init__()
        if (
            min(warmup_steps, pose_ramp_steps, image_weight, pose_weight, action_weight)
            < 0
        ):
            raise ValueError("Schedules and loss weights must be nonnegative")
        if type(sampler_steps) is not int or sampler_steps < 1:
            raise ValueError("sampler_steps must be a positive integer")
        self.model = model
        self.warmup_steps, self.pose_ramp_steps = warmup_steps, pose_ramp_steps
        self.image_weight, self.pose_weight = image_weight, pose_weight
        self.action_weight = action_weight
        self.sampler_steps = sampler_steps

    def pose_scale(self, step):
        return self.pose_weight * self.readout_ramp(step)

    def readout_ramp(self, step):
        if step < self.warmup_steps:
            return 0.0
        if self.pose_ramp_steps == 0:
            return 1.0
        return min(1.0, (step - self.warmup_steps) / self.pose_ramp_steps)

    def execute(self, batch, *, mode):
        embodiment = str(
            resolve_homogeneous_scalar(batch["embodiment"], label="embodiment")
        )
        if mode == "train":
            scale = self.pose_scale(int(batch["optimizer_step"]))
            action_scale = self.action_weight * self.readout_ramp(
                int(batch["optimizer_step"])
            )
            losses = self.model.training_losses(
                batch["world"],
                embodiment,
                pose_weight=scale,
                image_weight=self.image_weight,
                action_weight=action_scale,
            )
            batch[self.loss_key] = losses["total"]
            batch.update(
                {f"log/{self.log_prefix}_{k}": v.detach() for k, v in losses.items()}
            )
            image = batch["world"]["image"]
            batch[f"log/{self.log_prefix}_image_tokens"] = image["valid"].sum().float()
            batch[f"log/{self.log_prefix}_image_query_tokens"] = (
                (image["valid"] & ~image["observed"]).sum().float()
            )
            if self.model.objective in ("visual_proprio", "visual_state_action"):
                batch["log/pose_weight"] = losses["total"].new_tensor(scale)
            if self.model.objective == "visual_state_action":
                batch["log/action_weight"] = losses["total"].new_tensor(action_scale)
        else:
            batch[self.prediction_key] = self.model.sample(
                batch["world"], embodiment, steps=self.sampler_steps
            )
        return batch
