"""ABC-DiT observation stage: cameras, proprioception and task to condition tokens.

Writes ``condition`` as a ``(B, tokens, H)`` block, so the stock
``FlowNoisingStage`` / ``FlowDenoiserStage`` / ``FlowVelocityLossStage`` trio
trains ABC-DiT unchanged when the denoiser model is
``egomimic.models.abc_dit.ABCDiT`` and ``condition_as_tokens`` is set. Because
the ARC representation only changes ``target`` and ``action_horizon``, the same
graph trains both the time-indexed baseline and the hybrid ARC tokens.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F

from egomimic.models.abc_dit import IMAGENET_MEAN, IMAGENET_STD, ABCVisionStateEncoder
from egomimic.pipeline.core import Stage


def resize_pad_imagenet(images: torch.Tensor, size: int = 224) -> torch.Tensor:
    """Aspect-preserving resize + zero pad to ``size``, then ImageNet-normalize.

    Matches upstream ``resize_pad_normalize_batch`` for [0, 1] float input.
    """
    _, _, h, w = images.shape
    if (h, w) != (size, size):
        ratio = max(w / size, h / size)
        new_h, new_w = max(1, round(h / ratio)), max(1, round(w / ratio))
        images = F.interpolate(
            images,
            size=(new_h, new_w),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        top, left = (size - new_h) // 2, (size - new_w) // 2
        images = F.pad(images, (left, size - new_w - left, top, size - new_h - top))
    mean = images.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = images.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (images - mean) / (std + 1e-6)


class ABCObservationStage(Stage):
    """Encode camera keys and the state key into ABC-DiT condition tokens.

    ``mask_state_ratio`` is upstream's proprioception dropout: in training, that
    fraction of samples has its normalized state zeroed before embedding.
    ``task_key`` holds one prompt string per sample (the episode's
    ``task_description``), which the encoder embeds with frozen CLIP text.
    """

    writes = ("condition",)

    def __init__(
        self,
        encoder: ABCVisionStateEncoder,
        camera_keys: Sequence[str],
        state_key: str = "observations.state.ee_pose",
        task_key: str = "task",
        image_size: int = 224,
        mask_state_ratio: float = 0.1,
    ):
        super().__init__()
        self.encoder = encoder
        self.camera_keys = tuple(str(key) for key in camera_keys)
        if len(self.camera_keys) != len(encoder.apool):
            raise ValueError(
                f"{len(self.camera_keys)} camera keys but the encoder pools "
                f"{len(encoder.apool)} cameras"
            )
        self.state_key = str(state_key)
        self.image_size = int(image_size)
        self.mask_state_ratio = float(mask_state_ratio)
        if not 0.0 <= self.mask_state_ratio <= 1.0:
            raise ValueError("mask_state_ratio must be in [0, 1]")
        self.task_key = str(task_key)
        self.reads = (*self.camera_keys, self.state_key, self.task_key)

    def execute(self, batch: dict, *, mode: str) -> dict:
        state = batch[self.state_key].float()
        if mode == "train" and self.mask_state_ratio > 0:
            keep = (
                torch.rand(state.shape[0], 1, device=state.device)
                >= self.mask_state_ratio
            )
            state = state * keep
        images = []
        for key in self.camera_keys:
            image = batch[key]
            # Loader images are [0, 1] floats, possibly with singleton
            # time/view axes; the encoder takes the latest (B, 3, H, W) frame.
            image = image.reshape(len(image), -1, *image.shape[-3:])[:, -1]
            images.append(resize_pad_imagenet(image.float(), self.image_size))
        tasks = batch[self.task_key]
        tasks = [tasks] * len(state) if isinstance(tasks, str) else list(tasks)
        if len(tasks) != len(state) or not all(isinstance(t, str) for t in tasks):
            raise ValueError(f"{self.task_key} must be one prompt string per sample")
        batch["condition"] = self.encoder(images, state, tasks)
        return batch

    def forward(self, batch: dict) -> dict:
        return self.execute(batch, mode="train")
