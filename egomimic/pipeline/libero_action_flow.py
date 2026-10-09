"""Explicit LIBERO-10 observation boundary for the shared Action Flow graph.

The LIBERO replay keeps HWC images and separate robot state keys. This stage
converts one observation time into the CHW images and 64-D proprio condition
consumed by the same H384 Action Flow generator used for PushShapes. It never
changes action targets or creates a separate generative model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from egomimic.pipeline.core import Stage


class LiberoActionFlowObservationAdapter(Stage):
    reads = (
        "agentview_rgb",
        "robot0_eye_in_hand_rgb",
        "robot0_eef_pos",
        "robot0_eef_quat",
        "robot0_gripper_qpos",
        "task_uid",
    )
    writes = ("front_img_1", "front_img_2", "proprio_condition")

    def __init__(self, image_size=96, task_vocab_size=256, task_embed_dim=16):
        super().__init__()
        self.image_size = int(image_size)
        self.task_vocab_size = int(task_vocab_size)
        self.task_embed_dim = int(task_embed_dim)
        if min(self.image_size, self.task_vocab_size, self.task_embed_dim) <= 0:
            raise ValueError("adapter dimensions must be positive")
        self.task_embedding = nn.Embedding(self.task_vocab_size, self.task_embed_dim)
        self.proprio_projection = nn.Sequential(
            nn.Linear(9 + self.task_embed_dim, 64), nn.SiLU(), nn.Linear(64, 64)
        )

    def forward(self, batch):
        def latest(key, width):
            value = batch[key]
            if value.ndim != 3 or value.shape[-1] != width:
                raise ValueError(f"{key} must be [B,T,{width}]")
            return value[:, -1].float()

        state = torch.cat(
            (
                latest("robot0_eef_pos", 3),
                latest("robot0_eef_quat", 4),
                latest("robot0_gripper_qpos", 2),
            ),
            dim=-1,
        )
        task = batch["task_uid"]
        if task.ndim == 3 and task.shape[-1] == 1:
            task = task[..., 0]
        if task.ndim != 2 or task.shape[0] != state.shape[0]:
            raise ValueError("task_uid must be [B,T] or [B,T,1]")
        uid = task[:, -1]
        if not torch.isfinite(uid).all() or not torch.equal(uid, uid.round()):
            raise ValueError("task_uid must contain unnormalized integer IDs")
        uid = uid.long()
        if bool((uid < 0).any()) or bool((uid >= self.task_vocab_size).any()):
            raise ValueError("task_uid is outside embedding vocabulary")
        batch["proprio_condition"] = self.proprio_projection(
            torch.cat((state, self.task_embedding(uid)), dim=-1)
        )
        for source, dest in (
            ("agentview_rgb", "front_img_1"),
            ("robot0_eye_in_hand_rgb", "front_img_2"),
        ):
            image = batch[source]
            if image.ndim != 5 or image.shape[-1] != 3:
                raise ValueError(f"{source} must be [B,T,H,W,3]")
            image = image[:, -1].permute(0, 3, 1, 2).float()
            if image.shape[-2:] != (self.image_size, self.image_size):
                image = F.interpolate(
                    image,
                    size=(self.image_size, self.image_size),
                    mode="bilinear",
                    align_corners=False,
                )
            batch[dest] = image.contiguous()
        return batch
