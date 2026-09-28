"""Common action-reconstruction metrics for native OAT and ARC graphs."""

import torch

from egomimic.eval.eval import Eval
from egomimic.rldb.zarr.libero_dataset import EMBODIMENT


def action_metrics(prediction, target):
    if (
        prediction.shape != target.shape
        or prediction.ndim != 3
        or prediction.shape[-1] != 7
    ):
        raise ValueError("Expected matching (B,T,7) action chunks")
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise ValueError("Non-finite benchmark action")
    error = prediction - target
    return {
        "reconst_mse": error.square().mean(),
        "reconst_mae": error.abs().mean(),
        "translation_mse": error[..., :3].square().mean(),
        "rotation_command_mse": error[..., 3:6].square().mean(),
        "gripper_mse": error[..., 6].square().mean(),
        "gripper_sign_accuracy": ((prediction[..., 6] > 0) == (target[..., 6] > 0))
        .float()
        .mean(),
        "command_endpoint_error_m": (error[..., :3].sum(dim=1) * 0.05)
        .norm(dim=-1)
        .mean(),
    }


class LiberoActionEvaluator(Eval):
    def __init__(self):
        self.group = "valid"
        self.override_dict = {"limit_val_batches": 1.0}

    def set_validation_group(self, group):
        self.group = group

    def bind_data_context(self, *, normalizer):
        self.normalizer = normalizer

    def on_validation_start(self):
        pass

    def on_validation_step(self, batch, batch_idx, dataloader_idx=0):
        del batch_idx, dataloader_idx
        predictions = self.model.forward_eval(batch)
        for source, values in batch.items():
            target = self.normalizer.unnormalize(
                {"actions": values["actions"]}, EMBODIMENT
            )["actions"]
            predicted = self.normalizer.unnormalize(
                {"actions": predictions[source]["pred_action"]}, EMBODIMENT
            )["actions"]
            prefix = "Valid" if self.group == "valid" else f"Valid_{self.group}"
            for key, value in action_metrics(predicted, target).items():
                self.model.log(
                    f"{prefix}/{key}",
                    value,
                    batch_size=len(target),
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )

    def on_validation_end(self):
        pass
