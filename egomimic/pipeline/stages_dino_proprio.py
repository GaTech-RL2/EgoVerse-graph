"""Preserve the original DINO recipe's graph keys with the generic stage."""

from egomimic.pipeline.stages_image_proprio import ImageProprioStage


class DinoProprioStage(ImageProprioStage):
    loss_key = "loss/dino_proprio"
    prediction_key = "dino_proprio_prediction"
    log_prefix = "dino_proprio"
    writes_by_mode = {
        "train": ("loss/dino_proprio",),
        "inference": ("dino_proprio_prediction",),
    }
