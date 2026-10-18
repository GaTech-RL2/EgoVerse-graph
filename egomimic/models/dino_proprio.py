"""Compatibility class for the original unlaunched DINO/proprio recipes.

The feature-agnostic implementation lives in :mod:`image_proprio`. Its default
width guard and identity embeddings retain the original model's defaults.
"""

from egomimic.models.image_proprio import MODALITIES, ImageProprioModel


class DinoProprioModel(ImageProprioModel):
    def __init__(
        self,
        *args,
        use_camera_embedding=True,
        use_modality_embedding=True,
        use_embodiment_embedding=True,
        **kwargs,
    ):
        super().__init__(
            *args,
            use_camera_embedding=use_camera_embedding,
            use_modality_embedding=use_modality_embedding,
            use_embodiment_embedding=use_embodiment_embedding,
            **kwargs,
        )


__all__ = ["MODALITIES", "DinoProprioModel"]
