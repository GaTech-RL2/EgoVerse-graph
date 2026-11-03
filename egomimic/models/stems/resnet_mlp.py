"""ImageNet-normalized ResNet features followed by an HPT MLP stem.

Spatial augmentation (opt-in) and color jitter are applied only during training.
Spatial tokens are preserved for latent pooling in HPTStemStage.
"""

import torch
import torch.nn.functional as F
from torchvision import transforms

from egomimic.models.stems.hpt_stems import MLPPolicyStem, PolicyStem, ResNet


def random_rotate_translate_crop(
    images, max_rotation_deg=0.0, max_translate=0.0, min_crop=1.0
):
    """Resample each image with its own small rotation, shift and zoom-in crop.

    ``max_translate`` is a fraction of the image size and ``min_crop`` the
    smallest kept fraction of each side. Parameters are drawn per image, because
    torchvision's transforms draw one set per call, i.e. one per batch. Pixels
    from outside the frame mirror the image, and bicubic sampling keeps it sharp.
    """
    *batch, C, H, W = images.shape
    x = images.reshape(-1, C, H, W)

    def uniform(low, high):
        return x.new_empty(len(x)).uniform_(low, high)

    angle = torch.deg2rad(uniform(-max_rotation_deg, max_rotation_deg))
    crop = uniform(min_crop, 1.0)
    cos, sin = crop * angle.cos(), crop * angle.sin()
    # affine_grid coordinates span [-1, 1] on both axes, so H/W keeps the
    # rotation rigid on a non-square image and a shift of 2t moves t of the image.
    theta = torch.stack(
        [
            torch.stack(
                [cos, -sin * H / W, uniform(-max_translate, max_translate) * 2], -1
            ),
            torch.stack(
                [sin * W / H, cos, uniform(-max_translate, max_translate) * 2], -1
            ),
        ],
        1,
    )
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(
        x, grid, mode="bicubic", padding_mode="reflection", align_corners=False
    )
    return x.clamp(0.0, 1.0).reshape(*batch, C, H, W)


class ResNetMLPImageStem(PolicyStem):
    def __init__(
        self,
        output_dim=256,
        weights="DEFAULT",
        max_rotation_deg=0.0,
        max_translate=0.0,
        min_crop=1.0,
        jitter_hue=0.05,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.encoder = ResNet(output_dim=output_dim, weights=weights)
        self.mlp = MLPPolicyStem(
            input_dim=output_dim, output_dim=output_dim, widths=[output_dim]
        )
        self.spatial_aug = {
            "max_rotation_deg": max_rotation_deg,
            "max_translate": max_translate,
            "min_crop": min_crop,
        }
        self.use_spatial_aug = max_rotation_deg > 0 or max_translate > 0 or min_crop < 1
        self.jitter = transforms.ColorJitter(0.1, 0.1, 0.1, jitter_hue)
        self.normalize = transforms.Normalize(
            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        )

    def forward(self, images):
        if self.training:
            if self.use_spatial_aug:
                images = random_rotate_translate_crop(images, **self.spatial_aug)
            images = self.jitter(images)
        return self.mlp(self.encoder(self.normalize(images)))
