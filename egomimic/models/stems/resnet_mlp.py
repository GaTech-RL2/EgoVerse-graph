"""ImageNet-normalized ResNet features followed by an HPT MLP stem.

Color jitter is applied only during training. Spatial tokens are preserved
for latent pooling in HPTStemStage.
"""

from torchvision import transforms

from egomimic.models.stems.hpt_stems import MLPPolicyStem, PolicyStem, ResNet


class ResNetMLPImageStem(PolicyStem):
    def __init__(self, output_dim=256, weights="DEFAULT", **kwargs):
        super().__init__(**kwargs)
        self.encoder = ResNet(output_dim=output_dim, weights=weights)
        self.mlp = MLPPolicyStem(
            input_dim=output_dim, output_dim=output_dim, widths=[output_dim]
        )
        self.jitter = transforms.ColorJitter(0.1, 0.1, 0.1, 0.05)
        self.normalize = transforms.Normalize(
            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        )

    def forward(self, images):
        if self.training:
            images = self.jitter(images)
        return self.mlp(self.encoder(self.normalize(images)))
