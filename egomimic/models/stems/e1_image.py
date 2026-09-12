"""E1 image path: ImageNet normalization, ResNet, then the HPT MLP stem.

The campaign's encoder and stem were separate in the previous HPT runtime.
Keeping both here preserves that path while HPTStemStage owns latent pooling.
"""

from torchvision import transforms

from egomimic.models.stems.hpt_stems import MLPPolicyStem, PolicyStem, ResNet


class E1ImageStem(PolicyStem):
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
