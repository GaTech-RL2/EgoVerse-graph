"""Explicit encoder, augmentation and latent-pooling composition."""

import torch
from torch import nn

from egomimic.models.cores.hpt_utils import get_sinusoid_encoding_table


class EncodedPolicyStem(nn.Module):
    """Preserve recipes with a separate feature encoder before their MLP stem."""

    def __init__(
        self,
        stem,
        encoder=None,
        train_transform=None,
        eval_transform=None,
        feature_position_encoding=True,
    ):
        super().__init__()
        self.stem, self.encoder = stem, encoder
        self.train_transform, self.eval_transform = train_transform, eval_transform
        self.feature_position_encoding = feature_position_encoding
        self.specs = stem.specs

    def init_cross_attn(self, spec):
        self.stem.init_cross_attn(spec)

    def compute_latent(self, value):
        transform = self.train_transform if self.training else self.eval_transform
        if transform is not None:
            value = transform(value)
        if self.encoder is not None:
            value = self.encoder(value)
        if torch.is_tensor(value) and self.feature_position_encoding:
            if value.ndim < 2:
                raise ValueError("EncodedPolicyStem requires a batched feature tensor")
            shape = value.shape
            flat = value.reshape(shape[0], -1, shape[-1])
            position = get_sinusoid_encoding_table(0, flat.shape[1], flat.shape[2]).to(
                flat
            )
            value = (flat + position).reshape(shape)
        return self.stem.compute_latent(value)
