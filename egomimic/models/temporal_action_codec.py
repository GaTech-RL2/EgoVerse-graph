"""Context-free temporal U-Net interfaces for the shared Action Flow latent.

Skip connections stay inside each interface. No encoder activations reach the
decoder: every generated or reconstructed action must pass through the same
compact latent boundary. This is an architectural bias, not a smoothing filter.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from egomimic.models.denoising_nets import Conv1dBlock, Downsample1d


class _ResidualBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, groups):
        super().__init__()
        self.layers = nn.Sequential(
            Conv1dBlock(input_channels, output_channels, kernel_size, groups),
            Conv1dBlock(output_channels, output_channels, kernel_size, groups),
        )
        self.skip = (nn.Identity() if input_channels == output_channels
                     else nn.Conv1d(input_channels, output_channels, 1))

    def forward(self, value):
        return self.layers(value) + self.skip(value)


class TemporalUNet1D(nn.Module):
    """Unconditioned U-Net using the maintained temporal convolution blocks.

Exact-length linear interpolation followed by convolutions avoids transposed
convolution length ambiguity for odd lengths such as 100 -> 50 -> 25.
"""

    def __init__(self, input_dim, output_dim, channels, kernel_size=3, groups=8):
        super().__init__()
        channels = tuple(int(c) for c in channels)
        if (len(channels) < 2 or min(channels) <= 0 or
                any(c % groups for c in channels) or kernel_size % 2 != 1):
            raise ValueError("Need positive group-divisible channels and odd kernel")
        self.channels = channels
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        previous = int(input_dim)
        for index, width in enumerate(channels):
            self.down_blocks.append(nn.Sequential(
                _ResidualBlock(previous, width, kernel_size, groups),
                _ResidualBlock(width, width, kernel_size, groups),
            ))
            if index + 1 < len(channels):
                self.downsamples.append(Downsample1d(width))
            previous = width
        self.middle = nn.Sequential(
            _ResidualBlock(channels[-1], channels[-1], kernel_size, groups),
            _ResidualBlock(channels[-1], channels[-1], kernel_size, groups),
        )
        self.up_blocks = nn.ModuleList()
        current = channels[-1]
        for width in reversed(channels[:-1]):
            self.up_blocks.append(nn.Sequential(
                _ResidualBlock(current + width, width, kernel_size, groups),
                _ResidualBlock(width, width, kernel_size, groups),
            ))
            current = width
        self.output = nn.Conv1d(channels[0], int(output_dim), 1)
        self.gradient_checkpointing = True

    def _call(self, module, value):
        if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint(module, value, use_reentrant=False)
        return module(value)

    def forward(self, value):
        skips = []
        for index, block in enumerate(self.down_blocks):
            value = self._call(block, value)
            skips.append(value)
            if index < len(self.downsamples):
                value = self.downsamples[index](value)
        value = self._call(self.middle, value)
        for block, skip in zip(self.up_blocks, reversed(skips[:-1])):
            value = F.interpolate(value, size=skip.shape[-1], mode="linear",
                                  align_corners=False)
            value = self._call(block, torch.cat((value, skip), dim=1))
        return self.output(value)


class _TemporalInterface(nn.Module):
    @property
    def gradient_checkpointing(self):
        return self.backbone.gradient_checkpointing

    @gradient_checkpointing.setter
    def gradient_checkpointing(self, value):
        self.backbone.gradient_checkpointing = bool(value)

    @staticmethod
    def _check(value, length, width):
        if value.ndim != 3 or tuple(value.shape[1:]) != (length, width):
            raise ValueError(f"Expected (B, {length}, {width}), got {tuple(value.shape)}")


class TemporalActionEncoder(_TemporalInterface):
    """Deterministic action-only encoder: (B,H,A) -> (B,N,D)."""

    def __init__(self, action_dim, action_horizon, latent_dim, num_latent_tokens,
                 channels, kernel_size=3, groups=8, gradient_checkpointing=True):
        super().__init__()
        self.input_dim = int(action_dim)
        self.action_horizon = int(action_horizon)
        self.latent_dim = int(latent_dim)
        self.num_latent_tokens = int(num_latent_tokens)
        if min(self.input_dim, self.action_horizon, self.latent_dim,
               self.num_latent_tokens) <= 0:
            raise ValueError("Action/latent dimensions must be positive")
        self.backbone = TemporalUNet1D(self.input_dim, self.latent_dim, channels,
                                      kernel_size, groups)
        self.output_norm = nn.LayerNorm(self.latent_dim)
        self.gradient_checkpointing = gradient_checkpointing

    @property
    def blocks(self):
        # Two actual equal-width forward activations, not synthetic projections.
        return nn.ModuleList((self.backbone.down_blocks[0],
                              self.backbone.up_blocks[-1]))

    def canonicalize_diagnostic_block_output(self, output, block_index):
        return F.interpolate(output, size=self.num_latent_tokens, mode="linear",
                             align_corners=False).transpose(1, 2)

    def forward(self, content):
        self._check(content, self.action_horizon, self.input_dim)
        encoded = self.backbone(content.transpose(1, 2))
        encoded = F.interpolate(encoded, size=self.num_latent_tokens,
                                mode="linear", align_corners=False)
        return self.output_norm(encoded.transpose(1, 2))


class TemporalActionDecoder(_TemporalInterface):
    """Latent-only action decoder: (B,N,D) -> (B,H,A)."""

    def __init__(self, action_dim, action_horizon, latent_dim, num_latent_tokens,
                 channels, kernel_size=3, groups=8, gradient_checkpointing=True):
        super().__init__()
        self.action_dim = int(action_dim)
        self.action_horizon = int(action_horizon)
        self.latent_dim = int(latent_dim)
        self.num_latent_tokens = int(num_latent_tokens)
        if min(self.action_dim, self.action_horizon, self.latent_dim,
               self.num_latent_tokens) <= 0:
            raise ValueError("Action/latent dimensions must be positive")
        self.backbone = TemporalUNet1D(self.latent_dim, self.action_dim, channels,
                                      kernel_size, groups)
        self.gradient_checkpointing = gradient_checkpointing

    def forward(self, latent):
        self._check(latent, self.num_latent_tokens, self.latent_dim)
        value = F.interpolate(latent.transpose(1, 2), size=self.action_horizon,
                              mode="linear", align_corners=False)
        return self.backbone(value).transpose(1, 2)
