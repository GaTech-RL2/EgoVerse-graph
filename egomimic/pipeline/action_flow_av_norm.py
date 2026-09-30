"""Opt-in differentiable LayerNorm for Action Flow decoder JVPs only.

Ordinary reconstruction and inference keep PyTorch's native LayerNorm.  The
explicit formula is needed for the higher-order AV backward used by the
current U-Socket candidate; routed co-training applies it only to the decoder
selected by the homogeneous embodiment batch.
"""

from contextlib import contextmanager
import importlib
from types import MethodType

import torch
from lightning.pytorch.callbacks import Callback

from egomimic.pipeline.stages_action_flow import (
    ContentDecoderStage,
    RoutedContentDecoderStage,
)


def differentiable_layer_norm(self, x):
    if x.dtype not in (torch.float32, torch.float64):
        raise AssertionError("AV JVP must run outside autocast in FP32/FP64")
    dims = tuple(range(x.ndim - len(self.normalized_shape), x.ndim))
    centered = x - x.mean(dim=dims, keepdim=True)
    y = centered * torch.rsqrt(centered.square().mean(dim=dims, keepdim=True) + self.eps)
    if self.weight is not None:
        y = y * self.weight
    if self.bias is not None:
        y = y + self.bias
    return y


@contextmanager
def equivalent_norms(decoder, expected_count):
    missing = object()
    prior = []
    try:
        for module in decoder.modules():
            if isinstance(module, torch.nn.LayerNorm):
                prior.append((module, module.__dict__.get("forward", missing)))
                module.forward = MethodType(differentiable_layer_norm, module)
        if len(prior) != expected_count:
            raise AssertionError((len(prior), expected_count))
        yield
    finally:
        for module, previous in reversed(prior):
            if previous is missing:
                module.__dict__.pop("forward", None)
            else:
                module.forward = previous


def install_av_only(stage, *, expected_norms_per_decoder=25):
    if type(stage) not in (ContentDecoderStage, RoutedContentDecoderStage):
        raise TypeError("AV correction requires an Action Flow content decoder stage")
    if getattr(stage, "_av_norm_fix_installed", False):
        raise AssertionError("AV correction is already installed")
    if expected_norms_per_decoder <= 0:
        raise ValueError("expected_norms_per_decoder must be positive")
    if type(stage) is RoutedContentDecoderStage:
        decoders = tuple(stage.decoder.values())
    else:
        decoders = (stage.decoder,)
    if any(
        sum(isinstance(m, torch.nn.LayerNorm) for m in decoder.modules())
        != expected_norms_per_decoder
        for decoder in decoders
    ):
        raise AssertionError("unexpected decoder LayerNorm count")

    module = importlib.import_module(type(stage).__module__)
    original_forward = stage._forward_train
    original_jvp = module.jvp

    def forward(self, batch):
        if module.jvp is not original_jvp:
            raise AssertionError("unexpected/nested JVP dispatcher")
        decoder = self._decoder_for(batch)
        calls = []

        def dispatch(fn, primals, tangents, **kwargs):
            if fn is not decoder:
                raise AssertionError("unexpected JVP target or route")
            with equivalent_norms(decoder, expected_norms_per_decoder):
                answer = original_jvp(fn, primals, tangents, **kwargs)
            calls.append(True)
            return answer

        module.jvp = dispatch
        try:
            answer = original_forward(batch)
            if len(calls) != 1:
                raise AssertionError("AV correction did not cover exactly one JVP")
            return answer
        finally:
            module.jvp = original_jvp

    stage._forward_train = MethodType(forward, stage)
    stage._av_norm_fix_installed = True
    return stage


class AVOnlyLayerNormCallback(Callback):
    """Install the opt-in correction before the first Lightning training step."""

    def __init__(self, expected_norms_per_decoder=25, require_routed=True):
        super().__init__()
        self.expected_norms_per_decoder = int(expected_norms_per_decoder)
        self.require_routed = bool(require_routed)

    def setup(self, trainer, pl_module, stage):
        if stage != "fit":
            return
        stages = pl_module.model.pipeline.stages
        decoders = [
            item for item in stages
            if type(item) in (ContentDecoderStage, RoutedContentDecoderStage)
        ]
        if len(decoders) != 1:
            raise AssertionError("expected exactly one Action Flow decoder stage")
        decoder = decoders[0]
        if self.require_routed and type(decoder) is not RoutedContentDecoderStage:
            raise AssertionError("co-training requires two routed private decoders")
        if decoder.decoded_residual_key != "action_flow/decoded_velocity_residual":
            raise AssertionError("unexpected AV residual key")
        install_av_only(
            decoder,
            expected_norms_per_decoder=self.expected_norms_per_decoder,
        )
