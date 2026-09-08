"""Construct an exact-section graph with one shared private codec instance.

Hydra interpolation of a module configuration constructs independent modules;
it does not share their weights. This model-specific factory binds the encoder
and decoder explicitly without introducing model semantics into PipelineAlgo.
"""

from __future__ import annotations

import hydra
import torch
import torch.nn as nn

from egomimic.pipeline.algo import PipelineAlgo


class GraphSectionSequenceEncoder(nn.Module):
    def __init__(self, codec: nn.Module):
        super().__init__()
        # E owns only f; decoder-only R must not appear in E.parameters().
        self.graph = codec.graph

    def forward(self, content):
        return torch.cat((content, self.graph(content)), dim=-1)


def build_graph_section_pipeline(*, codec, stages, device=None):
    """Instantiate declared nodes, injecting the same codec at both boundaries."""
    shared = hydra.utils.instantiate(codec)
    bound_stages = []
    counts = {"encoder": 0, "decoder": 0}
    for config in stages:
        target = str(config.get("_target_", ""))
        if target == "egomimic.pipeline.stages_action_flow.ContentEncoderStage":
            if "encoder" in config:
                raise ValueError("graph encoder is injected; do not configure a second codec")
            stage = hydra.utils.instantiate(config, encoder=GraphSectionSequenceEncoder(shared))
            counts["encoder"] += 1
        elif target == "egomimic.pipeline.stages_action_flow.ContentDecoderStage":
            if "decoder" in config:
                raise ValueError("graph decoder is injected; do not configure a second codec")
            stage = hydra.utils.instantiate(config, decoder=shared)
            counts["decoder"] += 1
        else:
            stage = hydra.utils.instantiate(config)
        bound_stages.append(stage)
    if counts != {"encoder": 1, "decoder": 1}:
        raise ValueError(f"exact-section graph requires one encoder and decoder: {counts}")
    return PipelineAlgo(bound_stages, device=device)
