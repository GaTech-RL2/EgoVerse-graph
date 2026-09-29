"""Matched diffusion decoders for ARC geometry and timing streams.

The external timed LIBERO codec remains M x 12. Only the denoiser internally
separates ten geometry/gripper channels from timing channels 3 and 10. All
variants have identical parameter counts, observation conditioning, temporal
causality, output channels, and diffusion losses.
"""

import torch
from torch import nn

from egomimic.models.oat.diffusion import GraphDiffusionTransformer

DECODER_VARIANTS = ("shared", "separate", "shape_masked")
SHAPE_COLUMNS = (0, 1, 2, 4, 5, 6, 7, 8, 9, 11)
TIMING_COLUMNS = (3, 10)


class ArcStreamDiffusionTransformer(GraphDiffusionTransformer):
    """Split input/output streams with an equal total decoder parameter budget.

    ``shared`` uses one N-layer decoder over 2M tokens. ``shape_masked`` uses
    the same network but blocks every shape-query/timing-key attention edge.
    Timing can still attend to shape at the same or earlier waypoint.
    ``separate`` partitions the N distinct decoder layers into two independent
    N/2-layer stacks, each operating on M tokens. Width and total parameter
    count are held fixed; per-stream depth is explicitly halved in this arm.

    Observation/time conditioning and token-local final normalization are
    shared in all variants. Neither path consumes actions through memory.
    """

    def __init__(self, *, decoder_variant, **kwargs):
        if decoder_variant not in DECODER_VARIANTS:
            raise ValueError(f"Unknown ARC decoder variant: {decoder_variant}")
        if kwargs.get("input_dim") != 12 or kwargs.get("output_dim") != 12:
            raise ValueError("ARC stream decoders require the timed 12-D codec")
        layers = int(kwargs.get("n_layer", 4))
        if layers < 2 or layers % 2:
            raise ValueError(
                "An equal decoder budget requires a positive even layer count"
            )
        kwargs["n_layer"] = layers
        if (
            not kwargs.get("time_as_cond", True)
            or not kwargs.get("obs_as_cond", True)
            or kwargs.get("cond_dim", 0) <= 0
            or not kwargs.get("causal_attn", True)
        ):
            raise ValueError(
                "ARC stream comparison requires causal observation/time conditioning"
            )
        super().__init__(**kwargs)
        self.decoder_variant = decoder_variant
        self.total_decoder_layers = layers
        width = self.input_emb.out_features

        # Construct common modules in the same order for every variant. The
        # already initialized N decoder layers are reused, so common seed and
        # parameter initialization are identical before the topology changes.
        self.input_emb = nn.ModuleDict(
            {
                "shape": nn.Linear(len(SHAPE_COLUMNS), width),
                "timing": nn.Linear(len(TIMING_COLUMNS), width),
            }
        )
        self.head = nn.ModuleDict(
            {
                "shape": nn.Linear(width, len(SHAPE_COLUMNS)),
                "timing": nn.Linear(width, len(TIMING_COLUMNS)),
            }
        )
        self.stream_emb = nn.Embedding(2, width)
        for module in (*self.input_emb.values(), *self.head.values(), self.stream_emb):
            self._init_weights(module)
        self.register_buffer(
            "shape_columns", torch.tensor(SHAPE_COLUMNS), persistent=False
        )
        self.register_buffer(
            "timing_columns", torch.tensor(TIMING_COLUMNS), persistent=False
        )
        self.register_buffer(
            "output_order",
            torch.argsort(torch.tensor(SHAPE_COLUMNS + TIMING_COLUMNS)),
            persistent=False,
        )

        waypoint = torch.arange(self.horizon).repeat(2)
        blocked = waypoint[None, :] > waypoint[:, None]
        if decoder_variant in ("shape_masked", "separate"):
            blocked[: self.horizon, self.horizon :] = True
        if decoder_variant == "separate":
            blocked[self.horizon :, : self.horizon] = True
        self.mask = torch.zeros(2 * self.horizon, 2 * self.horizon).masked_fill(
            blocked, float("-inf")
        )
        # Repeat the original observation-memory mask by waypoint, not by the
        # concatenated token index, so both streams see identical observations.
        self.memory_mask = self.memory_mask.repeat(2, 1)

        if decoder_variant == "separate":
            original_layers = list(self.decoder.layers)
            half = layers // 2
            self.shape_decoder = nn.TransformerDecoder(original_layers[0], half)
            self.timing_decoder = nn.TransformerDecoder(original_layers[half], half)
            self.shape_decoder.layers = nn.ModuleList(original_layers[:half])
            self.timing_decoder.layers = nn.ModuleList(original_layers[half:])
            self.decoder = None

    def forward(self, sample, timestep, global_cond):
        if sample.ndim != 3 or tuple(sample.shape[1:]) != (self.horizon, 12):
            raise ValueError(f"Expected ARC input (B, {self.horizon}, 12)")
        if tuple(global_cond.shape) != (sample.shape[0], self.global_cond_dim):
            raise ValueError("ARC stream observation condition has the wrong shape")
        times = torch.as_tensor(timestep, device=sample.device)
        if times.ndim == 0:
            times = times[None]
        times = times.expand(sample.shape[0])
        observations = global_cond.reshape(
            sample.shape[0], self.observation_steps, self.observation_dim
        )
        condition = torch.cat(
            [self.time_emb(times).unsqueeze(1), self.cond_obs_emb(observations)], dim=1
        )
        memory = self.encoder(self.drop(condition + self.cond_pos_emb))

        shape = self.input_emb["shape"](sample.index_select(-1, self.shape_columns))
        timing = self.input_emb["timing"](sample.index_select(-1, self.timing_columns))
        shape = shape + self.pos_emb + self.stream_emb.weight[0]
        timing = timing + self.pos_emb + self.stream_emb.weight[1]
        tokens = self.drop(torch.cat([shape, timing], dim=1))
        m = self.horizon
        if self.decoder_variant == "separate":
            shape = self.shape_decoder(
                tokens[:, :m],
                memory,
                tgt_mask=self.mask[:m, :m],
                memory_mask=self.memory_mask[:m],
            )
            timing = self.timing_decoder(
                tokens[:, m:],
                memory,
                tgt_mask=self.mask[m:, m:],
                memory_mask=self.memory_mask[m:],
            )
        else:
            decoded = self.decoder(
                tokens, memory, tgt_mask=self.mask, memory_mask=self.memory_mask
            )
            shape, timing = decoded[:, :m], decoded[:, m:]
        packed = torch.cat(
            [
                self.head["shape"](self.ln_f(shape)),
                self.head["timing"](self.ln_f(timing)),
            ],
            dim=-1,
        )
        return packed.index_select(-1, self.output_order)
