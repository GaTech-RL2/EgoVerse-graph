"""Arc-length tokenize / detokenize stages for configured Planar graphs.

The Planar arc tokenizer already exists as a dataset transform
(:class:`~egomimic.rldb.zarr.planar_arc.TokenizePlanarArcLength`), which runs
per sample inside the loader. That placement is invisible to the graph: a
config's stage list shows the model consuming ``target`` with no indication
that the target is an arc token rather than a time-indexed chunk, and
``tools/config_graph.py`` cannot lint the boundary because nothing declares it.

These two stages move that boundary INTO the graph, as nodes with contracts:

* :class:`ArcTokenizeStage` replaces ``ActionTargetBuilder`` -- it reads the
  loader's time-indexed action chunk and writes the arc token as ``target``.
  It is the only writer of ``target``, so the rest of a DP or flow graph is
  unchanged; it simply models arc tokens.
* :class:`ArcDetokenizeStage` is the inverse, reading a predicted arc token and
  writing back a time-indexed chunk. It is declared ``inference_only``, the
  mirror of ``train_only``: ``pred_action`` exists only in the inference graph,
  and this runner treats an unsatisfied read as a configuration error rather
  than a reason to skip, so the restriction is stated rather than inferred.

Both stages are deliberately Planar-specific and live behind explicit
contracts: the generic runner in ``pipeline/core.py`` still knows nothing about
what an action, an angle, or a waypoint is.
"""

from __future__ import annotations

import numpy as np
import torch

from egomimic.pipeline.core import Stage
from egomimic.rldb.zarr.planar_arc import (
    PLANAR_ACTION_DIM,
    TokenizePlanarArcLength,
)


def _as_batched(actions: torch.Tensor, label: str) -> torch.Tensor:
    if not torch.is_tensor(actions):
        raise TypeError(f"{label} must be a tensor, got {type(actions).__name__}")
    if actions.ndim != 3:
        raise ValueError(f"{label} must be (B, T, D), got {tuple(actions.shape)}")
    return actions


class ArcTokenizeStage(Stage):
    """Encode a time-indexed action chunk as an arc-length token.

    Writes ``target`` directly rather than a namespaced key, so this stage takes
    ``ActionTargetBuilder``'s place in a stage list instead of sitting after it.
    Two writers of ``target`` would be a duplicate-writer lint error, and the
    graph would be ambiguous about which one the denoiser is modelling.

    Tokenization runs per sample on numpy through the same
    ``TokenizePlanarArcLength`` the loader-side transform uses, so a graph-side
    and a dataset-side tokenizer configured alike produce identical targets.
    """

    train_only = True
    writes = ("target",)

    def __init__(
        self,
        action_key: str = "actions",
        min_distance_unit: float = 200.0,
        resampled_vector_length: int = 100,
        dt: float = 1.0 / 30.0,
        rotation_radius: float = 0.0,
        hybrid_rotation_unit: float | None = None,
    ):
        super().__init__()
        self.action_key = str(action_key)
        if not self.action_key:
            raise ValueError("action_key must be non-empty")
        self.reads = (self.action_key,)
        self.num_waypoints = int(resampled_vector_length)
        self.tokenizer = TokenizePlanarArcLength(
            action_key="actions",
            output_action_key="actions",
            min_distance_unit=min_distance_unit,
            resampled_vector_length=resampled_vector_length,
            dt=dt,
            rotation_radius=rotation_radius,
            hybrid_rotation_unit=hybrid_rotation_unit,
        )

    def forward(self, batch: dict) -> dict:
        actions = _as_batched(batch[self.action_key], "ArcTokenizeStage input")
        native = actions.detach().cpu().numpy().astype(np.float64, copy=False)
        tokens = np.stack(
            [self.tokenizer.tokenize(sample) for sample in native], axis=0
        )
        batch["target"] = torch.as_tensor(
            tokens, dtype=actions.dtype, device=actions.device
        )
        batch.pop(self.action_key, None)
        return batch


class ArcDetokenizeStage(Stage):
    """Decode a predicted arc token back to a time-indexed action chunk.

    The token is ``num_waypoints`` waypoints uniform in SE(2) arc length
    followed by one timing row whose first field is the chunk's mean arc speed.
    Reconstruction walks the waypoint polyline at ``speed * dt * k`` for each
    output step k, which is the inverse of how the tokenizer laid the waypoints
    out; a zero-speed token (a stationary or degenerate chunk) holds the first
    waypoint, matching the tokenizer's own degenerate branch.

    Heading is stored as ``(cos, sin)`` and interpolated in that form, then
    renormalised. Interpolating the angle directly would need unwrapping and
    would cross the +/-pi seam mid-chunk.
    """

    # Inference-only rather than "blocked in train by a missing read": a stage
    # whose reads are unavailable is a configuration error in this runner, so a
    # mode restriction has to be declared, not inferred.
    inference_only = True
    reads = ("pred_action",)
    writes = ("pred_action_native", "log/*")

    def __init__(
        self,
        resampled_vector_length: int = 100,
        action_horizon: int = 16,
        dt: float = 1.0 / 30.0,
        native_action_dim: int = 3,
        zero_dist_epsilon: float = 1e-9,
    ):
        super().__init__()
        self.num_waypoints = int(resampled_vector_length)
        self.action_horizon = int(action_horizon)
        self.dt = float(dt)
        self.native_action_dim = int(native_action_dim)
        self.zero_dist_epsilon = float(zero_dist_epsilon)
        if self.num_waypoints < 2:
            raise ValueError("resampled_vector_length must be at least two")
        if self.action_horizon <= 0 or self.dt <= 0:
            raise ValueError("action_horizon and dt must be positive")
        if self.native_action_dim not in (2, 3, 4):
            raise ValueError("native_action_dim must be 2, 3, or 4")

    def _arc_positions(self, waypoints: torch.Tensor) -> torch.Tensor:
        """Cumulative translational arc length along each batch's polyline."""
        steps = torch.linalg.vector_norm(
            waypoints[:, 1:, :2] - waypoints[:, :-1, :2], dim=-1
        )
        zero = torch.zeros_like(steps[:, :1])
        return torch.cat((zero, torch.cumsum(steps, dim=1)), dim=1)

    def forward(self, batch: dict) -> dict:
        tokens = _as_batched(batch["pred_action"], "ArcDetokenizeStage input")
        expected = (self.num_waypoints + 1, PLANAR_ACTION_DIM)
        if tuple(tokens.shape[1:]) != expected:
            raise ValueError(
                f"ArcDetokenizeStage expects (B, {expected[0]}, {expected[1]}), "
                f"got {tuple(tokens.shape)}"
            )

        waypoints = tokens[:, : self.num_waypoints]
        speed = tokens[:, self.num_waypoints, 0].clamp_min(0.0)
        cumulative = self._arc_positions(waypoints)
        total = cumulative[:, -1]

        # Where along the polyline each output step lands. A zero-speed token
        # stays at 0 and so replays the first waypoint for the whole horizon.
        steps = torch.arange(
            self.action_horizon, device=tokens.device, dtype=tokens.dtype
        )
        targets = torch.minimum(
            speed[:, None] * self.dt * steps[None, :], total[:, None]
        )

        # Bracket each target between the two waypoints it falls between and
        # interpolate. searchsorted needs a contiguous, increasing key.
        upper = torch.searchsorted(cumulative.contiguous(), targets.contiguous())
        upper = upper.clamp(1, self.num_waypoints - 1)
        lower = upper - 1
        s_lo = torch.gather(cumulative, 1, lower)
        s_hi = torch.gather(cumulative, 1, upper)
        span = (s_hi - s_lo).clamp_min(self.zero_dist_epsilon)
        alpha = ((targets - s_lo) / span).clamp(0.0, 1.0).unsqueeze(-1)

        index_lo = lower.unsqueeze(-1).expand(-1, -1, PLANAR_ACTION_DIM)
        index_hi = upper.unsqueeze(-1).expand(-1, -1, PLANAR_ACTION_DIM)
        decoded = (1.0 - alpha) * torch.gather(
            waypoints, 1, index_lo
        ) + alpha * torch.gather(waypoints, 1, index_hi)

        heading = decoded[..., 2:4]
        norm = torch.linalg.vector_norm(heading, dim=-1, keepdim=True)
        heading = heading / norm.clamp_min(1e-8)
        theta = torch.atan2(heading[..., 1], heading[..., 0]).unsqueeze(-1)
        native = torch.cat((decoded[..., :2], theta, decoded[..., 4:5]), dim=-1)

        batch["pred_action_native"] = native[..., : self.native_action_dim]
        batch["log/ArcSpeed"] = speed.mean()
        batch["log/ArcChunkDistance"] = total.mean()
        return batch
