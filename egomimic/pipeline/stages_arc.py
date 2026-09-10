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

import math

import numpy as np
import torch

from egomimic.pipeline.core import Stage
from egomimic.rldb.zarr.planar_arc import (
    PLANAR_ACTION_DIM,
    TokenizePlanarArcLength,
    arc_token_rows,
    lambda_for_radius,
    validate_velocity_mode,
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
        velocity_mode: str = "mean",
    ):
        super().__init__()
        self.action_key = str(action_key)
        if not self.action_key:
            raise ValueError("action_key must be non-empty")
        self.reads = (self.action_key,)
        self.num_waypoints = int(resampled_vector_length)
        self.velocity_mode = validate_velocity_mode(velocity_mode)
        self.tokenizer = TokenizePlanarArcLength(
            action_key="actions",
            output_action_key="actions",
            min_distance_unit=min_distance_unit,
            resampled_vector_length=resampled_vector_length,
            dt=dt,
            rotation_radius=rotation_radius,
            hybrid_rotation_unit=hybrid_rotation_unit,
            velocity_mode=self.velocity_mode,
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

    ``rotation_radius`` MUST match the tokenizer's. The token's speed is a rate
    in the tokenizer's SE(2) metric -- translation plus ``lambda * rotation``,
    with lambda from :func:`lambda_for_radius` -- so the polyline this walks has
    to be measured in that same metric. Accumulating translation alone against
    an SE(2) rate traverses the window too fast by exactly the ratio between the
    two lengths, which for a rotating path is unbounded: a 60-unit translation
    carrying 2.5 rad at radius 30 measures 135 in SE(2), so the reconstruction
    runs 2.25x fast and saturates at a third of the horizon. Translation-only is
    correct only at ``rotation_radius=0``, where lambda is 0 and the two metrics
    coincide.

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
        rotation_radius: float = 0.0,
        velocity_mode: str = "mean",
        zero_dist_epsilon: float = 1e-9,
    ):
        super().__init__()
        self.velocity_mode = validate_velocity_mode(velocity_mode)
        self.num_waypoints = int(resampled_vector_length)
        self.action_horizon = int(action_horizon)
        self.dt = float(dt)
        self.native_action_dim = int(native_action_dim)
        self.rotation_radius = float(rotation_radius)
        if self.rotation_radius < 0:
            raise ValueError("rotation_radius must be non-negative")
        self.lambda_rot = lambda_for_radius(self.rotation_radius)
        self.zero_dist_epsilon = float(zero_dist_epsilon)
        if self.num_waypoints < 2:
            raise ValueError("resampled_vector_length must be at least two")
        if self.action_horizon <= 0 or self.dt <= 0:
            raise ValueError("action_horizon and dt must be positive")
        if self.native_action_dim not in (2, 3, 4):
            raise ValueError("native_action_dim must be 2, 3, or 4")

    def _arc_positions(self, waypoints: torch.Tensor) -> torch.Tensor:
        """Cumulative SE(2) arc length, in the tokenizer's metric.

        Mirrors ``planar_step_distance``: translation plus ``lambda`` times the
        scaled chordal rotation distance. The angular step is taken as the
        principal difference via atan2 rather than by unwrapping a decoded
        angle -- the waypoints are a dense resample so consecutive steps are
        small, and this cannot be tripped by the +/-pi seam.
        """
        steps = torch.linalg.vector_norm(
            waypoints[:, 1:, :2] - waypoints[:, :-1, :2], dim=-1
        )
        if self.lambda_rot:
            cos, sin = waypoints[..., 2], waypoints[..., 3]
            # cos/sin of the angle between consecutive headings.
            delta_cos = cos[:, 1:] * cos[:, :-1] + sin[:, 1:] * sin[:, :-1]
            delta_sin = sin[:, 1:] * cos[:, :-1] - cos[:, 1:] * sin[:, :-1]
            delta = torch.atan2(delta_sin, delta_cos).abs()
            steps = steps + self.lambda_rot * math.sqrt(2.0) * torch.sin(delta / 4.0)
        zero = torch.zeros_like(steps[:, :1])
        return torch.cat((zero, torch.cumsum(steps, dim=1)), dim=1)

    def _targets_from_mean(
        self, tokens: torch.Tensor, cumulative: torch.Tensor
    ) -> torch.Tensor:
        """Constant-speed arc positions from the single mean-rate row."""
        speed = tokens[:, self.num_waypoints, 0].clamp_min(0.0)
        total = cumulative[:, -1]
        steps = torch.arange(
            self.action_horizon, device=tokens.device, dtype=tokens.dtype
        )
        # A zero-speed token stays at 0 and replays the first waypoint.
        return torch.minimum(speed[:, None] * self.dt * steps[None, :], total[:, None])

    def _targets_from_per_waypoint(
        self, tokens: torch.Tensor, cumulative: torch.Tensor
    ) -> torch.Tensor:
        """Arc positions recovered by integrating the per-interval rates.

        Each interval takes ``arc_distance / rate`` seconds, so accumulating
        those gives the elapsed time at every waypoint. Sampling that curve at
        ``k * dt`` inverts it back to an arc position per control step, which is
        what lets a non-uniform chunk replay at its original pace.
        """
        rates = tokens[:, self.num_waypoints :, 0].clamp_min(0.0)
        interval_arc = cumulative[:, 1:] - cumulative[:, :-1]
        interval_rate = rates[:, :-1]
        moving = interval_arc > self.zero_dist_epsilon
        usable = interval_rate > self.zero_dist_epsilon
        duration = torch.zeros_like(interval_arc)
        duration = torch.where(
            moving & usable,
            interval_arc / interval_rate.clamp_min(self.zero_dist_epsilon),
            duration,
        )
        # A predicted nonzero interval with zero rate must hold, not teleport:
        # give it more time than the horizon so sampling never crosses it.
        stalled = self.dt * (self.action_horizon + 1)
        duration = torch.where(
            moving & ~usable, torch.full_like(duration, stalled), duration
        )
        elapsed = torch.cat(
            (torch.zeros_like(duration[:, :1]), torch.cumsum(duration, dim=1)), dim=1
        )
        steps = torch.arange(
            self.action_horizon, device=tokens.device, dtype=tokens.dtype
        )
        time_targets = (self.dt * steps)[None, :].expand(len(tokens), -1)
        # Invert time -> arc by interpolating the cumulative-time curve.
        upper = torch.searchsorted(
            elapsed.contiguous(), time_targets.contiguous(), right=True
        ).clamp(1, self.num_waypoints - 1)
        lower = upper - 1
        t_lo = torch.gather(elapsed, 1, lower)
        t_hi = torch.gather(elapsed, 1, upper)
        alpha = ((time_targets - t_lo) / (t_hi - t_lo).clamp_min(self.zero_dist_epsilon))
        alpha = alpha.clamp(0.0, 1.0)
        s_lo = torch.gather(cumulative, 1, lower)
        s_hi = torch.gather(cumulative, 1, upper)
        return s_lo + alpha * (s_hi - s_lo)

    def forward(self, batch: dict) -> dict:
        tokens = _as_batched(batch["pred_action"], "ArcDetokenizeStage input")
        rows = arc_token_rows(self.num_waypoints, self.velocity_mode)
        expected = (rows, PLANAR_ACTION_DIM)
        if tuple(tokens.shape[1:]) != expected:
            raise ValueError(
                f"ArcDetokenizeStage expects (B, {expected[0]}, {expected[1]}) for "
                f"velocity_mode={self.velocity_mode!r}, got {tuple(tokens.shape)}"
            )

        waypoints = tokens[:, : self.num_waypoints]
        cumulative = self._arc_positions(waypoints)
        if self.velocity_mode == "mean":
            targets = self._targets_from_mean(tokens, cumulative)
        else:
            targets = self._targets_from_per_waypoint(tokens, cumulative)

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
        if self.velocity_mode == "mean":
            batch["log/ArcSpeed"] = tokens[:, self.num_waypoints, 0].clamp_min(0.0).mean()
        else:
            batch["log/ArcSpeed"] = (
                tokens[:, self.num_waypoints :, 0].clamp_min(0.0).mean()
            )
        batch["log/ArcChunkDistance"] = cumulative[:, -1].mean()
        return batch
