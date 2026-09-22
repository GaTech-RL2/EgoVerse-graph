"""Native-control and duration-ARC codecs for chunked RL.

Spatial window selection and encoding are separate: ``native_window`` is an
uncompressed control with the exact same spatial boundaries as ``arc``.
Rotation budgets are radians, distances are in the units of action_scale.
No observation/action target offset is applied to transition replay.
"""
from __future__ import annotations

import math
import torch
from torch import nn


def interpolate(x, y, query):
    """Batched piecewise-linear interpolation; x[B,T], y[B,T,C]."""
    right = torch.searchsorted(x.contiguous(), query.contiguous(), right=True).clamp(1, x.shape[1] - 1)
    left = right - 1
    x0, x1 = x.gather(1, left), x.gather(1, right)
    weight = ((query - x0) / (x1 - x0).clamp_min(1e-8)).clamp(0, 1)
    a = y.gather(1, left[..., None].expand(-1, -1, y.shape[-1]))
    b = y.gather(1, right[..., None].expand(-1, -1, y.shape[-1]))
    return a + weight[..., None] * (b - a)


class ControlChunkCodec(nn.Module):
    def __init__(self, action_dim, native_horizon, kind="native", waypoints=4,
                 distance=0.1, rotation=None, translation_indices=None,
                 rotation_indices=None, action_scale=None, path_mode="delta",
                 temporal_floor=0.01):
        super().__init__()
        if kind not in {"native", "native_window", "arc"}:
            raise ValueError("kind must be native, native_window, or arc")
        if path_mode not in {"delta", "control"}:
            raise ValueError("path_mode must be delta or control")
        if native_horizon < 1 or waypoints < 2 or not math.isfinite(distance) or distance <= 0:
            raise ValueError("invalid horizon, waypoints, or spatial distance")
        if temporal_floor <= 0:
            raise ValueError("a positive temporal floor preserves stationary intervals")
        self.kind, self.path_mode = kind, path_mode
        self.action_dim, self.native_horizon = int(action_dim), int(native_horizon)
        self.waypoints, self.distance = int(waypoints), float(distance)
        self.rotation = None if rotation is None else float(rotation)
        self.translation_indices = tuple(range(action_dim) if translation_indices is None else translation_indices)
        self.rotation_indices = tuple(rotation_indices or ())
        if self.rotation_indices and (self.rotation is None or self.rotation <= 0):
            raise ValueError("rotation channels require a positive physical rotation budget")
        if not self.rotation_indices and self.rotation is not None:
            raise ValueError("rotation is not defined for this action space")
        self.register_buffer("action_scale", torch.tensor(action_scale or [1.] * action_dim))
        if self.action_scale.numel() != action_dim or not bool((self.action_scale > 0).all()):
            raise ValueError("action_scale must contain one positive scale per action")
        self.temporal_floor = float(temporal_floor)

    @property
    def encoded_dim(self):
        if self.kind == "arc":
            return self.waypoints * (self.action_dim + 1)
        return self.native_horizon * self.action_dim + int(self.kind == "native_window")

    def spatial_distances(self, actions):
        """Cumulative translation/control and angular travel in physical units."""
        physical = actions * self.action_scale
        if self.path_mode == "control":
            physical = torch.diff(physical, dim=1, prepend=physical[:, :1])
        translation = physical[..., self.translation_indices].norm(dim=-1).cumsum(1)
        angular = (physical[..., self.rotation_indices].norm(dim=-1).cumsum(1)
                   if self.rotation_indices else None)
        return translation, angular

    def spatial_clock(self, actions):
        translation, angular = self.spatial_distances(actions)
        progress = translation / self.distance
        return torch.maximum(progress, angular / self.rotation) if angular is not None else progress

    def window_lengths(self, actions):
        if self.kind == "native":
            return torch.full((len(actions),), self.native_horizon, device=actions.device, dtype=torch.long)
        return (self.spatial_clock(actions) <= 1).sum(1).clamp(1, self.native_horizon)

    def encode(self, actions):
        actions = actions[:, :self.native_horizon]
        if actions.shape[1:] != (self.native_horizon, self.action_dim):
            raise ValueError("codec needs a complete native action window")
        lengths = self.window_lengths(actions)
        if self.kind == "native":
            return actions.flatten(1), lengths
        if self.kind == "native_window":
            valid = torch.arange(self.native_horizon, device=actions.device)[None] < lengths[:, None]
            flattened = (actions * valid[..., None]).flatten(1)
            return torch.cat([flattened, (2 * lengths[:, None] / self.native_horizon - 1)], -1), lengths

        # Anchor at native time zero; first support at time one preserves a[t].
        zeros = torch.zeros_like(actions[:, :1])
        curve = (torch.cat([zeros, actions.cumsum(1)], 1) / self.native_horizon
                 if self.path_mode == "delta" else torch.cat([actions[:, :1], actions], 1))
        physical_step = torch.diff(curve, dim=1).mul(self.action_scale).norm(dim=-1)
        # Include all control channels (including grip) in support allocation.
        step = physical_step / physical_step.mean(1, keepdim=True).clamp_min(1e-8) + self.temporal_floor
        clock = torch.cat([torch.zeros_like(step[:, :1]), step.cumsum(1)], 1)
        end = clock.gather(1, lengths[:, None])
        fractions = torch.linspace(0, 1, self.waypoints, device=actions.device)[None]
        targets = clock[:, 1:2] + fractions * (end - clock[:, 1:2])
        native_t = torch.arange(self.native_horizon + 1, device=actions.device, dtype=actions.dtype)
        native_t = native_t[None].expand(len(actions), -1)
        times = interpolate(clock, native_t[..., None], targets).squeeze(-1)
        times[:, 0] = 1
        times[:, -1] = lengths
        supports = interpolate(native_t, curve, times)
        durations = torch.diff(times, dim=1, prepend=torch.zeros_like(times[:, :1]))
        tokens = torch.cat([supports, (2 * durations / self.native_horizon - 1)[..., None]], -1)
        return tokens.flatten(1), lengths

    def decode(self, latent):
        latent = latent.clamp(-1, 1)
        if self.kind == "native":
            actions = latent.view(-1, self.native_horizon, self.action_dim)
            return actions, torch.full((len(latent),), self.native_horizon, dtype=torch.long, device=latent.device)
        if self.kind == "native_window":
            lengths = ((latent[:, -1] + 1) * self.native_horizon / 2).round().long().clamp(1, self.native_horizon)
            return latent[:, :-1].view(-1, self.native_horizon, self.action_dim), lengths
        tokens = latent.view(-1, self.waypoints, self.action_dim + 1)
        durations = ((tokens[..., -1] + 1) * self.native_horizon / 2).clamp_min(0)
        # First support is always the first native action; snap the remaining
        # total time to a valid integer number of environment transitions.
        rest = durations[:, 1:]
        lengths = (1 + rest.sum(1)).round().long().clamp(1, self.native_horizon)
        rest = rest * ((lengths - 1) / rest.sum(1).clamp_min(1e-8))[:, None]
        times = torch.cat([torch.zeros_like(durations[:, :1]),
                           torch.ones_like(durations[:, :1]), 1 + rest.cumsum(1)], 1)
        values = tokens[..., :-1]
        anchor = torch.zeros_like(values[:, :1]) if self.path_mode == "delta" else values[:, :1]
        values = torch.cat([anchor, values], 1)
        queries = torch.arange(self.native_horizon + 1, device=latent.device, dtype=latent.dtype)[None].expand(len(latent), -1)
        queries = torch.minimum(queries, lengths[:, None])
        decoded = interpolate(times, values, queries)
        actions = (torch.diff(decoded, dim=1) * self.native_horizon
                   if self.path_mode == "delta" else decoded[:, 1:])
        return actions.clamp(-1, 1), lengths


class ShapeTimeControlChunkCodec(ControlChunkCodec):
    """Factor a commanded control path into geometry, extent, clock and duration.

    Shape supports are uniform in geometric arc progress, independent of native
    timestamps. A separate clock gives progress at each native control boundary;
    zero clock increments preserve holds. Total duration is one scalar, not the
    sum of M noisy interval predictions. M changes only geometric resolution.

    Delta controls integrate to a commanded displacement path. Direct controls
    describe a path in control space, with a separate initial control anchor;
    they are not mislabeled as Cartesian motion. Native/native_window retain
    their original contracts for comparison. This is a new checkpoint format.
    """
    def __init__(self, *args, geometry_units=None, extent_reference=1.,
                 duration_reference=1., factor_loss_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        if (not math.isfinite(extent_reference) or extent_reference <= 0
                or not math.isfinite(duration_reference) or duration_reference <= 0):
            raise ValueError("extent and duration references must be finite and positive")
        units = torch.tensor(geometry_units or [1.] * self.action_dim)
        if units.numel() != self.action_dim or not bool(torch.isfinite(units).all() and (units > 0).all()):
            raise ValueError("geometry_units needs one finite positive physical unit per channel")
        self.register_buffer("path_metric", self.action_scale / units)
        self.extent_reference, self.duration_reference = float(extent_reference), float(duration_reference)
        max_extent = self.native_horizon if self.path_mode == "delta" else (2 if self.native_horizon > 1 else 0)
        self.max_log_extent = math.log1p(max_extent / self.extent_reference)
        factors = {"shape": self.waypoints * self.action_dim, "extent": self.action_dim,
                   "clock": self.native_horizon, "duration": 1}
        if self.path_mode == "control":
            factors["anchor"] = self.action_dim
        self.factor_slices = {}
        offset = 0
        for name, size in factors.items():
            self.factor_slices[name] = slice(offset, offset + size)
            offset += size
        weights = dict(factor_loss_weights or {name: 1. for name in factors})
        if set(weights) != set(factors) or any(not math.isfinite(v) or v <= 0 for v in weights.values()):
            raise ValueError("factor_loss_weights must positively weight every representation factor")
        # Mean weight one retains the actor loss's overall scale. Increasing M
        # cannot silently reduce the relative supervision of clock or duration.
        scalar_weights = torch.cat([torch.full((size,), weights[name] / size) for name, size in factors.items()])
        scalar_weights *= len(scalar_weights) / sum(weights.values())
        self.register_buffer("factor_weights", scalar_weights)

    @property
    def encoded_dim(self):
        if self.kind != "arc":
            return super().encoded_dim
        return self.waypoints * self.action_dim + self.action_dim + self.native_horizon + 1 + (
            self.action_dim if self.path_mode == "control" else 0)

    @property
    def actor_loss_weights(self):
        return self.factor_weights if self.kind == "arc" else None

    def _lengths(self, latent):
        code = latent[:, self.factor_slices["duration"]].squeeze(-1)
        code = code.clamp(math.log(1 / self.duration_reference),
                          math.log(self.native_horizon / self.duration_reference))
        return (code.exp() * self.duration_reference).round().long().clamp(1, self.native_horizon)

    def _clock_weights(self, latent, lengths):
        valid = torch.arange(self.native_horizon, device=latent.device)[None] < lengths[:, None]
        if self.path_mode == "control":
            # The initial control is predicted by the anchor. Progress cannot
            # move away from it before the first native control is executed.
            valid[:, 0] = False
        weights = ((latent[:, self.factor_slices["clock"]] + 1) / 2).clamp(0, 1).square() * valid
        total = weights.sum(1, keepdim=True)
        # A stationary path has no identifiable geometric clock. Its canonical
        # clock is uniform, and its zero extent still yields stationary controls.
        return torch.where(total > 1e-12, weights / total.clamp_min(1e-12),
                           valid.to(latent.dtype) / valid.sum(1, keepdim=True).clamp_min(1))

    def project_latent(self, latent):
        """Apply codec bounds before Q ranking; canonicalize ignored padding."""
        if self.kind != "arc":
            return latent.clamp(-1, 1)
        lengths = self._lengths(latent)
        extent = latent[:, self.factor_slices["extent"]].clamp(0, self.max_log_extent)
        if self.path_mode == "control":
            # One native control has an anchor but no subsequent geometric path.
            extent = torch.where(lengths[:, None] > 1, extent, torch.zeros_like(extent))
        shape = latent[:, self.factor_slices["shape"]].clamp(-1, 1).reshape(-1, self.waypoints, self.action_dim)
        shape = torch.where(extent[:, None] > 0, shape, torch.zeros_like(shape))
        parts = [shape.flatten(1), extent, 2 * self._clock_weights(latent, lengths).sqrt() - 1,
                 (lengths.to(latent.dtype) / self.duration_reference).log()[:, None]]
        if self.path_mode == "control":
            parts.append(latent[:, self.factor_slices["anchor"]].clamp(-1, 1))
        return torch.cat(parts, -1)

    def encode(self, actions, *, lengths=None):
        if self.kind != "arc":
            if lengths is not None:
                raise ValueError("explicit lengths are only supported for ARC reference replay")
            return super().encode(actions)
        actions = actions[:, :self.native_horizon]
        if actions.shape[1:] != (self.native_horizon, self.action_dim):
            raise ValueError("codec needs a complete native action window")
        if lengths is None:
            lengths = self.window_lengths(actions)
        elif (lengths.shape != (len(actions),) or lengths.dtype != torch.long
              or bool(((lengths < 1) | (lengths > self.native_horizon)).any())):
            raise ValueError("explicit native lengths must be int64 in [1, native_horizon]")
        anchor = torch.zeros_like(actions[:, :1]) if self.path_mode == "delta" else actions[:, :1]
        path = (torch.cat([anchor, actions.cumsum(1)], 1) if self.path_mode == "delta"
                else torch.cat([anchor, actions], 1))
        native_t = torch.arange(self.native_horizon + 1, device=actions.device)[None].expand(len(actions), -1)
        index = torch.minimum(native_t, lengths[:, None])
        path = path.gather(1, index[..., None].expand(-1, -1, self.action_dim))
        relative = path - anchor
        weighted = relative * self.path_metric
        steps = torch.diff(weighted, dim=1).norm(dim=-1)
        travel = steps.sum(1, keepdim=True)
        progress = torch.cat([torch.zeros_like(steps[:, :1]), steps.cumsum(1)], 1) / travel.clamp_min(1e-12)
        knots = torch.arange(1, self.waypoints + 1, device=actions.device, dtype=actions.dtype)[None] / self.waypoints
        # Per-channel extents prevent a large grip/yaw coordinate from shrinking
        # the translation targets. Extents belong to shape, not to speed or cap.
        extent = relative.abs().amax(1)
        shape = interpolate(progress, relative, knots.expand(len(actions), -1)) / extent[:, None].clamp_min(1e-12)
        valid = torch.arange(self.native_horizon, device=actions.device)[None] < lengths[:, None]
        clock = torch.where(travel > 1e-12, steps / travel.clamp_min(1e-12),
                            valid.to(actions.dtype) / lengths[:, None])
        parts = [shape.flatten(1), torch.log1p(extent / self.extent_reference),
                 2 * clock.sqrt() - 1, (lengths.to(actions.dtype) / self.duration_reference).log()[:, None]]
        if self.path_mode == "control":
            parts.append(anchor[:, 0])
        return self.project_latent(torch.cat(parts, -1)), lengths

    def decode(self, latent):
        if self.kind != "arc":
            return super().decode(latent)
        latent = self.project_latent(latent)
        lengths = self._lengths(latent)
        shape = latent[:, self.factor_slices["shape"]].reshape(-1, self.waypoints, self.action_dim)
        shape = torch.cat([torch.zeros_like(shape[:, :1]), shape], 1)
        extent = latent[:, self.factor_slices["extent"]].expm1() * self.extent_reference
        weights = self._clock_weights(latent, lengths)
        progress = torch.cat([torch.zeros_like(weights[:, :1]), weights.cumsum(1)], 1).clamp(0, 1)
        knots = torch.linspace(0, 1, self.waypoints + 1, device=latent.device, dtype=latent.dtype)[None].expand(len(latent), -1)
        path = interpolate(knots, shape, progress) * extent[:, None]
        if self.path_mode == "delta":
            actions = torch.diff(path, dim=1)
        else:
            actions = path[:, 1:] + latent[:, self.factor_slices["anchor"]][:, None]
        return actions.clamp(-1, 1), lengths
