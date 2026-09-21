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

    def spatial_clock(self, actions):
        physical = actions * self.action_scale
        if self.path_mode == "control":
            physical = torch.diff(physical, dim=1, prepend=physical[:, :1])
        translation = physical[..., self.translation_indices].norm(dim=-1).cumsum(1) / self.distance
        progress = translation
        if self.rotation_indices:
            angular = physical[..., self.rotation_indices].norm(dim=-1).cumsum(1) / self.rotation
            progress = torch.maximum(progress, angular)
        return progress

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
