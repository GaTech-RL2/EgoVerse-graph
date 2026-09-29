"""Global geometry bases with independent local ARC clocks for LIBERO.

The three arms differ only in their geometry representation. Translation and
SO(3) use separate normalized arc coordinates and the same nonnegative-speed
clocks. Gripper is a full-budget global cosine series on the control grid.
Packing rows have no waypoint or time meaning; decoding has no learned weights.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Slerp

from egomimic.rldb.zarr.arc_global_basis import (
    _evaluate_clock,
    basis_matrix,
    evaluate_curve,
    fit_clock,
    fit_curve,
)
from egomimic.rldb.zarr.libero_arc import (
    LiberoArcCodec,
    _from_rotation6d,
    _integrate_commands,
    _rotation6d,
)


class LiberoArcGlobalCodec(LiberoArcCodec):
    """Packed 12-wide [xyz basis, rot6d basis, two clocks, grip, durations].

    Geometry fits use the same 257-point grid. Fourier uses a linear endpoint
    trend plus sine/cosine residuals, so open curves are not forced closed.
    Rotation6d is continuous across pi; projecting it back onto SO(3) preserves
    valid orientations but is an approximation, measured in native commands.
    Clock coefficients encode local speed via integral(relu(h(t))**2), not mean
    velocity. All-zero clocks denote a stationary stream, including full dwells.
    """

    mode = "global_basis"

    def __init__(
        self,
        basis="chebyshev",
        geometry=32,
        clock=31,
        clock_fit="sqrt_speed_linear",
        geometry_fit_grid="uniform",
        num_waypoints=None,
        **kwargs,
    ):
        self.basis, self.geometry, self.clock = str(basis), int(geometry), int(clock)
        self.clock_fit = str(clock_fit)
        self.geometry_fit_grid = str(geometry_fit_grid)
        if self.geometry_fit_grid not in {"uniform", "chebyshev_lobatto"}:
            raise ValueError("Unknown geometry fitting grid")
        if self.geometry_fit_grid != "uniform" and self.basis != "chebyshev":
            raise ValueError("Lobatto fitting is only supported for Chebyshev")
        horizon = int(kwargs.get("horizon", 32))
        self.scalars = 9 * self.geometry + 2 * self.clock + horizon + 2
        if self.scalars % 12:
            raise ValueError("Global ARC packing must have no unused scalar slots")
        rows = self.scalars // 12
        if num_waypoints is not None and int(num_waypoints) != rows:
            raise ValueError("Packed ARC rows differ from the model action horizon")
        super().__init__(num_waypoints=rows, **kwargs)
        if self.basis not in {"uniform", "chebyshev", "fourier"}:
            raise ValueError("Expected uniform, chebyshev or fourier geometry")
        if not 2 <= self.geometry <= 257 or not 2 <= self.clock <= 257:
            raise ValueError("Geometry/clock counts must lie within [2,257]")
        if self.horizon < 2:
            raise ValueError("Global grip series requires at least two control ticks")
        if self.clock_fit not in {"sqrt_speed_linear", "nonlinear"}:
            raise ValueError("Unknown clock fitting method")
        if self.basis != "uniform":
            basis_matrix(np.array([0.0]), self.geometry, self.basis)

    def _stream(self, cumulative, budget):
        """Retain source timestamps, including leading/internal/trailing holds."""
        frames = np.arange(self.horizon + 1, dtype=np.float64)
        if budget is not None and budget < cumulative[-1]:
            upper = int(np.searchsorted(cumulative, budget, side="left"))
            fraction = (budget - cumulative[upper - 1]) / (
                cumulative[upper] - cumulative[upper - 1]
            )
            stop = upper - 1 + fraction
            frames = np.r_[frames[frames < stop - 1e-12], stop]
        progress = np.interp(frames, np.arange(self.horizon + 1), cumulative)
        if progress[-1] > 1e-12:
            progress /= progress[-1]
        else:
            progress[:] = 0
        clock = fit_clock(
            frames / frames[-1], progress, self.clock, method=self.clock_fit
        ).coefficients
        return frames, progress, clock, frames[-1] * self.dt

    def encode(self, actions):
        actions = np.asarray(actions, dtype=np.float64)
        if actions.shape != (self.horizon, 7) or not np.isfinite(actions).all():
            raise ValueError(f"Expected finite ({self.horizon},7) LIBERO actions")
        xyz, rotation = _integrate_commands(
            actions, self.translation_scale, self.rotation_scale
        )
        t_arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xyz, axis=0), axis=-1))]
        r_arc = np.r_[0.0, np.cumsum((rotation[1:] * rotation[:-1].inv()).magnitude())]
        tf, tu, tc, td = self._stream(t_arc, self.max_translation)
        rotation_budget = (
            None
            if self.max_rotation_degrees is None
            else np.deg2rad(self.max_rotation_degrees)
        )
        rf, ru, rc, rd = self._stream(r_arc, rotation_budget)
        grid = np.linspace(0, 1, self.geometry if self.basis == "uniform" else 257)
        explicit_grid = None
        if self.geometry_fit_grid == "chebyshev_lobatto":
            grid = (1 - np.cos(np.linspace(0, np.pi, 257))) / 2
            explicit_grid = grid
        unique_t = np.r_[True, np.diff(tu) > 1e-12]
        positions = np.column_stack(
            [
                np.interp(tf, np.arange(self.horizon + 1), xyz[:, channel])
                for channel in range(3)
            ]
        )
        geometry_xyz = np.column_stack(
            [
                np.interp(grid, tu[unique_t], positions[unique_t, channel])
                for channel in range(3)
            ]
        )
        unique_r = np.r_[True, np.diff(ru) > 1e-12]
        sampled_rotations = Slerp(np.arange(self.horizon + 1), rotation)(rf)
        if unique_r.sum() > 1:
            geometry_rotation = _rotation6d(
                # Deduplicating a stationary tail can retain a terminal knot
                # a few ulps below 1. Query only the actual inclusive domain.
                Slerp(ru[unique_r], sampled_rotations[unique_r])(
                    np.clip(grid, ru[unique_r][0], ru[unique_r][-1])
                )
            )
        else:
            geometry_rotation = np.repeat(_rotation6d(rotation[:1]), len(grid), axis=0)
        if self.basis != "uniform":
            geometry_xyz = fit_curve(
                geometry_xyz, self.geometry, self.basis, sample_grid=explicit_grid
            )
            geometry_rotation = fit_curve(
                geometry_rotation, self.geometry, self.basis, sample_grid=explicit_grid
            )
        # Exact cosine-series interpolation on native grip ticks, shared by all
        # arms. It preserves grip-only actions without coupling them to motion.
        grip = fit_curve(actions[:, 6:7], self.horizon, "cosine")
        packed = (
            np.concatenate(
                [
                    geometry_xyz.ravel(),
                    geometry_rotation.ravel(),
                    tc,
                    rc,
                    grip.ravel(),
                    [td, rd],
                ]
            )
            .reshape(self.num_waypoints, 12)
            .astype(np.float32)
        )
        if not np.isfinite(packed).all():
            raise ValueError("Nonfinite global ARC target")
        return packed

    def _unpack(self, tokens):
        tokens = np.asarray(tokens, dtype=np.float64)
        if tokens.shape != (self.num_waypoints, 12) or not np.isfinite(tokens).all():
            raise ValueError("Incorrect global ARC shape or nonfinite token")
        flat = tokens.ravel()
        g, c = self.geometry, self.clock
        return (
            flat[: 3 * g].reshape(g, 3),
            flat[3 * g : 9 * g].reshape(g, 6),
            flat[9 * g : 9 * g + c],
            flat[9 * g + c : 9 * g + 2 * c],
            flat[9 * g + 2 * c : -2, None],
            flat[-2:],
        )

    def decode(self, tokens):
        xyz, rot, tc, rc, grip, durations = self._unpack(tokens)
        # Missing/nonpositive duration must not produce a zero-time teleport.
        durations = np.where(
            durations > 1e-8,
            np.minimum(durations, self.horizon * self.dt),
            (self.horizon + 1) * self.dt,
        )
        ticks = np.arange(self.horizon + 1) * self.dt
        tu = _evaluate_clock(tc, np.minimum(ticks / durations[0], 1))[0]
        ru = _evaluate_clock(rc, np.minimum(ticks / durations[1], 1))[0]
        if self.basis == "uniform":
            grid = np.linspace(0, 1, self.geometry)
            positions = np.column_stack(
                [np.interp(tu, grid, xyz[:, i]) for i in range(3)]
            )
            orientation = Slerp(grid, _from_rotation6d(rot))(ru)
        else:
            positions = evaluate_curve(xyz, tu, self.basis)
            orientation = _from_rotation6d(evaluate_curve(rot, ru, self.basis))
        native_grip = evaluate_curve(grip, np.linspace(0, 1, self.horizon), "cosine")[
            :, 0
        ].clip(-1, 1)
        decoded = np.column_stack(
            [
                np.diff(positions, axis=0) / self.translation_scale,
                (orientation[1:] * orientation[:-1].inv()).as_rotvec()
                / self.rotation_scale,
                native_grip,
            ]
        ).astype(np.float32)
        if not np.isfinite(decoded).all():
            raise ValueError("Nonfinite global ARC decoded controls")
        return decoded

    def represented_seconds(self, tokens):
        return float(np.clip(self._unpack(tokens)[-1].min(), 0, self.horizon * self.dt))

    def token_scale(self):
        scale = np.ones(self.scalars, dtype=np.float32)
        scale[: 3 * self.geometry] = self.translation_scale * self.horizon
        scale[-2:] = self.dt * self.horizon
        return scale.reshape(self.num_waypoints, 12)
