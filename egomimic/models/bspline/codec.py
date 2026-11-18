"""Adaptive B-spline targets and continuous-time execution.

Adapted from B-spline-policy/bspline-policy, revision
61ed5f42fced971d50a89b46417493790876ccd1 (MIT; see LICENSE).
The representation is [relative-to-observation knot time, control point...].
Times are demonstration frame indices, not equally spaced output tokens.
"""

from dataclasses import asdict, dataclass

import numpy as np
from scipy.interpolate import BSpline, generate_knots, make_lsq_spline
from scipy.optimize import minimize_scalar

UPSTREAM_COMMIT = "61ed5f42fced971d50a89b46417493790876ccd1"


@dataclass(frozen=True)
class FittedTrajectory:
    knots: np.ndarray
    control_points: np.ndarray
    length: int
    max_error: float


@dataclass(frozen=True)
class BSplineCodec:
    """Cubic BSP with 16 rows including six boundary-support knots by default."""

    num_knots: int = 16
    degree: int = 3
    max_error: float = 0.002
    smoothing: float = 1e-12
    knot_epsilon: float = 1e-6

    def __post_init__(self):
        if self.degree != 3:
            raise ValueError("The released BSP representation uses cubic splines")
        if type(self.num_knots) is not int or self.num_knots < 2 * self.degree + 2:
            raise ValueError("num_knots must include boundary support and a valid span")
        for name in ("max_error", "smoothing", "knot_epsilon"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")

    def context(self):
        return {
            "kind": "bspline_policy_v1",
            "upstream_commit": UPSTREAM_COMMIT,
            **asdict(self),
            "knot_units": "demonstration_frames_relative_to_observation",
            "fit_error_metric": "max_absolute_component_error",
        }

    def fit(self, actions):
        """FITPACK insertion until the released componentwise error bound is met.

        Fit native commands jointly, including gripper. Short episodes are
        endpoint-padded to four samples so they still have a cubic representation.
        Unlike upstream's warning-and-continue fallback, a failed bound is fatal.
        """
        actions = np.asarray(actions, dtype=np.float64)
        if (
            actions.ndim != 2
            or min(actions.shape) < 1
            or not np.isfinite(actions).all()
        ):
            raise ValueError("Expected nonempty finite (frames, action_dim) actions")
        length = len(actions)
        fit_data = np.concatenate(
            [actions, np.repeat(actions[-1:], max(0, 4 - length), axis=0)]
        )
        times = np.arange(len(fit_data), dtype=np.float64)
        for knots in generate_knots(times, fit_data, k=self.degree, s=self.smoothing):
            spline = make_lsq_spline(times, fit_data, knots, k=self.degree)
            error = float(np.abs(spline(times) - fit_data).max())
            if error < self.max_error:
                return FittedTrajectory(spline.t, spline.c, length, error)
        raise ValueError(
            f"BSP fitting failed tolerance {self.max_error}; error={error}"
        )

    def encode(self, trajectory, frame_index):
        """Select the nearest future knot, preserving upstream support and padding."""
        if not 0 <= frame_index < trajectory.length or int(frame_index) != frame_index:
            raise IndexError(frame_index)
        knots, points = trajectory.knots, trajectory.control_points
        unique = knots[self.degree : -self.degree]
        start = min(
            int(np.searchsorted(unique, frame_index, side="left")), len(unique) - 2
        )
        indices = np.arange(start, start + self.num_knots)
        local_knots = knots[np.minimum(indices, len(knots) - 1)] - frame_index
        local_points = points[np.minimum(indices, len(points) - 1)]
        return np.column_stack([local_knots, local_points]).astype(np.float32)

    def project_knots(self, knots):
        """Upstream forward correction: repair decreases, retain repeated boundaries."""
        knots = np.asarray(knots, dtype=np.float64).copy()
        if knots.shape != (self.num_knots,) or not np.isfinite(knots).all():
            raise ValueError("Invalid predicted knot vector")
        for index in range(1, len(knots)):
            if knots[index] < knots[index - 1]:
                knots[index] = knots[index - 1] + self.knot_epsilon
        # A collapsed prediction has no valid polynomial domain. Give it the
        # smallest positive span while preserving knot/control-point order.
        left, right = self.degree, len(knots) - self.degree - 1
        if knots[right] <= knots[left]:
            for index in range(left + 1, len(knots)):
                knots[index] = max(knots[index], knots[index - 1] + self.knot_epsilon)
        return knots

    def spline(self, parameters):
        parameters = np.asarray(parameters, dtype=np.float64)
        if (
            parameters.ndim != 2
            or parameters.shape[0] != self.num_knots
            or parameters.shape[1] < 2
            or not np.isfinite(parameters).all()
        ):
            raise ValueError("Expected finite (num_knots, 1 + action_dim) parameters")
        knots = self.project_knots(parameters[:, 0])
        # The final k+1 parameter rows pad coefficients to knot-vector length.
        return BSpline(knots, parameters[: -(self.degree + 1), 1:], self.degree)

    def decode(self, parameters, *, horizon=32, speed=1.0, start_time=0.0):
        """Sample at the existing controller rate without renormalizing knot times.

        LIBERO uses speed=1 and fixed 16-command replanning. If a segment ends
        earlier, hold its terminal command until replanning. Future-knot selection
        can put time zero before the valid span: use the first polynomial there,
        as in upstream's deployment wrapper. No phase alignment for delta actions.
        """
        if type(horizon) is not int or horizon < 1:
            raise ValueError("horizon must be a positive integer")
        if not np.isfinite(speed) or speed <= 0 or not np.isfinite(start_time):
            raise ValueError("Invalid execution clock")
        spline = self.spline(parameters)
        times = start_time + np.arange(horizon, dtype=np.float64) * speed
        # Upstream pads the terminal knot beyond multiplicity k+1. SciPy then
        # returns zero exactly at that boundary. Evaluate its left limit so a
        # held endpoint keeps the last command (especially a closed gripper).
        endpoint = np.nextafter(spline.t[-self.degree - 1], -np.inf)
        times = np.minimum(times, endpoint)
        values = spline(times)
        with np.errstate(over="ignore"):
            values = values.astype(np.float32)
        if not np.isfinite(values).all():
            raise ValueError("Non-finite decoded B-spline actions")
        return values

    def alignment_time(
        self, parameters, last_action, *, inference_latency, window_factor=1.5
    ):
        """Paper Eq. 2 for absolute commands only; latency is in knot/frame units.

        Search each polynomial span to avoid assuming the whole curve is unimodal.
        This helper is deliberately not used by the delta-command LIBERO graph.
        """
        if not np.isfinite(inference_latency) or inference_latency < 0:
            raise ValueError("inference_latency must be finite and nonnegative")
        if not np.isfinite(window_factor) or window_factor <= 1:
            raise ValueError("window_factor must be greater than one")
        spline = self.spline(parameters)
        last = np.asarray(last_action, dtype=np.float64)
        if last.shape != spline.c.shape[1:] or not np.isfinite(last).all():
            raise ValueError("last_action shape or values differ")
        upper = min(
            window_factor * inference_latency,
            np.nextafter(spline.t[-self.degree - 1], -np.inf),
        )
        if upper <= 0:
            return 0.0
        boundaries = np.unique(
            np.r_[0.0, spline.t[(spline.t > 0) & (spline.t < upper)], upper]
        )
        def loss(t):
            return float(np.mean((spline(t) - last) ** 2))

        candidates = list(boundaries)
        for left, right in zip(boundaries[:-1], boundaries[1:]):
            candidates.append(
                minimize_scalar(loss, bounds=(left, right), method="bounded").x
            )
        return float(min(candidates, key=loss))
