"""Shared anchored global bases and monotone local-speed clocks.

Reused from the human ARC study, source 8eea7328, without its planar adapter.
These functions are geometry- and embodiment-independent. Finite coefficients
are neither shift-invariant nor an injective encoding of arbitrary curves.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.optimize import least_squares


def basis_matrix(u: np.ndarray, count: int, basis: str) -> np.ndarray:
    u = np.asarray(u, dtype=np.float64)
    if count < 2 or not np.all(np.isfinite(u)):
        raise ValueError("finite coordinates and at least two coefficients required")
    if basis == "chebyshev":
        return np.polynomial.chebyshev.chebvander(2 * u - 1, count - 1)
    if basis == "cosine":
        return np.cos(np.pi * u[..., None] * np.arange(count))
    if basis == "fourier":
        if count < 4 or count % 2:
            raise ValueError("Fourier requires two endpoint slots and sin/cos pairs")
        phase = 2 * np.pi * u[..., None] * np.arange(1, count // 2)
        modes = np.stack((np.sin(phase), np.cos(phase) - 1), axis=-1)
        return np.concatenate(
            (np.ones_like(u)[..., None], u[..., None], modes.reshape(*u.shape, -1)),
            axis=-1,
        )
    raise ValueError(f"unknown basis: {basis}")


def anchored_matrix(u: np.ndarray, count: int, basis: str) -> np.ndarray:
    """First coefficient is the exact start; remaining modes vanish at u=0."""
    matrix = basis_matrix(u, count, basis)
    matrix[..., 1:] -= basis_matrix(np.array([0.0]), count, basis)[0, 1:]
    return matrix


@lru_cache(maxsize=64)
def _fit_inverse(samples: int, count: int, basis: str) -> np.ndarray:
    return _fit_inverse_at(tuple(np.linspace(0, 1, samples)), count, basis)


@lru_cache(maxsize=64)
def _fit_inverse_at(grid: tuple[float, ...], count: int, basis: str) -> np.ndarray:
    samples = len(grid)
    if count > samples:
        raise ValueError("underdetermined coefficient fit")
    matrix = anchored_matrix(np.asarray(grid), count, basis)[:, 1:]
    # Targets are ultimately stored in float32. An equispaced high-order
    # polynomial fit can produce enormous, cancellation-dependent coefficients
    # despite a finite pseudoinverse. Reject it before it becomes training data.
    if np.linalg.cond(matrix) > 1e6:
        raise ValueError(
            "Ill-conditioned coefficient fit; use a stable sampling grid or fewer modes"
        )
    return np.linalg.pinv(matrix)


def fit_curve(
    values: np.ndarray, count: int, basis: str, *, sample_grid=None
) -> np.ndarray:
    """Anchored least squares; defaults to the historical equally spaced grid.

    Chebyshev can explicitly use sorted Lobatto sample coordinates. The codec
    must sample the source curve at those exact coordinates, not relabel a
    uniformly sampled value array.
    """
    values = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(values)) or values.ndim != 2:
        raise ValueError("curve values must be a finite sample-by-channel matrix")
    if sample_grid is not None:
        sample_grid = np.asarray(sample_grid, dtype=np.float64)
        if (
            basis != "chebyshev"
            or sample_grid.shape != (len(values),)
            or not np.isfinite(sample_grid).all()
            or np.any(np.diff(sample_grid) <= 0)
            or sample_grid[0] != 0
            or sample_grid[-1] != 1
        ):
            raise ValueError(
                "Explicit Chebyshev sample grid must increase from zero to one"
            )
    coefficients = np.empty((count, values.shape[1]))
    coefficients[0] = values[0]
    if basis == "fourier":
        # Open-curve trend plus global periodic residual: no artificial closure.
        coefficients[1] = values[-1] - values[0]
        grid = np.linspace(0, 1, len(values))
        matrix = anchored_matrix(grid, count, basis)
        coefficients[2:] = _fourier_inverse(len(values), count) @ (
            values - matrix[:, :2] @ coefficients[:2]
        )
        return coefficients
    inverse = (
        _fit_inverse(len(values), count, basis)
        if sample_grid is None
        else _fit_inverse_at(tuple(sample_grid), count, basis)
    )
    coefficients[1:] = inverse @ (values - values[0])
    return coefficients


@lru_cache(maxsize=32)
def _fourier_inverse(samples: int, count: int) -> np.ndarray:
    if count > samples:
        raise ValueError("underdetermined coefficient fit")
    return np.linalg.pinv(
        basis_matrix(np.linspace(0, 1, samples), count, "fourier")[:, 2:]
    )


def evaluate_curve(coefficients: np.ndarray, u: np.ndarray, basis: str) -> np.ndarray:
    return anchored_matrix(u, len(coefficients), basis) @ coefficients


def _clock_integral(coefficients, tau, with_jac=False):
    """Exact piecewise polynomial integral, split at real roots of h.

    Moving root boundaries contribute zero to the Jacobian because h is zero
    there. Chebyshev arithmetic avoids ill-conditioned monomial conversion.
    """
    cheb = np.polynomial.chebyshev
    coefficients = np.asarray(coefficients, dtype=np.float64)
    x = 2 * np.asarray(tau, dtype=np.float64) - 1
    if not np.all(np.isfinite(coefficients)) or np.any(x < -1) or np.any(x > 1):
        raise ValueError("finite clock coefficients and times within [0,1] required")
    roots = cheb.chebroots(coefficients)
    real = np.real(roots[np.abs(np.imag(roots)) < 1e-9])
    breaks = np.r_[-1.0, np.sort(real[(real > -1) & (real < 1)]), 1.0]
    positive = cheb.chebval((breaks[:-1] + breaks[1:]) / 2, coefficients) > 0
    antiderivative = 0.5 * cheb.chebint(cheb.chebmul(coefficients, coefficients))
    area = np.zeros_like(x)
    jac = np.zeros((x.size, len(coefficients))) if with_jac else None
    if with_jac:
        # Integral of 2*h*T_j in tau equals integral of h*T_j in x.
        n = len(coefficients)
        products = np.zeros((2 * n - 1, n))
        i, j = np.indices((n, n))
        np.add.at(products, (i + j, j), coefficients[i] / 2)
        np.add.at(products, (np.abs(i - j), j), coefficients[i] / 2)
        jac_antiderivative = cheb.chebint(products, axis=0)
    for left, right in zip(breaks[:-1][positive], breaks[1:][positive], strict=True):
        upper = np.clip(x, left, right)
        area += cheb.chebval(upper, antiderivative) - cheb.chebval(left, antiderivative)
        if with_jac:
            jac += (
                cheb.chebval(upper, jac_antiderivative)
                - cheb.chebval(left, jac_antiderivative)[:, None]
            ).T
    return area, jac


def _evaluate_clock(coefficients, tau):
    area, _ = _clock_integral(coefficients, np.r_[np.asarray(tau), 1.0])
    mass = float(area[-1])
    if not np.all(np.isfinite(area)):
        raise ValueError("nonfinite clock")
    return (
        np.clip(area[:-1] / mass, 0, 1) if mass > 1e-20 else np.zeros_like(area[:-1])
    ), mass


def clock_grid(
    coefficients: np.ndarray, samples: int = 257
) -> tuple[np.ndarray, float]:
    """Evaluate exact integral; all-zero token explicitly means stationary pose."""
    return _evaluate_clock(coefficients, np.linspace(0, 1, samples))


def canonicalize_clock(coefficients: np.ndarray, samples: int = 257) -> np.ndarray:
    coefficients = np.asarray(coefficients, dtype=np.float64)
    _, mass = clock_grid(coefficients, samples)
    if mass <= 1e-20:
        if np.any(coefficients):
            raise ValueError("nonzero clock has no positive speed support")
        return coefficients.copy()
    return coefficients / np.sqrt(mass)


@dataclass(frozen=True)
class ClockFit:
    coefficients: np.ndarray
    nfev: int
    converged: bool

    def evaluate(self, tau: np.ndarray) -> np.ndarray:
        return _evaluate_clock(self.coefficients, tau)[0]


@lru_cache(maxsize=32)
def _clock_fit_inverse(count: int) -> np.ndarray:
    if count > 257:
        raise ValueError("underdetermined clock fit")
    return np.linalg.pinv(basis_matrix(np.linspace(0, 1, 257), count, "chebyshev"))


def fit_clock(
    tau: np.ndarray, progress: np.ndarray, count: int, method: str = "nonlinear"
) -> ClockFit:
    """Fit a deterministic nondecreasing clock, fixing its positive scale gauge.

    Optimizer residuals use a dense time grid plus a unit-area constraint. The
    result is always normalized again, so diffusion targets do not contain an
    arbitrary common coefficient scale. This is not a global-optimum claim.
    """
    if method not in {"nonlinear", "sqrt_speed_linear"}:
        raise ValueError("unknown clock fitting method")
    tau, progress = np.asarray(tau), np.asarray(progress)
    if (
        tau.ndim != 1
        or progress.shape != tau.shape
        or len(tau) < 2
        or not np.all(np.isfinite(tau))
        or not np.all(np.isfinite(progress))
        or np.any(np.diff(tau) <= 0)
        or np.any(np.diff(progress) < -1e-12)
        or not np.isclose(tau[0], 0)
        or not np.isclose(tau[-1], 1)
        or not np.isclose(progress[0], 0)
    ):
        raise ValueError("clock samples must span [0,1] with monotone progress")
    if progress[-1] <= 1e-12:
        return ClockFit(np.zeros(count), 0, True)
    if not np.isclose(progress[-1], 1):
        raise ValueError("moving clock must end at unit progress")
    grid = np.linspace(0, 1, 257)
    target = np.interp(grid, tau, progress)
    matrix = basis_matrix(grid, count, "chebyshev")
    speed = np.maximum(np.gradient(target, grid), 0)
    if method == "sqrt_speed_linear":
        # Deterministic linear target fit; decoding still uses the exact
        # normalized positive-polynomial integral. No per-sample optimizer.
        coefficients = _clock_fit_inverse(count) @ np.sqrt(speed)
        return ClockFit(canonicalize_clock(coefficients), 0, True)
    initial = np.linalg.lstsq(matrix, np.sqrt(speed), rcond=None)[0]
    initial = canonicalize_clock(initial)

    def residual_jac(coefficients):
        area, derivative = _clock_integral(coefficients, grid, with_jac=True)
        mass = max(float(area[-1]), 1e-20)
        normalized = area / mass
        jac = (derivative - normalized[:, None] * derivative[-1]) / mass
        # Weak high-mode penalty chooses a stable fit, not an unproved unique optimum.
        penalty = 1e-5 * np.arange(count) ** 2
        residual = np.concatenate(
            (normalized - target, [area[-1] - 1], penalty * coefficients)
        )
        jacobian = np.vstack((jac, derivative[-1], np.diag(penalty)))
        return residual, jacobian

    result = least_squares(
        lambda c: residual_jac(c)[0],
        initial,
        jac=lambda c: residual_jac(c)[1],
        max_nfev=60,
        ftol=1e-7,
        xtol=1e-7,
        gtol=1e-7,
    )
    return ClockFit(canonicalize_clock(result.x), result.nfev, bool(result.success))
