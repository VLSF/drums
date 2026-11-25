"""Fit the Dirichlet heat trace and reconstruct a Euclidean triangle."""

from itertools import permutations

import numpy as np
from scipy.optimize import brentq, minimize_scalar


ORDERINGS = np.array(list(permutations(range(3))))
MIN_GAMMA = 1.0 / 3.0
MIN_COT_SUM = 3.0 * np.sqrt(3.0)
GAMMA_ROUNDOFF = 1e-12
SYMMETRIC_COT_SUM_ROUNDOFF = 5e-10


def geometry(vertices):
    """Return [area, perimeter, corner heat coefficient] for (..., 3, 2) vertices."""
    vertices = np.asarray(vertices, dtype=float)
    if vertices.shape[-2:] != (3, 2) or not np.isfinite(vertices).all():
        raise ValueError("Expected finite triangle vertices with shape (..., 3, 2)")
    previous = np.roll(vertices, 1, axis=-2) - vertices
    following = np.roll(vertices, -1, axis=-2) - vertices
    cross = previous[..., 0] * following[..., 1] - previous[..., 1] * following[..., 0]
    area = np.abs(cross[..., 0]) / 2.0
    if np.any(area <= 0.0):
        raise ValueError("Triangle area must be positive")
    angles = np.arctan2(np.abs(cross), np.sum(previous * following, axis=-1))
    perimeter = np.linalg.norm(previous, axis=-1).sum(axis=-1)
    gamma = (np.pi * np.sum(1.0 / angles, axis=-1) - 1.0) / 24.0
    return np.stack((area, perimeter, gamma), axis=-1)


def _design(times):
    times = np.asarray(times, dtype=float)
    if (times.ndim != 1 or len(times) < 3 or not np.isfinite(times).all()
            or np.any(times <= 0.0) or len(np.unique(times)) < 3):
        raise ValueError("Expected at least three distinct, finite, positive times")
    t_min = float(times.min())
    u = times / t_min
    return np.column_stack((1.0 / u, 1.0 / np.sqrt(u), np.ones_like(u))), t_min


def fit_trace(eigenvalues, times):
    """Fit alpha/t + beta/sqrt(t) + gamma by ordinary least squares in trace space."""
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    if (eigenvalues.ndim != 1 or not len(eigenvalues) or
            not np.isfinite(eigenvalues).all() or np.any(eigenvalues <= 0.0)):
        raise ValueError("Expected a nonempty vector of finite, positive eigenvalues")
    design, t_min = _design(times)
    times = np.asarray(times, dtype=float)
    trace = np.zeros_like(times, dtype=np.longdouble)
    for start in range(0, len(eigenvalues), 8192):
        block = eigenvalues[start:start + 8192]
        trace += np.exp(-times[:, None] * block).sum(axis=1, dtype=np.longdouble)
    trace = np.asarray(trace, dtype=float)
    trace_norm = np.linalg.norm(trace)
    if trace_norm == 0.0:
        raise ValueError("The truncated trace underflowed on the chosen time grid")

    # Scaling the columns changes parameter units only, not the OLS weights.
    column_norms = np.linalg.norm(design, axis=0)
    scaled_design = design / column_norms
    scaled_fit, _, rank, singular_values = np.linalg.lstsq(scaled_design, trace, rcond=None)
    if rank < 3:
        raise ValueError("The heat-trace design is numerically rank deficient")
    fit = scaled_fit / column_norms
    alpha, beta, gamma = fit * np.array([t_min, np.sqrt(t_min), 1.0])
    return {
        "coefficients": np.array([alpha, beta, gamma]),
        "area": float(4.0 * np.pi * alpha),
        "perimeter": float(-8.0 * np.sqrt(np.pi) * beta),
        "gamma": float(gamma),
        "relative_residual": float(np.linalg.norm(trace - design @ fit) / trace_norm),
        "condition_number": float(singular_values[0] / singular_values[-1]),
        "trace": trace,
    }


def _cot_sum(angles):
    return float(np.sum(1.0 / np.tan(angles / 2.0)))


def _angles_with_reciprocal_sum(x, reciprocal_sum):
    # The other two normalized angles have the given sum and product.
    total = 1.0 - x
    product = total / (reciprocal_sum - 1.0 / x)
    discriminant = max(total * total - 4.0 * product, 0.0)
    y = 2.0 * product / (total + np.sqrt(discriminant))
    return np.pi * np.array([x, y, total - y])


def _triangle(angles, perimeter, method):
    angles = np.sort(angles)
    sine = np.sin(angles)
    sides = perimeter * sine / sine.sum()
    _, middle, long = sides
    # The endpoint at the origin has the angle opposite the shortest side.
    height = middle * np.sin(angles[0])
    horizontal = middle * np.cos(angles[0])
    return {
        "status": "ok",
        "method": method,
        "angles": angles,
        "sides": sides,
        "vertices": np.array([[0.0, 0.0], [long, 0.0], [horizontal, height]]),
        "area": float(long * height / 2.0),
        "perimeter": float(perimeter),
        "gamma": float((np.pi * np.sum(1.0 / angles) - 1.0) / 24.0),
    }


def reconstruct(area, perimeter, gamma):
    """Invert compatible invariants and classify inadmissible fits explicitly."""
    if not np.isfinite([area, perimeter, gamma]).all():
        return {"status": "nonfinite_coefficients"}
    if area <= 0.0 or perimeter <= 0.0:
        return {"status": "nonpositive_area_or_perimeter"}
    if gamma < MIN_GAMMA - GAMMA_ROUNDOFF:
        return {"status": "gamma_below_equilateral"}

    target = perimeter**2 / (4.0 * area)
    if target < MIN_COT_SUM - SYMMETRIC_COT_SUM_ROUNDOFF:
        return {"status": "isoperimetric_violation"}
    if gamma <= MIN_GAMMA + GAMMA_ROUNDOFF:
        if abs(target - MIN_COT_SUM) > SYMMETRIC_COT_SUM_ROUNDOFF:
            return {"status": "incompatible_coefficients"}
        return _triangle(np.full(3, np.pi / 3.0), perimeter, "strict")

    reciprocal_sum = 24.0 * gamma + 1.0
    discriminant = np.sqrt((reciprocal_sum - 1.0) * (reciprocal_sum - 9.0))
    lower = 2.0 / (reciprocal_sum - 3.0 + discriminant)
    upper = 4.0 / (reciprocal_sum + 3.0 + discriminant)

    def residual(x):
        return _cot_sum(_angles_with_reciprocal_sum(x, reciprocal_sum)) - target

    fl, fu = residual(lower), residual(upper)
    tolerance = 3e-14 * max(1.0, abs(target))
    if abs(fl) <= tolerance:
        x = lower
    elif abs(fu) <= tolerance:
        x = upper
    elif np.sign(fl) == np.sign(fu):
        return {"status": "incompatible_coefficients"}
    else:
        x = brentq(residual, lower, upper, xtol=5e-15)
    angles = _angles_with_reciprocal_sum(x, reciprocal_sum)
    return _triangle(angles, perimeter, "strict")


def constrained_reconstruct(eigenvalues, times, *, fit=None):
    """For an infeasible fit, search the two isosceles boundaries in trace space."""
    if fit is None:
        fit = fit_trace(eigenvalues, times)
    strict = reconstruct(fit["area"], fit["perimeter"], fit["gamma"])
    if strict["status"] == "ok":
        return {**strict, "relative_residual": fit["relative_residual"]}

    design, t_min = _design(times)
    trace = fit["trace"]
    perimeter_vector = -design[:, 1] / (8.0 * np.sqrt(np.pi))

    def cost(theta):
        angles = np.array([theta, theta, np.pi - 2.0 * theta])
        gamma = (np.pi * np.sum(1.0 / angles) - 1.0) / 24.0
        area_vector = design[:, 0] / (16.0 * np.pi * _cot_sum(angles))
        offset = gamma - trace
        u, v = area_vector, perimeter_vector
        # The trace residual is p**2*u + p*v + offset, with p=P/sqrt(t_min).
        stationary = np.roots([2.0 * (u @ u), 3.0 * (u @ v), v @ v + 2.0 * (u @ offset),
                               v @ offset])
        real_roots = stationary.real[(np.abs(stationary.imag) < 1e-9) & (stationary.real > 0.0)]
        scales = np.r_[0.0, real_roots]
        residuals = scales[:, None]**2 * u + scales[:, None] * v + offset
        norms = np.linalg.norm(residuals, axis=1)
        index = np.argmin(norms)
        return float(norms[index]), float(scales[index])

    theta_min = 1e-4
    candidates = []
    for left, right in ((theta_min, np.pi / 3.0), (np.pi / 3.0, (np.pi - theta_min) / 2.0)):
        theta = np.linspace(left, right, 33)
        values = [cost(angle)[0] for angle in theta]
        candidates.extend(zip(values, theta))
        # Refine sampled local minima and both endpoint neighborhoods.
        minima = [i for i in range(1, len(theta) - 1)
                  if values[i] <= min(values[i - 1], values[i + 1])]
        for i in [0, *minima, len(theta) - 1]:
            bounds = theta[max(i - 1, 0)], theta[min(i + 1, len(theta) - 1)]
            result = minimize_scalar(lambda angle: cost(angle)[0], bounds=bounds, method="bounded",
                                     options={"xatol": 1e-12})
            candidates.append((result.fun, result.x))

    error, theta = min(candidates)
    _, scale = cost(theta)
    if scale == 0.0:
        return {"status": "degenerate_boundary_fit"}
    triangle = _triangle(np.array([theta, theta, np.pi - 2.0 * theta]), scale * np.sqrt(t_min),
                         "boundary_least_squares")
    return {**triangle, "relative_residual": float(error / np.linalg.norm(trace))}


def vertex_error(reference, prediction):
    """Mean vertex-relative error after alignment and relabeling, with no scale fit."""
    reference = np.asarray(reference, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    if reference.shape != (3, 2) or prediction.shape != (3, 2):
        raise ValueError("Both triangles must have shape (3, 2)")
    if not np.isfinite(prediction).all():
        return float("inf")
    target = reference - reference.mean(axis=0, keepdims=True)
    radii = np.linalg.norm(target, axis=-1)
    if not np.isfinite(radii).all() or np.any(radii == 0.0):
        raise ValueError("Expected a finite reference with positive centered vertex radii")
    centered = prediction - prediction.mean(axis=0, keepdims=True)
    ordered = centered[ORDERINGS]
    u, _, vt = np.linalg.svd(ordered.transpose(0, 2, 1) @ target, full_matrices=False)
    aligned = ordered @ (u @ vt)
    relative = np.linalg.norm(target - aligned, axis=-1) / radii
    return float(np.min(relative.mean(axis=1)))
