"""Refine a scaling window in the small candidate subspace."""

import numpy as np

from scaling import quadrature


def project_taylor(solver, k, values, derivatives, vectors, order=8, basis_indices=None):
    """Project Bessel's recurrence before constructing the higher derivatives."""
    nodes, _ = quadrature(values.shape[1])
    xy = solver.left + nodes[:, None] * (solver.right - solver.left)
    r_squared = np.sum(xy**2, axis=1)[:, None]
    modes = np.arange(1., values.shape[0]+1.) if basis_indices is None else basis_indices+1.
    orders = solver.nu * modes
    q = (orders/k)**2
    max_power = order//2
    weighted = np.concatenate([vectors * q[:, None]**j
                               for j in range(max_power+1)], axis=1)
    n_candidates = vectors.shape[1]
    p = [list(np.split(values.T @ weighted, max_power+1, axis=1)),
         list(np.split(derivatives.T @ weighted[:, :max_power*n_candidates], max_power, axis=1))]
    for n in range(order-1):
        row = []
        k_linear = (n+1)*(2*n+1)/k
        r_term = n*n/(k*k)+r_squared
        for power in range((order-n-2)//2+1):
            numerator = k_linear*p[n+1][power] + r_term*p[n][power] - p[n][power+1]
            if n >= 1:
                numerator += 2*r_squared*p[n-1][power]/k
            if n >= 2:
                numerator += r_squared*p[n-2][power]/(k*k)
            row.append(-numerator/((n+1)*(n+2)))
        p.append(row)
    return [row[0] for row in p]


def value_and_derivatives(polynomial, delta):
    value = polynomial[-1]
    derivative = np.zeros_like(value)
    curvature = np.zeros_like(value)
    for coefficient in polynomial[-2::-1]:
        curvature = curvature*delta + 2*derivative
        derivative = derivative*delta + value
        value = value*delta + coefficient
    return value, derivative, curvature


