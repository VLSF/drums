"""Select and refine roots of the boundary scaling pencil."""

import numpy as np
from scipy import linalg
from scipy.linalg.lapack import dpstrf

from projected_fast import project_taylor, value_and_derivatives


def projected(solver, k, radius):
    values, derivatives = solver.evaluation(k)
    gram = np.asfortranarray(values @ values.T)
    tol = gram.diagonal().max()*solver.gram_tol
    factor, pivots, rank, _ = dpstrf(gram, lower=True, tol=tol, overwrite_a=True)
    if rank == 0:
        return np.empty(0), False
    selected = pivots[:rank] - 1
    factor = np.asfortranarray(factor[:rank, :rank])
    values, derivatives = values[selected], derivatives[selected]
    rate_form = values @ derivatives.T
    rate_form += rate_form.T.copy()
    white = linalg.solve_triangular(factor, rate_form, lower=True, check_finite=False)
    white = linalg.solve_triangular(factor, white.T, lower=True, check_finite=False).T
    white = (white + white.T)/2
    rates, vectors = linalg.eigh(white, driver="evd", check_finite=False, overwrite_a=True)
    keep = np.abs(rates) >= 2/(radius + .07)
    raw = -2/rates[keep]
    if len(raw) == 0:
        return raw, False
    vectors = linalg.solve_triangular(factor, vectors[:, keep], lower=True,
                                      trans="T", check_finite=False)
    # The selected basis may be noncontiguous after the pivoted Cholesky.
    polynomial = project_taylor(solver, k, values, derivatives, vectors,
                                basis_indices=selected)
    delta = raw[None, :].copy()
    for _ in range(3):
        v, d, h = value_and_derivatives(polynomial, delta)
        delta -= np.sum(d*v, axis=0)/np.sum(d*d+h*v, axis=0)
    v, d, _ = value_and_derivatives(polynomial, delta)
    refined = delta.ravel()
    residual = np.linalg.norm(v, axis=0)/np.linalg.norm(d, axis=0)
    core = (np.abs(raw) <= radius) | (np.abs(refined) <= radius)
    needs_repair = np.any(core & ((residual > .008) | (np.abs(raw-refined) > .035)
                                  | (np.abs(raw) < .00010) | ~np.isfinite(refined)))
    return k+np.sort(refined[np.abs(refined) <= radius]), bool(needs_repair)


def gap_cut(left, right, center, radius):
    ends = np.r_[center-radius, center+radius]
    both = np.concatenate((left, right))
    both = both[(both > ends[0]) & (both < ends[1])]
    candidates = np.sort(np.r_[ends, both])
    i = np.argmax(np.diff(candidates))
    return (candidates[i] + candidates[i+1])/2


def window(solver, center, radius, depth=0, force=False):
    """Recenter potentially ill-conditioned windows while preserving multiplicity."""
    root, repair = projected(solver, center, radius)
    if depth == 2 or (not repair and not force):
        return root, int(repair), 0
    shift = radius*.47
    sub_radius = radius*.70
    left, left_unresolved, left_redone = window(solver, center-shift, sub_radius, depth+1)
    right, right_unresolved, right_redone = window(solver, center+shift, sub_radius, depth+1)
    cut = gap_cut(left, right, center, radius*.12)
    repaired = np.r_[left[(left >= center-radius) & (left < cut)],
                     right[(right >= cut) & (right < center+radius)]]
    return repaired, left_unresolved + right_unresolved, 1 + left_redone + right_redone
