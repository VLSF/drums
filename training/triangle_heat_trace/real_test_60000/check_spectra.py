"""Validate agreement of numerical Dirichlet spectra and overlapping solver windows."""

import numpy as np


def _sorted_vector(values):
    points = np.asarray(values, dtype=float)
    if points.ndim != 1 or not np.all(np.isfinite(points)) or np.any(np.diff(points) < 0):
        raise ValueError('Expected a sorted finite one-dimensional array')
    return points


def _gap_nearest(target, low, high, joined):
    """Choose a cut between candidates, avoiding inconsistent endpoint rounding."""
    candidates = joined[(joined > low) & (joined < high)]
    boundaries = np.r_[low, candidates, high]
    size = np.diff(boundaries)
    centers = (boundaries[:-1] + boundaries[1:]) / 2
    best = np.lexsort((np.abs(centers - target), -size))[0]
    return centers[best], size[best]


def stitch_windows(windows, eigenfrequency_atol, lower=0.0, upper=None):
    """Assemble sorted eigenfrequencies, retaining duplicates as multiplicities.

    Each ascending window is (center, radius, candidate eigenfrequencies).
    Returns the spectrum and a disagreement list for overlap intervals.
    Do not interpret the result as certified when that list is nonempty.
    """
    windows = [(center, radius, _sorted_vector(values)) for center, radius, values in windows]
    if not windows or eigenfrequency_atol <= 0.0:
        raise ValueError('Expected a positive tolerance and at least one window')
    if upper is None:
        upper = windows[-1][0] + windows[-1][1]
    if lower < windows[0][0] - windows[0][1] or upper > windows[-1][0] + windows[-1][1]:
        raise ValueError('Requested interval extends beyond outer windows')

    cuts, disagreements = [lower], []
    for left, right in zip(windows[:-1], windows[1:]):
        lc, lr, lo = left
        rc, rr, ro = right
        low, high = max(lc - lr, rc - rr), min(lc + lr, rc + rr)
        if lc >= rc or low >= high:
            raise ValueError('Windows must have strictly increasing centers and positive overlaps')
        width = high - low
        joined = np.sort(np.r_[lo, ro])
        cut, gap_width = _gap_nearest((low + high) / 2, low + .25 * width, high - .25 * width, joined)
        begin, _ = _gap_nearest(low, low, low + .2 * width, joined)
        end, _ = _gap_nearest(high, high - .2 * width, high, joined)
        a, b = (values[(values >= begin) & (values < end)] for values in (lo, ro))
        discrepancy = np.max(np.abs(a - b), initial=0.0) if len(a) == len(b) else np.inf
        if discrepancy > eigenfrequency_atol or gap_width < 2 * eigenfrequency_atol:
            disagreements.append(dict(left=lc, right=rc, interval=(begin, end), seam=cut,
                                      counts=(len(a), len(b)), max_difference=discrepancy))
        cuts.append(cut)
    cuts.append(upper)
    if np.any(np.diff(cuts) <= 0):
        raise ValueError('Requested window boundaries do not form increasing cuts')
    pieces = [values[(values >= begin) & (values < end)]
              for (_, _, values), begin, end in zip(windows, cuts[:-1], cuts[1:])]
    return np.concatenate(pieces), disagreements


def _trace_design(times):
    times = np.asarray(times, dtype=float)
    u = times / np.min(times)
    columns = np.column_stack((1 / u, 1 / np.sqrt(u), np.ones(len(u))))
    return columns, np.linalg.pinv(columns)


def compare_spectra(first, second, times=None):
    """Compare two equal-length sorted physical eigenvalue spectra.

    The fitted quantity discrepancies are second minus first.  The trace
    difference is accumulated using expm1 to resolve small differences.
    """
    first, second = _sorted_vector(first), _sorted_vector(second)
    if len(first) == 0 or len(first) != len(second) or min(first[0], second[0]) <= 0:
        raise ValueError('Expected equal-length, nonempty positive eigenvalue arrays')
    if times is None:
        times = np.geomspace(1e-4, 2e-4, num=100)
    times = np.asarray(times, dtype=float)
    difference = second - first
    trace_change = np.zeros(len(times))
    for start in range(0, len(first), 8192):
        section = slice(start, start + 8192)
        smaller = np.minimum(first[section], second[section])
        delta = np.exp(-np.outer(times, smaller)) * np.expm1(-np.outer(times, np.abs(difference[section])))
        trace_change += delta @ np.sign(difference[section])
    _, pinv = _trace_design(times)
    delta_b = pinv @ trace_change
    return {
        'modes': len(first),
        'eigenvalue_max_abs_difference': float(np.max(np.abs(difference))),
        'wavenumber_max_abs_difference': float(np.max(np.abs(np.sqrt(second) - np.sqrt(first)))),
        'trace_max_abs_difference': float(np.max(np.abs(trace_change))),
        'estimated_area_change': float(delta_b[0] * 4 * np.pi * np.min(times)),
        'estimated_perimeter_change': float(delta_b[1] * -8 * np.sqrt(np.pi * np.min(times))),
        'estimated_gamma_change': float(delta_b[2]),
    }


