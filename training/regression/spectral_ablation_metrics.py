"""Shape metrics modulo rigid motions and cyclic/reversed vertex order.

Distances retain size and compare four boundary vertices, including the inserted
midpoint for triangles. They are representation-sensitive, not boundary distances.
FM diversity measures generated spread; it does not establish posterior calibration.
"""
import numpy as np


def dihedral_orderings(n=4):
    """All cyclic shifts and reversals; arbitrary permutations change a polygon."""
    indices = np.arange(n)
    return np.array([np.roll(sequence, shift)
                     for sequence in (indices, indices[::-1]) for shift in range(n)])


def align_to(reference, predictions):
    """Center and optimally rotate/reflect (..., n, 2) predictions onto reference."""
    reference = np.asarray(reference, dtype=float)
    predictions = np.asarray(predictions, dtype=float)
    if reference.ndim != 2 or reference.shape[1] != 2 or predictions.shape[-2:] != reference.shape:
        raise ValueError("Expected reference (n, 2) and predictions (..., n, 2)")
    target = reference - reference.mean(axis=0)
    flat = predictions.reshape(-1, *reference.shape)
    aligned = np.full_like(flat, np.nan)
    finite = np.isfinite(flat).all(axis=(1, 2))
    if finite.any() and np.isfinite(target).all():
        centered = flat[finite] - flat[finite].mean(axis=1, keepdims=True)
        candidates = centered[:, dihedral_orderings(len(reference)), :]
        u, _, vt = np.linalg.svd(candidates.swapaxes(-1, -2) @ target)
        candidates = candidates @ (u @ vt)
        best = np.argmin(np.sum((candidates - target) ** 2, axis=(-2, -1)), axis=1)
        aligned[finite] = candidates[np.arange(len(candidates)), best]
    return aligned.reshape(predictions.shape)


def shape_errors(target, samples):
    """Relative Frobenius error after alignment, retaining target size."""
    target = np.asarray(target, dtype=float)
    target = target - target.mean(axis=0)
    scale = np.linalg.norm(target)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Target must be finite with nonzero size")
    errors = np.linalg.norm(align_to(target, samples) - target, axis=(-2, -1)) / scale
    return np.where(np.isfinite(errors), errors, np.inf)


def mean_shape(samples, n_iterations=5):
    """Generalized Procrustes mean, computed without the ground truth."""
    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 3 or len(samples) == 0:
        raise ValueError("Expected a nonempty (draws, n, 2) array")
    if not np.isfinite(samples).all():
        return np.full(samples.shape[1:], np.nan)
    reference = samples[0] - samples[0].mean(axis=0)
    for _ in range(n_iterations):
        reference = align_to(reference, samples).mean(axis=0)
    return reference - reference.mean(axis=0)


def is_valid(polygon):
    """Finite simple polygon of positive area; adjacent collinear edges are allowed."""
    polygon = np.asarray(polygon, dtype=float)
    if polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3 or not np.isfinite(polygon).all():
        return False
    polygon = polygon - polygon.mean(axis=0)
    scale = np.linalg.norm(polygon, axis=1).max()
    if scale == 0:
        return False
    tolerance = 1e-10 * scale ** 2
    cross = lambda a, b: a[0] * b[1] - a[1] * b[0]
    n = len(polygon)
    for i in range(n):
        if np.any(np.linalg.norm(polygon[i + 1:] - polygon[i], axis=1) <= 1e-8 * scale):
            return False
    if abs(sum(cross(polygon[i], polygon[(i + 1) % n]) for i in range(n))) <= tolerance:
        return False

    def intersects(a, b, c, d):
        signs = [cross(b - a, c - a), cross(b - a, d - a),
                 cross(d - c, a - c), cross(d - c, b - c)]
        if signs[0] * signs[1] < -tolerance ** 2 and signs[2] * signs[3] < -tolerance ** 2:
            return True
        for value, point, start, end in ((signs[0], c, a, b), (signs[1], d, a, b),
                                          (signs[2], a, c, d), (signs[3], b, c, d)):
            if abs(value) <= tolerance and np.dot(point - start, point - end) <= tolerance:
                return True
        return False

    for i in range(n):
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue
            if intersects(polygon[i], polygon[(i + 1) % n], polygon[j], polygon[(j + 1) % n]):
                return False
    return True


def distribution_metrics(target, samples):
    """Point, sample, oracle-best, spread and validity metrics for one drum.

    Diversity is the mean pairwise aligned distance divided by target Frobenius
    size. Best-of-K uses the target to select a draw and is an oracle diagnostic.
    """
    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 3 or len(samples) == 0:
        raise ValueError("Expected a nonempty (draws, n, 2) array")
    errors = shape_errors(target, samples)
    scale = np.linalg.norm(np.asarray(target) - np.asarray(target).mean(axis=0))
    pairwise = []
    for i in range(len(samples) - 1):
        reference = samples[i] - samples[i].mean(axis=0)
        pairwise.extend(np.linalg.norm(align_to(reference, samples[i + 1:]) - reference,
                                       axis=(-2, -1)) / scale)
    diversity = float(np.mean(pairwise)) if pairwise else 0.0
    return dict(mean_draw_error=float(errors.mean()),
                mean_shape_error=float(shape_errors(target, mean_shape(samples))),
                best_of_k_error=float(errors.min()),
                diversity=diversity if np.isfinite(diversity) else float("inf"),
                validity_fraction=float(np.mean([is_valid(draw) for draw in samples])))
