"""Control dataset without ambiguity: the same 30k spectra, but the label of every spectrum is a canonical drum.

The index `domain` of a sample is NOT a property of the shape: the corners of the base triangle are labelled
at random, and relabelling T0 <-> T2 gives the same pair of drums with the two names exchanged. So "always
take domain 0" would still be a coin flip. The canonical choice used here: label the corners so that the angle
at T0 is not smaller than the angle at T2 and take the first domain of that pair. It is continuous in the
shape (for equal angles the two drums are congruent) and it is a function of the spectrum.

For every sample the labelled triangle is recovered from the polygon: the stored polygon is
pair[domain](T) after a random cyclic shift, so we try the 9 shifts, read T at the root corners, rebuild
the polygon and compare. Then the label is the stored drum if (domain == 0) == (angle(T0) >= angle(T2)),
and its twin otherwise. The spectra are not changed (the spectrum of a drum equals that of its twin up to the
mesh noise of ~2e-4). `twin_perimeters` becomes the other drum and `domain` is set to 0.

    python make_single_domain.py isospectral_30k.npz isospectral_30k_single.npz
"""
import sys
import numpy as np
from drums.data.isospectral_drums import isospectral_pair, _ROOT_CORNERS


def fit_residual(p, q):
    """Smallest RMS distance between the centred polygons p and q (same vertex labelling) over rotations/reflections."""
    p, q = p - p.mean(0), q - q.mean(0)
    U, _, Vt = np.linalg.svd(q.T @ p)
    return np.sqrt(np.mean(np.sum((p - q @ (U @ Vt))**2, axis=1)))


def angle(a, b, c):
    """Angle at the vertex a of the triangle (a, b, c)."""
    u, v = b - a, c - a
    return np.arccos(np.clip(u @ v / np.linalg.norm(u) / np.linalg.norm(v), -1, 1))


def recover_triangle(polygon, domain):
    """The polygon is rolled by a random shift, and isospectral_pair reverses the vertex order if the polygon
    came out clockwise, so the corners of the triangle are read from the shifted polygon or from its reversal."""
    for s in range(9):
        q = np.roll(polygon, -s, axis=0)
        for original in (q, q[::-1]):
            triangle = original[list(_ROOT_CORNERS[domain])]
            if fit_residual(q, isospectral_pair(triangle)[domain]) < 1e-8:
                return triangle
    return None


if __name__ == "__main__":
    d = dict(np.load(sys.argv[1]))
    P, Q, dom = d["perimeters"], d["twin_perimeters"], d["domain"]
    keep = np.zeros(len(P), dtype=bool)
    failed, flips = [], 0
    for i in range(len(P)):
        T = recover_triangle(P[i], dom[i])
        if T is None:
            failed.append(i)
            continue
        keep[i] = (dom[i] == 0) == (angle(T[0], T[1], T[2]) >= angle(T[2], T[0], T[1]))
    assert not failed, f"could not recover the triangle of {len(failed)} samples, e.g. {failed[:5]}"
    P_new, Q_new = np.where(keep[:, None, None], P, Q), np.where(keep[:, None, None], Q, P)
    d.update(perimeters=P_new, twin_perimeters=Q_new, domain=np.zeros_like(dom))
    np.savez(sys.argv[2], **d)
    print(f"the stored drum is the label for {keep.sum()} of {len(P)} samples, the twin for the others")
