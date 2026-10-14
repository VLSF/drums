import numpy as np


def align(vertices, rtol=1e-9):
    """Shift and rotate a drum: zero mean vertex, (first) longest vertex pointing up (+y)."""
    vertices_ = vertices - np.mean(vertices, axis=0, keepdims=True)
    r = np.linalg.norm(vertices_, axis=1)
    # first vertex among the (numerically) longest ones, robust to round-off ties
    ind = np.flatnonzero(r >= r.max() * (1 - rtol))[0]
    # angle of the longest vertex measured from +y; rotating by it maps the vertex onto +y
    phi = np.arctan2(vertices_[ind, 0], vertices_[ind, 1])
    return np.stack([
        vertices_[:, 0]*np.cos(phi) - vertices_[:, 1]*np.sin(phi),
        vertices_[:, 0]*np.sin(phi) + vertices_[:, 1]*np.cos(phi)
    ], axis=1)


def augment(perimeter):
    """All cyclic permutations of the vertex sequence and of the reversed sequence.

    Returns an array of shape (2n, n, 2).
    """
    perimeters = []
    for i in range(perimeter.shape[0]):
        perimeters.append(np.roll(perimeter, i, axis=0))
        perimeters.append(np.roll(perimeter[::-1], i, axis=0))
    return np.stack(perimeters, axis=0)


if __name__ == "__main__":
    try:
        from drums.data.spectrum import get_drum_spectrum
    except ImportError:
        from spectrum import get_drum_spectrum

    ok = True

    def check(name, cond):
        global ok
        ok &= bool(cond)
        print(f"{name}: {bool(cond)}")

    # a non-convex, asymmetric polygon (counter-clockwise)
    P = np.array([[0, 0], [3, 0], [3, 1], [1, 1], [1, 2], [0, 2.5]], dtype=float)

    # --- align ---
    A = align(P)
    r = np.linalg.norm(A, axis=1)
    k = np.argmax(r)
    print("aligned:\n", A)
    check("align: mean is zero", np.allclose(A.mean(axis=0), 0))
    check("align: longest vertex points up", np.allclose(A[k], [0, r[k]]) and k == np.argmax(np.linalg.norm(P - P.mean(0), axis=1)))
    check("align: distances preserved", np.allclose(r, np.linalg.norm(P - P.mean(0), axis=1)))
    check("align: idempotent", np.allclose(align(A), A))
    th = 0.7
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    check("align: shift/rotation invariant", np.allclose(align(P @ R.T + [5.0, -2.0]), A))
    sq = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    check("align: first of tied longest vertices is up", np.allclose(align(sq)[0], [0, np.sqrt(0.5)]))
    check("align: vertex on the x axis", np.allclose(align(np.array([[2.0, 0], [-1, 1], [-1, -1]]))[0], [0, 2]))

    # --- augment ---
    seq = np.arange(3)[:, None] * np.ones((1, 2))
    S = augment(seq)[..., 0].astype(int).tolist()
    expected = [[0, 1, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0], [1, 0, 2], [0, 2, 1]]
    print("augment of [0, 1, 2]:", S)
    check("augment: shape", augment(P).shape == (12, 6, 2))
    check("augment: all permutations present", sorted(S) == sorted(expected))
    check("augment: unique", len({a.tobytes() for a in augment(P)}) == 12)

    # --- spectrum invariance ---
    n_eig = 8
    lam0 = np.sort(get_drum_spectrum(P, N_eig=n_eig, a=3e-3)[0])
    cases = {"aligned": A, "shifted+rotated": P @ R.T + [5.0, -2.0]}
    for i, Q in enumerate(augment(P)):
        cases[f"augmented {i}"] = Q
    for name, Q in cases.items():
        lam = np.sort(get_drum_spectrum(Q, N_eig=n_eig, a=3e-3)[0])
        close = np.allclose(lam, lam0, rtol=1e-2)
        ok &= bool(close)
        if name in ("aligned", "shifted+rotated", "augmented 0", "augmented 3"):
            print(f"{name}\n  computed: {lam}\n  reference: {lam0}")
        print(f"spectrum invariance, {name}: {close}")
    assert ok, "preprocessing test failed"
