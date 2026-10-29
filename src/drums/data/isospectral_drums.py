import numpy as np

try:
    from drums.data.spectrum import get_drum_spectrum
    from drums.data.quadrilateral_drums import polygon_area, is_simple
    from drums.data.triangle_drums import random_triangle, triangle_area
    from drums.data.gww_drums import GWW, random_pose
except ImportError:  # run as a script from the data folder
    from spectrum import get_drum_spectrum
    from quadrilateral_drums import polygon_area, is_simple
    from triangle_drums import random_triangle, triangle_area
    from gww_drums import GWW, random_pose


# Exactly isospectral pairs of polygons built from an arbitrary triangle.
#
# Both domains consist of 7 copies of one triangle, any two copies that meet along an edge are
# mirror images in that edge (Buser, Conway, Doyle, Semmler, "Some planar isospectral domains",
# arXiv:1005.1839, 1994, pair 7_3, see Section 3: every such diagram "represents not a single
# pair of isospectral domains, but a whole family of pairs of isospectral domains, gotten by
# replacing the [...] triangles with general triangles"). For the right isosceles triangle with
# legs 1 the pair is the one of Gordon, Webb, Wolpert (GWW in gww_drums.py).
#
# Each domain is described by the 7 tiles (triples of indices into its 9 polygon vertices, the
# numbering is that of GWW), by the order in which the tiles are reflected out of the root tile
# 0 as pairs (tile, tile it is the mirror image of), and by the three polygon vertices of the
# root tile that carry the corners (T0, T1, T2) of the base triangle. The roles of T1 and T2 are
# exchanged in the second domain: this is the labelling in which the domains are isospectral
# (with the same labelling in both, the spectra differ by percents; the tests check it).
_TILES = (
    ((7, 4, 6), (6, 4, 5), (7, 3, 4), (7, 8, 3), (0, 1, 8), (1, 2, 3), (8, 1, 3)),
    ((8, 4, 7), (7, 5, 6), (7, 4, 5), (0, 1, 8), (8, 3, 4), (8, 1, 3), (1, 2, 3)),
)
_REFLECTIONS = (
    ((1, 0), (2, 0), (3, 2), (6, 3), (4, 6), (5, 6)),
    ((2, 0), (4, 0), (1, 2), (5, 4), (3, 5), (6, 5)),
)
_ROOT_CORNERS = ((4, 6, 7), (4, 8, 7))


def _reflect(p, a, b):
    """Mirror image of the point p in the line through a and b."""
    d = (b - a) / np.linalg.norm(b - a)
    v = p - a
    return a + 2*(v @ d)*d - v


def isospectral_pair(triangle):
    """The two isospectral polygons built from a triangle of shape (3, 2) (vertices T0, T1, T2).

    Returns (polygon_1, polygon_2), each of shape (9, 2) with counter-clockwise vertices, the
    first domain is GWW_1 and the second one is GWW_2 (up to a rigid motion) for the right
    isosceles triangle with legs 1. The polygons have the area 7 * (area of the triangle). They
    are not simple for some triangles (the boundary intersects itself), check with is_simple.
    """
    triangle = np.asarray(triangle, dtype=float)
    polygons = []
    for tiles, reflections, root in zip(_TILES, _REFLECTIONS, _ROOT_CORNERS):
        vertices = np.full((9, 2), np.nan)
        vertices[list(root)] = triangle
        for tile, parent in reflections:
            known = [v for v in tiles[tile] if v in tiles[parent]]      # the shared edge
            new = [v for v in tiles[tile] if v not in known][0]
            old = [v for v in tiles[parent] if v not in known][0]
            vertices[new] = _reflect(vertices[old], vertices[known[0]], vertices[known[1]])
        if vertices[:, 0] @ np.roll(vertices[:, 1], -1) - vertices[:, 1] @ np.roll(vertices[:, 0], -1) < 0:
            vertices = vertices[::-1]                                  # clockwise for a clockwise triangle
        polygons.append(vertices)
    return tuple(polygons)


def random_isospectral_drums(rng, r_min=0.25, r_max=1.5, angle_min=np.pi/6):
    """One of the two isospectral domains (each with probability 1/2) of a random triangle.

    The base triangle is drawn as in triangle_drums.random_triangle: inscribed in a circle of
    radius r uniform in [r_min, r_max], the interior angles are uniform on the simplex
    {sum = pi} restricted to min angle >= angle_min. Triangles for which the boundary of one of
    the two polygons intersects itself are rejected. The polygons are put in independent random
    poses (centred, rotated, random first vertex), see gww_drums.random_pose.

    Returns the vertices of the chosen domain and of its isospectral partner, both of shape
    (9, 2), and the index of the chosen domain, 0 or 1.
    """
    while True:
        triangle = random_triangle(rng, r_min, r_max, angle_min)
        pair = isospectral_pair(triangle)
        if all(is_simple(p) for p in pair):
            break
    domain = int(rng.uniform() >= 0.5)
    drum, twin = pair[domain], pair[1 - domain]
    return random_pose(rng, drum), random_pose(rng, twin), domain


def generate_isospectral_drums(n_samples, seed=0, N_eig=30, n_elements=4000,
                               r_min=0.25, r_max=1.5, angle_min=np.pi/6, return_all=False):
    """Dataset of isospectral drums: for every spectrum there are two different drums.

    A triangle is drawn at random, 7 copies of it are glued into the two isospectral domains
    of isospectral_pair, and one of them is the drum (each with probability 1/2). The two
    domains are isospectral for every triangle, so the spectrum does not tell which one is it.
    The spectra are finite-element eigenvalues on a mesh with the maximal element area
    area / n_elements for each drum. They are above the exact eigenvalues, by about 0.4% for the
    first, 0.6% for the 10th and 1.3% for the 30th eigenvalue (up to 1.6%; the error grows
    with the index and depends on the drum), so a spectrum computed with a finer mesh does not agree with the
    stored one; use a = area / n_elements to reproduce it. The two domains of a pair have the
    same exact spectrum, the stored spectrum of the twin agrees with the one of the drum
    to about 1e-3, with no systematic difference.

    Returns a dict with
        'perimeters':      (n_samples, 9, 2) vertices of the drums, counter-clockwise,
        'spectra':         (n_samples, N_eig) smallest Dirichlet eigenvalues, ascending,
        'domain':          (n_samples,) 0 if the drum is the first domain of the pair (GWW_1 like),
                           1 if it is the second one (GWW_2 like),
        'twin_perimeters': (n_samples, 9, 2) vertices of the other domain of the pair, which has
                           the same spectrum (its pose and first vertex are random, too).
    If return_all is True, the dictionary also contains lists (one entry per drum,
    the meshes differ in size) with everything get_drum_spectrum produces:
        'eigenfunctions': (N, N_eig) values at the mesh nodes, columns ordered as the spectrum,
        'triangulations': (N_triangles, 3) node indices of the triangles,
        'coordinates':    (N, 2) mesh nodes.
    """
    rng = np.random.default_rng(seed)
    perimeters, twins, spectra, domains = [], [], [], []
    extra = dict(eigenfunctions=[], triangulations=[], coordinates=[])
    for _ in range(n_samples):
        drum, twin, domain = random_isospectral_drums(rng, r_min, r_max, angle_min)
        a = polygon_area(drum) / n_elements
        lam, phi, X, T = get_drum_spectrum(drum, N_eig=N_eig, a=a)
        order = np.argsort(lam)
        perimeters.append(drum)
        twins.append(twin)
        spectra.append(lam[order])
        domains.append(domain)
        if return_all:  # only keep the mesh data if it is returned
            extra["eigenfunctions"].append(phi[:, order])
            extra["triangulations"].append(T)
            extra["coordinates"].append(X)
    data = dict(perimeters=np.stack(perimeters), spectra=np.stack(spectra), domain=np.array(domains),
                twin_perimeters=np.stack(twins))
    if return_all:
        data.update(extra)
    return data


def _congruence_distance(p, q):
    """Smallest RMS vertex distance between the polygons p and q over rotations/reflections and
    the 18 relabelings (cyclic shifts of the vertex sequence and of the reversed one)."""
    p = p - p.mean(axis=0)
    q = q - q.mean(axis=0)
    best = np.inf
    for qq in (q, q[::-1]):
        for k in range(len(q)):
            c = np.roll(qq, k, axis=0)
            U, _, Vt = np.linalg.svd(c.T @ p)
            best = min(best, np.sqrt(np.mean(np.sum((p - c @ (U @ Vt))**2, axis=1))))
    return best


if __name__ == "__main__":
    import time
    ok = True

    def check(name, cond):
        global ok
        print(f"{name}: {bool(cond)}")
        ok &= bool(cond)

    # --- the right isosceles triangle gives the GWW pair
    corner_0 = [GWW[0][v] for v in _ROOT_CORNERS[0]]
    pair = isospectral_pair(np.array(corner_0))
    for k in range(2):
        check(f"right isosceles triangle: domain {k + 1} is GWW_{k + 1} up to a rigid motion",
              _congruence_distance(pair[k], GWW[k]) < 1e-12)

    # --- the construction for general triangles
    rng = np.random.default_rng(3)
    checked, not_simple = 0, 0
    for _ in range(300):
        triangle = random_triangle(rng)
        p1, p2 = isospectral_pair(triangle)
        for p in (p1, p2):
            assert np.all(np.isfinite(p))
        ccw = lambda p: p[:, 0] @ np.roll(p[:, 1], -1) - p[:, 1] @ np.roll(p[:, 0], -1) > 0
        simple = is_simple(p1) and is_simple(p2)
        not_simple += not simple
        if simple:
            checked += 1
            ok &= bool(ccw(p1) and ccw(p2)
                       and np.isclose(polygon_area(p1), 7*triangle_area(triangle))
                       and np.isclose(polygon_area(p2), 7*triangle_area(triangle)))
    check("simple domains are counter-clockwise and have the area 7 * (area of the triangle)", ok)
    print(f"  {not_simple} of 300 random triangles give a polygon that is not simple (rejected)")
    check("most triangles are accepted", not_simple < 0.3 * 300)

    # --- isospectral but not congruent (the spectra agree up to the discretization error)
    for triangle in (random_triangle(rng), random_triangle(rng), random_triangle(rng)):
        p1, p2 = isospectral_pair(triangle)
        if not (is_simple(p1) and is_simple(p2)):
            continue
        lam = [np.sort(get_drum_spectrum(p, N_eig=30, a=polygon_area(p) / 4000)[0]) for p in (p1, p2)]
        gap = np.max(np.abs(lam[0] / lam[1] - 1))
        scale = np.sqrt(polygon_area(p1))
        print(f"  max relative difference of 30 eigenvalues {gap:.1e}, "
              f"distance between the domains / size {_congruence_distance(p1, p2) / scale:.3f}")
        check("the two domains of a pair are isospectral (rtol=2e-3)", gap < 2e-3)
    triangle = np.array([[0., 0.], [1., 0.], [0.8, 0.9]])
    p1, p2 = isospectral_pair(triangle)
    check("the pair of this triangle is not congruent", _congruence_distance(p1, p2) > 0.05)
    # control: with the same labelling of the corners in both domains the spectra differ
    saved = _ROOT_CORNERS
    _ROOT_CORNERS = (saved[0], (saved[1][0], saved[1][2], saved[1][1]))
    q1, q2 = isospectral_pair(triangle)
    _ROOT_CORNERS = saved
    lam = [np.sort(get_drum_spectrum(p, N_eig=30, a=polygon_area(p) / 4000)[0]) for p in (q1, q2)]
    check("control: the same labelling in both domains is not isospectral (> 1e-2)",
          is_simple(q1) and is_simple(q2) and np.max(np.abs(lam[0] / lam[1] - 1)) > 1e-2)

    # --- samplers and the dataset
    n_samples, N_eig = 20, 30
    t0 = time.time()
    data = generate_isospectral_drums(n_samples, seed=0, N_eig=N_eig)
    print(f"generated {n_samples} drums in {time.time() - t0:.1f} s")
    P, S, D, Q = data["perimeters"], data["spectra"], data["domain"], data["twin_perimeters"]
    check("keys", set(data) == {"perimeters", "spectra", "domain", "twin_perimeters"})
    check("shapes", P.shape == (n_samples, 9, 2) and Q.shape == P.shape and S.shape == (n_samples, N_eig)
          and D.shape == (n_samples,))
    check("finite and positive", np.all(np.isfinite(S)) and np.all(S > 0))
    check("sorted", np.all(np.diff(S, axis=1) >= 0))
    check("both domains present", set(D) == {0, 1})
    check("drums and twins are simple", all(is_simple(p) for p in P) and all(is_simple(p) for p in Q))
    check("a drum and its twin have the same area", np.allclose([polygon_area(p) for p in P], [polygon_area(q) for q in Q]))
    check("a drum and its twin are not congruent (except for rare symmetric triangles)",
          np.mean([_congruence_distance(p, q) > 1e-3 for p, q in zip(P, Q)]) > 0.8)
    twin_spectra = [np.sort(get_drum_spectrum(q, N_eig=N_eig, a=polygon_area(q) / 4000)[0]) for q in Q[:5]]
    check("the spectrum of the twin is the spectrum of the drum (rtol=2e-3)",
          np.allclose(twin_spectra, S[:5], rtol=2e-3))
    check("reproducible", np.allclose(generate_isospectral_drums(2, seed=0, N_eig=N_eig)["spectra"], S[:2]))
    samples = [random_isospectral_drums(np.random.default_rng(i)) for i in range(400)]
    frac = np.mean([d for _, _, d in samples])
    check(f"probability of the second domain is 1/2 (observed {frac:.3f})", abs(frac - 0.5) < 0.1)
    check("first vertex is random", len({int(np.argmin(p[:, 1])) for p, _, _ in samples}) == 9)

    full = generate_isospectral_drums(2, seed=0, N_eig=N_eig, return_all=True)
    phi, X, T = full["eigenfunctions"][0], full["coordinates"][0], full["triangulations"][0]
    check("return_all keys", set(full) == {"perimeters", "spectra", "domain", "twin_perimeters", "eigenfunctions",
                                          "triangulations", "coordinates"})
    check("return_all spectra unchanged", np.allclose(full["spectra"], S[:2]) and np.allclose(full["perimeters"], P[:2]))
    check("return_all sizes", len(full["eigenfunctions"]) == len(full["triangulations"]) == len(full["coordinates"]) == 2
          and phi.shape == (len(X), N_eig) and T.shape[1] == 3)
    assert ok, "isospectral_drums test failed"
