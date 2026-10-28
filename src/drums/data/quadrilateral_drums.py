import numpy as np

try:
    from drums.data.spectrum import get_drum_spectrum
except ImportError:  # run as a script from the data folder
    from spectrum import get_drum_spectrum


def polygon_area(polygon):
    """Area of a simple polygon (shoelace formula), positive for any orientation."""
    x, y = polygon[:, 0], polygon[:, 1]
    return 0.5 * np.abs(x @ np.roll(y, -1) - y @ np.roll(x, -1))


def interior_angles(polygon):
    """Interior angles in (0, 2 pi) of a simple counter-clockwise polygon; > pi at reflex vertices."""
    u = np.roll(polygon, -1, axis=0) - polygon   # to the next vertex
    v = np.roll(polygon, 1, axis=0) - polygon    # to the previous vertex
    cross = u[:, 0]*v[:, 1] - u[:, 1]*v[:, 0]
    return np.mod(np.arctan2(cross, np.sum(u*v, axis=1)), 2*np.pi)


def _ccw_order(points):
    """Indices that sort the points counter-clockwise around their centroid."""
    c = points.mean(axis=0)
    return np.argsort(np.arctan2(points[:, 1] - c[1], points[:, 0] - c[0]))


def _in_triangle(p, a, b, c):
    def side(p, q, r):
        return (q[0]-p[0])*(r[1]-p[1]) - (q[1]-p[1])*(r[0]-p[0])
    s = np.array([side(a, b, p), side(b, c, p), side(c, a, p)])
    return np.all(s > 0) or np.all(s < 0)


def random_quadrilateral(rng, convex=True, r_min=0.5, r_max=3.0, angle_min=np.pi/6,
                         convexity_margin=0.1, edge_ratio_min=0.2):
    """Random simple quadrilateral with the centroid of the vertices at the origin.

    Four points are drawn uniformly in a disc. If they are in convex position, the
    quadrilateral is their convex hull. Otherwise one point lies in the triangle of the
    other three and is inserted into a uniformly chosen edge of it, which gives a
    non-convex quadrilateral with exactly one reflex vertex. Samples of the wrong
    class are rejected, and so are those violating any of
        interior angles in [angle_min, 2 pi - angle_min],
        convex:     all angles <= pi - convexity_margin,
        non-convex: the reflex angle >= pi + convexity_margin,
        shortest edge >= edge_ratio_min * longest edge.
    The shape is scaled so that the largest distance from a vertex to the origin is
    uniform in [r_min, r_max]. Vertices are counter-clockwise, the first one is random.
    """
    while True:
        rho, phi = np.sqrt(rng.uniform(size=4)), rng.uniform(0, 2*np.pi, size=4)
        p = np.stack([rho*np.cos(phi), rho*np.sin(phi)], axis=1)
        inside = [k for k in range(4) if _in_triangle(p[k], *np.delete(p, k, axis=0))]
        if inside:
            if convex:
                continue
            outer = np.delete(p, inside[0], axis=0)
            outer = outer[_ccw_order(outer)]
            j = rng.integers(3)  # the interior point goes between vertices j and j+1
            polygon = np.insert(outer, j + 1, p[inside[0]], axis=0)
        else:
            if not convex:
                continue
            polygon = p[_ccw_order(p)]

        angles = interior_angles(polygon)
        edges = np.linalg.norm(np.roll(polygon, -1, axis=0) - polygon, axis=1)
        if angles.min() < angle_min or angles.max() > 2*np.pi - angle_min:
            continue
        if convex and angles.max() > np.pi - convexity_margin:
            continue
        if not convex and angles.max() < np.pi + convexity_margin:
            continue
        if edges.min() < edge_ratio_min * edges.max():
            continue
        break

    polygon = polygon - polygon.mean(axis=0)
    polygon *= rng.uniform(r_min, r_max) / np.linalg.norm(polygon, axis=1).max()
    return np.roll(polygon, rng.integers(4), axis=0)


def is_convex(polygon, tol=1e-12):
    """True if all interior angles of a counter-clockwise polygon are at most pi."""
    return bool(np.all(interior_angles(polygon) <= np.pi + tol))


def is_simple(polygon):
    """True if no two non-adjacent edges of the polygon intersect."""
    def ccw(a, b, c):
        return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])
    n = len(polygon)
    for i in range(n):
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:  # adjacent through the closing edge
                continue
            a, b = polygon[i], polygon[(i+1) % n]
            c, d = polygon[j], polygon[(j+1) % n]
            if ccw(a, b, c)*ccw(a, b, d) < 0 and ccw(c, d, a)*ccw(c, d, b) < 0:
                return False
    return True


def generate_quadrilateral_drums(n_samples, seed=0, N_eig=30, n_elements=4000,
                                 r_min=0.5, r_max=3.0, angle_min=np.pi/6,
                                 convexity_margin=0.1, edge_ratio_min=0.2,
                                 nonconvex_fraction=0.0, return_all=False):
    """Dataset of quadrilateral drums, convex only or a mixture of convex and non-convex.

    Each drum is non-convex with probability nonconvex_fraction, independently of the
    others (0: only convex drums, 0.5: an even mixture on average). The maximal element
    area of the mesh is area / n_elements for each drum, so all drums are resolved
    comparably, independently of their size.

    Returns a dict with
        'perimeters': (n_samples, 4, 2) vertices of the quadrilaterals, counter-clockwise,
        'spectra':    (n_samples, N_eig) smallest Dirichlet eigenvalues, ascending,
        'is_convex':  (n_samples,) bool.
    If return_all is True, the dictionary also contains lists (one entry per drum,
    the meshes differ in size) with everything get_drum_spectrum produces:
        'eigenfunctions': (N, N_eig) values at the mesh nodes, columns ordered as the spectrum,
        'triangulations': (N_triangles, 3) node indices of the triangles,
        'coordinates':    (N, 2) mesh nodes.
    """
    rng = np.random.default_rng(seed)
    perimeters, spectra, convex_flags = [], [], []
    extra = dict(eigenfunctions=[], triangulations=[], coordinates=[])
    for _ in range(n_samples):
        convex = rng.uniform() >= nonconvex_fraction
        quad = random_quadrilateral(rng, convex, r_min, r_max, angle_min,
                                    convexity_margin, edge_ratio_min)
        a = polygon_area(quad) / n_elements
        lam, phi, X, T = get_drum_spectrum(quad, N_eig=N_eig, a=a)
        order = np.argsort(lam)
        perimeters.append(quad)
        spectra.append(lam[order])
        convex_flags.append(convex)
        if return_all:  # only keep the mesh data if it is returned
            extra["eigenfunctions"].append(phi[:, order])
            extra["triangulations"].append(T)
            extra["coordinates"].append(X)
    data = dict(perimeters=np.stack(perimeters), spectra=np.stack(spectra),
                is_convex=np.array(convex_flags))
    if return_all:
        data.update(extra)
    return data


if __name__ == "__main__":
    import time
    ok = True

    def check(name, cond):
        global ok
        print(f"{name}: {bool(cond)}")
        ok &= bool(cond)

    # --- rectangles [0, p] x [0, q]: lambda = pi^2 (m^2/p^2 + n^2/q^2)
    for name, p, q in [("unit square", 1.0, 1.0), ("2x1 rectangle", 2.0, 1.0)]:
        rect = np.array([[0, 0], [p, 0], [p, q], [0, q]], dtype=float)
        lam, _, _, _ = get_drum_spectrum(rect, N_eig=10, a=polygon_area(rect) / 4000)
        lam = np.sort(lam)
        exact = np.sort([np.pi**2 * (m**2/p**2 + n**2/q**2) for m in range(1, 20) for n in range(1, 20)])[:10]
        print(f"{name}\n  computed: {lam}\n  expected: {exact}")
        check(f"{name} allclose (rtol=1e-2)", np.allclose(lam, exact, rtol=1e-2))

    # --- samplers
    rng = np.random.default_rng(1)
    convex_q = [random_quadrilateral(rng, True) for _ in range(300)]
    dart_q = [random_quadrilateral(rng, False) for _ in range(300)]
    check("convex sampler: convex, simple, ccw", all(is_convex(q) and is_simple(q) for q in convex_q))
    check("non-convex sampler: non-convex, simple, ccw",
          all(not is_convex(q) and is_simple(q) and len(np.flatnonzero(interior_angles(q) > np.pi)) == 1
              for q in dart_q))
    check("signed area positive (ccw)", all(np.sum(interior_angles(q)) > 2*np.pi - 1e-9 for q in convex_q + dart_q))
    angles = np.array([interior_angles(q) for q in convex_q + dart_q])
    check("angles in [30 deg, 330 deg]", angles.min() >= np.pi/6 - 1e-9 and angles.max() <= 11*np.pi/6 + 1e-9)
    check("convex angles <= pi - 0.1", max(interior_angles(q).max() for q in convex_q) <= np.pi - 0.1 + 1e-9)
    check("reflex angle >= pi + 0.1", min(interior_angles(q).max() for q in dart_q) >= np.pi + 0.1 - 1e-9)
    radii = np.array([np.linalg.norm(q, axis=1).max() for q in convex_q + dart_q])
    check("max vertex radius in [0.5, 3]", radii.min() >= 0.5 - 1e-9 and radii.max() <= 3 + 1e-9)
    check("centroid at the origin", all(np.allclose(q.mean(axis=0), 0) for q in convex_q + dart_q))
    check("first vertex is random", len({int(np.argmax(interior_angles(q))) for q in dart_q}) == 4)

    # --- dart: the spectrum does not depend on the position, orientation, labelling
    dart = dart_q[0]
    c, s = np.cos(0.7), np.sin(0.7)
    R = np.array([[c, -s], [s, c]])
    variants = {"rotated + shifted": dart @ R.T + [1.5, -2.0],
                "reflected + reversed": (dart * [-1, 1])[::-1],
                "cyclic shift": np.roll(dart, 2, axis=0)}
    lam0, _, _, _ = get_drum_spectrum(dart, N_eig=10, a=polygon_area(dart) / 4000)
    for name, v in variants.items():
        lam, _, _, _ = get_drum_spectrum(v, N_eig=10, a=polygon_area(v) / 4000)
        check(f"dart {name}: same spectrum (rtol=1e-2)", np.allclose(np.sort(lam), np.sort(lam0), rtol=1e-2))

    # --- mesh convergence for the worst case: the sampled dart with the deepest reflex vertex
    deep = max(dart_q, key=lambda q: interior_angles(q).max())
    print(f"deepest reflex angle: {np.degrees(interior_angles(deep).max()):.0f} deg")
    lams = {}
    for ne in (4000, 16000, 64000):
        lams[ne], _, _, _ = get_drum_spectrum(deep, N_eig=10, a=polygon_area(deep) / ne)
        lams[ne] = np.sort(lams[ne])
    err_coarse = np.max(np.abs(lams[4000] / lams[64000] - 1))
    err_mid = np.max(np.abs(lams[16000] / lams[64000] - 1))
    print(f"dart mesh convergence: max relative error w.r.t. n_elements=64000 is "
          f"{err_coarse:.2e} (4000), {err_mid:.2e} (16000)")
    check("default mesh resolves the dart to 1%", err_coarse < 1e-2)

    # --- dataset
    n_samples, N_eig = 20, 30
    t0 = time.time()
    data = generate_quadrilateral_drums(n_samples, seed=0, N_eig=N_eig, nonconvex_fraction=0.5)
    print(f"generated {n_samples} drums in {time.time() - t0:.1f} s")
    P, S, C = data["perimeters"], data["spectra"], data["is_convex"]
    check("keys", set(data) == {"perimeters", "spectra", "is_convex"})
    check("shapes", P.shape == (n_samples, 4, 2) and S.shape == (n_samples, N_eig) and C.shape == (n_samples,))
    check("finite and positive", np.all(np.isfinite(S)) and np.all(S > 0))
    check("sorted", np.all(np.diff(S, axis=1) >= 0))
    check("is_convex matches the geometry", all(is_convex(q) == c for q, c in zip(P, C)))
    check("mixture contains both classes", 0 < C.sum() < n_samples)
    check("reproducible", np.allclose(generate_quadrilateral_drums(
        2, seed=0, N_eig=N_eig, nonconvex_fraction=0.5)["spectra"], S[:2]))
    only_convex = generate_quadrilateral_drums(5, seed=0, N_eig=N_eig)
    check("default: only convex drums", only_convex["is_convex"].all() and all(is_convex(q) for q in only_convex["perimeters"]))

    full = generate_quadrilateral_drums(2, seed=0, N_eig=N_eig, nonconvex_fraction=0.5, return_all=True)
    phi, X, T = full["eigenfunctions"][0], full["coordinates"][0], full["triangulations"][0]
    check("return_all keys", set(full) == {"perimeters", "spectra", "is_convex", "eigenfunctions",
                                          "triangulations", "coordinates"})
    check("return_all spectra unchanged", np.allclose(full["spectra"], S[:2]) and np.allclose(full["perimeters"], P[:2]))
    check("return_all sizes", len(full["eigenfunctions"]) == len(full["triangulations"]) == len(full["coordinates"]) == 2
          and phi.shape == (len(X), N_eig) and T.shape[1] == 3)
    assert ok, "quadrilateral_drums test failed"
