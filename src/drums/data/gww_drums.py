import numpy as np

try:
    from drums.data.spectrum import get_drum_spectrum
    from drums.data.quadrilateral_drums import polygon_area, is_simple
except ImportError:  # run as a script from the data folder
    from spectrum import get_drum_spectrum
    from quadrilateral_drums import polygon_area, is_simple


# Isospectral drums from Gordon, Webb, Wolpert, "Isospectral plane domains and surfaces via
# Riemannian orbifolds", https://link.springer.com/article/10.1007/BF01231320 (counter-clockwise).
GWW_1 = np.array([[2, 0], [3, 1], [3, 2], [2, 2], [1, 2], [1, 3], [0, 2], [1, 1], [2, 1]], dtype=float)
GWW_2 = np.array([[2, 0], [2, 1], [3, 1], [2, 2], [1, 2], [1, 3], [0, 3], [0, 2], [1, 1]], dtype=float)
GWW = (GWW_1, GWW_2)


def perturbed_gww(rng, domain, delta):
    """Vertices of GWW[domain] with independent uniform noise in [-delta, delta] in each coordinate.

    The noise is drawn again until the polygon is simple: for delta > 0.3 the noise can make
    the boundary intersect itself (in a few percent of the cases for delta close to 0.5).
    """
    while True:
        polygon = GWW[domain] + rng.uniform(-delta, delta, size=GWW[domain].shape)
        if is_simple(polygon):
            return polygon


def random_pose(rng, polygon):
    """Centre the polygon at its mean vertex, rotate it uniformly and start from a random vertex.

    Without it the fixed position and starting vertex of the two drums would tell them apart.
    """
    polygon = polygon - polygon.mean(axis=0)
    t = rng.uniform(0, 2*np.pi)
    R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    return np.roll(polygon @ R.T, rng.integers(len(polygon)), axis=0)


def random_gww(rng, delta):
    """One of the two GWW drums (each with probability 1/2), perturbed, in a random pose.

    Returns the vertices, shape (9, 2), and the index of the drum, 0 or 1.
    """
    domain = int(rng.uniform() >= 0.5)
    return random_pose(rng, perturbed_gww(rng, domain, delta)), domain


def generate_gww_drums(n_samples, seed=0, N_eig=30, n_elements=4000, delta=0.1, return_all=False):
    """Dataset of the two isospectral GWW drums with randomly perturbed vertices.

    Each drum is GWW_1 or GWW_2 with probability 1/2. Every coordinate of its 9 vertices is
    shifted by an independent uniform random number from [-delta, delta], 0 < delta < 0.5. The
    drum is then centred, rotated and its first vertex is chosen at random. The maximal element
    area of the mesh is area / n_elements for each drum.

    Returns a dict with
        'perimeters': (n_samples, 9, 2) vertices of the drums, counter-clockwise,
        'spectra':    (n_samples, N_eig) smallest Dirichlet eigenvalues, ascending,
        'domain':     (n_samples,) 0 if the drum is a perturbed GWW_1, 1 if GWW_2.
    If return_all is True, the dictionary also contains lists (one entry per drum,
    the meshes differ in size) with everything get_drum_spectrum produces:
        'eigenfunctions': (N, N_eig) values at the mesh nodes, columns ordered as the spectrum,
        'triangulations': (N_triangles, 3) node indices of the triangles,
        'coordinates':    (N, 2) mesh nodes.
    """
    if not 0 < delta < 0.5:
        raise ValueError(f"delta must be in (0, 0.5), got {delta}")
    rng = np.random.default_rng(seed)
    perimeters, spectra, domains = [], [], []
    extra = dict(eigenfunctions=[], triangulations=[], coordinates=[])
    for _ in range(n_samples):
        drum, domain = random_gww(rng, delta)
        a = polygon_area(drum) / n_elements
        lam, phi, X, T = get_drum_spectrum(drum, N_eig=N_eig, a=a)
        order = np.argsort(lam)
        perimeters.append(drum)
        spectra.append(lam[order])
        domains.append(domain)
        if return_all:  # only keep the mesh data if it is returned
            extra["eigenfunctions"].append(phi[:, order])
            extra["triangulations"].append(T)
            extra["coordinates"].append(X)
    data = dict(perimeters=np.stack(perimeters), spectra=np.stack(spectra), domain=np.array(domains))
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

    # --- the two drums are isospectral: the spectra agree up to the discretization error,
    #     which is estimated by refining the mesh
    lams = {}
    for k, g in enumerate(GWW):
        for ne in (4000, 16000, 64000):
            lam, _, _, _ = get_drum_spectrum(g, N_eig=10, a=polygon_area(g) / ne)
            lams[k, ne] = np.sort(lam)
    print("GWW_1:", lams[0, 64000])
    print("GWW_2:", lams[1, 64000])
    gap = np.max(np.abs(lams[0, 64000] / lams[1, 64000] - 1))
    err = max(np.max(np.abs(lams[k, 4000] / lams[k, 64000] - 1)) for k in range(2))
    print(f"max relative difference between GWW_1 and GWW_2 (finest mesh): {gap:.2e}")
    print(f"max relative error of the default mesh w.r.t. n_elements=64000: {err:.2e}")
    check("GWW_1 and GWW_2 isospectral (rtol=1e-2)", gap < 1e-2)
    check("default mesh resolves GWW to 1%", err < 1e-2)
    check("GWW drums are simple and counter-clockwise, area 3.5",
          all(is_simple(g) and np.isclose(polygon_area(g), 3.5) for g in GWW)
          and all(g[:, 0] @ np.roll(g[:, 1], -1) - g[:, 1] @ np.roll(g[:, 0], -1) > 0 for g in GWW))

    # --- samplers
    rng = np.random.default_rng(1)
    for delta in (0.1, 0.49):
        noisy = [perturbed_gww(rng, d, delta) for d in (0, 1) for _ in range(500)]
        shifts = np.array([p - GWW[d] for d in (0, 1) for p in noisy[500*d:500*(d+1)]])
        check(f"delta={delta}: noise within [-delta, delta] and uses the whole range",
              np.abs(shifts).max() <= delta and np.abs(shifts).max() > 0.95*delta)
        check(f"delta={delta}: noise has zero mean and uniform variance delta^2/3",
              np.allclose(shifts.mean(), 0, atol=0.02*delta) and np.isclose(shifts.var(), delta**2/3, rtol=0.05))
        check(f"delta={delta}: drums are simple", all(is_simple(p) for p in noisy))

    drum = perturbed_gww(rng, 0, 0.3)
    posed = random_pose(rng, drum)
    edges = lambda p: np.linalg.norm(np.roll(p, -1, axis=0) - p, axis=1)
    check("pose: centred, area and counter-clockwise orientation kept",
          np.allclose(posed.mean(axis=0), 0) and np.isclose(polygon_area(posed), polygon_area(drum))
          and posed[:, 0] @ np.roll(posed[:, 1], -1) - posed[:, 1] @ np.roll(posed[:, 0], -1) > 0)
    check("pose: rigid motion and cyclic shift only",
          any(np.allclose(edges(posed), np.roll(edges(drum), k)) for k in range(9)))
    samples = [random_gww(rng, 0.1) for _ in range(2000)]
    frac = np.mean([d for _, d in samples])
    check(f"probability of GWW_2 is 1/2 (observed {frac:.3f})", abs(frac - 0.5) < 0.05)
    check("first vertex is random", len({int(np.argmin(p[:, 1])) for p, _ in samples}) == 9)
    for bad in (0.0, 0.5, -0.1):
        try:
            generate_gww_drums(1, delta=bad)
            check(f"delta={bad} rejected", False)
        except ValueError:
            check(f"delta={bad} rejected", True)

    # --- dataset
    n_samples, N_eig = 20, 30
    t0 = time.time()
    data = generate_gww_drums(n_samples, seed=0, N_eig=N_eig, delta=0.3)
    print(f"generated {n_samples} drums in {time.time() - t0:.1f} s")
    P, S, D = data["perimeters"], data["spectra"], data["domain"]
    check("keys", set(data) == {"perimeters", "spectra", "domain"})
    check("shapes", P.shape == (n_samples, 9, 2) and S.shape == (n_samples, N_eig) and D.shape == (n_samples,))
    check("finite and positive", np.all(np.isfinite(S)) and np.all(S > 0))
    check("sorted", np.all(np.diff(S, axis=1) >= 0))
    check("both drums present", set(D) == {0, 1})
    check("simple", all(is_simple(p) for p in P))
    check("reproducible", np.allclose(generate_gww_drums(2, seed=0, N_eig=N_eig, delta=0.3)["spectra"], S[:2]))

    full = generate_gww_drums(2, seed=0, N_eig=N_eig, delta=0.3, return_all=True)
    phi, X, T = full["eigenfunctions"][0], full["coordinates"][0], full["triangulations"][0]
    check("return_all keys", set(full) == {"perimeters", "spectra", "domain", "eigenfunctions",
                                          "triangulations", "coordinates"})
    check("return_all spectra unchanged", np.allclose(full["spectra"], S[:2]) and np.allclose(full["perimeters"], P[:2]))
    check("return_all sizes", len(full["eigenfunctions"]) == len(full["triangulations"]) == len(full["coordinates"]) == 2
          and phi.shape == (len(X), N_eig) and T.shape[1] == 3)
    assert ok, "gww_drums test failed"
