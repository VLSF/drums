import numpy as np
import scipy.sparse as sp

try:
    from drums.data.spectrum import get_drum_spectrum, local_matrices
except ImportError:  # run as a script from the data folder
    from spectrum import get_drum_spectrum, local_matrices


def triangle_area(triangle):
    x, y = triangle[:, 0], triangle[:, 1]
    return np.abs(0.5 * ((x[1]-x[0])*(y[2]-y[0]) - (x[2]-x[0])*(y[1]-y[0])))


def random_triangle(rng, r_min=0.5, r_max=3.0, angle_min=np.pi/6):
    """Random triangle inscribed in a circle of radius r centred at the origin.

    The interior angles are uniform on the simplex {sum = pi} restricted to
    min angle >= angle_min (rejection sampling). The radius is uniform in
    [r_min, r_max], the rotation is uniform. Vertices are counter-clockwise.
    """
    while True:
        angles = np.pi * rng.dirichlet(np.ones(3))
        if angles.min() >= angle_min:
            break
    r = rng.uniform(r_min, r_max)
    # an inscribed angle is half of the central angle of the opposite arc
    t = rng.uniform(0, 2*np.pi) + np.cumsum([0, 2*angles[2], 2*angles[0]])
    return r * np.stack([np.cos(t), np.sin(t)], axis=1)


def generate_triangle_drums(n_samples, seed=0, N_eig=30, n_elements=4000,
                            r_min=0.5, r_max=3.0, angle_min=np.pi/6, return_all=False):
    """Dataset of triangular drums.

    The maximal element area of the mesh is area / n_elements for each triangle,
    so all drums are resolved comparably, independently of their size.

    Returns a dict with
        'perimeters': (n_samples, 3, 2) vertices of the triangles,
        'spectra':    (n_samples, N_eig) smallest Dirichlet eigenvalues, ascending.
    If return_all is True, the dictionary also contains lists (one entry per drum,
    the meshes differ in size) with everything get_drum_spectrum produces:
        'eigenfunctions': (N, N_eig) values at the mesh nodes, columns ordered as the spectrum,
        'triangulations': (N_triangles, 3) node indices of the triangles,
        'coordinates':    (N, 2) mesh nodes.
    """
    rng = np.random.default_rng(seed)
    perimeters, spectra = [], []
    extra = dict(eigenfunctions=[], triangulations=[], coordinates=[])
    for _ in range(n_samples):
        triangle = random_triangle(rng, r_min, r_max, angle_min)
        a = triangle_area(triangle) / n_elements
        lam, phi, X, T = get_drum_spectrum(triangle, N_eig=N_eig, a=a)
        order = np.argsort(lam)
        perimeters.append(triangle)
        spectra.append(lam[order])
        if return_all:  # only keep the mesh data if it is returned
            extra["eigenfunctions"].append(phi[:, order])
            extra["triangulations"].append(T)
            extra["coordinates"].append(X)
    data = dict(perimeters=np.stack(perimeters), spectra=np.stack(spectra))
    if return_all:
        data.update(extra)
    return data


def min_angle(triangle):
    angles = []
    for i in range(3):
        u = triangle[(i+1) % 3] - triangle[i]
        v = triangle[(i+2) % 3] - triangle[i]
        angles.append(np.arccos(u @ v / np.linalg.norm(u) / np.linalg.norm(v)))
    return min(angles)


if __name__ == "__main__":
    import time
    ok = True

    # --- right isosceles triangle with legs L: lambda = pi^2 (m^2 + n^2) / L^2, m > n >= 1
    L = 2.0
    tri = np.array([[0, 0], [L, 0], [0, L]], dtype=float)
    lam, _, _, _ = get_drum_spectrum(tri, N_eig=10, a=triangle_area(tri) / 4000)
    lam = np.sort(lam)
    exact = np.sort([np.pi**2 * (m**2 + n**2) / L**2 for m in range(2, 30) for n in range(1, m)])[:10]
    close = np.allclose(lam, exact, rtol=1e-2)
    ok &= close
    print(f"right isosceles triangle\n  computed: {lam}\n  expected: {exact}\n  allclose (rtol=1e-2): {close}")

    # --- dataset
    n_samples, N_eig = 20, 30
    t0 = time.time()
    data = generate_triangle_drums(n_samples, seed=0, N_eig=N_eig)
    print(f"generated {n_samples} drums in {time.time() - t0:.1f} s")
    P, S = data["perimeters"], data["spectra"]
    checks = {
        "shapes": P.shape == (n_samples, 3, 2) and S.shape == (n_samples, N_eig),
        "finite and positive": np.all(np.isfinite(S)) and np.all(S > 0),
        "sorted": np.all(np.diff(S, axis=1) >= 0),
        "min angle >= 30 deg": min(min_angle(t) for t in P) >= np.pi/6 - 1e-9,
        "circumradius in [0.5, 3]": np.all((np.linalg.norm(P, axis=2) >= 0.5 - 1e-9) & (np.linalg.norm(P, axis=2) <= 3 + 1e-9)),
        "reproducible": np.allclose(generate_triangle_drums(2, seed=0, N_eig=N_eig)["spectra"], S[:2]),
    }
    check_default = set(data) == {"perimeters", "spectra"}
    print(f"default keys are perimeters and spectra: {check_default}")
    ok &= check_default

    # --- return_all: same spectra, consistent extra data, Rayleigh quotients match eigenvalues
    full = generate_triangle_drums(2, seed=0, N_eig=N_eig, return_all=True)
    keys = set(full) == {"perimeters", "spectra", "eigenfunctions", "triangulations", "coordinates"}
    same = np.allclose(full["spectra"], S[:2]) and np.allclose(full["perimeters"], P[:2])
    phi, X, T = full["eigenfunctions"][0], full["coordinates"][0], full["triangulations"][0]
    sizes = len(full["eigenfunctions"]) == len(full["triangulations"]) == len(full["coordinates"]) == 2 \
        and phi.shape == (len(X), N_eig) and T.shape[1] == 3
    # eigenfunction k must satisfy K phi = lambda M phi: check via Rayleigh quotient on the mesh
    Ke_Me = [local_matrices(X[t]) for t in T]
    rows = np.repeat(T[:, :, None], 3, axis=2).ravel(); cols = np.repeat(T[:, None, :], 3, axis=1).ravel()
    K = sp.coo_matrix((np.array([np.asarray(k) for k, _ in Ke_Me]).ravel(), (rows, cols)), shape=(len(X),) * 2).tocsr()
    M = sp.coo_matrix((np.array([np.asarray(m) for _, m in Ke_Me]).ravel(), (rows, cols)), shape=(len(X),) * 2).tocsr()
    rq = np.array([phi[:, k] @ (K @ phi[:, k]) / (phi[:, k] @ (M @ phi[:, k])) for k in range(N_eig)])
    rayleigh = np.allclose(rq, full["spectra"][0], rtol=1e-6)
    for name, c in [("return_all keys", keys), ("return_all spectra unchanged", same),
                    ("return_all sizes", sizes), ("eigenfunction order matches spectrum", rayleigh)]:
        print(f"{name}: {bool(c)}")
        ok &= bool(c)

    for name, c in checks.items():
        print(f"{name}: {bool(c)}")
        ok &= bool(c)
    print("smallest angle (deg):", np.degrees(min(min_angle(t) for t in P)))
    assert ok, "triangle_drums test failed"
