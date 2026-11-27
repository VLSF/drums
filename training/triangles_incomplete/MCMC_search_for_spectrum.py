import jax
import jax.numpy as jnp
import time

from jax.lax import scan, cond
from jax import random, vmap
from functools import partial


def local_matrices(xy):
    """
    Computes local stiffness (Ke) and mass (Me) matrices for a linear triangular element.
    xy: (3, 2) array of element vertices.
    """
    x, y = xy[:, 0], xy[:, 1]
    b = jnp.roll(y, -1) - jnp.roll(y, -2)  # y_{i+1} - y_{i+2}
    c = jnp.roll(x, -2) - jnp.roll(x, -1)  # x_{i+2} - x_{i+1}
    A = 0.5 * ((x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0]))
    Ke = (jnp.outer(b, b) + jnp.outer(c, c)) / (4.0 * A)
    Me = A / 12.0 * (jnp.ones((3, 3)) + jnp.eye(3))
    return Ke, Me


def get_grid_topology(N: int):
    """
    Precomputes the grid topology and barycentric coordinates for N subdivisions.
    Strictly pure function: deterministic output for any integer N, without caching side effects.
    Returns:
        U: (num_nodes, 3) barycentric coordinates [w, u, v] with w + u + v = 1
        T: (num_triangles, 3) element connectivity matrix
        free: (num_free_nodes,) 1D array of interior node indices
        rows: (num_triangles * 9,) row indices for K and M assembly
        cols: (num_triangles * 9,) col indices for K and M assembly
    """
    nodes = []
    node_indices = {}
    idx = 0
    free = []

    for j in range(N + 1):
        for i in range(N + 1 - j):
            u = i / N
            v = j / N
            w = 1.0 - u - v
            nodes.append([w, u, v])
            node_indices[(i, j)] = idx
            # Interior nodes are not on any of the 3 triangle edges
            if not (i == 0 or j == 0 or i + j == N):
                free.append(idx)
            idx += 1

    U = jnp.array(nodes)
    free = jnp.array(free)

    T = []
    for j in range(N):
        for i in range(N - j):
            # Upward triangle (CCW)
            n1 = node_indices[(i, j)]
            n2 = node_indices[(i + 1, j)]
            n3 = node_indices[(i, j + 1)]
            T.append([n1, n2, n3])

            # Downward triangle (CCW)
            if i < N - 1 - j:
                n4 = node_indices[(i + 1, j + 1)]
                T.append([n2, n4, n3])

    T = jnp.array(T)
    rows = jnp.broadcast_to(T[:, :, None], (len(T), 3, 3)).ravel()
    cols = jnp.broadcast_to(T[:, None, :], (len(T), 3, 3)).ravel()

    return U, T, free, rows, cols


def create_triangle_grid(triangle, N: int = 35):
    """
    Constructs the 2D mesh nodes X and element connectivity T for a general triangle.
    triangle: (3, 2) array of vertices.
    N: number of subdivisions along each edge.
    """
    U, T, free, rows, cols = get_grid_topology(N)
    X = U @ triangle
    return X, T, free, rows, cols


@partial(jax.jit, static_argnames=['N'])
def get_triangle_spectrum(triangle, N: int = 35):
    """
    Computes the Dirichlet Laplace spectrum for any general triangle purely in JAX.

    Parameters:
        triangle: (3, 2) array of triangle vertices.
        N: static integer, grid subdivision density (default: 35).

    Returns:
        spectrum: (num_free_nodes,) eigenvalues sorted in ascending order.
    """
    # Ensure standard counter-clockwise orientation for positive element area
    signed_area = 0.5 * (
        (triangle[1, 0] - triangle[0, 0]) * (triangle[2, 1] - triangle[0, 1])
        - (triangle[2, 0] - triangle[0, 0]) * (triangle[1, 1] - triangle[0, 1])
    )
    triangle = jax.lax.cond(
        signed_area < 0,
        lambda t: jnp.stack([t[0], t[2], t[1]], axis=0),
        lambda t: t,
        triangle,
    )

    U, T, free, rows, cols = get_grid_topology(N)
    X = U @ triangle

    # Compute element stiffness and mass matrices in parallel
    ke, me = jax.vmap(local_matrices)(X[T])

    num_nodes = U.shape[0]
    K = jnp.zeros((num_nodes, num_nodes)).at[rows, cols].add(ke.ravel())
    M = jnp.zeros((num_nodes, num_nodes)).at[rows, cols].add(me.ravel())

    # Apply Dirichlet boundary conditions (zero on boundary nodes)
    Kf = K[free][:, free]
    Mf = M[free][:, free]

    # Enforce exact symmetry
    Kf = (Kf + Kf.T) / 2.0
    Mf = (Mf + Mf.T) / 2.0

    # Solve generalized symmetric eigenvalue problem Kf u = lambda Mf u
    L = jnp.linalg.cholesky(Mf)
    temp = jax.scipy.linalg.solve_triangular(L, Kf, lower=True)
    C = jax.scipy.linalg.solve_triangular(L, temp.T, lower=True).T
    C = (C + C.T) / 2.0

    spectrum = jnp.linalg.eigvalsh(C)
    return spectrum


# ==============================================================================
# Invariant Geometry & Prior Specification
# ==============================================================================

# Reference canonical equilateral triangle centered at origin
MU_TEMPLATE = jnp.array([
    [0.0, 1.0],
    [-jnp.sqrt(3.0) / 2.0, -0.5],
    [jnp.sqrt(3.0) / 2.0, -0.5]
])

# All 6 permutations of 3 vertices (dihedral / symmetric group S3)
PERMUTATIONS = jnp.array([
    [0, 1, 2], [1, 2, 0], [2, 0, 1],
    [0, 2, 1], [2, 1, 0], [1, 0, 2]
])


def align(v1, v2):
    """
    Centre v1 and v2 and rotate/reflect v2 onto v1 via orthogonal Procrustes (SVD).
    """
    v1_ = v1 - jnp.mean(v1, axis=0, keepdims=True)
    v2_ = v2 - jnp.mean(v2, axis=0, keepdims=True)
    U, S, Vt = jnp.linalg.svd(v2_.T @ v1_, full_matrices=False)
    R = U @ Vt
    return v1_, v2_ @ R


def distance(target, prediction):
    """
    Computes mean vertex distance after optimal orthogonal Procrustes alignment.
    """
    target_, prediction_ = align(target, prediction)
    return jnp.mean(jnp.linalg.norm(target_ - prediction_, axis=1))


def invariant_distance(target, prediction):
    """
    Computes the Procrustes distance invariant under ISO(2) rigid motions,
    reflections O(2), and all vertex permutations S3.
    """
    return jnp.min(vmap(lambda ind: distance(target, prediction[ind]))(PERMUTATIONS))


def log_prior_density(triangle, sigma_p: float = 0.4):
    """
    Log-prior density defined by the invariant Procrustes distance to MU_TEMPLATE.
    In Kendall's shape space, this corresponds to an isotropic Gaussian centered
    at the equilateral template, invariant to translation, rotation, and vertex relabeling.
    """
    d = invariant_distance(MU_TEMPLATE, triangle)
    return - (d**2) / (2.0 * sigma_p**2)


def is_valid_triangle(triangle, min_svd: float = 6e-2, min_area: float = 1e-2):
    """
    Safeguard that prevents triangles from collapsing to collinear or degenerate configurations.
    Returns True if the triangle is well-conditioned for the FEM solver.
    """
    sigma_min = jnp.min(jnp.linalg.svdvals(triangle))
    signed_area = 0.5 * (
        (triangle[1, 0] - triangle[0, 0]) * (triangle[2, 1] - triangle[0, 1])
        - (triangle[2, 0] - triangle[0, 0]) * (triangle[1, 1] - triangle[0, 1])
    )
    return (sigma_min > min_svd) & (jnp.abs(signed_area) > min_area)


def get_invariants(triangle):
    """
    Computes geometric invariants independent of Euclidean motions and vertex ordering:
        s: semiperimeter
        A: area
        r: inradius (A / s)
        R: circumradius invariant (l1 * l2 * l3 / A)
    """
    V = [[0, 1], [1, 2], [0, 2]]
    l = [jnp.linalg.norm(triangle[v[0]] - triangle[v[1]]) for v in V]
    s = (l[0] + l[1] + l[2]) / 2.0
    A = jnp.sqrt(jnp.maximum(0.0, s * (s - l[0]) * (s - l[1]) * (s - l[2])))
    r = A / s
    R_circ = l[0] * l[1] * l[2] / A
    return s, A, r, R_circ


# ==============================================================================
# MCMC Sampling from the Prior
# ==============================================================================

@partial(jax.jit, static_argnames=['N_steps'])
def sample_prior_mcmc(key, N_steps: int = 300, sigma_p: float = 0.4, sigma_t: float = 0.08):
    """
    Samples a centered triangle from the invariant-distance prior:
        p(T) ~ exp(-D_inv(T, mu)^2 / (2 * sigma_p^2))
    via fast JAX-compiled MCMC without evaluating the FEM eigensolver.
    """
    def step(carry, _):
        T, k = carry
        k_prop, k_accept, k_next = random.split(k, 3)
        T_prop = T + sigma_t * random.normal(k_prop, T.shape)
        T_prop = T_prop - jnp.mean(T_prop, axis=0, keepdims=True)

        valid = is_valid_triangle(T_prop)
        safe_T_prop = jnp.where(valid, T_prop, T)

        d_curr = invariant_distance(MU_TEMPLATE, T)
        d_prop = invariant_distance(MU_TEMPLATE, safe_T_prop)
        log_alpha = (d_curr**2 - d_prop**2) / (2.0 * sigma_p**2)

        coin = random.uniform(k_accept, ())
        accepted = valid & (jnp.log(coin) <= jnp.minimum(0.0, log_alpha))
        T_new = jnp.where(accepted, T_prop, T)
        return (T_new, k_next), T_new

    carry = (MU_TEMPLATE, key)
    carry, _ = scan(step, carry, jnp.arange(N_steps))
    return carry[0]


# ==============================================================================
# MCMC Sampling from the Posterior
# ==============================================================================

def standard_step(triangle_, triangle, log_spectrum, log_target_spectrum, sigma_s, sigma_p, key):
    """
    Evaluates the spectral log-likelihood and invariant prior ratio for a valid proposal.
    """
    log_spectrum_ = jnp.log(get_triangle_spectrum(triangle_, N=35)[:log_spectrum.shape[0]])

    norm_s = jnp.sum((log_spectrum - log_target_spectrum)**2) / (2.0 * sigma_s**2)
    norm_s_ = jnp.sum((log_spectrum_ - log_target_spectrum)**2) / (2.0 * sigma_s**2)

    log_alpha = (
        norm_s - norm_s_
        + log_prior_density(triangle_, sigma_p)
        - log_prior_density(triangle, sigma_p)
    )
    log_alpha = jnp.minimum(0.0, log_alpha)

    coin = random.uniform(key, ())
    accepted = jnp.log(coin) <= log_alpha

    new_triangle = jnp.where(accepted, triangle_, triangle)
    new_log_spectrum = jnp.where(accepted, log_spectrum_, log_spectrum)
    return new_triangle, new_log_spectrum


def reject_step(triangle_, triangle, log_spectrum, log_target_spectrum, sigma_s, sigma_p, key):
    """Rejects proposal directly if geometric safeguard fails."""
    return triangle, log_spectrum


@jax.jit
def metropolis_hastings_jax_step(
    triangle, log_spectrum, log_target_spectrum, sigma_s, sigma_t, sigma_p, key
):
    """
    Single Metropolis-Hastings transition kernel with collapse safeguard.
    """
    keys = random.split(key, 3)
    triangle_ = triangle + sigma_t * random.normal(keys[0], triangle.shape)
    triangle_ = triangle_ - jnp.mean(triangle_, axis=0, keepdims=True)

    # Geometric safeguard against triangle collapse
    valid = is_valid_triangle(triangle_)
    safe_triangle_ = jnp.where(valid, triangle_, triangle)

    new_triangle, new_log_spectrum = cond(
        valid,
        lambda args: standard_step(*args),
        lambda args: reject_step(*args),
        (safe_triangle_, triangle, log_spectrum, log_target_spectrum, sigma_s, sigma_p, keys[1])
    )
    return new_triangle, new_log_spectrum, keys[2]


def metropolis_hastings_jax_scan(carry, ind):
    triangle, log_spectrum, log_target_spectrum, sigma_s, sigma_t, sigma_p, key = carry
    triangle, log_spectrum, key = metropolis_hastings_jax_step(
        triangle, log_spectrum, log_target_spectrum, sigma_s, sigma_t, sigma_p, key
    )
    return [triangle, log_spectrum, log_target_spectrum, sigma_s, sigma_t, sigma_p, key], (triangle, log_spectrum)


def test_run(log_target_spectrum, key, sigma_s: float = 0.1, sigma_t: float = 0.08, sigma_p: float = 0.4, N_steps: int = 1000):
    """
    Runs MCMC on the posterior starting from an independent draw from the prior.
    """
    k_init, k_chain = random.split(key)

    # Initialize from an independent sample from the prior
    triangle = sample_prior_mcmc(k_init, N_steps=300, sigma_p=sigma_p, sigma_t=sigma_t)
    log_spectrum = jnp.log(get_triangle_spectrum(triangle, N=35)[:log_target_spectrum.shape[0]])

    T_ = jnp.array([triangle])
    S_ = jnp.array([log_spectrum])

    carry = [triangle, log_spectrum, log_target_spectrum, sigma_s, sigma_t, sigma_p, k_chain]
    carry, (T, S) = scan(metropolis_hastings_jax_scan, carry, jnp.arange(N_steps))

    T = jnp.concatenate([T_, T])
    S = jnp.concatenate([S_, S])
    return T, S


# ==============================================================================
# Verification and Test Suite
# ==============================================================================

if __name__ == "__main__":
    master_key = random.PRNGKey(55)
    k_targets, k_inits = random.split(master_key, 2)

    num_samples = 6
    target_keys = random.split(k_targets, num_samples)
    init_keys = random.split(k_inits, num_samples)

    print("Generating reference target spectra from independent prior samples...")
    # Sample reference triangles from the prior and compute their Dirichlet spectra
    target_triangles = jax.jit(vmap(lambda k: sample_prior_mcmc(k, N_steps=400, sigma_p=0.4, sigma_t=0.08)))(target_keys)
    target_spectra = jax.jit(vmap(lambda t: get_triangle_spectrum(t, N=35)[:2]))(target_triangles)

    for i in range(num_samples):
        print(f"Target {i}: spectrum = [{target_spectra[i, 0]:.4f}, {target_spectra[i, 1]:.4f}]")
    print()

    # Test MCMC on the resulting spectra starting from independent samples from prior
    for i in range(num_samples):
        target_spec = target_spectra[i]
        log_target_spec = jnp.log(target_spec)

        ts = time.time()
        T, S = test_run(log_target_spec, init_keys[i], sigma_s=0.1, sigma_t=0.08, sigma_p=0.4, N_steps=1000)
        tf = time.time()

        # Compute relative spectral errors across chain
        S_errors = jnp.linalg.norm(jnp.exp(S) / target_spec.reshape(1, -1) - 1.0, axis=1)

        print(f"sample {i}, time {tf - ts:.4f}")
        print("relative errors", *[f"{e:.4f}" for e in S_errors[:3]], "...", *[f"{e:.4f}" for e in S_errors[-3:]])

        # Report invariant metrics of the target vs sampled posterior
        target_s, target_A, target_r, target_R = get_invariants(target_triangles[i])
        post_s, post_A, post_r, post_R = vmap(get_invariants)(T[500:])  # post-burnin
        print(f"  Target invariants:    Area = {target_A:.4f}, s = {target_s:.4f}, inradius r = {target_r:.4f}, R = {target_R:.4f}")
        print(f"  Posterior (post 500): Area = {jnp.mean(post_A):.4f} +/- {jnp.std(post_A):.4f}, inradius r = {jnp.mean(post_r):.4f} +/- {jnp.std(post_r):.4f}")
        print()
