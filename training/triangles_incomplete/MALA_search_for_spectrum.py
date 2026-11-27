import jax
import jax.numpy as jnp
import time

from jax.lax import scan, cond, stop_gradient
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
            if not (i == 0 or j == 0 or i + j == N):
                free.append(idx)
            idx += 1

    U = jnp.array(nodes)
    free = jnp.array(free)

    T = []
    for j in range(N):
        for i in range(N - j):
            n1 = node_indices[(i, j)]
            n2 = node_indices[(i + 1, j)]
            n3 = node_indices[(i, j + 1)]
            T.append([n1, n2, n3])

            if i < N - 1 - j:
                n4 = node_indices[(i + 1, j + 1)]
                T.append([n2, n4, n3])

    T = jnp.array(T)
    rows = jnp.broadcast_to(T[:, :, None], (len(T), 3, 3)).ravel()
    cols = jnp.broadcast_to(T[:, None, :], (len(T), 3, 3)).ravel()

    return U, T, free, rows, cols


@partial(jax.jit, static_argnames=['N'])
def get_triangle_spectrum(triangle, N: int = 35):
    """
    Computes the Dirichlet Laplace spectrum for any general triangle purely in JAX.
    Differentiable via reverse-mode automatic differentiation.
    """
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

    ke, me = jax.vmap(local_matrices)(X[T])

    num_nodes = U.shape[0]
    K = jnp.zeros((num_nodes, num_nodes)).at[rows, cols].add(ke.ravel())
    M = jnp.zeros((num_nodes, num_nodes)).at[rows, cols].add(me.ravel())

    Kf = K[free][:, free]
    Mf = M[free][:, free]

    Kf = (Kf + Kf.T) / 2.0
    Mf = (Mf + Mf.T) / 2.0

    L = jnp.linalg.cholesky(Mf)
    temp = jax.scipy.linalg.solve_triangular(L, Kf, lower=True)
    C = jax.scipy.linalg.solve_triangular(L, temp.T, lower=True).T
    C = (C + C.T) / 2.0

    spectrum = jnp.linalg.eigvalsh(C)
    return spectrum


# ==============================================================================
# Invariant Geometry & Differentiable Invariant Prior
# ==============================================================================

MU_TEMPLATE = jnp.array([
    [0.0, 1.0],
    [-jnp.sqrt(3.0) / 2.0, -0.5],
    [jnp.sqrt(3.0) / 2.0, -0.5]
])

PERMUTATIONS = jnp.array([
    [0, 1, 2], [1, 2, 0], [2, 0, 1],
    [0, 2, 1], [2, 1, 0], [1, 0, 2]
])


def align_envelope(v1, v2):
    """
    Orthogonal Procrustes alignment using Danskin's envelope theorem.
    By stopping gradients through the optimal rotation matrix R, we obtain the
    exact analytical gradient of the minimum Procrustes distance while completely
    bypassing the SVD derivative singularity (1 / (S1^2 - S2^2)) when singular values collide.
    """
    v1_ = v1 - jnp.mean(v1, axis=0, keepdims=True)
    v2_ = v2 - jnp.mean(v2, axis=0, keepdims=True)
    U, S, Vt = jnp.linalg.svd(v2_.T @ v1_, full_matrices=False)
    R = stop_gradient(U @ Vt)
    return v1_, v2_ @ R


def distance_sq(target, prediction):
    """Squared Procrustes distance between two configurations."""
    target_, prediction_ = align_envelope(target, prediction)
    return jnp.mean(jnp.sum((target_ - prediction_)**2, axis=1))


def invariant_dist_sq(target, prediction):
    """
    Minimum squared Procrustes distance invariant under ISO(2), O(2), and S3 permutations.
    """
    return jnp.min(vmap(lambda ind: distance_sq(target, prediction[ind]))(PERMUTATIONS))


def log_prior_density(triangle, sigma_p: float = 0.4):
    """
    Proper invariant prior density: exp(-D_inv^2(T, mu) / (2 * sigma_p^2)).
    """
    d_sq = invariant_dist_sq(MU_TEMPLATE, triangle)
    return - 0.5 * d_sq / (sigma_p**2)


def is_valid_triangle(triangle, min_svd: float = 6e-2, min_area: float = 1e-2):
    """
    Safeguard that prevents triangles from collapsing to collinear or degenerate configurations.
    """
    sigma_min = jnp.min(jnp.linalg.svdvals(triangle))
    signed_area = 0.5 * (
        (triangle[1, 0] - triangle[0, 0]) * (triangle[2, 1] - triangle[0, 1])
        - (triangle[2, 0] - triangle[0, 0]) * (triangle[1, 1] - triangle[0, 1])
    )
    return (sigma_min > min_svd) & (jnp.abs(signed_area) > min_area)


def get_invariants(triangle):
    """Computes scale- and shape-invariants: semiperimeter s, area A, inradius r, circumradius R."""
    V = [[0, 1], [1, 2], [0, 2]]
    l = [jnp.linalg.norm(triangle[v[0]] - triangle[v[1]]) for v in V]
    s = (l[0] + l[1] + l[2]) / 2.0
    A = jnp.sqrt(jnp.maximum(0.0, s * (s - l[0]) * (s - l[1]) * (s - l[2])))
    r = A / s
    R_circ = l[0] * l[1] * l[2] / A
    return s, A, r, R_circ


# ==============================================================================
# Fast Prior Sampling via MCMC
# ==============================================================================

@partial(jax.jit, static_argnames=['N_steps'])
def sample_prior_mcmc(key, N_steps: int = 300, sigma_p: float = 0.4, sigma_t: float = 0.08):
    """Samples a centered triangle from the invariant-distance prior in < 1 ms."""
    def step(carry, _):
        T, k = carry
        k_prop, k_accept, k_next = random.split(k, 3)
        T_prop = T + sigma_t * random.normal(k_prop, T.shape)
        T_prop = T_prop - jnp.mean(T_prop, axis=0, keepdims=True)

        valid = is_valid_triangle(T_prop)
        safe_T_prop = jnp.where(valid, T_prop, T)

        d_curr_sq = invariant_dist_sq(MU_TEMPLATE, T)
        d_prop_sq = invariant_dist_sq(MU_TEMPLATE, safe_T_prop)
        log_alpha = (d_curr_sq - d_prop_sq) / (2.0 * sigma_p**2)

        coin = random.uniform(k_accept, ())
        accepted = valid & (jnp.log(coin) <= jnp.minimum(0.0, log_alpha))
        T_new = jnp.where(accepted, T_prop, T)
        return (T_new, k_next), T_new

    carry = (MU_TEMPLATE, key)
    carry, _ = scan(step, carry, jnp.arange(N_steps))
    return carry[0]


# ==============================================================================
# MALA: Metropolis-Adjusted Langevin Algorithm
# ==============================================================================

def log_posterior(triangle, log_target_spectrum, sigma_s: float = 0.1, sigma_p: float = 0.4, N_grid: int = 35):
    """
    Log-posterior target density U(T) = ln p(lambda* | T) + ln p_prior(T).
    """
    log_spec = jnp.log(get_triangle_spectrum(triangle, N=N_grid)[:log_target_spectrum.shape[0]])
    log_lik = - 0.5 * jnp.sum((log_spec - log_target_spectrum)**2) / (sigma_s**2)
    log_prior = log_prior_density(triangle, sigma_p)
    return log_lik + log_prior


def mala_step(carry, ind, log_target_spectrum, tau: float = 4e-4, max_drift: float = 0.05, sigma_s: float = 0.1, sigma_p: float = 0.4, N_grid: int = 35):
    """
    Single MALA transition step with robust drift truncation (t-MALA):
      1. Propose state along Langevin drift: T' = T + drift(T) + sqrt(2*tau) * Z
      2. Evaluate U(T') and grad U(T') at the proposed state
      3. Compute asymmetric proposal ratio q(T|T') / q(T'|T)
      4. Accept/reject via Metropolis-Hastings criterion
    """
    triangle, U, G, key = carry
    k_prop, k_acc, k_next = random.split(key, 3)

    # 1. Langevin drift with norm truncation (Roberts & Tweedie 1996)
    G_norm = jnp.linalg.norm(G)
    scale = jnp.minimum(1.0, max_drift / (tau * G_norm + 1e-8))
    drift = tau * G * scale

    noise = random.normal(k_prop, triangle.shape)
    triangle_prop = triangle + drift + jnp.sqrt(2.0 * tau) * noise
    triangle_prop = triangle_prop - jnp.mean(triangle_prop, axis=0, keepdims=True)

    # 2. Geometric safeguard
    valid = is_valid_triangle(triangle_prop)
    safe_triangle_prop = jnp.where(valid, triangle_prop, triangle)

    # 3. Energy and gradient at proposed state
    target_fn = lambda t: log_posterior(t, log_target_spectrum, sigma_s, sigma_p, N_grid)
    U_prop, G_prop = jax.value_and_grad(target_fn)(safe_triangle_prop)
    G_prop = G_prop - jnp.mean(G_prop, axis=0, keepdims=True)

    G_prop_norm = jnp.linalg.norm(G_prop)
    scale_prop = jnp.minimum(1.0, max_drift / (tau * G_prop_norm + 1e-8))
    drift_prop = tau * G_prop * scale_prop

    # 4. Forward and reverse Langevin proposal log-probabilities:
    log_q_fwd = - jnp.sum((triangle_prop - triangle - drift)**2) / (4.0 * tau)
    log_q_rev = - jnp.sum((triangle - triangle_prop - drift_prop)**2) / (4.0 * tau)

    # 5. Hastings acceptance ratio
    log_alpha = U_prop - U + log_q_rev - log_q_fwd
    log_alpha = jnp.minimum(0.0, log_alpha)

    coin = random.uniform(k_acc, ())
    accepted = valid & (jnp.log(coin) <= log_alpha)

    # 6. State update (carrying forward U_prop and G_prop for the next step)
    new_triangle = jnp.where(accepted, triangle_prop, triangle)
    new_U = jnp.where(accepted, U_prop, U)
    new_G = jnp.where(accepted, G_prop, G)

    return [new_triangle, new_U, new_G, k_next], (accepted, new_triangle, new_U)


def test_run(log_target_spectrum, key, tau: float = 4e-4, max_drift: float = 0.05, sigma_s: float = 0.1, sigma_p: float = 0.4, N_steps: int = 1000, N_grid: int = 35):
    """
    Runs MALA on the posterior starting from an independent draw from the prior.
    """
    k_init, k_chain = random.split(key)

    # Initialize from an independent sample from the prior
    triangle = sample_prior_mcmc(k_init, N_steps=300, sigma_p=sigma_p, sigma_t=0.08)

    # Initial energy and gradient
    target_fn = lambda t: log_posterior(t, log_target_spectrum, sigma_s, sigma_p, N_grid)
    U0, G0 = jax.value_and_grad(target_fn)(triangle)
    G0 = G0 - jnp.mean(G0, axis=0, keepdims=True)

    scan_step = partial(
        mala_step,
        log_target_spectrum=log_target_spectrum,
        tau=tau,
        max_drift=max_drift,
        sigma_s=sigma_s,
        sigma_p=sigma_p,
        N_grid=N_grid,
    )

    carry = [triangle, U0, G0, k_chain]
    carry, (acc, T_chain, U_chain) = scan(scan_step, carry, jnp.arange(N_steps))

    T = jnp.concatenate([jnp.array([triangle]), T_chain])
    return T, acc, U_chain


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
    target_triangles = jax.jit(vmap(lambda k: sample_prior_mcmc(k, N_steps=400, sigma_p=0.4, sigma_t=0.08)))(target_keys)
    target_spectra = jax.jit(vmap(lambda t: get_triangle_spectrum(t, N=35)[:2]))(target_triangles)

    for i in range(num_samples):
        print(f"Target {i}: spectrum = [{target_spectra[i, 0]:.4f}, {target_spectra[i, 1]:.4f}]")
    print()

    # Test MALA on the resulting spectra starting from independent samples from prior
    for i in range(num_samples):
        target_spec = target_spectra[i]
        log_target_spec = jnp.log(target_spec)

        ts = time.time()
        T, acc, U_chain = test_run(log_target_spec, init_keys[i], tau=6e-4, sigma_s=0.1, sigma_p=0.4, N_steps=1000)
        tf = time.time()

        # Compute spectra along chain for error tracking
        chain_spectra = jax.jit(vmap(lambda t: get_triangle_spectrum(t, N=35)[:2]))(T)
        S_errors = jnp.linalg.norm(chain_spectra / target_spec.reshape(1, -1) - 1.0, axis=1)
        acc_rate = float(jnp.mean(acc))

        print(f"sample {i}, time {tf - ts:.4f}, acceptance rate {acc_rate:.2%}")
        print("relative errors", *[f"{e:.4f}" for e in S_errors[:3]], "...", *[f"{e:.4f}" for e in S_errors[-3:]])

        # Report invariant metrics of the target vs sampled posterior
        target_s, target_A, target_r, target_R = get_invariants(target_triangles[i])
        post_s, post_A, post_r, post_R = vmap(get_invariants)(T[500:])  # post-burnin
        print(f"  Target invariants:    Area = {target_A:.4f}, s = {target_s:.4f}, inradius r = {target_r:.4f}, R = {target_R:.4f}")
        print(f"  Posterior (post 500): Area = {jnp.mean(post_A):.4f} +/- {jnp.std(post_A):.4f}, inradius r = {jnp.mean(post_r):.4f} +/- {jnp.std(post_r):.4f}")
        print()
