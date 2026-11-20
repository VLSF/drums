import numpy as np
import triangle as tr
import jax
import jax.numpy as jnp
from jax import random, jit, value_and_grad, vmap
import optax
from drums.data import preprocessing


def align(v1, v2):
    """Centre v1 and v2 and rotate/reflect v2 onto v1 (orthogonal Procrustes)."""
    v1_ = v1 - jnp.mean(v1, axis=0, keepdims=True)
    v2_ = v2 - jnp.mean(v2, axis=0, keepdims=True)
    U, S, Vt = jnp.linalg.svd(v2_.T @ v1_, full_matrices=False)
    R = U @ Vt
    return v1_, v2_ @ R


def distance(target, prediction):
    target_, prediction_ = align(target, prediction)
    return jnp.mean(jnp.linalg.norm(target_ - prediction_, axis=1))


def invariant_distance(target, prediction, idx):
    return jnp.min(vmap(lambda ind: distance(target, prediction[ind]))(idx))


def local_matrices(xy):  # xy: (3, 2)
    x, y = xy[:, 0], xy[:, 1]
    b = jnp.roll(y, -1) - jnp.roll(y, -2)  # y_{i+1} - y_{i+2}
    c = jnp.roll(x, -2) - jnp.roll(x, -1)  # x_{i+2} - x_{i+1}
    A = 0.5 * ((x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0]))
    Ke = (jnp.outer(b, b) + jnp.outer(c, c)) / (4 * A)
    Me = A / 12 * (jnp.ones((3, 3)) + jnp.eye(3))
    return Ke, Me


def get_reference_data(R):
    l = R.shape[0]
    S = np.array([(i, (i + 1) % l) for i in range(l)])
    domain = dict(vertices=np.array(R), segments=S)
    mesh = tr.triangulate(domain, 'pq30a0.001')
    X = mesh['vertices']  # (N, 2)
    T = mesh['triangles']

    E = np.sort(np.vstack([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]), axis=1)
    uniq, cnt = np.unique(E, axis=0, return_counts=True)
    bnd = np.unique(uniq[cnt == 1])
    free = np.setdiff1d(np.arange(len(X)), bnd)

    Ke, Me = jax.jit(jax.vmap(local_matrices))(jnp.asarray(X)[T])  # (Ne,3,3) each
    Ke, Me = np.asarray(Ke), np.asarray(Me)

    rows = np.broadcast_to(T[:, :, None], Ke.shape).ravel()
    cols = np.broadcast_to(T[:, None, :], Ke.shape).ravel()

    return (
        jnp.array(rows),
        jnp.array(cols),
        jnp.array(T),
        jnp.array(free),
        jnp.array(X),
    )


def get_spectrum(p, rows, cols, T, free, X):
    A = jnp.array([[jnp.exp(p[0]), p[1]], [0.0, jnp.exp(p[2])]])

    x = X @ A

    ke, me = jax.vmap(local_matrices)(x[T])

    N = X.shape[0]
    K = jnp.zeros((N, N)).at[rows, cols].add(ke.ravel())
    M = jnp.zeros((N, N)).at[rows, cols].add(me.ravel())

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


def make_d_loss(n_eig):
    def loss_fn(p, target, rows, cols, T, free, X):
        spectrum = get_spectrum(p, rows, cols, T, free, X)
        return jnp.sum((jnp.log(spectrum[:n_eig]) - jnp.log(target[:n_eig])) ** 2)

    return jit(value_and_grad(loss_fn))


def get_canonical_reference():
    R = jnp.stack(
        [
            jnp.array([0, 1]),
            jnp.array([-jnp.sqrt(3) / 2, -1 / 2]),
            jnp.array([jnp.sqrt(3) / 2, -1 / 2]),
        ],
        axis=0,
    )
    R_ = jnp.linalg.inv(R[:2])
    return R @ R_


def run_optimization(
    sample_idx,
    n_eig,
    data,
    seed=32,
    steps=500,
    lr=1e-1,
    ref_data=None,
    R=None,
    idx=None,
):
    if R is None:
        R = get_canonical_reference()
    if idx is None:
        idx = preprocessing.augment(jnp.arange(R.shape[0], dtype=float)[:, None])[
            ..., 0
        ].astype(int)
    if ref_data is None:
        ref_data = get_reference_data(R)

    rows, cols, T, free, X = ref_data

    keys = random.split(random.PRNGKey(seed))
    triangle = jnp.array(data['perimeters'][sample_idx])
    target_spectrum = jnp.array(data['spectra'][sample_idx])

    p = 1e-2 * random.normal(keys[0], (3,))
    optim = optax.adam(lr)
    opt_state = optim.init(p)

    d_loss = make_d_loss(n_eig)

    for i in range(steps):
        loss_val, grads = d_loss(p, target_spectrum, rows, cols, T, free, X)
        updates, opt_state = optim.update(grads, opt_state)
        p = optax.apply_updates(p, updates)

    A = jnp.array([[jnp.exp(p[0]), p[1]], [0.0, jnp.exp(p[2])]])
    r = R @ A

    final_error = float(invariant_distance(triangle, r, idx))
    final_loss = float(loss_val)

    return final_loss, final_error


if __name__ == '__main__':
    data = jnp.load('triangle_drums.npz')
    R = get_canonical_reference()
    idx = preprocessing.augment(jnp.arange(R.shape[0], dtype=float)[:, None])[
        ..., 0
    ].astype(int)
    ref_data = get_reference_data(R)

    print('=' * 60)
    print('Baseline Experiments')
    print('=' * 60)

    # Experiment 1: sample=45, n_eig=10
    loss_1, err_1 = run_optimization(
        45, 10, data, seed=32, ref_data=ref_data, R=R, idx=idx
    )
    print(f'Experiment 1 (sample=45, n_eig=10, seed=32):')
    print(f'  Final Loss: {loss_1:.6e}, Invariant Error: {err_1:.6f}')

    # Experiment 2: sample=989, n_eig=10
    loss_2, err_2 = run_optimization(
        989, 10, data, seed=32, ref_data=ref_data, R=R, idx=idx
    )
    print(f'Experiment 2 (sample=989, n_eig=10, seed=32):')
    print(f'  Final Loss: {loss_2:.6e}, Invariant Error: {err_2:.6f}')

    # Experiment 3: sample=989, n_eig=2
    loss_3, err_3 = run_optimization(
        989, 2, data, seed=32, ref_data=ref_data, R=R, idx=idx
    )
    print(f'Experiment 3 (sample=989, n_eig=2, seed=32):')
    print(f'  Final Loss: {loss_3:.6e}, Invariant Error: {err_3:.6f}')

    print('\n' + '=' * 60)
    print('Seed Sensitivity Runs (n_eig=10)')
    print('=' * 60)

    for sample in [45, 989]:
        for s in [32, 42, 52]:
            loss_val, err_val = run_optimization(
                sample, 10, data, seed=s, ref_data=ref_data, R=R, idx=idx
            )
            print(
                f'Sample {sample}, Seed {s:2d}: Final Loss = {loss_val:.6e}, Invariant Error = {err_val:.6f}'
            )

    print('\n' + '=' * 60)
    print('Number of Eigenvalues Sensitivity Runs (seed=32)')
    print('=' * 60)

    for sample in [45, 989]:
        print(f'\n--- Sample {sample} ---')
        for n in [1, 2, 3, 4, 5, 10]:
            loss_val, err_val = run_optimization(
                sample, n, data, seed=32, ref_data=ref_data, R=R, idx=idx
            )
            print(
                f'n_eig = {n:2d}: Final Loss = {loss_val:.6e}, Invariant Error = {err_val:.6f}'
            )
