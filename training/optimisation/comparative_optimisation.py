import argparse
import json
import os
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import jit, random, value_and_grad, vmap

from drums.data import preprocessing
from drums.models.ring_transformer import RingTransformer
from drums.models.transformer import DrumTransformer


# ---------------------------------------------------------------------------
# Metric: Procrustes all-orderings relative error in %
# ---------------------------------------------------------------------------

def align(v1, v2):
    """Centre v1 and v2 and rotate/reflect v2 onto v1 (orthogonal Procrustes)."""
    v1_ = v1 - jnp.mean(v1, axis=0, keepdims=True)
    v2_ = v2 - jnp.mean(v2, axis=0, keepdims=True)
    U, S, Vt = jnp.linalg.svd(v2_.T @ v1_, full_matrices=False)
    R = U @ Vt
    return v1_, v2_ @ R


def relative_error(target, prediction):
    target_, prediction_ = align(target, prediction)
    return jnp.mean(
        jnp.linalg.norm(target_ - prediction_, axis=1)
        / jnp.linalg.norm(target_, axis=1)
    )


def orderings(N_v=3):
    return preprocessing.augment(np.arange(N_v, dtype=float)[:, None])[..., 0].astype(int)


def compute_rel_error_percent(target, prediction, idx):
    errs = vmap(lambda ind: relative_error(target, prediction[ind]))(idx)
    return float(jnp.min(errs)) * 100.0


# ---------------------------------------------------------------------------
# 1. Regression (DrumTransformer)
# ---------------------------------------------------------------------------

def train_and_eval_regression(triangles, spectra, test_indices, idx):
    print("=" * 60)
    print("Evaluating Regression (DrumTransformer)...")
    print("=" * 60)

    N_train = 900
    N_updates = 5000
    N_batch = 64
    lr = 1e-3

    # Normalize targets by max vertex norm in training set (as in procrustes_all_orderings.py)
    scale = float(jnp.max(jnp.linalg.norm(triangles[:N_train], axis=-1)))
    targets = triangles[:N_train] / scale
    features = spectra[:N_train]

    test_targets = triangles[test_indices]
    test_features = spectra[test_indices]

    k1, k2 = random.split(random.PRNGKey(17))
    model = DrumTransformer(3, 128, 30, k1, L=5, n_heads=4, mlp_ratio=4)

    warmup_steps = 500
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=lr,
        warmup_steps=warmup_steps,
        decay_steps=N_updates,
        end_value=1e-5,
    )
    optim = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(schedule, weight_decay=1e-4),
    )
    opt_state = optim.init(model)

    def regression_loss_single(m, feat, targ):
        pred = m(feat)
        errs = vmap(lambda ind: relative_error(targ, pred[ind]))(idx)
        return jnp.min(errs)

    def batch_loss(m, f_batch, t_batch):
        return jnp.mean(vmap(regression_loss_single, in_axes=(None, 0, 0))(m, f_batch, t_batch))

    compute_loss_and_grads = eqx.filter_value_and_grad(batch_loss)

    inds = random.choice(k2, N_train, (N_updates, N_batch))

    @eqx.filter_jit
    def step_fn(carry, batch_inds):
        m_curr, opt_curr = carry
        loss_val, grads = compute_loss_and_grads(m_curr, features[batch_inds], targets[batch_inds])
        updates, opt_curr = optim.update(grads, opt_curr, eqx.filter(m_curr, eqx.is_array))
        m_curr = eqx.apply_updates(m_curr, updates)
        return [m_curr, opt_curr], loss_val

    print(f"  Training DrumTransformer on {N_train} samples ({N_updates} updates)...")
    t_train0 = time.time()
    carry = [model, opt_state]
    for b_inds in inds:
        carry, loss_val = step_fn(carry, b_inds)
    model = carry[0]
    loss_val.block_until_ready()
    t_train1 = time.time()
    print(f"  Training finished in {t_train1 - t_train0:.2f}s. Final loss: {float(loss_val):.6f}")

    # Benchmark inference time on the 100 test samples
    predict_batch = jit(vmap(lambda f: model(f) * scale))

    # Warmup with full batch to compile
    _ = predict_batch(test_features).block_until_ready()

    n_reps = 10
    t_infer0 = time.time()
    for _ in range(n_reps):
        preds = predict_batch(test_features)
    preds.block_until_ready()
    t_infer1 = time.time()

    time_per_sample = (t_infer1 - t_infer0) / (n_reps * len(test_indices))

    # Compute errors
    errors = [
        compute_rel_error_percent(test_targets[i], preds[i], idx)
        for i in range(len(test_indices))
    ]

    return {
        "min": float(np.min(errors)),
        "median": float(np.median(errors)),
        "mean": float(np.mean(errors)),
        "max": float(np.max(errors)),
        "time_per_sample": float(time_per_sample),
    }


# ---------------------------------------------------------------------------
# 2. Exact Direct Optimization (Differentiating through FEM solver)
# ---------------------------------------------------------------------------

def local_matrices(xy):
    x, y = xy[:, 0], xy[:, 1]
    b = jnp.roll(y, -1) - jnp.roll(y, -2)
    c = jnp.roll(x, -2) - jnp.roll(x, -1)
    A = 0.5 * ((x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0]))
    Ke = (jnp.outer(b, b) + jnp.outer(c, c)) / (4 * A)
    Me = A / 12 * (jnp.ones((3, 3)) + jnp.eye(3))
    return Ke, Me


def get_canonical_reference():
    R = jnp.stack([
        jnp.array([0, 1]),
        jnp.array([-jnp.sqrt(3) / 2, -1 / 2]),
        jnp.array([jnp.sqrt(3) / 2, -1 / 2]),
    ], axis=0)
    R_ = jnp.linalg.inv(R[:2])
    return R @ R_


def get_reference_data(R):
    import triangle as tr
    l = R.shape[0]
    S = np.array([(i, (i + 1) % l) for i in range(l)])
    domain = dict(vertices=np.array(R), segments=S)
    mesh = tr.triangulate(domain, 'pq30a0.001')
    X = mesh['vertices']
    T = mesh['triangles']

    E = np.sort(np.vstack([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]), axis=1)
    uniq, cnt = np.unique(E, axis=0, return_counts=True)
    bnd = np.unique(uniq[cnt == 1])
    free = np.setdiff1d(np.arange(len(X)), bnd)

    Ke, Me = jax.jit(jax.vmap(local_matrices))(jnp.asarray(X)[T])
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


def get_fem_spectrum(p, rows, cols, T, free, X):
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
    return jnp.linalg.eigvalsh(C)


def eval_exact_optimization(triangles, spectra, test_indices, start_indices, idx, steps=500, lr=1e-1):
    print("=" * 60)
    print("Evaluating Exact FEM Optimization on 100 test samples...")
    print("=" * 60)

    R = get_canonical_reference()
    ref_data = get_reference_data(R)
    rows, cols, T, free, X = ref_data

    def fem_loss(p, target_spec):
        spec = get_fem_spectrum(p, rows, cols, T, free, X)
        return jnp.sum((jnp.log(spec[:10]) - jnp.log(target_spec[:10])) ** 2)

    d_fem_loss = jit(value_and_grad(fem_loss))
    optim = optax.adam(lr)

    def optimize_one(target_spec, p_init):
        opt_state = optim.init(p_init)
        def step_fn(carry, _):
            p, opt_curr = carry
            loss_val, grads = d_fem_loss(p, target_spec)
            updates, opt_curr = optim.update(grads, opt_curr)
            p = optax.apply_updates(p, updates)
            return (p, opt_curr), loss_val
        (p_fin, _), _ = jax.lax.scan(step_fn, (p_init, opt_state), None, length=steps)
        return p_fin

    optimize_one_jit = jit(optimize_one)

    # Initial parameter p_init mapping canonical R to the chosen start drum via RQ decomposition
    def get_initial_p(start_drum):
        import scipy.linalg
        mean = np.mean(np.array(start_drum), axis=0, keepdims=True)
        drum_c = np.array(start_drum) - mean
        A_mat = np.linalg.solve(np.array(R[:2]), drum_c[:2])
        U, Q = scipy.linalg.rq(A_mat)
        if U[0, 0] < 0:
            U[:, 0] *= -1
            Q[0, :] *= -1
        if U[1, 1] < 0:
            U[:, 1] *= -1
            Q[1, :] *= -1
        p0 = float(np.log(max(U[0, 0], 1e-8)))
        p1 = float(U[0, 1])
        p2 = float(np.log(max(U[1, 1], 1e-8)))
        return jnp.array([p0, p1, p2])

    # Warmup JIT
    p_dummy = jnp.zeros(3)
    _ = optimize_one_jit(spectra[test_indices[0]], p_dummy).block_until_ready()

    errors = []
    times = []

    for i, (t_idx, s_idx) in enumerate(zip(test_indices, start_indices)):
        target_drum = triangles[t_idx]
        target_spec = spectra[t_idx]
        start_drum = triangles[s_idx]

        p_init = get_initial_p(start_drum)

        t0 = time.time()
        p_final = optimize_one_jit(target_spec, p_init)
        p_final.block_until_ready()
        t1 = time.time()
        times.append(t1 - t0)

        A_final = jnp.array([[jnp.exp(p_final[0]), p_final[1]], [0.0, jnp.exp(p_final[2])]])
        pred_drum = R @ A_final

        err = compute_rel_error_percent(target_drum, pred_drum, idx)
        errors.append(err)

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Processed {i+1}/{len(test_indices)} drums | Last err: {err:.3f}% | Avg time/sample: {np.mean(times):.2f}s")

    return {
        "min": float(np.min(errors)),
        "median": float(np.median(errors)),
        "mean": float(np.mean(errors)),
        "max": float(np.max(errors)),
        "time_per_sample": float(np.mean(times)),
    }


# ---------------------------------------------------------------------------
# 3. Surrogate Optimization (RingTransformer)
# ---------------------------------------------------------------------------

def loss_single_surrogate(model, triangle, spectrum):
    spectrum_ = model.forward_flip(jnp.ones((1,)), triangle)[0] ** 2
    return jnp.sum((jnp.log(spectrum_) - jnp.log(spectrum)) ** 2)


def loss_batch_surrogate(model, triangles, spectra):
    return jnp.mean(vmap(loss_single_surrogate, in_axes=(None, 0, 0))(model, triangles, spectra))


compute_surrogate_loss_and_grads = eqx.filter_value_and_grad(loss_batch_surrogate)


def train_surrogate_model(triangles, spectra, n_updates, key, batch_size=10, lr=1e-4):
    n = triangles.shape[1]
    m = 1
    N_inv = spectra.shape[1]
    k_model, k_train = random.split(key)

    model = RingTransformer(n, m, 128, N_inv, k_model, L=4)
    optim = optax.adam(lr)
    opt_state = optim.init(model)

    N_train = triangles.shape[0]
    inds = random.choice(k_train, N_train, (n_updates, batch_size))

    def step_fn(carry, batch_inds):
        m_curr, opt_curr = carry
        loss_val, grads = compute_surrogate_loss_and_grads(
            m_curr, triangles[batch_inds], spectra[batch_inds]
        )
        updates, opt_curr = optim.update(grads, opt_curr, eqx.filter(m_curr, eqx.is_array))
        m_curr = eqx.apply_updates(m_curr, updates)
        return (m_curr, opt_curr), loss_val

    @eqx.filter_jit
    def run_training(m_init, opt_init):
        (m_final, _), history = jax.lax.scan(step_fn, (m_init, opt_init), inds)
        return m_final, history

    t0 = time.time()
    model, history = run_training(model, opt_state)
    history.block_until_ready()
    t1 = time.time()
    print(f"  Surrogate training finished in {t1 - t0:.2f}s. Final loss: {history[-1]:.6f}")
    return model


def eval_surrogate_optimization(model, triangles, spectra, test_indices, start_indices, idx, steps=1000, lr=1e-1):
    def surrogate_shape_loss(drum, target_spectrum):
        pred_spec = model.forward_flip(jnp.ones((1,)), drum)[0] ** 2
        return jnp.sum((jnp.log(target_spectrum) - jnp.log(pred_spec)) ** 2)

    compute_grad = jit(value_and_grad(surrogate_shape_loss))
    optim = optax.adam(lr)

    def optimize_one(target_spec, d_init):
        opt_state = optim.init(d_init)
        def step_fn(carry, _):
            drum_curr, opt_curr = carry
            loss_val, grads = compute_grad(drum_curr, target_spec)
            updates, opt_curr = optim.update(grads, opt_curr)
            drum_curr = optax.apply_updates(drum_curr, updates)
            return (drum_curr, opt_curr), loss_val
        (d_final, _), _ = jax.lax.scan(step_fn, (d_init, opt_state), None, length=steps)
        return d_final

    optimize_one_jit = jit(optimize_one)

    # Warmup JIT
    d_dummy = triangles[start_indices[0]]
    _ = optimize_one_jit(spectra[test_indices[0]], d_dummy).block_until_ready()

    errors = []
    times = []

    for i, (t_idx, s_idx) in enumerate(zip(test_indices, start_indices)):
        target_drum = triangles[t_idx]
        target_spec = spectra[t_idx]
        start_drum = triangles[s_idx]

        t0 = time.time()
        pred_drum = optimize_one_jit(target_spec, start_drum)
        pred_drum.block_until_ready()
        t1 = time.time()
        times.append(t1 - t0)

        err = compute_rel_error_percent(target_drum, pred_drum, idx)
        errors.append(err)

    return {
        "min": float(np.min(errors)),
        "median": float(np.median(errors)),
        "mean": float(np.mean(errors)),
        "max": float(np.max(errors)),
        "time_per_sample": float(np.mean(times)),
    }


# ---------------------------------------------------------------------------
# Main Runner
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["all", "regression", "exact", "surrogate_900", "surrogate_5000"], default="all")
    parser.add_argument("--out", default="task_2_results.json")
    args = parser.parse_args()

    small_data = jnp.load("triangle_drums.npz")
    triangles_sm = jnp.array(small_data["perimeters"])
    spectra_sm = jnp.array(small_data["spectra"])

    ext_data = jnp.load("triangle_drums_extended/triangle_drums.npz")
    triangles_ext = jnp.array(ext_data["perimeters"])
    spectra_ext = jnp.array(ext_data["spectra"])

    test_indices = np.arange(900, 1000)
    rng_start = np.random.default_rng(seed=42)
    start_indices = rng_start.choice(np.arange(0, 900), size=len(test_indices), replace=True)

    idx = orderings(3)

    results = {}
    if os.path.exists(args.out):
        try:
            with open(args.out, "r") as f:
                results = json.load(f)
        except Exception:
            results = {}

    # 1. Regression
    if args.mode in ["all", "regression"]:
        results["regression"] = train_and_eval_regression(triangles_sm, spectra_sm, test_indices, idx)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)

    # 2. Surrogate N=900
    if args.mode in ["all", "surrogate_900"]:
        print("=" * 60)
        print("Training Surrogate Model N=900 (15,000 updates, 166.7 epochs)...")
        print("=" * 60)
        m_900 = train_surrogate_model(triangles_sm[:900], spectra_sm[:900], n_updates=15000, key=random.PRNGKey(44))
        results["optim. (learned, N=900)"] = eval_surrogate_optimization(
            m_900, triangles_sm, spectra_sm, test_indices, start_indices, idx
        )
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)

    # 3. Surrogate N=5000
    if args.mode in ["all", "surrogate_5000"]:
        print("=" * 60)
        print("Training Surrogate Model N=5000 (83,333 updates, 166.7 epochs)...")
        print("=" * 60)
        m_5000 = train_surrogate_model(triangles_ext, spectra_ext, n_updates=83333, key=random.PRNGKey(44))
        results["optim. (learned, N=5000)"] = eval_surrogate_optimization(
            m_5000, triangles_sm, spectra_sm, test_indices, start_indices, idx
        )
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)

    # 4. Exact Optimization
    if args.mode in ["all", "exact"]:
        results["optimization (exact)"] = eval_exact_optimization(
            triangles_sm, spectra_sm, test_indices, start_indices, idx
        )
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print("FINAL SUMMARY TABLE")
    print("=" * 70)
    header = f"{'Method':<28} | {'min (%)':<8} | {'median (%)':<10} | {'mean (%)':<8} | {'max (%)':<8} | {'t, s/sample':<11}"
    print(header)
    print("-" * len(header))
    for k, v in results.items():
        print(f"{k:<28} | {v['min']:<8.3f} | {v['median']:<10.3f} | {v['mean']:<8.3f} | {v['max']:<8.3f} | {v['time_per_sample']:<11.5f}")


if __name__ == "__main__":
    main()
