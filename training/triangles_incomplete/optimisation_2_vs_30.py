"""
Direct Shape Optimization for Triangular Drum Recovery: 2 vs. 30 Eigenvalues.

Uses pure-JAX FEM eigensolver (differentiating end-to-end through generalized eigensolver)
to recover triangular drum geometry from 2 eigenvalues vs 30 eigenvalues on multimodal drums.
"""

import argparse
import os
import sys
import time

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# Ensure local imports work when run from training/triangles_incomplete
_cur_dir = os.path.dirname(os.path.abspath(__file__))
if _cur_dir not in sys.path:
    sys.path.insert(0, _cur_dir)

import jax
import jax.numpy as jnp
from jax import value_and_grad, jit, vmap
from jax.lax import scan
import numpy as np
import optax
import scipy.linalg

import MCMC_search_for_spectrum as rw_mcmc
from drums.data import preprocessing
from regression_2_vs_30 import MU1_OBTUSE, MU2_ACUTE, get_or_generate_dataset

R = rw_mcmc.MU_TEMPLATE
IDX = preprocessing.augment(np.arange(3, dtype=float)[:, None])[..., 0].astype(int)


def relative_error(target, prediction):
    v1_ = target - jnp.mean(target, axis=0, keepdims=True)
    v2_ = prediction - jnp.mean(prediction, axis=0, keepdims=True)
    U, S, Vt = jnp.linalg.svd(v2_.T @ v1_, full_matrices=False)
    prediction_aligned = v2_ @ U @ Vt
    return jnp.mean(jnp.linalg.norm(v1_ - prediction_aligned, axis=1) / jnp.linalg.norm(v1_, axis=1))


def invariant_rel_dist(target, prediction):
    return jnp.min(vmap(lambda i: relative_error(target, prediction[i]))(IDX))


def triangle_from_p(p):
    A = jnp.array([[jnp.exp(p[0]), p[1]], [0.0, jnp.exp(p[2])]])
    return R @ A


def get_initial_p(start_drum):
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


def make_batch_optimizer(n_eig, steps=300, lr=1e-1, mesh_n=35):
    def loss_fn(p, target_spec):
        t = triangle_from_p(p)
        spec = rw_mcmc.get_triangle_spectrum(t, N=mesh_n)[:n_eig]
        return jnp.sum((jnp.log(spec) - jnp.log(target_spec[:n_eig])) ** 2)

    val_and_grad_fn = value_and_grad(loss_fn)
    schedule = optax.cosine_decay_schedule(lr, steps)
    optim = optax.adam(schedule)

    def optimize_one(target_spec, p_init):
        opt_state = optim.init(p_init)

        def step(carry, _):
            p, opt_s = carry
            l_val, g = val_and_grad_fn(p, target_spec)
            updates, opt_s = optim.update(g, opt_s)
            p = optax.apply_updates(p, updates)
            return (p, opt_s), l_val

        (p_fin, _), losses = scan(step, (p_init, opt_state), None, length=steps)
        return p_fin, losses[-1]

    return jit(vmap(optimize_one))


def evaluate_optimization(data_path="mixture_prior_triangle_drums.npz", n_train=900, steps=300, lr=1e-1, batch_size=20):
    triangles, spectra30, labels = get_or_generate_dataset(data_path=data_path)
    test_triangles = triangles[n_train:]
    test_labels = labels[n_train:]
    test_specs30 = spectra30[n_train:]
    test_specs2 = test_specs30[:, :2]
    n_test = len(test_triangles)

    p_init1 = get_initial_p(MU1_OBTUSE)
    p_init2 = get_initial_p(MU2_ACUTE)

    opt_batch_2 = make_batch_optimizer(2, steps=steps, lr=lr)
    opt_batch_30 = make_batch_optimizer(30, steps=steps, lr=lr)

    def run_optimization(opt_fn, target_specs):
        p_fin_all = []
        for i in range(0, n_test, batch_size):
            b_specs = target_specs[i : i + batch_size]
            b_len = b_specs.shape[0]
            p_inits_1 = jnp.broadcast_to(p_init1, (b_len, 3))
            p_inits_2 = jnp.broadcast_to(p_init2, (b_len, 3))
            p_fin_1, l_1 = opt_fn(b_specs, p_inits_1)
            p_fin_2, l_2 = opt_fn(b_specs, p_inits_2)
            p_fin_1.block_until_ready()
            p_fin_2.block_until_ready()
            pick_1 = (l_1 < l_2)[:, None]
            p_best = jnp.where(pick_1, p_fin_1, p_fin_2)
            p_fin_all.append(np.array(p_best))
        return np.concatenate(p_fin_all, axis=0)

    print("\nRunning Direct Shape Optimization for 2 Eigenvalues...")
    t0 = time.time()
    p_fin_2 = run_optimization(opt_batch_2, test_specs2)
    t_2 = time.time() - t0
    rec_drums_2 = vmap(triangle_from_p)(jnp.array(p_fin_2))
    shape_err_2 = np.array(vmap(invariant_rel_dist)(test_triangles, rec_drums_2))
    print(f"2 Eigenvalues completed in {t_2:.2f} s ({t_2/n_test:.2f} s/sample)")

    print("\nRunning Direct Shape Optimization for 30 Eigenvalues...")
    t0 = time.time()
    p_fin_30 = run_optimization(opt_batch_30, test_specs30)
    t_30 = time.time() - t0
    rec_drums_30 = vmap(triangle_from_p)(jnp.array(p_fin_30))
    shape_err_30 = np.array(vmap(invariant_rel_dist)(test_triangles, rec_drums_30))
    print(f"30 Eigenvalues completed in {t_30:.2f} s ({t_30/n_test:.2f} s/sample)")

    d2_mu1 = np.array(vmap(lambda p: rw_mcmc.invariant_distance(MU1_OBTUSE, p))(rec_drums_2))
    d2_mu2 = np.array(vmap(lambda p: rw_mcmc.invariant_distance(MU2_ACUTE, p))(rec_drums_2))
    acc2 = float(np.mean(np.where(d2_mu1 < d2_mu2, 0, 1) == test_labels))

    d30_mu1 = np.array(vmap(lambda p: rw_mcmc.invariant_distance(MU1_OBTUSE, p))(rec_drums_30))
    d30_mu2 = np.array(vmap(lambda p: rw_mcmc.invariant_distance(MU2_ACUTE, p))(rec_drums_30))
    acc30 = float(np.mean(np.where(d30_mu1 < d30_mu2, 0, 1) == test_labels))

    print("\n" + "=" * 70)
    print("DIRECT INVERSE SHAPE OPTIMIZATION: 2 vs. 30 EIGENVALUES")
    print("=" * 70)
    print(f"{'Metric':<35} | {'2 Eigenvalues':<14} | {'30 Eigenvalues':<14}")
    print("-" * 70)
    print(f"{'Median Shape Error (%)':<35} | {np.median(shape_err_2)*100:<14.2f} | {np.median(shape_err_30)*100:<14.2f}")
    print(f"{'Mean Shape Error (%)':<35} | {np.mean(shape_err_2)*100:<14.2f} | {np.mean(shape_err_30)*100:<14.2f}")
    print(f"{'25th Percentile (Q1) (%)':<35} | {np.percentile(shape_err_2, 25)*100:<14.2f} | {np.percentile(shape_err_30, 25)*100:<14.2f}")
    print(f"{'75th Percentile (Q3) (%)':<35} | {np.percentile(shape_err_2, 75)*100:<14.2f} | {np.percentile(shape_err_30, 75)*100:<14.2f}")
    print(f"{'Min Shape Error (%)':<35} | {np.min(shape_err_2)*100:<14.2f} | {np.min(shape_err_30)*100:<14.2f}")
    print(f"{'Max Shape Error (%)':<35} | {np.max(shape_err_2)*100:<14.2f} | {np.max(shape_err_30)*100:<14.2f}")
    print(f"{'Mode Accuracy (Obtuse vs Acute)':<35} | {acc2*100:<13.1f}% | {acc30*100:<13.1f}%")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="mixture_prior_triangle_drums.npz")
    parser.add_argument("--n-train", type=int, default=900)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-1)
    parser.add_argument("--batch-size", type=int, default=20)
    args = parser.parse_args()
    evaluate_optimization(data_path=args.data, n_train=args.n_train, steps=args.steps, lr=args.lr, batch_size=args.batch_size)
