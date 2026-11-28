"""
Neural Regression for Triangular Drum Recovery: 2 vs. 30 Eigenvalues.

Compares DrumTransformer performance when given an incomplete spectrum (N_eig = 2)
versus the full spectrum (N_eig = 30) on a multimodal dataset consisting of
isospectral obtuse and acute triangles.
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

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import random, vmap
from jax.lax import scan

import MCMC_search_for_spectrum as rw_mcmc
from drums.data import preprocessing
from drums.models import transformer

# Mode templates sharing lambda ≈ [50.005, 79.989]
MU1_OBTUSE = jnp.array([
    [-1.3453413, -0.188223],
    [ 1.3468970, -0.188223],
    [-0.00155556, 0.3764459]
], dtype=jnp.float32)

MU2_ACUTE = jnp.array([
    [-0.31832823, -0.7295851],
    [ 0.31864205, -0.7295851],
    [-0.00031383,  1.4591701]
], dtype=jnp.float32)

IDX = preprocessing.augment(np.arange(3, dtype=float)[:, None])[..., 0].astype(int)


def relative_error(target, prediction):
    v1_ = target - jnp.mean(target, axis=0, keepdims=True)
    v2_ = prediction - jnp.mean(prediction, axis=0, keepdims=True)
    U, S, Vt = jnp.linalg.svd(v2_.T @ v1_, full_matrices=False)
    prediction_aligned = v2_ @ U @ Vt
    return jnp.mean(jnp.linalg.norm(v1_ - prediction_aligned, axis=1) / jnp.linalg.norm(v1_, axis=1))


def invariant_rel_dist(target, prediction):
    return jnp.min(vmap(lambda i: relative_error(target, prediction[i]))(IDX))


def sample_prior_mcmc_mode(key, template, n_steps=300, sigma_p=0.15, sigma_t=0.04):
    def step(carry, _):
        t, k = carry
        k_prop, k_accept, k_next = random.split(k, 3)
        t_prop = t + sigma_t * random.normal(k_prop, t.shape)
        t_prop = t_prop - jnp.mean(t_prop, axis=0, keepdims=True)
        valid = rw_mcmc.is_valid_triangle(t_prop)
        safe_t_prop = jnp.where(valid, t_prop, t)
        d_curr = rw_mcmc.invariant_distance(template, t)
        d_prop = rw_mcmc.invariant_distance(template, safe_t_prop)
        log_alpha = (d_curr**2 - d_prop**2) / (2.0 * sigma_p**2)
        coin = random.uniform(k_accept, ())
        accepted = valid & (jnp.log(coin) <= jnp.minimum(0.0, log_alpha))
        t_new = jnp.where(accepted, t_prop, t)
        return (t_new, k_next), t_new
    carry = (template, key)
    carry, _ = scan(step, carry, jnp.arange(n_steps))
    return carry[0]


def get_or_generate_dataset(data_path="mixture_prior_triangle_drums.npz", n_samples=1000, seed=42, mesh_n=35):
    if os.path.exists(data_path):
        data = np.load(data_path)
        triangles = jnp.array(data["perimeters"])
        labels = np.array(data["labels"])
        print(f"Loaded existing dataset from {data_path} ({len(triangles)} samples)")
    else:
        print(f"Generating bimodal dataset ({n_samples} samples)...")
        n_per_mode = n_samples // 2
        key = random.PRNGKey(seed)
        k1, k2, k_perm = random.split(key, 3)
        fn1 = jax.jit(vmap(lambda k: sample_prior_mcmc_mode(k, MU1_OBTUSE)))
        fn2 = jax.jit(vmap(lambda k: sample_prior_mcmc_mode(k, MU2_ACUTE)))
        t1 = fn1(random.split(k1, n_per_mode))
        t2 = fn2(random.split(k2, n_per_mode))
        triangles = jnp.concatenate([t1, t2], axis=0)
        labels = np.concatenate([np.zeros(n_per_mode, dtype=np.int32), np.ones(n_per_mode, dtype=np.int32)])
        perm = np.array(random.permutation(k_perm, n_samples))
        triangles = triangles[perm]
        labels = labels[perm]
        np.savez(data_path, perimeters=np.array(triangles), labels=labels)
        print(f"Saved dataset to {data_path}")

    # Compute spectra if not present
    spec30_fn = jax.jit(vmap(lambda t: rw_mcmc.get_triangle_spectrum(t, N=mesh_n)[:30]))
    spectra30 = spec30_fn(triangles)
    spectra30.block_until_ready()
    return triangles, spectra30, labels


def train_drum_transformer(features, targets, n_train=900, n_updates=5000, batch_size=64, lr=1e-3, seed=17):
    keys = random.split(random.PRNGKey(seed), 2)
    n_eig = features.shape[1]
    model = transformer.DrumTransformer(3, 128, n_eig, keys[0], L=5, n_heads=4, mlp_ratio=4, n_drums=1)
    schedule = optax.warmup_cosine_decay_schedule(0.0, lr, 250, n_updates, 1e-5)
    optim = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(schedule, weight_decay=1e-4))
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    def batch_loss(m, fb, tb):
        return jnp.mean(vmap(lambda f, t: jnp.min(vmap(lambda i: relative_error(t, m(f)[i]))(IDX)))(fb, tb))

    compute_grads = eqx.filter_value_and_grad(batch_loss)

    @eqx.filter_jit
    def step(carry, ind):
        m, o = carry
        l, g = compute_grads(m, features[ind], targets[ind])
        u, o = optim.update(g, o, eqx.filter(m, eqx.is_array))
        m = eqx.apply_updates(m, u)
        return [m, o], l

    batch_inds = random.choice(keys[1], n_train, shape=(n_updates, batch_size))
    carry = [model, opt_state]
    t0 = time.time()
    for b in batch_inds:
        carry, l_val = step(carry, b)
    t1 = time.time()
    print(f"Training DrumTransformer (N_eig={n_eig}) finished in {t1 - t0:.2f} s. Final loss: {float(l_val):.5f}")
    return carry[0]


def evaluate_regression(data_path="mixture_prior_triangle_drums.npz", n_train=900):
    triangles, spectra30, labels = get_or_generate_dataset(data_path=data_path)
    n_test = len(triangles) - n_train
    target_scale = float(jnp.max(jnp.linalg.norm(triangles[:n_train], axis=2)))
    targets = triangles / target_scale

    test_tri = triangles[n_train:]
    test_labels = labels[n_train:]

    # Model A: 2 Eigenvalues
    print("\n--- Training Model with 2 Eigenvalues ---")
    m2 = train_drum_transformer(spectra30[:, :2], targets, n_train=n_train)
    pred_drums2 = vmap(lambda f: m2(f) * target_scale)(spectra30[n_train:, :2])
    shape_err2 = np.array(vmap(invariant_rel_dist)(test_tri, pred_drums2))

    # Model B: 30 Eigenvalues
    print("\n--- Training Model with 30 Eigenvalues ---")
    m30 = train_drum_transformer(spectra30, targets, n_train=n_train)
    pred_drums30 = vmap(lambda f: m30(f) * target_scale)(spectra30[n_train:])
    shape_err30 = np.array(vmap(invariant_rel_dist)(test_tri, pred_drums30))

    # Mode identification accuracy
    d2_mu1 = np.array(vmap(lambda p: rw_mcmc.invariant_distance(MU1_OBTUSE, p))(pred_drums2))
    d2_mu2 = np.array(vmap(lambda p: rw_mcmc.invariant_distance(MU2_ACUTE, p))(pred_drums2))
    acc2 = float(np.mean(np.where(d2_mu1 < d2_mu2, 0, 1) == test_labels))

    d30_mu1 = np.array(vmap(lambda p: rw_mcmc.invariant_distance(MU1_OBTUSE, p))(pred_drums30))
    d30_mu2 = np.array(vmap(lambda p: rw_mcmc.invariant_distance(MU2_ACUTE, p))(pred_drums30))
    acc30 = float(np.mean(np.where(d30_mu1 < d30_mu2, 0, 1) == test_labels))

    print("\n" + "=" * 70)
    print("NEURAL REGRESSION BENCHMARK: 2 vs. 30 EIGENVALUES")
    print("=" * 70)
    print(f"{'Metric':<35} | {'2 Eigenvalues':<14} | {'30 Eigenvalues':<14}")
    print("-" * 70)
    print(f"{'Median Shape Error (%)':<35} | {np.median(shape_err2)*100:<14.2f} | {np.median(shape_err30)*100:<14.2f}")
    print(f"{'Mean Shape Error (%)':<35} | {np.mean(shape_err2)*100:<14.2f} | {np.mean(shape_err30)*100:<14.2f}")
    print(f"{'25th Percentile (Q1) (%)':<35} | {np.percentile(shape_err2, 25)*100:<14.2f} | {np.percentile(shape_err30, 25)*100:<14.2f}")
    print(f"{'75th Percentile (Q3) (%)':<35} | {np.percentile(shape_err2, 75)*100:<14.2f} | {np.percentile(shape_err30, 75)*100:<14.2f}")
    print(f"{'Min Shape Error (%)':<35} | {np.min(shape_err2)*100:<14.2f} | {np.min(shape_err30)*100:<14.2f}")
    print(f"{'Max Shape Error (%)':<35} | {np.max(shape_err2)*100:<14.2f} | {np.max(shape_err30)*100:<14.2f}")
    print(f"{'Mode Accuracy (Obtuse vs Acute)':<35} | {acc2*100:<13.1f}% | {acc30*100:<13.1f}%")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="mixture_prior_triangle_drums.npz")
    parser.add_argument("--n-train", type=int, default=900)
    args = parser.parse_args()
    evaluate_regression(data_path=args.data, n_train=args.n_train)
