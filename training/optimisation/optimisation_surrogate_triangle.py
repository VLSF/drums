import time
import numpy as np
import jax
import jax.numpy as jnp
from jax import random, vmap, jit, value_and_grad
import equinox as eqx
import optax

from drums.data import preprocessing
from drums.models.ring_transformer import RingTransformer


# ---------------------------------------------------------------------------
# Invariant Geometry Metrics
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Surrogate Training
# ---------------------------------------------------------------------------

def loss_single(model, triangle, spectrum):
    spectrum_ = model.forward_flip(jnp.ones((1,)), triangle)[0] ** 2
    return jnp.sum((jnp.log(spectrum_) - jnp.log(spectrum)) ** 2)


def loss_batch(model, triangles, spectra):
    return jnp.mean(vmap(loss_single, in_axes=(None, 0, 0))(model, triangles, spectra))


compute_loss_and_grads = eqx.filter_value_and_grad(loss_batch)


def train_surrogate(
    triangles,
    spectra,
    key,
    n_updates=15000,
    batch_size=10,
    lr=1e-4,
    L=4,
    N_f=128,
):
    """Train RingTransformer surrogate model on provided (triangles, spectra)."""
    n = triangles.shape[1]
    m = 1
    N_inv = spectra.shape[1]
    k_model, k_train = random.split(key)

    model = RingTransformer(n, m, N_f, N_inv, k_model, L)
    optim = optax.adam(lr)
    opt_state = optim.init(model)

    N_train = triangles.shape[0]
    inds = random.choice(k_train, N_train, (n_updates, batch_size))

    def step_fn(carry, batch_inds):
        m_curr, opt_curr = carry
        loss_val, grads = compute_loss_and_grads(
            m_curr, triangles[batch_inds], spectra[batch_inds]
        )
        updates, opt_curr = optim.update(
            grads, opt_curr, eqx.filter(m_curr, eqx.is_array)
        )
        m_curr = eqx.apply_updates(m_curr, updates)
        return (m_curr, opt_curr), loss_val

    @eqx.filter_jit
    def run_training(m_init, opt_init):
        (m_final, _), loss_history = jax.lax.scan(
            step_fn, (m_init, opt_init), inds
        )
        return m_final, loss_history

    print(f"  Training RingTransformer on {N_train} samples ({n_updates} updates, batch size {batch_size})...")
    t0 = time.time()
    model, history = run_training(model, opt_state)
    history.block_until_ready()
    t1 = time.time()
    print(f"  Training finished in {t1 - t0:.2f}s. Initial loss: {history[0]:.4f}, Final loss: {history[-1]:.6f}")

    return model, history


def get_surrogate_spectrum_errors(model, triangles, spectra):
    """Relative L2 error between surrogate predictions and ground truth spectra."""
    get_spec = lambda t: model.forward_flip(jnp.ones((1,)), t)[0] ** 2
    pred_spectra = vmap(get_spec)(triangles)
    return jnp.linalg.norm(pred_spectra - spectra, axis=1) / jnp.linalg.norm(spectra, axis=1)


# ---------------------------------------------------------------------------
# Direct Shape Optimization via Surrogate Model
# ---------------------------------------------------------------------------

def shape_loss(drum, target_spectrum, model):
    pred_spectrum = model.forward_flip(jnp.ones((1,)), drum)[0] ** 2
    return jnp.sum((jnp.log(target_spectrum) - jnp.log(pred_spectrum)) ** 2)


compute_shape_loss_and_grad = jit(value_and_grad(shape_loss))


def optimize_single_drum(
    initial_drum,
    target_spectrum,
    target_drum,
    model,
    idx,
    steps=1000,
    lr=1e-1,
):
    """Directly optimize triangle vertices using gradients from surrogate model."""
    optim = optax.adam(lr)
    opt_state = optim.init(initial_drum)

    def opt_step(carry, _):
        drum_curr, opt_curr = carry
        loss_val, grads = compute_shape_loss_and_grad(drum_curr, target_spectrum, model)
        updates, opt_curr = optim.update(grads, opt_curr)
        drum_curr = optax.apply_updates(drum_curr, updates)
        return (drum_curr, opt_curr), loss_val

    @jit
    def run_optimization(d_init, opt_init):
        (d_final, _), losses = jax.lax.scan(opt_step, (d_init, opt_init), None, length=steps)
        return d_final, losses

    d_final, losses = run_optimization(initial_drum, opt_state)
    initial_error = float(invariant_distance(target_drum, initial_drum, idx))
    final_error = float(invariant_distance(target_drum, d_final, idx))
    final_loss = float(losses[-1])

    return d_final, final_loss, initial_error, final_error


# ---------------------------------------------------------------------------
# Main Experiment Execution
# ---------------------------------------------------------------------------

def run_experiments():
    # Load datasets
    extended_path = "triangle_drums_extended/triangle_drums.npz"
    small_path = "triangle_drums.npz"

    data_ext = jnp.load(extended_path)
    triangles_ext = jnp.array(data_ext["perimeters"])
    spectra_ext = jnp.array(data_ext["spectra"])

    data_sm = jnp.load(small_path)
    triangles_sm = jnp.array(data_sm["perimeters"])
    spectra_sm = jnp.array(data_sm["spectra"])

    n_vertices = triangles_sm.shape[1]
    idx = preprocessing.augment(jnp.arange(n_vertices, dtype=float)[:, None])[..., 0].astype(int)

    # Deterministically select 10 test drums from the last 100 samples (indices 900..999)
    rng_test = np.random.default_rng(seed=42)
    test_indices = rng_test.choice(np.arange(900, 1000), size=10, replace=False)
    # Deterministically pick a random starting drum from the first 900 samples for each test drum
    start_indices = rng_test.choice(np.arange(0, 900), size=10, replace=False)

    print("=" * 70)
    print("SURROGATE DIRECT SHAPE OPTIMIZATION EXPERIMENTS")
    print("=" * 70)
    print(f"Test target drum indices (from last 100 samples): {test_indices.tolist()}")
    print(f"Initial starting drum indices (from training set): {start_indices.tolist()}")

    # -----------------------------------------------------------------------
    # Experiment 1: Surrogate trained on 5000 samples with matched epochs
    # (~166.7 epochs = 83,333 updates at batch size 10, matching 900 samples at 15k updates)
    # -----------------------------------------------------------------------
    n_updates_sm = 15000
    n_updates_ext = int(round(n_updates_sm * (triangles_ext.shape[0] / 900)))  # 83333 updates
    epochs = (n_updates_sm * 10) / 900

    print("\n" + "=" * 70)
    print(f"EXPERIMENT 1: Surrogate Trained on Extended Dataset (5000 samples, {n_updates_ext} updates, {epochs:.1f} epochs)")
    print("=" * 70)
    key_ext = random.PRNGKey(44)
    model_ext, _ = train_surrogate(
        triangles_ext, spectra_ext, key=key_ext, n_updates=n_updates_ext, batch_size=10, lr=1e-4
    )

    test_spec_errs_ext = get_surrogate_spectrum_errors(
        model_ext, triangles_sm[test_indices], spectra_sm[test_indices]
    )
    print(f"Surrogate mean relative spectrum error on 10 test drums: {float(jnp.mean(test_spec_errs_ext)):.4%}")

    results_ext = []
    print("\nRunning shape optimization for 10 test drums (Experiment 1)...")
    for k, (t_idx, s_idx) in enumerate(zip(test_indices, start_indices)):
        target_drum = triangles_sm[t_idx]
        target_spectrum = spectra_sm[t_idx]
        initial_drum = triangles_sm[s_idx]

        _, loss_val, init_err, fin_err = optimize_single_drum(
            initial_drum, target_spectrum, target_drum, model_ext, idx, steps=1000, lr=1e-1
        )
        results_ext.append({
            "test_idx": int(t_idx),
            "start_idx": int(s_idx),
            "loss": loss_val,
            "init_err": init_err,
            "final_err": fin_err,
            "spec_err": float(test_spec_errs_ext[k]),
        })
        print(f"  Drum #{k+1} (Target: {t_idx:3d}, Start: {s_idx:3d}): "
              f"Init Err = {init_err:.4f} -> Final Err = {fin_err:.4f} (Loss: {loss_val:.4e})")

    # -----------------------------------------------------------------------
    # Experiment 2: Surrogate trained on first 900 samples (small dataset)
    # (~166.7 epochs = 15,000 updates at batch size 10)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"EXPERIMENT 2: Surrogate Trained on Small Dataset (First 900 samples, {n_updates_sm} updates, {epochs:.1f} epochs)")
    print("=" * 70)
    key_sm = random.PRNGKey(44)
    model_sm, _ = train_surrogate(
        triangles_sm[:900], spectra_sm[:900], key=key_sm, n_updates=n_updates_sm, batch_size=10, lr=1e-4
    )

    test_spec_errs_sm = get_surrogate_spectrum_errors(
        model_sm, triangles_sm[test_indices], spectra_sm[test_indices]
    )
    print(f"Surrogate mean relative spectrum error on 10 test drums: {float(jnp.mean(test_spec_errs_sm)):.4%}")

    results_sm = []
    print("\nRunning shape optimization for 10 test drums (Experiment 2)...")
    for k, (t_idx, s_idx) in enumerate(zip(test_indices, start_indices)):
        target_drum = triangles_sm[t_idx]
        target_spectrum = spectra_sm[t_idx]
        initial_drum = triangles_sm[s_idx]

        _, loss_val, init_err, fin_err = optimize_single_drum(
            initial_drum, target_spectrum, target_drum, model_sm, idx, steps=1000, lr=1e-1
        )
        results_sm.append({
            "test_idx": int(t_idx),
            "start_idx": int(s_idx),
            "loss": loss_val,
            "init_err": init_err,
            "final_err": fin_err,
            "spec_err": float(test_spec_errs_sm[k]),
        })
        print(f"  Drum #{k+1} (Target: {t_idx:3d}, Start: {s_idx:3d}): "
              f"Init Err = {init_err:.4f} -> Final Err = {fin_err:.4f} (Loss: {loss_val:.4e})")

    # -----------------------------------------------------------------------
    # Experiment 3: Comparison and Analysis
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Comparison of Results")
    print("=" * 70)
    header = f"{'#':<3} | {'Target':<6} | {'Start':<5} | {'Init Err':<8} | {'Exp1 (5k)':<10} | {'Exp2 (900)':<10} | {'Rel Diff':<10}"
    print(header)
    print("-" * len(header))

    errs_ext = [r["final_err"] for r in results_ext]
    errs_sm = [r["final_err"] for r in results_sm]

    for k in range(len(results_ext)):
        t_idx = results_ext[k]["test_idx"]
        s_idx = results_ext[k]["start_idx"]
        i_err = results_ext[k]["init_err"]
        e1 = results_ext[k]["final_err"]
        e2 = results_sm[k]["final_err"]
        diff = (e1 - e2) / e2
        print(f"{k+1:<3} | {t_idx:<6} | {s_idx:<5} | {i_err:<8.4f} | {e1:<10.4f} | {e2:<10.4f} | {diff:<+10.1%}")

    print("-" * len(header))
    print(f"Mean Final Error:   Exp1 (5000 samples) = {np.mean(errs_ext):.4f}  |  Exp2 (900 samples) = {np.mean(errs_sm):.4f}")
    print(f"Median Final Error: Exp1 (5000 samples) = {np.median(errs_ext):.4f}  |  Exp2 (900 samples) = {np.median(errs_sm):.4f}")
    print(f"Min Final Error:    Exp1 (5000 samples) = {np.min(errs_ext):.4f}  |  Exp2 (900 samples) = {np.min(errs_sm):.4f}")
    print(f"Max Final Error:    Exp1 (5000 samples) = {np.max(errs_ext):.4f}  |  Exp2 (900 samples) = {np.max(errs_sm):.4f}")

    return results_ext, results_sm


if __name__ == "__main__":
    run_experiments()
