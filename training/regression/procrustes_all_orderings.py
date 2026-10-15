"""Example: train DrumTransformer on triangle drums with the Procrustes loss over all vertex orderings.

The prediction is matched to the target by the optimal orthogonal map (orthogonal
Procrustes, after centring). The loss is the smallest relative vertex error over all 2 N_v
orderings of the predicted vertices: all cyclic shifts of the given and of the reversed
order. Targets are the raw perimeters, only rescaled; the input spectra are not
normalised. The optimiser is AdamW with warm-up, cosine decay and gradient clipping.

    python training/regression/procrustes_all_orderings.py --data triangle_drums.npz --out results
"""
import argparse
import os

import equinox as eqx
import jax.numpy as jnp
import matplotlib
import numpy as np
import optax
from jax import random, vmap

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from drums.data import preprocessing, spectrum, triangle_drums
from drums.models import transformer


def align(v1, v2):
    """Centre v1 and v2 and rotate/reflect v2 onto v1 (orthogonal Procrustes)."""
    v1_ = v1 - jnp.mean(v1, axis=0, keepdims=True)
    v2_ = v2 - jnp.mean(v2, axis=0, keepdims=True)
    U, S, Vt = jnp.linalg.svd(v2_.T @ v1_, full_matrices=False)
    R = U @ Vt
    return v1_, v2_ @ R


def relative_error(target, prediction):
    target_, prediction_ = align(target, prediction)
    return jnp.mean(jnp.linalg.norm(target_ - prediction_, axis=1) / jnp.linalg.norm(target_, axis=1))


def orderings(N_v):
    """(2 N_v, N_v) index array: all cyclic shifts of the vertex order and of its reverse."""
    return preprocessing.augment(np.arange(N_v, dtype=float)[:, None])[..., 0].astype(int)


def orderings_errors(target, prediction, idx):
    return vmap(lambda ind: relative_error(target, prediction[ind]))(idx)


def regression_loss(model, feature, target, idx):
    return jnp.min(orderings_errors(target, model(feature), idx))


def batch_regression_loss(model, features, targets, idx):
    return jnp.mean(vmap(regression_loss, in_axes=(None, 0, 0, None))(model, features, targets, idx))


compute_loss_and_grads = eqx.filter_value_and_grad(batch_regression_loss)


@eqx.filter_jit
def step(carry, ind, optim, idx):
    model, opt_state, features, targets = carry
    loss, grads = compute_loss_and_grads(model, features[ind], targets[ind], idx)
    updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_array))
    model = eqx.apply_updates(model, updates)
    return [model, opt_state, features, targets], loss


@eqx.filter_jit
def all_errors(model, features, targets, idx):
    return vmap(regression_loss, in_axes=(None, 0, 0, None))(model, features, targets, idx)


def report(model, features, targets, idx, N_train, title):
    errors = all_errors(model, features, targets, idx)
    print(f"{title}: train error {jnp.mean(errors[:N_train]):.4f}, test error {jnp.mean(errors[N_train:]):.4f}")


def plot_shape(ax, model, features, targets, idx, sample):
    """Prediction (grey) rotated onto the target (pink), in the best vertex order."""
    gt, prediction = targets[sample], model(features[sample])
    prediction = prediction[idx[jnp.argmin(orderings_errors(gt, prediction, idx))]]
    gt, predicted = align(gt, prediction)
    ax.fill(predicted[:, 0], predicted[:, 1], facecolor="grey", edgecolor="black", linewidth=3)
    ax.fill(gt[:, 0], gt[:, 1], facecolor="pink", edgecolor="red", linewidth=3, alpha=0.5)
    ax.set_aspect("equal")
    ax.set_title(f"sample {sample}")


def plot_spectrum(ax, model, features, target_scale, sample):
    """Spectrum of the predicted drum (black) against the given one (red).

    The mesh resolution is the one used to generate the dataset (4000 elements per drum).
    """
    perimeter = np.array(model(features[sample]) * target_scale)
    a = triangle_drums.triangle_area(perimeter) / 4000
    predicted_spectrum = np.sort(spectrum.get_drum_spectrum(perimeter, a=a)[0])
    ax.plot(features[sample], "o", color="red")
    ax.plot(predicted_spectrum, ".", color="black")
    ax.set_title(f"spectrum, sample {sample}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="triangle_drums.npz")
    parser.add_argument("--out", default=".", help="folder for the figure")
    parser.add_argument("--n-updates", type=int, default=5000)
    args = parser.parse_args()

    N_train = 900
    learning_rate = 1e-3
    N_updates = args.n_updates
    N_batch = 64
    keys = random.split(random.PRNGKey(17), 2)

    N_v, N_f, N_eig, L, n_heads, mlp_ratio = 3, 128, 30, 5, 4, 4
    model = transformer.DrumTransformer(N_v, N_f, N_eig, keys[0], L, n_heads, mlp_ratio)
    idx = orderings(N_v)

    data = np.load(args.data)
    features = jnp.array(data["spectra"])
    perimeters = jnp.array(data["perimeters"])
    target_scale = jnp.max(jnp.linalg.norm(perimeters[:N_train], axis=2))
    targets = perimeters / target_scale

    report(model, features, targets, idx, N_train, "before training")

    schedule = optax.warmup_cosine_decay_schedule(0.0, learning_rate, 250, N_updates, 1e-5)
    optim = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(schedule, weight_decay=1e-4))
    opt_state = optim.init(eqx.filter(model, eqx.is_array))
    ind = random.choice(keys[1], N_train, shape=(N_updates, N_batch))
    carry = [model, opt_state, features, targets]
    history = []
    for i in ind:
        carry, loss_val = step(carry, i, optim, idx)
        history.append(loss_val)
    history = jnp.array(history)
    model = carry[0]

    report(model, features, targets, idx, N_train, "after training")

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    axes[0].plot(history)
    axes[0].set_yscale("log")
    axes[0].set_title("training loss")
    plot_shape(axes[1], model, features, targets, idx, 5)       # train sample
    plot_shape(axes[2], model, features, targets, idx, -8)      # test sample
    plot_spectrum(axes[3], model, features, target_scale, 7)
    os.makedirs(args.out, exist_ok=True)
    fig.savefig(os.path.join(args.out, "procrustes_all_orderings.png"), dpi=100, bbox_inches="tight")


if __name__ == "__main__":
    main()
