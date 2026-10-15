"""Example: train DrumTransformer on triangle drums with the Procrustes loss.

The prediction is matched to the target by the optimal orthogonal map (orthogonal
Procrustes, after centring), so the loss does not depend on the position and orientation
of the drum. Both the given and the reversed vertex order of the prediction are tried and
the smaller error is taken. Targets are the raw perimeters, only rescaled.

    python training/regression/procrustes_loss.py --data triangle_drums.npz --out results
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

from drums.data import spectrum
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


def regression_loss(model, feature, target):
    prediction = model(feature)
    return jnp.minimum(relative_error(target, prediction), relative_error(target, prediction[::-1]))


def batch_regression_loss(model, features, targets):
    return jnp.mean(vmap(regression_loss, in_axes=(None, 0, 0))(model, features, targets))


compute_loss_and_grads = eqx.filter_value_and_grad(batch_regression_loss)


@eqx.filter_jit
def step(carry, ind, optim):
    model, opt_state, features, targets = carry
    loss, grads = compute_loss_and_grads(model, features[ind], targets[ind])
    updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_array))
    model = eqx.apply_updates(model, updates)
    return [model, opt_state, features, targets], loss


@eqx.filter_jit
def all_errors(model, features, targets):
    return vmap(regression_loss, in_axes=(None, 0, 0))(model, features, targets)


def report(model, features, targets, N_train, title):
    errors = all_errors(model, features, targets)
    print(f"{title}: train error {jnp.mean(errors[:N_train]):.4f}, test error {jnp.mean(errors[N_train:]):.4f}")


def plot_shape(ax, model, features, targets, sample):
    """Prediction (grey) rotated onto the target (pink), in the better vertex order."""
    gt, prediction = targets[sample], model(features[sample])
    if relative_error(gt, prediction[::-1]) < relative_error(gt, prediction):
        prediction = prediction[::-1]
    gt, predicted = align(gt, prediction)
    ax.fill(predicted[:, 0], predicted[:, 1], facecolor="grey", edgecolor="black", linewidth=3)
    ax.fill(gt[:, 0], gt[:, 1], facecolor="pink", edgecolor="red", linewidth=3, alpha=0.5)
    ax.set_aspect("equal")
    ax.set_title(f"sample {sample}")


def plot_spectrum(ax, model, features, target_scale, sample):
    """Spectrum of the predicted drum (black) against the given one (red)."""
    perimeter = np.array(model(features[sample]) * target_scale)
    predicted_spectrum = np.sort(spectrum.get_drum_spectrum(perimeter)[0])
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

    data = np.load(args.data)
    features = jnp.array(data["spectra"])
    perimeters = jnp.array(data["perimeters"])
    target_scale = jnp.max(jnp.linalg.norm(perimeters[:N_train], axis=2))
    targets = perimeters / target_scale

    report(model, features, targets, N_train, "before training")

    optim = optax.adam(learning_rate)
    opt_state = optim.init(eqx.filter(model, eqx.is_array))
    ind = random.choice(keys[1], N_train, shape=(N_updates, N_batch))
    carry = [model, opt_state, features, targets]
    history = []
    for i in ind:
        carry, loss_val = step(carry, i, optim)
        history.append(loss_val)
    history = jnp.array(history)
    model = carry[0]

    report(model, features, targets, N_train, "after training")

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    axes[0].plot(history)
    axes[0].set_yscale("log")
    axes[0].set_title("training loss")
    plot_shape(axes[1], model, features, targets, 5)       # train sample
    plot_shape(axes[2], model, features, targets, -8)      # test sample
    plot_spectrum(axes[3], model, features, target_scale, 7)
    os.makedirs(args.out, exist_ok=True)
    fig.savefig(os.path.join(args.out, "procrustes_loss.png"), dpi=100, bbox_inches="tight")


if __name__ == "__main__":
    main()
