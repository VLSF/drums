"""Example: train DrumTransformer on triangle drums with the "align" loss.

Targets are aligned (centred, longest vertex up) and augmented with all cyclic
permutations and reversals of the vertex order. The prediction is aligned in the same
way, and the loss is the smallest relative vertex error over the augmented targets.

    python training/regression/align_loss.py --data triangle_drums.npz --out results
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

from drums.data import preprocessing, spectrum
from drums.models import transformer


def align(vertices):
    """Differentiable version of preprocessing.align (no tolerance for ties)."""
    vertices_ = vertices - jnp.mean(vertices, axis=0, keepdims=True)
    r = jnp.linalg.norm(vertices_, axis=1)
    ind = jnp.argmax(r)
    phi = jnp.arctan2(vertices_[ind, 0], vertices_[ind, 1])
    return jnp.stack([
        vertices_[:, 0]*jnp.cos(phi) - vertices_[:, 1]*jnp.sin(phi),
        vertices_[:, 0]*jnp.sin(phi) + vertices_[:, 1]*jnp.cos(phi)
    ], axis=1)


def regression_loss(model, feature, target):
    """target: (N_augmented, N_v, 2) equivalent descriptions of the same drum."""
    prediction = jnp.expand_dims(align(model(feature)), axis=0)
    relative_error = jnp.linalg.norm(target - prediction, axis=2) / jnp.linalg.norm(target, axis=2)
    return jnp.min(jnp.mean(relative_error, axis=1))


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
    predicted = align(model(features[sample]))
    gt = targets[sample, 0]
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
    targets = jnp.array([preprocessing.augment(preprocessing.align(p)) for p in data["perimeters"]])
    target_scale = jnp.max(jnp.linalg.norm(targets[:N_train], axis=3))
    targets = targets / target_scale

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
    plot_shape(axes[2], model, features, targets, -9)      # test sample
    plot_spectrum(axes[3], model, features, target_scale, 7)
    os.makedirs(args.out, exist_ok=True)
    fig.savefig(os.path.join(args.out, "align_loss.png"), dpi=100, bbox_inches="tight")


if __name__ == "__main__":
    main()
