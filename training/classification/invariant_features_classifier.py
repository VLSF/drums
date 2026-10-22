"""Example: classify pairs (drum shape, spectrum) as matching / non-matching with invariant features.

The drum enters only through its distances to N_drums learnable reference drums. The distance
is the Procrustes distance (after centring and the optimal rotation or reflection), minimised
over all 2 N_v orderings of the vertices, so it is invariant to all the symmetries of the
drum. These distances are concatenated with the spectrum and processed by an MLP with
residual blocks. Its single output is r = N2 - N1, and p(same) = sigmoid(-r). The loss is the
negative log likelihood of the labels.

Pairs: every drum is paired with its own spectrum (label 1) and with the spectrum of another
random drum of the same split (label 0), so the classes are balanced.

    python training/classification/invariant_features_classifier.py --data triangle_drums.npz
"""
import argparse
import os

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
import optax
from jax import random, vmap

matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
    """Smallest Procrustes distance over the vertex orderings idx of shape (2 N_v, N_v)."""
    return jnp.min(vmap(lambda ind: distance(target, prediction[ind]))(idx))


def invariant_features(drum, reference_drums, idx):
    return vmap(lambda ref: invariant_distance(drum, ref, idx))(reference_drums)


class Block(eqx.Module):
    norm: eqx.nn.LayerNorm
    mlp_in: eqx.nn.Linear
    mlp_out: eqx.nn.Linear

    def __init__(self, N_f, mlp_ratio, key):
        k_in, k_out = random.split(key, 2)
        self.norm = eqx.nn.LayerNorm(N_f)
        self.mlp_in = eqx.nn.Linear(N_f, mlp_ratio*N_f, key=k_in)
        self.mlp_out = eqx.nn.Linear(mlp_ratio*N_f, N_f, key=k_out)

    def __call__(self, x):
        return x + self.mlp_out(jax.nn.gelu(self.mlp_in(self.norm(x))))


class DrumMLP(eqx.Module):
    encoder: eqx.nn.Linear
    drums: jax.Array          # learnable reference drums, (N_drums, N_v, 2)
    blocks: list
    final_norm: eqx.nn.LayerNorm
    decoder: eqx.nn.Linear

    def __init__(self, N_v, N_f, N_eig, N_drums, key, L, mlp_ratio=4):
        k_enc, k_drums, k_blocks, k_dec = random.split(key, 4)
        self.encoder = eqx.nn.Linear(N_eig + N_drums, N_f, key=k_enc)
        self.drums = 0.1*random.normal(k_drums, (N_drums, N_v, 2))
        self.blocks = [Block(N_f, mlp_ratio, k) for k in random.split(k_blocks, L)]
        self.final_norm = eqx.nn.LayerNorm(N_f)
        self.decoder = eqx.nn.Linear(N_f, 1, key=k_dec)

    def __call__(self, drum, spectrum, idx):
        x = jnp.concatenate([spectrum, invariant_features(drum, self.drums, idx)])
        x = self.encoder(x)
        for block in self.blocks:
            x = block(x)
        return self.decoder(self.final_norm(x))[0]


def derangement(rng, n):
    """Random permutation without fixed points (a drum must not be paired with itself)."""
    while True:
        perm = rng.permutation(n)
        if not np.any(perm == np.arange(n)):
            return perm


def balanced_pairs(rng, spectra, drums):
    """Matching pairs, followed by the same spectra paired with other drums."""
    wrong = derangement(rng, len(spectra))
    targets = np.concatenate([np.ones(len(spectra)), np.zeros(len(spectra))])
    return (jnp.concatenate([spectra, spectra]), jnp.concatenate([drums, drums[wrong]]),
            jnp.asarray(targets))


def logprob_loss(model, drum, spectrum, target, idx):
    r = model(drum, spectrum, idx)
    # log p(same) = log sigmoid(-r), log p(distinct) = log sigmoid(r)
    return -(target * jax.nn.log_sigmoid(-r) + (1.0 - target) * jax.nn.log_sigmoid(r))


def batch_logprob_loss(model, drums, spectra, targets, idx):
    return jnp.mean(vmap(logprob_loss, in_axes=(None, 0, 0, 0, None))(model, drums, spectra, targets, idx))


compute_loss_and_grads = eqx.filter_value_and_grad(batch_logprob_loss)


@eqx.filter_jit
def step(carry, ind, optim, idx):
    model, opt_state, drums, spectra, targets = carry
    loss, grads = compute_loss_and_grads(model, drums[ind], spectra[ind], targets[ind], idx)
    updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_array))
    model = eqx.apply_updates(model, updates)
    return [model, opt_state, drums, spectra, targets], loss


@eqx.filter_jit
def logits(model, drums, spectra, idx):
    return vmap(model, in_axes=(0, 0, None))(drums, spectra, idx)


def all_pairs_logits(model, drums, spectra, idx):
    """R[i, j] = r for the spectrum i and the drum j: the diagonal holds the matching pairs."""
    n = len(drums)
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    flat = [logits(model, drums[jj], spectra[ii], idx) for ii, jj in
            zip(np.array_split(i.ravel(), 20), np.array_split(j.ravel(), 20))]
    return jnp.concatenate(flat).reshape(n, n)


def pair_metrics(R):
    """Accuracy on the matching pairs, on all non-matching pairs, balanced accuracy and ROC AUC."""
    n = R.shape[0]
    score = -R                                       # larger = more likely the same
    pos = np.diag(score)
    neg = score[~np.eye(n, dtype=bool)]
    auc = np.mean(pos[:, None] > neg[None, :]) + 0.5*np.mean(pos[:, None] == neg[None, :])
    tpr, tnr = np.mean(pos > 0), np.mean(neg < 0)
    return dict(tpr=tpr, tnr=tnr, balanced_accuracy=0.5*(tpr + tnr), auc=auc)


def run(data_path, seed=176, n_updates=3000, N_train=900, N_batch=64, learning_rate=1e-3,
        N_f=128, N_drums=100, L=6, mlp_ratio=4):
    keys = random.split(random.PRNGKey(seed), 3)
    rng = np.random.default_rng(seed)

    data = np.load(data_path)
    spectra = jnp.array(data["spectra"])
    perimeters = jnp.array(data["perimeters"])
    drums = perimeters / jnp.max(jnp.linalg.norm(perimeters[:N_train], axis=2))
    N_v, N_eig = drums.shape[1], spectra.shape[1]

    model = DrumMLP(N_v, N_f, N_eig, N_drums, keys[0], L, mlp_ratio)
    idx = preprocessing.augment(np.arange(N_v, dtype=float)[:, None])[..., 0].astype(int)

    train = balanced_pairs(rng, spectra[:N_train], drums[:N_train])
    test = balanced_pairs(rng, spectra[N_train:], drums[N_train:])

    schedule = optax.warmup_cosine_decay_schedule(0.0, learning_rate, 250, n_updates, 1e-5)
    optim = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(schedule, weight_decay=1e-4))
    opt_state = optim.init(eqx.filter(model, eqx.is_array))
    ind = random.choice(keys[2], len(train[2]), shape=(n_updates, N_batch))
    carry = [model, opt_state, train[1], train[0], train[2]]
    history = []
    for i in ind:
        carry, loss_val = step(carry, i, optim, idx)
        history.append(loss_val)
    model = carry[0]

    result = dict(history=np.array(history), model=model)
    for name, (s, d, t) in [("train", train), ("test", test)]:
        result[f"{name}_accuracy"] = float(jnp.mean((logits(model, d, s, idx) < 0) == (t == 1)))
    result["R_test"] = np.array(all_pairs_logits(model, drums[N_train:], spectra[N_train:], idx))
    result["all_pairs"] = pair_metrics(result["R_test"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="triangle_drums.npz")
    parser.add_argument("--out", default=".", help="folder for the figure")
    parser.add_argument("--seed", type=int, default=176)
    parser.add_argument("--n-updates", type=int, default=3000)
    args = parser.parse_args()

    result = run(args.data, args.seed, args.n_updates)
    print(f"balanced pairs: train accuracy {result['train_accuracy']:.4f}, test accuracy {result['test_accuracy']:.4f}")
    m = result["all_pairs"]
    print(f"all test pairs: accuracy on matching pairs {m['tpr']:.4f}, on non-matching pairs {m['tnr']:.4f}, "
          f"balanced {m['balanced_accuracy']:.4f}, ROC AUC {m['auc']:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(result["history"], color="black")
    axes[0].set_yscale("log")
    axes[0].set_title("training loss")
    cmap = plt.get_cmap("tab20")
    for i, ref in enumerate(result["model"].drums[:10]):
        axes[1].fill(ref[:, 0], ref[:, 1], facecolor="none", edgecolor=cmap(i), linewidth=2)
    axes[1].set_aspect("equal")
    axes[1].set_title("10 learned reference drums")
    os.makedirs(args.out, exist_ok=True)
    fig.savefig(os.path.join(args.out, "invariant_features_classifier.png"), dpi=100, bbox_inches="tight")


if __name__ == "__main__":
    main()
