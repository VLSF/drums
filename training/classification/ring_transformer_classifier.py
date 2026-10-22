"""Example: classify pairs (drum shape, spectrum) as matching / non-matching with the RingTransformer.

The model sees the spectrum as the context vector and the vertices of the drum as the polygon.
It is invariant to rotations, reflections, translations, cyclic shifts and the reversion of the
vertices, so it needs no alignment of the drum. Its single invariant output is r = N2 - N1, and
p(same) = sigmoid(-r). The loss is the negative log likelihood of the labels.

Pairs: every drum is paired with its own spectrum (label 1) and with the spectrum of another
random drum of the same split (label 0), so the classes are balanced.

    python training/classification/ring_transformer_classifier.py --data triangle_drums.npz
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

from drums.models import ring_transformer


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


def logit(model, drum, spectrum):
    """r = N2 - N1, the invariant output of the model (the symmetric forward pass)."""
    return model.forward_flip(spectrum, drum)[0][0]


def logprob_loss(model, drum, spectrum, target):
    r = logit(model, drum, spectrum)
    # log p(same) = log sigmoid(-r), log p(distinct) = log sigmoid(r)
    return -(target * jax.nn.log_sigmoid(-r) + (1.0 - target) * jax.nn.log_sigmoid(r))


def batch_logprob_loss(model, drums, spectra, targets):
    return jnp.mean(vmap(logprob_loss, in_axes=(None, 0, 0, 0))(model, drums, spectra, targets))


compute_loss_and_grads = eqx.filter_value_and_grad(batch_logprob_loss)


@eqx.filter_jit
def step(carry, ind, optim):
    model, opt_state, drums, spectra, targets = carry
    loss, grads = compute_loss_and_grads(model, drums[ind], spectra[ind], targets[ind])
    updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_array))
    model = eqx.apply_updates(model, updates)
    return [model, opt_state, drums, spectra, targets], loss


@eqx.filter_jit
def logits(model, drums, spectra):
    return vmap(logit, in_axes=(None, 0, 0))(model, drums, spectra)


def all_pairs_logits(model, drums, spectra):
    """R[i, j] = r for the spectrum i and the drum j: the diagonal holds the matching pairs."""
    n = len(drums)
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    flat = [logits(model, drums[jj], spectra[ii]) for ii, jj in
            zip(np.array_split(i.ravel(), 10), np.array_split(j.ravel(), 10))]
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


def run(data_path, seed=176, n_updates=5000, N_train=900, N_batch=64, learning_rate=1e-3,
        N_f=128, L=4, n_heads=4, mlp_ratio=4):
    keys = random.split(random.PRNGKey(seed), 3)
    rng = np.random.default_rng(seed)

    data = np.load(data_path)
    spectra = jnp.array(data["spectra"])
    perimeters = jnp.array(data["perimeters"])
    drums = perimeters / jnp.max(jnp.linalg.norm(perimeters[:N_train], axis=2))
    N_v, N_eig = drums.shape[1], spectra.shape[1]

    model = ring_transformer.RingTransformer(N_v, N_eig, N_f, 1, keys[0], L, n_heads, mlp_ratio)

    train = balanced_pairs(rng, spectra[:N_train], drums[:N_train])
    test = balanced_pairs(rng, spectra[N_train:], drums[N_train:])

    schedule = optax.warmup_cosine_decay_schedule(0.0, learning_rate, 250, n_updates, 1e-5)
    optim = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(schedule, weight_decay=1e-4))
    opt_state = optim.init(eqx.filter(model, eqx.is_array))
    ind = random.choice(keys[2], len(train[2]), shape=(n_updates, N_batch))
    carry = [model, opt_state, train[1], train[0], train[2]]
    history = []
    for i in ind:
        carry, loss_val = step(carry, i, optim)
        history.append(loss_val)
    model = carry[0]

    result = dict(history=np.array(history), model=model)
    for name, (s, d, t) in [("train", train), ("test", test)]:
        result[f"{name}_accuracy"] = float(jnp.mean((logits(model, d, s) < 0) == (t == 1)))
    result["R_test"] = np.array(all_pairs_logits(model, drums[N_train:], spectra[N_train:]))
    result["all_pairs"] = pair_metrics(result["R_test"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="triangle_drums.npz")
    parser.add_argument("--out", default=".", help="folder for the figure")
    parser.add_argument("--seed", type=int, default=176)
    parser.add_argument("--n-updates", type=int, default=5000)
    args = parser.parse_args()

    result = run(args.data, args.seed, args.n_updates)
    print(f"balanced pairs: train accuracy {result['train_accuracy']:.4f}, test accuracy {result['test_accuracy']:.4f}")
    m = result["all_pairs"]
    print(f"all test pairs: accuracy on matching pairs {m['tpr']:.4f}, on non-matching pairs {m['tnr']:.4f}, "
          f"balanced {m['balanced_accuracy']:.4f}, ROC AUC {m['auc']:.4f}")

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(result["history"], color="black")
    ax.set_yscale("log")
    ax.set_title("training loss")
    os.makedirs(args.out, exist_ok=True)
    fig.savefig(os.path.join(args.out, "ring_transformer_classifier.png"), dpi=100, bbox_inches="tight")


if __name__ == "__main__":
    main()
