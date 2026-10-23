"""Example: classify pairs (drum shape, spectrum) as matching / non-matching via predicted drums.

A single model is trained end-to-end on the pair labels:
  1. a DrumTransformer predicts K drums from the spectrum,
  2. the invariant distances between the given drum and the K predicted drums are computed
     (relative Procrustes distance, minimised over all vertex orderings),
  3. a small network (linear, or with one hidden layer) on the logarithms of the distances gives
     r = N2 - N1, with p(same) = sigmoid(-r).
The loss is the negative log likelihood of the labels. The predicted drums are not trained to
match the true drum directly: they only receive the gradient of the classification loss.

Pairs: every drum is paired with its own spectrum (label 1) and with the spectrum of another
random drum of the same split (label 0), so the classes are balanced.

    python training/classification/predicted_drums_classifier.py --data triangle_drums.npz
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


def invariant_error(target, prediction, idx):
    """Smallest relative Procrustes error over the vertex orderings idx of shape (2 N_v, N_v)."""
    return jnp.min(vmap(lambda ind: relative_error(target, prediction[ind]))(idx))


def predicted_drums(model, spectrum):
    """The K drums predicted from a spectrum, shape (K, N_v, 2); the model gives (N_v, 2 K)."""
    out = model(spectrum)
    return out.reshape(out.shape[0], -1, 2).transpose(1, 0, 2)


class Scorer(eqx.Module):
    """Small network from the K log-distances to r: linear for hidden=0, otherwise one hidden layer."""
    layers: list

    def __init__(self, K, hidden, key):
        k1, k2 = random.split(key)
        self.layers = [eqx.nn.Linear(K, 1, key=k1)] if hidden == 0 else \
            [eqx.nn.Linear(K, hidden, key=k1), eqx.nn.Linear(hidden, 1, key=k2)]

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = jax.nn.gelu(layer(x))
        return self.layers[-1](x)[0]


class PredictedDrumsClassifier(eqx.Module):
    """DrumTransformer -> K predicted drums -> log invariant distances to the drum -> scorer -> r."""
    regressor: transformer.DrumTransformer
    scorer: Scorer

    def __init__(self, N_v, N_f, N_eig, n_drums, hidden, key, L=5, n_heads=4, mlp_ratio=4):
        k_reg, k_score = random.split(key)
        self.regressor = transformer.DrumTransformer(N_v, N_f, N_eig, k_reg, L, n_heads, mlp_ratio, n_drums=n_drums)
        self.scorer = Scorer(n_drums, hidden, k_score)

    def __call__(self, drum, spectrum, idx):
        predicted = predicted_drums(self.regressor, spectrum)
        return self.scorer(jnp.log(vmap(lambda p: invariant_error(drum, p, idx))(predicted) + 1e-8))


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


def all_pairs_logits(model, drums, spectra, idx):
    """R[i, j] = r for the spectrum i and the drum j: the diagonal holds the matching pairs."""
    n = len(drums)
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    flat = [logits(model, drums[jj], spectra[ii], idx) for ii, jj in
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


def run(data_path, seed=176, n_drums=4, hidden=16, n_updates=5000, N_train=900, N_batch=64,
        learning_rate=1e-3, N_f=128, L=5):
    keys = random.split(random.PRNGKey(seed), 3)
    rng = np.random.default_rng(seed)

    data = np.load(data_path)
    spectra = jnp.array(data["spectra"])
    perimeters = jnp.array(data["perimeters"])
    drums = perimeters / jnp.max(jnp.linalg.norm(perimeters[:N_train], axis=2))
    N_v, N_eig = drums.shape[1], spectra.shape[1]
    idx = preprocessing.augment(np.arange(N_v, dtype=float)[:, None])[..., 0].astype(int)

    model = PredictedDrumsClassifier(N_v, N_f, N_eig, n_drums, hidden, keys[0], L)

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
    # how close the K drums predicted from the test spectra are to the true drums (a diagnostic only)
    predicted = vmap(predicted_drums, in_axes=(None, 0))(model.regressor, spectra[N_train:])
    result["regression_error"] = np.array(vmap(lambda p, d: vmap(lambda q: invariant_error(d, q, idx))(p))(
        predicted, drums[N_train:])).mean(axis=0)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="triangle_drums.npz")
    parser.add_argument("--out", default=".", help="folder for the figure")
    parser.add_argument("--seed", type=int, default=176)
    parser.add_argument("--n-drums", type=int, default=4, help="K, the number of predicted drums")
    parser.add_argument("--hidden", type=int, default=16, help="size of the hidden layer, 0 for a linear scorer")
    parser.add_argument("--n-updates", type=int, default=5000)
    args = parser.parse_args()

    result = run(args.data, args.seed, args.n_drums, args.hidden, args.n_updates)
    print(f"relative error of the K predicted drums to the true drum on the test drums: {np.round(result['regression_error'], 4)}")
    print(f"balanced pairs: train accuracy {result['train_accuracy']:.4f}, test accuracy {result['test_accuracy']:.4f}")
    m = result["all_pairs"]
    print(f"all test pairs: accuracy on matching pairs {m['tpr']:.4f}, on non-matching pairs {m['tnr']:.4f}, "
          f"balanced {m['balanced_accuracy']:.4f}, ROC AUC {m['auc']:.4f}")

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(result["history"], color="black")
    ax.set_yscale("log")
    ax.set_title("training loss")
    os.makedirs(args.out, exist_ok=True)
    fig.savefig(os.path.join(args.out, "predicted_drums_classifier.png"), dpi=100, bbox_inches="tight")


if __name__ == "__main__":
    main()
