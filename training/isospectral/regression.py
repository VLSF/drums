"""Neural regression baseline: DrumTransformer (spectrum features -> 9 vertices), full training set.

Two losses, both computed after the best alignment of the prediction to the label (minimum over the
18 vertex labelings of the optimal rotation/reflection, orthogonal Procrustes):
  relative  mean relative vertex error |v_k - p_k| / |v_k|  (L1-like; the loss of the earlier experiments)
  squared   squared Frobenius distance / |v|^2             (L2-like; for a bimodal label its minimiser is
                                                           the midpoint, the posterior mean)
The label of a spectrum is the drum of the dataset; the twin is used for the metrics, and for training only with --paired.

    python regression.py --features ratio --loss relative --out results/reg_ratio_relative.json
"""
import argparse
import json

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
from jax import vmap

from common import *
from metrics import evaluate
from drums.models.transformer import DrumTransformer

LOSS = None


def loss_fn(model, features, vertices, key):
    if LOSS == "relative":
        per = lambda f, v: relative_error(v, model(f))
    else:
        per = lambda f, v: best_align(v, model(f))[1] / jnp.sum(v**2)
    return jnp.mean(vmap(per)(features, vertices))


def main():
    global LOSS
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="isospectral_30k.npz")
    p.add_argument("--features", default="ratio")
    p.add_argument("--loss", default="relative", choices=["relative", "squared"])
    p.add_argument("--n-updates", type=int, default=30000)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--N-f", type=int, default=256)
    p.add_argument("--L", type=int, default=5)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--paired", action="store_true",
                   help="train on both drums of every pair: each training spectrum appears twice, with its drum and with its twin")
    p.add_argument("--out", default="regression.json")
    a = p.parse_args()
    LOSS = a.loss

    L, V, W, _ = load_data(a.data)
    L_tr, L_va, L_te = split(L)
    V_tr, V_va, V_te = split(V)
    W_tr, W_va, W_te = split(W)
    feats = Features(a.features, L_tr)
    F_tr, F_va, F_te = feats(L_tr), feats(L_va), feats(L_te)
    if a.paired:
        F_tr, V_tr = jnp.concatenate([F_tr, F_tr]), jnp.concatenate([V_tr, W_tr])

    key = jax.random.PRNGKey(a.seed)
    model = DrumTransformer(N_V, a.N_f, feats.dim, key, a.L)
    predict = eqx.filter_jit(lambda m, F: vmap(m)(F))
    err = eqx.filter_jit(lambda m, F, V: jnp.mean(vmap(lambda f, v: relative_error(v, m(f)))(F, V)))
    eval_fn = lambda m: f"val rel. error {float(err(m, F_va, V_va)):.4f}"
    model, hist = train(model, loss_fn, (F_tr, V_tr), a.n_updates, a.batch, a.lr, key, eval_fn, tag=f"{a.features}/{a.loss}")
    eqx.tree_serialise_leaves(a.out.replace(".json", ".eqx"), model)

    res = dict(args=vars(a), train_error=float(err(model, F_tr[:1500], V_tr[:1500])), val_error=float(err(model, F_va, V_va)))
    pred_te = predict(model, F_te)
    res["test"], per_drum = evaluate(pred_te, V_te, W_te)
    print(res)
    json.dump(res, open(a.out, "w"), indent=1)
    np.savez(a.out.replace(".json", "_per_drum.npz"), prediction=np.asarray(pred_te), **per_drum)


if __name__ == "__main__":
    main()
