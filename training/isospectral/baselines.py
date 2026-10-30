"""Non-neural baselines on the isospectral-drums dataset (full training set, 27k drums).

1. oracle NN  : the best training SHAPE for every test shape (knows the answer, only limited by
                how densely the training set covers shape space).
2. spectral kNN: nearest training spectra in a feature space; the shapes of the k neighbours are
                combined by the Procrustes mean (align to a common frame, then average).
Every prediction is scored against the truth and its isospectral twin (metrics.py).

    python baselines.py --data isospectral_30k.npz --out baselines.json
"""
import argparse
import json

import jax
import jax.numpy as jnp
import numpy as np
from jax import vmap

from common import *
from metrics import evaluate, references


def oracle_nn(train_v, test_v):
    return train_v[jnp.argmin(pairwise_distance(test_v, train_v), axis=1)]


def knn(feat_train, feat_test, train_v, k, chunk=250):
    out = []
    for i in range(0, len(feat_test), chunk):
        d = jnp.sum((feat_test[i:i + chunk, None] - feat_train[None])**2, axis=-1)
        idx = jnp.argsort(d, axis=1)[:, :k]
        nb = train_v[idx]                                   # (chunk, k, 9, 2)
        out.append(nb[:, 0] if k == 1 else vmap(procrustes_mean)(nb))
    return jnp.concatenate(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="isospectral_30k.npz")
    p.add_argument("--paired", action="store_true",
                   help="the training set has both drums of every pair (each spectrum twice, with its drum and with its twin)")
    p.add_argument("--out", default="baselines.json")
    a = p.parse_args()
    L, V, W, scale = load_data(a.data)
    L_tr, L_va, L_te = split(L)
    V_tr, V_va, V_te = split(V)
    W_tr, W_va, W_te = split(W)
    if a.paired:
        L_tr, V_tr = jnp.concatenate([L_tr, L_tr]), jnp.concatenate([V_tr, W_tr])
    res = dict(references=references(V_te, W_te))
    print("references:", res["references"])

    def report(name, pred):
        s, _ = evaluate(pred, V_te, W_te)
        res[name] = s
        print(f"  {name}: error {s['error']:.4f}, sharp {s['sharp']:.2f}, blurry {s['blurry']:.2f}, "
              f"e_set/sep {s['mean_set_over_sep']:.2f}, mid/sep {s['mid_over_sep']:.2f}, pick {s['pick']:.2f}", flush=True)

    print("oracle nearest training shape (to the truth):")
    report("oracle_nn", oracle_nn(V_tr, V_te))
    for kind in ["naive", "ratio", "ratio_white", "ratio_k10", "ratio_k20"]:
        feats = Features(kind, L_tr)
        F_tr, F_te = feats(L_tr), feats(L_te)
        print(f"features: {kind} (dim {feats.dim})")
        for k in (1, 5, 20):
            report(f"knn_{kind}_k{k}", knn(F_tr, F_te, V_tr, k))
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
