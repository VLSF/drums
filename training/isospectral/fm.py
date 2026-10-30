"""Flow matching (RingTransformer velocity, noise-aligned pairing) on the isospectral-drums dataset.

The target of each noise draw x_0 is the data shape relabeled/rotated/reflected so that it is as close as
possible to x_0 (18 labelings). After training M samples are drawn for every test spectrum and scored
against the truth and its isospectral twin (metrics.py), pooled over all draws:
  sharp / blurry / pick / mid     as for the regression (every draw is a prediction)
  mode balance                    per drum: fraction of the draws nearer to the truth than to the twin
                                  (0.5 expected, its spread over drums should be that of a binomial(M, 1/2));
                                  both_modes: fraction of drums for which the draws contain a sharp draw near
                                  the truth AND a sharp draw near the twin
  mean_of_draws                   the same metrics for the Procrustes mean of the M draws (blurry if bimodal)
  energy score                    E d(X, y) - 1/2 E d(X, X'), X, X' independent draws, y the label
With --paired the training set contains both drums of every pair (every spectrum twice).

    python fm.py --features ratio_white --out results_fm/fm_main_ratio_white.json
"""
import argparse
import json

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
from jax import random, vmap

from common import *
from metrics import per_drum, summarize
from drums.models.flow_matching import DrumRingTransformerVelocity


def fm_loss(model, spectra, vertices, key):
    k_noise, k_time = random.split(key)
    x0 = random.normal(k_noise, vertices.shape)
    x0 = x0 - jnp.mean(x0, axis=1, keepdims=True)
    x1 = vmap(lambda a, b: best_align(a, b)[0])(x0, vertices)       # aligned pairing
    t = random.uniform(k_time, (vertices.shape[0],))
    xt = (1 - t[:, None, None]) * x0 + t[:, None, None] * x1
    pred = vmap(model)(t, xt, spectra)
    return jnp.mean((pred - (x1 - x0))**2)


@eqx.filter_jit
def sample(model, spectra, key, n_steps):
    """Euler sampler, one draw per row of spectra."""
    def one(spectrum, key):
        x = random.normal(key, (N_V, 2))
        x = x - jnp.mean(x, axis=0, keepdims=True)
        dt = 1.0 / n_steps
        return jax.lax.fori_loop(0, n_steps, lambda i, x: x + dt * model(i * dt, x, spectrum), x)
    return vmap(one)(spectra, random.split(key, len(spectra)))


def evaluate(model, F, V, W, key, n_samples, n_steps, sep_min=0.1):
    n, M = len(F), n_samples
    S = jnp.stack([sample(model, F, k, n_steps) for k in random.split(key, M)], axis=1)     # (n, M, 9, 2)
    rep = lambda x: jnp.repeat(x, M, axis=0)
    m = per_drum(S.reshape(n * M, N_V, 2), rep(V), rep(W))
    res = dict(draws=summarize(m, sep_min))
    sel = m["sep"].reshape(n, M)[:, 0] > sep_min
    e_t, e_w, sep = (m[k].reshape(n, M) for k in ("e_truth", "e_twin", "sep"))
    sharp = np.minimum(e_t, e_w) < 0.25 * sep
    frac_truth = (e_t < e_w).mean(axis=1)
    both = ((sharp & (e_t < e_w)).any(axis=1)) & ((sharp & (e_w < e_t)).any(axis=1))
    res["mode_balance"] = dict(mean_fraction_nearer_truth=float(frac_truth[sel].mean()),
                               std_over_drums=float(frac_truth[sel].std()),
                               std_if_binomial=float(np.sqrt(0.25 / M)), both_modes=float(both[sel].mean()))
    mean_pred = vmap(procrustes_mean)(S)
    res["mean_of_draws"] = summarize(per_drum(mean_pred, V, W), sep_min)
    d_xy = vmap(lambda ss: vmap(lambda a: vmap(lambda b: relative_error(a, b))(ss))(ss))(S)  # (n, M, M)
    e_s = np.asarray(d_xy.sum(axis=(1, 2)) / (M * (M - 1)))
    res["energy_score"] = float((e_t.mean(axis=1) - 0.5 * e_s).mean())
    res["mean_error_draws"] = float(e_t.mean())
    return res, m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="isospectral_30k.npz")
    p.add_argument("--features", default="ratio_white")
    p.add_argument("--paired", action="store_true")
    p.add_argument("--n-updates", type=int, default=40000)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--N-f", type=int, default=256)
    p.add_argument("--L", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--n-samples", type=int, default=16)
    p.add_argument("--n-eval", type=int, default=1500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="fm.json")
    a = p.parse_args()

    L, V, W, _ = load_data(a.data)
    L_tr, L_va, L_te = split(L)
    V_tr, V_va, V_te = split(V)
    W_tr, W_va, W_te = split(W)
    feats = Features(a.features, L_tr)
    F_tr, F_va, F_te = feats(L_tr), feats(L_va), feats(L_te)
    if a.paired:
        F_tr, V_tr = jnp.concatenate([F_tr, F_tr]), jnp.concatenate([V_tr, W_tr])

    key = random.PRNGKey(a.seed)
    model = DrumRingTransformerVelocity(key, n_vertices=N_V, n_eig=feats.dim, N_f=a.N_f, L=a.L)
    val_loss = eqx.filter_jit(fm_loss)
    ke = random.PRNGKey(123)
    eval_fn = lambda m: f"val FM loss {float(val_loss(m, F_va, V_va, ke)):.4f}"
    model, hist = train(model, fm_loss, (F_tr, V_tr), a.n_updates, a.batch, a.lr, key, eval_fn, eval_every=5000, tag=a.features)
    eqx.tree_serialise_leaves(a.out.replace(".json", ".eqx"), model)

    n = a.n_eval
    res = dict(args=vars(a), n_train_pairs=int(len(F_tr)), final_train_loss=float(np.mean(hist[-500:])))
    for n_steps in (50, 200):
        res[f"test_steps{n_steps}"], per = evaluate(model, F_te[:n], V_te[:n], W_te[:n], random.PRNGKey(443), a.n_samples, n_steps)
        print(n_steps, "Euler steps:", json.dumps(res[f"test_steps{n_steps}"]), flush=True)
    json.dump(res, open(a.out, "w"), indent=1)
    np.savez(a.out.replace(".json", "_per_draw.npz"), **per)


if __name__ == "__main__":
    main()
