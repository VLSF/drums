"""Figures for isospectral_dataset_experiments.md (needs the results*/ folders with the trained models and per-drum outputs written by the runs): (1) distributions of e_set / sep, (2) example drums with the predictions of several methods."""
import numpy as np
import jax, jax.numpy as jnp
import equinox as eqx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from jax import random, vmap

from common import *
from fm import sample
from drums.models.transformer import DrumTransformer
from drums.models.flow_matching import DrumRingTransformerVelocity

# ------------------------------------------------------------------ figure 1
panels = [("regression, relative loss (Main)", "results/reg_ratio_relative_per_drum.npz"),
          ("regression, squared loss (Main)", "results/reg_ratio_squared_per_drum.npz"),
          ("regression, relative loss (Paired)", "results_paired/reg_ratio_relative_per_drum.npz"),
          ("flow matching, 16 draws (Main)", "results_fm/fm_main_ratio_white_per_draw.npz"),
          ("flow matching, 16 draws (Paired)", "results_fm/fm_paired_ratio_white_per_draw.npz"),
          ("regression, relative loss (Control)", "results_control/reg_ratio_relative_per_drum.npz")]
fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True)
for ax, (title, path) in zip(axes.ravel(), panels):
    m = np.load(path)
    sel = m["sep"] > 0.1
    r = np.minimum(m["e_truth"], m["e_twin"])[sel] / m["sep"][sel]
    ax.hist(np.clip(r, 0, 1.2), bins=48, range=(0, 1.2), color="#4c72b0", edgecolor="white")
    ax.axvline(0.25, color="k", ls=":", lw=1); ax.axvline(0.40, color="k", ls=":", lw=1)
    ax.set_title(f"{title}\nsharp {np.mean(r < 0.25):.2f}, blurry {np.mean(r > 0.4):.2f}", fontsize=10)
    ax.set_yticks([])
for ax in axes[1]: ax.set_xlabel("e_set / sep  (0: on one of the two drums, 0.5: midway)")
fig.tight_layout(); fig.savefig("fig_sharpness.png", dpi=110)

# ------------------------------------------------------------------ figure 2
L, V, W, _ = load_data("isospectral_30k.npz")
L_tr, _, L_te = split(L); V_tr, _, V_te = split(V); W_tr, _, W_te = split(W)
feats = Features("ratio", L_tr); feats_w = Features("ratio_white", L_tr)
key = random.PRNGKey(0)
reg_rel = eqx.tree_deserialise_leaves("results/reg_ratio_relative.eqx", DrumTransformer(N_V, 256, feats.dim, key, 5))
reg_sq = eqx.tree_deserialise_leaves("results/reg_ratio_squared.eqx", DrumTransformer(N_V, 256, feats.dim, key, 5))
fm = eqx.tree_deserialise_leaves("results_fm/fm_main_ratio_white.eqx",
                                 DrumRingTransformerVelocity(key, n_vertices=N_V, n_eig=feats_w.dim, N_f=256, L=3))
sep = np.asarray(vmap(relative_error)(V_te, W_te))
rows = np.argsort(-sep)[[5, 150, 400]]              # three drums with a large twin separation
titles = ["truth (pink) and twin (grey)", "regression, squared loss", "regression, relative loss", "flow matching, 4 draws"]
fig, axes = plt.subplots(len(rows), 4, figsize=(13, 3.3 * len(rows)))
def draw(ax, ref, shapes, colors, alphas):
    for s, c, a in zip(shapes, colors, alphas):
        ax.fill(s[:, 0], s[:, 1], facecolor=c, edgecolor="k", linewidth=1.5, alpha=a)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
for i, j in enumerate(rows):
    truth, twin = V_te[j], W_te[j]
    align = lambda s: best_align(truth, s)[0]
    p_sq, p_rel = align(reg_sq(feats(L_te[j:j+1])[0])), align(reg_rel(feats(L_te[j:j+1])[0]))
    draws = sample(fm, jnp.repeat(feats_w(L_te[j:j+1]), 4, axis=0), random.PRNGKey(7 + i), 200)
    draws = [align(d) for d in draws]
    draw(axes[i, 0], truth, [align(twin), truth], ["#aaaaaa", "#f4a6b8"], [0.7, 0.7])
    draw(axes[i, 1], truth, [align(twin), truth, p_sq], ["#aaaaaa", "#f4a6b8", "#4c72b0"], [0.3, 0.3, 0.7])
    draw(axes[i, 2], truth, [align(twin), truth, p_rel], ["#aaaaaa", "#f4a6b8", "#4c72b0"], [0.3, 0.3, 0.7])
    draw(axes[i, 3], truth, [align(twin), truth] + draws, ["#aaaaaa", "#f4a6b8"] + ["#4c72b0"] * 4, [0.2, 0.2] + [0.25] * 4)
    axes[i, 0].set_ylabel(f"test drum {j}\nsep {sep[j]:.2f}")
for ax, t in zip(axes[0], titles): ax.set_title(t, fontsize=10)
fig.tight_layout(); fig.savefig("fig_examples.png", dpi=110)
print("figures written")
