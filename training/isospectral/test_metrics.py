# run in the folder with isospectral_drums.npz: python src/drums/data/datasets_collection.py isospectral_drums <folder>
import numpy as np, jax, jax.numpy as jnp
from jax import vmap
from common import *
from metrics import evaluate, references
d = np.load("isospectral_drums.npz")
c = lambda x: (lambda x: x - x.mean(1, keepdims=True))(jnp.asarray(x, dtype=jnp.float32))
V, W = c(d["perimeters"])[:300], c(d["twin_perimeters"])[:300]
print("references:", references(V, W))
mid = vmap(lambda a, b: procrustes_mean(jnp.stack([a, b]), n_iter=3))(V, W)
rng = np.random.default_rng(0); pick = rng.uniform(size=300) < 0.5
rand_pick = jnp.where(jnp.asarray(pick)[:, None, None], V, W)
for name, p in [("truth", V), ("twin", W), ("random pick of the two", rand_pick), ("midpoint", mid)]:
    s, _ = evaluate(p, V, W)
    print(f"{name:24s} error {s['error']:.3f} error_set {s['error_set']:.3f} sharp {s['sharp']:.2f} blurry {s['blurry']:.2f} e_set/sep {s['mean_set_over_sep']:.2f} mid/sep {s['mid_over_sep']:.2f} pick {s['pick']:.2f} (n_sep {s['n_separated']}/300)")
