# run in the folder with isospectral_drums.npz: python src/drums/data/datasets_collection.py isospectral_drums <folder>
import numpy as np, jax, jax.numpy as jnp
from common import *
d = np.load("isospectral_drums.npz"); P = jnp.asarray(d["perimeters"]); P = P - P.mean(1, keepdims=True)
a = P[0]
th = 0.7; R = jnp.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
b = (a @ R.T * jnp.array([1., -1.]))[PERMS[5]] + 3.
al, r = best_align(a, b); print("residual after hidden transform (0):", float(r), float(jnp.abs(al - a).max()))
print("relative_error ~0:", float(relative_error(a, b)))
D = pairwise_distance(P[:5], P[:7]); print("pairwise self dist diag (0):", np.diag(np.asarray(D[:5, :5])))
print("pairwise vs best_align:", float(D[0, 3]), float(jnp.sqrt(best_align(P[0], P[3])[1])))
# averaging of randomly transformed copies
keys = jax.random.split(jax.random.PRNGKey(0), 6)
def rnd(k):
    t = jax.random.uniform(k, minval=0, maxval=6.28); Rr = jnp.array([[jnp.cos(t), -jnp.sin(t)], [jnp.sin(t), jnp.cos(t)]])
    p = PERMS[jax.random.randint(k, (), 0, 8)]
    return (a @ Rr.T)[p]
S = jnp.stack([rnd(k) for k in keys]); m = procrustes_mean(S)
print("mean of transformed copies equals shape:", float(relative_error(a, m)), " naive mean err:", float(relative_error(a, S.mean(0))))
