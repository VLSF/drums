"""Shared code for the isospectral-drums experiments: data, symmetries, alignment, features.

A drum is a (9, 2) array of vertices. Its spectrum does not change under rotations,
reflections, translations and relabelings of the vertices that preserve the cyclic order
(the dihedral group D9, 18 elements). Everything below treats shapes modulo this group,
but keeps the size: no scaling is allowed when aligning.
"""
import numpy as np
import jax
import jax.numpy as jnp
from jax import vmap

N_V = 9
# the 18 relabelings: cyclic shifts of the vertex sequence and of the reversed sequence
PERMS = np.array([np.roll(np.arange(N_V)[::s], k) for k in range(N_V) for s in (1, -1)])
N_TRAIN, N_VAL, N_TEST = 27000, 1500, 1500


# ----------------------------------------------------------------------------- data
def load_data(path):
    """Log-spectra, vertices of the drums and of their isospectral twins (same scale), the scale."""
    d = np.load(path)
    vertices = jnp.asarray(d["perimeters"], dtype=jnp.float32)
    twins = jnp.asarray(d["twin_perimeters"], dtype=jnp.float32)
    vertices = vertices - vertices.mean(axis=1, keepdims=True)
    twins = twins - twins.mean(axis=1, keepdims=True)
    scale = jnp.max(jnp.linalg.norm(vertices[:N_TRAIN], axis=-1))   # one shared scale, as in the notebook
    log_spectra = jnp.log(jnp.asarray(d["spectra"], dtype=jnp.float32))
    return log_spectra, vertices / scale, twins / scale, scale


def split(x):
    return x[:N_TRAIN], x[N_TRAIN:N_TRAIN + N_VAL], x[N_TRAIN + N_VAL:N_TRAIN + N_VAL + N_TEST]


# ------------------------------------------------------------------------ alignment
def _rotate_onto(a, b):
    """Rotation of b (n, 2) that minimises |a - R b|^2, and the residual |a|^2 + |b|^2 - 2|s|."""
    s_re = jnp.sum(a[:, 0]*b[:, 0] + a[:, 1]*b[:, 1])
    s_im = jnp.sum(a[:, 1]*b[:, 0] - a[:, 0]*b[:, 1])
    s = jnp.sqrt(s_re**2 + s_im**2 + 1e-12)
    c, sn = s_re / s, s_im / s
    rotated = jnp.stack([b[:, 0]*c - b[:, 1]*sn, b[:, 0]*sn + b[:, 1]*c], axis=1)
    return rotated, jnp.sum(a**2) + jnp.sum(b**2) - 2*s


def align_pair(a, b):
    """b rotated/reflected (optimal O(2) map) onto a; both are centred. Fixed vertex labelling."""
    b_rot, r_rot = _rotate_onto(a, b)
    b_ref, r_ref = _rotate_onto(a, b*jnp.array([1., -1.]))
    use_rot = r_rot <= r_ref
    return jnp.where(use_rot, b_rot, b_ref), jnp.where(use_rot, r_rot, r_ref)


def centre(x):
    return x - jnp.mean(x, axis=0, keepdims=True)


def best_align(a, b):
    """b relabeled (18 dihedral orderings) and rotated/reflected to be as close as possible to a.

    Returns (aligned b, squared Frobenius distance)."""
    cand, res = vmap(lambda p: align_pair(a, centre(b[p])))(PERMS)
    i = jnp.argmin(res)
    return cand[i], res[i]


def relative_error(target, prediction):
    """Mean over vertices of |target_k - aligned prediction_k| / |target_k|, best over the 18 labelings."""
    def error(p):
        aligned, _ = align_pair(target, centre(prediction[p]))
        return jnp.mean(jnp.linalg.norm(target - aligned, axis=1) / jnp.linalg.norm(target, axis=1))
    return jnp.min(vmap(error)(PERMS))


def pairwise_distance(A, B):
    """Frobenius distance modulo D9 and O(2) between all shapes of A (nA, 9, 2) and B (nB, 9, 2), centred."""
    za = A[..., 0] + 1j*A[..., 1]
    zb = B[..., 0] + 1j*B[..., 1]
    best = jnp.zeros((A.shape[0], B.shape[0]), dtype=A.dtype)
    with jax.default_matmul_precision("float32"):                  # TF32 would ruin the cancellation below
        for p in PERMS:
            zp = zb[:, p]
            best = jnp.maximum(best, jnp.abs(za @ jnp.conj(zp).T))     # rotations
            best = jnp.maximum(best, jnp.abs(za @ zp.T))               # reflections
    na = jnp.sum(A**2, axis=(1, 2))[:, None]
    nb = jnp.sum(B**2, axis=(1, 2))[None, :]
    return jnp.sqrt(jnp.maximum(na + nb - 2*best, 0.))


def procrustes_mean(shapes, n_iter=10):
    """Frechet-style mean of shapes (m, 9, 2) modulo D9 x O(2): align all to a reference, average, repeat.

    Plain averaging of the coordinates is meaningless because every sample has its own
    orientation and labelling. The reference starts at the medoid."""
    shapes = vmap(centre)(shapes)
    d = vmap(lambda a: vmap(lambda b: best_align(a, b)[1])(shapes))(shapes)
    ref = shapes[jnp.argmin(jnp.sum(d, axis=1))]
    for _ in range(n_iter):
        ref = jnp.mean(vmap(lambda b: best_align(ref, b)[0])(shapes), axis=0)
    return ref


def spread(shapes, mean):
    """RMS distance of the samples to the mean, relative to the size of the mean."""
    d2 = vmap(lambda b: best_align(mean, b)[1])(shapes)
    return jnp.sqrt(jnp.mean(d2) / jnp.sum(mean**2))


# ------------------------------------------------------------------------- features
class Features:
    """Input representations of the log-spectrum, fitted on the training set.

    naive        standardised log lambda_k (what the notebook uses)
    ratio        standardised log(lambda_k / lambda_1), k >= 2, and standardised log lambda_1
    ratio_white  as 'ratio', but PCA-whitened on the training set (shape directions get unit variance)
    A suffix _kNN keeps only the first NN eigenvalues, e.g. ratio_k10.
    """
    def __init__(self, kind, log_spectra_train):
        self.kind = kind
        base, _, k = kind.partition("_k")
        self.base, self.k = base, int(k) if k else log_spectra_train.shape[1]
        self.mu = None
        z = self._raw(log_spectra_train)
        self.mu, self.sigma = z.mean(0), jnp.maximum(z.std(0), 1e-6)
        if base == "ratio_white":
            u, s, vt = jnp.linalg.svd((z - self.mu) / self.sigma, full_matrices=False)
            self.proj = vt.T / (s / jnp.sqrt(len(z)))
        self.dim = self(log_spectra_train[:2]).shape[1]

    def _raw(self, L):
        L = L[:, :self.k]
        if self.base == "naive":
            return L
        return jnp.concatenate([L[:, 1:] - L[:, :1], L[:, :1]], axis=1)

    def __call__(self, L):
        z = (self._raw(L) - self.mu) / self.sigma
        return z @ self.proj if self.base == "ratio_white" else z


# --------------------------------------------------------------------------- training
def train(model, loss_fn, arrays, n_updates, batch, lr, key, eval_fn=None, eval_every=2000, tag=""):
    """AdamW + warm-up + cosine decay + gradient clipping. loss_fn(model, *batch_arrays, key) -> scalar."""
    import equinox as eqx
    import optax
    schedule = optax.warmup_cosine_decay_schedule(0., lr, 500, n_updates, 1e-6)
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(schedule, weight_decay=1e-4))
    state = optimizer.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def step(model, state, batch_arrays, key):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model, *batch_arrays, key)
        updates, state = optimizer.update(grads, state, eqx.filter(model, eqx.is_array))
        return eqx.apply_updates(model, updates), state, loss

    n = len(arrays[0])
    history = []
    for i in range(n_updates):
        key, k_batch, k_loss = jax.random.split(key, 3)
        idx = jax.random.randint(k_batch, (batch,), 0, n)
        model, state, loss = step(model, state, tuple(a[idx] for a in arrays), k_loss)
        history.append(float(loss))
        if eval_fn is not None and (i % eval_every == 0 or i == n_updates - 1):
            print(f"{tag} step {i}: train loss {np.mean(history[-200:]):.4f}  {eval_fn(model)}", flush=True)
    return model, history
