import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np


class RingAttention(eqx.Module):
    """Multi-head self-attention over (context token, n ring tokens) with a learnable bias.

    The bias added to the attention logits of every head is
        [[gamma, alpha, ..., alpha],
         [beta,  b_{(j - i) mod n}  ],     i, j = 0..n-1 for the ring tokens
         [...                       ]]
    i.e. a circulant matrix for the ring tokens, so the attention commutes with cyclic shifts
    of the ring tokens but not with arbitrary permutations.
    """
    n: int = eqx.field(static=True)
    n_heads: int = eqx.field(static=True)
    query: eqx.nn.Linear
    key: eqx.nn.Linear
    value: eqx.nn.Linear
    output: eqx.nn.Linear
    gamma: jax.Array     # (n_heads,)    context -> context
    alpha: jax.Array     # (n_heads,)    context query -> ring keys
    beta: jax.Array      # (n_heads,)    ring queries -> context key
    b: jax.Array         # (n_heads, n)  circulant part, ring -> ring

    def __init__(self, n, N_f, n_heads, key):
        assert N_f % n_heads == 0, "N_f must be divisible by n_heads"
        kq, kk, kv, ko = jr.split(key, 4)
        self.n, self.n_heads = n, n_heads
        self.query = eqx.nn.Linear(N_f, N_f, key=kq)
        self.key = eqx.nn.Linear(N_f, N_f, key=kk)
        self.value = eqx.nn.Linear(N_f, N_f, key=kv)
        self.output = eqx.nn.Linear(N_f, N_f, key=ko)
        self.gamma = jnp.zeros(n_heads)
        self.alpha = jnp.zeros(n_heads)
        self.beta = jnp.zeros(n_heads)
        self.b = jnp.zeros((n_heads, n))

    def bias(self):                                    # (n_heads, n + 1, n + 1)
        n, H = self.n, self.n_heads
        shift = (np.arange(n)[None, :] - np.arange(n)[:, None]) % n
        ring = self.b[:, shift]                        # (H, n, n), ring[h, i, j] = b[h, (j - i) mod n]
        first_row = jnp.concatenate(
            [self.gamma[:, None, None], jnp.broadcast_to(self.alpha[:, None, None], (H, 1, n))], axis=2)
        other_rows = jnp.concatenate(
            [jnp.broadcast_to(self.beta[:, None, None], (H, n, 1)), ring], axis=2)
        return jnp.concatenate([first_row, other_rows], axis=1)

    def __call__(self, x):                             # x: (n + 1, N_f)
        T, N_f = x.shape
        d = N_f // self.n_heads
        split = lambda y: y.reshape(T, self.n_heads, d)
        q, k, v = (split(jax.vmap(f)(x)) for f in (self.query, self.key, self.value))
        logits = jnp.einsum("thd,shd->hts", q, k) / jnp.sqrt(d) + self.bias()
        weights = jax.nn.softmax(logits, axis=-1)
        out = jnp.einsum("hts,shd->thd", weights, v).reshape(T, N_f)
        return jax.vmap(self.output)(out)


class Block(eqx.Module):
    """Pre-norm transformer block with RingAttention."""
    norm1: eqx.nn.LayerNorm
    attention: RingAttention
    norm2: eqx.nn.LayerNorm
    mlp_in: eqx.nn.Linear
    mlp_out: eqx.nn.Linear

    def __init__(self, n, N_f, n_heads, mlp_ratio, key):
        k_attn, k_in, k_out = jr.split(key, 3)
        self.norm1 = eqx.nn.LayerNorm(N_f)
        self.attention = RingAttention(n, N_f, n_heads, k_attn)
        self.norm2 = eqx.nn.LayerNorm(N_f)
        self.mlp_in = eqx.nn.Linear(N_f, mlp_ratio*N_f, key=k_in)
        self.mlp_out = eqx.nn.Linear(mlp_ratio*N_f, N_f, key=k_out)

    def __call__(self, x):
        x = x + self.attention(jax.vmap(self.norm1)(x))
        h = jax.vmap(self.norm2)(x)
        return x + jax.vmap(lambda t: self.mlp_out(jax.nn.gelu(self.mlp_in(t))))(h)


class RingTransformer(eqx.Module):
    """Transformer on a closed polygon X = (x_1, ..., x_n), x_i in R^2, with context c in R^m.

    Forward pass f(c, X) -> (c_tilde, Delta):
        c_tilde (N_inv,)  invariant to O(2) rotations/reflections, translations and cyclic shifts of X,
        Delta   (n, 2)    equivariant: rotated/reflected together with X, invariant to translations,
                          cyclically shifted together with X.
    Reversing the order of X is not a symmetry of f; `forward_flip` makes it one by averaging
    two passes. Normalise the input outside the model (Gram entries grow like the squared size).
    """
    n: int = eqx.field(static=True)
    ring_in: eqx.nn.Linear       # n Gram entries -> N_f, shared by all ring tokens
    context_in: eqx.nn.Linear    # m -> N_f
    blocks: list
    final_norm: eqx.nn.LayerNorm
    invariant_out: eqx.nn.Linear     # context token -> N_inv
    equivariant_out: eqx.nn.Linear   # ring tokens -> n coefficients

    def __init__(self, n, m, N_f, N_inv, key, L, n_heads=4, mlp_ratio=4):
        k_ring, k_ctx, k_blocks, k_inv, k_eq = jr.split(key, 5)
        self.n = n
        self.ring_in = eqx.nn.Linear(n, N_f, key=k_ring)
        self.context_in = eqx.nn.Linear(m, N_f, key=k_ctx)
        self.blocks = [Block(n, N_f, n_heads, mlp_ratio, k) for k in jr.split(k_blocks, L)]
        self.final_norm = eqx.nn.LayerNorm(N_f)
        self.invariant_out = eqx.nn.Linear(N_f, N_inv, key=k_inv)
        self.equivariant_out = eqx.nn.Linear(N_f, n, key=k_eq)

    def __call__(self, c, X):                    # c: (m,), X: (n, 2) -> (N_inv,), (n, 2)
        n = self.n
        X_ = X - jnp.mean(X, axis=0, keepdims=True)
        G = X_ @ X_.T
        cyc = (np.arange(n)[:, None] + np.arange(n)[None, :]) % n     # cyc[i, k] = (i + k) mod n
        ring = G[np.arange(n)[:, None], cyc]     # ring[i, k] = G[i, i + k]: row i, diagonal first
        x = jnp.concatenate([self.context_in(c)[None], jax.vmap(self.ring_in)(ring)], axis=0)
        for block in self.blocks:
            x = block(x)
        x = jax.vmap(self.final_norm)(x)
        c_tilde = self.invariant_out(x[0])
        f = jax.vmap(self.equivariant_out)(x[1:])                     # f[i, k]
        delta = jnp.einsum("ik,ikd->id", f, X_[cyc])                  # delta_i = sum_k f[i, k] x_{i + k}
        return c_tilde, delta

    def forward_flip(self, c, X):
        """Also invariant / equivariant to reversing the order of X."""
        c1, delta1 = self(c, X)
        c2, delta2 = self(c, X[::-1])
        return 0.5*(c1 + c2), 0.5*(delta1 + delta2[::-1])


if __name__ == "__main__":
    # the tests run in double precision: in float32 (on GPU even with reduced TF32 matmuls)
    # a batched and a single evaluation differ by up to ~1e-3
    jax.config.update("jax_enable_x64", True)
    n, m, N_f, N_inv, L, n_heads, mlp_ratio = 5, 4, 32, 6, 3, 4, 4
    key = jr.PRNGKey(0)
    model = RingTransformer(n, m, N_f, N_inv, key, L, n_heads, mlp_ratio)
    # the learnable bias is initialised with zeros: give it random values, otherwise the
    # circulant bias would not be tested at all
    ks = iter(jr.split(jr.PRNGKey(1), 4*L))
    for i in range(L):
        for name in ("gamma", "alpha", "beta", "b"):
            model = eqx.tree_at(lambda mod: getattr(mod.blocks[i].attention, name), model,
                                jr.normal(next(ks), getattr(model.blocks[i].attention, name).shape))
    ok = True

    def check(name, cond):
        global ok
        ok &= bool(cond)
        print(f"{name}: {bool(cond)}")

    def close(a, b, tol=1e-9):
        return bool(jnp.allclose(a, b, atol=tol, rtol=tol))

    kc, kx, ko, kb, kp = jr.split(jr.PRNGKey(2), 5)
    c = jr.normal(kc, (m,))
    X = jr.normal(kx, (n, 2))
    O_rot = jnp.array([[jnp.cos(0.9), -jnp.sin(0.9)], [jnp.sin(0.9), jnp.cos(0.9)]])
    O_ref = O_rot @ jnp.array([[1., 0.], [0., -1.]])        # reflection (det = -1)
    b = jr.normal(kb, (1, 2))
    s = 2

    def transform(X, O=None, shift=None, roll=0, flip=False):
        """Rigid motion x -> O x + b, cyclic shift (x_i -> x_{i + roll}) and optional reversion."""
        if O is not None:
            X = X @ O.T
        if shift is not None:
            X = X + shift
        X = jnp.roll(X, -roll, axis=0)
        return X[::-1] if flip else X

    def delta_transform(D, O=None, roll=0, flip=False):
        """How the output Delta must change for the same transformation (translation: nothing)."""
        if O is not None:
            D = D @ O.T
        D = jnp.roll(D, -roll, axis=0)
        return D[::-1] if flip else D

    cases = {
        "rotation": dict(O=O_rot), "reflection": dict(O=O_ref), "translation": dict(shift=b),
        "cyclic shift": dict(roll=s), "all together": dict(O=O_ref, shift=b, roll=s),
    }
    flip_cases = {"reversion": dict(flip=True), "reversion + all": dict(O=O_ref, shift=b, roll=s, flip=True)}

    for title, method, allowed in [("forward pass f", model, cases),
                                   ("forward pass with flip", model.forward_flip, {**cases, **flip_cases})]:
        print(f"--- {title}")
        c0, D0 = method(c, X)
        check("output shapes", c0.shape == (N_inv,) and D0.shape == (n, 2))
        for name, kw in allowed.items():
            c1, D1 = method(c, transform(X, **kw))
            check(f"c invariant under {name}", close(c1, c0))
            check(f"Delta equivariant under {name}", close(D1, delta_transform(D0, **{k: v for k, v in kw.items() if k != 'shift'})))

    print("--- sanity: the plain forward pass is not symmetric under reversion, the flip pass is")
    c0, D0 = model(c, X)
    c1, D1 = model(c, X[::-1])
    check("f: c changes under reversion", not close(c1, c0, tol=1e-3))
    check("f: Delta is not equivariant under reversion", not close(D1, D0[::-1], tol=1e-3))
    perm = jnp.array([1, 0, 2, 3, 4])    # a transposition is not a cyclic shift or a reversion
    check("f: not invariant under a general permutation", not close(model(c, X[perm])[0], c0, tol=1e-3))
    check("f: c depends on the context", not close(model(c + 1.0, X)[0], c0, tol=1e-3))

    # control: a bias that is not circulant must break the cyclic symmetry, so the tests above can fail
    circulant_bias = RingAttention.bias
    RingAttention.bias = lambda self: circulant_bias(self).at[:, 1, 2].add(3.0)
    check("control: non-circulant bias breaks the cyclic invariance",
          not close(model(c, transform(X, roll=s))[0], model(c, X)[0], tol=1e-3))
    RingAttention.bias = circulant_bias

    print("--- batching, gradients, parameters")
    cb = jr.normal(kc, (7, m)); Xb = jr.normal(kx, (7, n, 2))
    cb_out, Db_out = jax.vmap(model)(cb, Xb)
    check("vmap over a batch", cb_out.shape == (7, N_inv) and Db_out.shape == (7, n, 2) and close(Db_out[3], model(cb[3], Xb[3])[1]))

    def loss(mod, c, X):
        c_t, D = mod.forward_flip(c, X)
        return jnp.sum(c_t**2) + jnp.sum(D**2)

    grads = eqx.filter_grad(loss)(model, c, X)
    check("gradients finite", all(jnp.all(jnp.isfinite(g)) for g in jax.tree_util.tree_leaves(grads)))
    check("gradient reaches the bias", jnp.any(grads.blocks[0].attention.b != 0))

    block = (4*N_f + 4*(N_f*N_f + N_f) + n_heads*(3 + n) + (N_f*mlp_ratio*N_f + mlp_ratio*N_f) + (mlp_ratio*N_f*N_f + N_f))
    expected = (n*N_f + N_f) + (m*N_f + N_f) + L*block + 2*N_f + (N_f*N_inv + N_inv) + (N_f*n + n)
    n_params = sum(p.size for p in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array)))
    print(f"parameters: {n_params}, expected: {expected}")
    check("parameter count", n_params == expected)
    assert ok, "ring transformer test failed"
