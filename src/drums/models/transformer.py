import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr


class Block(eqx.Module):
    """Pre-norm transformer block: x + Attn(LN(x)), then x + MLP(LN(x))."""
    norm1: eqx.nn.LayerNorm
    attention: eqx.nn.MultiheadAttention
    norm2: eqx.nn.LayerNorm
    mlp_in: eqx.nn.Linear
    mlp_out: eqx.nn.Linear

    def __init__(self, N_f, n_heads, mlp_ratio, key):
        k_attn, k_in, k_out = jr.split(key, 3)
        self.norm1 = eqx.nn.LayerNorm(N_f)
        self.attention = eqx.nn.MultiheadAttention(
            n_heads, N_f, use_query_bias=True, use_key_bias=True, use_value_bias=True,
            use_output_bias=True, key=k_attn)
        self.norm2 = eqx.nn.LayerNorm(N_f)
        self.mlp_in = eqx.nn.Linear(N_f, mlp_ratio*N_f, key=k_in)
        self.mlp_out = eqx.nn.Linear(mlp_ratio*N_f, N_f, key=k_out)

    def __call__(self, x):                       # x: (tokens, N_f)
        h = jax.vmap(self.norm1)(x)
        x = x + self.attention(h, h, h)          # full (non-causal) self-attention
        h = jax.vmap(self.norm2)(x)
        return x + jax.vmap(lambda t: self.mlp_out(jax.nn.gelu(self.mlp_in(t))))(h)


class DrumTransformer(eqx.Module):
    """Predicts the N_v vertices of a drum from N_eig ascending eigenvalues.

    Tokens: [projected spectrum, N_v learnable vertex tokens] -> L blocks ->
    final norm -> linear decoder applied to the last N_v tokens.
    The model acts on a single drum (eigenvalues (N_eig,) -> vertices (N_v, 2));
    use jax.vmap for batches. The input is expected to be normalised already.

    The decoder gives 2 n_drums numbers for every vertex token, so the output has the shape
    (N_v, 2 n_drums): n_drums drums are predicted at once (by default one, shape (N_v, 2)).
    """
    encoder: eqx.nn.Linear
    vertex_tokens: jax.Array
    blocks: list
    final_norm: eqx.nn.LayerNorm
    decoder: eqx.nn.Linear

    def __init__(self, N_v, N_f, N_eig, key, L, n_heads=4, mlp_ratio=4, n_drums=1):
        k_enc, k_tok, k_blocks, k_dec = jr.split(key, 4)
        self.encoder = eqx.nn.Linear(N_eig, N_f, key=k_enc)
        self.vertex_tokens = jr.normal(k_tok, (N_v, N_f)) / jnp.sqrt(N_f)
        self.blocks = [Block(N_f, n_heads, mlp_ratio, k) for k in jr.split(k_blocks, L)]
        self.final_norm = eqx.nn.LayerNorm(N_f)
        self.decoder = eqx.nn.Linear(N_f, 2*n_drums, key=k_dec)

    def __call__(self, eigenvalues):             # (N_eig,) -> (N_v, 2 n_drums)
        N_v = self.vertex_tokens.shape[0]
        x = jnp.concatenate([self.encoder(eigenvalues)[None], self.vertex_tokens], axis=0)
        for block in self.blocks:
            x = block(x)
        x = jax.vmap(self.final_norm)(x[-N_v:])
        return jax.vmap(self.decoder)(x)


if __name__ == "__main__":
    import numpy as np
    import optax

    N_v, N_f, N_eig, L, n_heads, mlp_ratio = 3, 32, 30, 3, 4, 4
    model = DrumTransformer(N_v, N_f, N_eig, jr.PRNGKey(0), L, n_heads, mlp_ratio)
    ok = True

    def check(name, cond):
        global ok
        ok &= bool(cond)
        print(f"{name}: {bool(cond)}")

    x = jr.normal(jr.PRNGKey(1), (N_eig,))
    y = model(x)
    check("output shape (N_v, 2)", y.shape == (N_v, 2))
    check("output finite", jnp.all(jnp.isfinite(y)))

    xb = jr.normal(jr.PRNGKey(2), (5, N_eig))
    yb = jax.vmap(model)(xb)
    check("vmap over a batch", yb.shape == (5, N_v, 2) and np.allclose(yb[0], model(xb[0]), atol=1e-5))

    # parameter count: encoder + vertex tokens + L blocks + final norm + decoder
    block = 2*N_f + 4*(N_f*N_f + N_f) + 2*N_f + (N_f*mlp_ratio*N_f + mlp_ratio*N_f) + (mlp_ratio*N_f*N_f + N_f)
    expected = (N_eig*N_f + N_f) + N_v*N_f + L*block + 2*N_f + (2*N_f + 2)
    n_params = sum(p.size for p in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array)))
    print(f"parameters: {n_params}, expected: {expected}")
    check("parameter count", n_params == expected)

    def loss(m, x, t):
        return jnp.mean((jax.vmap(m)(x) - t)**2)

    grads = eqx.filter_grad(loss)(model, xb, jr.normal(jr.PRNGKey(3), (5, N_v, 2)))
    check("gradient reaches vertex tokens", jnp.any(grads.vertex_tokens != 0))
    check("gradient reaches encoder", jnp.any(grads.encoder.weight != 0))
    check("gradients finite", all(jnp.all(jnp.isfinite(g)) for g in jax.tree_util.tree_leaves(grads)))

    # overfit a small random batch
    t = jr.normal(jr.PRNGKey(4), (5, N_v, 2))
    opt = optax.adam(1e-3)
    state = opt.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def step(m, state):
        l, g = eqx.filter_value_and_grad(loss)(m, xb, t)
        u, state = opt.update(g, state)
        return eqx.apply_updates(m, u), state, l

    losses = []
    for _ in range(300):
        model, state, l = step(model, state)
        losses.append(float(l))
    print(f"loss: {losses[0]:.4f} -> {losses[-1]:.4f}")
    check("loss decreases when overfitting a batch", losses[-1] < 0.1 * losses[0])

    # several drums at once: the decoder gives 2 n_drums numbers for every vertex
    K = 4
    multi = DrumTransformer(N_v, N_f, N_eig, jr.PRNGKey(0), L, n_heads, mlp_ratio, n_drums=K)
    check("n_drums: output shape (N_v, 2 K)", multi(x).shape == (N_v, 2*K))
    check("n_drums: vmap over a batch", jax.vmap(multi)(xb).shape == (5, N_v, 2*K))
    n_multi = sum(p.size for p in jax.tree_util.tree_leaves(eqx.filter(multi, eqx.is_array)))
    check("n_drums: parameter count", n_multi == expected + (K - 1)*2*(N_f + 1))
    assert ok, "transformer test failed"
