import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from drums.models.transformer import Block
from drums.data.preprocessing import canonical_triangle

class DrumTransformerVelocity(eqx.Module):
    spectrum_encoder: eqx.nn.Linear
    coordinate_encoder: eqx.nn.Linear
    time_encoder: eqx.nn.Linear
    vertex_tokens: jax.Array
    blocks: list
    final_norm: eqx.nn.LayerNorm
    decoder: eqx.nn.Linear
    n_vertices: int

    def __init__(self, key, n_vertices=3, n_eig=30, N_f=128, L=3, n_heads=4):
        k_spec, k_coord, k_time, k_tok, k_blocks, k_dec = jr.split(key, 6)

        self.n_vertices = n_vertices
        self.spectrum_encoder = eqx.nn.Linear(n_eig, N_f, key=k_spec)
        self.coordinate_encoder = eqx.nn.Linear(2, N_f, key=k_coord)
        self.time_encoder = eqx.nn.Linear(1, N_f, key=k_time)

        self.vertex_tokens = jr.normal(k_tok, (n_vertices, N_f)) / jnp.sqrt(N_f)

        self.blocks = [
            Block(N_f, n_heads, mlp_ratio=4, key=k) for k in jr.split(k_blocks, L)
        ]

        self.final_norm = eqx.nn.LayerNorm(N_f)
        self.decoder = eqx.nn.Linear(N_f, 2, key=k_dec)

    def __call__(self, t, x_t, spectrum):
        vertices = jax.vmap(self.coordinate_encoder)(x_t)
        vertices = vertices + self.vertex_tokens + self.time_encoder(jnp.atleast_1d(t))

        spectrum_token = self.spectrum_encoder(spectrum)[None]
        tokens = jnp.concatenate([spectrum_token, vertices], axis=0)

        for block in self.blocks:
            tokens = block(tokens)

        vertices = jax.vmap(self.final_norm)(tokens[1:])
        return jax.vmap(self.decoder)(vertices)


class DrumVelocity(eqx.Module):
    mlp: eqx.nn.MLP
    n_vertices: int

    def __init__(self, key, n_vertices=3, n_eig=30):
        self.n_vertices = n_vertices
        self.mlp = eqx.nn.MLP(
            in_size=2 * n_vertices + 1 + n_eig,
            out_size=2 * n_vertices,
            width_size=512,
            depth=3,
            activation=jax.nn.gelu,
            key=key
        )

    def __call__(self, t, x_t, spectrum):
        inputs = jnp.concatenate([
            x_t.ravel(),
            jnp.atleast_1d(t),
            spectrum,
        ])
        return self.mlp(inputs).reshape(self.n_vertices, 2)


def flow_matching_loss(model, spectra, vertices, key):
    noise_key, time_key = jr.split(key)
    x_0 = jr.normal(noise_key, vertices.shape, dtype=vertices.dtype)
    t = jr.uniform(time_key, (vertices.shape[0],), dtype=vertices.dtype)

    time = t[:, None, None]
    x_t = (1 - time) * x_0 + time * vertices
    target_velocity = vertices - x_0

    predicted_velocity = jax.vmap(model)(t, x_t, spectra)
    return jnp.mean((predicted_velocity - target_velocity) ** 2)


def sample(model, spectrum, key, n_steps=30):
    """Integrate the velocity field from t=0 to t=1 using classical RK4."""
    dtype = spectrum.dtype
    x = jr.normal(key, (model.n_vertices, 2), dtype=dtype)
    dt = 1.0 / n_steps

    for i in range(n_steps):
        t = i * dt
        k1 = model(t, x, spectrum)
        k2 = model(t + dt / 2, x + dt * k1 / 2, spectrum)
        k3 = model(t + dt / 2, x + dt * k2 / 2, spectrum)
        k4 = model(t + dt, x + dt * k3, spectrum)
        x = x + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return x


def main(seed=17, canonical=False):
    import numpy as np
    import optax

    data = np.load("triangle_drums.npz")
    raw_spectra = np.array(data["spectra"])
    vertices = jnp.asarray(data["perimeters"], dtype=jnp.float32)
    n_train = int(0.9 * len(raw_spectra))

    # Normalize conditioning features.
    log_spectra = jnp.log(jnp.asarray(raw_spectra, dtype=jnp.float32))
    spectrum_mean = log_spectra[:n_train].mean(axis=0)
    spectrum_std = jnp.maximum(
        log_spectra[:n_train].std(axis=0), 1e-6
    )
    spectra = (log_spectra - spectrum_mean) / spectrum_std

    # One shared coordinate scale preserves relative drum sizes.
    target_scale = jnp.max(
        jnp.linalg.norm(vertices[:n_train], axis=-1)
    )
    if canonical:
        vertices = jnp.asarray(
            [canonical_triangle(triangle) for triangle in data["perimeters"]],
            dtype=jnp.float32,
        )
    vertices = vertices / target_scale

    key = jr.PRNGKey(seed)
    key, init_key = jr.split(key)
    model = DrumTransformerVelocity(init_key, N_f=256)

    n_updates = 5000
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=1e-3,
        warmup_steps=500,
        decay_steps=n_updates,
        end_value=1e-6,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(schedule, weight_decay=1e-4),
    )
    state = optimizer.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def train_step(model, state, spectra, vertices, key):
        loss, grads = eqx.filter_value_and_grad(flow_matching_loss)(
            model, spectra, vertices, key
        )
        updates, state = optimizer.update(grads, state, eqx.filter(model, eqx.is_array))
        return eqx.apply_updates(model, updates), state, loss

    history = []

    for update in range(n_updates):
        key, batch_key, loss_key = jr.split(key, 3)
        indices = jr.randint(batch_key, (64,), 0, n_train)

        model, state, loss = train_step(
            model, state, spectra[indices], vertices[indices], loss_key
        )
        history.append(float(loss))

        if update % 500 == 0:
            # Fixed noise/time draws make loss comparisons less noisy.
            test_loss = flow_matching_loss(
                model,
                spectra[n_train:],
                vertices[n_train:],
                jr.PRNGKey(123),
            )
            print(
                f"{update}: train={float(loss):.4f}, "
                f"test={float(test_loss):.4f}"
            )

    return model, history, spectra, vertices


if __name__ == "__main__":
    model, history, spectra, vertices = main()
