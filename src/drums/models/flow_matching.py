import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr


class DrumVelocity(eqx.Module):
    mlp: eqx.nn.MLP
    n_vertices: int

    def __init__(self, key, n_vertices=3, n_eig=30):
        self.n_vertices = n_vertices
        self.mlp = eqx.nn.MLP(
            in_size=2 * n_vertices + 1 + n_eig,
            out_size=2 * n_vertices,
            width_size=256,
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


def sample(model, spectrum, key, n_steps=100):
    dtype = model.mlp.layers[0].weight.dtype
    x = jr.normal(key, (model.n_vertices, 2), dtype=dtype)
    dt = 1.0 / n_steps

    for i in range(n_steps):
        x = x + dt * model(i * dt, x, spectrum)

    return x


def main():
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
    vertices = vertices / target_scale

    key = jr.PRNGKey(42)
    key, init_key = jr.split(key)
    model = DrumVelocity(init_key)

    n_updates = 20000
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=1e-3,
        warmup_steps=1000,
        decay_steps=n_updates,
        end_value=1e-5,
    )
    optimizer = optax.adamw(schedule, weight_decay=1e-4)
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

    return model, history


if __name__ == "__main__":
    model, history = main()
