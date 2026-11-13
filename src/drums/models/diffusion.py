"""Conditional DDPM on centered drum vertices with a ring noise predictor.

Train a first model with:
    python -m drums.models.diffusion --data triangle_drums.npz --out results/diffusion
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from drums.models.ring_transformer import RingTransformer


def center(x):
    """Center one polygon (N, 2) or a batch (B, N, 2)."""
    return x - x.mean(axis=-2, keepdims=True)


class DiffusionSchedule(eqx.Module):
    n_steps: int = eqx.field(static=True)
    betas: jax.Array
    alphas: jax.Array
    alpha_bar: jax.Array
    posterior_variance: jax.Array

    def __init__(
        self,
        n_steps=1000,
        beta_start=1e-4,
        beta_end=0.02,
    ):
        if n_steps < 1 or not 0 < beta_start <= beta_end < 1:
            raise ValueError("Invalid diffusion schedule")

        self.n_steps = n_steps
        zero = jnp.zeros((1,), dtype=jnp.float32)

        # Index zero is clean data: beta[0] = 0 and alpha_bar[0] = 1.
        self.betas = jnp.concatenate([
            zero,
            jnp.linspace(
                beta_start,
                beta_end,
                n_steps,
                dtype=jnp.float32,
            ),
        ])

        self.alphas = 1.0 - self.betas
        self.alpha_bar = jnp.cumprod(self.alphas)

        variance = (
            self.betas[1:] * (1.0 - self.alpha_bar[:-1]) / (1.0 - self.alpha_bar[1:])
        )
        self.posterior_variance = jnp.concatenate([zero, variance])


class DrumRingTransformerNoise(eqx.Module):
    """Ring predictor; use predict_noise to include optional Gaussian preconditioning."""
    ring: RingTransformer
    n_vertices: int = eqx.field(static=True)
    time_features: int = eqx.field(static=True)
    gaussian_skip: bool = eqx.field(static=True)
    data_std: float = eqx.field(static=True)

    def __init__(
        self,
        key,
        n_vertices=3,
        n_eig=30,
        N_f=128,
        L=3,
        n_heads=4,
        time_features=0,
        gaussian_skip=False,
        data_std=1.0,
    ):
        if time_features < 0 or time_features % 2:
            raise ValueError("time_features must be a nonnegative even integer")
        self.n_vertices = n_vertices
        self.time_features = time_features
        self.gaussian_skip = gaussian_skip
        self.data_std = float(data_std)
        if not 0 < self.data_std < float("inf"):
            raise ValueError("data_std must be finite and positive")

        self.ring = RingTransformer(
            n=n_vertices,
            m=n_eig + 1 + time_features,
            N_f=N_f,
            N_inv=1,
            key=key,
            L=L,
            n_heads=n_heads,
        )
        if gaussian_skip:
            # Begin at the analytic Gaussian estimate; learn its residual.
            self.ring = eqx.tree_at(
                lambda ring: (ring.equivariant_out.weight, ring.equivariant_out.bias),
                self.ring,
                (jnp.zeros_like(self.ring.equivariant_out.weight),
                 jnp.zeros_like(self.ring.equivariant_out.bias)),
            )

    def __call__(self, time, x_t, spectrum):
        """Return the raw centered output (a residual when gaussian_skip is enabled)."""
        time_context = jnp.atleast_1d(time)
        if self.time_features:
            half = self.time_features // 2
            frequencies = jnp.exp(-jnp.log(10000.) * jnp.arange(half) / half)
            phases = 1000. * time * frequencies
            time_context = jnp.concatenate([time_context, jnp.cos(phases), jnp.sin(phases)])
        context = jnp.concatenate([
            time_context,
            spectrum,
        ])

        _, predicted_noise = self.ring.forward_flip(context, x_t)
        return center(predicted_noise)


def predict_noise(model, x_t, spectrum, step, schedule):
    """Convert a raw model output to epsilon, optionally using VP Gaussian preconditioning.

    This adapts the EDM preconditioning idea to the existing VP epsilon target;
    neither the Gaussian corruption kernel nor the noise-MSE objective changes.
    """
    time = jnp.asarray(step, dtype=x_t.dtype) / schedule.n_steps
    if not getattr(model, "gaussian_skip", False):
        return model(time, x_t, spectrum)
    x_t = center(x_t)
    alpha_bar = schedule.alpha_bar[step]
    signal, noise = jnp.sqrt(alpha_bar), jnp.sqrt(1.0 - alpha_bar)
    variance = alpha_bar * model.data_std**2 + (1.0 - alpha_bar)
    residual = model(time, x_t / jnp.sqrt(variance), spectrum)
    return center(
        noise / variance * x_t
        + signal * model.data_std / jnp.sqrt(variance) * residual
    )


def q_sample(vertices, step, schedule, key):
    """Corrupt one drum at an integer step in 0..T; return input and noise target."""
    vertices = center(vertices)
    noise = center(jr.normal(key, vertices.shape, dtype=vertices.dtype))

    alpha_bar = schedule.alpha_bar[step]

    x_t = jnp.sqrt(alpha_bar) * vertices + jnp.sqrt(1. - alpha_bar) * noise

    return x_t, noise


def diffusion_loss(model, spectra, vertices, key, schedule):
    """Noise MSE for spectra (B, n_eig) and vertices (B, n_vertices, 2)."""
    time_key, noise_key = jr.split(key)
    batch_size = vertices.shape[0]

    steps = jr.randint(
        time_key,
        shape=(batch_size,),
        minval=1,
        maxval=schedule.n_steps + 1,
    )

    noise_keys = jr.split(noise_key, batch_size)

    x_t, noise = jax.vmap(
        q_sample,
        in_axes=(0, 0, None, 0),
    )(vertices, steps, schedule, noise_keys)

    predicted_noise = jax.vmap(predict_noise, in_axes=(None, 0, 0, 0, None))(
        model, x_t, spectra, steps, schedule,
    )

    return jnp.mean((predicted_noise - noise)**2)


def reverse_step(model, x_t, spectrum, step, schedule, key):
    """One DDPM update from step to step-1, for an integer step in 1..T."""
    predicted_noise = predict_noise(model, x_t, spectrum, step, schedule)

    mean = (
        x_t
        - schedule.betas[step]
        / jnp.sqrt(1.0 - schedule.alpha_bar[step])
        * predicted_noise
    ) / jnp.sqrt(schedule.alphas[step])

    noise = center(
        jr.normal(key, x_t.shape, dtype=x_t.dtype)
    )

    std = jnp.where(
        step > 1,
        jnp.sqrt(schedule.posterior_variance[step]),
        0.0,
    )

    return center(mean + std * noise)


@eqx.filter_jit
def sample(model, spectrum, key, schedule):
    """Generate one drum using every reverse diffusion step."""
    init_key, loop_key = jr.split(key)

    x_T = center(
        jr.normal(
            init_key,
            shape=(model.n_vertices, 2),
            dtype=spectrum.dtype,
        )
    )

    def body(i, carry):
        x_t, key = carry
        key, step_key = jr.split(key)

        step = schedule.n_steps - i
        x_prev = reverse_step(
            model,
            x_t,
            spectrum,
            step,
            schedule,
            step_key,
        )

        return x_prev, key

    x_0, _ = jax.lax.fori_loop(
        0,
        schedule.n_steps,
        body,
        (x_T, loop_key),
    )

    return x_0


def ddim_step(model, x_t, spectrum, step, previous_step, schedule, key, eta=0.0):
    """A respaced DDIM update using the original training schedule (Song et al., 2020)."""
    alpha_bar = schedule.alpha_bar[step]
    previous_alpha_bar = schedule.alpha_bar[previous_step]
    noise_prediction = predict_noise(model, x_t, spectrum, step, schedule)
    clean_prediction = (
        x_t - jnp.sqrt(1.0 - alpha_bar) * noise_prediction
    ) / jnp.sqrt(alpha_bar)
    variance = eta**2 * (1.0 - previous_alpha_bar) / (1.0 - alpha_bar) * (
        1.0 - alpha_bar / previous_alpha_bar
    )
    noise = center(jr.normal(key, x_t.shape, dtype=x_t.dtype))
    return center(
        jnp.sqrt(previous_alpha_bar) * clean_prediction
        + jnp.sqrt(jnp.maximum(1.0 - previous_alpha_bar - variance, 0.0)) * noise_prediction
        + jnp.sqrt(jnp.maximum(variance, 0.0)) * noise
    )


@eqx.filter_jit
def sample_ddim(model, spectrum, key, schedule, n_steps=200, eta=0.0):
    """Sample on a respaced grid; eta=0 is deterministic given the initial noise."""
    if not 1 <= n_steps <= schedule.n_steps or not 0 <= eta <= 1:
        raise ValueError("Require 1 <= sampling steps <= training steps and 0 <= eta <= 1")
    grid = jnp.rint(jnp.linspace(0, schedule.n_steps, n_steps + 1)).astype(jnp.int32)
    init_key, loop_key = jr.split(key)
    x_T = center(jr.normal(init_key, (model.n_vertices, 2), dtype=spectrum.dtype))

    def body(i, carry):
        x_t, key = carry
        key, step_key = jr.split(key)
        j = n_steps - i
        x_previous = ddim_step(
            model, x_t, spectrum, grid[j], grid[j - 1], schedule, step_key, eta,
        )
        return x_previous, key

    x_0, _ = jax.lax.fori_loop(0, n_steps, body, (x_T, loop_key))
    return x_0


def spectrum_features(spectra, kind="log_ratios"):
    """Features before standardization; retain the size information in lambda_1."""
    logs = jnp.log(jnp.asarray(spectra, dtype=jnp.float32))
    if kind == "log":
        return logs
    if kind == "log_ratios":
        return jnp.concatenate([logs[..., :1], logs[..., 1:] - logs[..., :1]], axis=-1)
    raise ValueError("features must be 'log' or 'log_ratios'")


def prepare_data(path, train_fraction=0.9, features="log_ratios", coordinate_scale="max"):
    """Use a sequential train/validation split and fit preprocessing on train only."""
    import numpy as np

    with np.load(path) as data:
        vertices = np.asarray(data["perimeters"], dtype=np.float32)
        raw_spectra = np.asarray(data["spectra"], dtype=np.float32)
    if (vertices.ndim != 3 or vertices.shape[-1] != 2 or vertices.shape[1] < 3
            or raw_spectra.ndim != 2 or raw_spectra.shape[1] < 1
            or len(raw_spectra) != len(vertices)):
        raise ValueError("Expected perimeters (N, n_vertices, 2) and spectra (N, n_eig)")
    if (not np.all(np.isfinite(vertices))
            or not np.all(np.isfinite(raw_spectra) & (raw_spectra > 0))):
        raise ValueError("Data must be finite, with positive eigenvalues")
    n_train = int(train_fraction * len(vertices))
    if not 0 < train_fraction < 1 or not 1 <= n_train < len(vertices):
        raise ValueError("The split must contain both training and validation samples")

    vertices = center(jnp.asarray(vertices))
    if coordinate_scale == "max":
        target_scale = jnp.max(jnp.linalg.norm(vertices[:n_train], axis=-1))
    elif coordinate_scale == "rms":
        # A centered polygon has 2*(n_vertices-1) Gaussian degrees of freedom.
        target_scale = jnp.sqrt(
            jnp.mean(jnp.sum(vertices[:n_train]**2, axis=(-2, -1)))
            / (2 * (vertices.shape[1] - 1))
        )
    else:
        raise ValueError("coordinate_scale must be 'max' or 'rms'")
    if not float(target_scale) > 0:
        raise ValueError("Training shapes must have nonzero size")
    values = spectrum_features(raw_spectra, features)
    mean = values[:n_train].mean(axis=0)
    std = jnp.maximum(values[:n_train].std(axis=0), 1e-6)
    preprocessing = dict(
        spectrum_mean=np.asarray(mean), spectrum_std=np.asarray(std),
        target_scale=np.asarray(target_scale), n_train=n_train, features=features,
        coordinate_scale=coordinate_scale,
    )
    return (values - mean) / std, vertices / target_scale, preprocessing


def main(
    seed=17, data="triangle_drums.npz", n_updates=5000, batch_size=64,
    n_steps=1000, N_f=128, L=3, n_heads=4, learning_rate=1e-3,
    eval_every=500, features="log_ratios", train_fraction=0.9, out=None,
    time_features=0, coordinate_scale="max", train_chunk=100, gaussian_skip=False,
):
    """Train a conditional DDPM; return model, history, spectra, vertices, schedule.

    If out is provided, save weights, configuration, preprocessing and loss history.
    The held-out split is validation; this entry point does not evaluate shape quality.
    """
    import json
    from pathlib import Path

    import numpy as np
    import optax

    if min(n_updates, batch_size, eval_every, train_chunk) < 1 or learning_rate <= 0:
        raise ValueError("Updates, batch size, evaluation interval and learning rate must be positive")
    spectra, vertices, preprocessing = prepare_data(data, train_fraction, features, coordinate_scale)
    n_train = preprocessing["n_train"]
    data_std = float(jnp.sqrt(
        jnp.mean(jnp.sum(vertices[:n_train]**2, axis=(-2, -1)))
        / (2 * (vertices.shape[1] - 1))
    ))
    schedule = DiffusionSchedule(n_steps=n_steps)
    key, init_key = jr.split(jr.PRNGKey(seed))
    model = DrumRingTransformerNoise(
        init_key, n_vertices=vertices.shape[1], n_eig=spectra.shape[1],
        N_f=N_f, L=L, n_heads=n_heads, time_features=time_features,
        gaussian_skip=gaussian_skip, data_std=data_std,
    )

    config = dict(
        seed=seed, data=str(data), n_updates=n_updates, batch_size=batch_size,
        n_steps=n_steps, beta_start=1e-4, beta_end=0.02,
        n_vertices=vertices.shape[1], n_eig=spectra.shape[1],
        N_f=N_f, L=L, n_heads=n_heads, time_features=time_features, learning_rate=learning_rate,
        features=features, coordinate_scale=coordinate_scale, train_chunk=train_chunk,
        gaussian_skip=gaussian_skip, data_std=data_std,
        train_fraction=train_fraction, n_train=n_train,
        n_validation=len(vertices) - n_train, jax=jax.__version__,
        equinox=eqx.__version__, optax=optax.__version__,
    )
    if out is not None:
        out = Path(out)
        if (out / "metadata.json").exists() or (out / "model.eqx").exists():
            raise FileExistsError(f"Choose a new output directory: {out}")
        out.mkdir(parents=True, exist_ok=True)
        (out / "metadata.json").write_text(json.dumps(config, indent=2) + "\n")
        np.savez(out / "preprocessing.npz", **preprocessing)

    lr = learning_rate
    if n_updates > 1:
        lr = optax.warmup_cosine_decay_schedule(
            init_value=0., peak_value=learning_rate,
            warmup_steps=min(500, max(1, n_updates // 10)),
            decay_steps=n_updates, end_value=1e-6,
        )
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.), optax.adamw(lr, weight_decay=1e-4),
    )
    state = optimizer.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def train_step(model, state, spectra, vertices, key):
        loss, grads = eqx.filter_value_and_grad(diffusion_loss)(
            model, spectra, vertices, key, schedule,
        )
        updates, state = optimizer.update(grads, state, eqx.filter(model, eqx.is_array))
        return eqx.apply_updates(model, updates), state, loss

    @eqx.filter_jit
    def update_chunk(model, state, key, spectra, vertices, count):
        def update(carry, _):
            model, state, key = carry
            key, batch_key, loss_key = jr.split(key, 3)
            indices = jr.randint(batch_key, (batch_size,), 0, n_train)
            model, state, loss = train_step(
                model, state, spectra[indices], vertices[indices], loss_key,
            )
            return (model, state, key), loss
        return jax.lax.scan(update, (model, state, key), None, length=count)

    validation_loss = eqx.filter_jit(diffusion_loss)
    validation_key = jr.PRNGKey(123)
    history, validation_steps, validation_history = [], [], []
    completed = 0
    while completed < n_updates:
        count = min(train_chunk, n_updates - completed, eval_every - completed % eval_every)
        (model, state, key), losses = update_chunk(
            model, state, key, spectra, vertices, count,
        )
        losses = np.asarray(losses)
        if not np.isfinite(losses).all():
            raise FloatingPointError(f"Nonfinite training loss after update {completed}")
        history.extend(losses.tolist())
        completed += count
        if completed == count or completed % eval_every == 0 or completed == n_updates:
            val = float(validation_loss(
                model, spectra[n_train:], vertices[n_train:], validation_key, schedule,
            ))
            if not np.isfinite(val):
                raise FloatingPointError(f"Nonfinite validation loss at update {completed}")
            validation_steps.append(completed)
            validation_history.append(val)
            print(f"{completed}: train={float(losses.mean()):.4f}, validation={val:.4f}", flush=True)

    if out is not None:
        eqx.tree_serialise_leaves(out / "model.eqx", model)
        np.savez(out / "history.npz", train_loss=history,
                 validation_steps=validation_steps, validation_loss=validation_history)
    return model, history, spectra, vertices, schedule


def load_checkpoint(folder):
    """Rebuild a saved model and schedule; return them with fitted preprocessing."""
    import json
    from pathlib import Path

    import numpy as np

    folder = Path(folder)
    config = json.loads((folder / "metadata.json").read_text())
    template = DrumRingTransformerNoise(
        jr.PRNGKey(config["seed"]),
        **{name: config[name] for name in ("n_vertices", "n_eig", "N_f", "L", "n_heads")},
        time_features=config.get("time_features", 0),
        gaussian_skip=config.get("gaussian_skip", False), data_std=config.get("data_std", 1.0),
    )
    model = eqx.tree_deserialise_leaves(folder / "model.eqx", template)
    schedule = DiffusionSchedule(
        **{name: config[name] for name in ("n_steps", "beta_start", "beta_end")},
    )
    with np.load(folder / "preprocessing.npz") as data:
        preprocessing = {name: data[name] for name in data.files}
    return model, schedule, preprocessing


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="triangle_drums.npz")
    parser.add_argument("--out", default="results/diffusion")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--n-updates", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-steps", type=int, default=1000,
                        help="Forward schedule length; short schedules may retain substantial signal")
    parser.add_argument("--width", type=int, default=128, dest="N_f")
    parser.add_argument("--layers", type=int, default=3, dest="L")
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--train-chunk", type=int, default=100,
                        help="Compile updates together to avoid per-update host synchronization")
    parser.add_argument("--features", choices=("log", "log_ratios"), default="log_ratios")
    parser.add_argument("--coordinate-scale", choices=("max", "rms"), default="max",
                        help="One training-set scale: maximum radius or RMS per centered degree of freedom")
    parser.add_argument("--time-features", type=int, default=0,
                        help="Extra sinusoidal time features; zero keeps the original scalar-time model")
    parser.add_argument("--gaussian-skip", action="store_true",
                        help="Predict epsilon using a train-fitted Gaussian skip and learned ring residual")
    main(**vars(parser.parse_args()))
