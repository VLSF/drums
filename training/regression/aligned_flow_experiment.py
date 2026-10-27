"""Train the aligned flow for 20k updates and evaluate the full drum dataset.

Run from the repository root with --train-only or --eval-only for either stage.
Checkpoints at 5k, 10k and 20k retain the model, Adam state and random key.
The first 5k updates use the original schedule; continuation decays 1e-4 to 1e-6.
Inference defaults to Euler/50 steps and 32 independent draws per drum.
Use --features log_ratios for standardized log(lambda_1) and log spectral ratios.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from drums.models.flow_matching import DrumRingTransformerVelocity

ORDERINGS = np.array([[0, 1, 2], [2, 1, 0], [2, 0, 1],
                      [1, 0, 2], [1, 2, 0], [0, 2, 1]])


def make_schedule(updates=20000):
    original = optax.warmup_cosine_decay_schedule(
        init_value=0., peak_value=1e-3, warmup_steps=500,
        decay_steps=5000, end_value=1e-6,
    )
    continuation = optax.cosine_decay_schedule(
        init_value=1e-4, decay_steps=max(updates - 5000, 1), alpha=.01,
    )
    return lambda step: jnp.where(step < 5000, original(step), continuation(step - 5000))


def make_optimizer(updates=20000):
    return optax.chain(optax.clip_by_global_norm(1.),
                       optax.adamw(make_schedule(updates), weight_decay=1e-4))


def template_model(seed=1729, width=256, updates=20000):
    key, init_key = jax.random.split(jax.random.PRNGKey(seed))
    model = DrumRingTransformerVelocity(init_key, N_f=width)
    optimizer = make_optimizer(updates)
    return model, optimizer.init(eqx.filter(model, eqx.is_array)), key, optimizer


def load_checkpoint(out, step=20000):
    """Load the existing (model, optimizer state, PRNG) checkpoint format."""
    out = Path(out)
    config = json.loads((out / "metadata.json").read_text())
    model, state, key, optimizer = template_model(
        config["seed"], config["width"], config["updates"],
    )
    model, state, key = eqx.tree_deserialise_leaves(
        out / f"checkpoint_{step:07d}.eqx", (model, state, key),
    )
    return model, state, key, optimizer


def preprocess(data_path, features="log"):
    data = np.load(data_path)
    vertices = np.asarray(data["perimeters"], dtype=np.float32)
    logs = np.log(np.asarray(data["spectra"], dtype=np.float32))
    n_train = int(.9 * len(vertices))
    if features == "log":
        values = logs
    elif features == "log_ratios":
        values = np.concatenate([logs[:, :1], logs[:, 1:] - logs[:, :1]], axis=1)
    else:
        raise ValueError("features must be 'log' or 'log_ratios'")
    mean = values[:n_train].mean(axis=0)
    std = np.maximum(values[:n_train].std(axis=0), 1e-6)
    vertices -= vertices.mean(axis=1, keepdims=True)
    scale = np.linalg.norm(vertices[:n_train], axis=-1).max()
    return dict(spectra=(values - mean) / std, vertices=vertices / scale,
                spectrum_mean=mean, spectrum_std=std, target_scale=scale, n_train=n_train)


def prepare_pairs(vertices, key):
    noise_key, time_key = jax.random.split(key)
    noise = jax.random.normal(noise_key, vertices.shape, dtype=vertices.dtype)
    noise -= noise.mean(axis=1, keepdims=True)

    def align_one(x0, x1):
        ordered = x1[ORDERINGS]
        u, _, vt = jnp.linalg.svd(ordered.swapaxes(1, 2) @ x0, full_matrices=False)
        candidates = ordered @ (u @ vt)
        return candidates[jnp.argmin(jnp.sum((candidates - x0)**2, axis=(1, 2)))]

    endpoint = jax.vmap(align_one)(noise, vertices)
    t = jax.random.uniform(time_key, (len(vertices),), dtype=vertices.dtype)
    return t, (1 - t[:, None, None]) * noise + t[:, None, None] * endpoint, endpoint - noise


def velocity_loss(model, spectra, t, x_t, velocity):
    return jnp.mean((jax.vmap(model)(t, x_t, spectra) - velocity)**2)


def train(args):
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    if (out / "metadata.json").exists() or list(out.glob("checkpoint_*.eqx")):
        raise ValueError("Training output already exists; choose a new --out directory")
    data = preprocess(args.data, args.features)
    np.savez(out / "preprocessing.npz", **data)
    model, state, key, optimizer = template_model(args.seed, args.width, args.updates)
    checkpoints = sorted({s for s in (5000, 10000, args.updates) if s <= args.updates})
    config = dict(seed=args.seed, width=args.width, updates=args.updates,
                  batch_size=args.batch_size, checkpoints=checkpoints,
                  layers=len(model.ring.blocks), heads=model.ring.blocks[0].attention.n_heads,
                  n_vertices=model.n_vertices, n_eig=data["spectra"].shape[1],
                  parameter_count=sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array))),
                  n_train=int(data["n_train"]), n_test=len(data["vertices"]) - int(data["n_train"]),
                  jax=jax.__version__, equinox=eqx.__version__, optax=optax.__version__,
                  schedule="Original cosine through 5000, then cosine 1e-4 to 1e-6; Adam moments retained",
                  noise="Centred standard Gaussian", features=args.features,
                  conditioning=("Column-standardized log eigenvalues" if args.features == "log"
                                else "Train-column-standardized log(lambda_1) and log(lambda_k/lambda_1)"),
                  feature_formula=("log(lambda_k), k=1..30" if args.features == "log"
                                   else "[log(lambda_1), log(lambda_2)-log(lambda_1), ..., log(lambda_30)-log(lambda_1)]"))
    if args.features == "log_ratios":
        config["feature_roles"] = dict(common_level="log(lambda_1); depends on size and shape",
                                       scale_invariant="log(lambda_k/lambda_1), k=2..30")
    (out / "metadata.json").write_text(json.dumps(config, indent=2) + "\n")
    device, cpu = jax.devices()[0], jax.devices("cpu")[0]
    spectra, vertices = (jax.device_put(data[name], cpu) for name in ("spectra", "vertices"))
    key = jax.device_put(key, cpu)
    pairs = jax.jit(prepare_pairs, device=cpu)
    n_train = int(data["n_train"])
    with jax.default_device(cpu):
        test_pairs = pairs(vertices[n_train:], jax.random.PRNGKey(123))
        rates = np.asarray(jax.jit(jax.vmap(make_schedule(args.updates)))(jnp.arange(args.updates)))
    test_inputs = tuple(jax.device_put(x, device) for x in (spectra[n_train:], *test_pairs))
    test_loss = eqx.filter_jit(velocity_loss)

    @eqx.filter_jit
    def step(model, state, spectra, t, x_t, velocity):
        loss, grads = eqx.filter_value_and_grad(velocity_loss)(model, spectra, t, x_t, velocity)
        updates, state = optimizer.update(grads, state, eqx.filter(model, eqx.is_array))
        return eqx.apply_updates(model, updates), state, loss

    start, rolling = time.perf_counter(), []
    with (out / "training.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["update", "train_loss", "test_velocity_loss",
                                              "learning_rate", "elapsed_seconds"])
        writer.writeheader()
        for update in range(args.updates):
            key, batch_key, loss_key = jax.random.split(key, 3)
            indices = jax.random.randint(batch_key, (args.batch_size,), 0, n_train)
            inputs = (spectra[indices], *pairs(vertices[indices], loss_key))
            model, state, loss = step(model, state, *(jax.device_put(x, device) for x in inputs))
            value = float(loss)
            if not np.isfinite(value):
                raise FloatingPointError(f"Nonfinite loss at update {update + 1}")
            rolling.append(value)
            completed = update + 1
            if completed == 1 or completed % 500 == 0 or completed in checkpoints:
                row = dict(update=completed, train_loss=float(np.mean(rolling)),
                           test_velocity_loss=float(test_loss(model, *test_inputs)),
                           learning_rate=float(rates[update]), elapsed_seconds=time.perf_counter() - start)
                writer.writerow(row)
                f.flush()
                print(row, flush=True)
                rolling.clear()
            if completed in checkpoints:
                target = out / f"checkpoint_{completed:07d}.eqx"
                temporary = target.with_suffix(".tmp.eqx")
                eqx.tree_serialise_leaves(temporary, (model, state, key))
                temporary.replace(target)


def make_initial_noise(seed, dataset_ids, draws):
    """Keys depend only on dataset/draw ID, preserving noise across batching."""
    def one_drum(dataset_id):
        key = jax.random.fold_in(jax.random.PRNGKey(seed), dataset_id)
        def one_draw(draw_id):
            x = jax.random.normal(jax.random.fold_in(key, draw_id), (3, 2), dtype=jnp.float32)
            return x - x.mean(axis=0, keepdims=True)
        return jax.vmap(one_draw)(jnp.arange(draws, dtype=jnp.uint32))
    return jax.vmap(one_drum)(jnp.asarray(dataset_ids, dtype=jnp.uint32))


@eqx.filter_jit
def sample_batch(model, spectra, noise, steps=50, method="euler"):
    dt = jnp.asarray(1 / steps, dtype=noise.dtype)
    def one(spectrum, x):
        def body(i, x):
            t = jnp.asarray(i, dtype=x.dtype) * dt
            k1 = model(t, x, spectrum)
            if method == "euler":
                return x + dt * k1
            if method == "heun":
                k2 = model(t + dt, x + dt * k1, spectrum)
                return x + dt * (k1 + k2) / 2
            k2 = model(t + dt / 2, x + dt * k1 / 2, spectrum)
            k3 = model(t + dt / 2, x + dt * k2 / 2, spectrum)
            k4 = model(t + dt, x + dt * k3, spectrum)
            return x + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        return jax.lax.fori_loop(0, steps, body, x)
    return jax.vmap(one)(spectra, noise)


def aligned_candidates(reference, prediction):
    """Six vertex orderings, each optimally rotated/reflected onto reference."""
    ordered = np.asarray(prediction)[..., ORDERINGS, :]
    u, _, vt = np.linalg.svd(ordered.swapaxes(-1, -2) @ reference, full_matrices=False)
    return ordered @ (u @ vt)


def relative_error(target, prediction):
    target, prediction = np.asarray(target, dtype=float), np.asarray(prediction, dtype=float)
    target = target - target.mean(axis=0, keepdims=True)
    prediction = prediction - prediction.mean(axis=0, keepdims=True)
    if not np.all(np.isfinite(prediction)):
        return float("inf")
    aligned = aligned_candidates(target, prediction)
    return float(np.min(np.mean(np.linalg.norm(target - aligned, axis=-1)
                                / np.linalg.norm(target, axis=-1), axis=-1)))


def mean_shape(samples, n_iterations=3):
    """Generalized Procrustes mean; uses generated shapes only, retaining size."""
    samples = np.asarray(samples, dtype=float)
    if not np.all(np.isfinite(samples)):
        return np.full((3, 2), np.nan)
    samples = samples - samples.mean(axis=1, keepdims=True)
    reference = samples[0]
    for _ in range(n_iterations):
        candidates = aligned_candidates(reference, samples)
        best = np.argmin(np.sum((candidates - reference)**2, axis=(-2, -1)), axis=1)
        reference = candidates[np.arange(len(samples)), best].mean(axis=0)
    return reference - reference.mean(axis=0, keepdims=True)


def summary_stats(errors):
    errors = np.asarray(errors)
    finite = errors[np.isfinite(errors)]
    names = ("mean", "median", "min", "max", "p5", "p95", "std")
    values = ([finite.mean(), np.median(finite), finite.min(), finite.max(),
               *np.percentile(finite, [5, 95]), finite.std()] if finite.size else [np.nan] * len(names))
    return dict(count=int(errors.size), nonfinite_count=int(errors.size - finite.size),
                **dict(zip(names, map(float, values))))


def write_full_dataset(out, samples, dataset_ids=None):
    """Save coordinates, per-drum errors, and all/train/test distribution stats."""
    out = Path(out)
    data = np.load(out / "preprocessing.npz")
    dataset_ids = np.arange(len(samples)) if dataset_ids is None else np.asarray(dataset_ids)
    targets = data["vertices"][dataset_ids]
    errors = np.array([[relative_error(target, draw) for draw in draws]
                       for target, draws in zip(targets, samples)])
    means = np.array([mean_shape(draws) for draws in samples])
    draw_error = errors.mean(axis=1)
    shape_error = np.array([relative_error(target, mean) for target, mean in zip(targets, means)])
    np.savez_compressed(out / "full_dataset.npz", dataset_ids=dataset_ids, samples=samples,
                        mean_shapes=means, draw_errors=errors, mean_draw_error=draw_error,
                        mean_shape_error=shape_error)
    rows = []
    for split, mask in (("all", np.ones(len(samples), dtype=bool)),
                        ("train", dataset_ids < int(data["n_train"])),
                        ("test", dataset_ids >= int(data["n_train"]))):
        for metric, values in (("mean_draw_error", draw_error), ("mean_shape_error", shape_error)):
            rows.append(dict(split=split, metric=metric, **summary_stats(values[mask])))
    with (out / "full_dataset_stats.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def evaluate(args):
    data = np.load(args.out / "preprocessing.npz")
    config = json.loads((args.out / "metadata.json").read_text())
    model, _, _, _ = load_checkpoint(args.out, config["updates"])
    device, cpu = jax.devices()[0], jax.devices("cpu")[0]
    dataset_ids = np.arange(len(data["spectra"]))
    with jax.default_device(cpu):
        noise = np.asarray(make_initial_noise(0, dataset_ids, args.draws)).reshape(-1, 3, 2)
    spectra = np.repeat(data["spectra"], args.draws, axis=0)
    predictions = []
    for offset in range(0, len(noise), args.chunk_size):
        c, x = spectra[offset:offset + args.chunk_size], noise[offset:offset + args.chunk_size]
        valid = len(x)
        if valid < args.chunk_size:
            c = np.pad(c, ((0, args.chunk_size - valid), (0, 0)), mode="edge")
            x = np.pad(x, ((0, args.chunk_size - valid), (0, 0), (0, 0)), mode="edge")
        prediction = sample_batch(model, jax.device_put(c, device), jax.device_put(x, device),
                                  steps=args.steps, method=args.method)
        predictions.append(np.asarray(prediction)[:valid])
    samples = np.concatenate(predictions).reshape(len(dataset_ids), args.draws, 3, 2)
    write_full_dataset(args.out, samples, dataset_ids)
    config["inference"] = dict(checkpoint=config["updates"], method=args.method, steps=args.steps,
                               draws=args.draws, seed=0, split="all")
    (args.out / "metadata.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"Saved full-dataset inference for {len(dataset_ids)} drums", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("triangle_drums.npz"))
    parser.add_argument("--out", type=Path, default=Path("results/aligned_flow_long_20261027"))
    parser.add_argument("--updates", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--features", choices=("log", "log_ratios"), default="log")
    parser.add_argument("--method", choices=("euler", "heun", "rk4"), default="euler")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--draws", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=256)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--train-only", action="store_true")
    mode.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()
    if min(args.updates, args.width, args.batch_size, args.steps, args.draws, args.chunk_size) < 1:
        parser.error("All counts must be positive")
    if not args.eval_only:
        train(args)
    if not args.train_only:
        evaluate(args)


if __name__ == "__main__":
    main()
