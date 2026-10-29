"""Compare direct regression and aligned ring flow matching with 30 or 5 eigenvalues.

Generate mixed_polygon_drums once with collect_mixed_dataset.py, then run:
    python training/regression/spectral_ablation.py --data mixed_polygon_drums.npz
All runs share a stratified 90/10 split, preprocessing, minibatches and model size.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from drums.models.flow_matching import DrumRingTransformerVelocity
from drums.models.transformer import DrumTransformer
from spectral_ablation_metrics import dihedral_orderings, distribution_metrics
from aligned_flow_experiment import sample_batch


def preprocess(path, seed=1729):
    """Fit all scales on train only; class labels are used only for split/reporting."""
    with np.load(path) as data:
        vertices = np.asarray(data["perimeters"], dtype=np.float32)
        spectra = np.asarray(data["spectra"], dtype=np.float32)
        families = (np.asarray(data["shape_label"]) if "shape_label" in data else
                    np.where(data["n_vertices"] == 3, "triangle",
                             np.where(data["is_convex"], "convex_quad", "nonconvex_quad")))
    if vertices.shape[1:] != (4, 2) or spectra.shape[0] != len(vertices):
        raise ValueError("Expected mixed_polygon_drums with (N, 4, 2) perimeters")
    if not np.all(np.isfinite(spectra) & (spectra > 0)) or not np.all(np.isfinite(vertices)):
        raise ValueError("Data must be finite, with positive eigenvalues")
    if np.any(np.diff(spectra, axis=1) < 0):
        raise ValueError("Eigenvalues must be ascending")
    rng = np.random.default_rng(seed)
    train, test = [], []
    for family in np.unique(families):
        ids = rng.permutation(np.flatnonzero(families == family))
        if len(ids) < 2:
            raise ValueError("Each family needs at least two examples for train/test")
        n_test = max(1, int(round(.1 * len(ids))))
        test.extend(ids[:n_test])
        train.extend(ids[n_test:])
    train, test = rng.permutation(train), rng.permutation(test)
    logs = np.log(spectra)
    # Retain scale through lambda_1; the remaining log ratios are scale invariant.
    features = np.concatenate([logs[:, :1], logs[:, 1:] - logs[:, :1]], axis=1)
    mean = features[train].mean(axis=0)
    std = np.maximum(features[train].std(axis=0), 1e-6)
    vertices = vertices - vertices.mean(axis=1, keepdims=True)
    scale = np.linalg.norm(vertices[train], axis=-1).max()
    if scale <= 0:
        raise ValueError("Training shapes must have nonzero size")
    return dict(features=(features - mean) / std, vertices=vertices / scale,
                train_ids=train, test_ids=test, families=families, raw_spectra=spectra,
                feature_mean=mean, feature_std=std, target_scale=scale)


def regression_loss(model, spectra, targets):
    """Relative squared Procrustes loss over eight boundary-preserving orderings.

    In 2D, ||A||_* = sqrt(||A||_F^2 + 2 |det A|), avoiding SVD gradients.
    Rotations/reflections and translations are removed; size is preserved.
    """
    predictions = jax.vmap(model)(spectra)
    predictions -= predictions.mean(axis=1, keepdims=True)
    ordered = predictions[:, dihedral_orderings(4)]
    cross = ordered.swapaxes(-1, -2) @ targets[:, None]
    det = cross[..., 0, 0] * cross[..., 1, 1] - cross[..., 0, 1] * cross[..., 1, 0]
    nuclear = jnp.sqrt(jnp.maximum(jnp.sum(cross**2, axis=(-2, -1)) + 2 * jnp.abs(det), 1e-12))
    norm_target = jnp.sum(targets**2, axis=(-2, -1))
    squared = norm_target[:, None] + jnp.sum(ordered**2, axis=(-2, -1)) - 2 * nuclear
    return jnp.mean(jnp.maximum(jnp.min(squared, axis=1), 0) / norm_target)


def prepare_pairs(vertices, key):
    """Align endpoints to centred Gaussian noise on CPU before interpolating."""
    noise_key, time_key = jax.random.split(key)
    noise = jax.random.normal(noise_key, vertices.shape, dtype=vertices.dtype)
    noise -= noise.mean(axis=1, keepdims=True)
    ordered = vertices[:, dihedral_orderings(4)]
    u, _, vt = jnp.linalg.svd(ordered.swapaxes(-1, -2) @ noise[:, None], full_matrices=False)
    aligned = ordered @ (u @ vt)
    best = jnp.argmin(jnp.sum((aligned - noise[:, None])**2, axis=(-2, -1)), axis=1)
    endpoint = aligned[jnp.arange(len(vertices)), best]
    t = jax.random.uniform(time_key, (len(vertices),), dtype=vertices.dtype)
    return t, (1 - t[:, None, None]) * noise + t[:, None, None] * endpoint, endpoint - noise


def velocity_loss(model, spectra, t, x_t, velocity):
    return jnp.mean((jax.vmap(model)(t, x_t, spectra) - velocity)**2)


def make_model(kind, n_eig, args):
    key = jax.random.PRNGKey(args.seed)
    if kind == "regression":
        return DrumTransformer(4, args.width, n_eig, key, args.layers)
    return DrumRingTransformerVelocity(key, n_vertices=4, n_eig=n_eig,
                                       N_f=args.width, L=args.layers)


def train(model, kind, n_eig, data, args, out):
    warmup = min(500, max(1, args.updates // 10))
    schedule = optax.warmup_cosine_decay_schedule(0., 1e-3, warmup, args.updates, 1e-6)
    optimizer = optax.chain(optax.clip_by_global_norm(1.), optax.adamw(schedule, weight_decay=1e-4))
    state = optimizer.init(eqx.filter(model, eqx.is_array))
    loss_fn = regression_loss if kind == "regression" else velocity_loss

    @eqx.filter_jit
    def step(model, state, *inputs):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model, *inputs)
        updates, state = optimizer.update(grads, state, eqx.filter(model, eqx.is_array))
        return eqx.apply_updates(model, updates), state, loss

    cpu, device = jax.devices("cpu")[0], jax.devices()[0]
    pairs = jax.jit(prepare_pairs, device=cpu)
    train_ids = data["train_ids"]
    features = data["features"][train_ids, :n_eig]
    vertices = data["vertices"][train_ids]
    rng = np.random.default_rng(args.seed)  # same batches for all four runs
    with jax.default_device(cpu):
        key = jax.random.PRNGKey(args.seed)
    start, losses = time.perf_counter(), []
    with (out / "training.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", "loss", "elapsed_seconds"])
        writer.writeheader()
        for update in range(args.updates):
            ids = rng.integers(len(train_ids), size=args.batch_size)
            inputs = (features[ids], vertices[ids])
            if kind == "flow":
                inputs = (inputs[0], *pairs(jax.device_put(inputs[1], cpu),
                                           jax.random.fold_in(key, update)))
            model, state, loss = step(model, state, *(jax.device_put(x, device) for x in inputs))
            value = float(loss)
            if not np.isfinite(value):
                raise FloatingPointError(f"Nonfinite {kind} loss at step {update + 1}")
            losses.append(value)
            if update == 0 or (update + 1) % 1000 == 0 or update + 1 == args.updates:
                row = dict(step=update + 1, loss=float(np.mean(losses)),
                           elapsed_seconds=time.perf_counter() - start)
                writer.writerow(row)
                handle.flush()
                print(f"{kind}/{n_eig}: {row}", flush=True)
                losses.clear()
    eqx.tree_serialise_leaves(out / "model.eqx", model)
    return model


def predict(model, kind, n_eig, data, args):
    """Held-out predictions; FM noise is identical for the 30- and 5-input runs."""
    ids = data["test_ids"]
    features = data["features"][ids, :n_eig]
    draws = args.draws if kind == "flow" else 1
    features = np.repeat(features, draws, axis=0)
    if kind == "flow":
        rng = np.random.default_rng(args.seed + 1)
        noise = rng.standard_normal((len(features), 4, 2)).astype(np.float32)
        noise -= noise.mean(axis=1, keepdims=True)
    regression_batch = eqx.filter_jit(jax.vmap(lambda m, x: m(x), in_axes=(None, 0)))
    predictions = []
    for offset in range(0, len(features), args.chunk_size):
        x = features[offset:offset + args.chunk_size]
        valid = len(x)
        x = np.pad(x, ((0, args.chunk_size - valid), (0, 0)), mode="edge")
        if kind == "regression":
            result = regression_batch(model, jnp.asarray(x))
        else:
            z = noise[offset:offset + valid]
            z = np.pad(z, ((0, args.chunk_size - valid), (0, 0), (0, 0)), mode="edge")
            result = sample_batch(model, jnp.asarray(x), jnp.asarray(z), steps=args.steps, method="euler")
        predictions.append(np.asarray(result)[:valid])
    return np.concatenate(predictions).reshape(len(ids), draws, 4, 2)


def evaluate(samples, kind, n_eig, data, out):
    ids = data["test_ids"]
    metrics = [distribution_metrics(target, draws)
               for target, draws in zip(data["vertices"][ids], samples)]
    values = {name: np.array([row[name] for row in metrics]) for name in metrics[0]}
    np.savez_compressed(out / "test_predictions.npz", test_ids=ids,
                        shape_label=data["families"][ids], samples=samples, **values)
    rows = []
    for family in ("all", *np.unique(data["families"])):
        mask = np.ones(len(ids), dtype=bool) if family == "all" else data["families"][ids] == family
        rows.append(dict(model=kind, n_eig=n_eig, family=family, count=int(mask.sum()),
                         **{name: float(v[mask].mean()) for name, v in values.items()}))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("results/spectral_ablation_20261028"))
    parser.add_argument("--updates", type=int, default=20000)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--n-eig", nargs="+", type=int, default=[30, 5])
    parser.add_argument("--models", nargs="+", choices=["regression", "flow"], default=["regression", "flow"])
    parser.add_argument("--draws", type=int, default=32)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()
    if args.updates < 2 or min(args.width, args.layers, args.batch_size, args.draws,
                               args.steps, args.chunk_size, *args.n_eig) < 1 or args.width % 4:
        parser.error("Use updates >= 2, positive counts, and width divisible by four")
    data = preprocess(args.data, args.seed)
    if max(args.n_eig) > data["features"].shape[1]:
        parser.error("Dataset has fewer eigenvalues than requested")
    args.out.mkdir(parents=True, exist_ok=True)
    config = dict(data=str(args.data.resolve()), data_sha256=hashlib.sha256(args.data.read_bytes()).hexdigest(),
                  seed=args.seed, updates=args.updates, width=args.width, layers=args.layers,
                  batch_size=args.batch_size, draws=args.draws, steps=args.steps,
                  features="train-standardized log(lambda_1), log(lambda_k/lambda_1)",
                  split="stratified 90/10 by triangle/convex quad/nonconvex quad",
                  learning_rate="warmup to 1e-3 then cosine to 1e-6", integrator="Euler",
                  n_train=len(data["train_ids"]), n_test=len(data["test_ids"]))
    metadata = args.out / "metadata.json"
    if metadata.exists() and json.loads(metadata.read_text()) != config:
        raise ValueError("Output configuration differs; choose a new --out directory")
    metadata.write_text(json.dumps(config, indent=2) + "\n")
    np.savez_compressed(args.out / "preprocessing.npz", **data)
    print(f"Device: {jax.devices()[0]}; train={config['n_train']}, test={config['n_test']}", flush=True)
    summary = args.out / "summary.csv"
    rows = list(csv.DictReader(summary.open())) if summary.exists() else []
    for kind in args.models:
        for n_eig in args.n_eig:
            out = args.out / f"{kind}_{n_eig}"
            out.mkdir(exist_ok=True)
            model = make_model(kind, n_eig, args)
            if args.eval_only or (out / "model.eqx").exists():
                model = eqx.tree_deserialise_leaves(out / "model.eqx", model)
            else:
                model = train(model, kind, n_eig, data, args, out)
            run_config = dict(model=kind, n_eig=n_eig,
                              parameter_count=sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array))),
                              jax=jax.__version__, equinox=eqx.__version__, optax=optax.__version__)
            (out / "metadata.json").write_text(json.dumps(run_config, indent=2) + "\n")
            rows = [r for r in rows if (r["model"], int(r["n_eig"])) != (kind, n_eig)]
            rows.extend(evaluate(predict(model, kind, n_eig, data, args), kind, n_eig, data, out))
            with summary.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            print(rows[-4:], flush=True)
            jax.clear_caches()


if __name__ == "__main__":
    main()
