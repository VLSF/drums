"""Evaluate a saved conditional DDPM on held-out triangles.

    python training/diffusion/triangle.py --checkpoint results/diffusion_triangle \
        --data triangle_drums.npz --out results/diffusion_triangle/evaluation

Alignment/error conventions match training/regression/aligned_flow_experiment.py.
Only generated shapes enter the Procrustes mean; targets enter evaluation only.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from drums.models.diffusion import load_checkpoint, prepare_data, sample, sample_ddim


ORDERINGS = np.array([[0, 1, 2], [2, 1, 0], [2, 0, 1],
                      [1, 0, 2], [1, 2, 0], [0, 2, 1]])


def aligned_candidates(reference, prediction):
    """All six orderings, each optimally rotated/reflected without rescaling."""
    ordered = np.asarray(prediction)[..., ORDERINGS, :]
    u, _, vt = np.linalg.svd(ordered.swapaxes(-1, -2) @ reference, full_matrices=False)
    return ordered @ (u @ vt)


def relative_error(target, prediction):
    target = np.asarray(target, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    if not np.all(np.isfinite(prediction)):
        return float("inf")
    target = target - target.mean(axis=0, keepdims=True)
    prediction = prediction - prediction.mean(axis=0, keepdims=True)
    candidates = aligned_candidates(target, prediction)
    return float(np.min(np.mean(
        np.linalg.norm(target - candidates, axis=-1)
        / np.linalg.norm(target, axis=-1), axis=-1,
    )))


def mean_shape(samples, n_iterations=3):
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
               *np.percentile(finite, [5, 95]), finite.std()] if finite.size
              else [None] * len(names))
    return dict(count=int(errors.size), nonfinite_count=int(errors.size - finite.size),
                **{name: None if value is None else float(value)
                   for name, value in zip(names, values)})


@eqx.filter_jit
def sample_batch(model, spectra, keys, schedule, sampler="ddpm", sampling_steps=200, eta=0.0,
                 guidance_scale=1., condition_present=1.):
    if sampler == "ddpm":
        return jax.vmap(lambda spectrum, key: sample(
            model, spectrum, key, schedule, guidance_scale=guidance_scale, condition_present=condition_present,
        ))(spectra, keys)
    return jax.vmap(lambda spectrum, key: sample_ddim(
        model, spectrum, key, schedule, n_steps=sampling_steps, eta=eta,
        guidance_scale=guidance_scale, condition_present=condition_present,
    ))(spectra, keys)


def evaluate(args):
    started = time.perf_counter()
    config = json.loads((args.checkpoint / "metadata.json").read_text())
    guidance_scale = getattr(args, "guidance_scale", 1.)
    condition_present = getattr(args, "condition_present", 1)
    presence_enabled = config.get("condition_presence", False)
    if not np.isfinite(guidance_scale) or condition_present not in (0, 1):
        raise ValueError("Guidance scale must be finite and condition_present must be 0 or 1")
    if not presence_enabled and (guidance_scale != 1. or condition_present != 1):
        raise ValueError("Guidance and unconditional generation require a presence-enabled checkpoint")
    dropout = config.get("cond_drop_prob", 0.)
    if not np.isfinite(dropout) or not 0 <= dropout <= 1:
        raise ValueError("Checkpoint cond_drop_prob must be in [0, 1]")
    if (condition_present == 0 or guidance_scale != 1.) and dropout <= 0:
        raise ValueError("Unconditional sampling and CFG require a null branch trained with cond_drop_prob > 0")
    if condition_present == 1 and guidance_scale != 0. and dropout >= 1:
        raise ValueError("Conditional sampling and CFG require cond_drop_prob < 1")
    model, schedule, preprocessing = load_checkpoint(args.checkpoint)
    spectra, vertices, fitted = prepare_data(
        args.data, train_fraction=config["train_fraction"], features=config["features"],
        coordinate_scale=config.get("coordinate_scale", "max"),
    )
    if model.n_vertices != 3 or vertices.shape[1:] != (3, 2):
        raise ValueError("This diagnostic expects three-vertex drums")
    for name in ("spectrum_mean", "spectrum_std", "target_scale", "n_train"):
        if not np.allclose(fitted[name], preprocessing[name], rtol=1e-6, atol=1e-6):
            raise ValueError(f"Dataset preprocessing differs from checkpoint: {name}")
    n_train = int(preprocessing["n_train"])
    ids = np.arange(n_train, min(len(vertices), n_train + args.n_eval))
    if not len(ids):
        raise ValueError("No held-out samples available")
    features = np.repeat(np.asarray(spectra)[ids], args.n_samples, axis=0)
    targets = np.asarray(vertices)[ids]
    device, cpu = jax.devices()[0], jax.devices("cpu")[0]

    # Keys are independent of chunk boundaries and stable for dataset/draw IDs.
    with jax.default_device(cpu):
        base_key = jr.PRNGKey(args.seed)
        dataset_keys = jax.vmap(lambda i: jr.fold_in(base_key, i))(
            jnp.asarray(ids, dtype=jnp.uint32),
        )
        keys = jax.vmap(lambda key: jax.vmap(lambda i: jr.fold_in(key, i))(
            jnp.arange(args.n_samples, dtype=jnp.uint32),
        ))(dataset_keys)
        keys = np.asarray(keys).reshape(-1, 2)

    sampling_started = time.perf_counter()
    predictions = []
    for offset in range(0, len(features), args.chunk_size):
        c = features[offset:offset + args.chunk_size]
        k = keys[offset:offset + args.chunk_size]
        valid = len(c)
        c = np.pad(c, ((0, args.chunk_size - valid), (0, 0)), mode="edge")
        k = np.pad(k, ((0, args.chunk_size - valid), (0, 0)), mode="edge")
        draws = sample_batch(model, jax.device_put(c, device),
                             jax.device_put(k, device), schedule,
                             args.sampler, args.sampling_steps, args.eta,
                             guidance_scale, condition_present)
        predictions.append(np.asarray(draws)[:valid])
        print(f"Generated {min(offset + valid, len(features))}/{len(features)} draws", flush=True)
    samples = np.concatenate(predictions).reshape(len(ids), args.n_samples, 3, 2)
    sampling_seconds = time.perf_counter() - sampling_started

    # NumPy SVD keeps geometry evaluation on CPU, including when sampling on Metal.
    errors = np.array([[relative_error(target, draw) for draw in draws]
                       for target, draws in zip(targets, samples)])
    means = np.array([mean_shape(draws) for draws in samples])
    shape_errors = np.array([relative_error(target, mean)
                             for target, mean in zip(targets, means)])
    sampling_steps = schedule.n_steps if args.sampler == "ddpm" else args.sampling_steps
    calls_per_step = 1 if condition_present == 0 or guidance_scale in (0., 1.) else 2
    nfe = sampling_steps * calls_per_step
    unconditional = condition_present == 0 or guidance_scale == 0.
    suffix = "_g" + str(float(guidance_scale)).removesuffix(".0") if presence_enabled else ""
    if presence_enabled and condition_present == 0:
        suffix += "_unconditional"
    summary_name, samples_name = f"summary{suffix}.json", f"samples{suffix}.npz"
    root = Path(__file__).resolve().parents[2]
    source_hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in (
        "src/drums/models/diffusion.py", "src/drums/models/ring_transformer.py", "training/diffusion/triangle.py",
    )}
    summary = dict(
        args={name: str(value) if isinstance(value, Path) else value
              for name, value in vars(args).items()},
        checkpoint_training=config, split="held_out_validation",
        n_eval=int(len(ids)), n_samples=args.n_samples, sampler=args.sampler.upper(),
        sampler_steps=sampling_steps,
        eta=args.eta if args.sampler == "ddim" else None, device=str(device),
        guidance_scale=guidance_scale, condition_present=condition_present,
        predictor_condition_masks=([0] if unconditional else [1] if guidance_scale == 1. else [0, 1]),
        condition_presence=presence_enabled, training_cond_drop_prob=config.get("cond_drop_prob", 0.),
        effective_conditioning="unconditional" if unconditional else "conditional",
        metrics_interpretation=("Unconditional prior draws paired with targets for diagnostics; not conditional reconstruction performance"
                                if unconditional else "Conditional reconstruction metrics"),
        predictor_evaluations_per_draw=nfe,
        requested_predictor_evaluations=len(features) * nfe,
        executed_predictor_evaluations=((len(features) + args.chunk_size - 1) // args.chunk_size) * args.chunk_size * nfe,
        source_sha256=source_hashes,
        output_files=dict(summary=summary_name, samples=samples_name),
        per_draw_error=summary_stats(errors),
        per_shape_mean_draw_error=summary_stats(errors.mean(axis=1)),
        mean_shape_error=summary_stats(shape_errors),
        historical_FM_log_ratios_reference=dict(
            mean_draw_error=0.01527, mean_shape_error=0.00873,
            updates=20000, width=256, draws=32, sampler="Euler", steps=50,
            note="Historical run with different training budget; not a matched comparison",
        ),
        sampling_seconds=sampling_seconds, elapsed_seconds=time.perf_counter() - started,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out / samples_name, dataset_ids=ids, samples=samples,
                        targets=targets, mean_shapes=means, draw_errors=errors,
                        mean_shape_error=shape_errors, target_scale=preprocessing["target_scale"],
                        guidance_scale=guidance_scale, condition_present=condition_present,
                        condition_presence=presence_enabled, training_cond_drop_prob=config.get("cond_drop_prob", 0.))
    (args.out / summary_name).write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("triangle_drums.npz"))
    parser.add_argument("--n-eval", type=int, default=100)
    parser.add_argument("--n-samples", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=443)
    parser.add_argument("--sampler", choices=("ddpm", "ddim"), default="ddpm")
    parser.add_argument("--sampling-steps", type=int, default=200)
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--guidance-scale", type=float, default=1.)
    parser.add_argument("--condition-present", type=int, choices=(0, 1), default=1)
    args = parser.parse_args(argv)
    if min(args.n_eval, args.n_samples, args.chunk_size) < 1:
        parser.error("Evaluation counts must be positive")
    if args.sampling_steps < 1 or not 0 <= args.eta <= 1:
        parser.error("Sampling steps must be positive and eta must be in [0, 1]")
    return evaluate(args)


if __name__ == "__main__":
    main()
