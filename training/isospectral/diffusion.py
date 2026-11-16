"""Conditional DDPM experiments with the same Conway splits and metrics as fm.py.

Examples (from the repository root):
    python training/isospectral/diffusion.py --data training/isospectral/isospectral_30k.npz --out results/ddpm_main_ratio --features ratio
    python training/isospectral/diffusion.py --out results/ddpm_main_ratio --eval-only
    python training/isospectral/diffusion.py --out results/ddpm_main_ratio --resume --train-only

Training and validation never use test labels. The historical asymmetric "energy
score" is reproduced for comparability and labeled as a diagnostic, not a proper
scoring rule. Checkpoints include optimizer state and the random key.
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import random, vmap

from common import Features, N_TRAIN, N_VAL, N_TEST, N_V, procrustes_mean, relative_error
from metrics import summarize
from drums.models.diffusion import (
    DiffusionSchedule, DrumRingTransformerNoise, center, diffusion_loss,
    sample as sample_one, sample_ddim, reverse_step,
)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_guidance(config, guidance_scale=1., condition_present=1.):
    if not np.isfinite(guidance_scale) or condition_present not in (0, 1):
        raise ValueError("Guidance scale must be finite and condition_present must be 0 or 1")
    if not config.get("condition_presence", False) and (guidance_scale != 1. or condition_present != 1):
        raise ValueError("Guidance and unconditional generation require a presence-enabled checkpoint")
    dropout = config.get("cond_drop_prob", 0.)
    if not np.isfinite(dropout) or not 0 <= dropout <= 1:
        raise ValueError("Checkpoint cond_drop_prob must be in [0, 1]")
    if (condition_present == 0 or guidance_scale != 1.) and dropout <= 0:
        raise ValueError("Unconditional sampling and CFG require a null branch trained with cond_drop_prob > 0")
    if condition_present == 1 and guidance_scale != 0. and dropout >= 1:
        raise ValueError("Conditional sampling and CFG require cond_drop_prob < 1")


def evaluation_prefix(config, evaluation):
    prefix = f"{evaluation['split']}_{evaluation['sampler']}{evaluation['sampling_steps']}_{evaluation['weights']}"
    if evaluation["sampler"] == "ddim":
        prefix += f"_eta{evaluation['eta']:g}"
    if evaluation.get("sampling_block", 0):
        prefix += f"_block{evaluation['sampling_block']}"
    if config.get("condition_presence", False):
        prefix += "_g" + str(float(evaluation.get("guidance_scale", 1.))).removesuffix(".0")
        if evaluation.get("condition_present", 1) == 0:
            prefix += "_unconditional"
    return prefix


def transform_features(log_spectra, kind, preprocessing):
    """Apply already-fitted common.Features without refitting on evaluation data."""
    base, _, suffix = kind.partition("_k")
    values = log_spectra[:, :int(suffix)] if suffix else log_spectra
    if base != "naive":
        values = jnp.concatenate([values[:, 1:] - values[:, :1], values[:, :1]], axis=1)
    values = (values - preprocessing["feature_mean"]) / preprocessing["feature_std"]
    if base == "ratio_white":
        values = values @ preprocessing["feature_projection"]
    return values


def prepare_data(path, config, preprocessing=None):
    """Split source rows before adding twins; use one scale fitted on train only."""
    with np.load(path) as source:
        vertices = np.asarray(source["perimeters"], dtype=np.float32)
        twins = np.asarray(source["twin_perimeters"], dtype=np.float32)
        spectra = np.asarray(source["spectra"], dtype=np.float32)
    total = config["n_train"] + config["n_val"] + config["n_test"]
    if (vertices.ndim != 3 or vertices.shape[1:] != (N_V, 2)
            or twins.shape != vertices.shape or spectra.ndim != 2
            or len(vertices) != len(spectra) or len(vertices) < total):
        raise ValueError(f"Expected at least {total} Conway rows with nine vertices and twins")
    if (not np.isfinite(vertices).all() or not np.isfinite(twins).all()
            or not (np.isfinite(spectra) & (spectra > 0)).all()):
        raise ValueError("Coordinates must be finite and spectra finite and positive")
    # Fit PCA on CPU as in the reference experiments, including on Metal runs.
    cpu = jax.devices("cpu")[0]
    with jax.default_device(cpu):
        vertices = center(jnp.asarray(vertices[:total]))
        twins = center(jnp.asarray(twins[:total]))
        logs = jnp.log(jnp.asarray(spectra[:total]))
        n_train = config["n_train"]
        if preprocessing is None:
            scale_kind = config.get("coordinate_scale", "max")
            if scale_kind == "max":
                scale = jnp.max(jnp.linalg.norm(vertices[:n_train], axis=-1))
            elif scale_kind == "rms":
                squared_norm = jnp.sum(vertices[:n_train] ** 2, axis=(-2, -1))
                scale = jnp.sqrt(jnp.mean(squared_norm) / (2 * (N_V - 1)))
            else:
                raise ValueError("coordinate_scale must be max or rms")
            if not float(scale) > 0:
                raise ValueError("Training shapes must have positive size")
            fitted = Features(config["features"], logs[:n_train])
            preprocessing = dict(
                target_scale=np.asarray(scale),
                feature_mean=np.asarray(fitted.mu),
                feature_std=np.asarray(fitted.sigma),
            )
            if fitted.base == "ratio_white":
                preprocessing["feature_projection"] = np.asarray(fitted.proj)
        features = transform_features(logs, config["features"], preprocessing)
        vertices = vertices / preprocessing["target_scale"]
        twins = twins / preprocessing["target_scale"]
        if "data_std" not in preprocessing:
            preprocessing = dict(preprocessing)
            preprocessing["data_std"] = np.asarray(jnp.sqrt(
                jnp.mean(jnp.sum(vertices[:n_train] ** 2, axis=(-2, -1))) / (2 * (N_V - 1)),
            ))
        indices = (0, n_train, n_train + config["n_val"], total)
        result = {}
        for name, first, last in zip(("train", "validation", "test"), indices[:-1], indices[1:]):
            result[name] = tuple(np.asarray(x[first:last]) for x in (features, vertices, twins))
    return result, preprocessing


def make_model(config):
    return DrumRingTransformerNoise(
        random.PRNGKey(config["seed"]), n_vertices=N_V,
        n_eig=config["n_eig"], N_f=config["N_f"], L=config["L"],
        n_heads=config["n_heads"], time_features=config.get("time_features", 0),
        gaussian_skip=config.get("gaussian_skip", False), data_std=config.get("data_std", 1.),
        condition_presence=config.get("condition_presence", False),
    )


def make_schedule(config):
    return DiffusionSchedule(
        n_steps=config["n_steps"], beta_start=config["beta_start"], beta_end=config["beta_end"],
    )


def make_optimizer(config):
    rate = config["lr"]
    if config["n_updates"] > 1:
        rate = optax.warmup_cosine_decay_schedule(
            0., rate, min(500, max(1, config["n_updates"] // 10)), config["n_updates"], 1e-6,
        )
    return optax.chain(optax.clip_by_global_norm(1.), optax.adamw(rate, weight_decay=1e-4))


def checkpoint_template(config):
    model = make_model(config)
    state = make_optimizer(config).init(eqx.filter(model, eqx.is_array))
    return (model, model, state, random.fold_in(random.PRNGKey(config["seed"]), 1),
            jnp.asarray(0, dtype=jnp.int32), np.full(config["n_updates"], np.nan, dtype=np.float32))


def save_checkpoint(folder, model, ema, state, key, completed, history):
    temporary = folder / "checkpoint.tmp.eqx"
    eqx.tree_serialise_leaves(
        temporary, (model, ema, state, key, jnp.asarray(completed, dtype=jnp.int32), history),
    )
    temporary.replace(folder / "checkpoint.eqx")
    # Separate model weights are convenient for plotting without optimizer construction.
    eqx.tree_serialise_leaves(folder / "model.eqx", model)
    eqx.tree_serialise_leaves(folder / "ema.eqx", ema)


def load_checkpoint(folder, config):
    return eqx.tree_deserialise_leaves(folder / "checkpoint.eqx", checkpoint_template(config))


def train(folder, config, arrays, validation, resume=False, stop_after=None):
    device = jax.devices()[0]
    optimizer = make_optimizer(config)
    schedule = make_schedule(config)
    carry = load_checkpoint(folder, config) if resume else checkpoint_template(config)
    model, ema, state, key, completed, history = carry
    completed = int(completed)
    target = config["n_updates"] if stop_after is None else min(config["n_updates"], stop_after)
    if target < completed:
        raise ValueError("Checkpoint already exceeds --stop-after; training cannot roll back")
    history = np.asarray(history).copy()
    F, V, W = arrays
    if config["paired"]:
        F, V = np.concatenate([F, F]), np.concatenate([V, W])
    F, V = jax.device_put(F, device), jax.device_put(V, device)
    F_val, V_val, _ = validation
    F_val, V_val = jax.device_put(F_val, device), jax.device_put(V_val, device)

    @eqx.filter_jit
    def update_chunk(model, ema, state, key, F, V, count):
        def update(carry, _):
            model, ema, state, key = carry
            key, batch_key, loss_key = random.split(key, 3)
            indices = random.randint(batch_key, (config["batch"],), 0, len(F))
            loss, grads = eqx.filter_value_and_grad(diffusion_loss)(
                model, F[indices], V[indices], loss_key, schedule,
                cond_drop_prob=config.get("cond_drop_prob", 0.),
            )
            updates, state = optimizer.update(grads, state, eqx.filter(model, eqx.is_array))
            model = eqx.apply_updates(model, updates)
            if config["ema_decay"]:
                decay = config["ema_decay"]
                ema = jax.tree_util.tree_map(lambda old, new: decay*old + (1-decay)*new, ema, model)
            else:
                ema = model
            return (model, ema, state, key), loss
        return jax.lax.scan(update, (model, ema, state, key), None, length=count)

    val_loss = eqx.filter_jit(diffusion_loss)
    timings_file = folder / "training.json"
    timing = json.loads(timings_file.read_text()) if resume and timings_file.exists() else {}
    previous_seconds = timing.get("elapsed_seconds", 0.)
    validation_history = timing.get("validation", [])
    started = time.perf_counter()
    while completed < target:
        boundary = min(config["checkpoint_every"] - completed % config["checkpoint_every"],
                       config["eval_every"] - completed % config["eval_every"])
        count = min(config["train_chunk"], target - completed, boundary)
        (model, ema, state, key), losses = update_chunk(model, ema, state, key, F, V, count)
        losses = np.asarray(losses)
        if not np.isfinite(losses).all():
            raise FloatingPointError(f"Nonfinite training loss after update {completed}")
        history[completed:completed + count] = losses
        completed += count
        if completed % 1000 == 0 and completed % config["eval_every"] != 0:
            print(json.dumps(dict(update=completed,
                                  train_loss=float(history[max(0, completed-500):completed].mean()),
                                  elapsed_seconds=previous_seconds + time.perf_counter() - started)), flush=True)
        if completed % config["eval_every"] == 0 or completed == target:
            value = float(val_loss(model, F_val, V_val, random.PRNGKey(123), schedule))
            ema_value = (float(val_loss(ema, F_val, V_val, random.PRNGKey(123), schedule))
                         if config["ema_decay"] else value)
            if not np.isfinite(value) or not np.isfinite(ema_value):
                raise FloatingPointError(f"Nonfinite validation loss at update {completed}")
            elapsed = previous_seconds + time.perf_counter() - started
            validation = dict(update=completed, loss=value, ema_loss=ema_value, elapsed_seconds=elapsed)
            if config.get("condition_presence", False):
                unconditional = float(val_loss(
                    model, F_val, V_val, random.PRNGKey(123), schedule, condition_present=0.,
                ))
                unconditional_ema = (float(val_loss(
                    ema, F_val, V_val, random.PRNGKey(123), schedule, condition_present=0.,
                )) if config["ema_decay"] else unconditional)
                if not np.isfinite(unconditional) or not np.isfinite(unconditional_ema):
                    raise FloatingPointError(f"Nonfinite unconditional validation loss at update {completed}")
                validation.update(conditional_loss=value, conditional_ema_loss=ema_value,
                                  unconditional_loss=unconditional, unconditional_ema_loss=unconditional_ema)
            validation_history.append(validation)
            print(json.dumps(dict(validation, train_loss=float(history[max(0, completed-500):completed].mean()),
                                  validation_loss=value, validation_loss_ema=ema_value)), flush=True)
        if completed % config["checkpoint_every"] == 0 or completed == target:
            save_checkpoint(folder, model, ema, state, key, completed, history)
            timing = dict(completed_updates=completed, n_train_pairs=len(F),
                          elapsed_seconds=previous_seconds + time.perf_counter() - started,
                          validation=validation_history)
            timings_file.write_text(json.dumps(timing, indent=2) + "\n")
            diagnostics = dict(validation_loss=[x["loss"] for x in validation_history],
                               validation_loss_ema=[x["ema_loss"] for x in validation_history])
            if config.get("condition_presence", False):
                diagnostics.update({f"validation_{name}": [x[name] for x in validation_history]
                                    for name in ("conditional_loss", "conditional_ema_loss",
                                                 "unconditional_loss", "unconditional_ema_loss")})
            np.savez(folder / "history.npz", train_loss=history[:completed],
                     validation_steps=[x["update"] for x in validation_history], **diagnostics)
    timing["completed_updates"] = completed
    return model, ema, history, timing


@eqx.filter_jit
def sample_batch(model, features, keys, schedule, sampler="ddpm", sampling_steps=200, eta=0.,
                 guidance_scale=1., condition_present=1.):
    """One independent DDPM draw per row, using the core sampler unchanged."""
    if sampler == "ddpm":
        return vmap(lambda feature, key: sample_one(
            model, feature, key, schedule, guidance_scale=guidance_scale, condition_present=condition_present,
        ))(features, keys)
    if sampler == "ddim":
        return vmap(lambda feature, key: sample_ddim(
            model, feature, key, schedule, n_steps=sampling_steps, eta=eta,
            guidance_scale=guidance_scale, condition_present=condition_present,
        ))(features, keys)
    raise ValueError("sampler must be ddpm or ddim")


@eqx.filter_jit
def initialize_ddpm_batch(model, features, keys):
    """Use the same per-draw initial Gaussian and loop key as core.sample."""
    def one(key):
        init_key, loop_key = random.split(key)
        x = center(random.normal(init_key, (model.n_vertices, 2), dtype=features.dtype))
        return x, loop_key
    return vmap(one)(keys)


@eqx.filter_jit
def ddpm_block(model, x, features, keys, schedule, offset, count, guidance_scale=1., condition_present=1.):
    """Execute consecutive DDPM steps; offset is a runtime JAX scalar."""
    def body(i, carry):
        x, keys = carry
        split_keys = vmap(random.split)(keys)
        loop_keys, step_keys = split_keys[:, 0], split_keys[:, 1]
        step = schedule.n_steps - (offset + i)
        previous = vmap(lambda vertices, feature, key: reverse_step(
            model, vertices, feature, step, schedule, key,
            guidance_scale=guidance_scale, condition_present=condition_present,
        ))(x, features, step_keys)
        return previous, loop_keys
    return jax.lax.fori_loop(0, count, body, (x, keys))


def sample_batch_blocked(model, features, keys, schedule, block_size=200,
                         guidance_scale=1., condition_present=1.):
    """Run every DDPM step in bounded execution blocks without changing its chain."""
    if block_size < 1:
        raise ValueError("DDPM sampling block size must be positive")
    x, loop_keys = initialize_ddpm_batch(model, features, keys)
    for offset in range(0, schedule.n_steps, block_size):
        count = min(block_size, schedule.n_steps - offset)
        # The dynamic offset lets all equal-sized blocks reuse one executable.
        offset_array = jnp.asarray(offset, dtype=jnp.int32)
        x, loop_keys = ddpm_block(model, x, features, loop_keys, schedule, offset_array, count,
                                  guidance_scale, condition_present)
        jax.block_until_ready((x, loop_keys))
    return x


def generate_samples(model, features, key, schedule, n_samples=16, chunk_size=4,
                     sampler="ddpm", sampling_steps=200, eta=0., sampling_block=0,
                     guidance_scale=1., condition_present=1.):
    """Sample chunks of spectra, with flat batch size chunk_size * n_samples.

    Four spectra (64 draws by default) is a conservative Metal setting. Keys
    depend on the global draw index, so changing chunk size preserves the draws.
    Positive sampling_block executes DDPM in synchronized blocks of that size.
    """
    if sampling_block < 0 or (sampling_block and sampler != "ddpm"):
        raise ValueError("sampling_block must be nonnegative and is supported only for DDPM")
    n = len(features)
    result = np.empty((n, n_samples, N_V, 2), dtype=np.float32)
    device = jax.devices()[0]
    started = time.perf_counter()
    for offset in range(0, n, chunk_size):
        block = features[offset:offset + chunk_size]
        valid = len(block)
        block = np.pad(block, ((0, chunk_size - valid), (0, 0)), mode="edge")
        repeated = np.repeat(block, n_samples, axis=0)
        indices = jnp.arange(offset * n_samples, (offset + chunk_size) * n_samples, dtype=jnp.uint32)
        keys = vmap(lambda index: random.fold_in(key, index))(indices)
        repeated, keys = jax.device_put(repeated, device), jax.device_put(keys, device)
        if sampling_block:
            draws = np.asarray(sample_batch_blocked(model, repeated, keys, schedule, sampling_block,
                                                    guidance_scale, condition_present))
        else:
            draws = np.asarray(sample_batch(model, repeated, keys, schedule, sampler, sampling_steps, eta,
                                            guidance_scale, condition_present))
        draws = draws.reshape(chunk_size, n_samples, N_V, 2)[:valid]
        if not np.isfinite(draws).all():
            raise FloatingPointError(f"Nonfinite generated coordinates at evaluation row {offset}")
        result[offset:offset + valid] = draws
        if offset == 0 or offset + chunk_size >= n or (offset // chunk_size + 1) % 10 == 0:
            print(json.dumps(dict(sampled_spectra=min(offset + valid, n), total=n,
                                  elapsed_seconds=time.perf_counter() - started)), flush=True)
    return result


@eqx.filter_jit
def score_chunk(samples, truth, twin):
    """The reference metrics, avoiding repeated computation of each twin midpoint."""
    distance_draws = vmap(lambda target, draws: vmap(lambda draw: relative_error(target, draw))(draws))
    sep = vmap(relative_error)(truth, twin)
    midpoint = vmap(lambda a, b: procrustes_mean(jnp.stack([a, b]), n_iter=3))(truth, twin)
    e_truth = distance_draws(truth, samples)
    per_draw = dict(e_truth=e_truth, e_twin=distance_draws(twin, samples),
                    sep=jnp.broadcast_to(sep[:, None], e_truth.shape),
                    e_mid=distance_draws(midpoint, samples))
    mean = vmap(procrustes_mean)(samples)
    mean_metrics = dict(e_truth=vmap(relative_error)(truth, mean),
                        e_twin=vmap(relative_error)(twin, mean), sep=sep,
                        e_mid=vmap(relative_error)(midpoint, mean))
    d_xy = vmap(lambda draws: vmap(lambda a: vmap(lambda b: relative_error(a, b))(draws))(draws))(samples)
    m = samples.shape[1]
    legacy_energy = e_truth.mean(axis=1) - 0.5 * d_xy.sum(axis=(1, 2)) / (m * (m - 1))
    return per_draw, mean_metrics, legacy_energy


def score_samples(samples, truth, twin, sep_min=0.1, chunk_size=16):
    if len(samples) != len(truth) or truth.shape != twin.shape or samples.shape[1] < 2:
        raise ValueError("Expected matching arrays and at least two draws per spectrum")
    cpu = jax.devices("cpu")[0]
    per_draw, mean_metrics, energies = {}, {}, []
    with jax.default_device(cpu):
        for offset in range(0, len(samples), chunk_size):
            s, v, w = (np.asarray(x[offset:offset + chunk_size]) for x in (samples, truth, twin))
            valid = len(s)
            pad = lambda x: np.pad(x, ((0, chunk_size-valid), *((0, 0),) * (x.ndim-1)), mode="edge")
            draw, mean, energy = score_chunk(*(jax.device_put(pad(x), cpu) for x in (s, v, w)))
            for target, values in ((per_draw, draw), (mean_metrics, mean)):
                for name, value in values.items():
                    target.setdefault(name, []).append(np.asarray(value)[:valid])
            energies.append(np.asarray(energy)[:valid])
    per_draw = {name: np.concatenate(values) for name, values in per_draw.items()}
    mean_metrics = {name: np.concatenate(values) for name, values in mean_metrics.items()}
    energy = np.concatenate(energies)
    n, m = samples.shape[:2]
    selected = per_draw["sep"][:, 0] > sep_min
    sharp = np.minimum(per_draw["e_truth"], per_draw["e_twin"]) < 0.25 * per_draw["sep"]
    nearer = per_draw["e_truth"] < per_draw["e_twin"]
    nearer_twin = per_draw["e_twin"] < per_draw["e_truth"]
    both = (sharp & nearer).any(1) & (sharp & nearer_twin).any(1)
    flat = {name: values.reshape(n*m) for name, values in per_draw.items()}
    result = dict(
        draws=summarize(flat, sep_min),
        mode_balance=dict(mean_fraction_nearer_truth=float(nearer[selected].mean()),
                          std_over_drums=float(nearer.mean(1)[selected].std()),
                          std_if_binomial=float(np.sqrt(0.25/m)), both_modes=float(both[selected].mean())),
        mean_of_draws=summarize(mean_metrics, sep_min),
        energy_score=float(energy.mean()), mean_error_draws=float(per_draw["e_truth"].mean()),
        energy_score_interpretation="Legacy asymmetric-distance diagnostic; not established as a proper scoring rule",
        n_spectra=n, n_samples=m, sep_min=sep_min,
    )
    return result, flat, mean_metrics


TRAINING_KEYS = (
    "features", "paired", "n_updates", "batch", "N_f", "L", "n_heads", "lr", "seed",
    "n_steps", "beta_start", "beta_end", "n_train", "n_val", "n_test",
    "ema_decay",
    "time_features",
    "coordinate_scale",
    "gaussian_skip",
    "condition_presence", "cond_drop_prob",
)
EVALUATION_KEYS = ("weights", "n_samples", "n_eval", "chunk_size", "metric_chunk", "sep_min",
                   "eval_seed", "eval_split", "sampler", "sampling_steps", "eta", "sampling_block",
                   "guidance_scale", "condition_present")


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path)
    p.add_argument("--out", type=Path, required=True, help="A new run directory")
    p.add_argument("--features", default="ratio")
    p.add_argument("--coordinate-scale", choices=("max", "rms"), default="max",
                   help="Train-derived coordinate scale: maximum radius or centered-subspace RMS")
    p.add_argument("--paired", action="store_true")
    p.add_argument("--n-updates", type=int, default=40000)
    p.add_argument("--stop-after", type=int,
                   help="Stop at this absolute update without changing the frozen training budget or LR schedule")
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--N-f", type=int, default=256)
    p.add_argument("--L", type=int, default=3)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--time-features", type=int, default=0,
                   help="Even sinusoidal time-embedding dimension; 0 preserves scalar conditioning")
    p.add_argument("--gaussian-skip", action="store_true",
                   help="Use the optional analytic Gaussian skip in the noise predictor")
    p.add_argument("--condition-presence", action="store_true",
                   help="Add the explicit condition-presence bit needed for CFG training")
    p.add_argument("--cond-drop-prob", type=float, default=0.,
                   help="Independent per-example conditioning dropout during training")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--ema-decay", type=float, default=0., help="0 disables EMA; otherwise use a decay in (0, 1)")
    p.add_argument("--weights", choices=("raw", "ema"), default="raw")
    p.add_argument("--sampler", choices=("ddpm", "ddim"), default="ddpm")
    p.add_argument("--sampling-steps", type=int, default=200, help="DDIM steps; DDPM always uses all training steps")
    p.add_argument("--sampling-block", type=int, default=0,
                   help="DDPM execution block size with synchronization between blocks; 0 uses the original sampler")
    p.add_argument("--eta", type=float, default=0., help="DDIM stochasticity")
    p.add_argument("--guidance-scale", type=float, default=1., help="0: unconditional; 1: conditional; other values: CFG")
    p.add_argument("--condition-present", type=int, choices=(0, 1), default=1,
                   help="0 ignores the spectrum and samples the unconditional learned branch")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-steps", type=int, default=1000)
    p.add_argument("--beta-start", type=float, default=1e-4)
    p.add_argument("--beta-end", type=float, default=0.02)
    p.add_argument("--n-train", type=int, default=N_TRAIN)
    p.add_argument("--n-val", type=int, default=N_VAL)
    p.add_argument("--n-test", type=int, default=N_TEST)
    p.add_argument("--eval-every", type=int, default=5000)
    p.add_argument("--checkpoint-every", type=int, default=5000)
    p.add_argument("--train-chunk", type=int, default=100)
    p.add_argument("--n-samples", type=int, default=16)
    p.add_argument("--n-eval", type=int, default=1500)
    p.add_argument("--chunk-size", type=int, default=4,
                   help="Spectra per chunk; the flat draw batch is chunk-size * n-samples (default 64)")
    p.add_argument("--metric-chunk", type=int, default=16)
    p.add_argument("--sep-min", type=float, default=0.1)
    p.add_argument("--eval-seed", type=int, default=443)
    p.add_argument("--eval-split", choices=("validation", "test"), default="test")
    stage = p.add_mutually_exclusive_group()
    stage.add_argument("--train-only", action="store_true")
    stage.add_argument("--eval-only", action="store_true")
    p.add_argument("--resume", action="store_true")
    return p


def main(argv=None):
    argument_parser = parser()
    command_line = sys.argv[1:] if argv is None else list(argv)
    args = argument_parser.parse_args(command_line)
    config = vars(args).copy()
    folder = args.out
    config["out"] = str(folder.resolve())
    if min(args.n_samples, args.n_eval, args.chunk_size, args.metric_chunk) < 1 or args.n_samples < 2:
        raise ValueError("Evaluation requires positive counts and at least two samples")
    if args.eval_only and args.resume:
        raise ValueError("--resume applies to training, not --eval-only")
    if args.stop_after is not None and args.stop_after < 1:
        raise ValueError("--stop-after must be a positive absolute update")
    if args.sampling_block < 0 or (args.sampling_block and args.sampler != "ddpm"):
        raise ValueError("--sampling-block must be nonnegative and is supported only for DDPM")
    if args.eval_only or args.resume:
        stored = json.loads((folder / "config.json").read_text())
        if args.resume:
            # Resume uses the original training configuration, including its LR schedule.
            # Explicit training flags are frozen, including explicitly requested defaults.
            defaults = vars(parser().parse_args(["--out", str(folder)]))
            options = {token.split("=", 1)[0] for token in command_line}
            explicit = {action.dest for action in argument_parser._actions
                        if options.intersection(action.option_strings)}
            for name in TRAINING_KEYS:
                if name in explicit and config[name] != stored.get(name, defaults[name]):
                    raise ValueError(f"Cannot change {name} while resuming")
        config = stored.copy()
        config.setdefault("coordinate_scale", "max")
        config.setdefault("gaussian_skip", False)
        config.setdefault("data_std", 1.)
        config.setdefault("condition_presence", False)
        config.setdefault("cond_drop_prob", 0.)
        if not args.train_only:
            validate_guidance(config, args.guidance_scale, args.condition_present)
        config["data"] = str(args.data.resolve()) if args.data else stored["data"]
        if sha256_file(config["data"]) != stored["data_sha256"]:
            raise ValueError("Dataset does not match the checkpoint's SHA256")
        with np.load(folder / "preprocessing.npz") as source:
            preprocessing = {name: source[name] for name in source.files}
        arrays, _ = prepare_data(config["data"], config, preprocessing)
    else:
        if min(args.n_updates, args.batch, args.n_train, args.n_val, args.n_test,
               args.eval_every, args.checkpoint_every, args.train_chunk) < 1 or args.lr <= 0:
            raise ValueError("Training counts and learning rate must be positive")
        if not 0 <= args.ema_decay < 1:
            raise ValueError("EMA decay must be in [0, 1)")
        if not np.isfinite(args.cond_drop_prob) or not 0 <= args.cond_drop_prob <= 1:
            raise ValueError("Condition dropout probability must be in [0, 1]")
        if args.cond_drop_prob and not args.condition_presence:
            raise ValueError("Condition dropout requires --condition-presence")
        if not args.train_only:
            validate_guidance(config, args.guidance_scale, args.condition_present)
        if (folder / "config.json").exists():
            raise FileExistsError(f"Run exists; use --resume or --eval-only: {folder}")
        config["data"] = str((args.data or Path("isospectral_30k.npz")).resolve())
        config["data_sha256"] = sha256_file(config["data"])
        arrays, preprocessing = prepare_data(config["data"], config)
        config["n_eig"] = arrays["train"][0].shape[1]
        config["data_std"] = float(preprocessing["data_std"])
        config["out"] = str(folder.resolve())
        config["versions"] = dict(jax=jax.__version__, equinox=eqx.__version__, optax=optax.__version__, numpy=np.__version__)
        config["devices"] = [str(device) for device in jax.devices()]
        root = Path(__file__).resolve().parents[2]
        source_paths = ("src/drums/models/diffusion.py", "src/drums/models/ring_transformer.py",
                        "training/isospectral/diffusion.py", "training/isospectral/common.py",
                        "training/isospectral/metrics.py")
        config["source_sha256"] = {name: sha256_file(root / name) for name in source_paths}
        config["matched_reference_split"] = (args.n_train, args.n_val, args.n_test) == (N_TRAIN, N_VAL, N_TEST)
        for name in ("train_only", "eval_only", "resume", "stop_after", *EVALUATION_KEYS):
            config.pop(name, None)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        np.savez(folder / "preprocessing.npz", **preprocessing)

    if args.eval_only:
        model, ema, _, _, completed, history = load_checkpoint(folder, config)
        completed = int(completed)
        timing_path = folder / "training.json"
        timing = json.loads(timing_path.read_text()) if timing_path.exists() else {}
    else:
        model, ema, history, timing = train(folder, config, arrays["train"], arrays["validation"],
                                            args.resume, args.stop_after)
        completed = timing["completed_updates"]
    if args.train_only:
        return
    if args.weights == "ema":
        if not config["ema_decay"]:
            raise ValueError("EMA weights requested but EMA was disabled during training")
        model = ema

    F, V, W = arrays[args.eval_split]
    n = min(args.n_eval, len(F))
    started = time.perf_counter()
    samples = generate_samples(model, F[:n], random.PRNGKey(args.eval_seed), make_schedule(config),
                               args.n_samples, args.chunk_size, args.sampler,
                               args.sampling_steps, args.eta, args.sampling_block,
                               args.guidance_scale, args.condition_present)
    sample_seconds = time.perf_counter() - started
    started = time.perf_counter()
    result, per_draw, means = score_samples(samples, V[:n], W[:n], args.sep_min, args.metric_chunk)
    metric_seconds = time.perf_counter() - started
    sampling_steps = config["n_steps"] if args.sampler == "ddpm" else args.sampling_steps
    unconditional = args.condition_present == 0 or args.guidance_scale == 0.
    calls_per_step = 1 if unconditional or args.guidance_scale == 1. else 2
    nfe = sampling_steps * calls_per_step
    root = Path(__file__).resolve().parents[2]
    evaluation_sources = {name: sha256_file(root / name) for name in (
        "src/drums/models/diffusion.py", "src/drums/models/ring_transformer.py", "training/isospectral/diffusion.py",
    )}
    summary = dict(config=config, evaluation=dict(split=args.eval_split, seed=args.eval_seed,
                   n_eval=n, n_samples=args.n_samples, chunk_size=args.chunk_size,
                   weights=args.weights, sampler=args.sampler,
                   sampling_block=args.sampling_block,
                   sampling_steps=sampling_steps,
                   eta=args.eta if args.sampler == "ddim" else None,
                   guidance_scale=args.guidance_scale, condition_present=args.condition_present,
                   predictor_condition_masks=([0] if unconditional else [1] if args.guidance_scale == 1. else [0, 1]),
                   condition_presence=config.get("condition_presence", False),
                   training_cond_drop_prob=config.get("cond_drop_prob", 0.),
                   source_sha256=evaluation_sources,
                   effective_conditioning="unconditional" if unconditional else "conditional",
                   predictor_evaluations_per_draw=nfe,
                   requested_predictor_evaluations=n * args.n_samples * nfe,
                   executed_predictor_evaluations=((n + args.chunk_size - 1) // args.chunk_size)
                       * args.chunk_size * args.n_samples * nfe,
                   metrics_interpretation=("Unconditional prior draws paired with targets for diagnostics; not conditional reconstruction performance"
                                           if unconditional else "Conditional reconstruction and mode-coverage metrics"),
                   completed_updates=completed, devices=[str(device) for device in jax.devices()],
                   versions=dict(jax=jax.__version__, equinox=eqx.__version__, optax=optax.__version__)),
                   n_train_pairs=config["n_train"] * (2 if config["paired"] else 1),
                   final_train_loss=float(np.asarray(history)[max(0, completed-500):completed].mean()),
                   timing=dict(training_seconds=timing.get("elapsed_seconds"),
                               sample_seconds=sample_seconds, metric_seconds=metric_seconds),
                   **{f"{args.eval_split}_steps{sampling_steps}": result})
    prefix = evaluation_prefix(config, summary["evaluation"])
    (folder / f"{prefix}_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    np.savez_compressed(folder / f"{prefix}_samples.npz", samples=samples,
                        truth=V[:n], twin=W[:n], spectra_features=F[:n],
                        target_scale=preprocessing["target_scale"],
                        guidance_scale=args.guidance_scale, condition_present=args.condition_present,
                        condition_presence=config.get("condition_presence", False),
                        training_cond_drop_prob=config.get("cond_drop_prob", 0.))
    np.savez_compressed(folder / f"{prefix}_per_draw.npz", **per_draw)
    np.savez_compressed(folder / f"{prefix}_mean_metrics.npz", **means)
    print(json.dumps(result, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    main()
