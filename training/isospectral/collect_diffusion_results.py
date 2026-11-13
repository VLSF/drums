"""Validate and collect a completed four-run Conway diffusion suite.

    python training/isospectral/collect_diffusion_results.py \
        --root results/isospectral_diffusion_20261112 \
        --publish-summary training/isospectral/summary

Requires all eight final test DDIM50/200 EMA eta=1 summaries and all four
fem_test200/summary.json files. No training, inference, plotting or prose writing.
"""

import argparse
import csv
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path


RUNS = tuple((arm, features) for arm in ("main", "paired") for features in ("ratio", "ratio_white"))
STEPS = (50, 200)
REFERENCE = dict(n_updates=40000, batch=128, N_f=256, L=3, n_heads=4, lr=1e-3,
                 seed=0, n_steps=1000, beta_start=1e-4, beta_end=0.02,
                 n_train=27000, n_val=1500, n_test=1500, n_eig=30)
SHARED_KEYS = (*REFERENCE, "coordinate_scale", "time_features", "ema_decay", "gaussian_skip", "data_std")
METRICS = dict(
    error=("draws", "error"), error_set=("draws", "error_set"),
    sharp=("draws", "sharp"), blurry=("draws", "blurry"),
    e_set_over_sep=("draws", "mean_set_over_sep"), mid_over_sep=("draws", "mid_over_sep"),
    pick=("draws", "pick"), both_modes=("mode_balance", "both_modes"),
    mean_fraction_nearer_truth=("mode_balance", "mean_fraction_nearer_truth"),
    std_fraction_nearer_truth=("mode_balance", "std_over_drums"),
    mean_of_draws_sharp=("mean_of_draws", "sharp"),
    mean_of_draws_blurry=("mean_of_draws", "blurry"),
    mean_of_draws_e_set_over_sep=("mean_of_draws", "mean_set_over_sep"),
    legacy_energy_diagnostic=("energy_score",),
)


@lru_cache(maxsize=None)
def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha256_file(path))


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def equal(actual, expected):
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-8)
    return actual == expected


def normalized_config(config):
    return dict(config, gaussian_skip=config.get("gaussian_skip", False),
                data_std=config.get("data_std", 1.),
                coordinate_scale=config.get("coordinate_scale", "max"),
                time_features=config.get("time_features", 0))


def check_fields(actual, expected, context):
    for key, value in expected.items():
        require(equal(actual.get(key), value), f"{context}: expected {key}={value!r}, got {actual.get(key)!r}")


def metric_values(result):
    values = {}
    for name, path in METRICS.items():
        value = result
        for key in path:
            value = value[key]
        values[name] = value
    return values


def difference(current, reference):
    return current - reference if isinstance(current, (int, float)) and isinstance(reference, (int, float)) else None


def clean_json(value):
    if isinstance(value, dict):
        return {key: clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def validate_fem(fem, run, dataset_hash, evaluation, sample_path):
    check_fields(fem, dict(evaluation_prefix="test_ddim200_ema_eta1", split="test",
                          data_sha256=dataset_hash, n_elements=4000,
                          solved_eigenvalues=30, scale_fitting=False), str(run / "fem_test200"))
    require(Path(fem["run"]).resolve() == run.resolve(), f"FEM source run differs: {run}")
    check_fields(fem["evaluation"], {key: evaluation[key] for key in (
        "split", "seed", "n_eval", "n_samples", "weights", "sampler", "sampling_steps", "eta", "completed_updates",
    )}, f"{run}: FEM evaluation")
    require(fem["samples_sha256"] == sha256_file(sample_path.resolve()), f"FEM sample SHA256 differs: {run}")
    rows, ids = fem["selected_evaluation_rows"], fem["selected_dataset_ids"]
    require(len(rows) > 0 and len(set(rows)) == len(rows), f"FEM rows must be a nonempty unique subset: {run}")
    require(ids == [28500 + row for row in rows] and all(0 <= row < 1500 for row in rows),
            f"FEM test row mapping differs: {run}")
    draws = fem["draws_per_selected_row"]
    require(isinstance(draws, int) and 1 <= draws <= 16, f"Invalid FEM draw count: {run}")
    all_draws = fem["all_saved_draws"]
    require(all_draws["total_draws"] == 24000
            and all_draws["valid_draws"] + all_draws["invalid_draws"] == 24000,
            f"FEM all-draw denominator differs: {run}")
    for kind in ("generated", "target", "twin"):
        group = fem["selected_cases"][kind]
        count = len(rows) * draws if kind == "generated" else len(rows)
        require(group["total_cases"] == count and sum(group[name] for name in (
            "successful_cases", "invalid_cases", "solver_error_cases")) == count,
            f"FEM failure counts do not retain all requested {kind} cases: {run}")


def collect(root, baseline_root):
    required = [root / f"conway_{arm}_{features}" / name
                for arm, features in RUNS
                for name in ("config.json", "training.json", "fem_test200/summary.json",
                             "test_ddim200_ema_eta1_samples.npz",
                             *(f"test_ddim{steps}_ema_eta1_summary.json" for steps in STEPS))]
    missing = [str(path) for path in required if not path.is_file()]
    require(not missing, "Suite is incomplete; missing:\n" + "\n".join(missing))
    rows, published, first_config, first_fem = [], {}, None, None
    for arm, features in RUNS:
        run = root / f"conway_{arm}_{features}"
        config = normalized_config(read(run / "config.json"))
        check_fields(config, dict(REFERENCE, features=features, paired=arm == "paired"), str(run))
        require(0 < config["ema_decay"] < 1, f"EMA must be enabled: {run}")
        if first_config is None:
            first_config = config
        check_fields(config, {key: first_config[key] for key in (*SHARED_KEYS, "data_sha256")}, str(run))
        dataset = Path(config["data"]).resolve()
        require(sha256_file(dataset) == config["data_sha256"], f"Dataset SHA256 differs: {run}")
        training = read(run / "training.json")
        n_train_pairs = 54000 if arm == "paired" else 27000
        check_fields(training, dict(completed_updates=40000, n_train_pairs=n_train_pairs), str(run / "training.json"))
        require(training.get("validation") and training["validation"][-1]["update"] == 40000,
                f"Final 40k validation is missing: {run}")
        fm_path = baseline_root / f"fm_{arm}_{features}.json"
        fm = read(fm_path)
        check_fields(fm["args"], dict(features=features, paired=arm == "paired", n_updates=40000,
                                     batch=128, N_f=256, L=3, lr=1e-3, seed=0,
                                     n_eval=1500, n_samples=16), str(fm_path))
        results, evaluations, sources, timings = {}, {}, {}, {}
        final_train_loss = None
        for steps in STEPS:
            path = run / f"test_ddim{steps}_ema_eta1_summary.json"
            summary = read(path)
            check_fields(normalized_config(summary["config"]),
                         {key: config[key] for key in (*SHARED_KEYS, "features", "paired", "data_sha256")}, str(path))
            evaluation = summary["evaluation"]
            check_fields(evaluation, dict(split="test", seed=443, n_eval=1500, n_samples=16,
                                         weights="ema", sampler="ddim", sampling_steps=steps,
                                         eta=1., completed_updates=40000), str(path))
            check_fields(summary, dict(n_train_pairs=n_train_pairs), str(path))
            key = f"test_steps{steps}"
            result, baseline = summary[key], fm[key]
            check_fields(result, dict(n_spectra=1500, n_samples=16, sep_min=0.1), str(path))
            check_fields(result["draws"], {name: baseline["draws"][name] for name in ("n_separated", "mean_sep")}, str(path))
            check_fields(result["mean_of_draws"],
                         dict(n_separated=result["draws"]["n_separated"] // 16), str(path))
            check_fields(result["mode_balance"], dict(std_if_binomial=0.125), str(path))
            if final_train_loss is not None:
                require(equal(summary["final_train_loss"], final_train_loss), f"Final training loss differs between samplers: {run}")
            final_train_loss = summary["final_train_loss"]
            current, reference = metric_values(result), metric_values(baseline)
            row = dict(run=run.name, arm=arm, features=features, sampler="DDIM", eta=1.,
                       steps=steps, weights="EMA", completed_updates=40000,
                       n_eval=1500, n_samples=16, n_train_pairs=n_train_pairs,
                       **current, **{f"fm_{name}": value for name, value in reference.items()},
                       **{f"delta_{name}": difference(current[name], reference[name]) for name in current},
                       sampling_seconds=summary["timing"]["sample_seconds"],
                       training_seconds=training["elapsed_seconds"],
                       diffusion_source=str(path.resolve()), fm_source=str(fm_path.resolve()))
            rows.append(row)
            results[key], evaluations[key], sources[key], timings[key] = result, evaluation, source(path), summary["timing"]
        fem_path = run / "fem_test200" / "summary.json"
        fem = read(fem_path)
        validate_fem(fem, run, config["data_sha256"], evaluations["test_steps200"],
                     run / "test_ddim200_ema_eta1_samples.npz")
        fem_protocol = {key: fem[key] for key in (
            "seed", "selected_dataset_ids", "draws_per_selected_row", "n_elements", "solved_eigenvalues", "target_scale",
        )}
        if first_fem is None:
            first_fem = fem_protocol
        check_fields(fem_protocol, first_fem, f"{run}: matched FEM subset")
        generated = fem["selected_cases"]["generated"]
        rows[-1].update(
            fem_all_draw_valid_fraction=fem["all_saved_draws"]["valid_fraction"],
            fem_generated_cases=generated["total_cases"],
            fem_success_fraction_all_cases=generated["success_fraction_all_cases"],
            fem_below_1pct_fraction_all_cases=generated["below_1pct_fraction_all_cases"],
            fem_mean_all30_mare_successes=generated["mean_all30_mare_successes"],
            fem_median_all30_mare_successes=generated["median_all30_mare_successes"],
            fem_invalid_cases=generated["invalid_cases"], fem_solver_error_cases=generated["solver_error_cases"],
            fem_target_mean_all30_mare=fem["selected_cases"]["target"]["mean_all30_mare_successes"],
            fem_twin_mean_all30_mare=fem["selected_cases"]["twin"]["mean_all30_mare_successes"],
        )
        training_metadata = {key: training[key] for key in ("completed_updates", "n_train_pairs", "elapsed_seconds")}
        training_metadata["final_validation"] = training["validation"][-1]
        name = f"diffusion_{arm}_{features}"
        published[name] = dict(
            method="conditional_diffusion", args=dict(config, n_eval=1500, n_samples=16), config=config,
            n_train_pairs=n_train_pairs, final_train_loss=final_train_loss,
            training=training_metadata, **results, evaluation=evaluations, timing=timings,
            sampler=dict(name="DDIM", eta=1., trained_steps=1000, evaluated_steps=list(STEPS), weights="EMA"),
            fem_test_steps200=fem,
            matched_flow_matching=dict(sampler="Euler", source=source(fm_path),
                                       deltas={f"test_steps{row['steps']}": {key: value for key, value in row.items()
                                               if key.startswith("delta_")} for row in rows[-2:]}),
            provenance=dict(config=source(run / "config.json"), training=source(run / "training.json"),
                            evaluations=sources, fem=source(fem_path), dataset=source(dataset),
                            fem_samples=source(run / "test_ddim200_ema_eta1_samples.npz")),
            energy_score_interpretation="Legacy asymmetric-distance diagnostic; not established as a proper scoring rule",
        )
    protocol = dict(reference_training=REFERENCE,
                    shared_settings={key: first_config[key] for key in SHARED_KEYS if key not in REFERENCE},
                    test=dict(n_eval=1500, n_samples=16, split="test", seed=443,
                              sampler="DDIM", steps=list(STEPS), eta=1., weights="EMA", sep_min=0.1),
                    fem=first_fem, dataset_sha256=first_config["data_sha256"],
                    deltas="Diffusion minus historical flow matching at the same step count",
                    historical_dataset_provenance="Historical FM files do not contain a dataset hash; split, separation counts and mean separation match",
                    energy_score_interpretation="Legacy asymmetric-distance diagnostic; not established as a proper scoring rule")
    return clean_json(dict(protocol=protocol, rows=rows, runs=published))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, help="Defaults to ROOT/summary")
    parser.add_argument("--publish-summary", type=Path, help="Optionally write the four lightweight diffusion_*.json files here")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    baseline_root = Path(__file__).resolve().parent / "summary"
    result = collect(root, baseline_root)  # Validate everything before creating outputs.
    result["provenance"] = dict(root=str(root), collector=source(Path(__file__)),
                                 collected_at_utc=datetime.now(timezone.utc).isoformat())
    for run in result["runs"].values():
        run["protocol"] = result["protocol"]
        run["provenance"]["collection"] = result["provenance"]
    out = (args.out or root / "summary").resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "comparison.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    columns = list(dict.fromkeys(key for row in result["rows"] for key in row))
    with (out / "comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(result["rows"])
    if args.publish_summary:
        args.publish_summary.mkdir(parents=True, exist_ok=True)
        for name, run in result["runs"].items():
            (args.publish_summary / f"{name}.json").write_text(json.dumps(run, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(rows=len(result["rows"]), runs=len(result["runs"]),
                          csv=str(out / "comparison.csv"), json=str(out / "comparison.json"),
                          published_to=str(args.publish_summary) if args.publish_summary else None), indent=2))
    return result


if __name__ == "__main__":
    main()
