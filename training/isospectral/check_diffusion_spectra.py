"""Check saved Conway diffusion draws with the forward FEM solver on CPU.

Example (from the repository root):
    python training/isospectral/check_diffusion_spectra.py \
        --run results/ddpm_main_ratio \
        --evaluation-prefix test_ddim200_raw_eta0 \
        --n-eval 30 --draws 4 --workers 4 --outfolder results/ddpm_spectra

Rows are selected using a fixed independent random seed. Every selected row uses
the first saved draws; selection never depends on generated geometry or error.
Targets and twins are remeshed as numerical controls. Failure counts always use
the full requested case count; spectral averages are labeled as success-only.
"""

import os

# FEM enables float64 globally, so this checker must run separately from training.
os.environ["JAX_PLATFORMS"] = "cpu"
for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_variable] = "1"

import argparse
import csv
import hashlib
import json
from multiprocessing import get_context
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "regression"))
from spectral_ablation_metrics import is_valid
from spectral_consistency import polygon_spectrum
from drums.data.quadrilateral_drums import is_simple


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def quick_validity(polygon):
    """Finite nondegenerate simple boundary, using the existing crossing test."""
    polygon = np.asarray(polygon, dtype=float)
    if polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3:
        return False
    if not np.isfinite(polygon).all():
        return False
    centered = polygon - polygon.mean(0)
    scale = np.linalg.norm(centered, axis=1).max()
    if scale <= 0:
        return False
    distances = np.linalg.norm(centered[:, None] - centered[None], axis=-1)
    if np.any(distances[np.triu_indices(len(polygon), 1)] <= 1e-8 * scale):
        return False
    x, y = centered.T
    twice_area = abs(x @ np.roll(y, -1) - y @ np.roll(x, -1))
    return bool(twice_area > 1e-10 * scale**2 and is_simple(centered))


def solve_case(task):
    """A spawn-safe worker; unsuccessful solves retain a NaN spectrum and a row."""
    kind, evaluation_row, dataset_id, draw_id, polygon, supplied_spectrum = task
    row = dict(kind=kind, evaluation_row=evaluation_row, dataset_id=dataset_id,
               draw_id=draw_id, status="ok", message="",
               all30_mare=float("nan"), first5_mare=float("nan"), high25_mare=float("nan"))
    eigenvalues = np.full(30, np.nan)
    if not is_valid(polygon):
        row["status"] = "invalid"
    else:
        try:
            eigenvalues = polygon_spectrum(polygon, n_eig=30, n_elements=4000)
            error = np.abs(eigenvalues / supplied_spectrum - 1)
            row.update(all30_mare=float(error.mean()), first5_mare=float(error[:5].mean()),
                       high25_mare=float(error[5:].mean()))
        except Exception as error:
            row.update(status="solver_error", message=f"{type(error).__name__}: {error}")
    return row, eigenvalues


def summarize_cases(rows):
    summaries = {}
    for kind in ("generated", "target", "twin"):
        cases = [row for row in rows if row["kind"] == kind]
        success = [row for row in cases if row["status"] == "ok"]
        total = len(cases)
        summaries[kind] = dict(
            total_cases=total, successful_cases=len(success),
            invalid_cases=sum(row["status"] == "invalid" for row in cases),
            solver_error_cases=sum(row["status"] == "solver_error" for row in cases),
            success_fraction_all_cases=len(success) / total,
            below_1pct_fraction_all_cases=sum(row["all30_mare"] < .01 for row in success) / total,
            mean_all30_mare_successes=float(np.mean([r["all30_mare"] for r in success])) if success else None,
            median_all30_mare_successes=float(np.median([r["all30_mare"] for r in success])) if success else None,
            mean_first5_mare_successes=float(np.mean([r["first5_mare"] for r in success])) if success else None,
            mean_high25_mare_successes=float(np.mean([r["high25_mare"] for r in success])) if success else None,
        )
    return summaries


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--evaluation-prefix", required=True, help="Exact saved prefix, without _samples.npz")
    parser.add_argument("--n-eval", type=int, default=30)
    parser.add_argument("--draws", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261112)
    parser.add_argument("--outfolder", type=Path, required=True)
    args = parser.parse_args(argv)
    if min(args.n_eval, args.draws, args.workers) < 1:
        parser.error("Counts must be positive")
    run = args.run.resolve()
    prefix = args.evaluation_prefix
    if Path(prefix).name != prefix:
        parser.error("--evaluation-prefix must be a filename prefix inside --run")
    config = json.loads((run / "config.json").read_text())
    evaluation = json.loads((run / f"{prefix}_summary.json").read_text())["evaluation"]
    split = evaluation["split"]
    if split not in ("test", "validation"):
        raise ValueError("Expected a validation or test evaluation")
    source_path = Path(config["data"])
    source_hash = sha256_file(source_path)
    if config.get("data_sha256") and source_hash != config["data_sha256"]:
        raise ValueError("Dataset SHA256 does not match the frozen training configuration")
    with np.load(run / "preprocessing.npz", allow_pickle=False) as fitted:
        scale = float(fitted["target_scale"])
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid saved coordinate scale")
    samples_path = run / f"{prefix}_samples.npz"
    with np.load(samples_path, allow_pickle=False) as saved:
        samples = np.asarray(saved["samples"], dtype=float)
        truth = np.asarray(saved["truth"], dtype=float)
        twin = np.asarray(saved["twin"], dtype=float)
        np.testing.assert_allclose(saved["target_scale"], scale, rtol=1e-7)
    if samples.ndim != 4 or samples.shape[2:] != (9, 2) or truth.shape != samples.shape[:1] + (9, 2) or twin.shape != truth.shape:
        raise ValueError("Expected saved samples (rows, draws, 9, 2) and matching truth/twin arrays")
    if args.n_eval > len(samples) or args.draws > samples.shape[1]:
        raise ValueError("Requested more rows or draws than the exact saved evaluation contains")
    if int(evaluation["n_eval"]) != len(samples):
        raise ValueError("Saved evaluation metadata and sample row count disagree")
    first = config["n_train"] + (config["n_val"] if split == "test" else 0)
    dataset_ids = first + np.arange(len(samples))
    with np.load(source_path, allow_pickle=False) as source:
        targets = np.asarray(source["perimeters"][dataset_ids], dtype=float)
        twins = np.asarray(source["twin_perimeters"][dataset_ids], dtype=float)
        supplied = np.asarray(source["spectra"][dataset_ids, :30], dtype=float)
    if supplied.shape != (len(samples), 30):
        raise ValueError("Thirty supplied eigenvalues are required")
    # Check the row mapping before assigning physical input spectra to predictions.
    np.testing.assert_allclose((targets - targets.mean(1, keepdims=True)) / scale, truth, atol=2e-6, rtol=2e-6)
    np.testing.assert_allclose((twins - twins.mean(1, keepdims=True)) / scale, twin, atol=2e-6, rtol=2e-6)
    selected = np.sort(np.random.default_rng(args.seed).choice(len(samples), args.n_eval, replace=False))
    physical = samples * scale
    out = args.outfolder.resolve()
    if any((out / name).exists() for name in ("metadata.json", "per_case.csv", "summary.json")):
        raise FileExistsError(f"Choose a new output folder: {out}")
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    validity = np.asarray([quick_validity(polygon) for polygon in physical.reshape(-1, 9, 2)]).reshape(samples.shape[:2])
    np.savez_compressed(out / "all_draw_validity.npz", validity=validity, dataset_ids=dataset_ids)
    metadata = dict(
        run=str(run), evaluation_prefix=prefix, evaluation=evaluation,
        data=str(source_path), data_sha256=source_hash, samples_sha256=sha256_file(samples_path),
        split=split, seed=args.seed, selected_evaluation_rows=selected.tolist(),
        selected_dataset_ids=dataset_ids[selected].tolist(), draws_per_selected_row=args.draws,
        selection="Independent fixed random row subset; first saved draws; no selection by quality",
        n_elements=4000, solved_eigenvalues=30, target_scale=scale, scale_fitting=False,
        spectral_error="mean(abs(recomputed_eigenvalues / supplied_eigenvalues - 1))",
        all_draw_validity_criterion="Finite coordinates, distinct vertices, positive area, existing is_simple crossing test",
        selected_validity_criterion="Robust spectral_ablation_metrics.is_valid before FEM",
        denominator="All requested cases retained; failures counted; spectral means and medians use successful solves only",
        workers=args.workers,
    )
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    tasks = []
    for row in selected:
        common = (int(row), int(dataset_ids[row]))
        tasks.append(("target", *common, -1, targets[row], supplied[row]))
        tasks.append(("twin", *common, -1, twins[row], supplied[row]))
        for draw in range(args.draws):
            tasks.append(("generated", *common, draw, physical[row, draw], supplied[row]))
    fields = ["kind", "evaluation_row", "dataset_id", "draw_id", "status", "message",
              "all30_mare", "first5_mare", "high25_mare"]
    rows, eigenvalues = [], []
    with (out / "per_case.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        with get_context("spawn").Pool(args.workers) as pool:
            for index, (row, values) in enumerate(pool.imap(solve_case, tasks), start=1):
                rows.append(row)
                eigenvalues.append(values)
                writer.writerow(row)
                handle.flush()
                if index == 1 or index % 30 == 0 or index == len(tasks):
                    print(json.dumps(dict(completed=index, total=len(tasks),
                                          failures=sum(r["status"] != "ok" for r in rows),
                                          elapsed_seconds=time.perf_counter()-started)), flush=True)
    references = np.asarray([task[-1] for task in tasks])
    eigenvalues = np.asarray(eigenvalues)
    np.savez_compressed(
        out / "forward_spectra.npz", spectra=eigenvalues, supplied_spectra=references,
        per_mode_relative_error=np.abs(eigenvalues / references - 1),
        polygons=np.asarray([task[-2] for task in tasks]),
        **{name: np.asarray([row[name] for row in rows]) for name in fields},
    )
    summary = dict(
        **metadata,
        all_saved_draws=dict(total_draws=int(validity.size), valid_draws=int(validity.sum()),
                             invalid_draws=int((~validity).sum()), valid_fraction=float(validity.mean())),
        selected_cases=summarize_cases(rows), elapsed_seconds=time.perf_counter()-started,
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["selected_cases"], indent=2), flush=True)
    return summary


if __name__ == "__main__":
    main()
