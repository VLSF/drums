"""Forward FEM check of saved regression/FM predictions against their input spectra.

Runs on a fixed balanced test subset and fixed FM draw IDs, without retraining.
Coordinates are restored to their original units; no per-shape scale fitting.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np

from spectral_ablation_metrics import is_valid, shape_errors


def polygon_spectrum(vertices, n_eig=30, n_elements=4000):
    """Preserve the boundary, reverse clockwise order, and use the collection mesh."""
    from drums.data.spectrum import get_drum_spectrum

    vertices = np.asarray(vertices, dtype=float)
    if not is_valid(vertices):
        raise ValueError("Invalid polygon")
    vertices = vertices - vertices.mean(axis=0)
    x, y = vertices.T
    signed_area = .5 * (x @ np.roll(y, -1) - y @ np.roll(x, -1))
    if signed_area < 0:
        vertices = vertices[::-1]
    values = np.sort(get_drum_spectrum(vertices, N_eig=n_eig,
                                       a=abs(signed_area) / n_elements)[0])
    if values.shape != (n_eig,) or not np.all(np.isfinite(values) & (values > 0)):
        raise ValueError("Solver returned an invalid spectrum")
    return values


def spectral_errors(predicted, target, n_eig):
    """Mean absolute relative eigenvalue errors; the observed band depends on n_eig."""
    relative = np.abs(np.asarray(predicted) / np.asarray(target) - 1)
    return dict(first5_mare=float(relative[:5].mean()),
                high25_mare=float(relative[5:].mean()),
                all30_mare=float(relative.mean()),
                observed_mare=float(relative[:n_eig].mean()))


def select_ids(test_ids, labels, per_family=30, seed=20261029):
    rng = np.random.default_rng(seed)
    return np.sort(np.concatenate([rng.choice(test_ids[labels == label], per_family, replace=False)
                                   for label in np.unique(labels)]))


def summarize(rows):
    summaries = []
    for model, n_eig in dict.fromkeys((r["model"], r["n_eig"]) for r in rows):
        group = [r for r in rows if (r["model"], r["n_eig"]) == (model, n_eig)]
        for label in ("all", *sorted({r["shape_label"] for r in group})):
            selected = group if label == "all" else [r for r in group if r["shape_label"] == label]
            success = [r for r in selected if r["status"] == "ok"]
            mean = lambda key: float(np.mean([r[key] for r in success])) if success else float("nan")
            summaries.append(dict(model=model, n_eig=n_eig, shape_label=label,
                                  count=len(selected), success_count=len(success),
                                  invalid_count=sum(r["status"] == "invalid" for r in selected),
                                  solver_error_count=sum(r["status"] == "solver_error" for r in selected),
                                  shape_error_mean=mean("shape_error"),
                                  **{f"{key}_mean": mean(key) for key in
                                     ("first5_mare", "high25_mare", "all30_mare", "observed_mare")},
                                  observed_mare_median=float(np.median([r["observed_mare"] for r in success]))
                                  if success else float("nan")))
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/spectral_ablation_balanced_20261028"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--per-family", type=int, default=30)
    parser.add_argument("--draws", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261029)
    args = parser.parse_args()
    if min(args.per_family, args.draws) < 1:
        parser.error("Counts must be positive")
    out = args.out or args.results / "spectral_consistency_20261029"
    if (out / "metadata.json").exists():
        parser.error("Output already exists; choose a new --out directory")
    config = json.loads((args.results / "metadata.json").read_text())
    with np.load(config["data"], allow_pickle=False) as data:
        targets = data["perimeters"]
        target_spectra = data["spectra"]
        labels = data["shape_label"]
        n_elements = int(data["n_elements"])
    with np.load(args.results / "preprocessing.npz") as data:
        test_ids = data["test_ids"]
        scale = float(data["target_scale"])
    ids = select_ids(test_ids, labels[test_ids], args.per_family, args.seed)
    # Each task uses the same targets; FM uses the first fixed draws, never a best draw.
    tasks = [("reference", 30, int(i), -1, targets[i]) for i in ids]
    for model in ("regression", "flow"):
        for n_eig in (30, 5):
            with np.load(args.results / f"{model}_{n_eig}" / "test_predictions.npz") as saved:
                np.testing.assert_array_equal(saved["test_ids"], test_ids)
                np.testing.assert_array_equal(saved["shape_label"], labels[test_ids])
                draws = 1 if model == "regression" else args.draws
                if draws > saved["samples"].shape[1]:
                    parser.error("Requested more draws than the saved predictions contain")
                for position in np.flatnonzero(np.isin(test_ids, ids)):
                    i = int(test_ids[position])
                    for draw in range(draws):
                        polygon = saved["samples"][position, draw].astype(float) * scale
                        tasks.append((model, n_eig, i, draw, polygon))
    out.mkdir(parents=True, exist_ok=True)
    metadata = dict(results=str(args.results.resolve()), data=config["data"],
                    data_sha256=config["data_sha256"], per_family=args.per_family,
                    draws=args.draws, seed=args.seed, test_ids=ids.tolist(),
                    n_elements=n_elements, solved_eigenvalues=30, target_scale=scale,
                    spectral_error="mean(abs(predicted_eigenvalue / input_eigenvalue - 1))",
                    sample_selection="Fixed balanced target subset; first saved FM draws",
                    reference="Stored four-point target remeshed: numerical/remeshing baseline",
                    scale_fitting=False)
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    rows, spectra, polygons = [], [], []
    start = time.perf_counter()
    fields = ["model", "n_eig", "test_id", "shape_label", "draw_id", "status", "message",
              "shape_error", "first5_mare", "high25_mare", "all30_mare", "observed_mare"]
    with (out / "per_draw.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for model, n_eig, i, draw, polygon in tasks:
            row = dict(model=model, n_eig=n_eig, test_id=i, shape_label=str(labels[i]),
                       draw_id=draw, status="ok", message="",
                       shape_error=float(shape_errors(targets[i], polygon)),
                       **{key: float("nan") for key in fields[-4:]})
            values = np.full(30, np.nan)
            if not is_valid(polygon):
                row["status"] = "invalid"
            else:
                try:
                    values = polygon_spectrum(polygon, 30, n_elements)
                    row.update(spectral_errors(values, target_spectra[i], n_eig))
                except Exception as error:
                    row.update(status="solver_error", message=f"{type(error).__name__}: {error}")
            rows.append(row)
            spectra.append(values)
            polygons.append(polygon)
            writer.writerow(row)
            handle.flush()
            if len(rows) == 1 or len(rows) % 50 == 0 or len(rows) == len(tasks):
                print(f"{len(rows)}/{len(tasks)}: {model}/{n_eig}; "
                      f"elapsed={time.perf_counter() - start:.1f}s; "
                      f"failures={sum(r['status'] != 'ok' for r in rows)}", flush=True)
    generated = np.asarray(spectra)
    target = target_spectra[[r["test_id"] for r in rows]]
    np.savez_compressed(out / "forward_spectra.npz", spectra=generated, target_spectra=target,
                        per_mode_relative_error=np.abs(generated / target - 1),
                        polygons=np.asarray(polygons), targets=targets[[r["test_id"] for r in rows]],
                        **{key: np.asarray([r[key] for r in rows]) for key in fields if key != "message"})
    summaries = summarize(rows)
    with (out / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summaries[0].keys())
        writer.writeheader()
        writer.writerows(summaries)
    for row in summaries:
        if row["shape_label"] == "all":
            print(row, flush=True)


if __name__ == "__main__":
    main()
