"""Reproduce the heat-trace window and finite-spectrum diagnostics on saved modes."""

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import platform

import numpy as np
import scipy
from scipy.special import erf


TIME_POINTS = 100
PASS_LIMIT = 0.02
BASELINE_MODES = 60_000
BASELINE_WINDOW = (1e-4, 2e-4)
EXAMPLE_INDEX = 900
WINDOWS = (
    ("baseline", *BASELINE_WINDOW),
    ("widen_fixed_left", 1e-4, 2e-3),
    ("shift_2p5e_4", 2.5e-4, 5e-4),
    ("shift_1e_3", 1e-3, 2e-3),
    ("shift_5e_3", 5e-3, 1e-2),
    ("shift_1e_2", 1e-2, 2e-2),
)
ADDITIONAL_WINDOWS = (
    ("widen_fixed_left_3e_4", 1e-4, 3e-4),
    ("widen_fixed_left_5e_4", 1e-4, 5e-4),
    ("widen_fixed_left_1e_3", 1e-4, 1e-3),
    ("shift_1p5e_4", 1.5e-4, 3e-4),
    ("shift_2e_4", 2e-4, 4e-4),
    ("shift_3e_4", 3e-4, 6e-4),
    ("shift_4e_4", 4e-4, 8e-4),
    ("shift_5e_4", 5e-4, 1e-3),
    ("shift_7p5e_4", 7.5e-4, 1.5e-3),
    ("shift_2e_3", 2e-3, 4e-3),
)


def fingerprint(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return {"filename": path.name, "sha256": digest.hexdigest()}


def import_reconstructor(path):
    spec = importlib.util.spec_from_file_location("drums_heat_reconstruct", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load reconstruction module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fit_weyl(reconstructor, spectrum, times, raw):
    """Fit the truncated continuum model using the first two Weyl density terms."""
    design, t_min = reconstructor._design(times)
    edge = times * spectrum[-1]
    design[:, 0] *= -np.expm1(-edge)
    design[:, 1] *= erf(np.sqrt(edge))
    column_norms = np.linalg.norm(design, axis=0)
    scaled_fit, _, rank, singular_values = np.linalg.lstsq(
        design / column_norms, raw["trace"], rcond=None
    )
    if rank < 3:
        raise ValueError("The Weyl-tail design is numerically rank deficient")
    fitted = scaled_fit / column_norms
    return {
        "area": float(fitted[0] * 4 * np.pi * t_min),
        "perimeter": float(fitted[1] * -8 * np.sqrt(np.pi * t_min)),
        "gamma": float(fitted[2]),
        "condition_number": float(singular_values[0] / singular_values[-1]),
        "relative_residual": float(
            np.linalg.norm(design @ fitted - raw["trace"]) / np.linalg.norm(raw["trace"])
        ),
    }


def shape_metrics(reconstructor, solution, truth):
    metrics = {"status": solution["status"]}
    if solution["status"] == "ok":
        error = float(reconstructor.vertex_error(truth, solution["vertices"]))
        metrics.update(
            relative_vertex_error=error,
            passes_2pct=error <= PASS_LIMIT,
            angles_deg=np.rad2deg(solution["angles"]).tolist(),
            method=solution["method"],
        )
    else:
        metrics.update(relative_vertex_error=None, passes_2pct=False)
    return metrics


def counts(rows, method="strict"):
    outcomes = [row[method] for row in rows]
    return {
        "n": len(rows),
        "feasible": sum(result["status"] == "ok" for result in outcomes),
        "passes_2pct": sum(result["passes_2pct"] for result in outcomes),
        "status_counts": dict(sorted(Counter(result["status"] for result in outcomes).items())),
    }


def solve_case(reconstructor, index, spectrum, vertices, times, tail_model, fallback):
    raw = reconstructor.fit_trace(spectrum, times)
    fit = fit_weyl(reconstructor, spectrum, times, raw) if tail_model else raw
    estimates = [fit[k] for k in ("area", "perimeter", "gamma")]
    solution = reconstructor.reconstruct(*estimates)
    row = {
        "index": int(index),
        "t_min_lambda_last": float(times[0] * spectrum[-1]),
        "fit_area_perimeter_gamma": estimates,
        "condition_number": fit["condition_number"],
        "relative_trace_residual": fit["relative_residual"],
        "strict": shape_metrics(reconstructor, solution, vertices),
    }
    if fallback and not tail_model:
        repaired = reconstructor.constrained_reconstruct(spectrum, times, fit=raw)
        row["with_fallback"] = shape_metrics(reconstructor, repaired, vertices)
    if index == EXAMPLE_INDEX:
        area, perimeter, gamma = reconstructor.geometry(vertices)
        short_time_terms = (
            area / (4 * np.pi * times) - perimeter / (8 * np.sqrt(np.pi * times)) + gamma
        )
        b = 4 * np.pi * times[0] / area
        row["reference_geometry"] = [float(area), float(perimeter), float(gamma)]
        row["trace"] = raw["trace"].tolist()
        row["three_term_trace_at_reference_geometry"] = short_time_terms.tolist()
        row["reference_minus_truncated_trace_at_t_min"] = float(short_time_terms[0] - raw["trace"][0])
        row["continuous_polya_tail_upper_bound_at_t_min"] = float(
            np.exp(-b * (len(spectrum) + 1)) / (-np.expm1(-b))
        )
    return row


def run_trial(reconstructor, name, interval, modes, indices, spectra, vertices,
              baseline_failures, *, tail_model=False, fallback=False):
    times = np.geomspace(*interval, TIME_POINTS)
    rows = [
        solve_case(
            reconstructor, index, spectrum[:modes], triangle, times, tail_model,
            fallback or index == EXAMPLE_INDEX,
        )
        for index, spectrum, triangle in zip(indices, spectra, vertices)
    ]
    original_failures = [row for row in rows if row["index"] in baseline_failures]
    feasible_errors = [row["strict"]["relative_vertex_error"] for row in rows
                       if row["strict"]["status"] == "ok"]
    result = {
        "name": name,
        "interval": list(interval),
        "modes": modes,
        "trace_model": "two_term_weyl_at_lambda_K" if tail_model else "three_term_heat_trace",
        "count": counts(rows),
        "baseline_incompatible_count": counts(original_failures),
        "baseline_incompatible_feasible_indices": [
            row["index"] for row in original_failures if row["strict"]["status"] == "ok"
        ],
        "baseline_incompatible_pass_indices": [
            row["index"] for row in original_failures if row["strict"]["passes_2pct"]
        ],
        "strict_median_relative_error_among_feasible": (
            float(np.median(feasible_errors)) if feasible_errors else None
        ),
        "cases": rows,
    }
    if fallback:
        result["with_fallback_count"] = counts(rows, "with_fallback")
    return result


def compare_nested_modes(reconstructor, baseline, reduced):
    """Verify the coefficient shift caused by removing known modes 50,001–60,000."""
    high, low = [
        next(row for row in trial["cases"] if row["index"] == EXAMPLE_INDEX)
        for trial in (baseline, reduced)
    ]
    times = np.geomspace(*BASELINE_WINDOW, TIME_POINTS)
    design, t_min = reconstructor._design(times)
    missing = np.asarray(high["trace"]) - np.asarray(low["trace"])
    projected = np.linalg.lstsq(design, -missing, rcond=None)[0]
    projected *= [4 * np.pi * t_min, -8 * np.sqrt(np.pi * t_min), 1]
    actual = np.asarray(low["fit_area_perimeter_gamma"]) - high["fit_area_perimeter_gamma"]
    return {
        "dataset_index": EXAMPLE_INDEX,
        "modes_removed": [50_001, 60_000],
        "trace_lost_at_t_min": float(missing[0]),
        "trace_lost_at_t_max": float(missing[-1]),
        "fit_50000_minus_fit_60000_area_perimeter_gamma": actual.tolist(),
        "projection_of_negative_lost_trace_area_perimeter_gamma": projected.tolist(),
        "projection_max_abs_discrepancy": float(np.max(np.abs(actual - projected))),
    }


def render_figure(path, trials):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

    labels = ["Original window", "Widened at fixed start", r"$[2.5,5]\times10^{-4}$",
              r"$[1,2]\times10^{-3}$", r"$[5,10]\times10^{-3}$", r"$[1,2]\times10^{-2}$"]
    blue, pale, grey = "#075f91", "#eec384", "#e3e7eb"
    figure = Figure(figsize=(11.2, 4.7), dpi=160)
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 3, sharey=True, gridspec_kw={"width_ratios": [1.35, 1, .78]})
    axis_window, axis_all, axis_fail = axes
    y = np.arange(len(trials))
    for pos, trial in zip(y, trials):
        left, right = trial["interval"]
        axis_window.plot([left, right], [pos, pos], marker="o", linewidth=2., markersize=4.5,
                         color=blue if pos != 1 else "#8b500c")
    axis_window.set_xscale("log")
    ticks = [1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2]
    axis_window.xaxis.set_major_locator(FixedLocator(ticks))
    axis_window.xaxis.set_minor_locator(NullLocator())
    axis_window.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    axis_window.tick_params(axis="x", labelrotation=30)
    axis_window.set_xlim(8.6e-5, 2.3e-2)
    axis_window.set_xlabel("Time")
    axis_window.set_yticks(y, labels)
    axis_window.set_ylim(len(y)-.5, -.5)
    axis_window.set_title("Time windows", weight="bold", loc="left")
    original = trials[0]
    original_recovery = sum(
        row["with_fallback"]["passes_2pct"]
        for row in original["cases"] if row["strict"]["status"] != "ok"
    )
    for axis, key, limit, ticks, label, fallback_passes in (
        (axis_all, "count", 100, [0, 25, 50, 75, 100], "All 100 test cases",
         original["with_fallback_count"]["passes_2pct"]),
        (axis_fail, "baseline_incompatible_count", 49, [0, 10, 20, 30, 40, 49],
         "49 original rejections", original_recovery),
    ):
        passed = np.array([trial[key]["passes_2pct"] for trial in trials])
        feasible = np.array([trial[key]["feasible"] for trial in trials])
        axis.barh(y, passed, height=.58, color=blue, zorder=2)
        axis.barh(y, feasible-passed, left=passed, height=.58, color=pale, zorder=2)
        axis.barh(y, limit-feasible, left=feasible, height=.58, color=grey, zorder=2)
        axis.axvline(fallback_passes, color="#4b344f", linestyle="--", lw=1.05, zorder=3)
        for yy, n_ok in zip(y, passed):
            if n_ok:
                axis.text(n_ok-1.25, yy, str(n_ok), va="center", ha="right", color="white",
                          fontsize=9, fontweight="bold")
        axis.set(xlim=(0, limit), xticks=ticks, xlabel="Cases")
        axis.set_title(label, weight="bold", loc="left")
        axis.tick_params(axis="y", labelleft=False)
    for axis in axes:
        axis.set_axisbelow(True)
        axis.grid(axis="x", color="#d7dee3", linewidth=.65)
        axis.tick_params(color="#8f9ba4", labelcolor="#27323b", length=3, labelsize=9)
        for border in ["top", "right", "left"]:
            axis.spines[border].set_visible(False)
        axis.spines["bottom"].set_color("#a6b0ba")
    legend = [
        Patch(facecolor=blue, label="Vertex error ≤ 2%"),
        Patch(facecolor=pale, label="Feasible, vertex error > 2%"),
        Patch(facecolor=grey, label="Rejected"),
        Line2D([0], [0], color="#4b344f", linestyle="--", lw=1.05,
               label="Original window with fallback"),
    ]
    figure.suptitle("Strict triangle reconstruction across time windows",
                    y=.96, fontsize=15, fontweight="bold")
    figure.subplots_adjust(left=.14, right=.98, bottom=.22, top=.82, wspace=.29)
    figure.legend(handles=legend, loc="lower center", ncols=4, frameon=False,
                  fontsize=9, bbox_to_anchor=(.55, .01))
    figure.savefig(path, dpi=160, facecolor="white")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--spectra", type=Path, required=True)
    parser.add_argument("--reconstruct", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--figure", type=Path)
    parser.add_argument("--weyl", action="store_true", help="Also fit the two-term Weyl-tail model")
    arguments = parser.parse_args()

    reconstructor = import_reconstructor(arguments.reconstruct)
    with np.load(arguments.spectra, allow_pickle=False) as input_modes:
        indices = input_modes["indices"]
        spectra = input_modes["spectra"]
    if not np.array_equal(indices, np.arange(900, 1000)) or spectra.shape != (100, BASELINE_MODES):
        parser.error("Expected indices 900–999 and 60,000 eigenvalues per triangle")
    if np.any(np.diff(spectra, axis=1) < 0):
        parser.error("Expected ascending eigenvalues, preserving numerical multiplicities")
    with np.load(arguments.dataset, allow_pickle=False) as archive:
        vertices = archive["perimeters"][indices]

    baseline = run_trial(
        reconstructor, "baseline", BASELINE_WINDOW, BASELINE_MODES, indices, spectra, vertices,
        set(), fallback=True,
    )
    failures = {row["index"] for row in baseline["cases"] if row["strict"]["status"] != "ok"}
    baseline["baseline_incompatible_count"] = counts(
        [row for row in baseline["cases"] if row["index"] in failures]
    )
    windows = [
        run_trial(reconstructor, name, (left, right), BASELINE_MODES, indices, spectra, vertices,
                  failures)
        for name, left, right in WINDOWS[1:]
    ]
    additional_windows = [
        run_trial(reconstructor, name, (left, right), BASELINE_MODES, indices, spectra, vertices,
                  failures)
        for name, left, right in ADDITIONAL_WINDOWS
    ]
    reduced = run_trial(reconstructor, "baseline_50000", BASELINE_WINDOW, 50_000, indices,
                        spectra, vertices, failures)
    position = np.flatnonzero(indices == EXAMPLE_INDEX)[0]
    extra_example = solve_case(
        reconstructor, EXAMPLE_INDEX, spectra[position], vertices[position],
        np.geomspace(2e-4, 4e-4, TIME_POINTS), False, True,
    )
    extra_example["interval"] = [2e-4, 4e-4]
    results = {
        "inputs": {
            name: fingerprint(path)
            for name, path in (("dataset", arguments.dataset), ("spectra", arguments.spectra),
                               ("reconstruct", arguments.reconstruct))
        },
        "dependencies": {
            "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__
        },
        "protocol": {
            "indices": indices.tolist(),
            "time_points_logarithmic": TIME_POINTS,
            "relative_vertex_error_cutoff": PASS_LIMIT,
            "baseline_incompatible_indices": sorted(failures),
            "tail_inference": "Standard and Weyl fits use the spectrum. Reference geometry enters the audit.",
        },
        "window_trials": [baseline, *windows],
        "additional_window_trials": additional_windows,
        "baseline_50000": reduced,
        "row_900_extra_window": extra_example,
        "row_900_nested_mode_comparison": compare_nested_modes(reconstructor, baseline, reduced),
    }
    if arguments.weyl:
        results["baseline_two_term_weyl"] = run_trial(
            reconstructor, "baseline_two_term_weyl", BASELINE_WINDOW, BASELINE_MODES,
            indices, spectra, vertices, failures, tail_model=True,
        )
    arguments.results.parent.mkdir(parents=True, exist_ok=True)
    arguments.results.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n",
                                 encoding="utf-8")
    if arguments.figure:
        arguments.figure.parent.mkdir(parents=True, exist_ok=True)
        render_figure(arguments.figure, results["window_trials"])
    for trial in results["window_trials"] + additional_windows + [reduced] + (
        [results["baseline_two_term_weyl"]] if arguments.weyl else []
    ):
        all_rows = trial["count"]
        rejected = trial["baseline_incompatible_count"]
        print(f"{trial['name']}: {all_rows['feasible']} feasible, "
              f"{all_rows['passes_2pct']} at 2% overall; "
              f"{rejected['feasible']}/{rejected['passes_2pct']} among original rejections")


if __name__ == "__main__":
    main()
