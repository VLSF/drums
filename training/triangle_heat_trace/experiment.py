"""Run the triangle heat-trace benchmark with exact, finite Dirichlet spectra."""

import argparse
import hashlib
import json
import platform
from collections import Counter
from pathlib import Path

import numpy as np
import scipy

from reconstruct import constrained_reconstruct, fit_trace, geometry, reconstruct, vertex_error


FAMILIES = {
    "equilateral": np.array([[0., 0.], [1., 0.], [.5, np.sqrt(3.) / 2.]]),
    "right_isosceles": np.array([[0., 0.], [1., 0.], [0., 1.]]),
    "30_60_90": np.array([[0., 0.], [.5, 0.], [0., np.sqrt(3.) / 2.]]),
}
TIME_POINTS = 80
HOLDOUT_COUNT = 100
ERROR_TARGET = .02
TAIL_REFERENCE_EXPONENT = 50.


def exact_spectrum(family, eigenvalue_limit):
    """Enumerate the complete unit-template spectrum up to a cutoff, with multiplicities."""
    if family not in FAMILIES:
        raise ValueError(f"Unknown analytic template family: {family}")
    right = family == "right_isosceles"
    multiplier = np.pi**2 if right else 16. * np.pi**2 / 9.
    max_index = int(np.sqrt(eigenvalue_limit / multiplier))
    modes = []
    for m in range(1, max_index + 1):
        # The equilateral triangle retains both index orders. Cutting along an
        # altitude or a square diagonal keeps precisely the antisymmetric modes.
        stop = max_index + 1 if family == "equilateral" else m
        n = np.arange(1, stop, dtype=np.int64)
        quadratic = m*m + n*n
        if not right:
            quadratic += m*n
        eigenvalues = multiplier * quadratic
        modes.append(eigenvalues[eigenvalues <= eigenvalue_limit])
    if not modes:
        return np.empty(0)
    return np.sort(np.concatenate(modes))


def scaled_spectrum_reference(family, base_area, areas, mode_count, t_min):
    """Cache enough unit-template values for every fit and start-of-window tail check."""
    max_scale_squared = max(1., float(np.max(areas) / base_area))
    reference_limit = TAIL_REFERENCE_EXPONENT * max_scale_squared / t_min
    count_limit = 1.1 * 4. * np.pi * mode_count / base_area
    limit = max(reference_limit, count_limit)
    values = exact_spectrum(family, limit)
    while len(values) < mode_count:
        limit *= 2.
        values = exact_spectrum(family, limit)
    return values


def sorted_lengths_angles(vertices):
    u = np.roll(vertices, 1, axis=0) - vertices
    v = np.roll(vertices, -1, axis=0) - vertices
    cross = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
    angles = np.arctan2(np.abs(cross), np.sum(u * v, axis=1))
    return np.sort(np.linalg.norm(u, axis=1)), np.sort(angles)


def reconstruction_metrics(solution, reference, true_sides, true_angles):
    result = {key: solution[key] for key in ("status", "method", "relative_residual")
              if key in solution}
    if solution["status"] != "ok":
        result["passes_2pct"] = False
        return result
    vertices = np.asarray(solution["vertices"])
    sides, angles = sorted_lengths_angles(vertices)
    error = vertex_error(reference, vertices)
    result.update(
        passes_2pct=bool(error <= ERROR_TARGET),
        relative_vertex_error=float(error),
        max_relative_side_error=float(np.max(np.abs(sides / true_sides - 1.))),
        max_angle_error_degrees=float(np.rad2deg(np.max(np.abs(angles - true_angles)))),
        estimated_sides=sides.tolist(),
        estimated_angles_degrees=np.rad2deg(angles).tolist(),
        estimated_vertices=vertices.tolist(),
    )
    return result


def evaluate_case(family, vertices, scale_squared, spectrum, mode_count, times,
                  case_kind, dataset_index=None):
    reference = vertices * np.sqrt(scale_squared)
    truth = geometry(reference)
    true_sides, true_angles = sorted_lengths_angles(reference)
    eigenvalues = spectrum[:mode_count] / scale_squared
    fit = fit_trace(eigenvalues, times)
    estimated = np.array([fit["area"], fit["perimeter"], fit["gamma"]], dtype=float)
    strict = reconstruct(*estimated)
    constrained = constrained_reconstruct(eigenvalues, times, fit=fit)
    # Use the same exact spectrum to estimate the omitted tail up to t_min*lambda=50.
    # Modal weights beyond that point are negligible for this diagnostic.
    reference_end = np.searchsorted(spectrum, TAIL_REFERENCE_EXPONENT * scale_squared / times[0],
                                    side="right")
    omitted = spectrum[mode_count:reference_end] / scale_squared
    tail = float(np.exp(-times[0] * omitted).sum())
    trace_start = float(fit["trace"][0])
    return {
        "family": family,
        "case_kind": case_kind,
        "dataset_index": dataset_index,
        "area": float(truth[0]),
        "true_perimeter": float(truth[1]),
        "true_gamma": float(truth[2]),
        "true_vertices": reference.tolist(),
        "true_angles_degrees": np.rad2deg(true_angles).tolist(),
        "true_sides": true_sides.tolist(),
        "lambda_first": float(eigenvalues[0]),
        "lambda_last": float(eigenvalues[-1]),
        "t_min_lambda_last": float(times[0] * eigenvalues[-1]),
        "trace_at_t_min": trace_start,
        "estimated_omitted_tail_at_t_min": tail,
        "estimated_relative_trace_tail_at_t_min": tail / (trace_start + tail),
        "fit": {
            "alpha_beta_gamma": np.asarray(fit["coefficients"]).tolist(),
            "area_perimeter_gamma": estimated.tolist(),
            "relative_invariant_errors": np.abs(estimated / truth - 1.).tolist(),
            "relative_trace_residual": float(fit["relative_residual"]),
            "condition_number": float(fit["condition_number"]),
        },
        "strict": reconstruction_metrics(strict, reference, true_sides, true_angles),
        "constrained": reconstruction_metrics(constrained, reference, true_sides, true_angles),
    }


def summarize(cases):
    summary = {
        "n_cases": len(cases),
        "area_range": [min(row["area"] for row in cases), max(row["area"] for row in cases)],
        "median_relative_invariant_errors": np.median(
            [row["fit"]["relative_invariant_errors"] for row in cases], axis=0
        ).tolist(),
        "median_t_min_lambda_last": float(np.median([row["t_min_lambda_last"] for row in cases])),
        "median_relative_trace_residual": float(np.median([row["fit"]["relative_trace_residual"] for row in cases])),
    }
    for mode in ("strict", "constrained"):
        records = [row[mode] for row in cases]
        statuses = Counter(row["status"] for row in records)
        valid = [row for row in records if row["status"] == "ok"]
        errors = [row["relative_vertex_error"] for row in valid]
        n_passed = sum(row["passes_2pct"] for row in records)
        summary[mode] = {
            "n_valid": len(valid),
            "n_pass_2pct": n_passed,
            "fraction_pass_2pct_all_cases": n_passed / len(cases),
            "status_counts": dict(sorted(statuses.items())),
            "method_counts_on_valid_only": dict(sorted(Counter(
                row.get("method", mode) for row in valid
            ).items())),
            "median_relative_vertex_error_on_valid_only": float(np.median(errors)) if errors else None,
            "p90_relative_vertex_error_on_valid_only": float(np.percentile(errors, 90)) if errors else None,
            "median_max_angle_error_degrees_on_valid_only": float(np.median(
                [row["max_angle_error_degrees"] for row in valid]
            )) if valid else None,
            "median_max_relative_side_error_on_valid_only": float(np.median(
                [row["max_relative_side_error"] for row in valid]
            )) if valid else None,
        }
    return summary


def run(dataset_path, mode_count, t_min, t_max):
    with np.load(dataset_path, allow_pickle=False) as archive:
        perimeters = np.asarray(archive["perimeters"], dtype=float)
    if perimeters.ndim != 3 or perimeters.shape[1:] != (3, 2) or len(perimeters) < HOLDOUT_COUNT:
        raise ValueError(f"Expected at least {HOLDOUT_COUNT} triangles in dataset field 'perimeters'")
    areas = np.asarray(geometry(perimeters))[:, 0]
    if not np.all(np.isfinite(areas) & (areas > 0)):
        raise ValueError("Dataset includes non-finite or degenerate triangles")
    test_indices = np.arange(len(areas) - HOLDOUT_COUNT, len(areas))
    test_areas = areas[test_indices]
    area_rank = np.argsort(test_areas, kind="stable")
    area_groups = np.array_split(area_rank, 4)
    times = np.geomspace(t_min, t_max, TIME_POINTS)
    median_area = float(np.median(areas))
    controls, held_out, by_family, by_area = [], [], {}, {}

    for family, vertices in FAMILIES.items():
        base_area = float(geometry(vertices)[0])
        all_sizes = np.r_[test_areas, median_area]
        reference = scaled_spectrum_reference(family, base_area, all_sizes, mode_count, t_min)
        first_lambda = {"equilateral": 16*np.pi**2/3, "right_isosceles": 5*np.pi**2,
                        "30_60_90": 112*np.pi**2/9}[family]
        np.testing.assert_allclose(reference[0], first_lambda, rtol=1e-14)
        controls.append(evaluate_case(family, vertices, 1., reference, mode_count, times, "unit"))
        controls.append(evaluate_case(family, vertices, median_area / base_area, reference,
                                      mode_count, times, "full_dataset_median_area"))
        family_cases = [evaluate_case(family, vertices, area / base_area, reference, mode_count,
                                      times, "held_out_area", int(index))
                        for area, index in zip(test_areas, test_indices)]
        held_out.extend(family_cases)
        by_family[family] = summarize(family_cases)
        by_area[family] = [summarize([family_cases[i] for i in group]) for group in area_groups]
        counts = by_family[family]
        print(f"{family}: strict {counts['strict']['n_pass_2pct']}/{HOLDOUT_COUNT}, "
              f"constrained {counts['constrained']['n_pass_2pct']}/{HOLDOUT_COUNT} at 2%", flush=True)

    return {
        "protocol": {
            "dataset": dataset_path.name,
            "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
            "dataset_n_triangles": len(areas),
            "held_out_index_range_inclusive": [int(test_indices[0]), int(test_indices[-1])],
            "held_out_areas_only": True,
            "full_dataset_area_median": median_area,
            "mode_count": mode_count,
            "eigenvalues": "Exact Dirichlet spectra of three templates, including multiplicity; no FEM input",
            "t_window": [t_min, t_max],
            "t_grid": "80 geometrically spaced points, endpoints included",
            "fit": "Unweighted least squares in the raw heat trace",
            "design_condition_number_after_column_scaling": controls[0]["fit"]["condition_number"],
            "target": "Vertex-relative error <= 2%, centered rigid/reflected alignment and six permutations, no scale fit",
            "reference_tail_exponent": TAIL_REFERENCE_EXPONENT,
            "area_quartiles": "Held-out areas in increasing order, four groups of 25, identical per family",
            "versions": {"python": platform.python_version(), "numpy": np.__version__,
                         "scipy": scipy.__version__},
        },
        "controls": controls,
        "held_out": {"by_family": by_family, "by_area_quartile": by_area, "cases": held_out},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("triangle_drums.npz"))
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("results.json"))
    parser.add_argument("--modes", type=int, default=50_000)
    parser.add_argument("--t-min", type=float, default=1.e-4)
    parser.add_argument("--t-max", type=float, default=2.e-4)
    args = parser.parse_args()
    if args.modes < 1 or not np.isfinite([args.t_min, args.t_max]).all() or not 0 < args.t_min < args.t_max:
        parser.error("Expected at least one mode and finite 0 < t_min < t_max")
    results = run(args.dataset, args.modes, args.t_min, args.t_max)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
