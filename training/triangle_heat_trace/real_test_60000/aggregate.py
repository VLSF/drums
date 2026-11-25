"""Stitch, audit, and evaluate settled triangle spectral checkpoints."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import scipy

from check_control import validate as validate_control
from check_spectra import stitch_windows
from experiment import reconstruction_metrics, sorted_lengths_angles, summarize
from reconstruct import constrained_reconstruct, fit_trace, geometry, reconstruct


MODE_COUNT = 60_000
TEST_TIME = np.geomspace(1e-4, 2e-4, 100)
AUDIT_TIME = np.geomspace(4e-5, 8e-5, 100)
OVERLAP_ATOL = 5e-4


def _polya_tail_bound(t_min, area):
    b = 4 * np.pi * t_min / area
    return float(np.exp(-b * (MODE_COUNT + 1)) / (-np.expm1(-b)))


def _true_short_time_trace(invariants, times):
    area, perimeter, gamma = invariants
    return area / (4 * np.pi * times) - perimeter / (8 * np.sqrt(np.pi * times)) + gamma


def _fit_fields(fit, truth):
    estimated = np.array([fit['area'], fit['perimeter'], fit['gamma']])
    return {
        'alpha_beta_gamma': fit['coefficients'].tolist(),
        'area_perimeter_gamma': estimated.tolist(),
        'invariant_errors': (estimated - truth).tolist(),
        'relative_invariant_errors': np.abs(estimated / truth - 1.).tolist(),
        'relative_trace_residual': fit['relative_residual'],
        'condition_number': fit['condition_number'],
    }


def evaluate(path, vertices, reference_30, index):
    with np.load(path, allow_pickle=False) as saved:
        if not np.array_equal(vertices, saved['vertices']):
            raise ValueError(f'Checkpoint {path} has mismatched dataset vertices for row {index}')
        centers, offsets, candidates = (saved[k] for k in ('centers', 'offsets', 'values'))
        radius = float(saved['radius'])
        area = float(saved['area'])
        if centers[-1] < 877. or ('finished' in saved and not bool(saved['finished'])):
            return None, None, {'status': 'spectral_scan_incomplete', 'k_max': float(centers[-1])}
        spectral_settings = {key: float(saved[key]) for key in ('step', 'phase', 'pad', 'seconds')}
        for key in ('repairs', 'max_rss_bytes'):
            if key in saved:
                spectral_settings[key] = int(saved[key])
        if 'flagged_windows' in saved:
            spectral_settings['flagged_windows'] = sorted(set(saved['flagged_windows'].tolist()))
        if 'targeted_repair_full_band_audits' in saved:
            spectral_settings['targeted_repair'] = {
                'source_flags_before_repair': True,
                'band_audit_columns': saved['targeted_repair_audit_columns'].tolist(),
                'band_audits': saved['targeted_repair_full_band_audits'].tolist(),
                'baseline_count_disagreement_centers': (
                    saved['baseline_overlap_count_disagreement_centers'].tolist()
                ),
            }
        if 'interior_normalized_mps_band_audits' in saved:
            spectral_settings['local_boundary_tension_checks'] = json.loads(
                str(saved['interior_normalized_mps_band_audits']))
    if (len(offsets) != len(centers) + 1 or offsets[0] != 0
            or offsets[-1] != len(candidates) or np.any(np.diff(offsets) < 0)):
        raise ValueError(f'Unusable checkpoint offsets in {path}')
    truth = geometry(vertices)
    if not np.isclose(truth[0], area, atol=0., rtol=1.e-13):
        raise ValueError(f'Checkpoint normalization area disagrees with row {index}')
    windows = [(center, radius, candidates[begin:end])
               for center, begin, end in zip(centers, offsets[:-1], offsets[1:])]
    all_modes, discrepancies = stitch_windows(windows, OVERLAP_ATOL,
                                               lower=4.15, upper=centers[-1])
    if len(all_modes) < MODE_COUNT:
        return None, None, {'status': 'insufficient_modes', 'n_candidates': len(all_modes)}

    roots = all_modes[:MODE_COUNT]
    eigenvalues = roots ** 2 / area
    fit = fit_trace(eigenvalues, TEST_TIME)
    strict = reconstruct(fit['area'], fit['perimeter'], fit['gamma'])
    constrained = constrained_reconstruct(eigenvalues, TEST_TIME, fit=fit)
    sides, angles = sorted_lengths_angles(vertices)

    # Diagnostic only: the same numerical spectrum on an auxiliary unit-area copy.
    normalized_truth = truth / [area, np.sqrt(area), 1.]
    normalized_fit = fit_trace(roots ** 2, AUDIT_TIME)
    normalized_trace = _true_short_time_trace(normalized_truth, AUDIT_TIME)
    test_trace = _true_short_time_trace(truth, TEST_TIME)
    # Moving mode counts reveal candidate count drift that may evade overlap checks.
    expected_count = (roots**2 - normalized_truth[1] * roots) / (4*np.pi) + truth[2]
    count_drift = np.arange(MODE_COUNT) + .5 - expected_count
    block_averages = np.mean(count_drift.reshape(60, 1000), axis=1)
    count_discrepancies = [x for x in discrepancies if x['counts'][0] != x['counts'][1]]
    overlap_range = [(float(x['left']), float(x['right']), list(x['counts']))
                     for x in count_discrepancies]
    row = {
        'dataset_index': index,
        'area': float(truth[0]),
        'true_perimeter': float(truth[1]),
        'true_gamma': float(truth[2]),
        'true_vertices': vertices.tolist(),
        'true_sides': sides.tolist(),
        'true_angles_degrees': np.degrees(angles).tolist(),
        'lambda_first': float(eigenvalues[0]),
        'lambda_last': float(eigenvalues[-1]),
        't_min_lambda_last': float(TEST_TIME[0] * eigenvalues[-1]),
        'polya_tail_upper_bound_at_t_min': _polya_tail_bound(TEST_TIME[0], area),
        'trace_at_t_min': float(fit['trace'][0]),
        'physical_trace_deficit_at_t_min_vs_true_three_terms': float(test_trace[0] - fit['trace'][0]),
        'fit': _fit_fields(fit, truth),
        'strict': reconstruction_metrics(strict, vertices, sides, angles),
        'constrained': reconstruction_metrics(constrained, vertices, sides, angles),
        'spectral_stitch': {
            **spectral_settings, 'radius': radius, 'windows': len(windows),
            'final_k': float(centers[-1]), 'candidate_count_through_final_k': len(all_modes),
            'overlap_disagreements': len(discrepancies),
            'overlap_count_disagreements': len(overlap_range),
            'overlap_count_disagreement_windows': overlap_range,
            'weyl_staircase_error_mean_per_1000_candidates': block_averages.tolist(),
            'weyl_staircase_max_absolute_1000_candidate_mean': float(np.max(np.abs(block_averages))),
            'max_relative_delta_vs_30_stored_fem': float(np.max(np.abs(reference_30 / eigenvalues[:30] - 1.))),
        },
        'normalized_spectrum_audit': {
            't_min_lambda_last': float(AUDIT_TIME[0] * roots[-1] ** 2),
            'polya_tail_upper_bound_at_t_min': _polya_tail_bound(AUDIT_TIME[0], 1.),
            'trace_linf_difference_vs_true_three_terms': float(np.max(np.abs(normalized_trace - normalized_fit['trace']))),
            'fit': _fit_fields(normalized_fit, normalized_truth),
        },
    }
    return row, eigenvalues, None


def run(dataset, folder, indices, overrides):
    with np.load(dataset, allow_pickle=False) as archive:
        vertices, spectra_30 = archive['perimeters'], archive['spectra']
    rows, spectra, pending = [], [], []
    for index in indices:
        path = overrides.get(index, folder / f'{index}.npz')
        if path.exists():
            row, values, status = evaluate(path, vertices[index], spectra_30[index], index)
        else:
            row, values, status = None, None, {'status': 'missing_checkpoint'}
        if row is None:
            pending.append({'dataset_index': index, **status})
        else:
            rows.append(row)
            spectra.append(values)

    ranked = sorted(rows, key=lambda r: r['area'])
    area_quartiles = [summarize([ranked[i] for i in sub])
                      for sub in np.array_split(np.arange(len(ranked)), 4) if len(sub)]
    audit = [row['normalized_spectrum_audit'] for row in rows]
    audit_summary = {} if not rows else {
        'median_trace_linf_difference_vs_true_three_terms': float(np.median(
            [a['trace_linf_difference_vs_true_three_terms'] for a in audit])),
        'max_trace_linf_difference_vs_true_three_terms': max(
            a['trace_linf_difference_vs_true_three_terms'] for a in audit),
        'median_absolute_gamma_error': float(np.median(
            [abs(a['fit']['invariant_errors'][2]) for a in audit])),
        'max_absolute_gamma_error': max(abs(a['fit']['invariant_errors'][2]) for a in audit),
        'median_relative_error_area_perimeter_gamma': np.median(
            [a['fit']['relative_invariant_errors'] for a in audit], axis=0).tolist(),
        'rows_with_overlap_count_disagreements': [r['dataset_index'] for r in rows
                                                 if r['spectral_stitch']['overlap_count_disagreements']],
    }
    result = {
        'protocol': {
            'dataset_file': dataset.name,
            'dataset_sha256': hashlib.sha256(dataset.read_bytes()).hexdigest(),
            'index_base': 0,
            'indices': indices,
            't_window': [float(TEST_TIME[0]), float(TEST_TIME[-1])],
            'time_points_geometric': len(TEST_TIME),
            'mode_count': MODE_COUNT,
            'condition': 'Dirichlet Laplacian',
            'spectral_solver': 'corner-adapted Fourier–Bessel windowed scaling with projected refinement',
            'ordering': 'ascending, retaining observed multiplicities',
            'overlap_wavenumber_atol_normalized': OVERLAP_ATOL,
            'audit_on_area_one': True,
            'audit_t_window': [float(AUDIT_TIME[0]), float(AUDIT_TIME[-1])],
            'target': 'mean relative vertex error <=2%, rigid alignment/reflection and six vertex permutations, preserving scale',
            'versions': {'python': platform.python_version(), 'numpy': np.__version__,
                         'scipy': scipy.__version__},
        },
        'n_completed': len(rows),
        'pending': pending,
        'summary': summarize(rows) if rows else None,
        'by_area_quartile_among_completed': area_quartiles,
        'normalized_spectrum_audit_summary': audit_summary,
        'cases': rows,
    }
    return result, np.array(spectra).reshape(len(rows), MODE_COUNT)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('checkpoints', type=Path)
    parser.add_argument('--indices', type=int, nargs='+', default=list(range(900, 1000)))
    parser.add_argument('--override', action='append', default=[], metavar='INDEX=CHECKPOINT')
    parser.add_argument('--control', action='append', default=[], metavar='FAMILY=CHECKPOINT')
    parser.add_argument('--results', type=Path)
    parser.add_argument('--spectra', type=Path)
    parser.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args()
    overrides = {}
    for pair in args.override:
        index, path = pair.split('=', 1)
        overrides[int(index)] = Path(path)
    outcome, spectra = run(args.dataset, args.checkpoints, args.indices, overrides)
    if args.control:
        outcome['solver_controls'] = {}
        for entry in args.control:
            family, path = entry.split('=', 1)
            outcome['solver_controls'][family] = validate_control(
                Path(path), family, 2.9715312024669283)
    if outcome['pending'] and not args.allow_partial:
        states = dict(Counter(row['status'] for row in outcome['pending']))
        parser.exit(2, f"{outcome['n_completed']}/{len(args.indices)} complete; pending {states}\n")
    if args.results:
        args.results.parent.mkdir(parents=True, exist_ok=True)
        tmp = args.results.with_suffix('.tmp')
        tmp.write_text(json.dumps(outcome, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        tmp.replace(args.results)
    if args.spectra:
        args.spectra.parent.mkdir(parents=True, exist_ok=True)
        tmp = args.spectra.with_suffix('.tmp')
        with tmp.open('wb') as output:
            np.savez_compressed(output, indices=np.array([r['dataset_index'] for r in outcome['cases']]),
                                spectra=spectra)
        tmp.replace(args.spectra)
    print(f"{outcome['n_completed']}/{len(args.indices)} complete")
    if outcome['summary']:
        for name in ('strict', 'constrained'):
            summ = outcome['summary'][name]
            print(name, f"feasible {summ['n_valid']}, passes {summ['n_pass_2pct']}, "
                        f"median {summ['median_relative_vertex_error_on_valid_only']}")


if __name__ == '__main__':
    main()
