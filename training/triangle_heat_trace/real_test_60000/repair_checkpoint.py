"""Independently rescan selected bands of a saved Fourier–Bessel spectrum."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from bessel_table import BesselTable
from check_spectra import stitch_windows
from robust import window
from scaling import ScalingSolver


def windows_from(data):
    centers, offsets, values, radius = (data[x] for x in ('centers', 'offsets', 'values', 'radius'))
    radii = data['radii'] if 'radii' in data else np.full(len(centers), radius, dtype=float)
    return [(c, float(r), values[a:b]) for c, r, a, b in zip(centers, radii, offsets[:-1], offsets[1:])]


def stitch_checkpoint(windows):
    roots, discrepancies = stitch_windows(windows, .0005, lower=4.15, upper=windows[-1][0])
    count_discrepancies = [[float(d['left']), float(d['right']), list(d['counts'])]
                           for d in discrepancies if d['counts'][0] != d['counts'][1]]
    return roots, {
        'candidate_count_through_final_k': len(roots),
        'overlap_disagreements': len(discrepancies),
        'overlap_count_disagreements': len(count_discrepancies),
        'overlap_count_disagreement_windows': count_discrepancies,
    }


def rebuild(solver, low, high, step=.17, phase=.37):
    radius = .5 * step + .043
    start = low - (1. - phase) * step
    centers = np.arange(start, high+step, step)
    windows = []
    unresolved = 0
    for k in centers:
        candidates, flagged, _ = window(solver, k, radius)
        unresolved += flagged
        windows.append((k, radius, candidates))
    eigenfrequencies, checks = stitch_windows(windows, .0005, lower=centers[0], upper=centers[-1])
    count_disagreements = sum(d['counts'][0] != d['counts'][1] for d in checks)
    return eigenfrequencies, [unresolved, len(checks), count_disagreements]


def replace(windows, selected, roots):
    for index in selected:
        k, radius, _ = windows[index]
        windows[index] = (k, radius, roots[(roots >= k-radius) & (roots <= k+radius)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--low-scan', type=Path)
    parser.add_argument('--bands', nargs='*', default=[], metavar='LO:HI')
    parser.add_argument('--step', type=float, default=.17)
    parser.add_argument('--phase', type=float, default=.37)
    parser.add_argument('--provenance', type=Path)
    args = parser.parse_args()
    started = perf_counter()
    with np.load(args.checkpoint, allow_pickle=False) as source:
        data = {k: source[k] for k in source.files}
    if args.checkpoint.resolve() == args.output.resolve():
        parser.error('Choose a separate output to preserve the original checkpoint')
    if 'finished' in data and not bool(data['finished']):
        parser.error('The baseline checkpoint is incomplete')
    windows = windows_from(data)
    _, baseline = stitch_checkpoint(windows)
    centers = data['centers']
    radius = max(r for _, r, _ in windows)
    reports = []
    independent_low_scan_seconds = 0.
    if args.low_scan:
        low_data = np.load(args.low_scan, allow_pickle=False)
        if not np.array_equal(low_data['vertices'], data['vertices']):
            raise ValueError('Low scan uses different triangle vertices')
        low_ks, low_audit = stitch_checkpoint(windows_from(low_data))
        selected = np.flatnonzero(centers < low_data['centers'][-1] - 2 * radius)
        if not len(selected):
            parser.error('The low scan does not cover any baseline window with margin')
        replace(windows, selected, low_ks)
        low_counts = low_audit['overlap_count_disagreements']
        reports.append([4.15, float(centers[selected[-1]]), len(low_data.get('flagged_windows', [])),
                        low_audit['overlap_disagreements'], low_counts])
        independent_low_scan_seconds = float(low_data['seconds'])
        print(f'Low scan replaced {len(selected)} windows with {low_counts} overlap count discrepancies', flush=True)
    if args.bands:
        solver = ScalingSolver(data['vertices'], basis_pad=14.)
        max_k = max(float(b.split(':')[1]) for b in args.bands) + 3
        solver.table = BesselTable(solver.basis_size(max_k), solver.nu, max_k * solver.radius)
        for band in args.bands:
            low, high = (float(q) for q in band.split(':'))
            selected = np.flatnonzero((centers >= low) & (centers <= high))
            if not len(selected):
                parser.error(f'The band {band} does not cover a baseline window')
            roots, audit = rebuild(solver, low - radius, high + radius, args.step, args.phase)
            replace(windows, selected, roots)
            reports.append([low, high, *audit])
            print(f'{band}: {len(roots)} candidate roots, {len(selected)} replaced windows, {audit}', flush=True)
    offsets = np.r_[0, np.cumsum([len(w[2]) for w in windows])]
    _, after = stitch_checkpoint(windows)
    columns = ('normalized_k_low', 'normalized_k_high', 'local_flags',
               'overlap_disagreements', 'overlap_count_disagreements')
    original_discrepancies = baseline['overlap_count_disagreement_windows']
    data['baseline_overlap_disagreements'] = baseline['overlap_disagreements']
    data['baseline_overlap_count_disagreement_centers'] = np.asarray(
        [row[:2] for row in original_discrepancies]).reshape(-1, 2)
    data['baseline_overlap_count_disagreement_counts'] = np.asarray(
        [row[2] for row in original_discrepancies], dtype=int).reshape(-1, 2)
    if 'flagged_windows' in data:
        data['flagged_windows_original_baseline'] = data['flagged_windows']
        data['flagged_windows_provenance'] = np.asarray('Baseline solver flags before targeted band replacement')
    repair_time = perf_counter()-started
    data.update(offsets=offsets, values=np.concatenate([w[2] for w in windows]),
                targeted_repair_full_band_audits=np.asarray(reports).reshape(-1, len(columns)),
                targeted_repair_audit_columns=np.asarray(columns),
                baseline_elapsed_seconds=float(data['seconds']),
                targeted_repair_elapsed_seconds=repair_time,
                targeted_repair_low_scan_seconds=independent_low_scan_seconds,
                seconds=float(data['seconds']) + independent_low_scan_seconds + repair_time)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix('.tmp.npz')
    np.savez(temporary, **data)
    temporary.replace(args.output)
    if args.provenance:
        provenance = {
            'baseline_checkpoint': str(args.checkpoint),
            'output_checkpoint': str(args.output),
            'independent_low_scan': str(args.low_scan) if args.low_scan else None,
            'candidate_replacement': 'independently recomputed modes in each selected baseline window',
            'baseline': baseline, 'repaired': after,
            'repair_band_audit_columns': columns, 'repair_band_audits': reports,
            'independent_band_step': args.step, 'independent_band_phase': args.phase,
        }
        args.provenance.parent.mkdir(parents=True, exist_ok=True)
        args.provenance.write_text(json.dumps(provenance, indent=2) + '\n', encoding='utf8')


if __name__ == '__main__':
    main()
