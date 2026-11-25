"""Replace narrow spectral bands using independent boundary-tension minima."""

import argparse
import json
from pathlib import Path

import numpy as np

from repair_checkpoint import stitch_checkpoint, windows_from


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('probes', type=Path, nargs='+')
    parser.add_argument('--provenance', type=Path)
    args = parser.parse_args()
    if args.checkpoint.resolve() == args.output.resolve():
        parser.error('Choose a separate output checkpoint')
    with np.load(args.checkpoint, allow_pickle=False) as source:
        original = {k: source[k] for k in source.files}
    windows = windows_from(original)
    _, before = stitch_checkpoint(windows)
    audit = []
    for report in args.probes:
        data = json.loads(report.read_text(encoding='utf8'))
        low, high = data['grid'][0], data['grid'][-1]
        minima = np.asarray(data['minima'], dtype=float).reshape(-1, 2)
        candidates = minima[:, 0]
        if np.any((candidates <= low) | (candidates >= high)) or np.any(~np.isfinite(minima)):
            raise ValueError(f'Invalid boundary-tension minima in {report}')
        for i, (center, radius, roots) in enumerate(windows):
            if center+radius <= low or center-radius >= high:
                continue
            replace = (candidates >= center-radius) & (candidates <= center+radius)
            preserve = (roots <= low) | (roots >= high)
            windows[i] = (center, radius, np.sort(np.r_[roots[preserve], candidates[replace]]))
        audit.append({'probe': str(report), 'band': [low, high], 'minima': minima.tolist()})
    _, after = stitch_checkpoint(windows)
    original['offsets'] = np.r_[0, np.cumsum([len(w[2]) for w in windows])]
    original['values'] = np.concatenate([w[2] for w in windows])
    original['interior_normalized_mps_band_audits'] = np.asarray(json.dumps(audit))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix('.tmp.npz')
    np.savez(temp, **original)
    temp.replace(args.output)
    if args.provenance:
        provenance = {'input_checkpoint':str(args.checkpoint), 'output_checkpoint':str(args.output),
                      'baseline':before, 'repaired':after, 'mps_bands':audit}
        args.provenance.write_text(json.dumps(provenance, indent=2) + '\n', encoding='utf8')


if __name__ == '__main__':
    main()
