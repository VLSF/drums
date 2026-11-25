"""Independent local frequency check by interior-normalized boundary tension."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import linalg, optimize

from bessel_table import BesselTable
from scaling import ScalingSolver


class Tension:
    def __init__(self, vertices, upper, seed=2841):
        self.solver = solver = ScalingSolver(vertices, basis_pad=18., point_factor=.82)
        basis_size = solver.basis_size(upper)
        solver.table = BesselTable(basis_size, solver.nu, upper * solver.radius)
        weights = np.random.default_rng(seed).dirichlet(np.ones(3), size=max(400, 2 * basis_size))
        internal = weights[:, :2] @ np.stack([solver.left, solver.right])
        self.radii = np.linalg.norm(internal, axis=1)
        self.angles = np.arctan2(internal[:, 1], internal[:, 0])

    def __call__(self, k):
        solver = self.solver
        boundary, _ = solver.evaluation(k)
        n = boundary.shape[0]
        interior, _ = solver.table.evaluate(k * self.radii, n)
        interior *= np.sin(solver.nu * np.arange(1, n+1)[:, None] * self.angles)
        interior /= np.sqrt(len(self.radii))
        combined = np.concatenate([boundary, interior], axis=1)
        q, r, _ = linalg.qr(combined.T, mode='economic', pivoting=True, check_finite=False)
        diag = np.abs(np.diag(r))
        rank = int(np.count_nonzero(diag > diag[0] * 1e-10))
        return float(linalg.svdvals(q[:boundary.shape[1], :rank], check_finite=False)[-1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('lower', type=float)
    parser.add_argument('upper', type=float)
    parser.add_argument('--step', type=float, default=.001)
    parser.add_argument('--seed', type=int, default=2841)
    args = parser.parse_args()
    with np.load(args.checkpoint, allow_pickle=False) as d:
        tension = Tension(d['vertices'], args.upper + 1., seed=args.seed)
    grid = np.arange(args.lower, args.upper+args.step/2, args.step)
    values = np.array([tension(k) for k in grid])
    minima = []
    for i in range(1, len(grid)-1):
        if values[i] <= values[i-1] and values[i] < values[i+1]:
            root = optimize.minimize_scalar(tension, bounds=(grid[i-1], grid[i+1]), method='bounded',
                                            options={'xatol': 5e-12})
            minima.append([float(root.x), float(root.fun)])
    report = {'grid': grid.tolist(), 'tensions': values.tolist(), 'minima': minima}
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(minima)


if __name__ == '__main__':
    main()
