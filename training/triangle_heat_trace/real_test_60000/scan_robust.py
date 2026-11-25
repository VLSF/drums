"""Solve overlapping Fourier–Bessel windows with local recentering."""

from pathlib import Path
import resource
import sys
from time import perf_counter

import numpy as np

from bessel_table import BesselTable
from check_spectra import stitch_windows
from robust import window
from scaling import ScalingSolver


def solve(vertices, path, step=.44, phase=0., stop=877., pad=12., verbose=False):
    start = perf_counter()
    solver = ScalingSolver(vertices, basis_pad=pad)
    radius = .67*step
    centers = np.arange(4.15 + phase*step, stop+step, step)
    k_max = centers[-1]+radius
    solver.table = BesselTable(solver.basis_size(k_max), solver.nu, k_max*solver.radius)
    data = []
    n_repairs = 0
    flagged = []
    path = Path(path)

    def save(finished):
        temp = path.with_suffix(".tmp.npz")
        offsets = np.r_[0, np.cumsum([len(values) for values in data])]
        peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform != "darwin":
            peak_rss *= 1024
        np.savez(temp, area=solver.area, vertices=vertices, centers=centers[:len(data)],
                 radius=radius, offsets=offsets, values=np.concatenate(data),
                 step=step, phase=phase, pad=pad, seconds=perf_counter()-start,
                 repairs=n_repairs, flagged_windows=np.asarray(flagged), finished=finished,
                 max_rss_bytes=peak_rss)
        temp.replace(path)

    for i, k in enumerate(centers):
        result, unresolved, repairs = window(solver, k, radius)
        data.append(result)
        n_repairs += repairs
        if unresolved:
            flagged.append(float(k))
        if verbose and i % 150 == 0:
            print(f"k={k:.2f}, {i+1}/{len(centers)}, repairs {n_repairs}, "
                  f"{perf_counter()-start:.1f} s", flush=True)
        if i % 250 == 0 and i < len(centers)-1:
            save(False)
    index_by_center = dict(zip(centers, range(len(centers))))
    for _ in range(2):
        windows = [(k, radius, values) for k, values in zip(centers, data)]
        _, disagreements = stitch_windows(windows, .0005, lower=4.15, upper=centers[-1])
        suspect = {index_by_center[c] for issue in disagreements
                   for c in (issue["left"], issue["right"])}
        if not suspect:
            break
        for index in sorted(suspect):
            result, unresolved, repairs = window(solver, centers[index], radius, force=True)
            data[index] = result
            n_repairs += repairs
            if unresolved:
                flagged.append(float(centers[index]))
    save(True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--index", type=int, default=902)
    parser.add_argument("--control", choices=["equilateral", "right_isosceles", "30_60_90"])
    parser.add_argument("--stop", type=float, default=877.)
    parser.add_argument("--step", type=float, default=.44)
    parser.add_argument("--phase", type=float, default=0.)
    parser.add_argument("--pad", type=float, default=12.)
    arguments = parser.parse_args()
    if arguments.control:
        from experiment import FAMILIES
        vertices = FAMILIES[arguments.control]
    else:
        with np.load(arguments.dataset, allow_pickle=False) as data:
            vertices = data["perimeters"][arguments.index]
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    solve(vertices, arguments.output, stop=arguments.stop, step=arguments.step, phase=arguments.phase,
          pad=arguments.pad, verbose=True)
