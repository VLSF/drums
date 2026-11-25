"""Compute Dirichlet spectra for rows 900–999 of triangle_drums.npz."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter

import numpy as np

from scan_robust import solve


def solve_row(index, vertices, checkpoints):
    path = checkpoints / f"{index}.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as existing:
            if bool(existing["finished"]):
                if not np.array_equal(existing["vertices"], vertices):
                    raise ValueError(f"Checkpoint {path} belongs to another geometry")
                return index, float(existing["seconds"])
    started = perf_counter()
    solve(vertices, path)
    return index, perf_counter() - started


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("checkpoints", type=Path)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--indices", nargs="+", type=int, default=list(range(900, 1000)))
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    with np.load(args.dataset, allow_pickle=False) as dataset:
        vertices = dataset["perimeters"]
    if (len(set(args.indices)) != len(args.indices)
            or any(i < 0 or i >= len(vertices) for i in args.indices)):
        parser.error("Expected unique, valid zero-based dataset indices")
    args.checkpoints.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = [pool.submit(solve_row, index, vertices[index], args.checkpoints)
                for index in args.indices]
        for done, job in enumerate(as_completed(jobs), 1):
            index, seconds = job.result()
            print(f"Row {index} completed in {seconds:.1f}s. {done}/{len(jobs)} finished, "
                  f"{perf_counter()-started:.1f}s elapsed", flush=True)


if __name__ == "__main__":
    main()
