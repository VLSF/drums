"""Collect the single shared dataset for the spectral ablation experiment."""
import argparse
from pathlib import Path

import numpy as np


def _load_generators():
    # Loading dataset helpers must not change the training process's JAX backend.
    try:
        from drums.data import triangle_drums, quadrilateral_drums
    except ImportError:
        import triangle_drums, quadrilateral_drums
    return triangle_drums, quadrilateral_drums


def triangle_as_quadrilateral(vertices):
    """Insert a longest-edge midpoint, preserving the exact triangular boundary."""
    vertices = np.asarray(vertices, dtype=float)
    edge = int(np.argmax(np.sum((np.roll(vertices, -1, axis=0) - vertices) ** 2, axis=1)))
    midpoint = (vertices[edge] + vertices[(edge + 1) % 3]) / 2
    return np.insert(vertices, edge + 1, midpoint, axis=0)


def generate_mixed_polygon_drums(n_triangles=1000, n_quadrilaterals=2000,
                                 seed=0, N_eig=30, nonconvex_fraction=0.5,
                                 n_elements=4000):
    """Return one shuffled dataset with a common four-point boundary representation.

    Exactly round(n_quadrilaterals * nonconvex_fraction) quadrilaterals are
    non-convex. Triangle spectra use their original three vertices; inserting
    a midpoint afterwards preserves their boundary without repeated vertices.
    The defaults give 1000 drums of each shape class. shape_label, n_vertices,
    and is_convex are evaluation labels, not conditioning inputs.
    """
    if min(n_triangles, n_quadrilaterals) < 0 or n_triangles + n_quadrilaterals == 0:
        raise ValueError("counts must be nonnegative, with at least one drum")
    if not 0 <= nonconvex_fraction <= 1 or N_eig < 1 or n_elements < 1:
        raise ValueError("fraction must be in [0, 1]; N_eig and n_elements must be positive")
    triangles, quadrilaterals = _load_generators()
    streams = np.random.SeedSequence(seed).spawn(4)
    seeds = [int(s.generate_state(1)[0]) for s in streams[:3]]
    n_nonconvex = int(round(n_quadrilaterals * nonconvex_fraction))
    counts = (n_triangles, n_quadrilaterals - n_nonconvex, n_nonconvex)
    pieces = []
    for group, (count, child_seed) in enumerate(zip(counts, seeds)):
        if not count:
            continue
        common = dict(seed=child_seed, N_eig=N_eig, n_elements=n_elements)
        if group == 0:
            data = triangles.generate_triangle_drums(count, **common)
            perimeter = np.stack([triangle_as_quadrilateral(p) for p in data["perimeters"]])
        else:
            data = quadrilaterals.generate_quadrilateral_drums(
                count, nonconvex_fraction=float(group == 2), **common)
            perimeter = data["perimeters"]
        pieces.append(dict(perimeters=perimeter, spectra=data["spectra"],
                           n_vertices=np.full(count, 3 if group == 0 else 4, dtype=np.int8),
                           is_convex=np.full(count, group != 2, dtype=bool),
                           shape_label=np.full(count, ("triangle", "convex_quad", "nonconvex_quad")[group])))
    order = np.random.default_rng(streams[3]).permutation(sum(counts))
    result = {name: np.concatenate([p[name] for p in pieces])[order]
              for name in ("perimeters", "spectra", "n_vertices", "is_convex", "shape_label")}
    result.update(sample_id=np.arange(sum(counts))[order], seed=np.asarray(seed),
                  n_elements=np.asarray(n_elements))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("save_here", type=Path)
    parser.add_argument("--n-triangles", type=int, default=1000)
    parser.add_argument("--n-quadrilaterals", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-eig", type=int, default=30)
    parser.add_argument("--n-elements", type=int, default=4000)
    parser.add_argument("--nonconvex-fraction", type=float, default=.5)
    args = parser.parse_args(argv)
    path = args.save_here / "mixed_polygon_drums.npz"
    if path.exists():
        parser.error(f"{path} already exists; choose a new folder")
    data = generate_mixed_polygon_drums(args.n_triangles, args.n_quadrilaterals,
                                       args.seed, args.n_eig, args.nonconvex_fraction,
                                       args.n_elements)
    args.save_here.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        np.savez_compressed(handle, **data)
    print(f"Saved {len(data['spectra'])} drums to {path}", flush=True)


if __name__ == "__main__":
    main()
