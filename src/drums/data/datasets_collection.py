import argparse
import os

os.environ["JAX_PLATFORMS"] = "cpu"  # must be set before jax is imported

import numpy as np

try:
    from drums.data import triangle_drums, quadrilateral_drums, gww_drums
except ImportError:
    import triangle_drums, quadrilateral_drums, gww_drums


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect a named dataset of drums.")
    parser.add_argument("dataset_name", choices=["triangle_drums", "convex_quadrilateral_drums", "mixed_quadrilateral_drums", "gww_drums"])
    parser.add_argument("save_here", help="folder to save the dataset in (created if missing)")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing dataset file")
    args = parser.parse_args()

    path = os.path.join(args.save_here, f"{args.dataset_name}.npz")
    if os.path.exists(path) and not args.overwrite:  # fail before the expensive computation
        parser.error(f"{path} already exists; use --overwrite to replace it")

    if args.dataset_name == "triangle_drums":
        n_samples = 1000
        data = triangle_drums.generate_triangle_drums(n_samples)
    elif args.dataset_name == "convex_quadrilateral_drums":
        n_samples = 1000
        data = quadrilateral_drums.generate_quadrilateral_drums(n_samples, nonconvex_fraction=0.0)
    elif args.dataset_name == "mixed_quadrilateral_drums":
        n_samples = 1000
        data = quadrilateral_drums.generate_quadrilateral_drums(n_samples, nonconvex_fraction=0.5)
    elif args.dataset_name == "gww_drums":
        n_samples = 1000
        data = gww_drums.generate_gww_drums(n_samples, delta=0.1)

    os.makedirs(args.save_here, exist_ok=True)
    if os.path.exists(path) and not args.overwrite:  # created while we were computing
        parser.error(f"{path} appeared in the meantime; use --overwrite to replace it")
    np.savez_compressed(path, **data)
