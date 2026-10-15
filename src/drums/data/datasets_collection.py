import argparse
import os

os.environ["JAX_PLATFORMS"] = "cpu"  # must be set before jax is imported

import numpy as np

try:
    from drums.data import triangle_drums
except ImportError:
    import triangle_drums


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect a named dataset of drums.")
    parser.add_argument("dataset_name", choices=["triangle_drums"])
    parser.add_argument("save_here", help="folder to save the dataset in (created if missing)")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing dataset file")
    args = parser.parse_args()

    path = os.path.join(args.save_here, f"{args.dataset_name}.npz")
    if os.path.exists(path) and not args.overwrite:  # fail before the expensive computation
        parser.error(f"{path} already exists; use --overwrite to replace it")

    if args.dataset_name == "triangle_drums":
        n_samples = 1000
        data = triangle_drums.generate_triangle_drums(n_samples)

    os.makedirs(args.save_here, exist_ok=True)
    if os.path.exists(path) and not args.overwrite:  # created while we were computing
        parser.error(f"{path} appeared in the meantime; use --overwrite to replace it")
    np.savez_compressed(path, **data)
