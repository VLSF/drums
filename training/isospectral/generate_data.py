"""Generate 30k isospectral drums (drums.data.isospectral_drums) in parallel, chunks with independent seeds.

    python generate_data.py --n 30000 --workers 16 --out isospectral_30k.npz
"""
import argparse
import os
os.environ.update(JAX_PLATFORMS="cpu", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
import time
from multiprocessing import get_context

import numpy as np


def chunk(args):
    seed, n = args
    from drums.data.isospectral_drums import generate_isospectral_drums
    return generate_isospectral_drums(n, seed=seed)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=30000)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--chunk", type=int, default=100)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--out", default="isospectral_30k.npz")
    a = p.parse_args()
    jobs = [(a.seed * 100000 + i, a.chunk) for i in range(a.n // a.chunk)]
    t = time.time()
    with get_context("spawn").Pool(a.workers) as pool:
        res = []
        for i, r in enumerate(pool.imap(chunk, jobs)):
            res.append(r)
            if i % 20 == 0:
                print(f"{i + 1}/{len(jobs)} chunks, {time.time() - t:.0f}s", flush=True)
    np.savez(a.out, **{k: np.concatenate([r[k] for r in res]) for k in res[0]})
    print("done", time.time() - t, "s")
