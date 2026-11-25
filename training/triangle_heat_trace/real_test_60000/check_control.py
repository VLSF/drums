"""Compare a completed numerical control with its analytic Dirichlet spectrum."""

import argparse
import json
from pathlib import Path

import numpy as np

from check_spectra import compare_spectra, stitch_windows
from experiment import FAMILIES, exact_spectrum


def validate(path, family, area):
    with np.load(path, allow_pickle=False) as checkpoint:
        if not bool(checkpoint["finished"]):
            raise ValueError(f"The control scan is incomplete: {path}")
        if not np.array_equal(checkpoint["vertices"], FAMILIES[family]):
            raise ValueError(f"Checkpoint {path} does not contain the requested template")
        centers = checkpoint["centers"]
        offsets = checkpoint["offsets"]
        values = checkpoint["values"]
        radius = float(checkpoint["radius"])
        template_area = float(checkpoint["area"])
    windows = [(center, radius, values[low:high])
               for center, low, high in zip(centers, offsets[:-1], offsets[1:])]
    computed, overlap_issues = stitch_windows(windows, .0005, lower=4.15, upper=centers[-1])
    exact = np.sqrt(template_area*exact_spectrum(family, centers[-1]**2/template_area))
    comparison = {
        "family": family,
        "normalized_frequency_cutoff": float(centers[-1]),
        "computed_count": len(computed),
        "exact_count": len(exact),
        "overlap_disagreements": len(overlap_issues),
        "overlap_count_disagreements": sum(item["counts"][0] != item["counts"][1]
                                         for item in overlap_issues),
        "physical_area": area,
    }
    if min(len(computed), len(exact)) >= 60_000:
        comparison.update(compare_spectra(exact[:60_000]**2/area, computed[:60_000]**2/area))
    return comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("family", choices=list(FAMILIES))
    parser.add_argument("--area", type=float, default=2.9715312024669283)
    args = parser.parse_args()
    if args.area <= 0:
        parser.error("--area must be positive")
    print(json.dumps(validate(args.checkpoint, args.family, args.area), indent=2))


if __name__ == "__main__":
    main()
