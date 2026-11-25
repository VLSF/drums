"""Plot the pass counts and area dependence from a triangle benchmark result."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


FAMILIES = {
    "equilateral": ("Equilateral", "#326694", "o"),
    "right_isosceles": ("Right isosceles", "#a34338", "s"),
    "30_60_90": ("30°–60°–90°", "#167368", "^"),
}


def plot(results, output):
    held_out = results["held_out"]
    figure, (passes_axis, error_axis) = plt.subplots(
        1, 2, figsize=(11.2, 4.5), layout="constrained",
        gridspec_kw={"width_ratios": [1, 1.35]},
    )
    mode_count = results["protocol"]["mode_count"]
    figure.suptitle(f"Triangle reconstruction with {mode_count:,} Dirichlet modes", fontsize=15)

    positions = list(range(len(FAMILIES)))
    for method, label, offset, color in (
        ("strict", "Strict", -0.18, "#596784"),
        ("constrained", "Geometry-constrained", 0.18, "#337e9f"),
    ):
        counts = [held_out["by_family"][family][method]["n_pass_2pct"] for family in FAMILIES]
        bars = passes_axis.bar([i + offset for i in positions], counts, 0.34,
                               label=label, color=color)
        passes_axis.bar_label(bars, padding=3, fontsize=9)
    passes_axis.set_xticks(positions, [label for label, _, _ in FAMILIES.values()])
    passes_axis.set_ylim(0, 113)
    passes_axis.set_ylabel("Cases within 2% (out of 100 per shape)")
    passes_axis.set_title("Reconstruction accuracy")
    passes_axis.legend(loc="upper right", frameon=False)
    passes_axis.grid(axis="y", alpha=0.22)
    passes_axis.set_axisbelow(True)

    for family, (label, color, marker) in FAMILIES.items():
        cases = [case for case in held_out["cases"]
                 if case["family"] == family and case["constrained"]["status"] == "ok"]
        error_axis.scatter(
            [case["area"] for case in cases],
            [100 * case["constrained"]["relative_vertex_error"] for case in cases],
            label=label, color=color, marker=marker, s=17, alpha=0.75, linewidths=0,
        )
    error_axis.axhline(2, color="#374151", linestyle="--", linewidth=1, label="2% target")
    error_axis.set_ylim(bottom=-0.9)
    error_axis.set_xlabel("Triangle area")
    error_axis.set_ylabel("Mean relative vertex error (%)")
    error_axis.set_title("Geometry-constrained error by area")
    error_axis.grid(alpha=0.22)
    error_axis.set_axisbelow(True)
    error_axis.legend(loc="upper left", frameon=False)

    for axis in (passes_axis, error_axis):
        axis.spines[["top", "right"]].set_visible(False)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    plot(json.loads(args.results.read_text(encoding="utf-8")), args.output)


if __name__ == "__main__":
    main()
