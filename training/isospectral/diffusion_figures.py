"""Plot saved Conway diffusion evaluations against historical FM/regression results.

    python training/isospectral/diffusion_figures.py \
        --run results/ddpm_main_ratio --run results/ddpm_paired_ratio \
        --evaluation-prefix test_ddim200_raw_eta0 --out results/diffusion_figures

This script reads saved evaluations only. Example rows are chosen by the ranked
truth/twin separation, and always display the first draws without quality selection.
"""

import argparse
import csv
import json
import math
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PINK, GREY, BLUE = "#f4a6b8", "#aaaaaa", "#4c72b0"
METRICS = (("sharp", "Sharp draws"), ("blurry", "Blurry draws"),
           ("mean_set_over_sep", "Mean e_set / separation"),
           ("both_modes", "Both modes among draws"))


def clean_json(value):
    if isinstance(value, dict):
        return {key: clean_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean_json(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def result_block(summary):
    blocks = [value for value in summary.values()
              if isinstance(value, dict) and "draws" in value and "mode_balance" in value]
    if len(blocks) != 1:
        raise ValueError("Expected exactly one evaluation block containing draws and mode_balance")
    return blocks[0]


def comparison_row(label, source, result, config, evaluation, path):
    draws = result.get("draws", result)
    mode = result.get("mode_balance", {})
    return dict(
        label=label, source=source, summary_file=str(path),
        evaluation_prefix=path.name.removesuffix("_summary.json") if path.name.endswith("_summary.json") else path.stem,
        arm="paired" if config.get("paired", False) else "main",
        features=config.get("features"), split=evaluation.get("split", "test"),
        sampler=evaluation.get("sampler"), sampling_steps=evaluation.get("sampling_steps"),
        weights=evaluation.get("weights"), seed=config.get("seed"),
        completed_updates=evaluation.get("completed_updates", config.get("n_updates")),
        n_eval=evaluation.get("n_eval"), n_samples=evaluation.get("n_samples", 1),
        error=draws.get("error"), sharp=draws.get("sharp"), blurry=draws.get("blurry"),
        mean_set_over_sep=draws.get("mean_set_over_sep"),
        both_modes=mode.get("both_modes"),
        mean_fraction_nearer_truth=mode.get("mean_fraction_nearer_truth"),
        std_fraction_nearer_truth=mode.get("std_over_drums"),
        legacy_energy_diagnostic=result.get("energy_score"),
    )


def readable_label(config, evaluation):
    arm = "Paired" if config.get("paired", False) else "Main"
    features = str(config.get("features", "spectral features")).replace("_", " ")
    completed = evaluation.get("completed_updates", config.get("n_updates"))
    updates = f"{completed:,} updates" if completed is not None else "saved checkpoint"
    split = str(evaluation.get("split", "evaluation")).title()
    sampler = str(evaluation.get("sampler", "DDPM")).upper()
    steps = evaluation.get("sampling_steps")
    sampling = f"{sampler} {steps:,}" if steps is not None else sampler
    if sampler == "DDIM" and evaluation.get("eta") is not None:
        sampling += f" (η={evaluation['eta']:g})"
    weights = "EMA" if evaluation.get("weights") == "ema" else "Raw weights"
    return (f"Conditional diffusion · {arm} · {features} · {updates}\n"
            f"{split} · {sampling} · {weights}")


def historical_rows(baseline_steps=200):
    root = Path(__file__).resolve().parent / "summary"
    rows = []
    for arm in ("main", "paired"):
        for features in ("ratio", "ratio_white"):
            path = root / f"fm_{arm}_{features}.json"
            summary = json.loads(path.read_text())
            config = summary["args"]
            evaluation = dict(split="test", sampler="Euler FM", sampling_steps=baseline_steps,
                              weights="raw", n_eval=config["n_eval"], n_samples=config["n_samples"])
            rows.append(comparison_row(
                f"FM {arm}, {features.replace('_', ' ')}\nhistorical, Euler {baseline_steps}", "historical",
                summary[f"test_steps{baseline_steps}"], config, evaluation, path,
            ))
        path = root / f"reg_{arm}_ratio_squared.json"
        summary = json.loads(path.read_text())
        config = dict(summary["args"], paired=arm == "paired")
        rows.append(comparison_row(
            f"Regression {arm}, squared\nhistorical, ratio", "historical",
            summary["test"], config, dict(split="test", sampler="regression", n_samples=1), path,
        ))
    return rows


def load_evaluations(folders, prefix=None):
    evaluations = []
    for folder in folders:
        paths = ([folder / f"{prefix}_summary.json"] if prefix
                 else sorted(folder.glob("*_summary.json")))
        if not paths:
            raise FileNotFoundError(f"No saved evaluation summaries in {folder}")
        for path in paths:
            summary = json.loads(path.read_text())
            name = path.name.removesuffix("_summary.json")
            with np.load(folder / f"{name}_samples.npz") as data:
                arrays = {key: data[key] for key in ("samples", "truth", "twin")}
            with np.load(folder / f"{name}_per_draw.npz") as data:
                per_draw = {key: data[key] for key in ("e_truth", "e_twin", "sep")}
            samples = arrays["samples"]
            n, m = samples.shape[:2]
            if (samples.shape[2:] != (9, 2) or arrays["truth"].shape != (n, 9, 2)
                    or arrays["twin"].shape != (n, 9, 2)
                    or any(value.size != n*m for value in per_draw.values())):
                raise ValueError(f"Mismatched saved samples and metrics: {path}")
            per_draw = {key: value.reshape(n, m) for key, value in per_draw.items()}
            block = result_block(summary)
            evaluation = summary["evaluation"]
            label = readable_label(summary["config"], evaluation)
            row = comparison_row(label, "diffusion", block, summary["config"], evaluation, path)
            evaluations.append(dict(label=label, prefix=name, folder=folder, summary=summary,
                                    result=block, arrays=arrays, per_draw=per_draw, row=row))
    return evaluations


def comparison_figure(rows, out):
    fig, axes = plt.subplots(1, 4, figsize=(17, max(5, 0.44 * len(rows) + 1.4)))
    labels = [row["label"] for row in rows]
    y = np.arange(len(rows))
    for ax, (metric, title) in zip(axes, METRICS):
        values = np.array([np.nan if row[metric] is None else row[metric] for row in rows], dtype=float)
        colors = [BLUE if row["source"] == "diffusion" else GREY for row in rows]
        ax.barh(y, values, color=colors, edgecolor="white")
        ax.set_yticks(y, labels if ax is axes[0] else [""] * len(rows), fontsize=8)
        ax.set_ylim(len(rows)-0.5, -0.5)
        ax.set_title(title, fontsize=11)
        ax.grid(axis="x", alpha=0.2)
        ax.set_axisbelow(True)
        if metric != "mean_set_over_sep":
            ax.set_xlim(0, 1.12)
        else:
            finite = values[np.isfinite(values)]
            ax.set_xlim(0, max(float(finite.max()) * 1.22, 0.1) if finite.size else 1.)
        for i, value in enumerate(values):
            if np.isfinite(value):
                ax.text(value, i, f" {value:.3f}", va="center", fontsize=8)
            else:
                ax.text(0.02, i, "not measured", va="center", color="0.45", fontsize=8)
    fig.suptitle("Conway inverse problem: saved diffusion evaluations and historical baselines", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out / "comparison.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def distribution_figures(evaluations, out):
    columns = min(3, len(evaluations))
    rows = math.ceil(len(evaluations) / columns)
    for kind in ("sharpness", "mode_balance"):
        fig, axes = plt.subplots(rows, columns, squeeze=False,
                                 figsize=(5 * columns, 3.5 * rows))
        for ax, evaluation in zip(axes.flat, evaluations):
            per_draw = evaluation["per_draw"]
            separated = per_draw["sep"][:, 0] > evaluation["result"].get("sep_min", 0.1)
            if kind == "sharpness":
                ratio = (np.minimum(per_draw["e_truth"], per_draw["e_twin"])[separated]
                         / per_draw["sep"][separated]).ravel()
                finite = ratio[np.isfinite(ratio)]
                if len(finite):
                    ax.hist(np.clip(finite, 0, 1.2-1e-9), bins=48, range=(0, 1.2),
                            weights=np.ones(len(finite)) / len(finite), color=BLUE, edgecolor="white")
                ax.axvline(0.25, color="k", ls=":", lw=1)
                ax.axvline(0.40, color="k", ls=":", lw=1)
                ax.set_xlabel("e_set / separation (right bin includes ≥1.2)")
                ax.set_ylabel("Fraction of finite separated draws")
            else:
                m = per_draw["e_truth"].shape[1]
                fractions = (per_draw["e_truth"] < per_draw["e_twin"]).mean(1)[separated]
                bins = (np.arange(m + 2) - 0.5) / m
                if len(fractions):
                    ax.hist(fractions, bins=bins, weights=np.ones(len(fractions)) / len(fractions),
                            color=BLUE, edgecolor="white", label="Saved draws")
                pmf = np.array([math.comb(m, k) / 2**m for k in range(m + 1)])
                ax.plot(np.arange(m+1) / m, pmf, "k.-", lw=1, ms=3, label=f"Binomial({m}, 0.5)")
                ax.set_xlabel("Fraction of draws nearer the labelled truth")
                ax.set_ylabel("Fraction of separated spectra")
                ax.legend(fontsize=8)
            ax.set_title(evaluation["label"], fontsize=9)
        for ax in list(axes.flat)[len(evaluations):]:
            ax.set_visible(False)
        fig.tight_layout()
        fig.savefig(out / f"{kind}.png", dpi=160, bbox_inches="tight")
        plt.close(fig)


def best_align(reference, prediction):
    """NumPy O(2)/D9 alignment; plotting only, with no scale fitting."""
    if not np.isfinite(prediction).all():
        return np.full_like(prediction, np.nan, dtype=float)
    reference = reference - reference.mean(0, keepdims=True)
    prediction = prediction - prediction.mean(0, keepdims=True)
    n = len(prediction)
    permutations = np.array([np.roll(np.arange(n)[::s], k)
                             for k in range(n) for s in (1, -1)])
    ordered = prediction[permutations]
    u, _, vt = np.linalg.svd(ordered.swapaxes(-1, -2) @ reference, full_matrices=False)
    candidates = ordered @ (u @ vt)
    return candidates[np.argmin(np.sum((candidates-reference)**2, axis=(1, 2)))]


def example_figure(evaluation, out):
    arrays = evaluation["arrays"]
    separation = evaluation["per_draw"]["sep"][:, 0]
    threshold = evaluation["result"].get("sep_min", 0.1)
    eligible = np.flatnonzero(np.isfinite(separation) & (separation > threshold))
    if not len(eligible):
        raise ValueError(f"No example pairs pass separation > {threshold}: {evaluation['prefix']}")
    ordered = eligible[np.argsort(-separation[eligible], kind="stable")]
    ranks = np.unique(np.rint(np.array([0.1, 0.5, 0.9]) * (len(ordered)-1)).astype(int))
    indices = ordered[ranks]
    n_draws = min(4, arrays["samples"].shape[1])
    fig, axes = plt.subplots(len(indices), n_draws+1, squeeze=False,
                             figsize=(3 * (n_draws+1), 2.7 * len(indices) + 0.6),
                             layout="compressed")
    for row, (index, rank) in enumerate(zip(indices, ranks)):
        truth = np.asarray(arrays["truth"][index], dtype=float)
        truth = truth - truth.mean(0, keepdims=True)
        twin = best_align(truth, arrays["twin"][index])
        draws = [best_align(truth, draw) for draw in arrays["samples"][index, :n_draws]]
        all_points = np.concatenate([truth, twin, *[draw for draw in draws if np.isfinite(draw).all()]])
        low, high = all_points.min(0), all_points.max(0)
        middle, half = (low+high)/2, max(float((high-low).max()) * 0.56, 1e-6)
        for column, ax in enumerate(axes[row]):
            for shape, color in ((twin, GREY), (truth, PINK)):
                ax.fill(shape[:, 0], shape[:, 1], facecolor=color, edgecolor="k",
                        linewidth=0.8, alpha=0.65 if column == 0 else 0.25)
            if column:
                shape = draws[column-1]
                if np.isfinite(shape).all():
                    ax.fill(shape[:, 0], shape[:, 1], facecolor=BLUE, edgecolor="k",
                            linewidth=0.8, alpha=0.65)
                else:
                    ax.text(0.5, 0.5, "Nonfinite draw", transform=ax.transAxes,
                            color=BLUE, ha="center", va="center")
            ax.set_aspect("equal")
            ax.set_xlim(middle[0]-half, middle[0]+half)
            ax.set_ylim(middle[1]-half, middle[1]+half)
            ax.set_xticks([])
            ax.set_yticks([])
        axes[row, 0].set_ylabel(f"Evaluation row {index}\neligible rank {rank}/{len(ordered)}\nsep={separation[index]:.3f}", fontsize=8)
    for column, ax in enumerate(axes[0]):
        ax.set_title("Truth (pink), twin (grey)" if column == 0 else f"Draw {column-1} (blue)", fontsize=9)
    fig.suptitle(evaluation["label"], fontsize=12)
    fig.supxlabel(f"Rows follow fixed separation ranks among pairs with sep > {threshold:g}; first {n_draws} draws shown.", fontsize=9)
    filename = re.sub(r"[^A-Za-z0-9_.-]+", "_", evaluation["folder"].name + "_" + evaluation["prefix"])
    fig.savefig(out / f"examples_{filename}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    config, settings = evaluation["summary"]["config"], evaluation["summary"]["evaluation"]
    split = settings["split"]
    offset = {"train": 0, "validation": config["n_train"],
              "test": config["n_train"] + config["n_val"]}[split]
    all_ranks = np.empty(len(separation), dtype=int)
    all_ranks[np.argsort(-separation, kind="stable")] = np.arange(len(separation))
    return dict(run=str(evaluation["folder"]), evaluation_prefix=evaluation["prefix"],
                eligibility_threshold=threshold, eligible_spectrum_count=int(len(eligible)),
                evaluation_spectrum_count=int(len(separation)), rank_population="eligible evaluation pairs",
                local_evaluation_indices=indices.tolist(), descending_separation_ranks=ranks.tolist(),
                global_dataset_indices=(indices + offset).tolist(),
                eligible_descending_separation_ranks=ranks.tolist(),
                all_evaluation_descending_separation_ranks=all_ranks[indices].tolist(),
                displayed_draw_indices=list(range(n_draws)))


def make_figures(args):
    evaluations = load_evaluations(args.run, args.evaluation_prefix)
    baseline_steps = getattr(args, "baseline_steps", 200)
    rows = [evaluation["row"] for evaluation in evaluations] + historical_rows(baseline_steps)
    args.out.mkdir(parents=True, exist_ok=True)
    comparison_figure(rows, args.out)
    distribution_figures(evaluations, args.out)
    examples = [example_figure(evaluation, args.out) for evaluation in evaluations]
    rows = clean_json(rows)
    with (args.out / "comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (args.out / "comparison.json").write_text(json.dumps(
        dict(rows=rows, examples=examples,
             historical_fm_sampling_steps=baseline_steps,
             historical_baselines="Committed single-seed FM/regression summaries; no regenerated baseline samples",
             legacy_energy="Asymmetric-distance diagnostic, not an established proper scoring rule"),
        indent=2, allow_nan=False,
    ) + "\n")
    print(f"Saved comparison, sharpness, mode balance, and example figures in {args.out}", flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, action="append", required=True,
                        help="Run folder; repeat to compare configurations")
    parser.add_argument("--evaluation-prefix", help="Exact saved prefix, e.g. test_ddim200_raw_eta0")
    parser.add_argument("--baseline-steps", type=int, choices=(50, 200), default=200,
                        help="Use the matching historical FM Euler evaluation")
    parser.add_argument("--out", type=Path, required=True)
    make_figures(parser.parse_args())


if __name__ == "__main__":
    main()
