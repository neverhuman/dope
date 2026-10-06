#!/usr/bin/env python3
"""Three IEEE figures from panel-stats.json, the lineage record, and loss TSVs.

Type 42 serif fonts. Lineage-bootstrap intervals only. No per-seed whiskers.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "serif",
    "font.serif": ["TeX Gyre Termes", "Nimbus Roman", "Liberation Serif"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
})
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[3]
RESULTS = REPO / "research" / "benchmark" / "results"
GEN = REPO / "docs" / "whitepaper" / "generated"
FIG = REPO / "docs" / "whitepaper" / "figures"
LOSS_ROOT = Path("/mnt/fast-scratch/dope-benchmark/dope-s3-loss-log-v1/replay")
SEED = 20261005
DRAWS = 10_000
BYTE_CAP = 10240
AUDITORS = ("catboost", "linear", "mlp")
METHODS = (
    ("DOPE", "DOPE", "#0072B2", "o"),
    ("GaussianCopula", "Gaussian copula", "#E69F00", "s"),
    ("Chow-Liu", "Chow-Liu", "#009E73", "^"),
    ("independent_marginals", "Independent", "#D55E00", "D"),
)
Y_LIM = (-1.5, 1.6)


def load_json(path):
    return json.loads(path.read_text())


def series_points(record, auditor):
    found = {key: [] for key, *_rest in METHODS}

    def add(label, group):
        utility = group["utility"][auditor]
        charged = group.get("charged_artifact_bytes")
        if not utility.get("complete_informative_sample_group"):
            return
        if not isinstance(charged, int) or charged <= 0:
            return
        value = utility.get("median_retention")
        if not isinstance(value, (int, float)) or not np.isfinite(value):
            return
        found[label].append((float(np.log10(charged)), float(value)))

    for row in record["rows"]:
        add("DOPE", row["sizes"]["4"])
    for method, _label, _color, _marker in METHODS[1:]:
        for sizes in record["comparators"][method].values():
            add(method, sizes["4"])
    return found


def draw_retention(record, stats):
    figure, axes = plt.subplots(1, 3, figsize=(7.16, 2.55), sharey=True)
    titles = {"catboost": "CatBoost", "linear": "Linear", "mlp": "MLP"}
    for axis, auditor in zip(axes, AUDITORS):
        points = series_points(record, auditor)
        outside = 0
        for key, label, color, marker in METHODS:
            cloud = points[key]
            inside = [(x, y) for x, y in cloud if Y_LIM[0] <= y <= Y_LIM[1]]
            outside += len(cloud) - len(inside)
            if inside:
                axis.scatter(
                    [item[0] for item in inside],
                    [item[1] for item in inside],
                    s=12,
                    c=color,
                    marker=marker,
                    linewidths=0,
                    label=label,
                    zorder=2,
                )
            summary = stats["blocks"]["density"][auditor]["methods"][key]
            if cloud and summary["median"] is not None and Y_LIM[0] <= summary["median"] <= Y_LIM[1]:
                x_med = float(np.median([item[0] for item in cloud]))
                axis.errorbar(
                    [x_med],
                    [summary["median"]],
                    yerr=[[summary["median"] - summary["lo"]], [summary["hi"] - summary["median"]]],
                    fmt=marker,
                    color=color,
                    markersize=5.5,
                    markerfacecolor="white",
                    markeredgewidth=0.8,
                    elinewidth=0.8,
                    capsize=2,
                    zorder=3,
                )
        axis.axvline(np.log10(BYTE_CAP), color="#666666", linestyle="--", linewidth=0.7)
        axis.set_ylim(*Y_LIM)
        axis.set_title(f"{titles[auditor]} ({outside} outside)")
        axis.set_xlabel("log10 charged bytes")
        if auditor == "catboost":
            axis.set_ylabel("retention (dimensionless)")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.tight_layout(pad=0.35, rect=(0, 0, 1, 0.90))
    figure.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.98))
    path = FIG / "retention-bytes.pdf"
    figure.savefig(path)
    plt.close(figure)
    return path


def draw_paired(stats):
    figure = plt.figure(figsize=(7.16, 3.45))
    grid = figure.add_gridspec(2, 3, height_ratios=(1.35, 1.0), hspace=0.55, wspace=0.35)
    pairs = stats["blocks"]["density"]["catboost"]["pairs"]
    order = (
        ("GaussianCopula", "Gaussian copula", "#E69F00"),
        ("Chow-Liu", "Chow-Liu", "#009E73"),
        ("independent_marginals", "Independent", "#D55E00"),
    )
    for column, (key, label, color) in enumerate(order):
        axis = figure.add_subplot(grid[0, column])
        values = np.sort(np.asarray(pairs[key]["differences"], dtype=float))
        axis.scatter(values, np.arange(1, values.size + 1), s=8, c=color, marker="o", linewidths=0)
        axis.axvline(0, color="#333333", linewidth=0.6)
        axis.set_xlim(-1.5, 1.5)
        axis.set_xlabel("DOPE minus comparator")
        if column == 0:
            axis.set_ylabel("lineages, sorted")
        lo, hi = pairs[key]["lo"], pairs[key]["hi"]
        outside = int(np.sum((values < -1.5) | (values > 1.5)))
        axis.set_title(f"{label}\nmedian {pairs[key]['median_difference']:.3f} [{lo:.3f}, {hi:.3f}] ({outside} outside)")
    axis = figure.add_subplot(grid[1, :])
    friedman = stats["blocks"]["density"]["catboost"]["friedman"]
    ranks = friedman["average_ranks"]
    labels = list(ranks)
    palette = {key: color for key, _label, color, _marker in METHODS}
    axis.axhline(0, color="#bbbbbb", linewidth=0.6)
    for key in labels:
        axis.scatter([ranks[key]], [0], s=36, c=palette[key], zorder=3)
        axis.text(ranks[key], 0.18, key.replace("independent_marginals", "Independent").replace("GaussianCopula", "Gaussian copula"), ha="center", va="bottom", fontsize=8)
    cd = friedman["cd"]
    axis.plot([1.0, 1.0 + cd], [-0.45, -0.45], color="#222222", linewidth=1.0)
    axis.plot([1.0, 1.0], [-0.52, -0.38], color="#222222", linewidth=1.0)
    axis.plot([1.0 + cd, 1.0 + cd], [-0.52, -0.38], color="#222222", linewidth=1.0)
    axis.text(1.0 + cd / 2, -0.62, f"critical difference {cd:.3f}", ha="center", va="top", fontsize=8)
    axis.set_xlim(0.8, 4.3)
    axis.set_ylim(-0.9, 0.7)
    axis.set_yticks([])
    axis.set_xlabel("average rank (1 is highest retention)")
    axis.set_title(f"CatBoost, {friedman['n']} lineages")
    path = FIG / "paired-cdf.pdf"
    figure.savefig(path)
    plt.close(figure)
    return path


def loss_matrix(profile):
    paths = sorted((LOSS_ROOT / profile).glob("*/loss.tsv"))
    rows = []
    for path in paths:
        table = np.loadtxt(path)
        if table.ndim != 2 or table.shape[1] != 3:
            continue
        rows.append(table[:, 1:])
    if not rows:
        return None
    width = min(row.shape[0] for row in rows)
    stacked = np.stack([row[:width] for row in rows], axis=0)
    return stacked


def bootstrap_band(values, rng):
    """values is (lineages, steps). Returns median, lo, hi at every step."""
    count, steps = values.shape
    median = np.median(values, axis=0)
    index = rng.integers(0, count, size=(DRAWS, count))
    lo = np.empty(steps)
    hi = np.empty(steps)
    stride = 8
    chosen = list(range(0, steps, stride))
    if chosen[-1] != steps - 1:
        chosen.append(steps - 1)
    lo[:] = np.nan
    hi[:] = np.nan
    for step in chosen:
        samples = np.median(values[:, step][index], axis=1)
        lo[step], hi[step] = np.quantile(samples, [0.025, 0.975])
    known = np.asarray(chosen)
    for slot in range(steps):
        if np.isfinite(lo[slot]):
            continue
        left = known[known < slot].max()
        right = known[known > slot].min()
        weight = (slot - left) / (right - left)
        lo[slot] = (1 - weight) * lo[left] + weight * lo[right]
        hi[slot] = (1 - weight) * hi[left] + weight * hi[right]
    return median, lo, hi


def draw_loss():
    rng = np.random.default_rng(SEED)
    profiles = ("features12_steps2048", "features12_steps8192")
    figure, axes = plt.subplots(1, 2, figsize=(7.16, 2.55), sharey=True)
    summary = {"seed": SEED, "draws": DRAWS, "stride_note": "band evaluated every 8 steps and linearly filled", "profiles": {}}
    if not LOSS_ROOT.is_dir():
        summary["error"] = "loss replay directory is absent"
        (GEN / "loss-curves.json").write_text(json.dumps(summary, indent=2) + "\n")
        plt.close(figure)
        return None
    for axis, profile in zip(axes, profiles):
        matrix = loss_matrix(profile)
        if matrix is None:
            summary["profiles"][profile] = {"error": "no loss.tsv"}
            continue
        train, train_lo, train_hi = bootstrap_band(matrix[:, :, 0], rng)
        valid, valid_lo, valid_hi = bootstrap_band(matrix[:, :, 1], rng)
        steps = np.arange(train.size)
        floor = 1e-8
        axis.plot(steps, np.maximum(train, floor), color="#0072B2", linewidth=1.0, label="train")
        axis.fill_between(steps, np.maximum(train_lo, floor), np.maximum(train_hi, floor), color="#0072B2", alpha=0.25, linewidth=0)
        axis.plot(steps, np.maximum(valid, floor), color="#E69F00", linewidth=1.0, label="validation")
        axis.fill_between(steps, np.maximum(valid_lo, floor), np.maximum(valid_hi, floor), color="#E69F00", alpha=0.25, linewidth=0)
        axis.set_xlabel("AdamW step")
        axis.set_ylabel("hidden-basis MSE" if profile.endswith("2048") else "")
        axis.set_yscale("log")
        axis.set_ylim(3e-4, 0.7)
        title = "2,048 steps" if profile.endswith("2048") else "8,192 steps, not displayed"
        axis.set_title(f"{title}, {matrix.shape[0]} lineages")
        summary["profiles"][profile] = {
            "lineages": int(matrix.shape[0]),
            "steps": int(train.size),
            "step0_train": float(train[0]),
            "step0_train_lo": float(train_lo[0]),
            "step0_train_hi": float(train_hi[0]),
            "step0_validation": float(valid[0]),
            "step0_validation_lo": float(valid_lo[0]),
            "step0_validation_hi": float(valid_hi[0]),
            "final_train": float(train[-1]),
            "final_train_lo": float(train_lo[-1]),
            "final_train_hi": float(train_hi[-1]),
            "final_validation": float(valid[-1]),
            "final_validation_lo": float(valid_lo[-1]),
            "final_validation_hi": float(valid_hi[-1]),
        }
    axes[0].legend(frameon=False)
    figure.tight_layout(pad=0.35)
    path = FIG / "loss-curves.pdf"
    figure.savefig(path)
    plt.close(figure)
    (GEN / "loss-curves.json").write_text(json.dumps(summary, indent=2) + "\n")
    return path


def main():
    stats = load_json(GEN / "panel-stats.json")
    record = load_json(RESULTS / "s3-lineage-record.json")
    paths = [draw_retention(record, stats), draw_paired(stats), draw_loss()]
    print("\n".join(str(path) for path in paths if path))


if __name__ == "__main__":
    main()
