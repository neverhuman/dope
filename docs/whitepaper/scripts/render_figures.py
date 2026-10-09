#!/usr/bin/env python3
"""Four paper figures from panel-stats.json, the lineage record, and loss TSVs.

Type 42 DejaVu Sans. Lineage-bootstrap intervals only. No per-seed whiskers.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("SOURCE_DATE_EPOCH", "1760000000")

import matplotlib

matplotlib.use("Agg")
import figure_style

# Composed at 7.16in and placed at \textwidth (516pt). TeX reads that
# PDF as 517.45pt, so 8.1pt here remains at least 8pt after placement.
FONT_PT = 8.1
figure_style.apply(matplotlib, size=FONT_PT)
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[3]
RESULTS = REPO / "research" / "benchmark" / "results"
GEN = REPO / "docs" / "whitepaper" / "generated"
FIG = REPO / "docs" / "whitepaper" / "figures"
from paper_receipts import loss_matrix as committed_loss_matrix
SEED = 20261005
DRAWS = 10_000
BYTE_CAP = 10240
AUDITORS = ("catboost", "linear", "mlp")
METHODS = (
    ("DOPE", "DOPE (ours)", figure_style.DOPE, "o"),
    ("GaussianCopula", "Gaussian copula", figure_style.GAUSSIAN, "s"),
    ("Chow-Liu", "Chow-Liu", figure_style.CHOW, "^"),
    ("independent_marginals", "Independent", figure_style.INDEPENDENT, "D"),
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


def _save(figure, path):
    figure.savefig(path)
    plt.close(figure)


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_hashes():
    names = ("retention-bytes.pdf", "retention-bars.pdf", "paired-cdf.pdf", "loss-curves.pdf")
    payload = {
        "format": "dope-paper-figure-hashes",
        "pdf_sha256": {name: _sha256(FIG / name) for name in names},
    }
    (GEN / "figure-hashes.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def draw_retention(record, stats, dest=None):
    figure, axes = plt.subplots(1, 3, figsize=(7.16, 2.85), sharey=True)
    titles = {"catboost": "CatBoost", "linear": "Linear", "mlp": "MLP"}
    for axis, auditor in zip(axes, AUDITORS):
        points = series_points(record, auditor)
        outside = 0
        for key, label, color, marker in METHODS:
            cloud = points[key]
            inside = [(x, y) for x, y in cloud if Y_LIM[0] <= y <= Y_LIM[1]]
            outside += len(cloud) - len(inside)
            dominant = key == "DOPE"
            if inside:
                axis.scatter(
                    [item[0] for item in inside],
                    [item[1] for item in inside],
                    s=16 if dominant else 12,
                    c=color,
                    marker=marker,
                    linewidths=0,
                    label=label,
                    zorder=4 if dominant else 2,
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
                    markersize=6.2 if dominant else 5.5,
                    markerfacecolor="white",
                    markeredgewidth=0.9 if dominant else 0.8,
                    elinewidth=0.9 if dominant else 0.8,
                    capsize=2,
                    zorder=5 if dominant else 3,
                )
        axis.axvline(np.log10(BYTE_CAP), color="#666666", linestyle="--", linewidth=0.7, zorder=1)
        axis.set_ylim(*Y_LIM)
        axis.set_title(f"{titles[auditor]} ({outside} outside)")
        axis.set_xlabel("log10 charged bytes")
        figure_style.panel(axis, grid="both")
        if auditor == "catboost":
            axis.set_ylabel("retention (dimensionless)")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.tight_layout(pad=0.35, rect=(0, 0.14, 1, 1))
    figure_style.legend_below(figure, handles, labels, ncol=4)
    path = Path(dest) if dest is not None else FIG / "retention-bytes.pdf"
    _save(figure, path)
    return path


def draw_paired(stats, dest=None):
    figure = plt.figure(figsize=(7.16, 3.85))
    grid = figure.add_gridspec(2, 3, height_ratios=(1.5, 1.0), hspace=0.78, wspace=0.38)
    pairs = stats["blocks"]["density"]["catboost"]["pairs"]
    order = (
        ("GaussianCopula", "Gaussian copula", figure_style.GAUSSIAN, "s"),
        ("Chow-Liu", "Chow-Liu", figure_style.CHOW, "^"),
        ("independent_marginals", "Independent", figure_style.INDEPENDENT, "D"),
    )
    for column, (key, label, color, marker) in enumerate(order):
        axis = figure.add_subplot(grid[0, column])
        values = np.sort(np.asarray(pairs[key]["differences"], dtype=float))
        axis.scatter(values, np.arange(1, values.size + 1), s=14, c=color, marker=marker, linewidths=0)
        axis.axvline(0, color="#333333", linewidth=0.6)
        axis.set_xlim(-1.5, 1.5)
        # One shared phrase. Repeating it under every panel collides in DejaVu Sans.
        if column == 1:
            axis.set_xlabel("retention difference (dimensionless)", fontsize=7)
        if column == 0:
            axis.set_ylabel("sorted lineage index")
        lo, hi = pairs[key]["lo"], pairs[key]["hi"]
        outside = int(np.sum((values < -1.5) | (values > 1.5)))
        figure_style.panel(axis, grid="both")
        axis.set_title(
            f"{label}\n{pairs[key]['median_difference']:.3f} [{lo:.3f}, {hi:.3f}] ({outside} outside)",
            fontsize=6.6,
        )
    axis = figure.add_subplot(grid[1, :])
    friedman = stats["blocks"]["density"]["catboost"]["friedman"]
    ranks = friedman["average_ranks"]
    labels = list(ranks)
    palette = {key: color for key, _label, color, _marker in METHODS}
    markers = {key: marker for key, _label, _color, marker in METHODS}
    axis.axhline(0, color="#000000", linewidth=0.6)
    for key in labels:
        axis.scatter([ranks[key]], [0], s=36, c=palette[key], marker=markers[key], zorder=3)
        axis.text(ranks[key], 0.18, key.replace("independent_marginals", "Independent").replace("GaussianCopula", "Gaussian copula"), ha="center", va="bottom", fontsize=FONT_PT)
    cd = friedman["cd"]
    axis.plot([1.0, 1.0 + cd], [-0.45, -0.45], color="#222222", linewidth=1.0)
    axis.plot([1.0, 1.0], [-0.52, -0.38], color="#222222", linewidth=1.0)
    axis.plot([1.0 + cd, 1.0 + cd], [-0.52, -0.38], color="#222222", linewidth=1.0)
    axis.text(1.0 + cd / 2, -0.62, f"critical difference {cd:.3f}", ha="center", va="top", fontsize=FONT_PT)
    axis.set_xlim(0.8, 4.3)
    axis.set_ylim(-0.9, 0.7)
    axis.set_yticks([])
    axis.set_xlabel("average rank (1 is highest retention)")
    axis.set_title(f"CatBoost, {friedman['n']} lineages", pad=6)
    path = Path(dest) if dest is not None else FIG / "paired-cdf.pdf"
    _save(figure, path)
    return path


def loss_matrix(profile):
    return np.asarray(committed_loss_matrix(profile), dtype=float)


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


def draw_loss(dest=None, summary_dest=None):
    """Regenerate curves and bootstrap intervals from authenticated committed TSVs."""
    rng = np.random.default_rng(SEED)
    profiles = ("features12_steps2048", "features12_steps8192")
    figure, axes = plt.subplots(1, 2, figsize=(7.16, 2.55), sharey=True)
    summary = {"seed": SEED, "draws": DRAWS, "stride_note": "band evaluated every 8 steps and linearly filled", "profiles": {}}
    for axis, profile in zip(axes, profiles):
        matrix = loss_matrix(profile)
        train, train_lo, train_hi = bootstrap_band(matrix[:, :, 0], rng)
        valid, valid_lo, valid_hi = bootstrap_band(matrix[:, :, 1], rng)
        steps = np.arange(train.size)
        floor = 1e-8
        axis.plot(steps, np.maximum(train, floor), color=figure_style.TRAIN, linestyle="-", linewidth=1.3, label="train")
        axis.fill_between(steps, np.maximum(train_lo, floor), np.maximum(train_hi, floor), color=figure_style.TRAIN, alpha=0.18, linewidth=0)
        axis.plot(steps, np.maximum(valid, floor), color=figure_style.VALIDATION, linestyle="--", linewidth=1.0, label="validation")
        axis.fill_between(steps, np.maximum(valid_lo, floor), np.maximum(valid_hi, floor), color=figure_style.VALIDATION, alpha=0.18, linewidth=0)
        axis.set_xlabel("AdamW step")
        axis.set_ylabel("hidden-basis MSE" if profile.endswith("2048") else "")
        axis.set_yscale("log")
        axis.set_ylim(3e-4, 0.7)
        title = "2,048 steps" if profile.endswith("2048") else "8,192 steps, not displayed"
        figure_style.panel(axis, grid="both")
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
    path = Path(dest) if dest is not None else FIG / "loss-curves.pdf"
    _save(figure, path)
    summary_path = Path(summary_dest) if summary_dest is not None else GEN / "loss-curves.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return path


# Same colors as the density marks, plus the two neural generators.
BAR_STYLE = {
    "DOPE": ("DOPE\n(ours)", figure_style.DOPE),
    "GaussianCopula": ("Gaussian\ncopula", figure_style.GAUSSIAN),
    "Chow-Liu": ("Chow-Liu", figure_style.CHOW),
    "independent_marginals": ("Independent\nmarginals", figure_style.INDEPENDENT),
    "TVAE": ("TVAE", figure_style.TVAE),
    "CTGAN": ("CTGAN", figure_style.CTGAN),
}


def _median_label(value):
    """Match sig3 on the retention scale so the bar text agrees with the tables."""
    if abs(value) >= 0.01:
        return f"{value:.3f}"
    return f"{value:.3g}"


def _paired_bars(block, baseline_keys):
    """One DOPE bar plus each baseline, on that block's paired CatBoost lineages."""
    anchor_key = baseline_keys[0]
    anchor = block["pairs"][anchor_key]
    rows = [("DOPE", anchor["dope_median"], anchor["dope_lo"], anchor["dope_hi"])]
    for key in baseline_keys:
        pair = block["pairs"][key]
        if int(pair["n"]) != int(anchor["n"]):
            raise SystemExit(f"{key} paired count {pair['n']} != {anchor['n']}")
        if abs(pair["dope_median"] - anchor["dope_median"]) > 5e-12:
            raise SystemExit(f"{key} DOPE median does not match {anchor_key}")
        rows.append((key, pair["other_median"], pair["other_lo"], pair["other_hi"]))
    return rows, int(anchor["n"])


def draw_retention_bars(stats, dest=None, *, readme=False):
    """CatBoost retention medians. Density and neural blocks stay unpooled.

    ``readme=True`` is the same bars for the repository front page. It does
    not redraw the paper PDF.
    """
    panels = (
        (
            _paired_bars(
                stats["blocks"]["density"]["catboost"],
                ("GaussianCopula", "Chow-Liu", "independent_marginals"),
            ),
            "Density block",
        ),
        (
            _paired_bars(stats["blocks"]["neural"]["catboost"], ("CTGAN", "TVAE")),
            "Neural block",
        ),
    )
    figure, axes = plt.subplots(1, 2, figsize=(7.16, 2.95), sharey=True)
    for axis, ((rows, count), title) in zip(axes, panels):
        positions = np.arange(len(rows))
        medians = [row[1] for row in rows]
        lowers = [row[1] - row[2] for row in rows]
        uppers = [row[3] - row[1] for row in rows]
        colors = [BAR_STYLE[row[0]][1] for row in rows]
        axis.bar(positions, medians, width=0.72, color=colors, edgecolor="none", zorder=2)
        axis.errorbar(
            positions,
            medians,
            yerr=[lowers, uppers],
            fmt="none",
            ecolor="#222222",
            elinewidth=0.7,
            capsize=2.4,
            capthick=0.7,
            zorder=3,
        )
        axis.axhline(0, color="#333333", linewidth=0.6, zorder=1)
        for xpos, row in zip(positions, rows):
            median, lo, hi = row[1], row[2], row[3]
            label = _median_label(median)
            if median >= 0:
                axis.text(xpos, hi + 0.025, label, ha="center", va="bottom", color="#1a1a1a")
            else:
                axis.text(xpos, lo - 0.025, label, ha="center", va="top", color="#1a1a1a")
        axis.set_xticks(positions)
        axis.set_xticklabels([BAR_STYLE[row[0]][0] for row in rows])
        axis.set_title(f"{title}, {count} lineages")
        figure_style.panel(axis, grid="y")
    if readme:
        axes[0].set_ylabel("retention (dimensionless)")
        for axis in axes:
            axis.set_xlabel("method")
    else:
        axes[0].set_ylabel("CatBoost retention (dimensionless)")
    axes[0].set_ylim(-0.42, 1.22)
    figure.tight_layout(pad=0.45)
    path = Path(dest) if dest is not None else FIG / "retention-bars.pdf"
    if readme:
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=160)
        plt.close(figure)
        return path
    _save(figure, path)
    return path


def _check():
    stats = load_json(GEN / "panel-stats.json")
    record = load_json(RESULTS / "s3-lineage-record.json")
    recorded = load_json(GEN / "figure-hashes.json")["pdf_sha256"]
    with tempfile.TemporaryDirectory() as tmp:
        fresh = {
            "retention-bytes.pdf": draw_retention(record, stats, Path(tmp) / "retention-bytes.pdf"),
            "retention-bars.pdf": draw_retention_bars(stats, Path(tmp) / "retention-bars.pdf"),
            "paired-cdf.pdf": draw_paired(stats, Path(tmp) / "paired-cdf.pdf"),
            "loss-curves.pdf": draw_loss(Path(tmp) / "loss-curves.pdf", Path(tmp) / "loss-curves.json"),
        }
        if (Path(tmp) / "loss-curves.json").read_bytes() != (GEN / "loss-curves.json").read_bytes():
            raise SystemExit("loss bootstrap summary does not reproduce from committed traces")
        for name, path in fresh.items():
            digest = _sha256(path)
            committed = _sha256(FIG / name)
            if digest != recorded[name] or committed != recorded[name]:
                raise SystemExit(f"{name} hash {digest} does not match the committed figure")
    print("figure hashes match")


def main():
    import sys
    if "--check" in sys.argv:
        _check()
        return
    stats = load_json(GEN / "panel-stats.json")
    record = load_json(RESULTS / "s3-lineage-record.json")
    paths = [draw_retention(record, stats), draw_retention_bars(stats), draw_paired(stats), draw_loss()]
    _write_hashes()
    print("\n".join(str(path) for path in paths if path))


if __name__ == "__main__":
    main()
