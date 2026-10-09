#!/usr/bin/env python3
"""Page-1 teaser from the committed density-block panel.

Reads docs/whitepaper/generated/panel-stats.json only. Each panel is one
auditor. Rows are DOPE-minus-comparator paired differences: filled marks are
lineages, and the open mark is the stored median with its stored 95%
lineage-bootstrap interval. No median is recomputed.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("SOURCE_DATE_EPOCH", "1760000000")

import matplotlib

matplotlib.use("Agg")
from matplotlib.lines import Line2D

import figure_style

figure_style.apply(matplotlib, size=8.0)
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[3]
GEN = REPO / "docs" / "whitepaper" / "generated"
FIG = REPO / "docs" / "whitepaper" / "figures"
STATS_NAME = "panel-stats.json"
PDF_NAME = "teaser.pdf"
HASH_NAME = "teaser-figure-hashes.json"
AUDITORS = (
    ("catboost", "CatBoost"),
    ("linear", "Linear"),
    ("mlp", "MLP"),
)
# Same markers as the body paired-difference figure. Not sorted by the median.
COMPARATORS = (
    ("GaussianCopula", "Gaussian copula", figure_style.GAUSSIAN, "s"),
    ("Chow-Liu", "Chow-Liu", figure_style.CHOW, "^"),
    ("independent_marginals", "Independent", figure_style.INDEPENDENT, "D"),
)
# The body figure uses this window and counts the points that fall outside it.
CLIP = 1.5


def _interval(median, lo, hi, label):
    if not all(isinstance(item, (int, float)) for item in (median, lo, hi)):
        raise SystemExit(f"{label} interval is not numeric")
    if not (lo <= median <= hi):
        raise SystemExit(f"{label} interval is not ordered")
    return float(median), float(lo), float(hi)


def density_differences(stats):
    """One panel per auditor. Differences and intervals are stored, not refit."""
    block = stats["blocks"]["density"]
    panels = []
    for key, title in AUDITORS:
        pairs = block[key]["pairs"]
        rows = []
        count = None
        for method, label, color, marker in COMPARATORS:
            pair = pairs[method]
            values = [float(item) for item in pair["differences"]]
            if len(values) != int(pair["n"]):
                raise SystemExit(f"{key} {method} difference count {len(values)} != {pair['n']}")
            if count is None:
                count = int(pair["n"])
            elif int(pair["n"]) != count:
                raise SystemExit(f"{key} {method} paired count {pair['n']} != {count}")
            median, lo, hi = _interval(
                pair["median_difference"], pair["lo"], pair["hi"], f"{key} {method} difference"
            )
            rows.append((label, color, marker, values, median, lo, hi))
        panels.append((title, count, rows))
    return panels


def _offsets(index, count):
    """Deterministic strip. Rank order is the only vertical jitter."""
    if count <= 1:
        return [float(index)]
    return [index - 0.32 + 0.64 * rank / (count - 1) for rank in range(count)]


def draw(stats, dest):
    panels = density_differences(stats)
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(figure_style.TEASER_WIDTH_IN, 2.95),
        sharex=True,
        sharey=True,
    )
    for axis, (title, count, rows) in zip(axes, panels):
        outside = 0
        for index, (_label, color, marker, values, median, lo, hi) in enumerate(rows):
            ordered = sorted(values)
            outside += sum(value < -CLIP or value > CLIP for value in ordered)
            axis.scatter(
                ordered,
                _offsets(index, len(ordered)),
                s=7,
                c=color,
                marker=marker,
                linewidths=0,
                zorder=2,
            )
            axis.errorbar(
                [median],
                [index],
                xerr=[[median - lo], [hi - median]],
                fmt=marker,
                color=color,
                markerfacecolor="white",
                markeredgewidth=0.8,
                markersize=5.5,
                elinewidth=0.8,
                capsize=2.0,
                capthick=0.7,
                zorder=4,
            )
        axis.axvline(0, color="#333333", linewidth=0.6, zorder=1)
        axis.set_xlim(-CLIP, CLIP)
        axis.set_title(f"{title}, {count}\n{outside} outside the frame", fontsize=7.2)
        figure_style.panel(axis, grid="x")
    axes[0].set_yticks([0, 1, 2])
    axes[0].set_yticklabels(["Copula", "Chow-Liu", "Independent"], fontsize=6.5)
    axes[0].set_ylim(-0.55, 2.55)
    for axis in axes[1:]:
        axis.tick_params(labelleft=False)
    axes[1].set_xlabel("retention difference (dimensionless)", fontsize=7)
    handles = [
        Line2D(
            [0],
            [0],
            marker=marker,
            color="none",
            markerfacecolor=color,
            markeredgecolor=color,
            linestyle="None",
            markersize=5,
            label=label,
        )
        for _key, label, color, marker in COMPARATORS
    ]
    labels = [label for _key, label, _color, _marker in COMPARATORS]
    figure.tight_layout(pad=0.35, rect=(0.02, 0.16, 1, 0.98))
    figure_style.legend_below(figure, handles, labels, ncol=3)
    path = Path(dest)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, metadata={"CreationDate": None, "ModDate": None})
    plt.close(figure)
    return path


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def hash_document(path):
    return {"pdf_sha256": {PDF_NAME: _sha256(path)}}


def write_outputs():
    stats = json.loads((GEN / STATS_NAME).read_text())
    path = draw(stats, FIG / PDF_NAME)
    payload = hash_document(path)
    (GEN / HASH_NAME).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    for title, count, rows in density_differences(stats):
        rendered = ", ".join(f"{label} {median:.3f} [{lo:.3f}, {hi:.3f}]" for label, _c, _m, _v, median, lo, hi in rows)
        print(f"{title} n={count}: {rendered}")
    print(path)
    print(payload["pdf_sha256"][PDF_NAME])


def check_outputs():
    stats = json.loads((GEN / STATS_NAME).read_text())
    recorded = json.loads((GEN / HASH_NAME).read_text())["pdf_sha256"]
    with tempfile.TemporaryDirectory() as tmp:
        fresh = draw(stats, Path(tmp) / PDF_NAME)
        digest = _sha256(fresh)
    committed = _sha256(FIG / PDF_NAME)
    if digest != recorded[PDF_NAME] or committed != recorded[PDF_NAME]:
        raise SystemExit(f"{PDF_NAME} hash {digest} does not match the committed figure")
    print("teaser hash matches")


def main():
    if "--check" in sys.argv:
        check_outputs()
        return
    write_outputs()


if __name__ == "__main__":
    main()
