#!/usr/bin/env python3
"""Page-1 teaser from the committed density-block panel.

Reads docs/whitepaper/generated/panel-stats.json only. Paired medians and
95% lineage-bootstrap intervals are plotted as stored. No other ledger is
opened, and no median is recomputed or re-selected.
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
from matplotlib.patches import Patch

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
# Fixed order, the same series in every panel. Not sorted by the median.
SERIES = (
    ("DOPE", "DOPE (ours)", figure_style.DOPE),
    ("GaussianCopula", "Gaussian copula", figure_style.GAUSSIAN),
    ("Chow-Liu", "Chow-Liu", figure_style.CHOW),
    ("independent_marginals", "Independent marginals", figure_style.INDEPENDENT),
)


def sig3(value):
    """Same retention rounding as compute_panel.sig3. Display only."""
    if abs(value) >= 0.01:
        return f"{value:.3f}"
    return f"{value:.3g}"


def _interval(median, lo, hi, label):
    if not all(isinstance(item, (int, float)) for item in (median, lo, hi)):
        raise SystemExit(f"{label} interval is not numeric")
    if not (lo <= median <= hi):
        raise SystemExit(f"{label} interval is not ordered")
    return float(median), float(lo), float(hi)


def density_rows(stats):
    """One row per auditor: four paired (median, lo, hi) tuples and the count."""
    block = stats["blocks"]["density"]
    panels = []
    for key, title in AUDITORS:
        pairs = block[key]["pairs"]
        anchor = pairs["GaussianCopula"]
        dope = _interval(anchor["dope_median"], anchor["dope_lo"], anchor["dope_hi"], f"{key} DOPE")
        count = int(anchor["n"])
        rows = {"DOPE": dope}
        for method, _label, _color in SERIES[1:]:
            pair = pairs[method]
            if int(pair["n"]) != count:
                raise SystemExit(f"{key} {method} paired count {pair['n']} != {count}")
            if abs(pair["dope_median"] - dope[0]) > 5e-12:
                raise SystemExit(f"{key} {method} DOPE median does not match the anchor")
            rows[method] = _interval(
                pair["other_median"], pair["other_lo"], pair["other_hi"], f"{key} {method}"
            )
        panels.append((title, count, rows))
    return panels


def draw(stats, dest):
    panels = density_rows(stats)
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(figure_style.TEASER_WIDTH_IN, 2.62),
        sharey=True,
    )
    for axis, (title, count, rows) in zip(axes, panels):
        positions = list(range(len(SERIES)))
        medians = [rows[key][0] for key, _label, _color in SERIES]
        lowers = [rows[key][0] - rows[key][1] for key, _label, _color in SERIES]
        uppers = [rows[key][2] - rows[key][0] for key, _label, _color in SERIES]
        colors = [color for _key, _label, color in SERIES]
        axis.bar(
            positions,
            medians,
            width=0.72,
            color=colors,
            edgecolor="none",
            zorder=2,
        )
        axis.errorbar(
            positions,
            medians,
            yerr=[lowers, uppers],
            fmt="none",
            ecolor="#333333",
            elinewidth=0.7,
            capsize=2.2,
            capthick=0.7,
            zorder=3,
        )
        axis.axhline(0, color="#333333", linewidth=0.6, zorder=1)
        for xpos, (key, _label, _color) in zip(positions, SERIES):
            median, lo, hi = rows[key]
            text = sig3(median)
            if median >= 0:
                axis.text(xpos, hi + 0.03, text, ha="center", va="bottom", fontsize=6.5, color=figure_style.INK)
            else:
                axis.text(xpos, lo - 0.03, text, ha="center", va="top", fontsize=6.5, color=figure_style.INK)
        axis.set_xticks(positions)
        axis.set_xticklabels([])
        axis.set_title(f"{title}, {count} lineages")
        figure_style.panel(axis, grid="y")
    axes[0].set_ylabel("retention (dimensionless)")
    axes[0].set_ylim(-0.28, 1.42)
    handles = [Patch(facecolor=color, edgecolor="none", label=label) for _key, label, color in SERIES]
    labels = [label for _key, label, _color in SERIES]
    # Left and top insets keep the rotated y label and the bold titles
    # inside the page. A zero left rect clips "retention" by about 4pt.
    figure.tight_layout(pad=0.35, rect=(0.03, 0.16, 1, 0.98))
    figure_style.legend_below(figure, handles, labels, ncol=2)
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
    for title, count, rows in density_rows(stats):
        rendered = ", ".join(f"{label} {sig3(rows[key][0])}" for key, label, _color in SERIES)
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
