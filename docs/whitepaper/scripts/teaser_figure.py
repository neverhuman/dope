#!/usr/bin/env python3
"""Page-1 teaser: the DOPE generator minus ARF, TabSyn, the copula, and Chow-Liu.

The open mark and bar are the stored family-cluster median and interval on
the size-4n review receipt. Filled marks are the stored paired differences
whose stored median equals that receipt. Author-default ARF points come from
retained-evidence.json. Copula and Chow-Liu points come from the density
block of panel-stats.json. No median is recomputed, and native-selected ARF
stays off this figure. Bound TabSyn points use its matched cohort, with a separate count.
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
from paper_emit import _REVIEW_PANEL_SHA256

figure_style.apply(matplotlib, size=8.0)
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[3]
GEN = REPO / "docs" / "whitepaper" / "generated"
FIG = REPO / "docs" / "whitepaper" / "figures"
STATS_NAME = "panel-stats.json"
RECEIPT_NAME = "review-receipts.json"
EVIDENCE_NAME = "retained-evidence.json"
PDF_NAME = "teaser.pdf"
HASH_NAME = "teaser-figure-hashes.json"
WAVE2_NAME = "review-wave2.json"
WAVE2_SHA256 = "692becfe7766cd799936eee9d3b281036dfc7dfc9423d1766b4211ec22e492bb"
AUDITORS = (
    ("catboost", "CatBoost"),
    ("linear", "Linear"),
    ("mlp", "MLP"),
)
# method, configuration, point source, tick, legend, color, marker.
# Order is the row order. ARF is the full-panel baseline.
COMPARATORS = (
    ("ARF", "author_default", "retained", "ARF", "ARF", figure_style.ARF, "o"),
    ("TabSyn", "scaled_200_vae_1000_diffusion", "wave2", "TabSyn", "TabSyn", figure_style.TABSYN, "D"),
    ("GaussianCopula", "native_selected", "panel", "Copula", "Gaussian copula", figure_style.GAUSSIAN, "s"),
    ("Chow-Liu", "native_selected", "panel", "Chow-Liu", "Chow-Liu", figure_style.CHOW, "^"),
)
# The body figure uses this window and counts the points that fall outside it.
CLIP = 1.5


def _interval(median, lo, hi, label):
    if not all(isinstance(item, (int, float)) for item in (median, lo, hi)):
        raise SystemExit(f"{label} interval is not numeric")
    if not (lo <= median <= hi):
        raise SystemExit(f"{label} interval is not ordered")
    return float(median), float(lo), float(hi)


def _review_index(receipt):
    chosen = {}
    for row in receipt["paired"]:
        if row.get("size") != 4:
            continue
        key = (row["method"], row["configuration"], row["auditor"])
        if key in chosen:
            raise SystemExit(f"duplicate review row {key}")
        chosen[key] = row
    return chosen


def _arf_author_default(evidence):
    chosen = {}
    for row in evidence["arf_paired"]:
        if row.get("configuration") != "author_default":
            continue
        auditor = row["auditor"]
        if auditor in chosen:
            raise SystemExit(f"duplicate ARF author-default point row for {auditor}")
        chosen[auditor] = row
    return chosen


def _point_row(source, auditor, method, arf_rows, density, tabsyn_rows):
    if source == "wave2":
        row = tabsyn_rows[auditor]
        return row["differences"], row["summary"]["median"], row["summary"]["n"]
    if source == "retained":
        row = arf_rows.get(auditor)
        if row is None:
            raise SystemExit(f"missing author-default ARF points for {auditor}")
        return row["differences"], row["median_difference"], row["n"]
    if source == "panel":
        pair = density[auditor]["pairs"][method]
        return pair["differences"], pair["median_difference"], pair["n"]
    raise SystemExit(f"unknown point source {source}")


def paired_panels():
    """One panel per auditor. Bars follow the review receipt; points follow the matching series."""
    receipt_path = GEN / RECEIPT_NAME
    raw = receipt_path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != _REVIEW_PANEL_SHA256:
        raise SystemExit("review receipt drift")
    review = _review_index(json.loads(raw))
    wave2_raw = (GEN / WAVE2_NAME).read_bytes()
    if hashlib.sha256(wave2_raw).hexdigest() != WAVE2_SHA256:
        raise SystemExit("wave2 paired point receipt drift")
    wave2 = json.loads(wave2_raw)
    tabsyn_rows = {row["auditor"]: row for row in wave2["tabsyn_teaser"]}
    for auditor, row in tabsyn_rows.items():
        review[("TabSyn", "scaled_200_vae_1000_diffusion", auditor)] = row["summary"]
    evidence = json.loads((GEN / EVIDENCE_NAME).read_text())
    stats = json.loads((GEN / STATS_NAME).read_text())
    arf_rows = _arf_author_default(evidence)
    density = stats["blocks"]["density"]
    panels = []
    for auditor, title in AUDITORS:
        rows = []
        counts = set()
        for method, configuration, source, tick, label, color, marker in COMPARATORS:
            key = (method, configuration, auditor)
            receipt = review.get(key)
            if receipt is None:
                raise SystemExit(f"missing review row {key}")
            values_raw, stored_median, stored_n = _point_row(source, auditor, method, arf_rows, density, tabsyn_rows)
            values = [float(item) for item in values_raw]
            if len(values) != int(stored_n) or int(stored_n) != int(receipt["n"]):
                raise SystemExit(f"{key} difference count does not match the review receipt")
            if float(stored_median) != float(receipt["median"]):
                raise SystemExit(f"{key} stored median does not match the review receipt")
            median, lo, hi = _interval(
                receipt["median"], receipt["lo"], receipt["hi"], f"{key} family-cluster"
            )
            if source != "wave2":
                counts.add(int(receipt["n"]))
            rows.append((tick, label, color, marker, values, median, lo, hi))
        if len(counts) != 1:
            raise SystemExit(f"{auditor} comparators do not share one paired count")
        panels.append((title, str(counts.pop()) + "/" + str(tabsyn_rows[auditor]["summary"]["n"]), rows))
    return panels


def _offsets(index, count):
    """Deterministic strip. Rank order is the only vertical jitter."""
    if count <= 1:
        return [float(index)]
    return [index - 0.32 + 0.64 * rank / (count - 1) for rank in range(count)]


def draw(panels, dest):
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(figure_style.TEASER_WIDTH_IN, 2.95),
        sharex=True,
        sharey=True,
    )
    for axis, (_title, count, rows) in zip(axes, panels):
        outside = 0
        # First comparator is the headline row, so it sits at the top of the axis.
        for index, (_tick, _label, color, marker, values, median, lo, hi) in enumerate(rows):
            center = len(rows) - 1 - index
            ordered = sorted(values)
            outside += sum(value < -CLIP or value > CLIP for value in ordered)
            axis.scatter(
                ordered,
                _offsets(center, len(ordered)),
                s=7,
                c=color,
                marker=marker,
                linewidths=0,
                zorder=2,
            )
            axis.errorbar(
                [median],
                [center],
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
        axis.set_title(f"{_title}, n={count}\n{outside} outside the frame", fontsize=7.2)
        figure_style.panel(axis, grid="x")
    ticks = [tick for tick, _label, _color, _marker, _values, _median, _lo, _hi in panels[0][2]]
    axes[0].set_yticks(list(range(len(ticks))))
    axes[0].set_yticklabels(list(reversed(ticks)), fontsize=6.5)
    axes[0].set_ylim(-0.55, len(ticks) - 0.45)
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
        for _tick, label, color, marker, _values, _median, _lo, _hi in panels[0][2]
    ]
    labels = [label for _tick, label, _color, _marker, _values, _median, _lo, _hi in panels[0][2]]
    figure.tight_layout(pad=0.35, rect=(0.02, 0.16, 1, 0.98))
    figure_style.legend_below(figure, handles, labels, ncol=4)
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
    panels = paired_panels()
    path = draw(panels, FIG / PDF_NAME)
    payload = hash_document(path)
    (GEN / HASH_NAME).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    for title, count, rows in panels:
        rendered = ", ".join(
            f"{label} {median:.3f} [{lo:.3f}, {hi:.3f}]"
            for _tick, label, _color, _marker, _values, median, lo, hi in rows
        )
        print(f"{title} n={count}: {rendered}")
    print(path)
    print(payload["pdf_sha256"][PDF_NAME])


def check_outputs():
    panels = paired_panels()
    recorded = json.loads((GEN / HASH_NAME).read_text())["pdf_sha256"]
    with tempfile.TemporaryDirectory() as tmp:
        fresh = draw(panels, Path(tmp) / PDF_NAME)
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
