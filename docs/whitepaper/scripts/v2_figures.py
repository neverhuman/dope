#!/usr/bin/env python3
"""Page-1 forest plot and the utility-per-byte figure, from v2-headline.json.

Both figures use the emphasis form: DOPE in its accent green, every
comparator in recessive grey with a direct ink label. Identity never rests
on color alone. The README variants are the same drawings as PNG files.
`--check` redraws into a temporary directory and compares SHA-256 digests.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("SOURCE_DATE_EPOCH", "1760000000")

import matplotlib

matplotlib.use("Agg")

import figure_style

figure_style.apply(matplotlib, size=7.6)
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
GEN = REPO / "docs" / "whitepaper" / "generated"
FIG = REPO / "docs" / "whitepaper" / "figures"
README_DIR = REPO / "docs" / "readme"
HEADLINE = GEN / "v2-headline.json"
ACCENT = "#76B900"
ACCENT_EDGE = "#3F6B00"
CONTEXT = "#7A7A7A"
CONTROL = "#A8A8A8"
BAND = "#EDEDED"
CAP = 10240
FULL_PANEL = 90
AUDITORS = (("catboost", "CatBoost"), ("linear", "Linear"), ("mlp", "MLP"))
OUTPUTS = {
    "teaser.pdf": (FIG, "teaser-figure-hashes.json"),
    "pareto.pdf": (FIG, "pareto-figure-hashes.json"),
    "paired.png": (README_DIR, "readme-figure-hashes.json"),
    "pareto.png": (README_DIR, "readme-figure-hashes.json"),
}


def _rows(headline: dict) -> list[dict]:
    """Comparator rows for the forest plot: full panel, smaller cohorts, then controls."""
    rows = [arm for arm in headline["arms"] if arm["code"] != "Dope" and arm["contrasts"]["catboost"]]
    rank = {"generator": 0, "diagnostic": 2, "control": 3}

    def key(arm):
        small = arm["contrasts"]["catboost"]["n"] < FULL_PANEL
        return (rank.get(arm["role"], 1) + (1 if small and arm["role"] == "generator" else 0),
                -(arm["level"]["median"] or 0.0))
    return sorted(rows, key=key)


def draw_forest(headline: dict, path: Path, width: float = 5.5, dpi: int | None = None) -> Path:
    rows = _rows(headline)
    margin = headline.get("margin", 0.02)
    height = 0.42 + 0.215 * len(rows)
    figure, axes = plt.subplots(1, 3, figsize=(width, height), sharey=True)
    lo_all = min(arm["contrasts"][a]["hl_lo"] for arm in rows for a, _ in AUDITORS if arm["contrasts"][a])
    hi_all = max(arm["contrasts"][a]["hl_hi"] for arm in rows for a, _ in AUDITORS if arm["contrasts"][a])
    left, right = max(-1.0, min(-0.1, lo_all - 0.05)), min(1.2, max(0.2, hi_all + 0.05))
    for axis, (auditor, title) in zip(axes, AUDITORS):
        figure_style.panel(axis, grid="x")
        axis.axvspan(-margin, margin, color=BAND, zorder=0, linewidth=0)
        axis.axvline(0.0, color=ACCENT_EDGE, linewidth=0.9, zorder=1)
        for index, arm in enumerate(rows):
            row = arm["contrasts"][auditor]
            if row is None:
                continue
            y = len(rows) - 1 - index
            colour = CONTROL if arm["role"] == "control" else CONTEXT
            hollow = row["n"] < FULL_PANEL
            lo, hi = max(left, row["hl_lo"]), min(right, row["hl_hi"])
            axis.plot([lo, hi], [y, y], color=colour, linewidth=1.4, solid_capstyle="round", zorder=2)
            axis.plot([row["hl"]], [y], marker="o", markersize=4.6, zorder=3,
                      markerfacecolor="white" if hollow else colour, markeredgecolor=colour, markeredgewidth=1.0)
            if auditor == "mlp":
                axis.annotate(f"n={row['n']}", xy=(1.0, y), xycoords=("axes fraction", "data"),
                              xytext=(3, 0), textcoords="offset points", va="center", fontsize=6.4,
                              color=figure_style.INK)
        axis.set_xlim(left, right)
        axis.set_title(title)
    axes[0].set_yticks(range(len(rows)))
    axes[0].set_yticklabels([arm["display"] for arm in reversed(rows)])
    axes[0].set_ylim(-0.6, len(rows) - 0.4)
    figure.supxlabel("DOPE minus comparator, Hodges-Lehmann retention difference (95% nested interval)",
                     fontsize=figure_style_size(), y=0.01)
    figure.subplots_adjust(left=0.27, right=0.93, bottom=0.2, top=0.9, wspace=0.08)
    return _save(figure, path, dpi)


def figure_style_size() -> float:
    return matplotlib.rcParams["axes.labelsize"]


def draw_pareto(headline: dict, path: Path, width: float = 5.5, dpi: int | None = None) -> Path:
    figure, axis = plt.subplots(figsize=(width, 2.7))
    figure_style.panel(axis, grid="both")
    axis.set_xscale("log")
    axis.axvline(CAP, color=figure_style.SPINE, linestyle=(0, (4, 3)), linewidth=0.8, zorder=1)
    axis.annotate("10,240-byte cap", xy=(CAP, 1.0), xycoords=("data", "axes fraction"), xytext=(3, -9),
                  textcoords="offset points", fontsize=6.4, color=figure_style.INK)
    xs = [arm["bytes"][key] for arm in headline["arms"] if arm["bytes"] and arm["role"] in {"headline", "generator"}
          for key in ("q1", "q3")]
    right_edge = 10 ** (math.ceil(math.log10(max(xs))) + 0.3) if xs else None
    for arm in headline["arms"]:
        level = arm["level"]
        if arm["role"] == "control":
            axis.axhline(level["median"], color=CONTROL, linestyle=(0, (2, 2)), linewidth=0.9, zorder=1)
            axis.annotate(arm["display"], xy=(1.0, level["median"]), xycoords=("axes fraction", "data"),
                          xytext=(-3, 2), textcoords="offset points", ha="right", va="bottom",
                          fontsize=6.4, color=figure_style.INK)
            continue
        if arm["bytes"] is None or arm["role"] == "diagnostic":
            continue
        size = arm["bytes"]["median"]
        is_dope = arm["code"] == "Dope"
        colour = ACCENT if is_dope else CONTEXT
        hollow = (level["n"] or 0) < FULL_PANEL
        axis.plot([arm["bytes"]["q1"], arm["bytes"]["q3"]], [level["median"]] * 2, color=colour,
                  linewidth=1.1, alpha=0.8, zorder=2)
        if level["lo"] is not None:
            axis.plot([size, size], [level["lo"], level["hi"]], color=colour, linewidth=1.1, zorder=2)
        axis.plot([size], [level["median"]], marker="o", markersize=6.5 if is_dope else 4.8, zorder=3,
                  markerfacecolor="white" if hollow else colour,
                  markeredgecolor=ACCENT_EDGE if is_dope else colour, markeredgewidth=1.0)
        near_right = right_edge is not None and size > right_edge / 40
        axis.annotate(arm["display"], xy=(size, level["median"]), xytext=(-4 if near_right else 4, 3),
                      textcoords="offset points", ha="right" if near_right else "left",
                      fontsize=6.6 if is_dope else 6.2, color=figure_style.INK,
                      fontweight="bold" if is_dope else "normal")
    if xs:
        axis.set_xlim(10 ** math.floor(math.log10(min(xs)) - 0.2), right_edge)
    axis.set_xlabel("median charged artifact bytes (log scale; bar = interquartile range)")
    axis.set_ylabel("CatBoost retention at 4n")
    figure.subplots_adjust(left=0.1, right=0.98, bottom=0.17, top=0.95)
    return _save(figure, path, dpi)


def _save(figure, path: Path, dpi: int | None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".png":
        figure.savefig(path, dpi=dpi or 200, metadata={"Software": None})
    else:
        figure.savefig(path, metadata={"CreationDate": None, "ModDate": None})
    plt.close(figure)
    return path


def _draw_all(headline: dict, root: Path | None) -> dict[str, Path]:
    def place(name):
        directory = OUTPUTS[name][0] if root is None else root
        return directory / name
    return {
        "teaser.pdf": draw_forest(headline, place("teaser.pdf")),
        "pareto.pdf": draw_pareto(headline, place("pareto.pdf")),
        "paired.png": draw_forest(headline, place("paired.png"), width=8.0, dpi=200),
        "pareto.png": draw_pareto(headline, place("pareto.png"), width=8.0, dpi=200),
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_outputs(headline: dict) -> None:
    drawn = _draw_all(headline, None)
    documents: dict[str, dict] = {}
    for name, path in drawn.items():
        documents.setdefault(OUTPUTS[name][1], {})[name] = _sha256(path)
    for hash_name, digests in documents.items():
        key = "png_sha256" if hash_name.startswith("readme") else "pdf_sha256"
        (GEN / hash_name).write_text(json.dumps({key: digests}, indent=2, sort_keys=True) + "\n")
        print(hash_name, digests)


def check_outputs(headline: dict) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        fresh = {name: _sha256(path) for name, path in _draw_all(headline, Path(tmp)).items()}
    for name, (directory, hash_name) in OUTPUTS.items():
        recorded = json.loads((GEN / hash_name).read_text())
        digest = (recorded.get("pdf_sha256") or recorded.get("png_sha256") or {}).get(name)
        if fresh[name] != digest or _sha256(directory / name) != digest:
            raise SystemExit(f"{name} does not match {hash_name}")
    print("v2 figure hashes match")


def main() -> None:
    headline = json.loads(HEADLINE.read_text())
    if "--check" in sys.argv:
        check_outputs(headline)
        return
    write_outputs(headline)


if __name__ == "__main__":
    main()
