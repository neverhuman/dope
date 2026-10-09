"""UNREAL figure palette for the whitepaper.

DejaVu Sans, Type 42. DOPE uses the same green as the UNREAL reference
series. Other methods keep distinct hues and the existing markers or hatches.
This module does not read measurements.
"""

from __future__ import annotations

import os

os.environ.setdefault("SOURCE_DATE_EPOCH", "1760000000")

# Bright green sampled from the UNREAL page-1 series (about rgb 118, 185, 0).
DOPE = "#76B900"
GAUSSIAN = "#222222"
CHOW = "#E69F00"
INDEPENDENT = "#0072B2"
TVAE = "#56B4E9"
CTGAN = "#CC79A7"
FOREST = "#666666"
ARF = "#CC79A7"
TABSYN = "#6A3D9A"
TABDDPM = "#0072B2"
TRAIN = DOPE
VALIDATION = "#D55E00"
GRID = "#E6E6E6"
SPINE = "#4A4A4A"
INK = "#1A1A1A"

COLORS = {
    "DOPE": DOPE,
    "GaussianCopula": GAUSSIAN,
    "Chow-Liu": CHOW,
    "independent_marginals": INDEPENDENT,
    "TVAE": TVAE,
    "CTGAN": CTGAN,
    "Forest-Flow": FOREST,
    "ARF": ARF,
    "TabSyn": TABSYN,
    "TabDDPM": TABDDPM,
}

# Native size of every figure that check_paper.py still measures.
FIG_WIDTH_IN = 7.16
# Page-1 teaser is drawn at the UNREAL text width. It is not in that check.
TEASER_WIDTH_IN = 5.5


def apply(matplotlib_module, size=8.1):
    """Sans-serif DejaVu, embedded as TrueType. Safe to call more than once."""
    matplotlib_module.rcParams.update({
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": size,
        "axes.labelsize": size,
        "axes.titlesize": size,
        "axes.titleweight": "bold",
        "xtick.labelsize": size,
        "ytick.labelsize": size,
        "legend.fontsize": size,
        "axes.linewidth": 0.6,
        "axes.edgecolor": SPINE,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
    })


def panel(axis, grid="y"):
    """Light grid under the marks. Spines stay so the frame matches UNREAL."""
    axis.set_axisbelow(True)
    axis.grid(True, axis=grid, linewidth=0.4, color=GRID, zorder=0)
    for side in ("top", "right", "left", "bottom"):
        axis.spines[side].set_visible(True)
        axis.spines[side].set_color(SPINE)
        axis.spines[side].set_linewidth(0.6)


def legend_below(figure, handles, labels, ncol):
    """One shared legend in the bottom margin, not inside the panels."""
    figure.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=ncol,
        frameon=False,
        handlelength=1.5,
        columnspacing=1.0,
        borderaxespad=0.2,
    )


def cohort_color(label):
    """Color a published-cohort name by method family. Not a ranking."""
    text = label.lower()
    if "chow" in text:
        return CHOW
    if "gaussian" in text or "copula" in text:
        return GAUSSIAN
    if "forest" in text:
        return FOREST
    if text.startswith("arf") or " arf" in text:
        return ARF
    if "tabddpm" in text:
        return TABDDPM
    if "tabsyn" in text:
        return TABSYN
    return GAUSSIAN
