#!/usr/bin/env python3
"""Page-1 hero, the utility-per-byte figure and the README forest, from v2-headline.json.

The page-1 hero (teaser.pdf) is a single panel showing how much predictive
structure each generator keeps per charged byte at synthetic size 4n. It is
drawn at the text width and placed at that width, so every label keeps
HERO_MIN_POINTS on the page.

All figures use the emphasis form: DOPE in its accent green, every
comparator in recessive grey with a direct ink label. Identity never rests
on color alone. The README includes both the utility-per-byte PNG and the
three-auditor paired-difference forest PNG.
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

import declared_facts
import figure_style
from figure_spec import TEASER_ROW_PAD, TEASER_X_LIM

figure_style.apply(matplotlib, size=7.6)
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from research.benchmark.review_fixes.v2_panel import STRONGEST_MIN_LINEAGES  # noqa: E402

GEN = REPO / "docs" / "whitepaper" / "generated"
FIG = REPO / "docs" / "whitepaper" / "figures"
README_DIR = REPO / "docs" / "readme"
HEADLINE = GEN / "v2-headline.json"
ACCENT = "#76B900"
ACCENT_EDGE = "#3F6B00"
DEFAULT_GREEN = "#A6D96A"
CONTEXT = "#7A7A7A"
CONTROL = "#A8A8A8"
BAND = "#D6D6D6"
# The cap, the full-panel lineage floor and the interval level are read, not typed.
CAP = declared_facts.byte_cap(declared_facts.read(declared_facts.CONTRACT, REPO))
FULL_PANEL = STRONGEST_MIN_LINEAGES
INTERVAL_LEVEL = declared_facts.interval_level(declared_facts.read(declared_facts.PREDECLARE, REPO))
FOREST_TICKS = (0.0, 0.3, 0.6)
# Readability floor for the page-1 hero at its printed size (ledger C053).
HERO_MIN_POINTS = 7.0
HERO_HEIGHT_IN = 2.58
# Table 1 reports all rows; HERO_OMITTED_ROWS guards the forest window contract.
HERO_OMITTED_ROWS = ("Ind", "Ctgan")
AUDITORS = (("catboost", "CatBoost"), ("linear", "Linear"), ("mlp", "MLP"))
# The figures draw these arms only; every other arm stays in the supplement tables.
FIGURE_CODES = ("TabSyn", "Arf", "Gauss", "Chow", "Ind", "Tvae", "Ctgan", "Forest", "TabDdpm", "TabDiff",
                "Great", "Synthpop", "Smote", "Pred", "Boot")
V2_PANEL = REPO / "research" / "benchmark" / "results" / "review-fixes-v2" / "panel.json"
MFS_PANEL = REPO / "research" / "benchmark" / "results" / "mfs-v3-density-panel.json"
OUTPUTS = {
    "teaser.pdf": (FIG, "teaser-figure-hashes.json"),
    "pareto.pdf": (FIG, "pareto-figure-hashes.json"),
    "paired.png": (README_DIR, "readme-figure-hashes.json"),
    "pareto.png": (README_DIR, "readme-figure-hashes.json"),
}


def _multimodel_summary() -> dict:
    """Load multi-model (CatBoost, Linear, MLP) and foundation-model (Kumo, TabICL, Mitra) + driver stats."""
    classical: dict[tuple[str, str], dict[str, tuple[float, float, float]]] = {}
    if V2_PANEL.is_file():
        v2 = json.loads(V2_PANEL.read_text())
        for row in v2.get("levels", ()):
            if row.get("size") == 4:
                key = (row["method"], row["configuration"])
                classical.setdefault(key, {})[row["auditor"]] = (
                    float(row["median"]), float(row["lo"]), float(row["hi"])
                )
    tfm: dict[str, dict[str, tuple[float, float, float]]] = {}
    driver: dict[str, float] = {}
    if MFS_PANEL.is_file():
        mfs = json.loads(MFS_PANEL.read_text())
        for aud, block in mfs.get("retention_summary", {}).items():
            tfm[aud] = {
                m: (float(v["median"]), float(v["lo"]), float(v["hi"]))
                for m, v in block.get("methods", {}).items()
            }
        by_method: dict[str, list[float]] = {}
        for lin in mfs.get("lineages", ()):
            vals = sorted(
                float(s["holdout"]["driver_fidelity"])
                for s in lin.get("seeds", ())
                if isinstance(s.get("holdout", {}).get("driver_fidelity"), (int, float))
            )
            if vals:
                mid = len(vals) // 2
                med = vals[mid] if len(vals) % 2 == 1 else 0.5 * (vals[mid - 1] + vals[mid])
                by_method.setdefault(lin["method"], []).append(med)
        for m, vals in by_method.items():
            vals.sort()
            mid = len(vals) // 2
            driver[m] = vals[mid] if len(vals) % 2 == 1 else 0.5 * (vals[mid - 1] + vals[mid])
    return {"classical": classical, "tfm": tfm, "driver": driver}


def _rows(headline: dict) -> list[dict]:
    """Comparator rows for the forest plot: full panel, smaller cohorts, then controls."""
    rows = [arm for arm in headline["arms"] if arm["code"] in FIGURE_CODES and arm["contrasts"]["catboost"]]
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
    _forest_panels(axes, rows, margin, value_size=6.0, count_size=6.4)
    figure.supxlabel(f"DOPE minus comparator, Hodges-Lehmann retention difference ({INTERVAL_LEVEL}% nested interval)",
                     fontsize=figure_style_size(), y=0.01)
    figure.subplots_adjust(left=0.27, right=0.93, bottom=0.2, top=0.9, wspace=0.08)
    return _save(figure, path, dpi)


def _forest_panels(axes, rows: list[dict], margin: float, value_size: float, count_size: float) -> None:
    """One paired-difference panel per auditor, rows shared; the MLP panel prints each CatBoost cohort."""
    # A fixed window keeps the close comparisons readable; a row outside it is drawn at the edge
    # with an arrow and its value printed, so nothing is dropped.
    left, right = TEASER_X_LIM
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
            # Hollow marks a small cohort; an auditor's own count drops uninformative lineages.
            cohort = arm["contrasts"]["catboost"]["n"]
            hollow = cohort < FULL_PANEL
            lo, hi = max(left, row["hl_lo"]), min(right, row["hl_hi"])
            if hi > lo:
                axis.plot([lo, hi], [y, y], color=colour, linewidth=1.4, solid_capstyle="round", zorder=2)
            # An interval cut by the window ends in an arrowhead at that edge.
            for bound, edge, sign in ((row["hl_hi"], right, 1), (row["hl_lo"], left, -1)):
                if sign * (bound - edge) > 0 and left <= row["hl"] <= right:
                    axis.annotate("", xy=(edge, y), xytext=(edge - sign * 0.06, y),
                                  arrowprops={"arrowstyle": "-|>", "color": colour, "lw": 1.0,
                                              "shrinkA": 0, "shrinkB": 0, "mutation_scale": 7}, zorder=2)
            point = row["hl"]
            if left <= point <= right:
                axis.plot([point], [y], marker="o", markersize=4.6, zorder=3,
                          markerfacecolor="white" if hollow else colour, markeredgecolor=colour, markeredgewidth=1.0)
            else:
                edge = right if point > right else left
                axis.plot([edge], [y], marker=">" if point > right else "<", markersize=5.0, zorder=3,
                          color=colour)
                axis.annotate(f"{point:.2f}", xy=(edge, y), xytext=(-6 if point > right else 6, 0),
                              textcoords="offset points", ha="right" if point > right else "left", va="center",
                              fontsize=value_size, color=figure_style.INK,
                              bbox={"boxstyle": "square,pad=0.1", "facecolor": "white", "edgecolor": "none"})
            if auditor == "mlp":
                axis.annotate(f"n={cohort}", xy=(1.0, y), xycoords=("axes fraction", "data"),
                              xytext=(3, 0), textcoords="offset points", va="center", fontsize=count_size,
                              color=figure_style.INK)
        axis.set_xlim(left, right)
        axis.set_xticks(FOREST_TICKS)
        axis.set_title(title)
    axes[0].set_yticks(range(len(rows)))
    axes[0].set_yticklabels([arm["display"] for arm in reversed(rows)])
    axes[0].set_ylim(-TEASER_ROW_PAD, len(rows) - 1 + TEASER_ROW_PAD)


def _architecture_panel(axis, summary: dict, font_size: float, compact: bool = False) -> None:
    """Panel (b): Zero-shot tabular foundation models, classical learners, and holdout driver fidelity."""
    figure_style.panel(axis, grid="x")
    classical = summary["classical"]
    tfm = summary["tfm"]
    driver = summary["driver"]

    if compact:
        metrics = [
            ("Driver fidelity", "driver", "driver"),
            ("Mitra-v2 (TFM)", "tfm", "mitra_v2"),
            ("TabICLv2 (TFM)", "tfm", "tabicl2"),
            ("NVIDIA Kumo-L", "tfm", "kumo_tabular_l"),
        ]
    else:
        metrics = [
            ("CatBoost (4n)", "classical", "catboost"),
            ("Linear (4n)", "classical", "linear"),
            ("Neural MLP (4n)", "classical", "mlp"),
            ("Driver fidelity", "driver", "driver"),
            ("Mitra-v2 (TFM)", "tfm", "mitra_v2"),
            ("TabICLv2 (TFM)", "tfm", "tabicl2"),
            ("NVIDIA Kumo-L", "tfm", "kumo_tabular_l"),
        ]
    methods = [
        ("DOPE (1.9 KB)", ("DOPE", "headline"), "DOPE", ACCENT, ACCENT_EDGE, "o", 5.8, False, 4),
        ("TabSyn (42.3 MB)", ("TabSyn", "scaled_200_vae_1000_diffusion"), None, "#3B5B8C", "#3B5B8C", "s", 4.4, False, 3),
        ("ARF (1.4 MB)", ("ARF", "author_default"), None, "#756BB1", "#756BB1", "D", 4.2, False, 3),
        ("Copula (5.5 KB)", ("GaussianCopula", "native_selected"), "GaussianCopula", "#525252", "#525252", "^", 4.5, False, 2),
        ("Chow-Liu (29.9 KB)", ("Chow-Liu", "native_selected"), "Chow-Liu", "#8C8C8C", "#8C8C8C", "o", 4.2, True, 2),
    ]
    axis.axvline(1.0, color=CONTROL, linestyle=(0, (2, 2)), linewidth=0.8, zorder=1)
    for m_label, cls_key, tfm_key, face, edge, marker, msize, hollow, z in methods:
        xs, ys = [], []
        for idx, (_, kind, key) in enumerate(metrics):
            val = None
            if kind == "classical" and cls_key in classical and key in classical[cls_key]:
                val = classical[cls_key][key][0]
            elif kind == "tfm" and tfm_key and key in tfm and tfm_key in tfm[key]:
                val = tfm[key][tfm_key][0]
            elif kind == "driver" and tfm_key and tfm_key in driver:
                val = driver[tfm_key]
            if val is not None:
                xs.append(val)
                ys.append(idx)
        if xs:
            axis.plot(
                xs, ys, linestyle="none", marker=marker, markersize=msize,
                markerfacecolor="white" if hollow else face, markeredgecolor=edge,
                markeredgewidth=1.0, label=m_label, zorder=z
            )
    axis.set_yticks(range(len(metrics)))
    axis.set_yticklabels([m[0] for m in metrics], fontsize=font_size)
    axis.set_ylim(-TEASER_ROW_PAD, len(metrics) - 1 + TEASER_ROW_PAD)
    x_bounds = (0.31, 1.18)
    axis.set_xlim(*x_bounds)
    axis.set_xticks((0.4, 0.6, 0.8, 1.0, 1.1))
    axis.set_xlabel("holdout retention / feature driver agreement (1.0 = real-trained parity)", labelpad=1.2, fontsize=font_size)
    axis.set_title("(b) Zero-shot foundation models (TFM) & holdout driver fidelity" if compact else
                   "(b) Zero-shot foundation models (TFM), classical learners & driver fidelity",
                   loc="left", pad=2.0, fontweight="bold", fontsize=font_size)
    axis.legend(loc="lower left", frameon=True, facecolor="white", edgecolor="none",
                fontsize=font_size, handletextpad=0.3, borderpad=0.15, labelspacing=0.12)


def draw_hero(headline: dict, path: Path) -> Path:
    """Page-1 hero: two-panel Money Plot showing (a) utility vs. bytes and (b) multi-model & TFM transfer."""
    _, right = TEASER_X_LIM
    for arm in _rows(headline):
        if arm["code"] in HERO_OMITTED_ROWS and any(
                row is None or row["hl_lo"] <= right for row in (arm["contrasts"][key] for key, _ in AUDITORS)):
            raise ValueError(f"{arm['display']} is omitted from the hero but an interval reaches its window")
    summary = _multimodel_summary()
    floor = HERO_MIN_POINTS
    sizes = {key: floor for key in ("font.size", "axes.labelsize", "axes.titlesize", "xtick.labelsize",
                                     "ytick.labelsize", "legend.fontsize")}
    with matplotlib.rc_context(sizes):
        figure = plt.figure(figsize=(figure_style.TEASER_WIDTH_IN, HERO_HEIGHT_IN))
        axis_a = figure.add_axes([0.105, 0.565, 0.87, 0.38])
        axis_b = figure.add_axes([0.215, 0.13, 0.76, 0.245])
        _frontier(axis_a, headline, cap_size=floor, label_size=floor, dope_size=floor + 0.6, control_size=floor,
                  offsets=HERO_LABEL_OFFSETS, names=HERO_NAMES,
                  summary=summary)
        axis_a.set_xlabel("median charged artifact bytes (log scale; bar = interquartile range)", labelpad=1.0)
        axis_a.set_ylabel("Retention (4n)")
        axis_a.set_title("(a) Retention at 4n vs. artifact bytes (DOPE 1.9 KB: \u25b2 MLP, \u25a0 Linear, \u25cf CatBoost)",
                         loc="left", pad=2.0, fontweight="bold")
        _architecture_panel(axis_b, summary, font_size=floor, compact=True)
        return _save(figure, path, None)


def text_sizes(figure) -> list[tuple[str, float]]:
    """Every visible non-empty text in a drawn figure with its size in points."""
    found = []
    for artist in figure.findobj(matplotlib.text.Text):
        if artist.get_visible() and artist.get_text().strip():
            found.append((artist.get_text(), float(artist.get_fontsize())))
    return found


def figure_style_size() -> float:
    return matplotlib.rcParams["axes.labelsize"]


# Candidate label offsets in points, tried in order; the first that clears every placed label wins.
LABEL_OFFSETS = ((4, 3, "left", "bottom"), (4, -3, "left", "top"), (-4, 3, "right", "bottom"),
                 (-4, -3, "right", "top"), (0, 7, "center", "bottom"), (0, -7, "center", "top"),
                 (6, 9, "left", "bottom"), (6, -9, "left", "top"), (-6, 9, "right", "bottom"),
                 (-6, -9, "right", "top"))
CONTROL_POSITIONS = (0.62, 0.98, 0.02, 0.26, 0.74)
# The hero's labels are larger, so they are shorter and also try offsets one label height further out.
# Panel (b) and the results figure keep the full names, including each configuration.
HERO_NAMES = {"DopeDef": "DOPE untuned", "TabSyn": "TabSyn", "Arf": "ARF", "Forest": "Forest-Flow", "Tvae": "TVAE",
              "Ctgan": "CTGAN", "Gauss": "Gaussian copula", "Chow": "Chow-Liu", "Ind": "Independent",
              "Pred": "Predictor-only control", "Boot": "Real rows resampled"}
HERO_LABEL_OFFSETS = LABEL_OFFSETS + ((7, 0, "left", "center"), (-7, 0, "right", "center"), (6, -14, "left", "top"),
                                      (-6, -14, "right", "top"), (8, 14, "left", "bottom"), (-8, 14, "right", "bottom"),
                                      (0, -16, "center", "top"), (0, 16, "center", "bottom"), (14, -6, "left", "top"),
                                      (-14, -6, "right", "top"), (14, 6, "left", "bottom"), (-14, 6, "right", "bottom"))


def _place(axis, text: str, anchors, placed: list, avoid=(), **style):
    """Annotate at the first anchor and offset whose box stays in the axes and clears placed labels and marks."""
    renderer = axis.figure.canvas.get_renderer()
    frame = axis.get_window_extent(renderer)
    first = None
    for xy, xycoords, offsets in anchors:
        for dx, dy, ha, va in offsets:
            label = axis.annotate(text, xy=xy, xycoords=xycoords, xytext=(dx, dy), textcoords="offset points",
                                  ha=ha, va=va, **style)
            box = label.get_window_extent(renderer).expanded(1.04, 1.1)
            inside = (box.x0 >= frame.x0 and box.x1 <= frame.x1 and box.y0 >= frame.y0 and box.y1 <= frame.y1)
            if inside and not any(box.overlaps(other) for other in (*placed, *avoid)):
                placed.append(box)
                if first is not None:
                    first.remove()
                return label
            if first is None:
                first = label
            else:
                label.remove()
    # Nothing clears: keep the first candidate rather than drop the label.
    placed.append(first.get_window_extent(renderer))
    return first


def draw_pareto(headline: dict, path: Path, width: float = 5.5, dpi: int | None = None,
                floor: float | None = None) -> Path:
    summary = _multimodel_summary()
    if floor is None:
        figure = plt.figure(figsize=(width, 4.7))
        axis_a = figure.add_axes([0.09, 0.55, 0.88, 0.39])
        axis_b = figure.add_axes([0.18, 0.11, 0.79, 0.33])
        _frontier(axis_a, headline, cap_size=7.2, label_size=7.0, dope_size=7.8, control_size=7.0,
                  offsets=HERO_LABEL_OFFSETS, names=HERO_NAMES, summary=summary)
        axis_a.set_xlabel("median charged artifact bytes (log scale; bar = interquartile range)", fontsize=7.6)
        axis_a.set_ylabel("Retention at 4n", fontsize=7.6)
        axis_a.set_title("(a) Multi-model retention at 4n vs. artifact bytes (DOPE 1.9 KB: \u25b2 MLP, \u25a0 Linear, \u25cf CatBoost)",
                         loc="left", pad=3.5, fontweight="bold", fontsize=8.2)
        _architecture_panel(axis_b, summary, font_size=7.4)
        return _save(figure, path, dpi)
    figure, axis = plt.subplots(figsize=(width, 2.7))
    # The frame is final before any label is measured.
    figure.subplots_adjust(left=0.1, right=0.98, bottom=0.17, top=0.95)
    # The paper figure prints at the text width, so its labels keep the page-1 floor.
    _frontier(axis, headline, cap_size=floor, label_size=floor, dope_size=floor + 0.6, control_size=floor,
              offsets=HERO_LABEL_OFFSETS)
    axis.set_xlabel("median charged artifact bytes (log scale; bar = interquartile range)")
    axis.set_ylabel("CatBoost retention at 4n")
    return _save(figure, path, dpi)


def _frontier(axis, headline: dict, cap_size: float, label_size: float, dope_size: float,
              control_size: float, offsets=LABEL_OFFSETS, names=None, dope_offsets=None,
              summary: dict | None = None) -> None:
    """Retention at 4n against median charged bytes, labels placed after every mark and limit."""
    figure = axis.figure
    figure_style.panel(axis, grid="both")
    axis.set_xscale("log")
    axis.axvline(CAP, color=figure_style.SPINE, linestyle=(0, (4, 3)), linewidth=0.8, zorder=1)
    # The untuned default sits beside the reported profile, so the byte axis shows what the profile choice adds.
    drawn = [arm for arm in headline["arms"] if arm["code"] in ("Dope", "DopeDef") or arm["code"] in FIGURE_CODES]
    plotted = [arm for arm in drawn if (arm["role"] in {"headline", "generator"} or arm["code"] == "DopeDef")
               and arm["bytes"] is not None]
    controls = [arm for arm in drawn if arm["role"] == "control"]
    xs = [arm["bytes"][key] for arm in plotted for key in ("q1", "q3")]
    ys = [value for arm in plotted for value in (arm["level"]["lo"], arm["level"]["hi"], arm["level"]["median"])
          if value is not None] + [arm["level"]["median"] for arm in controls]
    if xs:
        axis.set_xlim(10 ** math.floor(math.log10(min(xs)) - 0.2), 10 ** (math.ceil(math.log10(max(xs))) + 0.3))
    if ys:
        pad = 0.08 * (max(ys) - min(ys) or 1.0)
        axis.set_ylim(min(ys) - pad, max(ys) + 2.2 * pad)
    for arm in controls:
        axis.axhline(arm["level"]["median"], color=CONTROL, linestyle=(0, (2, 2)), linewidth=0.9, zorder=1)
    marks: dict[str, list] = {}
    dope_cls = (summary or {}).get("classical", {}).get(("DOPE", "headline"), {})
    for arm in plotted:
        level, size = arm["level"], arm["bytes"]["median"]
        is_dope = arm["code"] == "Dope"
        colour = ACCENT if is_dope else (DEFAULT_GREEN if arm["code"] == "DopeDef" else CONTEXT)
        hollow = (level["n"] or 0) < FULL_PANEL
        drawn_marks = axis.plot([arm["bytes"]["q1"], arm["bytes"]["q3"]], [level["median"]] * 2, color=colour,
                                linewidth=1.1, alpha=0.8, zorder=2)
        if level["lo"] is not None:
            drawn_marks += axis.plot([size, size], [level["lo"], level["hi"]], color=colour, linewidth=1.1, zorder=2)
        drawn_marks += axis.plot([size], [level["median"]], marker="o", markersize=6.5 if is_dope else 4.8, zorder=3,
                                 markerfacecolor="white" if hollow else colour,
                                 markeredgecolor=ACCENT_EDGE if is_dope else colour, markeredgewidth=1.0)
        if is_dope and dope_cls:
            if "mlp" in dope_cls:
                drawn_marks += axis.plot([size, size], [level["median"], dope_cls["mlp"][0] + 0.04],
                                         color=ACCENT, linewidth=1.0, linestyle=(0, (1.5, 1.5)), zorder=2)
            for aud, mk in (("linear", "s"), ("mlp", "^")):
                if aud in dope_cls:
                    drawn_marks += axis.plot([size], [dope_cls[aud][0]], marker=mk, markersize=4.8, zorder=4,
                                             markerfacecolor=ACCENT, markeredgecolor=ACCENT_EDGE, markeredgewidth=0.9)
        marks[arm["code"]] = drawn_marks
    renderer = figure.canvas.get_renderer()
    boxes = {code: [line.get_window_extent(renderer).padded(2.5) for line in lines] for code, lines in marks.items()}
    # Labels go on after every mark and limit is fixed, DOPE first, then by retention.
    placed: list = []
    _place(axis, f"{CAP:,}-byte cap", [((CAP, 1.0), ("data", "axes fraction"), ((3, -3, "left", "top"),))],
           placed, fontsize=cap_size, color=figure_style.INK)
    order = sorted(plotted, key=lambda arm: (arm["code"] != "Dope", -(arm["level"]["median"] or 0.0)))
    for arm in order:
        is_dope = arm["code"] == "Dope"
        others = [box for code, owned in boxes.items() if code != arm["code"] for box in owned]
        _place(axis, (names or {}).get(arm["code"], arm["display"]),
               [((arm["bytes"]["median"], arm["level"]["median"]), "data",
                 (dope_offsets or offsets) if is_dope else offsets)],
               placed, avoid=others, fontsize=dope_size if is_dope else label_size, color=figure_style.INK,
               fontweight="bold" if is_dope else "normal")
    for arm in controls:
        anchors = [((fraction, arm["level"]["median"]), ("axes fraction", "data"),
                    ((0, 2, "left" if fraction < 0.1 else ("right" if fraction > 0.9 else "center"), "bottom"),
                     (0, -2, "left" if fraction < 0.1 else ("right" if fraction > 0.9 else "center"), "top")))
                   for fraction in CONTROL_POSITIONS]
        _place(axis, (names or {}).get(arm["code"], arm["display"]), anchors, placed, avoid=[box for owned in boxes.values() for box in owned],
               fontsize=control_size, color=figure_style.INK)


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
        "teaser.pdf": draw_hero(headline, place("teaser.pdf")),
        "pareto.pdf": draw_pareto(headline, place("pareto.pdf"), floor=HERO_MIN_POINTS),
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
