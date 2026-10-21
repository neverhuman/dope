"""Draw the authenticated historical size/transfer and foundation panels.

The renderer supplies canonical rows and interval levels. This module only
validates display cohorts and draws; it fits no model and computes no metric.
"""
from __future__ import annotations

import csv
import io

def compression_rows(csv_bytes: bytes) -> list[dict]:
    """Keep the declared, comparable classical cohort; never pool smaller panels.

    This is a display of historical aggregate points, not a fitted frontier or
    a per-dataset dominance test. Include the untuned product default so the
    selected research profile cannot be mistaken for out-of-box behavior.
    """
    rows = list(csv.DictReader(io.StringIO(csv_bytes.decode())))
    ids = ("Dope", "DopeDef", "TabSyn", "Arf", "ArfNat", "Gauss", "Chow", "Ind")
    selected = []
    for key in ids:
        matches = [row for row in rows if row["id"] == key]
        if len(matches) != 1:
            raise ValueError("compression figure: missing or duplicate method")
        row = matches[0]
        if (row["cohort_id"] != "legacy-v2-catboost-99" or row["observed_lineages"] != "99"
                or row["prepared_lineages"] != "100" or row["cohort_complete"] != "true"
                or row["evidence_status"] != "selection_exposed_validation"):
            raise ValueError("compression figure: incomparable cohort")
        values = [float(row[key]) for key in ("artifact_bytes", "retention", "interval_lo", "interval_hi")]
        size, retention, low, high = values
        if not (900 <= size <= 1e8 and -0.1 <= low <= retention <= high <= 1.15):
            raise ValueError("compression figure: out-of-axis value")
        selected.append(row)
    return selected


def draw_hero(f_summary: dict, six_bars: list[dict], percent: int, *, profile: str,
                          classical_rows: list[dict], classical_percent: int):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    # Match the full manuscript text width: fonts remain legible at print size.
    width_in, height_in = (6.50, 3.55) if profile == "paper" else (10.0, 5.46)
    base_pt = 9.5 if profile == "paper" else 13.1
    footer_pt = base_pt - 0.3
    matplotlib.rcParams.update({"svg.hashsalt": "dope-historical-validation-v1", "font.family": "DejaVu Sans", "font.size": base_pt})
    figure = plt.figure(figsize=(width_in, height_in))
    bottom, height = (0.38, 0.41) if profile == "paper" else (0.33, 0.46)
    rate_axis = figure.add_axes((0.10, bottom, 0.46, height))
    axis = figure.add_axes((0.765, bottom, 0.20, height))

    labels = {
        "Dope": ("DOPE research", (10, 15)),
        "DopeDef": ("DOPE default", (7, -30)),
        "TabSyn": ("TabSyn", (-42, 15)),
        "Arf": ("ARF default", (-55, 18)),
        "ArfNat": ("ARF selected", (7, -23)),
        "Gauss": ("Copula", (7, -1)),
        "Chow": ("Chow-Liu", (7, -1)),
        "Ind": ("Independent", (7, 5)),
    }
    if profile == "paper":
        labels["Arf"] = ("ARF default", (-8, -17))
        labels["ArfNat"] = ("ARF selected", (8, -38))
        labels["Dope"] = ("DOPE research", (6, 15))
        labels["TabSyn"] = ("TabSyn", (-35, 10))
    for row in classical_rows:
        key = row["id"]
        size, val, lo, hi = [float(row[k]) for k in ("artifact_bytes", "retention", "interval_lo", "interval_hi")]
        color = "#21664d" if key.startswith("Dope") else "#596879"
        rate_axis.errorbar(size, val, yerr=[[val - lo], [hi - val]], fmt="o",
                           color=color, ecolor=color, capsize=2, markersize=5 if key == "Dope" else 4,
                           markerfacecolor="white" if key == "DopeDef" else color, linewidth=0.9, zorder=3)
        label, offset = labels[key]
        rate_axis.annotate(label, (size, val), xytext=offset, textcoords="offset points",
                           fontsize=base_pt - 0.5, weight="bold" if key == "Dope" else "normal",
                           arrowprops={"arrowstyle": "-", "lw": 0.5, "color": color})
    rate_axis.set_xscale("log")
    rate_axis.set_xlim(900, 1e8)
    rate_axis.set_xticks([1e3, 1e4, 1e5, 1e6, 1e7, 1e8])
    rate_axis.set_xticklabels(["1 KB", "10 KB", "100 KB", "1 MB", "10 MB", "100 MB"])
    rate_axis.tick_params(axis="x", labelsize=base_pt - 0.5)
    rate_axis.set_ylim(-0.1, 1.15)
    rate_axis.set_yticks([0, 0.5, 1.0], ["0%", "50%", "100%"])
    rate_axis.axhline(1.0, color="#a4abb1", lw=0.7, ls="--")
    rate_axis.set_ylabel("Lift retained")
    rate_axis.set_xlabel("Charged fitted state (log scale; decimal units)", fontsize=base_pt - 0.5)
    rate_axis.set_title("A  Matched CatBoost validation", loc="left", fontsize=base_pt + 0.3, weight="bold", pad=9)
    rate_axis.grid(alpha=0.15)
    rate_axis.spines[["top", "right"]].set_visible(False)
    by_id = {row["id"]: row for row in classical_rows}
    dope, tabsyn = by_id["Dope"], by_id["TabSyn"]
    ratio = float(tabsyn["artifact_bytes"]) / float(dope["artifact_bytes"])
    figure.text(0.10, 0.18, f"DOPE / TabSyn median state: {ratio:,.0f}x smaller.\n"
                "CatBoost: inconclusive, not equivalence.",
                fontsize=footer_pt, color="#243543", linespacing=1.4, va="top")

    y_positions = [5.10, 4.42, 3.10, 2.42, 1.10, 0.42]
    group_centers = [4.76, 2.76, 0.76]
    group_labels = [bar["auditor_display"] for bar in six_bars[::2]]
    dope_bytes, copula_bytes = six_bars[0]["matched_artifact_bytes"], six_bars[1]["matched_artifact_bytes"]

    for y_pos, bar in zip(y_positions, six_bars):
        is_dope = bar["method"] == "DOPE"
        color, hatch = ("#2f6b25", "") if is_dope else ("#7a7a7a", "//")
        val, lo, hi = bar["retention_pct"], bar["interval_lo_pct"], bar["interval_hi_pct"]
        rect = axis.barh(y_pos, val, height=0.54, color=color, edgecolor="#1a1a1a", linewidth=0.85, zorder=2)[0]
        if hatch:
            rect.set_hatch(hatch)
        axis.errorbar(val, y_pos, xerr=[[val - lo], [hi - val]], fmt="none",
                      ecolor="#111111", elinewidth=1.25, capsize=3.0, capthick=1.0, zorder=3)
        label_text = f"{val:.1f}%"
        axis.text(3.0, y_pos, label_text, va="center", ha="left",
                  fontsize=base_pt - 0.5, fontweight="bold" if is_dope else "normal",
                  color="#ffffff" if is_dope else "#111111", zorder=4,
                  bbox=None if is_dope else {"facecolor": "#dddddd", "edgecolor": "none", "pad": 0.2})

    axis.set_xlim(0.0, 100.0)
    axis.set_xticks([0, 50, 100])
    axis.set_xticklabels(["0%", "50%", "100%"], fontsize=base_pt - 0.2)
    axis.set_ylim(-0.15, 5.75)
    axis.set_yticks(group_centers)
    axis.set_yticklabels(group_labels, fontsize=base_pt - 0.5, fontweight="bold")
    axis.set_xlabel("Lift retained", fontsize=base_pt - 0.5, labelpad=4)
    axis.set_title("B  Foundation models", loc="center", fontsize=base_pt + 0.3, weight="bold", pad=9)
    figure.suptitle("Preserve predictive lift. Charge every byte.", x=0.50, y=0.965,
                   ha="center", fontsize=base_pt + 2, fontweight="bold")
    figure.text(0.5, 0.865, f"Selection-exposed validation | A: historical {classical_percent}% interval; B: historical {percent}% interval",
                ha="center", fontsize=base_pt - 0.3, color="#43505c")
    axis.grid(axis="x", alpha=0.22, zorder=0)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

    handles = [
        Patch(facecolor="#2f6b25", edgecolor="#1a1a1a", linewidth=0.85, label=f"DOPE ({dope_bytes:,} B)"),
        Patch(facecolor="#7a7a7a", edgecolor="#1a1a1a", linewidth=0.85, hatch="//", label=f"Gaussian copula ({copula_bytes:,} B)"),
    ]
    figure.legend(handles=handles, loc="center", bbox_to_anchor=(0.79, 0.16), ncol=1,
                  fontsize=base_pt - 0.5, frameon=False, handlelength=1.2, labelspacing=0.4)
    figure.text(0.50, 0.018, "A: 99 lineages; B: separate 97-lineage cohort; both 4n. TDL and cold closure unmeasured.\n"
                             "Cross-model feature-driver preservation: not yet measured. Pretraining overlap unknown.",
                ha="center", va="bottom", fontsize=footer_pt, color="#222222", linespacing=1.18)
    return figure, plt
