"""Render figures, the README result block and paper macros from a summary.

    python3 bench/real-world/report.py --results bench/real-world/results/confirm
    python3 bench/real-world/render.py --results bench/real-world/results/confirm

Every number shown in the README block, the paper and the figures comes
from ``summary.json`` (itself computed from ``records.jsonl``); nothing is
typed by hand. ``--check`` re-renders in memory and fails on any drift.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_tex import (  # noqa: E402
    BUDGET_ORDER, ROOT, config_macros, kb, macros, multilevel_convergence_data, multilevel_readme_lines,
    ret, row, table_ablation, table_equal_bytes, table_main, table_multilevel, table_per_dataset,
    tdl_stats, throughput_tex,
)

BEGIN, END = "<!-- BEGIN DOPE1 RESULTS -->", "<!-- END DOPE1 RESULTS -->"
# Palette (validated categorical slots 1-2 + neutral; see the dataviz method).
BLUE, ORANGE, GRAY, INK, INK2, GRID, SURFACE = "#2a78d6", "#eb6834", "#8a8984", "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
GENERATORS = [("copula", "Gaussian copula"), ("arf", "ARF"), ("tvae", "TVAE"), ("ctgan", "CTGAN")]
GRAYS = ["#b8b7b2", "#8a8984", "#62615d", "#3d3c39"]  # neutral steps for the generator bars


def frontier(summary: dict, method: str) -> list[dict]:
    rows = [row(summary, method, b) for b in BUDGET_ORDER]
    return [r for r in rows if r and r["datasets"] >= max(3, len(summary["datasets"]) // 2)]



def setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9.5, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
        "xtick.color": INK2, "ytick.color": INK2, "axes.facecolor": SURFACE, "figure.facecolor": SURFACE,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
        "axes.spines.right": False, "svg.hashsalt": "dope", "pdf.fonttype": 42,
    })
    return plt


FLOOR = -0.25  # y-axis floor; medians below it are drawn on it with a downward marker


def kib_ticks(ax, lo: float, hi: float) -> None:
    """Powers-of-four byte ticks (1 KiB, 4 KiB, ...) spanning [lo, hi]."""
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
    ticks = [256.0 * 4 ** k for k in range(14) if lo / 2 <= 256.0 * 4 ** k <= hi * 2]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: kb(v).replace(".0 ", " ")))


def hero(summary: dict, plt, title: bool = True):
    # The paper version is smaller on the page, so it is drawn smaller to keep text legible.
    fig, ax = plt.subplots(figsize=(7.2, 4.1) if title else (6.3, 3.6))
    ax.set_xscale("log")
    xs_all = []
    ends = {}
    for method, color, label in (("rows", ORANGE, "Real rows, xz-compressed\nto the same bytes"), ("dope", BLUE, "DOPE")):
        pts = frontier(summary, method)
        xs = [p["bytes"]["median"] for p in pts]
        ys = [p["robust"]["median"] for p in pts]
        lo = [max(p["robust"]["lo"], FLOOR) for p in pts]
        hi = [max(p["robust"]["hi"], FLOOR) for p in pts]
        yc = [max(y, FLOOR) for y in ys]
        xs_all += xs
        ax.fill_between(xs, lo, hi, color=color, alpha=0.14 if method == "dope" else 0.07, linewidth=0)
        ax.plot(xs, yc, color=color, linewidth=2.2, solid_capstyle="round", zorder=3)
        for x, y, c in zip(xs, ys, yc):
            clipped = y < FLOOR
            ax.scatter([x], [c], s=60 if clipped else 46, marker="v" if clipped else "o", color=color,
                       edgecolor=SURFACE, linewidth=1.8, zorder=4)
            if clipped:
                ax.annotate(f"{y:.1f}".replace("-", "−"), (x, c), xytext=(0, 7), textcoords="offset points",
                            ha="center", color=INK2, fontsize=7.5)
        if xs:
            ends[method] = (xs[-1], yc[-1], color, label)
    for method, (x, y, color, label) in ends.items():
        ax.annotate(label, (x, y), xytext=(8, -2 if method == "dope" else -14), textcoords="offset points",
                    color=color, fontsize=9, fontweight="bold", va="center")
    head = row(summary, "dope", "4k")
    if head and head["datasets"]:
        x, y = head["bytes"]["median"], head["robust"]["median"]
        ax.annotate(f"default program\n{kb(x)}, retention {y:.2f}", (x, y), xytext=(18, -38),
                    textcoords="offset points", ha="left", color=INK, fontsize=8.5,
                    arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    clipped = 0
    for method, name in GENERATORS:
        r = row(summary, method, "-")
        if not r or r["datasets"] == 0:
            continue
        x, y = r["bytes"]["median"], r["robust"]["median"]
        xs_all.append(x)
        yc = max(y, FLOOR)
        ax.errorbar([x], [yc], yerr=[[max(0, yc - max(r["robust"]["lo"], FLOOR))], [max(0, max(r["robust"]["hi"], FLOOR) - yc)]],
                    fmt="v" if y < FLOOR else "o", color=GRAY, ms=6.5, mec=SURFACE, mew=1.5, elinewidth=1.2,
                    capsize=0, zorder=3)
        suffix = "" if r["datasets"] == len(summary["datasets"]) else f" ({r['datasets']} tables)"
        if y < FLOOR:  # below the axis: print the value, alternating sides so labels do not collide
            clipped += 1
            left = clipped % 2 == 1
            ax.annotate(f"{name}{suffix}: {y:.2f}".replace("-", "−"), (x, yc), xytext=(-7 if left else 7, 9),
                        textcoords="offset points", ha="right" if left else "left", color=INK2, fontsize=8)
        else:
            ax.annotate(f"{name}{suffix}", (x, yc), xytext=(7, 4), textcoords="offset points", color=INK2, fontsize=8)
    k = row(summary, "kernel", "-")
    if k and k["datasets"]:
        x, y = k["bytes"]["median"], k["robust"]["median"]
        xs_all.append(x)
        yc = max(y, FLOOR)
        ax.errorbar([x], [yc], yerr=[[max(0, yc - max(k["robust"]["lo"], FLOOR))], [max(0, max(k["robust"]["hi"], FLOOR) - yc)]],
                    fmt="D", color=BLUE, mfc=SURFACE, mec=BLUE, ms=6.5, mew=1.6, elinewidth=1.2, capsize=0, zorder=4)
        ax.annotate("research kernel\n(dope-kernel)", (x, yc), xytext=(-8, -4), textcoords="offset points",
                    ha="right", va="top", color=INK2, fontsize=8)
    ax.axhline(0.9, color=INK2, linewidth=0.9, linestyle=(0, (1, 2.5)), zorder=1)
    ax.text(0.995, 0.9, "τ = 0.9", transform=ax.get_yaxis_transform(), va="bottom", ha="right", color=INK2, fontsize=8)
    ax.text(0.015, 0.97, "↖ better: more retention, fewer bytes", transform=ax.transAxes, ha="left", va="top",
            color=INK2, fontsize=8)
    ax.set_ylim(FLOOR - 0.04, 1.08)
    if xs_all:
        kib_ticks(ax, min(xs_all), max(xs_all))
        ax.set_xlim(min(xs_all) / 3.2, max(xs_all) * 6)  # room for the kernel's label on the left
    ax.set_xlabel("Bytes that sampling reads (log scale)")
    ax.set_ylabel("Robust retention\n(worst of linear, CatBoost, MLP)")
    if title:
        n = len(summary["datasets"])
        ax.set_title(f"Predictive structure per byte, median over {n} public tables", loc="left", color=INK,
                     fontsize=11)
    fig.tight_layout()
    return fig


def coverage(summary: dict, plt):
    import numpy as np
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    ax.set_xscale("log")
    n = len(summary["datasets"])
    for method, color, name in (("dope", BLUE, "DOPE"), ("rows", ORANGE, "Real rows (xz)")):
        for tau, style in ((0.9, "-"), (0.8, (0, (4, 2)))):
            vals = sorted(v for v in summary["tdl"][f"{method}@{tau}"].values() if v is not None)
            xs = np.array([256.0] + vals + [vals[-1] * 4 if vals else 1e7])
            ys = np.array([0] + list(range(1, len(vals) + 1)) + [len(vals)]) / n
            ax.step(xs, ys, where="post", color=color, linewidth=2, linestyle=style, label=f"{name}, τ = {tau}")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Bytes (TDL: smallest tested artifact reaching retention τ)")
    ax.set_ylabel("Share of tables reached")
    lo, hi = ax.get_xlim()
    kib_ticks(ax, lo, hi)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    fig.tight_layout()
    return fig


def families(summary: dict, plt):
    import numpy as np
    head = row(summary, "dope", "4k")
    rows4 = row(summary, "rows", "4k")
    others = [(m, n) for m, n in GENERATORS if row(summary, m, "-")]
    groups = [("DOPE @ 4 KiB", head, BLUE), ("Rows @ 4 KiB", rows4, ORANGE)] + [(n, row(summary, m, "-"), GRAY) for m, n in others]
    fams = ["linear", "gbdt", "mlp"]
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    width = 0.8 / len(groups)
    for g, (name, r, color) in enumerate(groups):
        vals = [max(-0.2, r["families"].get(f, np.nan)) if r else np.nan for f in fams]
        xs = np.arange(len(fams)) + (g - (len(groups) - 1) / 2) * width
        shade = color if color != GRAY else GRAYS[(g - 2) % len(GRAYS)]
        ax.bar(xs, vals, width * 0.86, color=shade, label=name)
    ax.set_xticks(range(len(fams)), ["Linear / logistic", "CatBoost", "MLP"])
    ax.set_ylabel("Median retention")
    ax.set_ylim(-0.2, 1.1)
    ax.axhline(0, color=INK2, linewidth=0.8)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    fig.tight_layout()
    return fig


def tables(summary: dict, plt):
    """Per table at the default budget: DOPE against real rows compressed to the same bytes."""
    pdd = summary["per_dataset"]
    dope, rows = pdd.get("dope@4k", {}), pdd.get("rows@4k", {})
    names = [d for d in summary["datasets"] if (dope.get(d) or {}).get("robust") is not None]
    names.sort(key=lambda d: dope[d]["robust"])
    fig, ax = plt.subplots(figsize=(6.4, 0.32 * len(names) + 1.1))
    for i, d in enumerate(names):
        a = dope[d]["robust"]
        b = (rows.get(d) or {}).get("robust")
        if b is not None:
            ax.plot([max(b, FLOOR), max(a, FLOOR)], [i, i], color=GRID, linewidth=2.2, zorder=1)
            ax.scatter([max(b, FLOOR)], [i], s=36, marker="v" if b < FLOOR else "o", color=ORANGE,
                       edgecolor=SURFACE, linewidth=1.2, zorder=2)
        if dope[d].get("status") == "failed":  # the schema alone exceeded the budget: no program
            ax.scatter([0], [i], s=40, marker="x", color=BLUE, linewidth=1.6, zorder=3)
            ax.annotate("did not fit in 4 KiB", (0, i), xytext=(7, -3), textcoords="offset points",
                        color=INK2, fontsize=7.5)
        else:
            ax.scatter([max(a, FLOOR)], [i], s=44, color=BLUE, edgecolor=SURFACE, linewidth=1.2, zorder=3)
    ax.set_yticks(range(len(names)), names)
    ax.axvline(0.9, color=INK2, linewidth=0.9, linestyle=(0, (1, 2.5)))
    ax.set_xlim(FLOOR - 0.04, 1.08)
    ax.set_xlabel("Robust retention at 4 KiB (worst of linear, CatBoost, MLP)")
    ax.grid(axis="y", visible=False)
    ax.scatter([], [], color=BLUE, s=36, label="DOPE")
    ax.scatter([], [], color=ORANGE, s=36, label="Real rows, xz-compressed to the same bytes")
    ax.legend(loc="lower left", frameon=False, fontsize=8, bbox_to_anchor=(0, 1.0), ncol=2)
    fig.tight_layout()
    return fig


def audit(summary: dict, plt):
    """Compile-time self-audit against measured robust retention, one point per fitted DOPE cell."""
    au = summary.get("self_audit") or {"pairs": []}
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    pts = [(a, max(m, FLOOR)) for a, m in au["pairs"]]
    if pts:
        a, m = zip(*pts)
        ax.scatter(a, m, s=16, color=BLUE, alpha=0.7, edgecolor="none")
    ax.plot([FLOOR, 1.1], [FLOOR, 1.1], color=INK2, linewidth=0.9, linestyle=(0, (1, 2.5)))
    ax.set_xlim(FLOOR, 1.1)
    ax.set_ylim(FLOOR - 0.04, 1.1)
    ax.set_xlabel("Self-audit at compile time")
    ax.set_ylabel("Measured robust retention")
    fig.tight_layout()
    return fig


def convergence(summary: dict, plt):
    """DOPE-MLC-v2 and DOPE-MLC-v3 training and <5% full-validation loss curves across Training Rounds R0..R8."""
    import numpy as np
    snap = multilevel_convergence_data()
    v2_old = snap["v2_full_corpus"]
    eps_old, full_old = v2_old["epochs"], v2_old["full_1891_val_final_reloaded_ep7_best_state"]
    t1 = snap["v2_full_corpus_maxtrain97_val3pct"]["rounds"]
    t2 = snap["v2_disjoint1000_maxtrain97_devval345"]["rounds"]
    t3_a3 = snap["v3_maxtrain97_val3pct"]["a3_rounds"]
    t3_vars = snap["v3_maxtrain97_val3pct"]["matched_budget_ablation_progression_A0_to_A3"]
    xs = np.array([r["raw_receipt_epoch_field"] for r in t1], dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.2))
    ax_a, ax_b, ax_c, ax_d = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]
    # Panel A: Track 1 <5% Val (684/684 splits, 115/3,188 DS) vs Preserved 10%-Val
    t1_val = np.array([r["val_full"]["mean_total_loss"] for r in t1])
    t1_se = np.array([r["val_full"]["dataset_grouped_se"] for r in t1])
    t1_tr = np.array([r["train_monitor_600"]["mean_total_loss"] for r in t1])
    old_val = np.array([e["val_monitor_600"]["mean_total_loss"] for e in eps_old])
    xf = 8.85
    ax_a.fill_between(xs, t1_val - t1_se, t1_val + t1_se, color=BLUE, alpha=0.16, lw=0)
    ax_a.plot(xs, t1_val, marker="s", ms=4.2, lw=2.0, color=BLUE, label="Full 684/684 val ±1 SE (3.59% splits, R8* 0.34216, -16.5%)")
    ax_a.plot(xs, t1_tr, marker="o", ms=3.8, lw=1.6, color="#15803d", ls="--", label="Train monitor (600/18,395 splits, R8 0.35175)")
    ax_a.plot(xs, old_val, marker="^", ms=3.8, lw=1.5, color=GRAY, ls=":", label="Preserved 10%-val monitor (600/1,891, R7* 0.37667)")
    ax_a.plot([xf], [full_old["mean_total_loss"]], marker="D", ms=5.5, color="#b91c1c", ls="none", label="Preserved full 1,891 eval (R7*: 0.37753)")
    ax_a.axvline(8.42, color=GRAY, ls=":", lw=1.0)
    ax_a.set_xticks(list(range(9)) + [xf], ["R0", "R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8*", "Full\n1891"])
    ax_a.set_xlabel("Training Round (R1..R8 = 24,527 visits, 1.333× passes)")
    ax_a.set_ylabel("Multi-task objective loss")
    ax_a.set_title("A. DOPE-MLC-v2 Full Corpus (<5% Val: 684 Splits / 115 DS)", loc="left", color=INK, fontsize=8.6)
    ax_a.legend(frameon=False, fontsize=6.5, loc="upper right")
    # Panel B: Track 1 Per-Tier Hierarchy (<5% Val, 684/684 splits)
    for key, col, mk, lbl in (
        ("medium", ORANGE, "^", "Val Medium (4 KiB, R8* 0.36407 ± 0.01608)"),
        ("high", BLUE, "o", "Val High (16 KiB, R8* 0.33844 ± 0.01635)"),
        ("ultra", "#15803d", "*", "Val Ultra (64 KiB, R8* 0.32396 ± 0.01672)"),
    ):
        ys = np.array([r["val_full"][key]["total_loss"] for r in t1])
        se = np.array([r["val_full"][key]["dataset_grouped_se"] for r in t1])
        ax_b.fill_between(xs, ys - se, ys + se, color=col, alpha=0.11, lw=0)
        ax_b.plot(xs, ys, marker=mk, ms=5.5 if mk == "*" else 4.2, lw=1.9, color=col, label=lbl)
    ax_b.set_xticks(range(9), ["R0", "R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8*"])
    ax_b.set_xlabel("Training Round (100% of 684 unseen splits evaluated/round)")
    ax_b.set_ylabel("Unseen-dataset validation loss (±1 SE)")
    ax_b.set_title("B. DOPE-MLC-v2 Per-Tier Hierarchy (Ultra < High < Medium)", loc="left", color=INK, fontsize=8.6)
    ax_b.legend(frameon=False, fontsize=6.7, loc="upper right")
    # Panel C: Track 2 Strictly Disjoint-1,000 (<5% Val: 345/345 dev_val splits, 58 DS)
    t2_val = np.array([r["val_full"]["mean_total_loss"] for r in t2])
    t2_se = np.array([r["val_full"]["dataset_grouped_se"] for r in t2])
    t2_tr = np.array([r["train_monitor_600"]["mean_total_loss"] for r in t2])
    ax_c.fill_between(xs, t2_val - t2_se, t2_val + t2_se, color=BLUE, alpha=0.16, lw=0)
    ax_c.plot(xs, t2_val, marker="s", ms=4.2, lw=2.0, color=BLUE, label="Full 345/345 dev_val ±1 SE (2.80% splits, R5* 0.57363)")
    ax_c.plot([5], [t2_val[5]], marker="o", ms=8.0, fillstyle="none", mew=1.6, color="#b91c1c", ls="none", label="Saved best_state (R5* 0.57363 ± 0.02131; R8 0.57371)")
    ax_c.plot(xs, t2_tr, marker="o", ms=3.8, lw=1.6, color=ORANGE, ls="--", label="Train monitor (600/11,976 splits, R8 0.41492)")
    ax_c.set_xticks(range(9), ["R0", "R1", "R2", "R3", "R4", "R5*", "R6", "R7", "R8"])
    ax_c.set_xlabel("Training Round (R1..R8 = 15,968 visits, 1.333× passes)")
    ax_c.set_ylabel("Disjoint-1,000 objective loss")
    ax_c.set_title("C. DOPE-MLC-v2 Disjoint-1,000 (<5% Val: 345 Splits / 58 DS)", loc="left", color=INK, fontsize=8.6)
    ax_c.legend(frameon=False, fontsize=6.5, loc="upper right")
    # Panel D: Track 3 DOPE-MLC-v3 Disjoint-1,069 (<5% Val: 75/2,304 loaded cells = 3.26%)
    a3_val = np.array([r["val_full"]["mean_total_loss"] for r in t3_a3])
    a3_se = np.array([r["val_full"]["dataset_grouped_se"] for r in t3_a3])
    ax_d.fill_between(xs, a3_val - a3_se, a3_val + a3_se, color="#15803d", alpha=0.14, lw=0)
    for vkey, col, ls, mk, lbl in (
        ("A0_v2_baseline_140", GRAY, ":", "v", "A0 baseline (140-d, R5* 0.68700 | R8 0.68709)"),
        ("A1_plus_16_building_blocks", ORANGE, "--", "^", "A1 +68 building blocks (R7* 0.68531 | R8 0.68547)"),
        ("A2_plus_48_inter_landmark_stats", BLUE, "-.", "o", "A2 +48 inter-landmark+attn (R8* 0.68489)"),
        ("A3_full_v3_with_col_gate_head", "#15803d", "-", "s", "A3 full v3 + col_gate ±1 SE (R8* 0.68492 ± 0.03877)"),
    ):
        ys = np.array([pt["val_mean_total_loss"] for pt in t3_vars[vkey]["round_val_loss_trajectory"]])
        ax_d.plot(xs, ys, marker=mk, ms=4.0, lw=2.0 if vkey.startswith("A3") else 1.5, color=col, ls=ls, label=lbl)
    ax_d.set_xticks(range(9), ["R0", "R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8*"])
    ax_d.set_xlabel("Training Round (4,457 visits / 2,229 cells = 1.9996× passes)")
    ax_d.set_ylabel("Full 75/75 loaded dev_val loss (3.26% pool)")
    ax_d.set_title("D. DOPE-MLC-v3 Disjoint-1,069 A0..A3 (<5% Val: 75/2,304 Cells)", loc="left", color=INK, fontsize=8.6)
    ax_d.legend(frameon=False, fontsize=6.4, loc="upper right")
    fig.tight_layout()
    return fig


def figure_bytes(fig, fmt: str) -> bytes:
    buf = io.BytesIO()
    meta = {"Creator": "dope bench", "CreationDate": None} if fmt == "pdf" else {"Software": None}
    fig.savefig(buf, format=fmt, dpi=200, metadata=meta)
    return buf.getvalue()


def readme_block(summary: dict) -> str:
    n = len(summary["datasets"])
    head = row(summary, "dope", "4k")
    lines = [BEGIN, "<!-- Generated by bench/real-world/render.py from results/confirm/summary.json; do not edit. -->",
             f"Measured on {n} public tables (UCI and PMLB), held-out real test rows. Robust retention is the worst of "
             "three auditor families (linear/logistic, CatBoost, MLP) trained only on the method's rows; "
             "medians over tables with 95% bootstrap intervals.", "",
             "| Method | Median bytes | Robust retention | Tables ≥ 0.9 |", "|---|---:|---:|---:|"]
    order = [("dope", "2k", "DOPE, 2 KiB budget"), ("dope", "4k", "**DOPE, 4 KiB budget**"),
             ("dope", "16k", "DOPE, 16 KiB budget"),
             ("kernel", "-", "DOPE research kernel (`dope-kernel fit-table`)"),
             ("rows", "4k", "Real rows, xz-compressed to 4 KiB"), ("rows", "64k", "Real rows, xz-compressed to 64 KiB")]
    order += [(m, "-", f"{name} ({'arfpy' if m == 'arf' else 'SDV'} defaults)") for m, name in GENERATORS]
    order += [("rows", "full", "All real rows, xz-compressed (reference)")]
    for m, b, label in order:
        r = row(summary, m, b)
        if not r or r["datasets"] == 0:
            continue
        rr = r["robust"]
        cohort = "" if r["datasets"] == n else f" ({r['datasets']} tables)"
        lines.append(f"| {label}{cohort} | {kb(r['bytes']['median'])} | {ret(rr['median'], tex=False)} "
                     f"[{ret(rr['lo'], tex=False)}, {ret(rr['hi'], tex=False)}] "
                     f"| {r['reach90']}/{r['datasets']} |")
    k_d, med_d = tdl_stats(summary, "dope@0.9")
    k_r, med_r = tdl_stats(summary, "rows@0.9")
    lines += ["", f"Transfer Description Length at τ = 0.9: DOPE reaches it on {k_d}/{n} tables "
              f"(median {kb(med_d)}); xz-compressed real rows reach it on {k_r}/{n} (median {kb(med_r)})."]
    eq = summary.get("equal_bytes", {})
    if eq:
        wins, losses = sum(c["wins"] for c in eq.values()), sum(c["losses"] for c in eq.values())
        cells = sum(c["n"] for c in eq.values())
        per = ", ".join(f"{b.replace('k', ' KiB')} {c['wins']}/{c['ties']}/{c['losses']}"
                        for b, c in sorted(eq.items(), key=lambda kv: BUDGET_ORDER.index(kv[0])))
        failed = [(b, k) for b, k in sorted(summary.get("dope_failed", {}).items(),
                                            key=lambda kv: BUDGET_ORDER.index(kv[0])) if k]
        note = ""
        if failed:
            where = " and ".join(f"{b.replace('k', ' KiB')} on {k} table{'s' if k != 1 else ''}" for b, k in failed)
            note = (f" DOPE could not fit within {where}: the schema plus the leanest program alone exceeded "
                    f"{'that budget' if len(failed) == 1 else 'those budgets'}.")
        lines.append(f"At equal bytes, wherever DOPE fits the budget, it beats xz-compressed real rows in {wins} of "
                     f"{cells} paired (table, budget) comparisons and loses {losses} (wins/ties/losses by budget: "
                     f"{per}).{note}")
    kern = summary.get("kernel")
    if kern and kern["vs"]:
        per = "; ".join(f"at {b.replace('k', ' KiB')} {c['wins']}/{c['ties']}/{c['losses']}, Hodges–Lehmann "
                        f"{c['hl']:+.3f} [{c['lo']:+.3f}, {c['hi']:+.3f}]"
                        for b, c in sorted(kern["vs"].items(), key=lambda kv: BUDGET_ORDER.index(kv[0])))
        lines.append(f"Against the repository's research kernel (median {kb(kern['bytes'])}) on the "
                     f"{kern['supported']} regression and binary tables it supports, DOPE's robust retention "
                     f"wins/ties/losses are {per}. On the {kern['unsupported']} multiclass tables the kernel refuses "
                     f"the target or fits its class codes as a regression; its row above counts them as measured.")
    lines += ["", "![Per table at 4 KiB: DOPE's robust retention against real rows xz-compressed to the same bytes]"
              "(bench/real-world/figures/tables.png)", "",
              "![TDL coverage: share of tables reaching the retention threshold within a byte budget]"
              "(bench/real-world/figures/coverage.png)",
              *multilevel_readme_lines(),
              END]
    return "\n".join(lines)



def outputs(results: Path) -> dict[str, bytes]:
    summary = json.loads((results / "summary.json").read_text())
    plt = setup_matplotlib()
    files: dict[str, bytes] = {}
    for name, fn in (("hero", hero), ("coverage", coverage), ("families", families), ("tables", tables),
                     ("audit", audit), ("convergence", convergence)):
        fig = fn(summary, plt)
        files[f"bench/real-world/figures/{name}.png"] = figure_bytes(fig, "png")
        if name == "hero":  # the paper's caption carries the title
            plt.close(fig)
            fig = hero(summary, plt, title=False)
        files[f"paper/generated/dope1-{name}.pdf"] = figure_bytes(fig, "pdf")
        plt.close(fig)
    files["bench/real-world/results/readme-block.md"] = (readme_block(summary) + "\n").encode()
    files["paper/generated/dope1-macros.tex"] = macros(summary).encode()
    files["paper/generated/dope1-config.tex"] = config_macros().encode()
    files["paper/generated/dope1-table.tex"] = table_main(summary).encode()
    files["paper/generated/dope1-multilevel.tex"] = table_multilevel().encode()
    files["paper/generated/dope1-ablation.tex"] = table_ablation(summary).encode()
    files["paper/generated/dope1-equalbytes.tex"] = table_equal_bytes(summary).encode()
    files["paper/generated/dope1-pertable.tex"] = table_per_dataset(summary).encode()
    tp = results.parent / "throughput.json"
    if tp.is_file():
        macros_tp, table_tp = throughput_tex(json.loads(tp.read_text()))
        files["paper/generated/dope1-throughput-macros.tex"] = macros_tp.encode()
        files["paper/generated/dope1-throughput.tex"] = table_tp.encode()
    digest = {k: hashlib.sha256(v).hexdigest() for k, v in sorted(files.items())}
    files["bench/real-world/results/render-manifest.json"] = (json.dumps(
        {"summary_sha256": hashlib.sha256((results / "summary.json").read_bytes()).hexdigest(), "files": digest},
        indent=1) + "\n").encode()
    return files


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=ROOT / "bench/real-world/results/confirm")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    drift = []
    for name, content in outputs(a.results).items():
        path = ROOT / name
        if a.check:
            # Raster/PDF bytes depend on the matplotlib build; text outputs must match exactly.
            if name.endswith((".md", ".tex")) and (not path.is_file() or path.read_bytes() != content):
                drift.append(name)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
    if drift:
        print("render drift: " + ", ".join(drift), file=sys.stderr)
        return 1
    print("DOPE 1.0 benchmark surfaces are current" if a.check else "rendered")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
