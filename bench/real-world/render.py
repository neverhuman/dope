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
    """2x2 dashboard: (A-B) v2 & v3 train monitor vs <5% full-val loss; (C) value vs wire bytes; (D) testing scale."""
    import numpy as np
    snap = multilevel_convergence_data()
    v2_old = snap["v2_full_corpus"]
    eps_old, full_old = v2_old["epochs"], v2_old["full_1891_val_final_reloaded_ep7_best_state"]
    t1 = snap["v2_full_corpus_maxtrain97_val3pct"]["rounds"]
    t2 = snap["v2_disjoint1000_maxtrain97_devval345"]["rounds"]
    t3_a3 = snap["v3_maxtrain97_val3pct"]["a3_rounds"]
    t3_a0 = snap["v3_maxtrain97_val3pct"]["matched_budget_ablation_progression_A0_to_A3"]["A0_v2_baseline_140"]
    vs = snap["value_and_testing_scale_summary"]
    xs = np.array([r["raw_receipt_epoch_field"] for r in t1], dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 6.8))
    ax_a, ax_b, ax_c, ax_d = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]
    # Panel A: Track 1 DOPE-MLC-v2 Full Corpus (<5% Val: 684/684 splits, 115 DS) + Per-Tier Hierarchy
    t1_val = np.array([r["val_full"]["mean_total_loss"] for r in t1])
    t1_se = np.array([r["val_full"]["dataset_grouped_se"] for r in t1])
    t1_tr = np.array([r["train_monitor_600"]["mean_total_loss"] for r in t1])
    t1_ult = np.array([r["val_full"]["ultra"]["total_loss"] for r in t1])
    t1_med = np.array([r["val_full"]["medium"]["total_loss"] for r in t1])
    old_val = np.array([e["val_monitor_600"]["mean_total_loss"] for e in eps_old])
    xf = 8.85
    ax_a.fill_between(xs, t1_val - t1_se, t1_val + t1_se, color=BLUE, alpha=0.16, lw=0)
    ax_a.plot(xs, t1_val, marker="s", ms=4.2, lw=2.1, color=BLUE, label="Full 684/684 val mean ±1 SE (3.59% splits, R8* 0.34216, -16.5%)")
    ax_a.plot(xs, t1_tr, marker="o", ms=3.8, lw=1.7, color="#15803d", ls="--", label="600/18,395 train monitor (subsampled, R0 0.41216 → R8 0.35175)")
    ax_a.plot(xs, t1_ult, marker="*", ms=5.0, lw=1.4, color="#0f766e", ls="-.", label="Full 684/684 val Ultra 64 KiB (R8* 0.32396 ± 0.01672)")
    ax_a.plot(xs, t1_med, marker="^", ms=3.6, lw=1.3, color=ORANGE, ls="-.", label="Full 684/684 val Medium 4 KiB (R8* 0.36407 ± 0.01608)")
    ax_a.plot(xs, old_val, marker="d", ms=3.5, lw=1.3, color=GRAY, ls=":", label="Preserved 10%-val monitor (600/1,891, R7* 0.37667 | full 0.37753)")
    ax_a.plot([xf], [full_old["mean_total_loss"]], marker="D", ms=5.5, color="#b91c1c", ls="none")
    ax_a.axvline(8.42, color=GRAY, ls=":", lw=0.9)
    ax_a.set_xticks(list(range(9)) + [xf], ["R0", "R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8*", "Full\n1891"])
    ax_a.set_ylim(0.312, 0.455)
    ax_a.set_xlabel("Training Round (24,527 visits / 18,395 splits = 1.333× passes)")
    ax_a.set_ylabel("DOPE-MLC-v2 multi-task loss")
    ax_a.set_title("A. DOPE-MLC-v2 Train Monitor vs. Full <5% Unseen Val Loss (3,188 DS)", loc="left", color=INK, fontsize=8.3)
    ax_a.legend(frameon=False, fontsize=5.8, loc="upper right")
    # Panel B: Disjoint-1,000 (v2) & Disjoint-1,069 (v3) Train Monitor vs Full <5% dev_val Loss
    t2_val = np.array([r["val_full"]["mean_total_loss"] for r in t2])
    t2_se = np.array([r["val_full"]["dataset_grouped_se"] for r in t2])
    t2_tr = np.array([r["train_monitor_600"]["mean_total_loss"] for r in t2])
    a3_val = np.array([r["val_full"]["mean_total_loss"] for r in t3_a3])
    a3_se = np.array([r["val_full"]["dataset_grouped_se"] for r in t3_a3])
    a3_tr = np.array([r["train_monitor_600"]["mean_total_loss"] for r in t3_a3])
    a0_val = np.array([pt["val_mean_total_loss"] for pt in t3_a0["round_val_loss_trajectory"]])
    ax_b.fill_between(xs, a3_val - a3_se, a3_val + a3_se, color="#15803d", alpha=0.13, lw=0)
    ax_b.fill_between(xs, t2_val - t2_se, t2_val + t2_se, color=BLUE, alpha=0.14, lw=0)
    ax_b.plot(xs, a3_val, marker="s", ms=4.0, lw=2.0, color="#15803d", label="v3 A3 full 75/75 dev_val ±1 SE (3.26% of 2,304 pool, R8* 0.68492)")
    ax_b.plot(xs, a0_val, marker="v", ms=3.5, lw=1.3, color=GRAY, ls=":", label="v3 A0 baseline 75/75 dev_val (140-d, R5* 0.68700 | R8 0.68709)")
    ax_b.plot(xs, t2_val, marker="s", ms=4.0, lw=2.0, color=BLUE, label="v2 Disjoint-1k full 345/345 dev_val ±1 SE (2.80%, R5* 0.57363)")
    ax_b.plot([5], [t2_val[5]], marker="o", ms=7.5, fillstyle="none", mew=1.5, color="#b91c1c", ls="none")
    ax_b.plot(xs, a3_tr, marker="o", ms=3.6, lw=1.6, color="#15803d", ls="--", label="v3 A3 600/2,229 train monitor (1.9996× passes, R8 0.46132)")
    ax_b.plot(xs, t2_tr, marker="o", ms=3.6, lw=1.6, color=ORANGE, ls="--", label="v2 Disjoint-1k 600/11,976 train monitor (1.333× passes, R8 0.41492)")
    ax_b.set_xticks(range(9), ["R0", "R1", "R2", "R3", "R4", "R5*", "R6", "R7", "R8*"])
    ax_b.set_ylim(0.395, 0.855)
    ax_b.set_xlabel("Training Round (v2: 15,968 / 1.333×; v3: 4,457 / 2.00×)")
    ax_b.set_ylabel("Multi-task objective loss")
    ax_b.set_title("B. Disjoint-1,000 (v2) & 1,069 (v3) Train vs. <5% Val Loss", loc="left", color=INK, fontsize=8.3)
    ax_b.legend(frameon=False, fontsize=5.8, loc="upper right")
    # Panel C: Value vs Measured Wire Bytes Across 1,000 (v2) & 1,069 (v3) Selection-Exposed Datasets
    ax_c.set_xscale("log")
    v2_t = vs["v2_value_by_tier_1000ds"]
    v3_t = vs["v3_value_by_role_and_tier_1069ds"]
    tiers_v2 = ("medium", "high", "ultra")
    tiers_v3 = ("small", "medium", "high", "ultra")
    x_v2 = [v2_t[t]["exact_v2_latent_wire_median_bytes"] for t in tiers_v2]
    y_v2 = [v2_t[t]["dope_mean_retention_R"] for t in tiers_v2]
    lo_v2 = [y_v2[i] - v2_t[t]["dope_retention_R_ci95"][0] for i, t in enumerate(tiers_v2)]
    hi_v2 = [v2_t[t]["dope_retention_R_ci95"][1] - y_v2[i] for i, t in enumerate(tiers_v2)]
    ax_c.errorbar(x_v2, y_v2, yerr=[lo_v2, hi_v2], marker="o", ms=5.2, lw=2.1, color=BLUE, capsize=2.5,
                  label="DOPE-MLC-v2 (1,000 DS, exact v2 med wire, 95% CI; Elo 1651.3 #1)")
    for m_key, col, ls, mk, lbl in (
        ("rows", ORANGE, "--", "s", "Rows @ nominal 4/16/64 KiB cap (1,000 DS, 95% CI; Elo 1593.3)"),
        ("copula", GRAY, ":", "^", "Copula @ nominal 4/16/64 KiB cap (1,000 DS, 95% CI; Elo 1545.8)"),
    ):
        xm = [v2_t[t]["nominal_cap_bytes"] for t in tiers_v2]
        ym = [v2_t[t][f"{m_key}_mean_retention_R"] for t in tiers_v2]
        lom = [ym[i] - v2_t[t][f"{m_key}_retention_R_ci95"][0] for i, t in enumerate(tiers_v2)]
        him = [v2_t[t][f"{m_key}_retention_R_ci95"][1] - ym[i] for i, t in enumerate(tiers_v2)]
        ax_c.errorbar(xm, ym, yerr=[lom, him], marker=mk, ms=4.2, lw=1.5, color=col, ls=ls, capsize=2.0, label=lbl)
    x_v3p = [v3_t["eval_gradient_disjoint_v2_panel"][t]["mean_dml3_complete_wire_bytes"] for t in tiers_v3]
    y_v3p = [v3_t["eval_gradient_disjoint_v2_panel"][t]["sender_scaled_6aud_mean_clipped_R"] for t in tiers_v3]
    x_v3m = [v3_t["eval_gradient_disjoint_multiclass_panel"][t]["mean_dml3_complete_wire_bytes"] for t in tiers_v3]
    y_v3m = [v3_t["eval_gradient_disjoint_multiclass_panel"][t]["sender_scaled_6aud_mean_clipped_R"] for t in tiers_v3]
    ax_c.plot(x_v3p, y_v3p, marker="D", ms=4.4, lw=1.8, color="#15803d", ls="-.",
              label="DOPE-MLC-v3 1,000-DS panel (mean DML3+CSC1 wire, 340-row 6-aud R)")
    ax_c.plot(x_v3m, y_v3m, marker="P", ms=5.0, lw=1.7, color="#7c3aed", ls="-.",
              label="DOPE-MLC-v3 69-DS multiclass (+0.253 ΔR vs copula, Elo 1540.9)")
    kib_ticks(ax_c, 800.0, 65536.0)
    ax_c.set_xlim(700.0, 82000.0)
    ax_c.set_ylim(0.64, 0.94)
    ax_c.set_xlabel("Wire bytes (v2: exact median wire; v3: mean DML3+CSC1; baselines: cap)")
    ax_c.set_ylabel("Mean holdout retention R (6 auditors)")
    ax_c.set_title("C. Value vs. Wire Bytes Across 1,000 (v2) & 1,069 (v3) Datasets", loc="left", color=INK, fontsize=8.3)
    ax_c.legend(frameon=False, fontsize=5.8, loc="lower right")
    # Panel D: Distinct Testing & Verification Scale (No Overlapping-Cohort Summation)
    sc = vs["testing_scale_distinct_units"]
    bars = [
        ("CI/cpu test assertions\n(656 Rust + 1,801 Py; 834+ unique)", sc["ci_cpu_run_40159022870_test_assertions"]["total_executed_assertions_in_job"], "#475569"),
        ("Reconciled catalog datasets\n(1,069 unique eval; 1,000 shared)", sc["reconciled_catalog_datasets_total"], "#0f766e"),
        ("v3 completed DS-tier cells\n(1,069 DS × 4 tiers, 0 failed)", sc["v3_completed_dataset_tier_cells"], "#15803d"),
        ("v2 bit-exact wire replay cells\n(1,000 DS × 3 tiers × 2 ckpts)", sc["v2_exact_wire_replay_cells_verified"], BLUE),
        ("v2 pre-calculated CV splits\n(18,395 train + 684 <5% val)", sc["v2_precalculated_cv_splits_total"], "#1d4ed8"),
        ("v3 6-auditor cell evaluations\n(4,276 cells × 6 auditors/domain)", sc["v3_six_auditor_evaluations_per_domain"], "#7c3aed"),
        ("v2 downstream auditor fits\n(1,000 reg/bin DS × 6 auditors)", sc["v2_downstream_auditor_fits_total"], ORANGE),
    ]
    y_pos = np.arange(len(bars))
    vals = [b[1] for b in bars]
    cols = [b[2] for b in bars]
    ax_d.barh(y_pos, vals, height=0.62, color=cols, alpha=0.88)
    ax_d.set_xscale("log")
    ax_d.set_xlim(1000.0, 260000.0)
    ax_d.set_yticks(y_pos, [b[0] for b in bars], fontsize=6.7)
    ax_d.grid(axis="y", visible=False)
    for idx_b, val_b in enumerate(vals):
        ax_d.text(val_b * 1.12, idx_b, f"{val_b:,}", va="center", ha="left", fontsize=7.2, fontweight="bold", color=INK)
    ax_d.set_xlabel("Verified count (log scale; distinct non-summed receipt units)")
    ax_d.set_title("D. Testing & Verification Scale (1,069 Unique Eval DS)", loc="left", color=INK, fontsize=8.3)
    fig.tight_layout(rect=(0.01, 0.01, 0.99, 0.99))
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
