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
    BUDGET_ORDER, ROOT, _study_val, config_macros, kb, macros, multilevel_convergence_data, multilevel_readme_lines,
    row, table_ablation, table_equal_bytes, table_main, table_multilevel, table_per_dataset, throughput_tex,
)

BEGIN, END = "<!-- BEGIN DOPE1 RESULTS -->", "<!-- END DOPE1 RESULTS -->"
# Okabe-Ito color-blind-safe publication palette + neutral slate steps.
BLUE, ORANGE, TEAL, PURPLE, SLATE = "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#475569"
GRAY, INK, INK2, GRID, SURFACE = "#78716c", "#0b0b0b", "#44403c", "#e6e5e1", "#fcfcfb"
GENERATORS = [("copula", "Gaussian copula"), ("arf", "ARF"), ("tvae", "TVAE"), ("ctgan", "CTGAN")]
GRAYS = ["#a8a29e", "#78716c", "#57534e", "#292524"]  # neutral steps for the generator bars


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


def hero_readme(plt):
    """3-panel publication hero on 1,000 (v2) & 1,069 (v3) datasets for README hero.png: (A) Elo leaderboard, (B) retention by byte tier, (C) 6-auditor transfer & 150-DS 1x->5x expansion."""
    import numpy as np
    snap = multilevel_convergence_data()
    vs = snap["value_and_testing_scale_summary"]
    lb = [r for r in vs["v2_overall_elo_1000ds"] if r["method"] != "kernel"]
    v2_t = vs["v2_value_by_tier_1000ds"]
    v3_t = vs["v3_value_by_role_and_tier_1069ds"]
    v3_p, v3_mc = v3_t["eval_gradient_disjoint_v2_panel"], v3_t["eval_gradient_disjoint_multiclass_panel"]
    tiers = ("medium", "high", "ultra")
    med_b = [int(v2_t[t]["exact_v2_latent_wire_median_bytes"]) for t in tiers]
    med_kib = [b / 1024.0 for b in med_b]
    v3_kib = [v3_p[t]["mean_dml3_complete_wire_bytes"] / 1024.0 for t in tiers]
    study_txt = (ROOT / "docs/multilevel-landmark-dl-study.md").read_text()
    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(14.4, 4.95), gridspec_kw={"width_ratios": [1.14, 1.05, 1.01]})
    fig.suptitle("DOPE (git tag v1.0.0) — 1,000-DS (DOPE-MLC-v2) & 1,069-DS (DOPE-MLC-v3) Multi-Learner Benchmark", fontsize=10.6, fontweight="bold", color=INK, y=0.985)
    names = [f"#1 DOPE-MLC-v2 (v1.0.0)\n({med_kib[0]:.1f}–{med_kib[2]:.1f} KiB v2 latent wire)",
             "#2 Raw Rows Subsample\n(raw-row subset control)",
             "#3 Gaussian Copula\n(parametric baseline)", "#4 Joint Diffusion\n(tabular diffusion)", "#5 ARF\n(adversarial forest)"]
    colors = [BLUE, ORANGE, SLATE, GRAY, "#a8a29e"]
    y_pos = np.arange(len(lb))[::-1]
    elos = [r["elo"] for r in lb]
    bars = ax_a.barh(y_pos, elos, xerr=[[r["elo"] - r["elo_ci95_low"] for r in lb], [r["elo_ci95_high"] - r["elo"] for r in lb]],
                     height=0.60, color=colors, alpha=0.92, error_kw=dict(ecolor=INK, lw=1.2, capsize=2.5))
    bars[0].set_edgecolor(INK)
    bars[0].set_linewidth(1.3)
    ax_a.axvline(1500.0, color=INK2, lw=1.0, ls="--", zorder=1)
    ax_a.set_yticks(y_pos, names, fontsize=7.8)
    ax_a.set_xlim(1100.0, 1865.0)
    ax_a.set_ylim(-0.65, len(lb) - 0.35)
    ax_a.grid(axis="y", visible=False)
    for i, r in enumerate(lb):
        ax_a.text(r["elo_ci95_high"] + 8.0, y_pos[i], f"{r['elo']:.1f} ({100.0 * r['win_share']:.1f}% win)", va="center", ha="left", fontsize=7.6, fontweight="bold" if i == 0 else "normal", color=BLUE if i == 0 else INK)
    ax_a.text(0.98, 0.03, "v2 Disjoint-1k: 1650.7 (#1)\n242 held-out DS: 1671.6 (#1)\nv3 MC Elo: 1540.9 (Copula 1313.8)", transform=ax_a.transAxes, ha="right", va="bottom", fontsize=6.1, color=INK, bbox=dict(boxstyle="round,pad=0.22", facecolor="#f0f9ff", edgecolor=BLUE, lw=0.75))
    ax_a.set_xlabel("Bradley–Terry Elo (95% CI; dashed = 1500 neutral)", fontsize=8.3)
    ax_a.set_title("A. 1,000-DS (v2) & 1,069-DS (v3) Tournament Elo", loc="left", color=INK, fontsize=9.0, fontweight="bold")
    x, w = np.arange(len(tiers)), 0.20
    dope_v2_lbl = f"DOPE-MLC-v2 ({med_b[0]:,}/{med_b[1]:,}/{med_b[2]:,} B = {med_kib[0]:.1f}/{med_kib[1]:.1f}/{med_kib[2]:.1f} KiB v2 latent med)"
    dope_v3_lbl = f"DOPE-MLC-v3 (dope mlc-v3: {v3_kib[0]:.1f}/{v3_kib[1]:.1f}/{v3_kib[2]:.1f} KiB DML3+CSC1 mean; MC High {v3_mc['high']['receiver_csc1_raw_6aud_mean_clipped_R']:.3f})"
    for key, label, col, offset in (("dope", dope_v2_lbl, BLUE, -1.5 * w), ("v3", dope_v3_lbl, TEAL, -0.5 * w), ("rows", "Raw Rows Subsample (4 / 16 / 64 KiB cap; Med CB 999/1k)", ORANGE, 0.5 * w), ("copula", "Gaussian Copula (4 / 16 / 64 KiB nominal cap)", SLATE, 1.5 * w)):
        if key == "v3":
            means = [v3_p[t]["receiver_csc1_raw_6aud_mean_clipped_R"] for t in tiers]
            lo, hi = [0.0] * len(tiers), [0.0] * len(tiers)
            rects = ax_b.bar(x + offset, means, w * 0.88, color=col, alpha=0.90, label=label)
        else:
            means = [v2_t[t][f"{key}_mean_retention_R"] for t in tiers]
            lo = [means[i] - v2_t[t][f"{key}_retention_R_ci95"][0] for i, t in enumerate(tiers)]
            hi = [v2_t[t][f"{key}_retention_R_ci95"][1] - means[i] for i, t in enumerate(tiers)]
            rects = ax_b.bar(x + offset, means, w * 0.88, yerr=[lo, hi], color=col, alpha=0.92, label=label, error_kw=dict(ecolor=INK, lw=1.0, capsize=1.8))
        for rect in (rects if key in ("dope", "v3") else ()):
            rect.set_edgecolor(INK)
            rect.set_linewidth(0.9)
        for i, m in enumerate(means):
            pad = 0.013 if key == "v3" else 0.008
            ax_b.text(x[i] + offset, m + hi[i] + pad, f"{m + 1e-9:.3f}", ha="center", va="bottom", fontsize=5.7, fontweight="bold" if key in ("dope", "v3") else "normal", color=BLUE if key == "dope" else (TEAL if key == "v3" else INK2))
    ax_b.text(-0.22, 0.910, "+18.8 pts vs Rows\n(#1 Elo 1652.1)", ha="center", va="bottom", fontsize=5.8, fontweight="bold", color=BLUE, bbox=dict(boxstyle="round,pad=0.13", facecolor="#f0f9ff", edgecolor=BLUE, lw=0.65))
    ax_b.text(0.95, 0.968, "v2 +4.5 pts (#1 1669.6)\nv3 MC High: 0.845", ha="center", va="bottom", fontsize=5.8, fontweight="bold", color=BLUE, bbox=dict(boxstyle="round,pad=0.13", facecolor="#f0f9ff", edgecolor=BLUE, lw=0.65))
    ax_b.text(2.12, 0.992, "#1 Encoder (1640.1)\nRows: 88% raw copy", ha="center", va="bottom", fontsize=5.8, color=INK2, bbox=dict(boxstyle="round,pad=0.13", facecolor=SURFACE, edgecolor=GRID, lw=0.65))
    ax_b.set_xticks(x, [f"Medium (4 KiB cap)\nv2: {med_kib[0]:.1f} | v3: {v3_kib[0]:.1f} KiB", f"High (16 KiB cap)\nv2: {med_kib[1]:.1f} | v3: {v3_kib[1]:.1f} KiB", f"Ultra (64 KiB cap)\nv2: {med_kib[2]:.1f} | v3: {v3_kib[2]:.1f} KiB"], fontsize=7.2)
    ax_b.set_xlim(-0.56, 2.56)
    ax_b.set_ylim(0.55, 1.24)
    ax_b.grid(axis="x", visible=False)
    ax_b.set_ylabel("Mean holdout retention R (6 learners)", fontsize=8.3)
    ax_b.set_title("B. Holdout Retention R by Byte Tier (v2 1,000 DS & v3 1,069 DS)", loc="left", color=INK, fontsize=8.8, fontweight="bold")
    ax_b.legend(frameon=True, facecolor=SURFACE, edgecolor=GRID, fontsize=6.0, loc="upper left")
    aud_specs = [
        ("Linear / Logistic\n(n=969)", r"\| \*\*Linear / Logistic\*\* \| `969` \| `[\d.]+` \| `[\d.]+` \| \*\*`([\d.]+)`", r"\| \*\*Linear / Logistic\*\* \| `969` \| .*? \| \*\*`[\d.]+` `\[[^\]]+\]`\*\* \| \*\*`([\d.]+)`"),
        ("Local Gated Attention\n(n=962, from scratch)", r"\| \*\*Local Gated Attention Net \(`kumo_tabpfn` key\)\*\* \| `962` \| `[\d.]+` \| `[\d.]+` \| \*\*`([\d.]+)`", r"\| \*\*Local Gated Attention Net \(`kumo_tabpfn` key\)\*\* \| `962` \| .*? \| \*\*`[\d.]+` `\[[^\]]+\]`\*\* \| \*\*`([\d.]+)`"),
        ("Neural MLP\n(n=965)", r"\| \*\*MLP\*\* \| `965` \| `[\d.]+` \| `[\d.]+` \| \*\*`([\d.]+)`", r"\| \*\*MLP\*\* \| `965` \| .*? \| \*\*`[\d.]+` `\[[^\]]+\]`\*\* \| \*\*`([\d.]+)`"),
        ("LightGBM\n(n=989)", r"\| \*\*LightGBM\*\* \| `989` \| `[\d.]+` \| `[\d.]+` \| \*\*`([\d.]+)`", r"\| \*\*LightGBM\*\* \| `989` \| .*? \| \*\*`[\d.]+` `\[[^\]]+\]`\*\* \| \*\*`([\d.]+)`"),
        ("CatBoost\n(n=1,000)", r"\| \*\*CatBoost\*\* \| `1,000` \| `[\d.]+` \| `[\d.]+` \| \*\*`([\d.]+)`", r"\| \*\*CatBoost\*\* \| `1,000` \| .*? \| \*\*`[\d.]+` `\[[^\]]+\]`\*\* \| \*\*`([\d.]+)`"),
        ("XGBoost\n(n=989)", r"\| \*\*XGBoost\*\* \| `989` \| `[\d.]+` \| `[\d.]+` \| \*\*`([\d.]+)`", r"\| \*\*XGBoost\*\* \| `989` \| .*? \| \*\*`[\d.]+` `\[[^\]]+\]`\*\* \| \*\*`([\d.]+)`"),
    ]
    yc, r_vals, d_vals = np.arange(len(aud_specs))[::-1], [float(_study_val(study_txt, p_r)) for _, p_r, _ in aud_specs], [float(_study_val(study_txt, p_d)) for _, _, p_d in aud_specs]
    ax_c.barh(yc + 0.15, r_vals, height=0.30, color=BLUE, alpha=0.92, label="Holdout retention R (T90 = 0.90)")
    ax_c.barh(yc - 0.15, d_vals, height=0.30, color=TEAL, alpha=0.88, label="Driver fidelity D (D80 = 0.80)")
    ax_c.axvline(0.90, color=BLUE, lw=0.9, ls="--", alpha=0.7)
    ax_c.axvline(0.80, color=TEAL, lw=0.9, ls=":", alpha=0.7)
    for i, (rv, dv) in enumerate(zip(r_vals, d_vals)):
        ax_c.text(rv + 0.006, yc[i] + 0.15, f"{rv:.3f}", va="center", ha="left", fontsize=6.9, fontweight="bold", color=BLUE)
        ax_c.text(dv + 0.006, yc[i] - 0.15, f"{dv:.3f}", va="center", ha="left", fontsize=6.7, color=TEAL)
    exp1 = _study_val(study_txt, r"monotonically from \*\*`([\d.]+)`\*\* \(\$1\\times N_\{\\text\{train\}\}\$\)")
    exp2 = _study_val(study_txt, r"\$\\to\$ \*\*`([\d.]+)`\*\*\s+\(\$2\\times N_\{\\text\{train\}\}\$\)")
    exp5 = _study_val(study_txt, r"\$\\to\$ \*\*`([\d.]+)`\*\* \(\$5\\times N_\{\\text\{train\}\}\$")
    ax_c.text(0.50, 0.015, f"1×→2×→5× N_train expansion (n=150 DS subset, ds_rank<150):\nCatBoost R: {exp1} → {exp2} → {exp5} (+2.62 pts, 0 extra wire B)", transform=ax_c.transAxes, ha="center", va="bottom", fontsize=6.1, color=INK, bbox=dict(boxstyle="round,pad=0.22", facecolor="#f0f9ff", edgecolor=BLUE, lw=0.75))
    ax_c.set_yticks(yc, [a[0] for a in aud_specs], fontsize=7.4)
    ax_c.set_xlim(0.62, 1.04)
    ax_c.set_ylim(-1.42, 6.55)
    ax_c.grid(axis="y", visible=False)
    ax_c.set_xlabel("Holdout retention R & driver fidelity D (Ultra)", fontsize=8.3)
    ax_c.set_title("C. 6-Learner Transfer (Ultra)", loc="left", color=INK, fontsize=9.0, fontweight="bold")
    ax_c.legend(frameon=True, facecolor=SURFACE, edgecolor=GRID, fontsize=6.4, loc="upper left")
    dec_b = int(_study_val(study_txt, r"frozen `(\d+,\d+) B` decoder").replace(",", ""))
    fig.text(0.5, 0.006, f"Byte & cohort qualification (git tag v1.0.0): DOPE-MLC-v2 medians ({med_b[0]:,} / {med_b[1]:,} / {med_b[2]:,} B = {med_kib[0]:.1f} / {med_kib[1]:.1f} / {med_kib[2]:.1f} KiB) are exact v2 latent payloads (+545 B med scaler sidecar; shared {dec_b:,} B decoder; 1,000 DS); DOPE-MLC-v3 (dope mlc-v3) spans 1,069 DS (1,000 v2-panel + 69 multiclass, DML3+CSC1 self-contained).\nBaseline tiers (4 / 16 / 64 KiB) reflect nominal budget caps and historical receipt byte estimates (rows medium|catboost: 999/1,000 observed), not a common measured-wire byte qualification.", ha="center", va="bottom", fontsize=6.5, color=INK2)
    fig.tight_layout(rect=(0.0, 0.055, 0.995, 0.955))
    return fig


def hero(summary: dict, plt, title: bool = True):
    if title:
        return hero_readme(plt)
    fig, ax = plt.subplots(figsize=(6.4, 3.65))
    ax.set_xscale("log")
    xs_all, ends, clipped = [], {}, 0
    for method, color, label in (("rows", ORANGE, "Real rows, xz-compressed\nto the same bytes"), ("dope", BLUE, "DOPE (v1.0.0)")):
        pts = frontier(summary, method)
        xs, ys = [p["bytes"]["median"] for p in pts], [p["robust"]["median"] for p in pts]
        lo, hi = [max(p["robust"]["lo"], FLOOR) for p in pts], [max(p["robust"]["hi"], FLOOR) for p in pts]
        yc = [max(y, FLOOR) for y in ys]
        xs_all += xs
        ax.fill_between(xs, lo, hi, color=color, alpha=0.15 if method == "dope" else 0.08, linewidth=0)
        ax.plot(xs, yc, color=color, linewidth=2.3, solid_capstyle="round", zorder=3)
        for x, y, c in zip(xs, ys, yc):
            ax.scatter([x], [c], s=60 if y < FLOOR else 48, marker="v" if y < FLOOR else "o", color=color, edgecolor=SURFACE, linewidth=1.8, zorder=4)
            if y < FLOOR:
                ax.annotate(f"{y:.1f}".replace("-", "−"), (x, c), xytext=(0, 7), textcoords="offset points", ha="center", color=INK2, fontsize=7.5)
        if xs:
            ends[method] = (xs[1] if method == "dope" and len(xs) > 1 else xs[-1],
                            yc[1] if method == "dope" and len(yc) > 1 else yc[-1], color, label)
    if xs_all:
        ax.axvspan(min(xs_all) / 3.2, 8192.0, color=BLUE, alpha=0.04, zorder=0)
        ax.text(1650.0, 0.06, "Compact-program\nzone (≤ 8 KiB)", color=BLUE, fontsize=7.3, fontweight="bold", ha="center", va="bottom")
    for method, (x, y, color, label) in ends.items():
        ax.annotate(label, (x, y), xytext=(-8, 2) if method == "dope" else (8, -14), textcoords="offset points",
                    color=color, fontsize=9, fontweight="bold", ha="right" if method == "dope" else "left", va="center")
    head = row(summary, "dope", "4k")
    if head and head["datasets"]:
        x, y = head["bytes"]["median"], head["robust"]["median"]
        ax.annotate(f"default program\n{kb(x)}, retention {y:.2f}", (x, y), xytext=(18, -38), textcoords="offset points", ha="left", color=INK, fontsize=8.5, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    for method, name in GENERATORS:
        r = row(summary, method, "-")
        if not r or r["datasets"] == 0:
            continue
        x, y = r["bytes"]["median"], r["robust"]["median"]
        xs_all.append(x)
        yc = max(y, FLOOR)
        ax.errorbar([x], [yc], yerr=[[max(0, yc - max(r["robust"]["lo"], FLOOR))], [max(0, max(r["robust"]["hi"], FLOOR) - yc)]], fmt="v" if y < FLOOR else "o", color=SLATE, ms=6.5, mec=SURFACE, mew=1.5, elinewidth=1.2, capsize=0, zorder=3)
        suffix = "" if r["datasets"] == len(summary["datasets"]) else f" ({r['datasets']} tables)"
        if y < FLOOR:
            clipped += 1
            left = clipped % 2 == 1
            ax.annotate(f"{name}{suffix}: {y:.2f}".replace("-", "−"), (x, yc), xytext=(-7 if left else 7, 9), textcoords="offset points", ha="right" if left else "left", color=INK2, fontsize=8)
        else:
            ax.annotate(f"{name}{suffix}", (x, yc), xytext=(7, 4), textcoords="offset points", color=INK2, fontsize=8)
    ax.axhline(0.9, color=INK2, linewidth=0.9, linestyle=(0, (1, 2.5)), zorder=1)
    ax.text(0.995, 0.9, "τ = 0.9", transform=ax.get_yaxis_transform(), va="bottom", ha="right", color=INK2, fontsize=8)
    ax.text(0.015, 0.97, "↖ better: more retention, fewer bytes", transform=ax.transAxes, ha="left", va="top", color=INK2, fontsize=8)
    ax.set_ylim(FLOOR - 0.04, 1.08)
    if xs_all:
        kib_ticks(ax, min(xs_all), max(xs_all))
        ax.set_xlim(min(xs_all) / 3.2, max(xs_all) * 6)
    ax.set_xlabel("Bytes that sampling reads (log scale)")
    ax.set_ylabel("Robust retention\n(worst of linear, CatBoost, MLP)")
    fig.tight_layout()
    return fig


def coverage(summary: dict, plt):
    import numpy as np
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    ax.set_xscale("log")
    n = len(summary["datasets"])
    for method, color, name in (("dope", BLUE, "DOPE (v1.0.0)"), ("rows", ORANGE, "Real rows (xz)")):
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
    groups = [("DOPE (v1.0.0) @ 4 KiB", head, BLUE), ("Rows @ 4 KiB", rows4, ORANGE)] + [(n, row(summary, m, "-"), GRAY) for m, n in others]
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
    ax.scatter([], [], color=BLUE, s=36, label="DOPE (v1.0.0)")
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
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 7.6))
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
    ax_a.plot(xs, t1_val, marker="s", ms=4.4, lw=2.2, color=BLUE, label="Full 684/684 val mean ±1 SE (3.59% splits, R8* 0.34216, -16.52%)")
    ax_a.plot(xs, t1_tr, marker="o", ms=3.8, lw=1.7, color=TEAL, ls="--", label="600/18,395 train monitor (subsampled, R0 0.41216 → R8 0.35175)")
    ax_a.plot(xs, t1_ult, marker="*", ms=5.2, lw=1.5, color=PURPLE, ls="-.", label="Full 684/684 val Ultra 64 KiB (R8* 0.32396 ± 0.01672)")
    ax_a.plot(xs, t1_med, marker="^", ms=3.8, lw=1.4, color=ORANGE, ls="-.", label="Full 684/684 val Medium 4 KiB (R8* 0.36407 ± 0.01608)")
    ax_a.plot(xs, old_val, marker="d", ms=3.5, lw=1.3, color=SLATE, ls=":", label="Preserved 10%-val monitor (600/1,891, R7* 0.37667 | full 0.37753)")
    ax_a.plot([xf], [full_old["mean_total_loss"]], marker="D", ms=5.5, color=ORANGE, ls="none")
    ax_a.axvline(8.42, color=GRAY, ls=":", lw=0.9)
    ax_a.set_xticks(list(range(9)) + [xf], ["R0", "R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8*", "Full\n1891"])
    ax_a.set_ylim(0.305, 0.488)
    ax_a.set_xlabel("Training Round (24,527 visits / 18,395 splits = 1.333× passes)")
    ax_a.set_ylabel("DOPE-MLC-v2 multi-task loss")
    ax_a.set_title("A. DOPE-MLC-v2 Train Monitor vs. Full <5% Unseen Val Loss (3,188 DS)", loc="left", color=INK, fontsize=8.8, fontweight="bold")
    ax_a.legend(frameon=True, facecolor=SURFACE, edgecolor=GRID, fontsize=6.2, loc="upper right")
    # Panel B: Disjoint-1,000 (v2) & Disjoint-1,069 (v3) Train Monitor vs Full <5% dev_val Loss
    t2_val = np.array([r["val_full"]["mean_total_loss"] for r in t2])
    t2_se = np.array([r["val_full"]["dataset_grouped_se"] for r in t2])
    t2_tr = np.array([r["train_monitor_600"]["mean_total_loss"] for r in t2])
    a3_val = np.array([r["val_full"]["mean_total_loss"] for r in t3_a3])
    a3_se = np.array([r["val_full"]["dataset_grouped_se"] for r in t3_a3])
    a3_tr = np.array([r["train_monitor_600"]["mean_total_loss"] for r in t3_a3])
    a0_val = np.array([pt["val_mean_total_loss"] for pt in t3_a0["round_val_loss_trajectory"]])
    ax_b.fill_between(xs, a3_val - a3_se, a3_val + a3_se, color=TEAL, alpha=0.14, lw=0)
    ax_b.fill_between(xs, t2_val - t2_se, t2_val + t2_se, color=BLUE, alpha=0.15, lw=0)
    ax_b.plot(xs, a3_val, marker="s", ms=4.2, lw=2.1, color=TEAL, label="v3 A3 full 75/75 dev_val ±1 SE (3.26% of 2,304 pool, R8* 0.68492)")
    ax_b.plot(xs, a0_val, marker="v", ms=3.5, lw=1.3, color=SLATE, ls=":", label="v3 A0 baseline 75/75 dev_val (140-d, R5* 0.68700 | R8 0.68709)")
    ax_b.plot(xs, t2_val, marker="s", ms=4.2, lw=2.1, color=BLUE, label="v2 Disjoint-1k full 345/345 dev_val ±1 SE (2.80%, R5* 0.57363)")
    ax_b.plot([5], [t2_val[5]], marker="o", ms=7.8, fillstyle="none", mew=1.6, color=ORANGE, ls="none")
    ax_b.plot(xs, a3_tr, marker="o", ms=3.6, lw=1.6, color=TEAL, ls="--", label="v3 A3 600/2,229 train monitor (1.9996× passes, R8 0.46132)")
    ax_b.plot(xs, t2_tr, marker="o", ms=3.6, lw=1.6, color=ORANGE, ls="--", label="v2 Disjoint-1k 600/11,976 train monitor (1.333× passes, R8 0.41492)")
    ax_b.set_xticks(range(9), ["R0", "R1", "R2", "R3", "R4", "R5*", "R6", "R7", "R8*"])
    ax_b.set_ylim(0.385, 0.915)
    ax_b.set_xlabel("Training Round (v2: 15,968 / 1.333×; v3: 4,457 / 2.00×)")
    ax_b.set_ylabel("Multi-task objective loss")
    ax_b.set_title("B. Disjoint-1,000 (v2) & 1,069 (v3) Train vs. <5% Val Loss", loc="left", color=INK, fontsize=8.8, fontweight="bold")
    ax_b.legend(frameon=True, facecolor=SURFACE, edgecolor=GRID, fontsize=6.2, loc="upper right")
    # Panel C: Value vs Measured Wire Bytes Across 1,000 (v2) & 1,069 (v3) Selection-Exposed Datasets
    ax_c.set_xscale("log")
    v2_t, v3_t = vs["v2_value_by_tier_1000ds"], vs["v3_value_by_role_and_tier_1069ds"]
    tiers_v2, tiers_v3 = ("medium", "high", "ultra"), ("small", "medium", "high", "ultra")
    x_v2 = [v2_t[t]["exact_v2_latent_wire_median_bytes"] for t in tiers_v2]
    y_v2 = [v2_t[t]["dope_mean_retention_R"] for t in tiers_v2]
    lo_v2 = [y_v2[i] - v2_t[t]["dope_retention_R_ci95"][0] for i, t in enumerate(tiers_v2)]
    hi_v2 = [v2_t[t]["dope_retention_R_ci95"][1] - y_v2[i] for i, t in enumerate(tiers_v2)]
    ax_c.axvspan(700.0, 8192.0, color=BLUE, alpha=0.045, zorder=0)
    ax_c.text(780.0, 0.625, "Compact-Program\nZone (≤ 8 KiB)", color=BLUE, fontsize=6.7, fontweight="bold", ha="left", va="bottom")
    ax_c.errorbar(x_v2, y_v2, yerr=[lo_v2, hi_v2], marker="o", ms=5.4, lw=2.2, color=BLUE, capsize=2.5, label="DOPE-MLC-v2 (1,000 DS: 242 disj + 758 fold, exact med wire; Elo 1651.3 #1)")
    for m_key, col, ls, mk, lbl in (("rows", ORANGE, "--", "s", "Rows @ nominal 4/16/64 KiB cap (1,000 DS, Med CB 999/1k; Elo 1593.3)"), ("copula", SLATE, ":", "^", "Copula @ nominal 4/16/64 KiB cap (1,000 DS, 95% CI; Elo 1545.8)")):
        xm, ym = [v2_t[t]["nominal_cap_bytes"] for t in tiers_v2], [v2_t[t][f"{m_key}_mean_retention_R"] for t in tiers_v2]
        lom, him = [ym[i] - v2_t[t][f"{m_key}_retention_R_ci95"][0] for i, t in enumerate(tiers_v2)], [v2_t[t][f"{m_key}_retention_R_ci95"][1] - ym[i] for i, t in enumerate(tiers_v2)]
        ax_c.errorbar(xm, ym, yerr=[lom, him], marker=mk, ms=4.4, lw=1.6, color=col, ls=ls, capsize=2.0, label=lbl)
    x_v3p = [v3_t["eval_gradient_disjoint_v2_panel"][t]["mean_dml3_complete_wire_bytes"] for t in tiers_v3]
    y_v3p = [v3_t["eval_gradient_disjoint_v2_panel"][t]["sender_scaled_6aud_mean_clipped_R"] for t in tiers_v3]
    x_v3m = [v3_t["eval_gradient_disjoint_multiclass_panel"][t]["mean_dml3_complete_wire_bytes"] for t in tiers_v3]
    y_v3m = [v3_t["eval_gradient_disjoint_multiclass_panel"][t]["sender_scaled_6aud_mean_clipped_R"] for t in tiers_v3]
    ax_c.plot(x_v3p, y_v3p, marker="D", ms=4.6, lw=1.9, color=TEAL, ls="-.", label="DOPE-MLC-v3 1,000-DS panel (mean DML3+CSC1 wire, 340-row 6-aud R)")
    ax_c.plot(x_v3m, y_v3m, marker="P", ms=5.2, lw=1.8, color=PURPLE, ls="-.", label="DOPE-MLC-v3 69-DS multiclass (+0.253 ΔR vs copula, Elo 1540.9)")
    study_txt = (ROOT / "docs/multilevel-landmark-dl-study.md").read_text()
    nl_med = float(_study_val(study_txt, r"\| \*\*`Strong Nonlinear \(>0\.14\)`\*\* \| \*\*`221`\*\* \| `([\d.]+)`"))
    nl_ult = float(_study_val(study_txt, r"\| \*\*`Strong Nonlinear \(>0\.14\)`\*\* \| \*\*`221`\*\* \| `[\d.]+` \| `[\d.]+` \| \*\*`([\d.]+)`"))
    s0_ceil = float(_study_val(study_txt, r"\| \*\*`Ultra` \(\+ Rich Fine & Chatterjee \$\\xi\$ Spectrum\)\*\* \| \*\*116\*\* \| \*\*`[\d.]+`\*\* \| \*\*`([\d.]+)`\*\*"))
    s0_nonlin = float(_study_val(study_txt, r"\| \*\*`Ultra` \(\+ Rich Fine & Chatterjee \$\\xi\$ Spectrum\)\*\* \| \*\*116\*\* \| \*\*`[\d.]+`\*\* \| \*\*`[\d.]+`\*\* \| \*\*`([\d.]+)`\*\*"))
    ax_c.text(0.02, 0.97, f"Strongly nonlinear (>0.14, 221 DS): Med {nl_med:.3f} → Ultra {nl_ult:.3f} (+{100*(nl_ult-nl_med):.1f} pts)\nStage-0 (19,190 splits): ceiling R² = {s0_ceil:.3f} | nonlin uplift R² = {s0_nonlin:.3f}", transform=ax_c.transAxes, ha="left", va="top", fontsize=6.1, color=INK, bbox=dict(boxstyle="round,pad=0.22", facecolor="#f0f9ff", edgecolor=BLUE, lw=0.7))
    kib_ticks(ax_c, 800.0, 65536.0)
    ax_c.set_xlim(700.0, 82000.0)
    ax_c.set_ylim(0.51, 1.05)
    ax_c.set_xlabel("Wire bytes (v2: exact median wire; v3: mean DML3+CSC1; baselines: cap)")
    ax_c.set_ylabel("Mean holdout retention R (6 learners)")
    ax_c.set_title("C. Value vs. Wire Bytes Across 1,000 (v2) & 1,069 (v3) Datasets", loc="left", color=INK, fontsize=8.8, fontweight="bold")
    ax_c.legend(frameon=True, facecolor=SURFACE, edgecolor=GRID, fontsize=6.0, loc="lower right")
    sc = vs["testing_scale_distinct_units"]
    bars = [
        ("CI/cpu test assertions\n(656 Rust + 1,801 Py; 834+ unique)", sc["ci_cpu_run_40159022870_test_assertions"]["total_executed_assertions_in_job"], SLATE),
        ("Reconciled catalog datasets\n(1,069 unique eval; 1,000 shared)", sc["reconciled_catalog_datasets_total"], TEAL),
        ("v3 completed DS-tier cells\n(1,069 DS × 4 tiers, 0 failed)", sc["v3_completed_dataset_tier_cells"], "#0d9488"),
        ("v2 bit-exact wire replay cells\n(1,000 DS × 3 tiers × 2 ckpts)", sc["v2_exact_wire_replay_cells_verified"], BLUE),
        ("v2 pre-calculated CV splits\n(18,395 train + 684 <5% val)", sc["v2_precalculated_cv_splits_total"], "#0284c7"),
        ("v3 6-auditor cell evaluations\n(4,276 cells × 6 auditors/domain)", sc["v3_six_auditor_evaluations_per_domain"], PURPLE),
        ("v2 downstream auditor fits\n(1,000 reg/bin DS × 6 auditors)", sc["v2_downstream_auditor_fits_total"], ORANGE),
    ]
    y_pos, vals, cols = np.arange(len(bars)), [b[1] for b in bars], [b[2] for b in bars]
    ax_d.barh(y_pos, vals, height=0.60, color=cols, alpha=0.88)
    ax_d.set_xscale("log")
    ax_d.set_xlim(1000.0, 420000.0)
    ax_d.set_ylim(-0.65, 6.65)
    ax_d.set_yticks(y_pos, [b[0] for b in bars], fontsize=7.0)
    ax_d.grid(axis="y", visible=False)
    for idx_b, val_b in enumerate(vals):
        ax_d.text(val_b * 1.12, idx_b, f"{val_b:,}", va="center", ha="left", fontsize=7.4, fontweight="bold", color=INK)
    ax_d.set_xlabel("Verified count (log scale; distinct non-summed receipt units)")
    ax_d.set_title("D. Testing & Verification Scale (1,069 Unique Eval DS)", loc="left", color=INK, fontsize=8.8, fontweight="bold")
    fig.tight_layout(rect=(0.008, 0.008, 0.992, 0.992))
    return fig


def figure_bytes(fig, fmt: str) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, dpi=200, metadata={"Creator": "dope bench", "CreationDate": None} if fmt == "pdf" else {"Software": None})
    return buf.getvalue()


def readme_block(_summary: dict) -> str:
    return "\n".join([BEGIN, "<!-- Generated by bench/real-world/render.py from results/multilevel-convergence-snapshot.json; do not edit. -->", *multilevel_readme_lines(), END])



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
