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

ROOT = Path(__file__).resolve().parents[2]
BEGIN, END = "<!-- BEGIN DOPE1 RESULTS -->", "<!-- END DOPE1 RESULTS -->"
# Palette (validated categorical slots 1-2 + neutral; see the dataviz method).
BLUE, ORANGE, GRAY, INK, INK2, GRID, SURFACE = "#2a78d6", "#eb6834", "#8a8984", "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
GENERATORS = [("copula", "Gaussian copula"), ("arf", "ARF"), ("tvae", "TVAE"), ("ctgan", "CTGAN")]
GRAYS = ["#b8b7b2", "#8a8984", "#62615d", "#3d3c39"]  # neutral steps for the generator bars
BUDGET_ORDER = ["1k", "2k", "4k", "8k", "16k", "64k", "full"]


def kb(x: float) -> str:
    if x is None:
        return "–"
    if x >= 1024 ** 2:
        return f"{x / 1024 ** 2:.1f} MiB"
    if x >= 1024:
        return f"{x / 1024:.1f} KiB"
    return f"{x:.0f} B"


def seconds(s: float) -> str:
    return "--" if s != s else (f"{s:.1f}~s" if s < 100 else f"{s / 60:.0f}~min")


def rate(r: float) -> str:
    if r != r:
        return "--"
    return f"{r / 1e6:.1f}M" if r >= 1e6 else f"{r / 1e3:.0f}k" if r >= 1e3 else f"{r:.0f}"


def row(summary: dict, method: str, budget: str) -> dict | None:
    return next((r for r in summary["methods"] if r["method"] == method and r["budget"] == budget), None)


def frontier(summary: dict, method: str) -> list[dict]:
    rows = [row(summary, method, b) for b in BUDGET_ORDER]
    return [r for r in rows if r and r["datasets"] >= max(3, len(summary["datasets"]) // 2)]


def tdl_stats(summary: dict, key: str) -> tuple[int, float | None]:
    vals = [v for v in summary["tdl"].get(key, {}).values() if v is not None]
    import statistics
    return len(vals), (statistics.median(vals) if vals else None)


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
              "(bench/real-world/figures/coverage.png)", END]
    return "\n".join(lines)


def ret(v: float, fmt: str = ".3f", tex: bool = True) -> str:
    """A retention for tables: values below -99 (a learner far worse than the null) print as a bound."""
    if v < -99:
        return "$<\\!-99$" if tex else "< −99"
    return signed(v, fmt) if tex else format(v, fmt)


def signed(v: float, fmt: str) -> str:
    """A number for TeX text mode, with a typographic minus sign."""
    s = format(v, fmt)
    return "\\ensuremath{-}" + s[1:] if s.startswith("-") else s


def macros(summary: dict) -> str:
    out = ["% Generated by bench/real-world/render.py from summary.json; do not edit."]
    def put(name: str, value: str):
        out.append(f"\\newcommand{{\\{name}}}{{{value}}}")
    put("OneTables", str(len(summary["datasets"])))
    names = {"dope": "Dope", "kernel": "Kernel", "rows": "Rows", "copula": "Copula", "arf": "Arf", "ctgan": "Ctgan", "tvae": "Tvae",
             "bootstrap": "Boot", "dope-independent": "DopeIndep", "dope-linear": "DopeLinear", "dope-8bit": "DopeEight",
             "dope-1regime": "DopeOneRegime", "dope-4x": "DopeFour", "dope-notrees": "DopeNoTrees",
             "dope-noaudit": "DopeNoAudit", "dope-seed0": "DopeSeedZero"}
    budgets = {"1k": "A", "2k": "B", "4k": "C", "8k": "D", "16k": "E", "64k": "F", "full": "Full", "-": ""}
    for r in summary["methods"]:
        key = names.get(r["method"])
        if not key:
            continue
        stem = f"One{key}{budgets[r['budget']]}"
        put(stem + "N", str(r["datasets"]))
        if r["datasets"] == 0:  # every fit failed at this budget; the text shows a pending marker
            continue
        rr = r["robust"]
        put(stem + "Ret", signed(rr["median"], ".3f"))
        put(stem + "Lo", signed(rr["lo"], ".3f"))
        put(stem + "Hi", signed(rr["hi"], ".3f"))
        put(stem + "Bytes", kb(r["bytes"]["median"]).replace(" ", "~"))
        put(stem + "Reach", str(r["reach90"]))
        if r["method"] not in ("rows", "bootstrap"):  # controls have no fit or sampler
            put(stem + "Fit", seconds(r["fit_s"]))
            put(stem + "Rate", rate(r["rows_per_s"]))
        for f, v in r["families"].items():
            put(stem + {"linear": "Lin", "gbdt": "Cb", "mlp": "Mlp"}[f], signed(v, ".3f"))
    for k, c in summary["contrasts"].items():
        m, b = k.split("@")
        key = names.get(m)
        if key:
            stem = f"OneDiff{key}{budgets[b]}"
            put(stem, signed(c["hl"], "+.3f"))
            put(stem + "Lo", signed(c["lo"], "+.3f"))
            put(stem + "Hi", signed(c["hi"], "+.3f"))
            put(stem + "Wtl", f"{c['wins']}/{c['ties']}/{c['losses']}")
    for b, c in summary.get("equal_bytes", {}).items():
        stem = f"OneEq{budgets[b]}"
        put(stem + "Wtl", f"{c['wins']}/{c['ties']}/{c['losses']}")
        put(stem + "P", f"{c['sign_p']:.2g}")
        put(stem + "N", str(c["n"]))
    kern = summary.get("kernel")
    if kern:
        put("OneKernelSupported", str(kern["supported"]))
        put("OneKernelUnsupported", str(kern["unsupported"]))
        for b, c in kern["vs"].items():
            stem = f"OneKernelVs{budgets[b]}"
            put(stem + "Wtl", f"{c['wins']}/{c['ties']}/{c['losses']}")
            put(stem + "Hl", signed(c["hl"], "+.3f"))
            put(stem + "Lo", signed(c["lo"], "+.3f"))
            put(stem + "Hi", signed(c["hi"], "+.3f"))
            put(stem + "P", f"{c['sign_p']:.2g}")
            put(stem + "N", str(c["n"]))
    au = summary.get("self_audit")
    if au:
        put("OneAuditN", str(au["n"]))
        put("OneAuditRho", f"{au['spearman']:.2f}")
        put("OneAuditMae", f"{au['mae']:.3f}")
        put("OneAuditBias", signed(au["bias"], "+.3f"))
    for m, c in summary.get("ablation", {}).items():
        key = names.get(m)
        if key:
            put(f"OneAbl{key}Wtl", f"{c['wins']}/{c['ties']}/{c['losses']}")
            put(f"OneAbl{key}P", f"{c['sign_p']:.2g}")
    eq = summary.get("equal_bytes", {})
    put("OneEqBudgets", str(len(eq)))
    put("OneEqWins", str(sum(c["wins"] for c in eq.values())))
    put("OneEqLosses", str(sum(c["losses"] for c in eq.values())))
    put("OneEqCells", str(sum(c["n"] for c in eq.values())))
    for b, k in summary.get("dope_failed", {}).items():
        put(f"OneDopeFailed{budgets[b]}", str(k))
    for key, name in (("dope@0.9", "OneTdlDope"), ("rows@0.9", "OneTdlRows"), ("dope@0.8", "OneTdlDopeEighty"),
                      ("rows@0.8", "OneTdlRowsEighty")):
        k, med = tdl_stats(summary, key)
        put(name + "N", str(k))
        put(name + "Median", kb(med).replace(" ", "~") if med else "not~reached")
    return "\n".join(out) + "\n"


def _const(path: str, pattern: str) -> str:
    import re
    text = (ROOT / path).read_text()
    m = re.search(pattern, text, re.M)
    if not m:
        raise ValueError(f"constant {pattern!r} not found in {path}")
    return m.group(1).replace("_", "")


def config_macros() -> str:
    """Hyperparameters read from the engine and harness sources (never retyped)."""
    src = "engine/dope-core/src/"
    vals = {
        "CfgBudget": str(int(_const(src + "compile.rs", r"DEFAULT_BUDGET: usize = ([\d_]+)")) // 1024),
        "CfgZGrid": _const(src + "marginal.rs", r"Z_GRID: f64 = ([\d.]+)").rstrip("0").rstrip("."),
        "CfgMaxDiscrete": _const(src + "marginal.rs", r"MAX_DISCRETE: usize = (\d+)"),
        "CfgAtomShare": str(round(100 * float(_const(src + "marginal.rs", r"ATOM_SHARE: f64 = ([\d.]+)")))),
        "CfgResidualKnots": _const(src + "noise.rs", r"RESIDUAL_KNOTS: usize = (\d+)"),
        "CfgMaxRegimes": _const(src + "regimes.rs", r"MAX_REGIMES: usize = (\d+)"),
        "CfgRegimeLadder": _const(src + "regimes.rs", r"REGIME_LADDER: \[usize; \d+\] = \[([^\]]+)\]").replace(" ", ""),
        "CfgSteps": _const(src + "train.rs", r"max_steps: ([\d_]+),"),
        "CfgLearningRate": _const(src + "train.rs", r"learning_rate: ([\de.-]+),"),
        "CfgWeightDecay": _const(src + "train.rs", r"weight_decay: ([\de.-]+),"),
        "CfgBatch": _const(src + "train.rs", r"batch: ([\d_]+),"),
        "CfgEvalEvery": _const(src + "train.rs", r"eval_every: (\d+),"),
        "CfgPatience": _const(src + "train.rs", r"patience: (\d+),"),
        "CfgTolerance": str(round(1000 * float(_const(src + "compile.rs", r"tolerance: ([\d.]+),"))) / 10),
        "CfgReadoutRows": _const(src + "target.rs", r"READOUT_ROWS: usize = ([\d_]+)"),
        "CfgScoreTable": _const(src + "sample.rs", r"SCORE_TABLE: usize = (\d+)"),
        "CfgMinLevel": _const(src + "schema.rs", r"min_level_count: (\d+)"),
        "CfgMaxLevels": _const(src + "schema.rs", r"max_levels: (\d+)"),
        "CfgLevels": _const(src + "forest.rs", r"LEVELS: usize = (\d+)"),
        "CfgMaxTrees": _const(src + "compile.rs", r"MAX_TREES: usize = ([\d_]+)"),
        "CfgLeafBits": _const(src + "compile.rs", r"TREE_LEAF_BITS: u8 = (\d+)"),
        "CfgShares": ", ".join(str(round(100 * float(v))) for v in
                               _const(src + "compile.rs", r"DEPENDENCE_SHARES: \[f64; \d+\] = \[([^\]]+)\]").split(",")),
        "CfgAuditRows": _const(src + "eval/audit.rs", r"SELF_AUDIT_ROWS: usize = ([\d_]+)"),
        "CfgMaxTrain": _const("bench/real-world/datasets.py", r"MAX_TRAIN, MAX_TEST, SEED = ([\d_]+),"),
        "CfgMaxTest": _const("bench/real-world/datasets.py", r"MAX_TRAIN, MAX_TEST, SEED = [\d_]+, ([\d_]+),"),
        "CfgCatIters": _const("bench/real-world/auditors.py", r"iterations=(\d+)"),
        "CfgCatDepth": _const("bench/real-world/auditors.py", r"depth=(\d+)"),
        "CfgDraws": _const("bench/real-world/report.py", r"DRAWS, SEED = ([\d_]+),"),
        "CfgTimeoutMin": str(int(_const("bench/real-world/run.py", r"GEN_TIMEOUT = (\d+)")) // 60),
        "CfgPairedDraws": _const("bench/real-world/report.py", r"PAIRED_DRAWS = ([\d_]+)"),
        "CfgTau": _const("bench/real-world/report.py", r"TAU, TAU_LOW = ([\d.]+),"),
        "CfgTauLow": _const("bench/real-world/report.py", r"TAU, TAU_LOW = [\d.]+, ([\d.]+)"),
        "CfgConf": str(round(100 * (1 - float(_const("bench/real-world/report.py", r"ALPHA = ([\d.]+)"))))),
    }
    plans = _const(src + "compile.rs", r"PLANS: \[\(usize, usize, usize\); \d+\] = \[([^;]+?)\];")
    import re
    tuples = re.findall(r"\((\d+), (\d+), (\d+)\)", plans)
    vals["CfgPlanRich"] = "({}, {}, {})".format(*tuples[0])
    vals["CfgPlanLean"] = "({}, {}, {})".format(*tuples[-1])
    vals["CfgPlans"] = str(len(tuples))
    def tex(v: str) -> str:  # thousands separators for long integers; powers of ten
        import re as _re
        if _re.fullmatch(r"1e-\d+", v):
            return f"$10^{{{v[2:]}}}$"
        return f"{int(v):,}".replace(",", "{,}") if v.isdigit() and len(v) > 4 else v
    return "\n".join(["% Generated by bench/real-world/render.py from engine and harness sources; do not edit."] +
                     [f"\\newcommand{{\\{k}}}{{{tex(v)}}}" for k, v in vals.items()]) + "\n"


def _cell(r: dict | None, fam: str | None = None) -> str:
    if not r or r["datasets"] == 0:
        return "--"
    if fam:
        v = r["families"].get(fam)
        return "--" if v is None else ret(v, ".2f")
    rr = r["robust"]
    return f"{ret(rr['median'])} [{ret(rr['lo'], '.2f')}, {ret(rr['hi'], '.2f')}]"


def table_main(summary: dict) -> str:
    n = len(summary["datasets"])
    order = [("dope", "2k", "DOPE, 2 KiB"), ("dope", "4k", "\\textbf{DOPE, 4 KiB (default)}"), ("dope", "16k", "DOPE, 16 KiB"),
             ("dope", "64k", "DOPE, 64 KiB"), ("kernel", "-", "DOPE research kernel"),
             ("rows", "4k", "Real rows (xz), 4 KiB"), ("rows", "64k", "Real rows (xz), 64 KiB"),
             ("rows", "full", "Real rows (xz), all"), ("copula", "-", "Gaussian copula (SDV)"), ("arf", "-", "ARF"),
             ("tvae", "-", "TVAE (SDV)"), ("ctgan", "-", "CTGAN (SDV)"), ("bootstrap", "-", "Real rows, resampled")]
    lines = ["\\begin{tabular}{lrrrrrr}", "\\toprule",
             "Method & Bytes & Robust retention [95\\% CI] & Linear & CatBoost & MLP & $\\geq 0.9$ \\\\", "\\midrule"]
    for m, b, label in order:
        r = row(summary, m, b)
        if not r or r["datasets"] == 0:
            continue
        tables = "" if r["datasets"] == n else f"$^{{{r['datasets']}}}$"
        lines.append(f"{label}{tables} & {kb(r['bytes']['median']).replace(' ', '~')} & {_cell(r)} & {_cell(r, 'linear')} & "
                     f"{_cell(r, 'gbdt')} & {_cell(r, 'mlp')} & {r['reach90']}/{r['datasets']} \\\\")
        if (m, b) in (("kernel", "-"), ("rows", "full")):
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def table_equal_bytes(summary: dict) -> str:
    """DOPE against xz-compressed real rows at each budget, paired over tables."""
    lines = ["\\begin{tabular}{lrrrrrr}", "\\toprule",
             "Budget & Tables & DOPE & Real rows (xz) & DOPE wins/ties/losses & Sign test $p$ & DOPE did not fit \\\\",
             "\\midrule"]
    for b in BUDGET_ORDER:
        c = summary.get("equal_bytes", {}).get(b)
        if not c:
            continue
        lines.append(f"{b.replace('k', '~KiB')} & {c['n']} & {ret(c['a'])} & {ret(c['b'])} & "
                     f"{c['wins']}/{c['ties']}/{c['losses']} & {c['sign_p']:.2g} & {c.get('dope_failed', 0)} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def table_ablation(summary: dict) -> str:
    order = [("dope-seed0", "-", "DOPE (default)"), ("dope-independent", "-", "independent features, one regime"),
             ("dope-1regime", "-", "one regime (plain Gaussian copula)"), ("dope-notrees", "-", "network only, no trees"),
             ("dope-linear", "-", "linear target model, no trees"), ("dope-noaudit", "-", "fixed byte split, no self-audit"),
             ("dope-8bit", "-", "uniform 8-bit network weights"), ("dope-4x", "-", "four times as many synthetic rows")]
    lines = ["\\begin{tabular}{lrrrrrr}", "\\toprule",
             "Variant (4 KiB, seed 0) & Bytes & Robust retention [95\\% CI] & Linear & CatBoost & MLP & Better/tied/worse \\\\",
             "\\midrule"]
    for m, b, label in order:
        r = row(summary, m, b)
        if not r or r["datasets"] == 0:
            continue
        c = summary.get("ablation", {}).get(m)
        wtl = f"{c['wins']}/{c['ties']}/{c['losses']}" if c else "--"
        lines.append(f"{label} & {kb(r['bytes']['median']).replace(' ', '~')} & {_cell(r)} & {_cell(r, 'linear')} & "
                     f"{_cell(r, 'gbdt')} & {_cell(r, 'mlp')} & {wtl} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def table_per_dataset(summary: dict) -> str:
    pd_ = summary["per_dataset"]
    meta = summary.get("meta", {})
    lines = ["\\begin{tabular}{llrrrrrrrrr}", "\\toprule",
             "Table & Task & Train & Cols & DOPE bytes & DOPE & Kernel & Rows 4 KiB & Copula & ARF & TDL$_{0.9}$ \\\\",
             "\\midrule"]
    def r(key, ds):
        c = pd_.get(key, {}).get(ds)
        if not c or c["robust"] is None:
            return "--"
        return "fail" if c.get("status") == "failed" else ret(c["robust"], ".2f")
    for ds in summary["datasets"]:
        m = meta.get(ds, {})
        dope = pd_.get("dope@4k", {}).get(ds) or {}
        t = summary["tdl"]["dope@0.9"].get(ds)
        lines.append(f"{ds} & {m.get('task', '')[:5]}. & {m.get('train_rows', '')} & {m.get('features', '')} & "
                     f"{kb(dope.get('bytes')).replace(' ', '~')} & {r('dope@4k', ds)} & {r('kernel@-', ds)} & {r('rows@4k', ds)} & "
                     f"{r('copula@-', ds)} & {r('arf@-', ds)} & {kb(t).replace(' ', '~') if t else 'n.r.'} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def throughput_tex(rows: list[dict]) -> tuple[str, str]:
    """Macros and a supplement table from results/throughput.json."""
    rates = sorted(r["rows_per_second"] for r in rows)
    import statistics
    at4 = [r for r in rows if r["budget"] == "4k"]
    meta_share = 100 * statistics.median(r["metadata"] / r["inflated"] for r in at4)
    stored_share = 100 * statistics.median(r["stored"] / r["inflated"] for r in at4)
    device = rows[0]["device"].replace("(", "").replace(")", "")
    macros = "\n".join([
        "% Generated by bench/real-world/render.py from results/throughput.json; do not edit.",
        f"\\newcommand{{\\OneThroughputMin}}{{{rate(rates[0])}}}",
        f"\\newcommand{{\\OneThroughputMax}}{{{rate(rates[-1])}}}",
        f"\\newcommand{{\\OneThroughputDevice}}{{{device}}}",
        f"\\newcommand{{\\OneMetaShare}}{{{meta_share:.0f}}}",
        f"\\newcommand{{\\OneStoredShare}}{{{stored_share:.0f}}}",
    ]) + "\n"
    lines = ["\\begin{tabular}{lrrrrrrr}", "\\toprule",
             "Table & Budget & Bytes & Inflated & Metadata & Features & Trees & Rows per second \\\\", "\\midrule"]
    for r in rows:
        lines.append(f"{r['dataset']} & {r['budget'].replace('k', '~KiB')} & {r['bytes']:,} & {r['inflated']:,} & "
                     f"{r['metadata']:,} & {r['features']} & {r['trees']} & {r['rows_per_second']:,} \\\\".replace(",", "{,}"))
    lines += ["\\bottomrule", "\\end{tabular}"]
    return macros, "\n".join(lines) + "\n"


def outputs(results: Path) -> dict[str, bytes]:
    summary = json.loads((results / "summary.json").read_text())
    plt = setup_matplotlib()
    files: dict[str, bytes] = {}
    for name, fn in (("hero", hero), ("coverage", coverage), ("families", families), ("tables", tables),
                     ("audit", audit)):
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
