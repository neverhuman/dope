"""Renderers for the compression frontier: figure, history-page block, TeX macros and tables.

Pure functions of the JSON written by compression_frontier.py. claim_registry.py calls
them to authenticate the committed HISTORY.md block and TeX files without recomputing the
bootstrap, so a hand edit to any rendered output fails the claims firewall.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

os.environ.setdefault("SOURCE_DATE_EPOCH", "1760000000")

FULL_PANEL_MIN = 80
EVIDENCE = "selection_exposed_validation"
README_BEGIN = "<!-- BEGIN COMPRESSION -->"
README_END = "<!-- END COMPRESSION -->"
HISTORY = "HISTORY.md"
FIGURE_PNG = "paper/generated/compression-frontier.png"
FIGURE_PDF = "paper/generated/compression-frontier.pdf"
DATA_JSON = "paper/generated/compression-frontier.json"

DOPE_GREEN = "#76B900"
DOPE_EDGE = "#3D6B00"
DOPE_INK = "#2F5A00"
GRAY = "#7A7A7A"
GRAY_LIGHT = "#BDBDBD"
INK = "#1A1A1A"
MUTED = "#555555"
GRID = "#E9E9E9"


# ----------------------------------------------------------------------------- text outputs

def fmt_bytes(value: float) -> str:
    if value >= 1e6:
        return f"{value / 1e6:.1f} MB"
    if value >= 1e3:
        return f"{value / 1e3:.1f} KB"
    return f"{value:.0f} B"


def fmt_ratio(value: float) -> str:
    if value >= 1000:
        return f"{value:,.0f}"
    return f"{value:.1f}"


def fmt_lift(value: float | None) -> str:
    if value is None:
        return "no lift"
    if value >= 0.01:
        return f"{value:.3f}"
    exponent = math.floor(math.log10(value))
    return f"{value / 10 ** exponent:.1f}e{exponent}"


def _ranked_generators(data: dict) -> list[dict]:
    """Generators ordered by retained-lift rank; those without positive lift last, in panel order."""
    rows = [r for r in data["rows"] if r["role"] == "generator"]
    return sorted(rows, key=lambda r: (r.get("lift_rank") is None, r.get("lift_rank") or 0))


def markdown_table(data: dict) -> str:
    lines = [
        "| Rank (lift/KiB) | Generator | Worst-auditor retention (4n) | CatBoost | Linear | MLP | Median charged bytes | Retained lift per KiB |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in _ranked_generators(data):
        a = row["auditors"]
        rank = row.get("lift_rank")
        name = f"**{row['display']}**" if row["method"] == "DOPE" else row["display"]
        if row["lineages"] < FULL_PANEL_MIN:
            name += f" ({row['lineages']} lineages)"
        worst = f"{row['worst_median']:.3f} [{row['worst_lo']:.3f}, {row['worst_hi']:.3f}]"
        lines.append(
            f"| {rank if rank else '—'} | {name} | {worst} | {a['catboost']['median']:.3f} | "
            f"{a['linear']['median']:.3f} | {a['mlp']['median']:.3f} | {fmt_bytes(row['bytes']['median'])} | "
            f"{fmt_lift(row['lift_per_kib'])} |")
    for row in data["rows"]:
        if row["role"] == "control":
            a = row["auditors"]
            lines.append(
                f"| — | {row['display']} *(control, not a generator)* | {row['worst_median']:.3f} | "
                f"{a['catboost']['median']:.3f} | {a['linear']['median']:.3f} | {a['mlp']['median']:.3f} | none | — |")
    return "\n".join(lines) + "\n"


def tex_macros(data: dict) -> str:
    rows = {(r["method"], r["configuration"]): r for r in data["rows"]}
    dope = rows[("DOPE", "headline")]
    tab = rows[("TabSyn", "scaled_200_vae_1000_diffusion")]
    gauss = rows[("GaussianCopula", "native_selected")]
    pred = rows[("predictor_only", "control")]
    untuned = rows[("DOPE", "product_default")]
    pair = data["dope_vs_tabsyn_worst"]
    raw = data["raw_table_float32_bytes"]
    dens = data["dope_density"]
    macro = {
        "CFLineages": str(data["lineages_all_auditors_informative"]),
        "CFDopeWorst": f"{dope['worst_median']:.3f}",
        "CFDopeWorstLo": f"{dope['worst_lo']:.3f}",
        "CFDopeWorstHi": f"{dope['worst_hi']:.3f}",
        "CFDopeBytes": f"{dope['bytes']['median']:,.0f}",
        "CFDopeKiB": f"{dope['bytes']['median'] / 1024:.2f}",
        "CFDopeUntunedWorst": f"{untuned['worst_median']:.3f}",
        "CFTabSynWorst": f"{tab['worst_median']:.3f}",
        "CFTabSynWorstLo": f"{tab['worst_lo']:.3f}",
        "CFTabSynWorstHi": f"{tab['worst_hi']:.3f}",
        "CFTabSynBytes": fmt_bytes(tab["bytes"]["median"]).replace(" ", "~"),
        "CFByteRatio": f"{data['byte_ratio_tabsyn_over_dope']:,.0f}",
        "CFPairHL": f"{pair['hl']:+.3f}",
        "CFPairLo": f"{pair['hl_lo']:+.3f}",
        "CFPairHi": f"{pair['hl_hi']:+.3f}",
        "CFPairN": str(pair["n"]),
        "CFGaussWorst": f"{gauss['worst_median']:.3f}",
        "CFPredWorst": f"{pred['worst_median']:.3f}",
        "CFPredCatBoost": f"{pred['auditors']['catboost']['median']:.3f}",
        "CFPredLinear": f"{pred['auditors']['linear']['median']:.3f}",
        "CFPredMlp": f"{pred['auditors']['mlp']['median']:.3f}",
        "CFDopeLift": f"{dope['lift_per_kib']:.3f}",
        "CFRunnerUp": data["runner_up"],
        "CFLiftRatio": f"{data['lift_ratio_dope_over_runner_up']:.1f}",
        "CFGenerators": str(len(data["lift_ranking"])),
        "CFRawTableMedian": fmt_bytes(raw["median"]).replace(" ", "~"),
        "CFBitsPerCell": f"{dens['bits_per_cell_median']:.1f}",
        "CFFloatRatio": f"{dens['float32_ratio_median']:.1f}",
        "CFSidecarShare": f"{100 * dens['sidecar_share_median']:.0f}",
    }
    for auditor in ("catboost", "linear", "mlp"):
        name = {"catboost": "CatBoost", "linear": "Linear", "mlp": "Mlp"}[auditor]
        macro[f"CFDope{name}"] = f"{dope['auditors'][auditor]['median']:.3f}"
        macro[f"CFTabSyn{name}"] = f"{tab['auditors'][auditor]['median']:.3f}"
    boot = rows[("real_bootstrap_4n", "control")]
    macro["CFIntervalLevel"] = "95\\%"
    macro["CFBootMlp"] = f"{boot['auditors']['mlp']['median']:.3f}"
    macro["CFGaussBytes"] = fmt_bytes(gauss["bytes"]["median"]).replace(" ", "~")
    labels = {"Kumo Large": "Kumo", "TabICLv2": "Tabicl", "Mitra-v2": "Mitra"}
    for row in data.get("foundation", []):
        who = "Dope" if row["method"].startswith("DOPE") else "Gauss"
        key = f"CFFm{labels[row['auditor']]}{who}"
        macro[key] = f"{row['retention']:.3f}"
        macro[key + "Lo"] = f"{row['lo']:.3f}"
        macro[key + "Hi"] = f"{row['hi']:.3f}"
    if data.get("foundation"):
        macro["CFFmLineages"] = str(data["foundation"][0]["lineages"])
    body = "".join(f"\\providecommand{{\\{k}}}{{{v}}}\n" for k, v in macro.items())
    return ("% Generated by docs/whitepaper/scripts/compression_frontier.py; do not edit.\n"
            f"% Evidence status: {EVIDENCE}.\n" + body)


def tex_table(data: dict) -> str:
    out = ["% Generated by docs/whitepaper/scripts/compression_frontier.py; do not edit.",
           "\\begin{tabular}{@{}rlcccccr@{}}", "\\toprule",
           "Rank & Generator & Worst auditor & CatBoost & Linear & MLP & Bytes & Lift/KiB \\\\", "\\midrule"]
    for row in _ranked_generators(data):
        a = row["auditors"]
        name = row["display"]
        if row["method"] == "DOPE":
            name = "\\textbf{" + name + "}"
        if row["lineages"] < FULL_PANEL_MIN:
            name += "$^\\dagger$"
        worst = f"{row['worst_median']:.3f} [{row['worst_lo']:.2f}, {row['worst_hi']:.2f}]"
        rank = str(row.get("lift_rank", "--"))
        lift = fmt_lift(row["lift_per_kib"])
        if "e" in lift:
            mant, exp = lift.split("e")
            lift = f"${mant}{{\\times}}10^{{{exp}}}$"
        out.append(f"{rank} & {name} & {worst} & {a['catboost']['median']:.3f} & {a['linear']['median']:.3f} & "
                   f"{a['mlp']['median']:.3f} & {fmt_bytes(row['bytes']['median']).replace(' ', '~')} & {lift} \\\\")
    out.append("\\midrule")
    for row in data["rows"]:
        if row["role"] == "control":
            a = row["auditors"]
            out.append(f"-- & \\emph{{{row['display']}}} & {row['worst_median']:.3f} & {a['catboost']['median']:.3f} & "
                       f"{a['linear']['median']:.3f} & {a['mlp']['median']:.3f} & -- & -- \\\\")
    out += ["\\bottomrule", "\\end{tabular}", ""]
    return "\n".join(out)


# ----------------------------------------------------------------------------- figure

def render(data: dict, formats: dict[str, Path]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, Rectangle

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9.5,
        "axes.edgecolor": "#4A4A4A", "axes.linewidth": 0.8, "pdf.fonttype": 42,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": INK,
        "svg.hashsalt": "dope-compression-frontier",
    })
    rows = {(r["method"], r["configuration"]): r for r in data["rows"]}
    fig = plt.figure(figsize=(7.2, 3.3))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.85, 1.0], wspace=0.62,
                            left=0.075, right=0.985, top=0.84, bottom=0.145)
    ax = fig.add_subplot(grid[0])
    bx = fig.add_subplot(grid[1])

    x_lo, x_hi = 1.5e2, 2e9
    y_floor, y_top = 0.40, 1.16
    ax.set_xscale("log")
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_floor, y_top)
    ax.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.grid(True, which="major", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    dope = rows[("DOPE", "headline")]
    dx, dy = dope["bytes"]["median"], dope["worst_median"]
    ax.add_patch(Rectangle((dx, y_floor), x_hi - dx, dy - y_floor, facecolor=DOPE_GREEN,
                           alpha=0.08, edgecolor="none", zorder=0))
    ax.text(1.2e7, 0.53, "DOPE dominates this region:\nmore bytes and less transfer", color=DOPE_INK,
            fontsize=6.6, ha="center", va="center")
    raw = data["raw_table_float32_bytes"]
    ax.axvspan(raw["q1"], raw["q3"], ymax=(1.0 - y_floor) / (y_top - y_floor), color="#000000",
               alpha=0.07, lw=0, zorder=0)
    ax.text(math.sqrt(raw["q1"] * raw["q3"]) * 1.05, 0.615, "raw table size (IQR)", fontsize=6.0, color=MUTED,
            ha="center", va="center", rotation=90)
    ax.axhline(0.90, color=INK, lw=0.8, ls=(0, (4, 3)), zorder=1)
    ax.text(1.6e5, 0.905, "T90 transfer bar", fontsize=6.6, color=INK, ha="center", va="bottom")
    for key, text, offset, va, xpos in ((("predictor_only", "control"), "predictor-only control", -0.006, "top", x_hi / 1.15),
                                        (("real_bootstrap_4n", "control"), "real rows (reference)", 0.006, "bottom", 1.2e5)):
        level = rows[key]["worst_median"]
        ax.axhline(level, color=GRAY, lw=0.7, ls=":", zorder=1)
        ax.text(xpos, level + offset, text, fontsize=6.3, color=MUTED, ha="right" if xpos > 1e8 else "center", va=va)

    placement = {
        "TabSyn": (1 / 1.35, 0.012, "right", "bottom"),
        "Forest-Flow": (1.0, 0.022, "center", "bottom"),
        "ARF": (1.0, -0.03, "center", "top"),
        "TVAE": (1.25, 0.0, "left", "center"),
        "Gaussian copula": (1.25, 0.0, "left", "center"),
        "Chow-Liu tree": (1.25, 0.0, "left", "center"),
        "DOPE, untuned default": (1 / 1.35, 0.0, "right", "center"),
    }
    off_scale = []
    for row in data["rows"]:
        if row["role"] == "control" or not row["bytes"]:
            continue
        x, y = row["bytes"]["median"], row["worst_median"]
        is_dope = row["method"] == "DOPE" and row["configuration"] == "headline"
        small = row["lineages"] < FULL_PANEL_MIN
        if y < y_floor:
            off_scale.append(row)
            continue
        if is_dope:
            face, edge, size, z, whisker = DOPE_GREEN, DOPE_EDGE, 10, 6, DOPE_EDGE
        elif row["method"] == "DOPE":
            face, edge, size, z, whisker = "white", "#8DB85A", 5, 3, "#C9DFAE"
        else:
            face, edge, size, z, whisker = ("white" if small else GRAY), "#4A4A4A", 6, 4, "#6A6A6A"
        ax.plot([row["bytes"]["q1"], row["bytes"]["q3"]], [y, y], color=whisker, lw=0.8, alpha=0.7, zorder=z - 1)
        ax.plot([x, x], [row["worst_lo"], row["worst_hi"]], color=whisker, lw=1.1, zorder=z - 1)
        ax.plot([x], [y], marker="o", ms=size, mfc=face, mec=edge, mew=1.2, zorder=z, ls="none")
        if is_dope:
            ax.text(x * 1.45, y - 0.008, "DOPE (ours)", fontsize=9.5, fontweight="bold", color=DOPE_INK,
                    ha="left", va="top", zorder=8)
            continue
        scale, dyy, ha, va = placement.get(row["display"], (1.0, -0.03, "center", "top"))
        text = row["display"] + (f" (n={row['lineages']})" if small else "")
        if row["method"] == "DOPE":
            text = "untuned\ndefault"
        ax.text(x * scale, y + dyy, text, fontsize=6.8, ha=ha, va=va,
                color=DOPE_INK if row["method"] == "DOPE" else MUTED, zorder=8)
    if off_scale:
        note = "below axis: " + ", ".join(f"{r['display']} ({r['worst_median']:+.2f})" for r in off_scale)
        ax.text(x_lo * 1.2, y_floor + 0.012, "↓ " + note, fontsize=6.0, color=MUTED, ha="left", va="bottom")

    tab = rows[("TabSyn", "scaled_200_vae_1000_diffusion")]
    ty = 1.09
    ax.add_patch(FancyArrowPatch((dx, ty), (tab["bytes"]["median"], ty), arrowstyle="<->",
                                 mutation_scale=9, color=DOPE_INK, lw=1.1, zorder=7))
    ax.plot([dx, dx], [dy + 0.03, ty], color=DOPE_INK, lw=0.6, ls=":", zorder=6)
    ax.plot([tab["bytes"]["median"]] * 2, [tab["worst_median"] + 0.02, ty], color=DOPE_INK, lw=0.6,
            ls=":", zorder=6)
    pair = data["dope_vs_tabsyn_worst"]
    mid = math.sqrt(dx * tab["bytes"]["median"])
    ax.text(mid, ty + 0.012, f"{data['byte_ratio_tabsyn_over_dope']:,.0f}× fewer bytes than TabSyn",
            fontsize=7.8, color=DOPE_INK, ha="center", va="bottom", fontweight="bold")
    ax.text(mid, ty - 0.012, f"transfer: paired Δ {pair['hl']:+.3f} [{pair['hl_lo']:+.3f}, {pair['hl_hi']:+.3f}]",
            fontsize=6.4, color=MUTED, ha="center", va="top")
    ax.set_xlabel("charged artifact bytes (median, log scale; bar = IQR)")
    ax.set_ylabel("worst-auditor retention at 4n")
    ax.set_title("(a) Transfer vs. bytes", loc="left", fontweight="bold", color=INK)

    ranked = sorted((r for r in data["rows"] if r.get("lift_rank")), key=lambda r: r["lift_rank"])
    positions = list(range(len(ranked)))[::-1]
    values = [r["lift_per_kib"] for r in ranked]
    colors = [DOPE_GREEN if r["method"] == "DOPE" else GRAY_LIGHT for r in ranked]
    edges = [DOPE_EDGE if r["method"] == "DOPE" else "#8A8A8A" for r in ranked]
    floor = 1e-6
    bx.barh(positions, [v - floor for v in values], left=floor, color=colors, edgecolor=edges,
            linewidth=0.8, height=0.64)
    bx.set_xscale("log")
    bx.set_xlim(floor, 40)
    bx.set_xticks([1e-6, 1e-4, 1e-2, 1])
    bx.set_yticks(positions)
    bx.set_yticklabels([f"#{r['lift_rank']}  {r['display']}" for r in ranked], fontsize=7.2)
    for tick, row in zip(bx.get_yticklabels(), ranked):
        tick.set_color(DOPE_INK if row["method"] == "DOPE" else INK)
        if row["method"] == "DOPE":
            tick.set_fontweight("bold")
    bx.tick_params(axis="y", length=0, pad=3)
    for side in ("top", "right"):
        bx.spines[side].set_visible(False)
    bx.grid(True, axis="x", which="major", color=GRID, linewidth=0.6)
    bx.set_axisbelow(True)
    for pos, row in zip(positions, ranked):
        bx.text(row["lift_per_kib"] * 1.4, pos, fmt_lift(row["lift_per_kib"]), fontsize=6.4, va="center",
                color=DOPE_INK if row["method"] == "DOPE" else MUTED,
                fontweight="bold" if row["method"] == "DOPE" else "normal")
    bx.set_xlabel("retained lift per KiB (log scale)")
    bx.set_title("(b) Retained lift per KiB", loc="left", fontweight="bold", color=INK)
    bx.text(values[0] * 1.4, positions[0] - 0.42, f"{data['lift_ratio_dope_over_runner_up']:.1f}× #2",
            fontsize=6.4, color=DOPE_INK, va="top", fontweight="bold")
    fig.text(0.075, 0.965,
             f"Historical selection-exposed validation · {data['lineages_all_auditors_informative']} PMLB "
             "regression lineages · 4n · worst of CatBoost, linear, MLP per lineage · 95% nested bootstrap",
             fontsize=6.4, color=MUTED, ha="left", va="center")
    for path in formats.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".png":
            fig.savefig(path, dpi=220, metadata={"Software": None})
        else:
            fig.savefig(path, metadata={"CreationDate": None})
    plt.close(fig)


def readme_block(data: dict) -> str:
    """The history-page block: figure, three numbered facts, evidence label."""
    rows = {(r["method"], r["configuration"]): r for r in data["rows"]}
    dope = rows[("DOPE", "headline")]
    tab = rows[("TabSyn", "scaled_200_vae_1000_diffusion")]
    pair = data["dope_vs_tabsyn_worst"]
    ranked = len(data["lift_ranking"])
    dope_kib = dope["bytes"]["median"] / 1024
    alt = ("Transfer versus charged bytes for every generator on the v2 panel, and retained "
           "predictive lift per KiB ranked; DOPE is the smallest point on the frontier.")
    lines = [
        README_BEGIN,
        f'<p align="center"><img src="{FIGURE_PNG}" width="100%" alt="{alt}"></p>',
        "",
        f"- **Smallest program on the transfer frontier.** DOPE's median program is "
        f"{dope['bytes']['median']:,.0f} bytes ({dope_kib:.2f} KiB) of dataset-specific state, "
        f"{data['byte_ratio_tabsyn_over_dope']:,.0f}× fewer bytes than TabSyn ({fmt_bytes(tab['bytes']['median'])}). "
        f"DOPE minus TabSyn in worst-auditor retention is {pair['hl']:+.3f} "
        f"[{pair['hl_lo']:+.3f}, {pair['hl_hi']:+.3f}] over {pair['n']} lineages, so the paired interval contains zero.",
        f"- **Rank 1 of {ranked} reported positive-lift generator estimates on retained lift per KiB** ({dope['lift_per_kib']:.3f} per KiB), "
        f"{data['lift_ratio_dope_over_runner_up']:.1f}× the runner-up ({data['runner_up']}); this is a descriptive ranking across the reported method-specific cohorts.",
        f"- **Every auditor family counts.** Worst of CatBoost, linear and MLP retention per lineage: "
        f"median {dope['worst_median']:.3f} [{dope['worst_lo']:.3f}, {dope['worst_hi']:.3f}] at synthetic size "
        f"{data['size']} on {data['lineages_all_auditors_informative']} PMLB regression lineages.",
        "",
        "<details><summary>Every generator, every auditor (generated table)</summary>",
        "",
        markdown_table(data).rstrip("\n"),
        "",
        "Auditor columns are paired within each method\'s reported cohort; smaller cohorts remain separate comparisons. Efficiency ranks are descriptive across available cohorts. Controls are diagnostics, not generators.",
        "</details>",
        "",
        "<sub>Historical selection-exposed validation: the panel also informed DOPE's profile choice, and the "
        "official test split is sealed. Generated by <code>docs/whitepaper/scripts/compression_frontier.py</code> "
        "from SHA-256-pinned ledgers; 95% nested cluster bootstrap intervals.</sub>",
        README_END,
    ]
    return "\n".join(lines)


def text_outputs(data: dict) -> dict[str, str]:
    """Every committed text output, keyed by repository-relative path."""
    return {
        DATA_JSON: json.dumps(data, indent=1, sort_keys=True) + "\n",
        "paper/generated/compression-macros.tex": tex_macros(data),
        "paper/generated/compression-table.tex": tex_table(data),
        "paper/generated/compression-table.md": markdown_table(data),
        "paper/generated/readme-compression.md": readme_block(data) + "\n",
    }


def readme_failures(readme: str, data: dict) -> list[str]:
    """The history page must carry the generated block exactly once, byte for byte."""
    if readme.count(README_BEGIN) != 1 or readme.count(README_END) != 1:
        return [f"{HISTORY}: compression block markers must appear exactly once"]
    start, end = readme.index(README_BEGIN), readme.index(README_END) + len(README_END)
    if readme[start:end] != readme_block(data):
        return [f"{HISTORY}: generated compression block drift"]
    return []


def strip_readme_block(readme: str) -> str:
    if README_BEGIN in readme and README_END in readme:
        start, end = readme.index(README_BEGIN), readme.index(README_END) + len(README_END)
        return readme[:start] + readme[end:]
    return readme
