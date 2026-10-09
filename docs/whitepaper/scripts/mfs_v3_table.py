#!/usr/bin/env python3
"""Density-block table and figures for the three clean-license tabular auditors.

Digits come from the lineage retentions in the MFS-v3 receipt. Stored medians
are checked and then ignored. CatBoost, linear, and MLP figures are untouched.
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

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from research.benchmark.mfs_v3_panel import KUMO_ESTIMATORS, REFUSED_ENCODERS, VECTOR_PIN
from research.benchmark.representation import TABULAR_AUDITORS

import compute_panel
import figure_style

RECEIPT = REPO / "research" / "benchmark" / "results" / "mfs-v3-density-panel.json"
GENERATED = REPO / "docs" / "whitepaper" / "generated"
FIGURES = REPO / "docs" / "whitepaper" / "figures"
TABLE_NAME = "mfs-v3-density-table.tex"
SCALAR_NAME = "mfs-v3-scalar.tex"
HASH_NAME = "mfs-v3-figure-hashes.json"
BAR_NAME = "mfs-v3-retention-bars.pdf"
PAIRED_NAME = "mfs-v3-paired.pdf"
METHODS = ("DOPE", "GaussianCopula", "Chow-Liu", "independent_marginals")
COMPARATORS = ("GaussianCopula", "Chow-Liu", "independent_marginals")
AUDITOR_TEX = {
    "kumo_tabular_l": "Kumo large",
    "mitra_v2": "Mitra",
    "tabicl2": "TabICL",
}
COMPARATOR_TEX = {
    "GaussianCopula": "Gaussian copula",
    "Chow-Liu": "Chow--Liu",
    "independent_marginals": "Indep.\\ marginals",
}
BAR_STYLE = {
    "DOPE": ("DOPE\n(ours)", figure_style.DOPE, ""),
    "GaussianCopula": ("Gaussian\ncopula", figure_style.GAUSSIAN, "//"),
    "Chow-Liu": ("Chow-Liu", figure_style.CHOW, "\\\\"),
    "independent_marginals": ("Indep.", figure_style.INDEPENDENT, "xx"),
}
MARKS = (
    ("GaussianCopula", "Gaussian copula", figure_style.GAUSSIAN, "s"),
    ("Chow-Liu", "Chow-Liu", figure_style.CHOW, "^"),
    ("independent_marginals", "Independent", figure_style.INDEPENDENT, "D"),
)
X_LIM = (-1.5, 1.5)


def _finite(value) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def require_receipt(receipt: dict) -> int:
    """Refuse a win claim, a refused encoder, or an auditor outside the pin."""
    if receipt.get("format") != "dope-mfs-v3-density-panel":
        raise ValueError("receipt format is not the density panel")
    if receipt.get("counts_as_dope_win") is not False or receipt.get("official_tests_opened") is not False:
        raise ValueError("receipt claims a win or an opened official test")
    if receipt.get("superiority") is not None:
        raise ValueError("superiority is not null")
    expected = set(TABULAR_AUDITORS)
    refused = set(REFUSED_ENCODERS)
    scored = 0
    for row in receipt["lineages"]:
        if row.get("counts_as_dope_win") is not False or row.get("official_tests_opened") is not False:
            raise ValueError("a lineage claims a win or an opened official test")
        if row.get("superiority") is not None:
            raise ValueError("a lineage claims superiority")
        names = set((row.get("retention") or {}))
        if names & refused:
            raise ValueError("a refused encoder is in the receipt")
        if names != expected:
            raise ValueError("a lineage auditor set is not the pinned trio")
        if row.get("score") is not None:
            scored += 1
    if scored != receipt.get("scored_lineages"):
        raise ValueError("scored lineage count does not match the receipt")
    return scored


def lineage_maps(receipt: dict) -> dict:
    """Auditor, method, dataset to the finite lineage retention."""
    found = {name: {method: {} for method in METHODS} for name in TABULAR_AUDITORS}
    identities = set()
    cohorts = {method: set() for method in METHODS}
    for row in receipt["lineages"]:
        method = row.get("method")
        dataset = row.get("dataset")
        if method not in METHODS or not isinstance(dataset, str) or not dataset:
            raise ValueError("unknown method or invalid lineage identity")
        if (method, dataset) in identities:
            raise ValueError("duplicate method-lineage-auditor key")
        identities.add((method, dataset))
        cohorts[method].add(dataset)
        for name, value in (row.get("retention") or {}).items():
            if name not in found:
                raise ValueError("unknown auditor")
            if value is not None and not _finite(value):
                raise ValueError("invalid retention value")
            if _finite(value):
                found[name][method][dataset] = float(value)
    expected = cohorts["DOPE"]
    if not expected or any(cohort != expected for cohort in cohorts.values()):
        raise ValueError("incomplete matched method-lineage matrix")
    return found


def _agree(stored, fresh, label: str) -> None:
    if stored is None or fresh is None:
        return
    if abs(float(stored) - float(fresh)) > 1e-9:
        raise ValueError(f"stored summary disagrees for {label}")


def measure(receipt: dict) -> dict:
    """Recompute medians, intervals, and Holm from the lineage retentions."""
    scored = require_receipt(receipt)
    maps = lineage_maps(receipt)
    n_cells = len(receipt["lineages"])
    n_lineages = len({row["dataset"] for row in receipt["lineages"]})
    if n_cells != len(METHODS) * n_lineages:
        raise ValueError("method-lineage cell count disagrees with matched matrix")
    stored = receipt.get("retention_summary") or {}
    block = {}
    for auditor in TABULAR_AUDITORS:
        pairs = {}
        dope = maps[auditor]["DOPE"]
        for comparator in COMPARATORS:
            pairs[comparator] = compute_panel.paired_test(
                dope,
                maps[auditor][comparator],
                None,
                len(COMPARATORS),
            )
        compute_panel.attach_holm(pairs, "wilcoxon_p", "holm_p")
        methods = {
            method: compute_panel.summarize(values.values())
            for method, values in maps[auditor].items()
        }
        prior = stored.get(auditor) or {}
        prior_methods = prior.get("methods") or {}
        prior_pairs = prior.get("pairs") or {}
        for method, summary in methods.items():
            _agree((prior_methods.get(method) or {}).get("median"), summary["median"], f"{auditor} {method}")
        for comparator, pair in pairs.items():
            _agree(
                (prior_pairs.get(comparator) or {}).get("median_difference"),
                pair["median_difference"],
                f"{auditor} {comparator}",
            )
        block[auditor] = {"methods": methods, "pairs": pairs}
    cleartext = sum(
        "cleartext_absent" in (row.get("failed_gates") or [])
        for row in receipt["lineages"]
    )
    return {
        "block": block,
        "cleartext_unscanned": cleartext,
        "method_lineage_cells": len(receipt["lineages"]),
        "n_lineages": len({row["dataset"] for row in receipt["lineages"]}),
        "n_methods": len(METHODS),
        "family_size": len(COMPARATORS),
        "scored": scored,
    }


def _tt(name: str) -> str:
    return "\\texttt{" + "\\_\\allowbreak{}".join(name.split("_")) + "}"


def table_tex(measured: dict) -> str:
    rows = []
    for auditor in TABULAR_AUDITORS:
        for comparator in COMPARATORS:
            pair = measured["block"][auditor]["pairs"][comparator]
            rows.append(
                f"{AUDITOR_TEX[auditor]} & {COMPARATOR_TEX[comparator]} & {pair['n']} & "
                f"{compute_panel.paired_median_tex(pair['dope_median'], pair['lo'], pair['hi'], 'dope')} [{compute_panel.sig3(pair['dope_lo'])}, {compute_panel.sig3(pair['dope_hi'])}] & "
                f"{compute_panel.paired_median_tex(pair['other_median'], pair['lo'], pair['hi'], 'other')} [{compute_panel.sig3(pair['other_lo'])}, {compute_panel.sig3(pair['other_hi'])}] & "
                f"{compute_panel.sig3(pair['median_difference'])} [{compute_panel.sig3(pair['lo'])}, {compute_panel.sig3(pair['hi'])}] & "
                f"{pair['wins']}/{pair['ties']}/{pair['losses']} & "
                f"${compute_panel.tex_p(pair['holm_p'])}$ \\\\"
            )
    body = "\n".join(rows)
    return (
        "\\begin{tabular}{@{}llrrrrrr@{}}\n\\toprule\n"
        "Auditor & Comparator & $n$ & DOPE & Other & Difference & W/T/L & Holm $p$ \\\\\n"
        "\\midrule\n"
        f"{body}\n\\bottomrule\n\\end{{tabular}}\n"
        f"\\par\\smallskip{{\\scriptsize Holm family: {measured['family_size']} tests per auditor; "
        "DOPE versus Gaussian copula, Chow--Liu, and independent marginals.}\\par\n"
    )


def scalar_tex(measured: dict) -> str:
    rows = "\n".join(
        f"{AUDITOR_TEX[name]} & {_tt(VECTOR_PIN[name])} \\\\"
        for name in TABULAR_AUDITORS
    )
    return (
        f"Kumo large averages {KUMO_ESTIMATORS} inverse-transformed quantile forecasts. "
        f"The receipt scores {measured['scored']} of {measured['method_lineage_cells']} method--lineage cells "
        f"({measured['n_lineages']} lineages $\\times$ {measured['n_methods']} methods), so the scalar is null. "
        f"Cleartext is unscanned on {measured['cleartext_unscanned']} method--lineage cells.\n"
        "\\begin{center}\\scriptsize\n"
        "\\begin{tabular}{@{}l>{\\raggedright\\arraybackslash}p{2.15in}@{}}\n"
        f"{rows}\n"
        "\\end{tabular}\n"
        "\\end{center}\n"
    )


def _fonts():
    import matplotlib

    matplotlib.use("Agg")
    figure_style.apply(matplotlib, size=8.1)
    import matplotlib.pyplot as plt

    return plt


def _bars(measured: dict, dest: Path) -> None:
    plt = _fonts()
    import numpy as np
    from matplotlib.patches import Patch

    figure, axes = plt.subplots(1, 3, figsize=(7.16, 3.45), sharey=True)
    for axis, auditor in zip(axes, TABULAR_AUDITORS):
        pairs = measured["block"][auditor]["pairs"]
        anchor = pairs["GaussianCopula"]
        rows = [("DOPE", anchor["dope_median"], anchor["dope_lo"], anchor["dope_hi"])]
        for key in COMPARATORS:
            pair = pairs[key]
            if int(pair["n"]) != int(anchor["n"]):
                raise SystemExit(f"{auditor} {key} paired count does not match")
            rows.append((key, pair["other_median"], pair["other_lo"], pair["other_hi"]))
        positions = np.arange(len(rows))
        medians = [row[1] for row in rows]
        lowers = [row[1] - row[2] for row in rows]
        uppers = [row[3] - row[1] for row in rows]
        colors = [BAR_STYLE[row[0]][1] for row in rows]
        hatches = [BAR_STYLE[row[0]][2] for row in rows]
        bars = axis.bar(positions, medians, width=0.72, color=colors, edgecolor="#222222", linewidth=0.4, zorder=2)
        for bar, hatch in zip(bars, hatches):
            bar.set_hatch(hatch)
        axis.errorbar(positions, medians, yerr=[lowers, uppers], fmt="none", ecolor="#222222", elinewidth=0.7, capsize=2.0, capthick=0.6, zorder=3)
        axis.axhline(0, color="#333333", linewidth=0.6, zorder=1)
        for xpos, row in zip(positions, rows):
            label = compute_panel.sig3(row[1])
            if row[1] >= 0:
                axis.text(xpos, row[3] + 0.03, label, ha="center", va="bottom", color="#1a1a1a")
            else:
                axis.text(xpos, row[2] - 0.03, label, ha="center", va="top", color="#1a1a1a")
        axis.set_xticks(positions)
        axis.set_xticklabels([])
        axis.tick_params(axis="x", length=0)
        axis.set_title(f"{AUDITOR_TEX[auditor]}, {anchor['n']} lineages")
        figure_style.panel(axis, grid="y")
    axes[0].set_ylabel("retention (dimensionless)")
    axes[0].set_ylim(-0.28, 1.18)
    handles = [
        Patch(
            facecolor=BAR_STYLE[key][1],
            edgecolor="#222222",
            linewidth=0.4,
            hatch=BAR_STYLE[key][2],
            label=BAR_STYLE[key][0].replace("\n", " "),
        )
        for key in ("DOPE", "GaussianCopula", "Chow-Liu", "independent_marginals")
    ]
    labels = [handle.get_label() for handle in handles]
    figure.tight_layout(pad=0.4, rect=(0, 0.12, 1, 1))
    figure_style.legend_below(figure, handles, labels, ncol=4)
    figure.savefig(dest)
    plt.close(figure)


def _paired(measured: dict, dest: Path) -> None:
    plt = _fonts()
    import numpy as np

    figure, axes = plt.subplots(3, 3, figsize=(7.16, 6.45), sharey="row")
    for row, auditor in enumerate(TABULAR_AUDITORS):
        for column, (key, label, color, marker) in enumerate(MARKS):
            axis = axes[row, column]
            pair = measured["block"][auditor]["pairs"][key]
            values = np.sort(np.asarray(pair["differences"], dtype=float))
            axis.scatter(values, np.arange(1, values.size + 1), s=12, c=color, marker=marker, linewidths=0)
            axis.axvline(0, color="#333333", linewidth=0.6)
            axis.set_xlim(*X_LIM)
            axis.set_xticks([-1.0, 0.0, 1.0])
            outside = int(np.sum((values < X_LIM[0]) | (values > X_LIM[1])))
            lo, hi = pair["lo"], pair["hi"]
            figure_style.panel(axis, grid="x")
            axis.set_title(
                f"{AUDITOR_TEX[auditor]}, {label}\n"
                f"{compute_panel.sig3(pair['median_difference'])} "
                f"[{compute_panel.sig3(lo)}, {compute_panel.sig3(hi)}]"
                f" ({outside} outside)",
                fontsize=6.4,
            )
            if row == 2:
                axis.set_xlabel("retention difference (dimensionless)")
            if column == 0:
                axis.set_ylabel("sorted lineage index")
    figure.tight_layout(pad=0.35)
    figure.savefig(dest)
    plt.close(figure)


def draw_figures(measured: dict, bar_path: Path, paired_path: Path) -> None:
    bar_path.parent.mkdir(parents=True, exist_ok=True)
    _bars(measured, bar_path)
    _paired(measured, paired_path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hash_document(bar_path: Path, paired_path: Path) -> dict:
    return {
        "pdf_sha256": {
            bar_path.name: _sha256(bar_path),
            paired_path.name: _sha256(paired_path),
        }
    }


def write_outputs(measured: dict) -> None:
    GENERATED.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    (GENERATED / TABLE_NAME).write_text(table_tex(measured))
    (GENERATED / SCALAR_NAME).write_text(scalar_tex(measured))
    draw_figures(measured, FIGURES / BAR_NAME, FIGURES / PAIRED_NAME)
    payload = hash_document(FIGURES / BAR_NAME, FIGURES / PAIRED_NAME)
    (GENERATED / HASH_NAME).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def check_outputs(measured: dict) -> None:
    """Regenerate beside the committed files and require the same bytes."""
    expected_table = table_tex(measured)
    expected_scalar = scalar_tex(measured)
    table_path = GENERATED / TABLE_NAME
    scalar_path = GENERATED / SCALAR_NAME
    if table_path.read_text() != expected_table:
        raise SystemExit("mfs-v3 density table does not match the receipt")
    if scalar_path.read_text() != expected_scalar:
        raise SystemExit("mfs-v3 scalar note does not match the receipt")
    recorded = json.loads((GENERATED / HASH_NAME).read_text())["pdf_sha256"]
    with tempfile.TemporaryDirectory() as tmp:
        bar_path = Path(tmp) / BAR_NAME
        paired_path = Path(tmp) / PAIRED_NAME
        draw_figures(measured, bar_path, paired_path)
        fresh = hash_document(bar_path, paired_path)["pdf_sha256"]
    for name in (BAR_NAME, PAIRED_NAME):
        committed = _sha256(FIGURES / name)
        if fresh[name] != recorded[name] or committed != recorded[name]:
            raise SystemExit(f"{name} hash does not match the receipt")
    print("mfs-v3 table matches the receipt")


def load_receipt() -> dict:
    """Authenticate the density-panel digest before any table or figure write."""
    return compute_panel.load(RECEIPT.name)


def main() -> None:
    receipt = load_receipt()
    measured = measure(receipt)
    if "--check" in sys.argv:
        check_outputs(measured)
        return
    write_outputs(measured)
    print(f"wrote {TABLE_NAME} scored {measured['scored']} of {measured['method_lineage_cells']} method-lineage cells")


if __name__ == "__main__":
    main()
