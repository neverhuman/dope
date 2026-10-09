"""Receipt-panel tables that print a configuration beside every method.

ARF has an author-default and a native-selected configuration in the same
panel. The TOST, RMSE, stored-fidelity and existing-seed tables print one row
per configuration, so each row names its configuration and no two rows share
an identity. Only rendering lives here; the committed panel is read unchanged.
"""

from __future__ import annotations

from research.benchmark.review_fixes.receipt_panel import (
    _measured_prefix,
    _metric_label,
    _metric_median,
    _table,
    ci,
    fmt,
    tex_name,
)
from research.benchmark.review_fixes.review_tex import assert_distinct

TOST_NOTE = (
    "% Wilcoxon and the TOST t-tests treat paired lineages as iid. "
    "The interval is the family-cluster bootstrap of the median. "
    "Holm families are separate for size n and size 4n "
    "(density 9, neural 6, ARF 6, Forest-Flow 3).\n"
)


def tost_table(payload: dict) -> str:
    rows = payload["paired"]
    assert_distinct(((r["size"], r["auditor"], r["method"], r["configuration"]) for r in rows), "TOST table")
    lines = []
    for row in rows:
        tost = row["tost"]
        lines.append(
            f"{row['size']}$n$ & {tex_name(row['auditor'])} & {tex_name(row['method'])} & "
            f"{tex_name(row['configuration'])} & {fmt(tost['mean'])} & "
            f"[{fmt(tost['ci90_lo'])}, {fmt(tost['ci90_hi'])}] & "
            f"{fmt(tost['p'], 3)} & {'yes' if tost['equivalent'] else 'no'} \\\\"
        )
    return TOST_NOTE + _table(
        "Size & Auditor & Comparator & Configuration & Mean diff. & 90\\% CI & TOST $p$ & Equivalent at $\\pm 0.02$",
        lines,
        "llllrrrl",
    )


def rmse_table(payload: dict) -> str:
    rows = [row for row in payload["rmse"] if row["auditor"] == "catboost"]
    assert_distinct(((r["method"], r["configuration"], r["size"]) for r in rows), "RMSE table")
    lines = [
        f"{tex_name(row['method'])} & {tex_name(row['configuration'])} & {row['size']}$n$ & "
        f"{row['n_regression']} & {fmt(row['synthetic_rmse_median'])} & {fmt(row['real_rmse_median'])} \\\\"
        for row in rows
    ]
    return _table(
        "Method & Configuration & Size & Regression lineages & Synthetic RMSE & Real TRTR RMSE",
        lines,
        "lllrrr",
    )


def fidelity_stored_table(payload: dict) -> str:
    rows = payload["fidelity_stored"]
    assert_distinct(
        ((r["method"], r["configuration"], r["size"], r["metric"]) for r in rows), "stored fidelity table")
    lines = [
        f"{tex_name(row['method'])} & {tex_name(row['configuration'])} & {row['size']}$n$ & "
        f"{tex_name(_metric_label(row))} & {row['n']} & {fmt(row['median'])} & {ci(row)} \\\\"
        for row in rows
    ]
    methods = ", ".join(sorted({str(row["method"]) for row in rows}))
    note = (
        "% Stored logistic C2ST, marginal KS, and pair correlation from the population ledgers. "
        "Each n counts lineages with three finite stored values. Cohorts are not matched, "
        "and this table is not a cross-method ranking. The stored C2ST is not the GBDT detector. "
        f"Methods in this file: {methods}. TabSyn is not in these ledgers.\n"
    )
    return note + _table(
        "Method & Configuration & Size & Metric & $n$ lineages & Median & Hierarchical CI",
        lines,
        "llllrrr",
    )


def seeds_existing_table(payload: dict) -> str:
    rows = payload["existing_multiseed"]
    assert_distinct(
        ((r["method"], r.get("selection_binding"), r["fit_seed"], r["row_multiplier"]) for r in rows),
        "existing-seed table",
    )
    lines = [
        f"{tex_name(row['method'])} & {tex_name(row.get('selection_binding'))} & {row['fit_seed']} & "
        f"{row['row_multiplier']}$n$ & {_measured_prefix(row)} & "
        f"{_metric_median(row.get('catboost_retention'))} & "
        f"{_metric_median(row.get('distance_mia_auc'))} & "
        f"{_metric_median(row.get('dcr_validation_median'))} \\\\"
        for row in rows
    ]
    return (
        "% Existing cpu-fivefit prefix. It is not pooled with the 97-lineage table. "
        "Measured/prefix counts finite CatBoost retentions over the prefix size. "
        "Configuration is the panel selection binding.\n"
    ) + _table(
        "Method & Configuration & Fit seed & Size & Measured/prefix & CatBoost retention & "
        "Distance MIA AUC & Holdout DCR median",
        lines,
        "lllllrrr",
    )


def configured_tables(payload: dict) -> dict[str, str]:
    """File name to TeX body for every table that carries a configuration column."""
    return {
        "review-tost.tex": tost_table(payload),
        "review-rmse.tex": rmse_table(payload),
        "review-fidelity-stored.tex": fidelity_stored_table(payload),
        "review-seeds-existing.tex": seeds_existing_table(payload),
    }
