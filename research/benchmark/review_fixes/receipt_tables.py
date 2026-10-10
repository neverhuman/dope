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
    ci,
    fmt,
    tex_name,
)
from research.benchmark.review_fixes.review_tex import assert_distinct, longtable


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
    return longtable(
        "\\textbf{v1 mean-difference equivalence, fit seed 11.} Mean DOPE-minus-comparator retention difference, "
        "its 90\\% $t$ interval, and the larger of the two one-sided $p$-values against $\\pm 0.02$. The test treats "
        "lineages as independent, and outliers move the mean; it is kept for continuity with the first version. "
        "Holm families as in the paired table; the equivalence $p$ is unadjusted.",
        "Size & Auditor & Comparator & Configuration & Mean diff. & 90\\% CI & TOST $p$ & Equivalent at $\\pm 0.02$",
        lines,
        "llllrrrl",
        label="tab:review-tost",
    )


def rmse_table(payload: dict) -> str:
    rows = [row for row in payload["rmse"] if row["auditor"] == "catboost"]
    assert_distinct(((r["method"], r["configuration"], r["size"]) for r in rows), "RMSE table")
    lines = [
        f"{tex_name(row['method'])} & {tex_name(row['configuration'])} & {row['size']}$n$ & "
        f"{row['n_regression']} & {fmt(row['synthetic_rmse_median'])} & {fmt(row['real_rmse_median'])} \\\\"
        for row in rows
    ]
    return longtable(
        "\\textbf{CatBoost auditor RMSE, fit seed 11.} Median over regression lineages of the validation RMSE of the "
        "auditor trained on synthetic rows and of the auditor trained on the real fit rows. RMSE is in each "
        "lineage's target units, so a median mixes scales and is no cross-method ranking. No Holm family.",
        "Method & Configuration & Size & Regression lineages & Synthetic RMSE & Real TRTR RMSE",
        lines,
        "lllrrr",
        label="tab:review-rmse",
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
    return longtable(
        "\\textbf{Stored fidelity from the population ledgers, fit seed 11.} Logistic two-sample AUC, mean marginal "
        "Kolmogorov--Smirnov distance, and pair-correlation fidelity; $n$ counts lineages with three finite stored "
        "values, with the family-cluster bootstrap interval. Cohorts differ across methods, so the rows are no "
        "ranking. The stored detector is logistic, not CatBoost, and TabSyn is absent from these ledgers. No Holm family.",
        "Method & Configuration & Size & Metric & $n$ lineages & Median & Hierarchical CI",
        lines,
        "llllrrr",
        label="tab:review-fidelity",
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
    return longtable(
        "\\textbf{Earlier five-fit prefix.} Medians on the lineage prefix of an earlier CPU five-fit run, never "
        "pooled with the full-panel tables. Measured/prefix counts finite CatBoost retentions over the prefix size, "
        "and the configuration is the panel's selection binding. No Holm family.",
        "Method & Configuration & Fit seed & Size & Measured/prefix & CatBoost retention & "
        "Distance MIA AUC & Holdout DCR median",
        lines,
        "lllllrrr",
        label="tab:review-prefix",
    )


def configured_tables(payload: dict) -> dict[str, str]:
    """File name to TeX body for every table that carries a configuration column."""
    return {
        "review-tost.tex": tost_table(payload),
        "review-rmse.tex": rmse_table(payload),
        "review-fidelity-stored.tex": fidelity_stored_table(payload),
        "review-seeds-existing.tex": seeds_existing_table(payload),
    }
