"""Manuscript reductions of merged validation metadata; no rows or models read."""
from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from scipy import stats

from compute_panel import DRAWS, SEED, holm, paired_test, sig3, summarize, tex_p

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "docs/whitepaper/generated"
SOURCES = {
    "classical": ("research/benchmark/results/retained-classical-validation/panel.json",
                  "f841daea8de49ef6a8f0d1d05159050981ebe80661ead64fc84ecbf9c0b29680"),
    "tabsyn": ("research/benchmark/results/tabsyn-eight-lineage-validation/panel.json",
               "3e49c1f8f99615f21abee3504afb645e6b61eaab2d88be2e279be0c0c1b91d76"),
    "first12": ("research/benchmark/results/common-A952-validation-v1/first12.completed.json",
                "8e039b1d57298f10bf7c05c2f3a8b14481a60ed663c1063ed30c273854f3bdfe"),
    "arf": ("research/benchmark/results/arf-matched-population-validation.json",
            "c69ce66e79b234bf5d1f9d56938450ba253ae39a3f6655a6fd586f890b8992f7"),
    "forest": ("research/benchmark/results/s3-matched-forest-confirmation-validation.json",
               "97a25be902954cad16c5b5802d0e0ba468dea104221efa2465434c94e6774a1f"),
    "forest_pilot": ("research/benchmark/results/pilot24-forestdiffusion-native.json",
                     "8024a22fd5cf9b83894bfea06414465b3f0265fc9a316602ad05f7c3323145b2"),
}
FIELDS = ("catboost_retention", "linear_retention", "mlp_retention",
          "marginal_error_mean", "pairwise_pearson_difference_mean", "contingency_tv_mean",
          "alpha_precision", "beta_recall", "c2st_catboost_auc", "c2st_logistic_auc",
          "dcr_fit_median", "dcr_validation_median", "nndr_fit_median",
          "distance_mia_auc", "domias_kde_auc")
PAIR_FIELDS = ("catboost_retention", "marginal_error_mean", "c2st_catboost_auc", "distance_mia_auc")
LABELS = {"ARF": "ARF", "Chow-Liu": "Chow--Liu", "GaussianCopula": "Copula", "TabSyn": "TabSyn"}


def load_sources(repo=REPO):
    docs = {}
    for name, (relative, digest) in SOURCES.items():
        buf = (repo / relative).read_bytes()
        if hashlib.sha256(buf).hexdigest() != digest:
            raise ValueError("committed validation source drift: " + name)
        docs[name] = json.loads(buf)
        if docs[name].get("official_tests_opened") is not False:
            raise ValueError("test seal is absent: " + name)
        for field in ("mfs_v2", "ptf_v1", "superiority"):
            if docs[name].get(field) is not None:
                raise ValueError("gated score is not null: " + name)
    return docs


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def groups_from_rows(rows):
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        if row["fit_seed"] != 11:
            raise ValueError("unexpected fit seed")
        key = (row["method"], row["selection_binding"], row["row_multiplier"], row["dataset"])
        groups[key].append((index, row))
    result = []
    for (method, config, size, dataset), group in sorted(groups.items()):
        if sorted(row["sample_seed"] for _, row in group) != [101, 211, 307]:
            raise ValueError("incomplete or duplicate sample group")
        group.sort(key=lambda item: item[1]["sample_seed"])
        values = {}
        for field in FIELDS:
            series = [row[field] for _, row in group]
            values[field] = sum(series) / 3 if all(finite(x) for x in series) else None
        result.append({"method": method, "configuration": config, "size": size,
                       "dataset": dataset, "values": values,
                       "source_pointers": ["/rows/" + str(i) for i, _ in group]})
    return result


def generic_pair(left, right, family):
    pair = paired_test(left, right, None, family)
    for prefix, replace in (("dope_", "left_"), ("other_", "right_")):
        for name in list(pair):
            if name.startswith(prefix):
                pair[replace + name[len(prefix):]] = pair.pop(name)
    # Directional sign tests and W/T/L are not used to rank privacy diagnostics.
    for name in ("sign_p", "wins", "ties", "losses"):
        pair.pop(name)
    pair["bootstrap_tail_adjusted_lo"] = pair.pop("familywise_lo")
    pair["bootstrap_tail_adjusted_hi"] = pair.pop("familywise_hi")
    pair["simultaneous_median_interval"] = median_interval(pair["differences"], family)
    return pair


def median_interval(differences, family):
    """Conservative order-statistic median interval, Bonferroni across a family.

    Requires independent identically distributed lineage differences. Small
    cohorts can require an unbounded interval; bootstrap extrema cannot fix it.
    """
    ordered = sorted(differences)
    n = len(ordered)
    tail = .05 / (2 * family)
    indices = [k for k in range(1, (n + 1) // 2 + 1) if stats.binom.cdf(k - 1, n, .5) <= tail]
    k = max(indices, default=0)
    return {"family_size": family, "nominal_simultaneous_confidence": .95,
            "order_index": k, "unbounded": k == 0,
            "lo": ordered[k - 1] if k else None,
            "hi": ordered[n - k] if k else None,
            "assumption": "independent identically distributed lineage differences; validation selection and fixed fits condition the estimand"}


def build(docs):
    panel = docs["classical"]
    if len(panel["rows"]) != 3648 or panel["complete_classical_lineages"] != 100:
        raise ValueError("unexpected complete cohort")
    groups = groups_from_rows(panel["rows"])
    eight = {x["dataset"] for x in groups if x["method"] == "TabSyn"}
    if len(eight) != 8:
        raise ValueError("unexpected TabSyn cohort")
    summaries = []
    for scope in ("classical100", "matched8"):
        chosen = [x for x in groups if (x["method"] != "TabSyn" if scope == "classical100" else x["dataset"] in eight)]
        keys = sorted({(x["method"], x["configuration"], x["size"]) for x in chosen})
        for method, config, size in keys:
            group = [x for x in chosen if (x["method"], x["configuration"], x["size"]) == (method, config, size)]
            summaries.append({"scope": scope, "method": method, "configuration": config, "size": size,
                              "metrics": {field: summarize(x["values"][field] for x in group if finite(x["values"][field])) for field in FIELDS}})
    pairs = []
    comparators = sorted({(x["method"], x["configuration"]) for x in groups if x["method"] != "TabSyn"})
    for metric in PAIR_FIELDS:
        left = {x["dataset"]: x["values"][metric] for x in groups if x["method"] == "TabSyn" and x["size"] == 4 and finite(x["values"][metric])}
        for method, config in comparators:
            right = {x["dataset"]: x["values"][metric] for x in groups if x["method"] == method and x["configuration"] == config and x["size"] == 4 and x["dataset"] in eight and finite(x["values"][metric])}
            pairs.append({"metric": metric, "left": "TabSyn/scaled_author_default", "right": method + "/" + config,
                          **generic_pair(left, right, len(PAIR_FIELDS) * len(comparators))})
    for pair, adjusted in zip(pairs, holm([x["wilcoxon_p"] for x in pairs])):
        pair["holm_p"] = adjusted
    arf_pairs = []
    lineage = docs["arf"]["lineage_groups"]
    for auditor in ("catboost", "linear", "mlp"):
        for config in ("author_default", "native_selected"):
            def mapping(method, configuration):
                return {row["dataset"]: row["utility"][auditor]["median_retention"] for row in lineage
                        if row["method"] == method and row["configuration"] == configuration and row["size_multiplier"] == 4
                        and row["utility"][auditor]["complete_informative"] and finite(row["utility"][auditor]["median_retention"])}
            arf_pairs.append({"auditor": auditor, "configuration": config,
                              **generic_pair(mapping("DOPE", "features12_steps2048"), mapping("ARF", config), 6)})
    for pair, adjusted in zip(arf_pairs, holm([x["wilcoxon_p"] for x in arf_pairs])):
        pair["holm_p"] = adjusted
    return {"format": "paper-retained-validation-evidence-v1", "bootstrap_seed": SEED, "bootstrap_draws": DRAWS,
            "sources": {k: {"path": p, "sha256": s} for k, (p, s) in SOURCES.items()},
            "groups": groups, "summaries": summaries, "matched8_paired": pairs, "arf_paired": arf_pairs,
            "official_tests_opened": False, "mfs_v2": None, "ptf_v1": None, "superiority": None,
            "procedure": {"sample_aggregation": "mean of all three sample seeds before lineage resampling; no seed-level pseudoreplication",
                          "intervals": "10000 percentile lineage-bootstrap resamples, approximate95% marginal; tail-adjusted paired bootstrap ranges are exploratory, not guaranteed familywise intervals. Conservative binomial/order-statistic median intervals use Bonferroni across each stated family; n8/family24 is unbounded",
                          "matched8_family": "exploratory 24 two-sided Wilcoxon tests: four metrics times six classical configurations, 4n, Holm correction across all24",
                          "arf_family": "six exploratory 4n retention tests: default/native times three auditors, Holm across six; legacy median-of-three sample aggregation retained separately",
                          "first12": "deterministic canonical-job batch, incomplete sample groups; per-cell values only; no bootstrap or method ranking"}}


def span(value):
    return "NA (0)" if not value["n"] else f"{sig3(value['median'])} [{sig3(value['lo'])}, {sig3(value['hi'])}] ({value['n']})"


def interval_text(row):
    value = row["simultaneous_median_interval"]
    return "Unbounded" if value["unbounded"] else f"[{sig3(value['lo'])}, {sig3(value['hi'])}]"


def render_tables(data, first12):
    outputs = {}
    for scope in ("classical100", "matched8"):
        lines = [r"\begin{tabular}{@{}llcccc@{}}", r"\toprule",
                 r"Method & Config. & CB retention & KS/TV & C2ST CB AUC & Distance MIA AUC \\",
                 r"\midrule"]
        for row in data["summaries"]:
            if row["scope"] == scope and row["size"] == 4:
                config = {"author_default": "Default", "native_selected": "Native", "scaled_author_default": "Scaled default"}[row["configuration"]]
                values = " & ".join(span(row["metrics"][x]) for x in PAIR_FIELDS)
                lines.append(f"{LABELS[row['method']]} & {config} & {values} " + r"\\")
        lines += [r"\bottomrule", r"\end{tabular}", ""]
        outputs["retained-" + scope + ".tex"] = "\n".join(lines)
    lines = [r"\begin{tabular}{@{}lcrrr@{}}", r"\toprule",
             r"TabSyn minus & $n$ & Median CB difference & Simultaneous median CI & Holm $p$ \\", r"\midrule"]
    for row in data["matched8_paired"]:
        if row["metric"] == "catboost_retention":
            label = row["right"].replace("GaussianCopula", "Copula").replace("Chow-Liu", "Chow--Liu").replace("author_default", "default").replace("native_selected", "native")
            lines.append(f"{label} & {row['n']} & {sig3(row['median_difference'])} & {interval_text(row)} & ${tex_p(row['holm_p'])}$ " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    outputs["retained-paired.tex"] = "\n".join(lines)
    lines = [r"\begin{tabular}{@{}llcrrr@{}}", r"\toprule",
             r"ARF config. & Auditor & $n$ & DOPE minus ARF & Simultaneous median CI & Holm $p$ \\", r"\midrule"]
    for row in data["arf_paired"]:
        config = "Default" if row["configuration"] == "author_default" else "Native"
        lines.append(f"{config} & {row['auditor']} & {row['n']} & {sig3(row['median_difference'])} & {interval_text(row)} & ${tex_p(row['holm_p'])}$ " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    outputs["retained-arf-paired.tex"] = "\n".join(lines)
    lines = [r"\begin{tabular}{@{}llrrrr@{}}", r"\toprule",
             r"Job prefix & Method & Pearson diff. & Alpha & Beta & Distance MIA AUC \\", r"\midrule"]
    for cell in first12["cells"]:
        diag = cell["diagnostics"]
        method = "/".join(sorted({x["method"] for x in cell["logical_aliases"]}))
        values = [diag["fidelity"]["pairwise_pearson_difference"]["value"]["mean"], diag["alpha_beta"]["value"]["alpha_precision"], diag["alpha_beta"]["value"]["beta_recall"], diag["privacy"]["distance_mia"]["value"]]
        lines.append(cell["physical_job_sha256"][:12] + " & " + method.replace("_", r"\_") + " & " + " & ".join(sig3(x) for x in values) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    outputs["retained-first12.tex"] = "\n".join(lines)
    return outputs


def main():
    docs = load_sources()
    data = build(docs)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "retained-evidence.json").write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    for name, text in render_tables(data, docs["first12"]).items():
        (OUT / name).write_text(text)
    coverage(docs)
    render_figure(data)
    print("Retained validation tables, lineage CIs and paired statistics regenerated")


def coverage(docs):
    path = REPO / "docs/whitepaper/coverage-plan.json"
    plan = json.loads(path.read_bytes())
    hours = (plan["default_fits"] + plan["native_trial_ceiling"]) * plan["per_fit_seconds"] / 3600 + plan["sampling_and_metrics_allowance_hours"]
    forecast = datetime.fromisoformat(plan["conditional_start_mt"]) + timedelta(hours=hours)
    forest_n = len({x["dataset"] for x in docs["forest"]["summary"] if x["method"] == "ForestDiffusion/Forest-Flow"
                    and x["configuration"] == "native_selected" and x["size_multiplier"] == 4
                    and x["utility"]["catboost"].get("complete_informative_sample_group")})
    rows = [
        {"method": "TabSyn", "complete_retained_lineages": 8, "complete_retained_sample_cells": 48,
         "coverage": "Scaled author-default subset, n/4n; full100/native tuning incomplete", "complete100": False},
        {"method": "TabDDPM", "complete_retained_lineages": None, "complete_retained_sample_cells": None,
         "coverage": "Contract/pilot receipts; no complete100 matched population publication", "complete100": False},
        {"method": "Forest-Flow", "complete_retained_lineages": forest_n, "complete_retained_sample_cells": None,
         "coverage": "Native-selected4n confirmation; full100 incomplete", "complete100": False},
        {"method": "Forest-Diffusion", "complete_retained_lineages": None, "complete_retained_sample_cells": None, "pilot_logical_cells": len(docs["forest_pilot"]["cells"]),
         "coverage": "Pilot includes author mode variants; no separate complete100 diffusion panel", "complete100": False},
        {"method": "ARF", "complete_retained_lineages": 100, "complete_retained_sample_cells": 1200,
         "coverage": "Default/native, n/4n, three sample seeds; fit11 complete", "complete100": True},
    ]
    for row in rows:
        observed = datetime.fromisoformat(docs["classical"]["execution_records"]["arf"]["recorded_UTC"]).astimezone(timezone(timedelta(hours=-6)))
        row["full100_eta_mt"] = observed.isoformat() if row["complete100"] else forecast.isoformat()
        if row["complete100"]:
            row["completion_observation_source"] = {"source": "classical", "pointer": "/execution_records/arf/recorded_UTC"}
        row["eta_is_observed_completion"] = row["complete100"]
        row["final_five_fit_complete"] = False
    record = {"plan": plan, "plan_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "calculated_slot_hours": hours, "rows": rows, "sources": {k: {"path": p, "sha256": h} for k, (p, h) in SOURCES.items()}}
    (OUT / "baseline-coverage.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    lines = [r"\begin{tabular}{@{}lp{8.3cm}l@{}}", r"\toprule", r"Method & Measured coverage & Full-population ETA (MDT) \\", r"\midrule"]
    for row in rows:
        lineages = str(row["complete_retained_lineages"]) if row["complete_retained_lineages"] is not None else "Partial"
        desc = row["coverage"].replace("full100", "full population").replace("complete100", "full population").replace("fit11", "fit seed 11").replace("selected4n", "selected $4n$")
        timestamp = datetime.fromisoformat(row["full100_eta_mt"])
        eta = ("Complete, " if row["complete100"] else "") + timestamp.strftime("%b. %d, %H:%M") + ("" if row["complete100"] else "*")
        lines.append(row["method"] + " & " + lineages + ": " + desc + " & " + eta + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    (OUT / "baseline-coverage.tex").write_text("\n".join(lines))


def render_figure(data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # The serif font ships with the pinned plotting dependency on every CI guest.
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42,
                         "font.family": "DejaVu Serif", "font.size": 8.1})

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.15))
    entries = [x for x in data["summaries"] if x["scope"] == "matched8" and x["size"] == 4]
    for axis, metric, label in zip(axes, ("catboost_retention", "distance_mia_auc"),
                                   ("Null-normalized retention", "Distance MIA AUC (empirical)")):
        for y, row in enumerate(entries):
            v = row["metrics"][metric]
            axis.plot([v["lo"], v["hi"]], [y, y], color="black", linewidth=1)
            axis.plot(v["median"], y, marker="o", color="black", markersize=3)
        axis.set_yticks(range(len(entries)), [LABELS[x["method"]].replace("--", "-") + " " +
                                            {"native_selected": "native", "author_default": "default", "scaled_author_default": "scaled"}[x["configuration"]] for x in entries])
        axis.invert_yaxis()
        axis.set_xlabel(label)
        axis.set_title("Matched eight, 4n; 95% lineage CI")
        axis.grid(axis="x", alpha=.2)
    fig.tight_layout()
    path = REPO / "docs/whitepaper/figures/retained-matched-eight.pdf"
    fig.savefig(path, metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)
    (OUT / "retained-figure-hashes.json").write_text(json.dumps({"pdf_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()}}, indent=2) + "\n")


if __name__ == "__main__":
    main()
