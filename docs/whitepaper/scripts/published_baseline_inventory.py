"""Paper coverage and descriptive figure inputs from pinned merged publications.

No private files, sample rows, checkpoints, forecasts, or method ranking are read.
Each displayed value retains a repository receipt path and JSON pointer.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

REPO = Path(__file__).resolve().parents[3]
SOURCES = {
    "tabsyn": ("tabsyn-eight-lineage-validation/panel.json", "3e49c1f8f99615f21abee3504afb645e6b61eaab2d88be2e279be0c0c1b91d76"),
    "tabddpm": ("tabddpm-twelve-lineage-retained-validation/panel.json", "877317666864c712b84feed7170f9dccd3587780ae3b5ebbb6cecf4513adda13"),
    "forest2": ("forest-first-two-fivefit-v1/panel.json", "b9dd57b0c1c95ff20105577fb10ba56f6964e6c5bb534900e84a0a35e1456c67"),
    "forest6": ("forest-next-six-fivefit-v1/panel.json", "376fc0b3c835e9eadde677d394eff5b56da91f2af3ea876e28f764ffbe1e1d08"),
    "forest5": ("forest-third-five-fivefit-v1/panel.json", "311af74fe4ba6635bf736b28c4436afe64f47ff017b2ae018c33bbf3ce69cd8b"),
    "classical": ("retained-classical-validation/panel.json", "f841daea8de49ef6a8f0d1d05159050981ebe80661ead64fc84ecbf9c0b29680"),
    "arf_prefix": ("cpu-fivefit-validation-v1/arf/panel.json", "d4a72b56358346346fc4643dfdd499aa06c92e803a71af3f8ecc8127e8b52ab9"),
    "arf_fits": ("arf-complete-additional-fits-v1/panel.json", "7acfd172db6b43537723aff63e47faed35cd1319fb371fef34ad4328ebe46cca"),
    "forest_legacy": ("s3-matched-forest-confirmation-validation.json", "97a25be902954cad16c5b5802d0e0ba468dea104221efa2465434c94e6774a1f"),
    "forest_pilot": ("pilot24-forestdiffusion-native.json", "8024a22fd5cf9b83894bfea06414465b3f0265fc9a316602ad05f7c3323145b2"),
}
METRICS = ("catboost_retention", "marginal_error_mean", "c2st_catboost_auc", "distance_mia_auc")


def source_ref(name, pointer):
    path, sha = SOURCES[name]
    return dict(path="research/benchmark/results/" + path, sha256=sha, pointer=pointer)


def load_sources(repo=REPO):
    docs = {}
    for name, (relative, sha) in SOURCES.items():
        path = repo / "research/benchmark/results" / relative
        if path.is_symlink():
            raise ValueError("symlinked publication")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != sha:
            raise ValueError("published baseline source drift: " + name)
        docs[name] = json.loads(raw)
        if docs[name].get("official_tests_opened") is not False:
            raise ValueError("publication test seal missing")
        for key in ("mfs_v2", "ptf_v1", "superiority", "release_safe_l3", "release_safe"):
            if docs[name].get(key) is not None:
                raise ValueError("publication gated claim is non-null")
    return docs


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def scalar(cell, metric):
    metrics = cell["metrics"]
    if metric == "catboost_retention":
        return metrics["utility"]["auditors"]["catboost"]["retention"]
    if metric == "marginal_error_mean":
        return mean(row["value"] for row in metrics["fidelity"]["marginals"])
    if metric == "c2st_catboost_auc":
        return metrics["detection"]["catboost"]["value"]
    return metrics["privacy"]["distance_mia"]["value"]


def build(docs):
    ts, td, arf = docs["tabsyn"], docs["tabddpm"], docs["arf_fits"]
    forests = [docs[name] for name in ("forest2", "forest6", "forest5")]
    forest_ids = [{c["dataset"] for c in p["cells"]} for p in forests]
    if sum(map(len, forest_ids)) != len(set.union(*forest_ids)):
        raise ValueError("overlapping published Forest cohorts")
    if any(p["fit_seeds"] != [11, 23, 37, 53, 71] or not p["five_fit_default_cohort_complete"] for p in forests):
        raise ValueError("incomplete published Forest cohort")
    rows = [
        dict(method="TabSyn", complete_retained_lineages=ts["lineages"], complete_retained_sample_cells=ts["sample_cells"],
             coverage=f"{ts['lineages']} scaled-default lineages; {ts['sample_cells']} cells; fit seed 11. Native tuning and full population pending.",
             complete100=False, five_fit_lineages=0, sources=[source_ref("tabsyn", "")]),
        dict(method="TabDDPM", complete_retained_lineages=td["lineages"], complete_retained_sample_cells=td["physical_metric_cells"],
             logical_sample_cells=td["logical_metric_cells"], retained_models=td["retained_models"],
             coverage=f"{td['lineages']} retained lineages; {td['physical_metric_cells']} physical/{td['logical_metric_cells']} logical cells; {td['retained_models']} models, fit seed 11. Default/native cohorts differ; native search incomplete.",
             complete100=False, five_fit_lineages=0, sources=[source_ref("tabddpm", "")]),
        dict(method="Forest-Flow", complete_retained_lineages=sum(map(len, forest_ids)),
             complete_retained_sample_cells=sum(p["common_sample_cells"] for p in forests),
             retained_models=sum(p["generator_fits"] for p in forests), five_fit_lineages=sum(map(len, forest_ids)),
             coverage=f"{sum(map(len, forest_ids))} default lineages at five fit seeds; {sum(p['generator_fits'] for p in forests)} models/{sum(p['common_sample_cells'] for p in forests)} cells. Separate legacy six-lineage default/native panel; full population pending.",
             complete100=False, sources=[source_ref(name, "") for name in ("forest2", "forest6", "forest5", "forest_legacy")]),
        dict(method="Forest-Diffusion", complete_retained_lineages=None, complete_retained_sample_cells=None,
             pilot_logical_cells=len(docs["forest_pilot"]["cells"]), five_fit_lineages=0,
             coverage="Pilot only; separate 100-lineage matched diffusion panel pending.", complete100=False,
             sources=[source_ref("forest_pilot", "")]),
        dict(method="ARF", complete_retained_lineages=len({c["dataset"] for c in docs["classical"]["rows"] if c["method"]=="ARF"}),
             complete_retained_sample_cells=len([c for c in docs["classical"]["rows"] if c["method"] == "ARF"]),
             published_additional_metric_cells=docs["arf_prefix"]["sample_cells"],
             published_additional_physical_fits=arf["physical_fit_counts"]["ARF"], five_fit_lineages=0,
             coverage=f"{len({c['dataset'] for c in docs['classical']['rows'] if c['method']=='ARF'})} default/native lineages at fit seed 11; {len([c for c in docs['classical']['rows'] if c['method']=='ARF'])} logical cells. {arf['physical_fit_counts']['ARF']} additional native-scored fits; {docs['arf_prefix']['sample_cells']} additional common cells published. Remaining publication pending.",
             complete100=True, sources=[source_ref(name, "") for name in ("classical", "arf_prefix", "arf_fits")]),
    ]
    for method in ("GaussianCopula", "Chow-Liu"):
        cells = [c for c in docs["classical"]["rows"] if c["method"] == method]
        lineages = len({c["dataset"] for c in cells})
        rows.append(dict(method=method, complete_retained_lineages=lineages,
            complete_retained_sample_cells=len(cells), five_fit_lineages=0,
            coverage=f"{lineages} default/native lineages at fit seed 11; {len(cells)} logical cells. Five-fit common coverage pending.",
            complete100=lineages == 100, sources=[source_ref("classical", "")]))
    for row in rows:
        row.update(final_five_fit_complete=False, full100_eta_mt=None, eta_is_observed_completion=False,
                   remaining_campaign_status="pending", production_certified=False)
    # Descriptive cohort points only: identical sample schedules are required,
    # and configurations/cohorts remain separate. No across-cohort inference.
    points = []
    inputs = [("tabsyn", "TabSyn scaled (8)", ts["cells"], "cells", None, "TabSyn"),
              ("tabddpm", "TabDDPM default (10)", td["rows"], "rows", "author_default", "TabDDPM"),
              ("tabddpm", "TabDDPM native (12)", td["rows"], "rows", "native_selected", "TabDDPM"),
              ("classical", "ARF default (100)", docs["classical"]["rows"], "rows", "author_default", "ARF"),
              ("classical", "ARF native (100)", docs["classical"]["rows"], "rows", "native_selected", "ARF"),
              ("arf_prefix", "ARF seed23 default (12)", docs["arf_prefix"]["cells"], "cells", "author_default", "ARF"),
              ("arf_prefix", "ARF seed23 native (12)", docs["arf_prefix"]["cells"], "cells", "native_selected", "ARF")]
    for method in ("GaussianCopula", "Chow-Liu"):
        for config, label in (("author_default", "default"), ("native_selected", "native")):
            inputs.append(("classical", method+" "+label+" (100)", docs["classical"]["rows"], "rows", config, method))
    for name, label, cells, collection, configuration, method in inputs:
        groups = defaultdict(list)
        for i, cell in enumerate(cells):
            if cell["row_multiplier"] != 4 or cell["method"] != method:
                continue
            if configuration is not None and cell["selection_binding"] != configuration:
                continue
            groups[cell["dataset"]].append((i, cell))
        for dataset, group in sorted(groups.items()):
            if sorted(c["sample_seed"] for _, c in group) != [101, 211, 307] or len({(c["fit_seed"], c["config_sha256"]) for _, c in group}) != 1:
                raise ValueError("incomplete descriptive sample group")
            for metric in METRICS:
                values = [scalar(c, metric) if "metrics" in c else c[metric] for _, c in group]
                points.append(dict(cohort=label, dataset=dataset, metric=metric,
                    value=sum(values)/3 if all(finite(v) for v in values) else None,
                    measured_fits=1, sample_seeds=[101, 211, 307], row_multiplier=4,
                    aggregation="mean_of_three_sample_seeds", support_status="measured" if all(finite(v) for v in values) else "unavailable",
                    sources=[source_ref(name, "/"+collection+"/"+str(i)) for i, _ in group]))
    for name in ("forest2", "forest6", "forest5"):
        for i, summary in enumerate(docs[name]["summary"]):
            if summary["row_multiplier"] != 4:
                continue
            for metric in METRICS:
                stat = summary["metrics"][metric]
                points.append(dict(cohort="Forest-Flow five fits (13)", dataset=summary["dataset"], metric=metric,
                    value=stat["mean"] if stat["measured_fits"] == 5 else None,
                    measured_fits=stat["measured_fits"], sample_seeds=[101,211,307], row_multiplier=4,
                    aggregation="mean_of_five_fit_means_after_three_sample_means", support_status="measured" if stat["measured_fits"]==5 else "unavailable",
                    sources=[source_ref(name, "/summary/"+str(i)+"/metrics/"+metric)]))
    return dict(format="paper-published-baseline-inventory-v1", rows=rows, figure_points=points,
        sources={name:source_ref(name, "") for name in SOURCES}, official_tests_opened=False,
        mfs_v2=None, ptf_v1=None, release_safe_l3=None, superiority=None,
        scope="Published validation evidence only; pending cells have no estimated value or completion date.",
        limitations=["Cohorts and fit counts differ; plot is descriptive, not a ranking or paired comparison.",
            "Forest fit SD is not a confidence interval; this plot shows lineage points and medians without CI bars.",
            "Native objectives select only within their own methods; native scores are never ranked across methods.",
            "Additional ARF metrics completed on scratch remain pending publication in this frozen merged-receipt inventory."])


def render_table(data):
    lines=[r"\begin{tabular}{@{}lp{8.7cm}l@{}}",r"\toprule",r"Method & Published measured coverage & Remaining campaign \\",r"\midrule"]
    for row in data["rows"]:
        lines.append(row["method"]+" & "+row["coverage"]+" & Pending "+r"\\")
    return "\n".join(lines+[r"\bottomrule",r"\end{tabular}",""])


def render_csv(data):
    output=io.StringIO()
    fields=["cohort","dataset","metric","value","measured_fits","row_multiplier","aggregation","support_status","source_pointers_json"]
    writer=csv.DictWriter(output,fieldnames=fields,lineterminator="\n");writer.writeheader()
    for point in data["figure_points"]:
        row={k:point[k] for k in fields if k != "source_pointers_json"}
        row["source_pointers_json"]=json.dumps(point["sources"],sort_keys=True,separators=(",",":"))
        writer.writerow(row)
    return output.getvalue()


def render_figure(data, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"pdf.fonttype":42,"ps.fonttype":42,"font.family":"DejaVu Serif","font.size":8.1})
    labels=list(dict.fromkeys(p["cohort"] for p in data["figure_points"]))
    fig,axes=plt.subplots(1,4,figsize=(7.16,4.0),sharey=True)
    for axis,metric,title in zip(axes,METRICS,("CB retention","KS/TV error","C2ST CB AUC","MIA AUC")):
        for y,label in enumerate(labels):
            values=[p["value"] for p in data["figure_points"] if p["cohort"]==label and p["metric"]==metric and finite(p["value"])]
            axis.scatter(values,[y]*len(values),s=5,color="#888888",alpha=.35,linewidths=0)
            if values:axis.scatter([median(values)],[y],s=24,marker="D",color="#222222",linewidths=.4)
        axis.set_title(title);axis.set_yticks(range(len(labels)),labels);axis.grid(axis="x",alpha=.2)
        if metric == "catboost_retention":
            # Preserve large negative ratios without hiding the central range.
            axis.set_xscale("symlog", linthresh=1)
            axis.set_xticks([-100, 0, 1], ["-100", "0", "1"])
        axis.tick_params(axis="x", labelsize=7)
        axis.tick_params(axis="y", labelsize=7)
    axes[0].invert_yaxis()
    fig.supxlabel("Gray: lineage means. Diamonds: medians. Retention: symlog, linear in [-1,1].\n"
                  "Different cohorts; descriptive only. Unavailable values remain null in the CSV.", fontsize=6.8)
    fig.subplots_adjust(left=.29, right=.995, bottom=.19, top=.92, wspace=.18)
    fig.savefig(path,metadata={"CreationDate":None,"ModDate":None});plt.close(fig)


def emit(repo=REPO):
    data=build(load_sources(repo));out=repo/"docs/whitepaper/generated"
    coverage = {k:v for k,v in data.items() if k != "figure_points"}
    for name,value in (("baseline-coverage.json",coverage),("published-baseline-kpis.json",data)):
        (out/name).write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n")
    (out/"baseline-coverage.tex").write_text(render_table(data))
    (out/"published-baseline-kpis.csv").write_text(render_csv(data))
    figure=repo/"docs/whitepaper/figures/published-baseline-cohorts.pdf";render_figure(data,figure)
    (out/"published-baseline-figure-hashes.json").write_text(json.dumps(dict(pdf_sha256={figure.name:hashlib.sha256(figure.read_bytes()).hexdigest()}),sort_keys=True,indent=2)+"\n")
    return data


if __name__ == "__main__":
    emit()
