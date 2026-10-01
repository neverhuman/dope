"""Publish rights-safe, validation-only DOPE versus native-tuned SDV pilot results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


SCRATCH = Path("/mnt/fast-scratch/dope-benchmark/pilot-24h")
DATASETS = ("Adult", "California", "News")
GROUPS = ("Adult", "regression", "News")
SIZES = (1, 4)


def sha(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def within(root: Path, relative: str, raw: str) -> Path:
    path = Path(raw).resolve()
    allowed = (root / relative).resolve()
    if (not path.is_relative_to(allowed) or "evaluator" in path.parts
            or path.name == "test.csv"):
        raise ValueError("result evidence leaves training-only scratch")
    return path


def within_any(root: Path, relatives: tuple[str, ...], raw: str) -> Path:
    for relative in relatives:
        try:
            return within(root, relative, raw)
        except ValueError:
            continue
    raise ValueError("result evidence leaves training-only scratch")


def check_metric(metric: dict) -> dict:
    if (metric["mfs_v2"] is not None or metric["gate_profile_complete"] is not False
            or metric["task"] not in ("binary", "regression")):
        raise ValueError("pilot metric has a gated score or invalid task")
    catboost = metric["utility"]["catboost"]
    value = catboost.get("retention")
    if value is not None and not math.isfinite(value):
        raise ValueError("nonfinite pilot retention")
    return {"catboost_retention": value,
            "catboost_informative": catboost.get("informative"),
            "catboost_trtr_loss": catboost.get("trtr_loss"),
            "catboost_tstr_loss": catboost.get("tstr_loss"),
            "null_loss": metric["null_loss"],
            "linear_retention": metric["utility"]["linear"].get("retention"),
            "mlp_retention": metric["utility"]["mlp"].get("retention"),
            "c2st_auc": metric["c2st_auc"],
            "exact_row_copies": metric["copy_counts"]["exact"],
            "near_row_copies": metric["copy_counts"]["near"],
            "real_vs_real_exact": metric["real_vs_real_control_counts"]["exact"]}


def selection_receipts(root: Path) -> list[dict]:
    directory = root / "sdv-selection-v1"
    lock_path = directory / "selection.lock.json"
    lock = read(lock_path)
    if (lock["validation_only"] is not True
            or lock["official_tests_opened"] is not False):
        raise ValueError("SDV selection is not training-derived validation")
    results = []
    for dataset in DATASETS:
        for method in ("CTGAN", "TVAE"):
            path = directory / f"{dataset}-{method}.json"
            selected = read(path)
            if (selected["dataset"] != dataset or selected["method"] != method
                    or selected["selection_lock_sha256"] != sha(lock_path)
                    or selected["official_tests_opened"] is not False
                    or selected["selected_trial"] is None
                    or len(selected["trials"]) > 8
                    or selected["total_tuning_wall_seconds"] > 43200):
                raise ValueError("native selection identity or budget changed")
            trials = selected["trials"]
            chosen = next(t for t in trials if t["job"]["trial"] == selected["selected_trial"])
            default = next(t for t in trials if t["job"]["trial"] == 0)
            if (chosen["status"] != "ok"
                    or chosen["native_kpi"] != selected["selected_native_kpi"]
                    or chosen["job"]["config"] != selected["selected_config"]):
                raise ValueError("native-selected trial changed")
            for trial in trials:
                receipt_path = within(root, trial["round"], trial["attempt_path"])
                if sha(receipt_path) != trial["attempt_sha256"]:
                    raise ValueError("native tuning trial receipt changed")
            results.append({"dataset": dataset, "method": method,
                            "objective": selected["objective"]["name"],
                            "direction": selected["objective"]["direction"],
                            "selected_trial": selected["selected_trial"],
                            "selected_native_value": chosen["native_kpi"]["value"],
                            "default_native_value": (default["native_kpi"]["value"]
                                                     if default["native_kpi"] else None),
                            "default_trial_status": default["status"],
                            "tuning_attempts": len(trials),
                            "tuning_wall_seconds": selected["total_tuning_wall_seconds"],
                            "selection_receipt_sha256": sha(path),
                            "selection_lock_sha256": sha(lock_path),
                            "default_config_sha256": digest(default["job"]["config"]),
                            "selected_config_sha256": digest(chosen["job"]["config"])})
    return results


def native_metric_rows(root: Path, selections: list[dict]) -> tuple[list[dict], list[dict]]:
    selection_map = {(row["dataset"], row["method"]): row for row in selections}
    rows = []
    failures = []
    for group in GROUPS:
        sample_lock_path = root / "sdv-final-samples-v1" / group / "round.lock.json"
        metric_lock_path = root / "sdv-final-metrics-v1" / group / "round.lock.json"
        readout_lock_path = root / "sdv-fixed-readout-v1" / group / "round.lock.json"
        sample_lock, metric_lock, readout_lock = map(read,
                                                    (sample_lock_path, metric_lock_path,
                                                     readout_lock_path))
        if (metric_lock["sample_round_sha256"] != sha(sample_lock_path)
                or sample_lock["readout_lock_sha256"] != sha(readout_lock_path)
                or metric_lock["official_tests_opened"] is not False):
            raise ValueError("fixed comparator sample or metric lock changed")
        for cell in readout_lock["cells"]:
            if cell["fit_status"] == "ok":
                continue
            failures.append({"dataset": cell["dataset"], "method": cell["method"],
                             "configuration": cell["track"],
                             "fit_seed": cell["fit_seed"],
                             "status": cell["fit_status"],
                             "fit_receipt_sha256": cell["fit_receipt_sha256"]})
        for job in metric_lock["jobs"]:
            path = metric_lock_path.parent / "cells" / f"{digest(job)}.json"
            receipt = read(path)
            sample_path = sample_lock_path.parent / "jobs" / digest(job) / "attempt-0001/receipt.json"
            sample = read(sample_path)
            fit_path = Path(job["fit_receipt_path"]).resolve()
            # Fit receipts may live in one of the three frozen fixed rounds.
            if not fit_path.is_relative_to((root / "sdv-final-seed23-v1").resolve()) \
                    and not fit_path.is_relative_to((root / "sdv-final-news-tvae-v1").resolve()):
                raise ValueError("native fixed fit path changed")
            fit = read(fit_path)
            selection = selection_map[job["dataset"], job["method"]]
            if (receipt["job"] != job or job not in sample_lock["jobs"]
                    or receipt["metric_lock_sha256"] != sha(metric_lock_path)
                    or receipt["sample_receipt_sha256"] != sha(sample_path)
                    or sample["round_sha256"] != sha(sample_lock_path)
                    or sample["job"] != job or sample["status"] != "ok"
                    or receipt["sample_sha256"] != sample["sample_sha256"]
                    or sha(within_any(root, ("sdv-final-samples-v1",
                                             "sdv-final-seed23-v1",
                                             "sdv-final-news-tvae-v1"),
                                      sample["sample_path"]))
                    != sample["sample_sha256"]
                    or sha(fit_path) != job["fit_receipt_sha256"]
                    or fit["job"]["dataset"] != job["dataset"]
                    or fit["job"]["method"] != job["method"]
                    or fit["job"]["track"] != job["track"]
                    or fit["artifact_bytes"] != job["artifact_bytes"]
                    or fit["job"]["native_selection_sha256"]
                    != selection["selection_receipt_sha256"]
                    or digest(fit["job"]["config"]) != job["config_sha256"]
                    or receipt["official_tests_opened"] is not False
                    or receipt["ptf_v1"] is not None or receipt["mfs_v2"] is not None):
                raise ValueError("native fixed validation lineage changed")
            expected_config = (selection["selected_config_sha256"]
                               if job["track"] in ("tuned", "default_and_tuned")
                               else selection["default_config_sha256"])
            if job["config_sha256"] != expected_config:
                raise ValueError("native comparator used wrong configuration")
            rows.append({"dataset": job["dataset"], "method": job["method"],
                         "configuration": job["track"], "fit_seed": job["fit_seed"],
                         "sample_seed": job["sample_seed"],
                         "size_multiplier": job["size_multiplier"],
                         "artifact_bytes": job["artifact_bytes"],
                         "within_l3_bytes": job["artifact_bytes"] <= 10240,
                         "metric_receipt_sha256": sha(path),
                         "sample_receipt_sha256": sha(sample_path),
                         "fit_receipt_sha256": sha(fit_path),
                         **check_metric(receipt["metrics"])})
    if len(rows) != 42 or len(failures) != 2:
        raise ValueError("fixed comparator metric or failure matrix incomplete")
    return rows, failures


def dope_q8_rows(root: Path) -> list[dict]:
    old_lock_path = root / "gpu-v3/gpu-validation-v3.lock.json"
    third_lock_path = root / "dope-q8-seed307-v1/round.lock.json"
    old_lock, third_lock = read(old_lock_path), read(third_lock_path)
    if (third_lock["prior_gpu_validation_lock_sha256"] != sha(old_lock_path)
            or third_lock["official_tests_opened"] is not False):
        raise ValueError("q8 third-seed source changed")
    rows = []
    fit_map = {(fit["dataset"], fit["fit_seed"]): fit for fit in old_lock["fits"]}
    if len(fit_map) != 6:
        raise ValueError("q8 fit set changed")
    for fit in fit_map.values():
        if (sha(within(root, "gpu-v3/fits", fit["receipt_path"]))
                != fit["receipt_sha256"]):
            raise ValueError("q8 fit receipt changed")
    for dataset in DATASETS:
        fit = fit_map[dataset, 23]
        for sample_seed in (101, 211):
            for size in SIZES:
                directory = (root / "gpu-v3/validation-samples" / dataset / "fit-23"
                             / f"sample-{sample_seed}-size-{size}")
                receipt_path = directory / "sample-receipt.json"
                receipt = read(receipt_path)
                metric_path = directory / "validation-metrics.json"
                if (receipt["dataset"] != dataset or receipt["fit_seed"] != 23
                        or receipt["sample_seed"] != sample_seed
                        or receipt["size_multiplier"] != size
                        or receipt["fit_receipt_sha256"] != fit["receipt_sha256"]
                        or receipt["metric_sha256"] != sha(metric_path)
                        or sha(directory / "sample.csv") != receipt["sample_sha256"]
                        or receipt["sample_sha256"] != receipt["deterministic_repeat_sha256"]
                        or receipt["official_tests_opened"] is not False
                        or receipt["ptf_v1"] is not None or receipt["mfs_v2"] is not None):
                    raise ValueError("q8 original pilot sample changed")
                rows.append({"dataset": dataset, "method": "DOPE",
                             "configuration": "q8_symbolic", "fit_seed": 23,
                             "sample_seed": sample_seed, "size_multiplier": size,
                             "artifact_bytes": receipt["artifact_bytes"],
                             "within_l3_bytes": receipt["artifact_bytes"] <= 10240,
                             "metric_receipt_sha256": sha(metric_path),
                             "sample_receipt_sha256": sha(receipt_path),
                             "fit_receipt_sha256": fit["receipt_sha256"],
                             **check_metric(read(metric_path))})
    for job in third_lock["jobs"]:
        if job["fit_seed"] != 23:
            continue
        path = third_lock_path.parent / "cells" / digest(job) / "receipt.json"
        receipt = read(path)
        if (receipt["job"] != job or receipt["round_sha256"] != sha(third_lock_path)
                or receipt["status"] != "ok"
                or receipt["sample_sha256"] != receipt["repeat_sha256"]
                or receipt["official_tests_opened"] is not False
                or receipt["ptf_v1"] is not None or receipt["mfs_v2"] is not None
                or sha(within(root, "dope-q8-seed307-v1", receipt["sample_path"]))
                != receipt["sample_sha256"]
                or job["fit_receipt_sha256"] != fit_map[job["dataset"], 23]["receipt_sha256"]):
            raise ValueError("q8 third sample seed changed")
        rows.append({"dataset": job["dataset"], "method": "DOPE",
                     "configuration": "q8_symbolic", "fit_seed": 23,
                     "sample_seed": 307, "size_multiplier": job["size_multiplier"],
                     "artifact_bytes": job["artifact_bytes"],
                     "within_l3_bytes": job["artifact_bytes"] <= 10240,
                     "metric_receipt_sha256": sha(path),
                     "sample_receipt_sha256": sha(path),
                     "fit_receipt_sha256": job["fit_receipt_sha256"],
                     **check_metric(receipt["metric"])})
    if len(rows) != 18:
        raise ValueError("q8 paired three-seed matrix incomplete")
    return rows


def dope_q10_news_rows(root: Path) -> list[dict]:
    lock_path = root / "dope-news-q10-v1/sample.lock.json"
    lock = read(lock_path)
    if (lock["l3_eligible"] is not False or lock["official_tests_opened"] is not False
            or len(lock["jobs"]) != 24):
        raise ValueError("News q10 over-cap sample lock changed")
    rows = []
    for job in lock["jobs"]:
        if job["fit_seed"] != 23 or job["size_multiplier"] not in SIZES:
            continue
        path = lock_path.parent / "samples" / digest(job) / "receipt.json"
        receipt = read(path)
        if (receipt["job"] != job or receipt["round_sha256"] != sha(lock_path)
                or receipt["status"] != "ok"
                or receipt["sample_sha256"] != receipt["repeat_sha256"]
                or sha(within(root, "dope-news-q10-v1", receipt["sample_path"]))
                != receipt["sample_sha256"]
                or sha(within(root, "dope-news-q10-v1/fits",
                              job["fit_receipt_path"])) != job["fit_receipt_sha256"]
                or job["artifact_bytes"] <= 10240
                or receipt["official_tests_opened"] is not False
                or receipt["ptf_v1"] is not None or receipt["mfs_v2"] is not None):
            raise ValueError("News q10 over-cap validation cell changed")
        rows.append({"dataset": "News", "method": "DOPE",
                     "configuration": "q10_overcap_quality", "fit_seed": 23,
                     "sample_seed": job["sample_seed"],
                     "size_multiplier": job["size_multiplier"],
                     "artifact_bytes": job["artifact_bytes"],
                     "within_l3_bytes": False,
                     "metric_receipt_sha256": sha(path),
                     "sample_receipt_sha256": sha(path),
                     "fit_receipt_sha256": job["fit_receipt_sha256"],
                     **check_metric(receipt["metric"])})
    if len(rows) != 6:
        raise ValueError("News q10 three-seed metric matrix incomplete")
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[row["dataset"], row["method"], row["configuration"],
               row["size_multiplier"]].append(row)
    result = []
    for (dataset, method, configuration, size), group in sorted(groups.items()):
        if len(group) != 3 or {row["sample_seed"] for row in group} != {101, 211, 307}:
            raise ValueError("three sample seeds missing from paired pilot summary")
        values = [row["catboost_retention"] for row in group]
        if any(value is None for value in values):
            raise ValueError("informative pilot comparison is missing")
        byte_counts = {row["artifact_bytes"] for row in group}
        if len(byte_counts) != 1:
            raise ValueError("fixed-fit artifact bytes changed between samples")
        result.append({"dataset": dataset, "method": method,
                       "configuration": configuration,
                       "fit_seed": 23, "size_multiplier": size,
                       "sample_seeds": [101, 211, 307],
                       "median_catboost_retention": statistics.median(values),
                       "min_catboost_retention": min(values),
                       "max_catboost_retention": max(values),
                       "artifact_bytes": byte_counts.pop(),
                       "within_l3_bytes": all(row["within_l3_bytes"] for row in group),
                       "exact_row_copies": sum(row["exact_row_copies"] for row in group)})
    if len(result) != 22:
        raise ValueError("pilot summary lacks a comparator or DOPE outcome")
    return result


def build(root: Path = SCRATCH) -> dict:
    selections = selection_receipts(root)
    native, failures = native_metric_rows(root, selections)
    rows = native + dope_q8_rows(root) + dope_q10_news_rows(root)
    dope_lock_path = root / "dope-conditional-v1/selection.lock.json"
    dope_selection = []
    for dataset in DATASETS:
        path = root / "dope-conditional-v1" / f"{dataset}-selection.json"
        item = read(path)
        if (item["selection_lock_sha256"] != sha(dope_lock_path)
                or len(item["trials"]) != 8
                or item["official_tests_opened"] is not False):
            raise ValueError("DOPE conditional selection changed")
        dope_selection.append({"dataset": dataset,
                               "selected_job": item["selected_job"],
                               "selected_diagnostic_mean_retention": item["selected_diagnostic_mean_retention"],
                               "selection_receipt_sha256": sha(path),
                               "fit_trials": len(item["trials"]),
                               "total_fit_seconds": item["total_fit_seconds"]})
    if len(rows) != 66:
        raise ValueError("neural comparison lacks a common metric receipt")
    return {"format": "dope-24h-native-neural-validation-comparison", "version": 1,
            "scope": "training_derived_validation_only", "official_tests_opened": False,
            "production_certified": False, "mfs_v2": None, "ptf_v1": None,
            "source_sha256": sha(Path(__file__)),
            "l3_byte_limit": 10240,
            "notes": ["The official tests were not opened. All utility values use training-derived validation.",
                      "CTGAN and TVAE configurations were selected by their own locked author-library ML efficacy, not by DOPE utility.",
                      "Native F1 and R2 values are only comparable within the same method and dataset.",
                      "The DOPE q8 pilot compiler candidate is symbolic; separate GPU neural and conditional research costs are reported outside this fixed-seed comparison.",
                      "News q10 is an over-cap quality result. Restricted projection packing is not production certification.",
                      "News retention is sensitive to a small real-model improvement over the null.",
                      "No MFS-v2 or PTF-v1 gate or paper superiority conclusion follows from validation outcomes."],
            "locks": {name: sha(root / name) for name in (
                "start-v2.json", "sdv-selection-v1/selection.lock.json",
                "sdv-final-metrics-v1/Adult/round.lock.json",
                "sdv-final-metrics-v1/regression/round.lock.json",
                "sdv-final-metrics-v1/News/round.lock.json",
                "gpu-v3/gpu-validation-v3.lock.json",
                "dope-q8-seed307-v1/round.lock.json",
                "dope-conditional-v1/selection.lock.json",
                "dope-news-q10-v1/sample.lock.json")},
            "counts": {"common_metric_cells": len(rows),
                       "fixed_fit_failures": len(failures),
                       "native_selection_cells": len(selections),
                       "dope_conditional_fit_trials": sum(x["fit_trials"] for x in dope_selection)},
            "native_validation_kpi": selections,
            "dope_validation_selection": dope_selection,
            "fixed_fit_failures": sorted(failures, key=lambda x: (x["dataset"], x["method"])),
            "cells": sorted(rows, key=lambda x: (x["dataset"], x["method"],
                                              x["configuration"], x["fit_seed"],
                                              x["sample_seed"], x["size_multiplier"])),
            "summary": summarize(rows)}


def render_table(report: dict) -> str:
    lines = ["# Adult / California / News native-neural validation comparison", "",
             "Training-derived validation only. Official tests remain sealed. MFS-v2 and "
             "production PTF-v1 are null. Medians use fixed fit seed 23, sample seeds "
             "101/211/307, and the same CatBoost auditor on real validation rows.", "",
             "| Dataset | Method | Configuration | n retention | 4n retention | Charged bytes | L3 bytes |",
             "| --- | --- | --- | ---: | ---: | ---: | --- |"]
    for dataset in DATASETS:
        keys = sorted({(item["method"], item["configuration"])
                       for item in report["summary"] if item["dataset"] == dataset})
        for method, configuration in keys:
            group = {item["size_multiplier"]: item for item in report["summary"]
                     if item["dataset"] == dataset and item["method"] == method
                     and item["configuration"] == configuration}
            lines.append(f"| {dataset} | {method} | {configuration} | "
                         f"{group[1]['median_catboost_retention']:.4f} | "
                         f"{group[4]['median_catboost_retention']:.4f} | "
                         f"{group[1]['artifact_bytes']:,} | "
                         f"{'yes' if group[1]['within_l3_bytes'] else 'no'} |")
    lines += ["", "A byte-eligible artifact is not production certified: generator, privacy, "
              "coverage, and release gates remain incomplete. News q10 is shown for "
              "unconstrained quality and exceeds the complete 10,240-byte L3 cap.",
              "", "## Native validation objectives", "",
              "Values are compared only within one method and dataset. Adult uses mean "
              "binary F1; California and News use regression R². A timeout has no KPI.", "",
              "| Dataset | Method | Native objective | Default | Selected | Tuning attempts |",
              "| --- | --- | --- | ---: | ---: | ---: |"]
    for item in report["native_validation_kpi"]:
        default = (f"{item['default_native_value']:.4f}" if item["default_native_value"] is not None
                   else item["default_trial_status"])
        lines.append(f"| {item['dataset']} | {item['method']} | {item['objective']} | "
                     f"{default} | {item['selected_native_value']:.4f} | "
                     f"{item['tuning_attempts']} |")
    lines += ["", "## Failed fixed fits", ""]
    for item in report["fixed_fit_failures"]:
        lines.append(f"- {item['dataset']} {item['method']} {item['configuration']}: "
                     f"{item['status']} (fit seed {item['fit_seed']}).")
    lines += ["", "All 66 common metric cells, 42 SDV fixed-seed metric receipts, "
              "native selections, 24 DOPE conditional tuning trials, artifact charges, "
              "copy counts, and immutable scratch hashes are in the JSON. The News "
              "real-model gain over the null is small, so its retention ratio is unstable. "
              "This is not the preregistered public-test paired analysis.", ""]
    return "\n".join(lines)


def render_csv(report: dict) -> str:
    output = io.StringIO()
    fields = ("dataset", "method", "configuration", "fit_seed", "size_multiplier",
              "median_catboost_retention", "min_catboost_retention",
              "max_catboost_retention", "artifact_bytes", "within_l3_bytes",
              "exact_row_copies")
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n",
                            extrasaction="ignore")
    writer.writeheader()
    writer.writerows(report["summary"])
    return output.getvalue()


def render_figure(report: dict, svg_path: Path, pdf_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["svg.hashsalt"] = "dope-pilot24-native-neural-v1"
    import matplotlib.pyplot as plt

    colors = {"DOPE": "#176e8c", "CTGAN": "#c0711b", "TVAE": "#7757a3"}
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharex=True)
    for ax, dataset in zip(axes, DATASETS):
        groups = sorted({(row["method"], row["configuration"])
                         for row in report["summary"] if row["dataset"] == dataset})
        for method, configuration in groups:
            points = {row["size_multiplier"]: row for row in report["summary"]
                      if row["dataset"] == dataset and row["method"] == method
                      and row["configuration"] == configuration}
            y = [points[size]["median_catboost_retention"] for size in SIZES]
            low = [points[size]["min_catboost_retention"] for size in SIZES]
            high = [points[size]["max_catboost_retention"] for size in SIZES]
            label = f"{method} {configuration.replace('_', ' ')}"
            ax.errorbar([0, 1], y,
                        yerr=[[v - lo for v, lo in zip(y, low)],
                              [hi - v for hi, v in zip(high, y)]],
                        marker="o", markersize=4, linewidth=1.5, capsize=2,
                        linestyle="-" if points[1]["within_l3_bytes"] else "--",
                        color=colors[method], label=label)
        ax.axhline(0, color="#666666", linewidth=0.7)
        ax.set_title(dataset)
        ax.set_xticks([0, 1], ["n", "4n"])
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Validation CatBoost TSTR/TRTR retention")
    legend = {}
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        legend.update(zip(labels, handles))
    fig.legend(list(legend.values()), list(legend), loc="lower center", ncol=3, fontsize=7,
               bbox_to_anchor=(0.5, -0.05))
    fig.suptitle("DOPE and native-tuned comparators · validation only · fit seed 23", fontsize=11)
    fig.tight_layout(rect=(0, 0.13, 1, 0.93))
    fig.savefig(svg_path, format="svg", bbox_inches="tight", metadata={"Date": None})
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight",
                metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)
    svg_path.write_text("\n".join(line.rstrip() for line in svg_path.read_text().splitlines()) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=SCRATCH)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--table", type=Path)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--svg", type=Path)
    parser.add_argument("--pdf", type=Path)
    parser.add_argument("--render-existing", type=Path)
    args = parser.parse_args()
    report = read(args.render_existing) if args.render_existing else build(args.root)
    encoded = json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if args.output:
        with args.output.open("w") as stream:
            stream.write(encoded)
    else:
        print(encoded, end="")
    for path, content in ((args.table, render_table(report)),
                          (args.csv, render_csv(report))):
        if path:
            path.write_text(content)
    if args.svg and args.pdf:
        render_figure(report, args.svg, args.pdf)


if __name__ == "__main__":
    main()
