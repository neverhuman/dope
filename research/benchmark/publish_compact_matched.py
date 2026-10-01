"""Publish study-owned compact references against DOPE on pilot validation."""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
from statistics import median

from . import publish_synthpop_matched
from .manifest import digest
from .score import sha256


HERE = Path(__file__).parent
ROOT = Path("/mnt/fast-scratch/dope-benchmark/pilot-24h")
NATIVE = ROOT / "compact-native-v1"
FINAL = ROOT / "compact-final-v1"
RUN = ROOT / "compact-final-validation-v1"
METHOD_LOCK = HERE / "methods.lock.json"
RESULT = HERE / "results/pilot24-compact-matched.json"
METHODS = ("independent_marginals", "Chow-Liu")
DATASETS = ("Adult", "California", "News")
KINDS = ("default", "tuned")
SEEDS = (101, 211, 307)
SIZES = (1, 4)
CAP = 10_240


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def native_selections() -> dict:
    lock_path = NATIVE / "round.lock.json"
    lock = read(lock_path)
    methods = read(METHOD_LOCK)["methods"]
    if (lock["format"] != "dope-benchmark-compact-native-round"
            or lock["method_lock_sha256"] != sha256(METHOD_LOCK)
            or len(lock["jobs"]) != 6
            or lock["official_tests_opened"] is not False):
        raise ValueError("compact native matrix changed")
    selections = {}
    for job in lock["jobs"]:
        method, dataset = job["method"], job["dataset"]
        if method not in METHODS or dataset not in DATASETS:
            raise ValueError("compact native job outside pilot")
        entry = methods[method]
        path = NATIVE / "results" / dataset / method / "selection.json"
        selection = read(path)
        trials = selection["trials"]
        if (selection["round_sha256"] != sha256(lock_path)
                or selection["dataset"] != dataset
                or selection["method"] != method
                or selection["objective"] != entry["native_objective"]
                or selection["test_opened"] is not False
                or selection["total_wall_seconds"] > 43_200
                or len(trials) != len(job["configurations"])
                or len(trials) > 8):
            raise ValueError("compact native selection changed")
        for index, (config, trial) in enumerate(zip(job["configurations"], trials)):
            attempt_path = (NATIVE / "results" / dataset / method
                            / f"trial-{index:02d}/attempt.json")
            attempt = read(attempt_path)
            native_metric = read(Path(attempt["metric_receipt_path"]))
            if (trial["config"] != config
                    or trial["attempt_receipt_sha256"] != sha256(attempt_path)
                    or trial != {**attempt,
                                 "attempt_receipt_path": str(attempt_path),
                                 "attempt_receipt_sha256": sha256(attempt_path)}
                    or attempt["identity"]["round_sha256"] != sha256(lock_path)
                    or attempt["identity"]["native_objective_sha256"]
                    != entry["native_objective"]["implementation_sha256"]
                    or attempt["status"] != "ok"
                    or sha256(Path(attempt["metric_receipt_path"]))
                    != attempt["metric_receipt_sha256"]
                    or native_metric["value"] != attempt["native_kpi"]
                    or native_metric["validation_sha256"]
                    != job["validation_sha256"]
                    or native_metric["implementation_sha256"]
                    != entry["native_objective"]["implementation_sha256"]):
                raise ValueError("compact native attempt changed")
        chosen = min(enumerate(trials),
                     key=lambda pair: (-pair[1]["native_kpi"],
                                       pair[1]["artifact_bytes"],
                                       digest(pair[1]["config"]), pair[0]))
        defaults = [trial for trial in trials
                    if trial["config"] == entry["default_config"]]
        if (len(defaults) != 1
                or chosen[0] != selection["selected_trial_index"]
                or chosen[1]["config"] != selection["selected_config"]):
            raise ValueError("compact native winner or author default changed")
        key = (dataset, method)
        if key in selections:
            raise ValueError("duplicate compact native selection")
        selections[key] = {"selection_sha256": sha256(path),
                           "objective": "mean_log_density",
                           "direction": "maximize",
                           "default_native_kpi": defaults[0]["native_kpi"],
                           "selected_native_kpi": chosen[1]["native_kpi"],
                           "selected_trial": chosen[0],
                           "tuning_trials": len(trials),
                           "tuning_wall_seconds": selection["total_wall_seconds"]}
    if set(selections) != {(dataset, method) for dataset in DATASETS
                           for method in METHODS}:
        raise ValueError("compact native selections incomplete")
    return selections


def build() -> dict:
    lock_path = RUN / "round.lock.json"
    manifest_path = RUN / "manifest.json"
    final_path = FINAL / "round.lock.json"
    lock, manifest, final = (read(path) for path in
                             (lock_path, manifest_path, final_path))
    if (lock["format"] != "dope-compact-pilot-fixed-seed-validation-round"
            or lock["final_round_sha256"] != sha256(final_path)
            or len(final["jobs"]) != 12
            or len(lock["cells"]) != 72
            or lock["failed_fit_cells"] or lock["failed_sample_cells"]
            or manifest["exact_cells"] != 72
            or manifest["round_lock_sha256"] != sha256(lock_path)
            or manifest["official_tests_opened"] is not False
            or lock["official_tests_opened"] is not False
            or lock["mfs_v2"] is not None or lock["ptf_v1"] is not None):
        raise ValueError("compact common validation matrix incomplete")
    replayed = {row["cell_digest"]: row for row in manifest["cells"]}
    if len(replayed) != 72:
        raise ValueError("compact replay matrix incomplete")
    native = native_selections()
    dope = publish_synthpop_matched.dope_reference()
    cells = []
    for cell in lock["cells"]:
        job = cell["job"]
        dataset, method, kind = job["dataset"], job["method"], job["kind"]
        seed = cell["sample_seed"]
        size = cell["size_multiplier"]
        selection = native[(dataset, method)]
        fit_path = Path(cell["fit_receipt_path"])
        fit = read(fit_path)
        code = digest(cell)
        metric_path = RUN / "metrics" / f"{code}.json"
        replay_path = RUN / "replay" / f"{code}.json"
        metric, replay = read(metric_path), read(replay_path)
        check = replayed[code]
        if (job["fit_seed"] != 23 or kind not in KINDS
                or seed not in SEEDS or size not in SIZES
                or (kind == "tuned" and job["configuration"]["selection_sha256"]
                    != selection["selection_sha256"])
                or sha256(fit_path) != cell["fit_receipt_sha256"]
                or fit["status"] != "ok"
                or fit["artifact_bytes"] != cell["artifact_bytes"]
                or metric["cell"] != cell
                or metric["round_lock_sha256"] != sha256(lock_path)
                or metric["metric_source_sha256"]
                != lock["metric_source_sha256"]
                or metric["metrics"]["implementation_sha256"]
                != lock["metric_source_sha256"]
                or metric["official_tests_opened"] is not False
                or metric["mfs_v2"] is not None
                or metric["ptf_v1"] is not None
                or check["metric_receipt_sha256"] != sha256(metric_path)
                or check["replay_receipt_sha256"] != sha256(replay_path)
                or replay["cell"] != cell
                or replay["status"] != "exact"
                or replay["round_lock_sha256"] != sha256(lock_path)
                or replay["metric_receipt_sha256"] != sha256(metric_path)
                or replay["official_tests_opened"] is not False
                or replay["mfs_v2"] is not None
                or replay["ptf_v1"] is not None
                or replay["metric_payload"] != {name: value
                    for name, value in metric["metrics"].items()
                    if name != "metric_seconds"}):
            raise ValueError("compact metric, replay, or fit changed")
        outcome = metric["metrics"]
        retention = outcome["utility"]["catboost"]["retention"]
        if (retention is None or outcome["mfs_v2"] is not None
                or outcome["gate_profile_complete"] is not False):
            raise ValueError("compact common validation outcome missing")
        reference = dope[(dataset, seed, size)]
        kpi = selection["default_native_kpi" if kind == "default"
                        else "selected_native_kpi"]
        copy = outcome["copy_counts"]
        cells.append({"dataset": dataset, "method": method,
                      "implementation": "study_owned_reference",
                      "configuration": kind, "fit_seed": 23,
                      "sample_seed": seed, "size_multiplier": size,
                      "native_objective": "mean_log_density",
                      "native_validation_kpi": kpi,
                      "compact_artifact_bytes": cell["artifact_bytes"],
                      "compact_within_l3_bytes": cell["artifact_bytes"] <= CAP,
                      "compact_retention": retention,
                      "compact_exact_copies": copy["exact"],
                      "compact_near_copies": copy["near"],
                      "compact_copy_gate_pass":
                          copy["exact"] == 0 and copy["near"] == 0,
                      "compact_release_safe": False if
                          cell["artifact_bytes"] > CAP or copy["exact"] > 0
                          or copy["near"] > 0 else None,
                      "dope_configuration": reference["configuration"],
                      "dope_artifact_bytes": reference["artifact_bytes"],
                      "dope_retention": reference["retention"],
                      "dope_release_safe": None,
                      "difference_dope_minus_compact":
                          reference["retention"] - retention,
                      "fit_receipt_sha256": sha256(fit_path),
                      "sample_attempt_sha256": cell["sample_attempt_sha256"],
                      "sample_receipt_sha256": cell["sample_receipt_sha256"],
                      "metric_receipt_sha256": sha256(metric_path),
                      "replay_receipt_sha256": sha256(replay_path),
                      "dope_metric_receipt_sha256":
                          reference["metric_receipt_sha256"]})
    expected = {(dataset, method, kind, seed, size)
                for dataset in DATASETS for method in METHODS for kind in KINDS
                for seed in SEEDS for size in SIZES}
    if {(row["dataset"], row["method"], row["configuration"],
         row["sample_seed"], row["size_multiplier"])
        for row in cells} != expected:
        raise ValueError("compact paired cells incomplete")
    cells.sort(key=lambda row: (row["dataset"], row["method"],
                                row["configuration"], row["size_multiplier"],
                                row["sample_seed"]))
    summary = []
    for dataset in DATASETS:
        for method in METHODS:
            for kind in KINDS:
                for size in SIZES:
                    rows = [row for row in cells if row["dataset"] == dataset
                            and row["method"] == method
                            and row["configuration"] == kind
                            and row["size_multiplier"] == size]
                    if len(rows) != 3:
                        raise ValueError("compact paired summary incomplete")
                    summary.append({"dataset": dataset, "method": method,
                                    "configuration": kind,
                                    "size_multiplier": size,
                                    "paired_sample_seeds": 3,
                                    "native_validation_kpi": rows[0][
                                        "native_validation_kpi"],
                                    "compact_artifact_bytes": rows[0][
                                        "compact_artifact_bytes"],
                                    "dope_artifact_bytes": rows[0][
                                        "dope_artifact_bytes"],
                                    "median_compact_retention": median(
                                        row["compact_retention"] for row in rows),
                                    "median_dope_retention": median(
                                        row["dope_retention"] for row in rows),
                                    "median_paired_difference": median(
                                        row["difference_dope_minus_compact"]
                                        for row in rows),
                                    "compact_cells_failing_copy_gate": sum(
                                        not row["compact_copy_gate_pass"]
                                        for row in rows)})
    return {"format": "dope-pilot24-compact-matched-validation",
            "version": 1, "scope": "training_derived_validation_only",
            "source_sha256": sha256(Path(__file__)),
            "implementation": "study_owned_references_not_author_code",
            "locks": {"method_source": sha256(METHOD_LOCK),
                      "native_round": sha256(NATIVE / "round.lock.json"),
                      "final_sample_round": sha256(final_path),
                      "final_validation_round": sha256(lock_path),
                      "exact_replay_manifest": sha256(manifest_path),
                      "dope_neural_report": sha256(
                          publish_synthpop_matched.NEURAL),
                      "dope_packed_news_report": sha256(
                          publish_synthpop_matched.PACKED)},
            "native_selections": {f"{dataset}/{method}": value
                                  for (dataset, method), value
                                  in sorted(native.items())},
            "l3_byte_cap": CAP, "cells": cells, "summary": summary,
            "notes": ["Independent marginals and discretized Chow-Liu are study-owned reference implementations; they are not original author implementations.",
                      "Each method's own frozen held-out mean log density selected its tuned configuration. Native KPI values compare configurations only within one method and dataset.",
                      "Fit seed 23 and three paired sample seeds at n and 4n are validation-only pilot evidence. Other frozen sample sizes are outside this common metric readout.",
                      "Byte and row-copy outcomes are shown per cell; no method is labeled release safe without the complete release gates.",
                      "News validation retention is unstable because the real-auditor improvement over the null target predictor is small; large negative values are retained as measured rather than clipped.",
                      "Official tests remain sealed; MFS-v2, PTF-v1, production certification, and paper superiority claims are unavailable."],
            "official_tests_opened": False, "mfs_v2": None,
            "ptf_v1": None, "production_certified": False}


def render(report: dict) -> tuple[str, str]:
    table = io.StringIO()
    fields = ("dataset", "method", "configuration", "size_multiplier",
              "paired_sample_seeds", "native_validation_kpi",
              "compact_artifact_bytes", "dope_artifact_bytes",
              "median_compact_retention", "median_dope_retention",
              "median_paired_difference", "compact_cells_failing_copy_gate")
    writer = csv.DictWriter(table, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(report["summary"])
    lines = ["# DOPE versus compact study references: paired pilot validation", "",
             "Independent marginals and Chow-Liu use study-owned implementations. "
             "Their tuned configurations maximize each method's frozen held-out "
             "mean log density. Native KPI values are comparable within a method "
             "and dataset only. The shared outcome is CatBoost TSTR/TRTR "
             "retention at fit seed 23 and three paired sample seeds.", "",
             "| Dataset | Method | Config | Size | Native log density | "
             "Reference retention | DOPE retention | Paired DOPE − reference | "
             "Reference bytes | Copy-gate failures |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in report["summary"]:
        lines.append(f"| {row['dataset']} | {row['method']} | "
                     f"{row['configuration']} | {row['size_multiplier']}n | "
                     f"{row['native_validation_kpi']:.6f} | "
                     f"{row['median_compact_retention']:.4f} | "
                     f"{row['median_dope_retention']:.4f} | "
                     f"{row['median_paired_difference']:+.4f} | "
                     f"{row['compact_artifact_bytes']:,} | "
                     f"{row['compact_cells_failing_copy_gate']}/3 |")
    lines += ["", "News retention has a small real-over-null utility denominator; "
              "large negative values are measured and are not clipped. "
              "JSON retains every seed, artifact charge, row-copy count, "
              "native selection, and receipt hash. Official tests remain sealed; "
              "MFS-v2, PTF-v1, and production certification are null. This "
              "three-dataset, one-fit-seed pilot is not a paper superiority "
              "result.", ""]
    return table.getvalue(), "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-json", action="store_true")
    args = parser.parse_args()
    report = read(RESULT) if args.from_json else build()
    if (report["format"] != "dope-pilot24-compact-matched-validation"
            or report["source_sha256"] != sha256(Path(__file__))
            or report["official_tests_opened"] is not False
            or report["mfs_v2"] is not None or report["ptf_v1"] is not None):
        raise ValueError("compact matched report changed")
    if not args.from_json:
        RESULT.write_text(json.dumps(report, sort_keys=True, indent=2,
                                     allow_nan=False) + "\n")
    csv_text, markdown = render(report)
    RESULT.with_suffix(".csv").write_text(csv_text)
    RESULT.with_suffix(".md").write_text(markdown)
    print(json.dumps({"cells": len(report["cells"]),
                      "summary_rows": len(report["summary"])}, sort_keys=True))


if __name__ == "__main__":
    main()
