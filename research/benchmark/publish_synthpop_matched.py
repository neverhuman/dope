"""Publish the fixed-fit synthpop CART and DOPE validation comparison."""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
from statistics import median

from .manifest import digest
from .score import sha256


HERE = Path(__file__).parent
ROOT = Path("/mnt/fast-scratch/dope-benchmark/pilot-24h")
RUN = ROOT / "synthpop-cart-final-validation-v1"
SAMPLES = ROOT / "synthpop-cart-final-samples-v1"
NATIVE = ROOT / "synthpop-cart-native-v1"
RESULT = HERE / "results/pilot24-synthpop-matched.json"
NEURAL = HERE / "results/pilot24-native-neural.json"
PACKED = HERE / "results/pilot24-news-q10-packed.json"
CAP = 10_240
DATASETS = ("Adult", "California", "News")
SEEDS = (101, 211, 307)
SIZES = (1, 4)


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def dope_reference() -> dict[tuple[str, int, int], dict]:
    neural = read(NEURAL)
    packed = read(PACKED)
    if (neural["official_tests_opened"] is not False
            or neural["mfs_v2"] is not None or neural["ptf_v1"] is not None
            or packed["official_tests_opened"] is not False
            or packed["mfs_v2"] is not None or packed["ptf_v1"] is not None
            or packed["locks"]["neural_comparison"] != sha256(NEURAL)
            or packed["packed_q10_artifact_bytes"] != 9_205
            or packed["packed_sample_checks_exact"] != 12
            or len(packed["paired_cells"]) != 12):
        raise ValueError("DOPE validation reference changed")
    rows = {}
    for cell in neural["cells"]:
        if (cell["method"] != "DOPE" or cell["dataset"] not in DATASETS
                or cell["dataset"] == "News"):
            continue
        if (cell["configuration"] != "q8_symbolic"
                or cell["fit_seed"] != 23 or not cell["within_l3_bytes"]):
            raise ValueError("DOPE q8 reference changed")
        key = (cell["dataset"], cell["sample_seed"], cell["size_multiplier"])
        if key in rows:
            raise ValueError("duplicate DOPE validation reference")
        rows[key] = {"retention": cell["catboost_retention"],
                     "artifact_bytes": cell["artifact_bytes"],
                     "metric_receipt_sha256": cell["metric_receipt_sha256"],
                     "sample_receipt_sha256": cell["sample_receipt_sha256"],
                     "configuration": "q8_symbolic"}
    for cell in packed["paired_cells"]:
        key = ("News", cell["sample_seed"], cell["size_multiplier"])
        value = {"retention": cell["dope_retention"],
                 "artifact_bytes": cell["dope_packed_artifact_bytes"],
                 "metric_receipt_sha256": cell["dope_metric_receipt_sha256"],
                 "sample_receipt_sha256": cell["dope_sample_check_sha256"],
                 "configuration": "q10_packed"}
        if key in rows and rows[key] != value:
            raise ValueError("News q10 comparator copies disagree")
        rows[key] = value
    expected = {(dataset, seed, size) for dataset in DATASETS
                for seed in SEEDS for size in SIZES}
    if set(rows) != expected:
        raise ValueError("DOPE comparison cells incomplete")
    return rows


def native_selections() -> dict[str, dict]:
    lock_path = NATIVE / "round.lock.json"
    lock = read(lock_path)
    if (lock["format"] != "dope-synthpop-cart-pilot-native-round"
            or lock["native_objective"]["name"] != "validation_cart_pMSE"
            or lock["native_objective"]["direction"] != "minimize"
            or lock["official_tests_opened"] is not False):
        raise ValueError("synthpop native objective changed")
    selections = {}
    for dataset in DATASETS:
        path = NATIVE / f"{dataset}-selection.json"
        choice = read(path)
        trials = choice["trials"]
        jobs = sorted((job for job in lock["jobs"] if job["dataset"] == dataset),
                      key=lambda job: job["trial"])
        if len(jobs) != 4 or [job["trial"] for job in jobs] != list(range(4)):
            raise ValueError("synthpop native job grid changed")
        for job, trial in zip(jobs, trials):
            receipt_path = NATIVE / "jobs" / digest(job) / "attempt-0001/receipt.json"
            receipt = read(receipt_path)
            if (trial["trial"] != job["trial"]
                    or trial["config"] != job["config"]
                    or trial["attempt_receipt_sha256"] != sha256(receipt_path)
                    or receipt["job"] != job
                    or receipt["round_sha256"] != sha256(lock_path)
                    or receipt["status"] != trial["status"]
                    or receipt["native_kpi"] != trial["native_kpi"]
                    or receipt["artifact_bytes"] != trial["artifact_bytes"]
                    or receipt["wall_seconds"] != trial["wall_seconds"]
                    or receipt["official_tests_opened"] is not False):
                raise ValueError("synthpop native attempt lineage changed")
            if receipt["status"] == "ok":
                metric_path = receipt_path.parent / "native-kpi.json"
                metric = read(metric_path)
                if (receipt["evidence_files"]["native-kpi.json"]
                        != sha256(metric_path)
                        or metric["value"] != receipt["native_kpi"]):
                    raise ValueError("synthpop native KPI value changed")
        eligible = [row for row in trials if row["status"] == "ok"]
        chosen = min(eligible, key=lambda row: (row["native_kpi"],
                     row["artifact_bytes"], digest(row["config"])))
        if (choice["dataset"] != dataset
                or choice["objective"] != lock["native_objective"]
                or choice["round_lock_sha256"] != sha256(lock_path)
                or choice["tuning_trials"] != 4 or len(trials) != 4
                or choice["tuning_wall_seconds"] > 43_200
                or choice["selected_trial"] != chosen["trial"]
                or choice["selected_config"] != chosen["config"]
                or choice["selected_native_kpi"] != chosen["native_kpi"]
                or choice["official_tests_opened"] is not False
                or choice["mfs_v2"] is not None or choice["ptf_v1"] is not None):
            raise ValueError("synthpop native selection changed")
        selections[dataset] = {"selection_sha256": sha256(path),
                               "default_native_kpi": trials[0]["native_kpi"],
                               "selected_native_kpi": chosen["native_kpi"],
                               "selected_trial": chosen["trial"],
                               "tuning_trials": 4,
                               "tuning_wall_seconds": choice["tuning_wall_seconds"]}
    return selections


def build() -> dict:
    lock_path = RUN / "round.lock.json"
    manifest_path = RUN / "manifest.json"
    lock = read(lock_path)
    manifest = read(manifest_path)
    sample_lock_path = SAMPLES / "round.lock.json"
    sample_lock = read(sample_lock_path)
    source_audit_path = HERE / "synthpop-cart-source.lock.json"
    native_lock = read(NATIVE / "round.lock.json")
    if (lock["format"] != "dope-synthpop-cart-pilot-validation-round"
            or lock["profile"] != "fit23_three_sample_seeds_n_4n"
            or lock["sample_round_sha256"] != sha256(sample_lock_path)
            or sample_lock["format"]
            != "dope-synthpop-cart-pilot-final-sample-round"
            or native_lock["source_audit_sha256"] != sha256(source_audit_path)
            or len(sample_lock["jobs"]) != 72
            or len(lock["cells"]) != 36
            or lock["failed_sample_cells"] or lock["failed_fit_cells"]
            or manifest["round_lock_sha256"] != sha256(lock_path)
            or manifest["exact_cells"] != 36
            or manifest["official_tests_opened"] is not False
            or lock["official_tests_opened"] is not False
            or lock["mfs_v2"] is not None or lock["ptf_v1"] is not None):
        raise ValueError("synthpop matched validation matrix incomplete")
    checks = {row["cell_digest"]: row for row in manifest["cells"]}
    if len(checks) != 36:
        raise ValueError("synthpop exact replay matrix incomplete")
    dope = dope_reference()
    native = native_selections()
    cells = []
    for cell in lock["cells"]:
        job = cell["job"]
        key = (job["dataset"], job["sample_seed"], job["size_multiplier"])
        code = digest(cell)
        metric_path = RUN / "metrics" / f"{code}.json"
        replay_path = RUN / "replay" / f"{code}.json"
        metric = read(metric_path)
        replay = read(replay_path)
        check = checks[code]
        fit_path = Path(job["fit_receipt_path"])
        fit = read(fit_path)
        if (job["fit_seed"] != 23 or job["kind"] not in ("default", "tuned")
                or job["sample_seed"] not in SEEDS
                or job["size_multiplier"] not in SIZES
                or job["selection_sha256"]
                != native[job["dataset"]]["selection_sha256"]
                or job["fit_trial"] != (0 if job["kind"] == "default"
                    else native[job["dataset"]]["selected_trial"])
                or sha256(fit_path) != job["fit_receipt_sha256"]
                or fit["status"] != "ok"
                or fit["job"]["dataset"] != job["dataset"]
                or fit["job"]["fit_seed"] != 23
                or fit["artifact_bytes"] <= CAP
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
                or replay["status"] != "exact"
                or replay["cell"] != cell
                or replay["round_lock_sha256"] != sha256(lock_path)
                or replay["metric_receipt_sha256"] != sha256(metric_path)
                or replay["official_tests_opened"] is not False
                or replay["mfs_v2"] is not None
                or replay["ptf_v1"] is not None
                or replay["metric_payload"] != {name: value
                    for name, value in metric["metrics"].items()
                    if name != "metric_seconds"}):
            raise ValueError("synthpop metric, replay, or artifact changed")
        outcome = metric["metrics"]
        retention = outcome["utility"]["catboost"]["retention"]
        if (retention is None or outcome["mfs_v2"] is not None
                or outcome["gate_profile_complete"] is not False):
            raise ValueError("synthpop common validation outcome changed")
        reference = dope[key]
        cells.append({"dataset": key[0], "configuration": job["kind"],
                      "fit_seed": 23, "sample_seed": key[1],
                      "size_multiplier": key[2],
                      "native_objective": "validation_cart_pMSE",
                      "native_validation_kpi": native[key[0]][
                          "default_native_kpi" if job["kind"] == "default"
                          else "selected_native_kpi"],
                      "synthpop_artifact_bytes": fit["artifact_bytes"],
                      "synthpop_within_l3_bytes": False,
                      "synthpop_retention": retention,
                      "synthpop_exact_copies": outcome["copy_counts"]["exact"],
                      "synthpop_near_copies": outcome["copy_counts"]["near"],
                      "synthpop_copy_gate_pass":
                          outcome["copy_counts"]["exact"] == 0
                          and outcome["copy_counts"]["near"] == 0,
                      "synthpop_release_safe": False,
                      "dope_configuration": reference["configuration"],
                      "dope_artifact_bytes": reference["artifact_bytes"],
                      "dope_release_safe": None,
                      "dope_retention": reference["retention"],
                      "difference_dope_minus_synthpop":
                          reference["retention"] - retention,
                      "fit_receipt_sha256": sha256(fit_path),
                      "sample_receipt_sha256": cell["sample_receipt_sha256"],
                      "metric_receipt_sha256": sha256(metric_path),
                      "replay_receipt_sha256": sha256(replay_path),
                      "dope_metric_receipt_sha256":
                          reference["metric_receipt_sha256"]})
    expected = {(dataset, kind, seed, size) for dataset in DATASETS
                for kind in ("default", "tuned") for seed in SEEDS
                for size in SIZES}
    if {(row["dataset"], row["configuration"], row["sample_seed"],
         row["size_multiplier"]) for row in cells} != expected:
        raise ValueError("synthpop paired metric cells missing")
    cells.sort(key=lambda row: (row["dataset"], row["configuration"],
                                row["size_multiplier"], row["sample_seed"]))
    summary = []
    for dataset in DATASETS:
        for kind in ("default", "tuned"):
            for size in SIZES:
                rows = [row for row in cells if row["dataset"] == dataset
                        and row["configuration"] == kind
                        and row["size_multiplier"] == size]
                if (len(rows) != 3
                        or len({row["synthpop_artifact_bytes"] for row in rows}) != 1
                        or len({row["dope_artifact_bytes"] for row in rows}) != 1
                        or len({row["native_validation_kpi"] for row in rows}) != 1):
                    raise ValueError("paired synthpop summary changed")
                summary.append({"dataset": dataset, "configuration": kind,
                                "size_multiplier": size,
                                "paired_sample_seeds": len(rows),
                                "native_validation_kpi": rows[0]["native_validation_kpi"],
                                "synthpop_artifact_bytes": rows[0]["synthpop_artifact_bytes"],
                                "dope_artifact_bytes": rows[0]["dope_artifact_bytes"],
                                "median_synthpop_retention": median(
                                    row["synthpop_retention"] for row in rows),
                                "median_dope_retention": median(
                                    row["dope_retention"] for row in rows),
                                "median_paired_difference": median(
                                    row["difference_dope_minus_synthpop"]
                                    for row in rows),
                                "synthpop_cells_with_exact_copies": sum(
                                    row["synthpop_exact_copies"] > 0 for row in rows),
                                "synthpop_cells_failing_copy_gate": sum(
                                    not row["synthpop_copy_gate_pass"]
                                    for row in rows)})
    return {"format": "dope-pilot24-synthpop-matched-validation",
            "version": 1, "scope": "training_derived_validation_only",
            "source_sha256": sha256(Path(__file__)),
            "locks": {"native_round": sha256(NATIVE / "round.lock.json"),
                      "source_audit": sha256(source_audit_path),
                      "synthpop_source_archive":
                          native_lock["source_archive_sha256"],
                      "final_sample_round": sha256(sample_lock_path),
                      "final_validation_round": sha256(lock_path),
                      "exact_replay_manifest": sha256(manifest_path),
                      "dope_neural_report": sha256(NEURAL),
                      "dope_packed_news_report": sha256(PACKED)},
            "native_selections": native, "l3_byte_cap": CAP,
            "cells": cells, "summary": summary,
            "notes": ["Synthpop CART tuning minimized its own frozen validation CART pMSE; values are comparable only across synthpop configurations on the same dataset.",
                      "Fit seed 23 and three paired sample seeds at n and 4n are validation-only pilot evidence. The other frozen sample sizes are outside this common metric readout.",
                      "All synthpop fitted artifacts exceed the L3 byte cap. Exact and near row copies are reported per sample; a copy failure is not a release-safe win.",
                      "DOPE uses the previously frozen q8 symbolic candidate on Adult/California and exactly replayed 9,205-byte packed q10 candidate on News.",
                      "Official tests remain sealed; MFS-v2, PTF-v1, production certification, and paper superiority claims are unavailable."],
            "official_tests_opened": False, "mfs_v2": None,
            "ptf_v1": None, "production_certified": False}


def render(report: dict) -> tuple[str, str]:
    table = io.StringIO()
    fields = ("dataset", "configuration", "size_multiplier",
              "paired_sample_seeds", "native_validation_kpi",
              "synthpop_artifact_bytes", "dope_artifact_bytes",
              "median_synthpop_retention", "median_dope_retention",
              "median_paired_difference", "synthpop_cells_with_exact_copies",
              "synthpop_cells_failing_copy_gate")
    writer = csv.DictWriter(table, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(report["summary"])
    lines = ["# DOPE versus synthpop CART: paired pilot validation", "",
             "Synthpop configurations were selected by their own frozen CART "
             "pMSE on training-derived validation rows. Native pMSE values "
             "compare synthpop configurations within a dataset only. The "
             "common outcome is CatBoost TSTR/TRTR retention at fit seed 23 "
             "and three paired sample seeds.", "",
             "| Dataset | Synthpop | Size | Native pMSE | Synthpop retention | "
             "DOPE retention | Paired DOPE − synthpop | Synthpop bytes | "
             "Copy-gate failures |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in report["summary"]:
        lines.append(f"| {row['dataset']} | {row['configuration']} | "
                     f"{row['size_multiplier']}n | "
                     f"{row['native_validation_kpi']:.6f} | "
                     f"{row['median_synthpop_retention']:.4f} | "
                     f"{row['median_dope_retention']:.4f} | "
                     f"{row['median_paired_difference']:+.4f} | "
                     f"{row['synthpop_artifact_bytes']:,} | "
                     f"{row['synthpop_cells_failing_copy_gate']}/3 |")
    lines += ["", "All synthpop artifacts exceed the 10,240-byte L3 limit. "
              "The JSON retains each paired seed, receipt hash, copy count, "
              "and failed release gate. Official tests remain sealed; MFS-v2, "
              "PTF-v1, and production certification are null. These three "
              "datasets and one fit seed do not support a paper superiority "
              "claim.", ""]
    return table.getvalue(), "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-json", action="store_true")
    args = parser.parse_args()
    report = read(RESULT) if args.from_json else build()
    if (report["format"] != "dope-pilot24-synthpop-matched-validation"
            or report["source_sha256"] != sha256(Path(__file__))
            or report["official_tests_opened"] is not False
            or report["mfs_v2"] is not None or report["ptf_v1"] is not None):
        raise ValueError("synthpop matched report changed")
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
