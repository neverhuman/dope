"""Publish the frozen California autoregressive pilot validation result."""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
from statistics import median

from .score import sha256


HERE = Path(__file__).parent
ROOT = Path("/mnt/fast-scratch/dope-benchmark/pilot-24h")
FIT = ROOT / "california-q10-research-v1"
EVAL = FIT / "validation-v1"
WORKER = ROOT / "prepared-v2/worker/California"
PRIOR = HERE / "results/pilot24-native-neural.json"
RESULT = HERE / "results/pilot24-california-q10.json"
SEEDS = (101, 211, 307)
SIZES = (1, 4)


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def build() -> dict:
    fit_lock = read(FIT / "round.lock.json")
    eval_lock = read(EVAL / "round.lock.json")
    prior = read(PRIOR)
    preflight = read(FIT / "launch-preflight-v1.json")
    worker = read(WORKER / "worker-manifest.json")
    if (fit_lock["format"] != "dope-california-q10-pilot-research-round"
            or eval_lock["format"]
            != "dope-california-q10-pilot-validation-round"
            or fit_lock["source_sha256"]
            != sha256(FIT / "source/round.py")
            or eval_lock["source_sha256"]
            != sha256(EVAL / "source/evaluate.py")
            or fit_lock["package_manifest_sha256"]
            != sha256(FIT / "package/package-manifest.json")
            or fit_lock["probe_source_sha256"]
            != sha256(FIT / "package/research/benchmark/gpu_probe.py")
            or fit_lock["gpu_binary_sha256"]
            != sha256(Path(fit_lock["gpu_binary_path"]))
            or eval_lock["fit_round_sha256"]
            != sha256(FIT / "round.lock.json")
            or eval_lock["gpu_binary_sha256"]
            != fit_lock["gpu_binary_sha256"]
            or eval_lock["pilot_metric_source_sha256"]
            != sha256(HERE / "pilot_metrics.py")
            or eval_lock["sample_seeds"] != list(SEEDS)
            or eval_lock["size_multipliers"] != list(SIZES)
            or eval_lock["train_sha256"]
            != sha256(WORKER / "train.csv")
            or eval_lock["validation_sha256"]
            != sha256(WORKER / "validation.csv")
            or eval_lock["projection_sha256"]
            != sha256(WORKER / "projection.json")
            or eval_lock["worker_manifest_sha256"]
            != sha256(WORKER / "worker-manifest.json")
            or worker["dataset_id"] != "California"
            or worker["train_rows"] != eval_lock["train_rows"]
            or (WORKER / "test.csv").exists()
            or len(eval_lock["fits"]) != 2
            or fit_lock["gpu_fit_compute_ceiling_seconds"] != 1200
            or fit_lock["fit_timeout_seconds"] != 600
            or fit_lock["gpu_vram_mib_cap"] != 16 * 1024
            or fit_lock["official_tests_opened"] is not False
            or fit_lock["mfs_v2"] is not None
            or fit_lock["ptf_v1"] is not None
            or eval_lock["official_tests_opened"] is not False
            or eval_lock["mfs_v2"] is not None
            or eval_lock["ptf_v1"] is not None
            or prior["official_tests_opened"] is not False
            or prior["mfs_v2"] is not None
            or prior["ptf_v1"] is not None
            or preflight["round_lock_sha256"]
            != sha256(FIT / "round.lock.json")
            or preflight["source_sha256"] != fit_lock["source_sha256"]
            or len(preflight["events"]) != 4
            or any(event["gpu_fit_started"] is not False
                   for event in preflight["events"])):
        raise ValueError("California frozen source, inputs, or release gates changed")
    package = read(FIT / "package/package-manifest.json")
    for name, expected in package["files"].items():
        if sha256(FIT / "package/research/benchmark" / name) != expected:
            raise ValueError("California deployed runtime changed")
    reference = {(row["sample_seed"], row["size_multiplier"]): row
                 for row in prior["cells"]
                 if row["method"] == "DOPE"
                 and row["dataset"] == "California"
                 and row["configuration"] == "q8_symbolic"
                 and row["fit_seed"] == 23}
    if set(reference) != {(seed, size) for seed in SEEDS for size in SIZES}:
        raise ValueError("California q8 comparison cells changed")
    cells = []
    fits = []
    for fit in eval_lock["fits"]:
        seed = fit["fit_seed"]
        dispatch_path = FIT / f"dispatch/{seed}/attempt-0001.json"
        dispatch = read(dispatch_path)
        fit_path = Path(fit["fit_receipt_path"])
        fitted = read(fit_path)
        model = Path(fit["model_path"])
        compile_report_path = fit_path.with_suffix(".dpk.report.json")
        compile_report = read(compile_report_path)
        if (seed not in (11, 23)
                or dispatch["status"] != "ok"
                or dispatch["job"] not in fit_lock["gpu_jobs"]
                or dispatch["round_lock_sha256"]
                != sha256(FIT / "round.lock.json")
                or fit["dispatch_receipt_sha256"] != sha256(dispatch_path)
                or fitted["status"] != "ok"
                or fitted["identity"]["seed"] != seed
                or fitted["identity"]["dataset"] != "California"
                or fit["fit_receipt_sha256"] != sha256(fit_path)
                or fit["model_sha256"] != sha256(model)
                or fitted["artifact_sha256"] != fit["model_sha256"]
                or fit["artifact_bytes"]
                != model.stat().st_size + (WORKER / "projection.json").stat().st_size
                or fit["artifact_bytes"] != fitted["artifact_bytes"]
                or fit["artifact_bytes"] > 10_240
                or fitted["elapsed_seconds"] > 600
                or compile_report["compliant"] is not False
                or compile_report["failed_gates"]
                != ["utility", "driver", "production_evidence_unmeasured"]):
            raise ValueError("California fit, charge, or compiler gate changed")
        fits.append({"fit_seed": seed, "host": dispatch["host"],
                     "status": fitted["status"],
                     "artifact_bytes": fit["artifact_bytes"],
                     "model_sha256": fit["model_sha256"],
                     "fit_receipt_sha256": fit["fit_receipt_sha256"],
                     "dispatch_receipt_sha256": sha256(dispatch_path),
                     "compile_report_sha256": sha256(compile_report_path),
                     "elapsed_seconds": fitted["elapsed_seconds"],
                     "peak_gpu_used_mib": fitted["peak_gpu_used_mib"],
                     "energy_joules_estimate": fitted["energy_joules_estimate"],
                     "compiler_failed_gates": compile_report["failed_gates"]})
        for size in SIZES:
            for sample_seed in SEEDS:
                path = EVAL / f"cells/fit{seed}-sample{sample_seed}-size{size}"
                receipt_path = path / "receipt.json"
                receipt = read(receipt_path)
                metric_path = path / "metrics.json"
                metric = read(metric_path)
                retention = metric["utility"]["catboost"]["retention"]
                if (receipt["status"] != "ok"
                        or receipt["fit_seed"] != seed
                        or receipt["sample_seed"] != sample_seed
                        or receipt["size_multiplier"] != size
                        or receipt["round_lock_sha256"]
                        != sha256(EVAL / "round.lock.json")
                        or receipt["fit_receipt_sha256"]
                        != fit["fit_receipt_sha256"]
                        or receipt["model_sha256"] != fit["model_sha256"]
                        or receipt["artifact_bytes"] != fit["artifact_bytes"]
                        or receipt["sample_sha256"]
                        != sha256(path / "sample.csv")
                        or receipt["repeat_sha256"]
                        != receipt["sample_sha256"]
                        or receipt["metric_sha256"] != sha256(metric_path)
                        or metric["implementation_sha256"]
                        != eval_lock["pilot_metric_source_sha256"]
                        or metric["rows"]["synthetic"]
                        != worker["train_rows"] * size
                        or retention is None
                        or metric["gate_profile_complete"] is not False
                        or metric["mfs_v2"] is not None
                        or receipt["official_tests_opened"] is not False
                        or receipt["mfs_v2"] is not None
                        or receipt["ptf_v1"] is not None):
                    raise ValueError("California sample or validation metric changed")
                baseline = reference[(sample_seed, size)] if seed == 23 else None
                cells.append({"fit_seed": seed, "sample_seed": sample_seed,
                              "size_multiplier": size,
                              "catboost_retention": retention,
                              "q8_same_seed_retention":
                              baseline["catboost_retention"] if baseline else None,
                              "difference_q10_minus_q8":
                              retention - baseline["catboost_retention"]
                              if baseline else None,
                              "exact_row_copies": metric["copy_counts"]["exact"],
                              "near_row_copies": metric["copy_counts"]["near"],
                              "sample_sha256": receipt["sample_sha256"],
                              "sample_receipt_sha256": sha256(receipt_path),
                              "metric_receipt_sha256": sha256(metric_path),
                              "q8_metric_receipt_sha256":
                              baseline["metric_receipt_sha256"]
                              if baseline else None,
                              "artifact_bytes": fit["artifact_bytes"],
                              "within_l3_bytes": True,
                              "release_safe": None})
    if len(cells) != 12 or len({(row["fit_seed"], row["sample_seed"],
                                row["size_multiplier"]) for row in cells}) != 12:
        raise ValueError("California 12-cell validation matrix incomplete")
    summary = []
    for seed in (11, 23):
        for size in SIZES:
            rows = [row for row in cells if row["fit_seed"] == seed
                    and row["size_multiplier"] == size]
            summary.append({"fit_seed": seed, "size_multiplier": size,
                            "median_catboost_retention": median(
                                row["catboost_retention"] for row in rows),
                            "median_q8_same_seed_retention": median(
                                row["q8_same_seed_retention"] for row in rows)
                            if seed == 23 else None,
                            "median_paired_difference": median(
                                row["difference_q10_minus_q8"] for row in rows)
                            if seed == 23 else None,
                            "exact_row_copies": sum(
                                row["exact_row_copies"] for row in rows),
                            "near_row_copies": sum(
                                row["near_row_copies"] for row in rows)})
    return {"format": "dope-pilot24-california-q10-validation", "version": 1,
            "scope": "training_derived_validation_only",
            "source_sha256": sha256(Path(__file__)),
            "fit_round_sha256": sha256(FIT / "round.lock.json"),
            "validation_round_sha256": sha256(EVAL / "round.lock.json"),
            "preflight_failure_receipt_sha256":
                sha256(FIT / "launch-preflight-v1.json"),
            "prior_comparator_report_sha256": sha256(PRIOR),
            "fit_attempts": fits, "cells": cells, "summary": summary,
            "status_counts": {"fit_ok": 2, "metric_ok": 12,
                              "preflight_non_gpu_failures": 4},
            "notes": ["The California candidate was chosen after observing the q8 utility gap on this same validation dataset. Its independent fit seed is not an independent dataset confirmation or paper superiority test.",
                      "The compiler selected a symbolic artifact on GPU hosts; negligible GPU memory was used. Both artifacts fit L3 bytes but failed utility and driver gates. No neural GPU training or release certification is claimed.",
                      "Native-tuned TVAE and synthpop CART comparisons are reported separately using their own frozen validation KPIs; this DOPE-only round does not retune those comparators."],
            "official_tests_opened": False, "mfs_v2": None,
            "ptf_v1": None, "production_certified": False}


def render(report: dict) -> tuple[str, str]:
    table = io.StringIO()
    fields = ("fit_seed", "sample_seed", "size_multiplier",
              "catboost_retention", "q8_same_seed_retention",
              "difference_q10_minus_q8", "exact_row_copies",
              "near_row_copies", "artifact_bytes", "sample_receipt_sha256",
              "metric_receipt_sha256")
    writer = csv.DictWriter(table, fieldnames=fields, lineterminator="\n",
                            extrasaction="ignore")
    writer.writeheader()
    writer.writerows(report["cells"])
    lines = ["# California DOPE autoregressive pilot validation", "",
             "Two frozen fit seeds completed on GPU hosts, each with three "
             "sample seeds at n and 4n. All 12 samples and validation metrics "
             "are receipt verified; no exact or near training-row copies were "
             "observed.", "",
             "| Fit seed | Size | Candidate median CatBoost retention | "
             "q8 fit-23 median | Median paired difference |",
             "|---:|---:|---:|---:|---:|"]
    for row in report["summary"]:
        baseline = row["median_q8_same_seed_retention"]
        difference = row["median_paired_difference"]
        baseline_text = f"{baseline:.4f}" if baseline is not None else "—"
        difference_text = f"{difference:+.4f}" if difference is not None else "—"
        lines.append(f"| {row['fit_seed']} | {row['size_multiplier']}n | "
                     f"{row['median_catboost_retention']:.4f} | "
                     f"{baseline_text} | {difference_text} |")
    lines += ["", "The candidate was chosen after seeing California validation "
              "utility, so this is descriptive research. Fit seed 23 is lower "
              "than q8 at both sizes. The compiler selected a symbolic artifact "
              "on GPU hosts; no neural GPU training is claimed. Both 2,525-byte "
              "artifacts failed utility and driver gates. Official tests remain "
              "sealed; MFS-v2, PTF-v1, and certification are null.", ""]
    return table.getvalue(), "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-json", action="store_true")
    args = parser.parse_args()
    report = read(RESULT) if args.from_json else build()
    if (report["format"] != "dope-pilot24-california-q10-validation"
            or report["source_sha256"] != sha256(Path(__file__))
            or report["official_tests_opened"] is not False
            or report["mfs_v2"] is not None
            or report["ptf_v1"] is not None):
        raise ValueError("California publication changed")
    if not args.from_json:
        RESULT.write_text(json.dumps(report, sort_keys=True, indent=2,
                                     allow_nan=False) + "\n")
    csv_text, markdown = render(report)
    RESULT.with_suffix(".csv").write_text(csv_text)
    RESULT.with_suffix(".md").write_text(markdown)
    print(json.dumps(report["status_counts"], sort_keys=True))


if __name__ == "__main__":
    main()
