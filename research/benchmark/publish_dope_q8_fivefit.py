"""Publish fixed q8 Adult/California fit-seed stability from sealed pilot receipts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
from statistics import median

from research.benchmark.manifest import digest
from research.benchmark.score import sha256


HERE = Path(__file__).resolve().parent
ROOT = Path("/mnt/fast-scratch/dope-benchmark/pilot-24h")
FIT = ROOT / "dope-q8-fivefit-v1"
EVAL = ROOT / "dope-q8-fivefit-validation-v2"
AUDIT = ROOT / "dope-q8-fivefit-audit-v1"
FAILED_FREEZE = ROOT / "dope-q8-fivefit-validation-v1/freeze-preflight-failure.json"
PREFLIGHT_LEDGER = EVAL / "launch-preflight-v1.json"
PRIOR = HERE / "results/pilot24-native-neural.json"
RESULT = HERE / "results/pilot24-dope-q8-fivefit.json"
DATASETS = ("Adult", "California")
FITS = (11, 23, 37, 53, 71)
SEEDS = (101, 211, 307)
SIZES = (1, 2, 4, 8)
METRIC_SIZES = (1, 4)
CAP = 10_240


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def verify_audit(eval_lock: dict) -> dict:
    lock = read(AUDIT / "round.lock.json")
    manifest = read(AUDIT / "manifest.json")
    selected = [job for job in eval_lock["jobs"]
                if job["fit_seed"] in (11, 71)
                and job["sample_seed"] in (101, 307)
                and job["size_multiplier"] in METRIC_SIZES]
    if (lock["format"] != "dope-pilot24-q8-fivefit-stratified-audit"
            or lock["source_sha256"] != sha256(AUDIT / "source/audit.py")
            or lock["parent_round_sha256"] != sha256(EVAL / "round.lock.json")
            or lock["metric_source_sha256"] != eval_lock["metric_source_sha256"]
            or lock["gpu_binary_sha256"] != eval_lock["gpu_binary_sha256"]
            or [cell["job"] for cell in lock["cells"]] != selected
            or len(selected) != 16
            or manifest["format"] != "dope-pilot24-q8-fivefit-audit-manifest"
            or manifest["round_lock_sha256"] != sha256(AUDIT / "round.lock.json")
            or manifest["exact_cells"] != 16
            or len(manifest["cells"]) != 16
            or any(row["official_tests_opened"] is not False
                   or row["mfs_v2"] is not None or row["ptf_v1"] is not None
                   for row in (lock, manifest))):
        raise ValueError("q8 stratified audit lineage changed")
    receipts = []
    for cell, item in zip(lock["cells"], manifest["cells"]):
        job_hash = digest(cell["job"])
        path = AUDIT / "cells" / f"{job_hash}.json"
        receipt = read(path)
        parent = Path(cell["sample_receipt_path"])
        metric_path = parent.parent / "attempt-0001/metrics.json"
        metric = read(metric_path)
        payload = {key: value for key, value in metric.items()
                   if key != "metric_seconds"}
        payload_hash = hashlib.sha256(json.dumps(
            payload, sort_keys=True, allow_nan=False).encode()).hexdigest()
        if (item["job_digest"] != job_hash
                or item["receipt_sha256"] != sha256(path)
                or receipt["cell"] != cell
                or receipt["round_lock_sha256"] != manifest["round_lock_sha256"]
                or receipt["status"] != "exact"
                or receipt["sample_sha256"] != cell["sample_sha256"]
                or sha256(AUDIT / "samples" / f"{job_hash}.csv")
                != cell["sample_sha256"]
                or sha256(parent) != cell["sample_receipt_sha256"]
                or sha256(metric_path) != cell["metric_sha256"]
                or receipt["metric_payload_sha256"] != payload_hash
                or receipt["official_tests_opened"] is not False
                or receipt["mfs_v2"] is not None or receipt["ptf_v1"] is not None):
            raise ValueError("q8 stratified replay receipt changed")
        receipts.append({"job_digest": job_hash, "receipt_path": str(path),
                         "receipt_sha256": sha256(path)})
    return {"round_lock_sha256": sha256(AUDIT / "round.lock.json"),
            "manifest_path": str(AUDIT / "manifest.json"),
            "manifest_sha256": sha256(AUDIT / "manifest.json"),
            "source_sha256": lock["source_sha256"], "exact_cells": 16,
            "selection_rule": lock["selection_rule"], "receipts": receipts}


def build() -> dict:
    fit_lock = read(FIT / "round.lock.json")
    eval_lock = read(EVAL / "round.lock.json")
    prior = read(PRIOR)
    preflight = read(FAILED_FREEZE)
    preflight_ledger = read(PREFLIGHT_LEDGER)
    package_root = ROOT / "california-q10-research-v1/package"
    package = read(package_root / "package-manifest.json")
    if (fit_lock["format"] != "dope-pilot24-q8-fivefit-replication-round"
            or fit_lock["source_sha256"] != sha256(FIT / "source/round.py")
            or fit_lock["runtime_manifest_sha256"]
            != sha256(package_root / "package-manifest.json")
            or fit_lock["probe_source_sha256"] != package["files"]["gpu_probe.py"]
            or fit_lock["gpu_binary_sha256"]
            != sha256(Path(fit_lock["gpu_binary_path"]))
            or fit_lock["baseline_report_sha256"] != sha256(PRIOR)
            or fit_lock["gpu_fit_compute_ceiling_seconds"] != 6_000
            or fit_lock["fit_timeout_seconds"] != 600
            or fit_lock["gpu_vram_mib_cap"] != 16 * 1024
            or len(fit_lock["jobs"]) != 10
            or eval_lock["format"] != "dope-pilot24-q8-fivefit-validation-round"
            or eval_lock["version"] != 2
            or eval_lock["source_sha256"]
            != sha256(EVAL / "source/evaluate.py")
            or eval_lock["metric_source_sha256"]
            != sha256(HERE / "pilot_metrics.py")
            or eval_lock["fit_round_sha256"] != sha256(FIT / "round.lock.json")
            or eval_lock["gpu_binary_sha256"]
            != fit_lock["gpu_binary_sha256"]
            or eval_lock["sample_seeds"] != list(SEEDS)
            or eval_lock["size_multipliers"] != list(SIZES)
            or eval_lock["metric_sizes"] != list(METRIC_SIZES)
            or len(eval_lock["fits"]) != 10 or len(eval_lock["jobs"]) != 120
            or preflight["status"] != "failed_before_validation_lock"
            or preflight["round_lock_created"] is not False
            or preflight["official_tests_opened"] is not False
            or preflight_ledger["format"]
            != "dope-pilot24-q8-fivefit-validation-preflight-ledger"
            or len(preflight_ledger["events"]) != 3
            or any(event["gpu_fit_started"] is not False
                   or event["metric_started"] is not False
                   for event in preflight_ledger["events"])
            or preflight_ledger["events"][1]["failure_receipt_sha256"]
            != sha256(FAILED_FREEZE)
            or any(preflight_ledger["events"][index]["source_sha256"]
                   != sha256(ROOT / "dope-q8-fivefit-validation-v1/source/evaluate.py")
                   for index in (0, 1))
            or preflight_ledger["events"][2]["source_sha256"]
            != eval_lock["source_sha256"]
            or preflight_ledger["events"][2]["round_lock_sha256"]
            != sha256(EVAL / "round.lock.json")
            or preflight_ledger["official_tests_opened"] is not False
            or prior["official_tests_opened"] is not False
            or any(x["official_tests_opened"] is not False
                   or x["mfs_v2"] is not None or x["ptf_v1"] is not None
                   for x in (fit_lock, eval_lock, prior))):
        raise ValueError("q8 five-fit source, lock, or test seal changed")
    for name, expected in package["files"].items():
        if sha256(package_root / "research/benchmark" / name) != expected:
            raise ValueError("q8 runtime package changed")
    expected_jobs = {(dataset, fit, sample, size)
                     for dataset in DATASETS for fit in FITS
                     for sample in SEEDS for size in SIZES}
    if {(row["dataset"], row["fit_seed"], row["sample_seed"],
         row["size_multiplier"]) for row in eval_lock["jobs"]} != expected_jobs:
        raise ValueError("q8 frozen sample matrix changed")
    fit_rows = []
    fit_index = {}
    for fit in eval_lock["fits"]:
        key = (fit["dataset"], fit["fit_seed"])
        if key in fit_index:
            raise ValueError("duplicate q8 fit")
        worker_root = ROOT / "prepared-v2/worker" / fit["dataset"]
        worker = read(worker_root / "worker-manifest.json")
        dispatch_root = FIT / "dispatch" / fit["dataset"] / str(fit["fit_seed"])
        dispatch_paths = [path for path in dispatch_root.glob("attempt-*.json")
                          if not path.name.endswith(".inventory.json")]
        successful = [path for path in dispatch_paths if read(path)["status"] == "ok"]
        if len(successful) != 1:
            raise ValueError("q8 fit success count changed")
        dispatch_path = successful[0]
        dispatch = read(dispatch_path)
        inventory_path = dispatch_path.with_suffix(".inventory.json")
        inventory = read(inventory_path)
        fit_path = Path(fit["fit_receipt_path"])
        receipt = read(fit_path)
        model = Path(fit["model_path"])
        compile_path = fit_path.with_suffix(".dpk.report.json")
        compile_report = read(compile_path)
        if (fit["dispatch_receipt_sha256"] != sha256(dispatch_path)
                or dispatch["round_lock_sha256"] != sha256(FIT / "round.lock.json")
                or dispatch["source_sha256"] != fit_lock["source_sha256"]
                or dispatch["child"]["receipt_sha256"] != sha256(fit_path)
                or dispatch["job"] not in fit_lock["jobs"]
                or dispatch["host"] != dispatch["job"]["host"]
                or dispatch["inventory_sha256"] != sha256(inventory_path)
                or inventory["host"].split(".")[0] != dispatch["host"]
                or inventory["active_gpu_processes"]
                or inventory["gpus"][0]["memory_free_mib"] < 17 * 1024
                or not set(range(16, 32)).issubset(inventory["allowed_cpus"])
                or receipt["status"] != "ok"
                or receipt["identity"]["dataset"] != fit["dataset"]
                or receipt["identity"]["seed"] != fit["fit_seed"]
                or receipt["identity"]["candidate"]
                != "compact_neural_residual_symbolic"
                or receipt["identity"]["target_weight"] != 2.0
                or receipt["identity"]["structural_penalty"] != 0.0
                or receipt["identity"]["round_lock_sha256"]
                != sha256(FIT / "round.lock.json")
                or receipt["identity"]["binary_sha256"]
                != fit_lock["gpu_binary_sha256"]
                or receipt["identity"]["train_sha256"]
                != fit["train_sha256"]
                or receipt["identity"]["validation_sha256"]
                != fit["validation_sha256"]
                or fit["fit_receipt_sha256"] != sha256(fit_path)
                or fit["model_sha256"] != sha256(model)
                or receipt["artifact_sha256"] != fit["model_sha256"]
                or receipt["artifact_bytes"] != fit["artifact_bytes"]
                or fit["artifact_bytes"]
                != model.stat().st_size + (worker_root / "projection.json").stat().st_size
                or fit["artifact_bytes"] > CAP
                or receipt["elapsed_seconds"] > 600
                or receipt["peak_gpu_used_mib"] > 16 * 1024
                or compile_report["selected_candidate"]
                != "compact_neural_residual_symbolic"
                or compile_report["compliant"] is not False
                or worker["dataset_id"] != fit["dataset"]
                or worker["train_rows"] != fit["train_rows"]
                or (worker_root / "test.csv").exists()
                or sha256(worker_root / "train.csv") != fit["train_sha256"]
                or sha256(worker_root / "validation.csv")
                != fit["validation_sha256"]
                or sha256(worker_root / "projection.json")
                != fit["projection_sha256"]):
            raise ValueError("q8 fit or byte charge changed")
        fit_index[key] = fit
        fit_rows.append({"dataset": fit["dataset"], "fit_seed": fit["fit_seed"],
                         "host": dispatch["host"],
                         "artifact_bytes": fit["artifact_bytes"],
                         "model_sha256": fit["model_sha256"],
                         "fit_receipt_path": str(fit_path),
                         "fit_receipt_sha256": fit["fit_receipt_sha256"],
                         "dispatch_receipt_path": str(dispatch_path),
                         "dispatch_receipt_sha256": sha256(dispatch_path),
                         "host_inventory_sha256": sha256(inventory_path),
                         "compile_report_sha256": sha256(compile_path),
                         "elapsed_seconds": receipt["elapsed_seconds"],
                         "peak_gpu_used_mib": receipt["peak_gpu_used_mib"],
                         "energy_joules_estimate": receipt["energy_joules_estimate"],
                         "compiler_failed_gates": compile_report["failed_gates"]})
    cells = []
    for job in eval_lock["jobs"]:
        fit = fit_index[(job["dataset"], job["fit_seed"])]
        if (job["track"] != "common_numeric"
                or job["configuration"] != "q8_symbolic"
                or job["candidate"] != "compact_neural_residual_symbolic"
                or job["target_weight"] != 2.0
                or job["structural_penalty"] != 0.0
                or job["fit_receipt_sha256"] != fit["fit_receipt_sha256"]
                or job["model_sha256"] != fit["model_sha256"]
                or job["row_count"] != fit["train_rows"] * job["size_multiplier"]):
            raise ValueError("q8 sample job changed")
        cell_root = EVAL / "cells" / digest(job)
        receipts = sorted(path for path in cell_root.glob("attempt-*.json")
                          if not path.name.endswith(".reservation.json"))
        if not receipts:
            raise ValueError("q8 sample job has no attempt receipt")
        for path in receipts:
            row = read(path)
            if (row["job"] != job
                    or row["round_lock_sha256"] != sha256(EVAL / "round.lock.json")
                    or row["official_tests_opened"] is not False
                    or row["mfs_v2"] is not None or row["ptf_v1"] is not None):
                raise ValueError("q8 sample attempt lineage changed")
        latest = read(receipts[-1])
        if latest["status"] != "ok" or len(receipts) != 1:
            raise ValueError("q8 sample failure requires explicit publisher revision")
        evidence = cell_root / receipts[-1].stem
        sample = evidence / "sample.csv"
        metric_path = evidence / "metrics.json"
        if (latest["fit_receipt_sha256"] != fit["fit_receipt_sha256"]
                or latest["model_sha256"] != fit["model_sha256"]
                or latest["artifact_bytes"] != fit["artifact_bytes"]
                or latest["sample_sha256"] != sha256(sample)
                or latest["repeat_sha256"] != latest["sample_sha256"]):
            raise ValueError("q8 deterministic sample or artifact changed")
        metric = None
        if job["size_multiplier"] in METRIC_SIZES:
            metric = read(metric_path)
            if (latest["metric_sha256"] != sha256(metric_path)
                    or latest["metric_replay_status"] != "exact"
                    or metric["implementation_sha256"]
                    != eval_lock["metric_source_sha256"]
                    or metric["rows"]["synthetic"] != job["row_count"]
                    or metric["gate_profile_complete"] is not False
                    or metric["mfs_v2"] is not None):
                raise ValueError("q8 metric or replay changed")
        elif latest["metric_sha256"] is not None:
            raise ValueError("unexpected q8 2n/8n metric")
        auditor = metric["utility"] if metric else None
        cells.append({"dataset": job["dataset"],
                      "fit_seed": job["fit_seed"],
                      "sample_seed": job["sample_seed"],
                      "size_multiplier": job["size_multiplier"],
                      "status": latest["status"],
                      "artifact_bytes": fit["artifact_bytes"],
                      "within_l3_bytes": True,
                      "catboost_retention": auditor["catboost"]["retention"]
                      if auditor else None,
                      "linear_retention": auditor["linear"]["retention"]
                      if auditor else None,
                      "mlp_retention": auditor["mlp"]["retention"]
                      if auditor else None,
                      "c2st_auc": metric["c2st_auc"] if metric else None,
                      "exact_row_copies": metric["copy_counts"]["exact"]
                      if metric else None,
                      "near_row_copies": metric["copy_counts"]["near"]
                      if metric else None,
                      "real_vs_real_exact":
                      metric["real_vs_real_control_counts"]["exact"]
                      if metric else None,
                      "real_vs_real_near":
                      metric["real_vs_real_control_counts"]["near"]
                      if metric else None,
                      "synthetic_near_rate":
                      metric["copy_counts"]["near"] / metric["rows"]["synthetic"]
                      if metric else None,
                      "real_control_near_rate":
                      metric["real_vs_real_control_counts"]["near"]
                      / metric["rows"]["validation"] if metric else None,
                      "sample_sha256": latest["sample_sha256"],
                      "sample_receipt_path": str(receipts[-1]),
                      "sample_receipt_sha256": sha256(receipts[-1]),
                      "metric_receipt_path": str(metric_path) if metric else None,
                      "metric_receipt_sha256": sha256(metric_path)
                      if metric else None,
                      "metric_replay_status": latest["metric_replay_status"],
                      "release_safe": None})
    if len(cells) != 120:
        raise ValueError("q8 sample matrix incomplete")
    prior_fit23 = {(row["dataset"], row["sample_seed"],
                    row["size_multiplier"]): row
                   for row in prior["cells"]
                   if row["method"] == "DOPE"
                   and row["configuration"] == "q8_symbolic"
                   and row["fit_seed"] == 23
                   and row["dataset"] in DATASETS
                   and row["size_multiplier"] in METRIC_SIZES}
    replicated = [row for row in cells if row["fit_seed"] == 23
                  and row["size_multiplier"] in METRIC_SIZES]
    if (len(prior_fit23) != 12 or len(replicated) != 12
            or any(row["catboost_retention"]
                   != prior_fit23[(row["dataset"], row["sample_seed"],
                                   row["size_multiplier"])]["catboost_retention"]
                   for row in replicated)):
        raise ValueError("q8 prior fit-23 validation values failed exact replay")
    summary = []
    for dataset in DATASETS:
        for seed in FITS:
            for size in METRIC_SIZES:
                rows = [row for row in cells if row["dataset"] == dataset
                        and row["fit_seed"] == seed
                        and row["size_multiplier"] == size]
                if len(rows) != 3 or any(row["catboost_retention"] is None for row in rows):
                    raise ValueError("q8 fit summary incomplete")
                summary.append({"dataset": dataset, "fit_seed": seed,
                                "size_multiplier": size,
                                "median_catboost_retention": median(
                                    row["catboost_retention"] for row in rows),
                                "min_catboost_retention": min(
                                    row["catboost_retention"] for row in rows),
                                "max_catboost_retention": max(
                                    row["catboost_retention"] for row in rows),
                                "exact_row_copies": sum(row["exact_row_copies"] for row in rows),
                                "near_row_copies": sum(row["near_row_copies"] for row in rows)})
    dataset_summary = []
    for dataset in DATASETS:
        for size in METRIC_SIZES:
            rows = [row for row in summary if row["dataset"] == dataset
                    and row["size_multiplier"] == size]
            dataset_summary.append({"dataset": dataset, "size_multiplier": size,
                                    "fit_seeds": 5, "sample_seeds_per_fit": 3,
                                    "median_of_fit_medians_catboost_retention": median(
                                        row["median_catboost_retention"] for row in rows),
                                    "min_fit_median_catboost_retention": min(
                                        row["median_catboost_retention"] for row in rows),
                                    "max_fit_median_catboost_retention": max(
                                        row["median_catboost_retention"] for row in rows)})
    energies = [row["energy_joules_estimate"] for row in fit_rows
                if row["energy_joules_estimate"] is not None]
    fit_cost = {"total_elapsed_seconds": sum(row["elapsed_seconds"] for row in fit_rows),
                "energy_joules_estimate_observed": sum(energies),
                "energy_receipts": len(energies),
                "peak_gpu_used_mib": max(row["peak_gpu_used_mib"] for row in fit_rows),
                "artifact_bytes_min": min(row["artifact_bytes"] for row in fit_rows),
                "artifact_bytes_max": max(row["artifact_bytes"] for row in fit_rows),
                "fits_within_l3_bytes": sum(row["artifact_bytes"] <= CAP for row in fit_rows)}
    audit = verify_audit(eval_lock)
    return {"format": "dope-pilot24-q8-fivefit-validation", "version": 1,
            "scope": "training_derived_validation_only",
            "source_sha256": sha256(Path(__file__)),
            "fit_round_sha256": sha256(FIT / "round.lock.json"),
            "validation_round_sha256": sha256(EVAL / "round.lock.json"),
            "preflight_failure_receipt_sha256": sha256(FAILED_FREEZE),
            "preflight_ledger_sha256": sha256(PREFLIGHT_LEDGER),
            "preflight_events": preflight_ledger["events"],
            "prior_native_neural_report_sha256": sha256(PRIOR),
            "fit_attempts": fit_rows, "fit_cost": fit_cost, "cells": cells,
            "fit_summary": summary, "dataset_summary": dataset_summary,
            "stratified_replay": audit,
            "status_counts": {"fit_ok": 10, "sample_ok": 120,
                              "metric_replay_exact": 60,
                              "prior_fit23_value_replays_exact": 12,
                              "stratified_replays_exact": 16,
                              "nonfit_nonmetric_preflight_failures": 3},
            "notes": ["This fixed q8 replication follows earlier validation-only family selection and uses five fit seeds on each pilot dataset. It is not an independent dataset confirmation or the preregistered public-test analysis.",
                      "Adult fits showed device-memory use; some California compiler fits had negligible GPU use. The source-pinned candidate is reported by its selected family without claiming every artifact is a trained neural generator.",
                      "Near-copy counts use a fixed normalized RMS threshold; real-vs-real controls can also match near training rows. The complete privacy gate requires calibrated attacks and remains unmeasured.",
                      "Existing CTGAN/TVAE native-tuned comparisons use the same training-derived datasets but only one fit seed, so their paired outcomes remain a separate panel.",
                      "All twelve fit-seed-23 CatBoost retention values at n/4n exactly reproduce the previous published q8 pilot values on Adult and California.",
                      "A separately frozen 16-cell audit replayed both datasets, the first and last fit and sample seeds, and n/4n. Sample hashes and metric payloads matched exactly, excluding only elapsed metric time.",
                      "All official tests remain sealed. Required privacy attacks, complete evaluator gates, and product coverage are absent; MFS-v2, PTF-v1, release safety and certification remain null."],
            "official_tests_opened": False, "mfs_v2": None,
            "ptf_v1": None, "production_certified": False}


def render(report: dict) -> tuple[str, str]:
    buffer = io.StringIO()
    fields = ("dataset", "fit_seed", "sample_seed", "size_multiplier",
              "status", "artifact_bytes", "catboost_retention",
              "linear_retention", "mlp_retention", "exact_row_copies",
              "near_row_copies", "real_vs_real_exact", "real_vs_real_near",
              "synthetic_near_rate", "real_control_near_rate",
              "sample_receipt_sha256",
              "metric_receipt_sha256")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n",
                            extrasaction="ignore")
    writer.writeheader()
    writer.writerows(report["cells"])
    lines = ["# DOPE q8: five fit seeds on Adult and California", "",
             "Fixed q8 configuration, five fits per dataset, three sample seeds, "
             "and n/2n/4n/8n output sizes. The shared CatBoost auditor was "
             "measured and replayed at n and 4n on training-derived validation. "
             "Official tests stayed sealed.", "",
             "| Dataset | Fit seed | n median retention | 4n median retention | Charged bytes |",
             "|---|---:|---:|---:|---:|"]
    for fit in report["fit_attempts"]:
        dataset, seed = fit["dataset"], fit["fit_seed"]
        one = next(row for row in report["fit_summary"]
                   if row["dataset"] == dataset and row["fit_seed"] == seed
                   and row["size_multiplier"] == 1)
        four = next(row for row in report["fit_summary"]
                    if row["dataset"] == dataset and row["fit_seed"] == seed
                    and row["size_multiplier"] == 4)
        lines.append(f"| {dataset} | {seed} | "
                     f"{one['median_catboost_retention']:.4f} | "
                     f"{four['median_catboost_retention']:.4f} | "
                     f"{fit['artifact_bytes']:,} |")
    lines += ["", "Median across fit-seed medians:", "",
              "| Dataset | n | 4n |", "|---|---:|---:|"]
    for dataset in DATASETS:
        one = next(row for row in report["dataset_summary"]
                   if row["dataset"] == dataset and row["size_multiplier"] == 1)
        four = next(row for row in report["dataset_summary"]
                    if row["dataset"] == dataset and row["size_multiplier"] == 4)
        lines.append(f"| {dataset} | "
                     f"{one['median_of_fit_medians_catboost_retention']:.4f} | "
                     f"{four['median_of_fit_medians_catboost_retention']:.4f} |")
    cost = report["fit_cost"]
    lines += ["", "## Fit costs and launch accounting", "",
              f"Ten successful fits consumed {cost['total_elapsed_seconds']:.2f} "
              "seconds of summed fit elapsed time under a frozen 6,000-second "
              "ceiling. Every fit stayed below 600 seconds and 16 GiB. "
              "These costs exclude earlier architecture research and metric evaluation.", "",
              "| Dataset | Fit seed | Host | Fit seconds | Peak GPU MiB | Estimated joules |",
              "|---|---:|---|---:|---:|---:|"]
    for fit in report["fit_attempts"]:
        energy = fit["energy_joules_estimate"]
        energy_text = f"{energy:.2f}" if energy is not None else "unavailable"
        lines.append(f"| {fit['dataset']} | {fit['fit_seed']} | {fit['host']} | "
                     f"{fit['elapsed_seconds']:.2f} | {fit['peak_gpu_used_mib']} | "
                     f"{energy_text} |")
    lines += ["", "Three preflight launch failures started no GPU fit or metric "
              "evaluation. Their source and failure receipts remain linked in the JSON.", "",
              "| Phase | Error | Cause |", "|---|---|---|"]
    for event in report["preflight_events"]:
        lines.append(f"| {event['phase']} | {event['error_type']} | {event['reason']} |")
    lines += ["", "All 120 sample cells matched repeat hashes; all 60 n/4n metric "
              "payloads replayed exactly. A separate frozen stratified audit "
              "replayed 16 cells exactly, excluding only metric elapsed time."]
    lines += ["", "Near-match rates use the fixed 1e-3 normalized RMS pilot "
              "threshold. The real-vs-real control uses held-out validation "
              "rows against training rows; rates are median per cell.", "",
              "| Dataset | Size | Synthetic near-match rate | Real control near-match rate |",
              "|---|---:|---:|---:|"]
    for dataset in DATASETS:
        for size in METRIC_SIZES:
            rows = [row for row in report["cells"]
                    if row["dataset"] == dataset
                    and row["size_multiplier"] == size]
            lines.append(f"| {dataset} | {size}n | "
                         f"{median(row['synthetic_near_rate'] for row in rows):.6f} | "
                         f"{median(row['real_control_near_rate'] for row in rows):.6f} |")
    lines += ["", "These are validation-only stability estimates. The existing "
              "native-tuned CTGAN and TVAE panel uses one fit seed and is "
              "reported separately. Every q8 artifact fits the 10,240-byte "
              "L3 byte cap, but other release gates are incomplete or failed. "
              "MFS-v2, PTF-v1, and certification are null.", ""]
    return buffer.getvalue(), "\n".join(lines)


def main() -> None:
    from jsonschema import Draft202012Validator

    parser = argparse.ArgumentParser()
    parser.add_argument("--from-json", action="store_true")
    args = parser.parse_args()
    report = read(RESULT) if args.from_json else build()
    schema = read(RESULT.with_suffix(".schema.json"))
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(report)
    if (report["format"] != "dope-pilot24-q8-fivefit-validation"
            or report["source_sha256"] != sha256(Path(__file__))
            or report["official_tests_opened"] is not False
            or report["mfs_v2"] is not None or report["ptf_v1"] is not None):
        raise ValueError("q8 publication changed")
    if not args.from_json:
        RESULT.write_text(json.dumps(report, sort_keys=True, indent=2,
                                     allow_nan=False) + "\n")
    csv_text, markdown = render(report)
    RESULT.with_suffix(".csv").write_text(csv_text)
    RESULT.with_suffix(".md").write_text(markdown)
    print(json.dumps(report["status_counts"], sort_keys=True))


if __name__ == "__main__":
    main()
