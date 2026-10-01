"""Five-fit matched validation outcomes; native selections remain unchanged."""

from __future__ import annotations
import argparse, csv, io, json, math
import re
from collections import Counter
from pathlib import Path
from statistics import median
from research.benchmark import pilot24_neural_comparison as prior_checks
from research.benchmark import publish_dope_q8_fivefit as dope_checks
from research.benchmark.manifest import digest
from research.benchmark.score import sha256, artifact_inventory

HERE = Path(__file__).resolve().parent
ROOT = Path("/mnt/fast-scratch/dope-benchmark/pilot-24h")
FIT = ROOT / "sdv-fivefit-v1"
EVAL = ROOT / "sdv-fivefit-validation-v1"
DATASETS = ("Adult", "California")
FIT_SEEDS = (11, 23, 37, 53, 71)
SAMPLE_SEEDS = (101, 211, 307)
SIZES = (1, 2, 4, 8)
AUDITORS = ("catboost", "linear", "mlp")
COMPARATOR_CONFIGS = {
    "Adult": (("CTGAN", "default"), ("CTGAN", "tuned"), ("TVAE", "default_and_tuned")),
    "California": (
        ("CTGAN", "default_and_tuned"),
        ("TVAE", "default"),
        ("TVAE", "tuned"),
    ),
}
PINS = {
    "Adult": "df6594af43ff4ba357beeffa8b9728019466b3c6e6498ffe460128f09a77880f",
    "California": "75f41549d3c5e7658bb00b6f3aedcf033cbf9c8ee6d17e82e22c647f899583bb",
    "evaluation": "1d419a59ce75ebae58902e7dff0d769e46101b26b5ac0b81a2b0bc583197442b",
}


def read(path):
    return json.loads(Path(path).read_text())


def sealed(value):
    if (
        value.get("official_tests_opened") is not False
        or value.get("mfs_v2") is not None
        or value.get("ptf_v1") is not None
    ):
        raise ValueError("test seal or score changed")


def evidence(path, receipt):
    for name, h in receipt["evidence_files"].items():
        p = Path(path) / name
        if (
            p.is_symlink()
            or not p.resolve().is_relative_to(Path(path).resolve())
            or sha256(p) != h
        ):
            raise ValueError("immutable evidence changed")


def fit_row(lock_path, job):
    lock = read(lock_path)
    paths = sorted(
        (lock_path.parent / "jobs" / digest(job)).glob("attempt-*/receipt.json")
    )
    if not paths:
        raise ValueError("fixed fit pending")
    rows = []
    for path in paths:
        receipt = read(path)
        if receipt["job"] != job or receipt["round_sha256"] != sha256(lock_path):
            raise ValueError("fit/retry identity changed")
        evidence(path.parent, receipt)
        if (
            receipt["validation_only"] is not True
            or receipt["mfs_v2"] is not None
            or receipt["ptf_v1"] is not None
        ):
            raise ValueError("fit is not validation only")
        host = read(path.parent / "host.json")
        if (
            host["active_gpu_processes"]
            or host["gpus"][0]["memory_free_mib"] < 17 * 1024
            or host["host"] != receipt["host"]
            or not set(receipt["cpu_affinity"]).issubset(host["allowed_cpus"])
        ):
            raise ValueError("fit admission changed")
        fitted = receipt["fit"]
        if (receipt["status"] == "ok" and fitted["wall_seconds"] > 600) or fitted[
            "peak_device_used_mib"
        ] > 16384:
            raise ValueError("fit cap exceeded")
        charged = receipt.get("artifact_bytes")
        if receipt["status"] == "ok":
            inventory, actual = artifact_inventory(
                path.parent / "artifact", ["model.pt", "model.json", "projection.json"]
            )
            if actual != charged or inventory != receipt["artifact_inventory"]:
                raise ValueError("artifact charge changed")
            if (
                sha256(path.parent / "artifact/projection.json")
                != job["projection_sha256"]
            ):
                raise ValueError("artifact projection changed")
        rows.append(
            {
                "dataset": job["dataset"],
                "method": job["method"],
                "configuration": job["track"],
                "fit_seed": job["fit_seed"],
                "job_digest": digest(job),
                "config_sha256": digest(job["config"]),
                "job": job,
                "attempt": path.parent.name,
                "status": receipt["status"],
                "native_kpi": receipt.get("native_kpi"),
                "artifact_bytes": charged,
                "within_l3_bytes": charged <= 10240 if charged is not None else None,
                "fit_seconds": fitted["wall_seconds"],
                "total_attempt_seconds": receipt["wall_seconds"],
                "peak_device_used_mib": fitted["peak_device_used_mib"],
                "energy_joules_estimate": fitted["energy_joules_estimate"],
                "host": receipt["host"],
                "error_type": receipt.get("error_type"),
                "fit_receipt_path": str(path),
                "fit_receipt_sha256": sha256(path),
                "round_lock_sha256": sha256(lock_path),
                "release_safe": None,
            }
        )
    return rows


def measured_cell(job, receipt, path, metric):
    sealed(metric) if "official_tests_opened" in metric else None
    if (
        metric["mfs_v2"] is not None
        or metric["gate_profile_complete"] is not False
        or metric["implementation_sha256"] != sha256(HERE / "pilot_metrics.py")
        or metric["rows"]["synthetic"] != job["row_count"]
    ):
        raise ValueError("common metric implementation changed")
    return {
        "dataset": job["fit_job"]["dataset"],
        "method": job["fit_job"]["method"],
        "configuration": job["fit_job"]["track"],
        "fit_seed": job["fit_job"]["fit_seed"],
        "sample_seed": job["sample_seed"],
        "size_multiplier": job["size_multiplier"],
        "status": "ok",
        "artifact_bytes": receipt["artifact_bytes"],
        "within_l3_bytes": receipt["artifact_bytes"] <= 10240,
        "release_safe": None,
        "fit_receipt_sha256": receipt["fit_receipt_sha256"],
        "sample_receipt_path": str(path),
        "sample_receipt_sha256": sha256(path),
        "metric_receipt_sha256": receipt["metric_sha256"],
        "metric_replay_status": receipt["metric_replay_status"],
        "sample_sha256": receipt["sample_sha256"],
        "synthetic_rows": metric["rows"]["synthetic"],
        "validation_rows": metric["rows"]["validation"],
        "train_rows": metric["rows"]["train"],
        "marginal_ks_mean": metric["marginal_ks_mean"],
        "pair_correlation_fidelity": metric["pair_correlation_fidelity"],
        "null_loss": metric["null_loss"],
        "utility": metric["utility"],
        "c2st_auc": metric["c2st_auc"],
        "copy_counts": metric["copy_counts"],
        "real_vs_real_control_counts": metric["real_vs_real_control_counts"],
    }


def dope_research_cost():
    """Account completed pilot research fits separately from fixed replications."""
    rounds = {
        "gpu-v3": 6,
        "gpu-refine-v5": 6,
        "gpu-neural-v1": 12,
        "dope-conditional-v1": 24,
        "dope-news-autoreg-confirm-v2": 1,
        "dope-news-q10-v1": 1,
        "dope-news-q10-multiseed-v1": 3,
        "california-q10-research-v1": 2,
    }
    lock_paths = list(ROOT.glob("*lock.json")) + list(ROOT.glob("*/round.lock.json"))
    locks = {sha256(path): path for path in lock_paths}
    attempts = []
    for stage, expected in rounds.items():
        paths = sorted(
            path
            for path in (ROOT / stage / "fits").glob("*/attempt-*.json")
            if re.fullmatch(r"attempt-\d+\.json", path.name)
        )
        if len(paths) != expected:
            raise ValueError("prior DOPE pilot cost matrix changed")
        for path in paths:
            row = read(path)
            identity = row["identity"]
            lock_path = locks.get(identity["round_lock_sha256"])
            if (
                row["format"] != "dope-benchmark-gpu-research-attempt"
                or row["validation_only"] is not True
                or row["ptf_v1"] is not None
                or lock_path is None
                or identity["fit_deadline_seconds"] != 600
                or not math.isfinite(row["elapsed_seconds"])
                or row["elapsed_seconds"] < 0
                or sha256(path.with_suffix(".log")) != row["log_sha256"]
            ):
                raise ValueError("prior DOPE pilot cost receipt changed")
            if row["status"] == "ok":
                artifact = path.with_suffix(".dpk")
                projection = (
                    ROOT
                    / "prepared-v2/worker"
                    / identity["dataset"]
                    / "projection.json"
                )
                if (
                    sha256(artifact) != row["artifact_sha256"]
                    or sha256(projection) != identity["projection_sha256"]
                    or artifact.stat().st_size + projection.stat().st_size
                    != row["artifact_bytes"]
                ):
                    raise ValueError("prior DOPE research artifact changed")
            attempts.append(
                {
                    "stage": stage,
                    "dataset": identity["dataset"],
                    "candidate": identity["candidate"],
                    "fit_seed": identity["seed"],
                    "status": row["status"],
                    "failure_code": row["failure_code"],
                    "host": row["host"],
                    "elapsed_seconds": row["elapsed_seconds"],
                    "energy_joules_estimate": row["energy_joules_estimate"],
                    "peak_gpu_used_mib": row["peak_gpu_used_mib"],
                    "receipt_path": str(path),
                    "receipt_sha256": sha256(path),
                    "round_lock_path": str(lock_path),
                    "round_lock_sha256": identity["round_lock_sha256"],
                    "binary_sha256": identity["binary_sha256"],
                    "probe_source_sha256": identity["probe_source_sha256"],
                }
            )
    return {
        "scope": "completed_named_pilot_research_fit_rounds_only",
        "total_elapsed_seconds": sum(row["elapsed_seconds"] for row in attempts),
        "fit_attempts": len(attempts),
        "status_counts": dict(Counter(row["status"] for row in attempts)),
        "all_campaign_r_and_d_total_seconds": None,
        "all_campaign_r_and_d_accounting_complete": False,
        "attempts": attempts,
        "notes": [
            "These prior pilot research rounds are charged separately from the ten fixed q8 fits.",
            "This scope includes News and California follow-up fits; it does not claim equal total research spend.",
            "Earlier S3 discovery/confirmation, CPU development, source audits and other campaign costs require the separate complete campaign ledger.",
        ],
    }


def prior_samples(fit_index):
    """Reconcile every original fit-23 size, including failed-fit identities."""
    rows = []
    for group in ("Adult", "regression"):
        lock_path = ROOT / "sdv-final-samples-v1" / group / "round.lock.json"
        lock = read(lock_path)
        sealed(lock)
        for job in lock["jobs"]:
            if job["dataset"] not in DATASETS:
                continue
            fit = fit_index[job["dataset"], job["method"], job["track"], 23]
            path = (
                lock_path.parent
                / "jobs"
                / prior_checks.digest(job)
                / "attempt-0001/receipt.json"
            )
            receipt = read(path)
            sealed(receipt)
            sample = prior_checks.within_any(
                ROOT,
                ("sdv-final-samples-v1", "sdv-final-seed23-v1"),
                receipt["sample_path"],
            )
            process_path = (
                Path(fit["fit_receipt_path"]).parent
                if receipt["inherited_from_fit"]
                else path.parent
            ) / "sample.receipt.json"
            process = read(process_path)
            if (
                receipt["job"] != job
                or receipt["round_sha256"] != sha256(lock_path)
                or receipt["status"] != "ok"
                or process["status"] != "ok"
                or fit["status"] != "ok"
                or job["fit_receipt_sha256"] != fit["fit_receipt_sha256"]
                or job["config_sha256"] != fit["config_sha256"]
                or job["artifact_bytes"] != fit["artifact_bytes"]
                or job["row_count"] != fit["job"]["train_rows"] * job["size_multiplier"]
                or any(
                    job[f"{part}_sha256"] != fit["job"][f"{part}_sha256"]
                    for part in ("train", "validation")
                )
                or sha256(process_path) != receipt["sample_process_receipt_sha256"]
                or sha256(sample) != receipt["sample_sha256"]
                or process["child"]["sha256"] != receipt["sample_sha256"]
                or process["child"]["rows"] != job["row_count"]
            ):
                raise ValueError("prior wider sample lineage changed")
            for name in ("stdout", "stderr"):
                if (
                    sha256(process_path.with_name(f"sample.{name}"))
                    != process[f"{name}_sha256"]
                ):
                    raise ValueError("prior wider sample process evidence changed")
            rows.append(
                {
                    "job": job,
                    "status": "ok",
                    "receipt_path": str(path),
                    "receipt_sha256": sha256(path),
                    "sample_sha256": receipt["sample_sha256"],
                    "sample_process_receipt_sha256": sha256(process_path),
                    "fit_receipt_sha256": fit["fit_receipt_sha256"],
                    "round_lock_sha256": sha256(lock_path),
                    "source_unavailable": False,
                    "contributes_dope_win": False,
                }
            )
    for fit in fit_index.values():
        if fit["fit_seed"] != 23 or fit["status"] == "ok":
            continue
        for size in SIZES:
            for seed in SAMPLE_SEEDS:
                rows.append(
                    {
                        "job": {
                            "dataset": fit["dataset"],
                            "method": fit["method"],
                            "track": fit["configuration"],
                            "fit_seed": 23,
                            "sample_seed": seed,
                            "size_multiplier": size,
                        },
                        "status": "fit_unavailable",
                        "receipt_path": None,
                        "receipt_sha256": None,
                        "sample_sha256": None,
                        "sample_process_receipt_sha256": None,
                        "fit_receipt_sha256": fit["fit_receipt_sha256"],
                        "round_lock_sha256": fit["round_lock_sha256"],
                        "source_unavailable": False,
                        "contributes_dope_win": False,
                    }
                )
    return rows


def sample_matrix_counts(new_samples, prior_samples):
    keys = []
    for row in new_samples + prior_samples:
        job = row["job"]
        fit = job.get("fit_job", job)
        keys.append(
            (
                fit["dataset"],
                fit["method"],
                fit["track"],
                fit["fit_seed"],
                job["sample_seed"],
                job["size_multiplier"],
            )
        )
    expected = {
        (d, m, c, f, s, z)
        for d in DATASETS
        for m, c in COMPARATOR_CONFIGS[d]
        for f in FIT_SEEDS
        for s in SAMPLE_SEEDS
        for z in SIZES
    }
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("complete five-fit comparator sample matrix differs")
    return dict(Counter(row["status"] for row in new_samples + prior_samples))


def build():
    eval_path = EVAL / "round.lock.json"
    lock = read(eval_path)
    sealed(lock)
    if (
        sha256(eval_path) != PINS["evaluation"]
        or lock["source_sha256"] != sha256(EVAL / "source/evaluate.py")
        or lock["metric_source_sha256"] != sha256(HERE / "pilot_metrics.py")
        or lock["selection_policy"] != "no_tuning_fixed_configs_only"
        or len(lock["jobs"]) != 288
    ):
        raise ValueError("SDV evaluation lock changed")
    dope = dope_checks.build()
    published_dope = read(HERE / "results/pilot24-dope-q8-fivefit.json")
    if dope != published_dope:
        raise ValueError("published DOPE proof no longer regenerates")
    selections = prior_checks.selection_receipts(ROOT)
    old_cells, old_failures = prior_checks.native_metric_rows(ROOT, selections)
    published_prior = read(HERE / "results/pilot24-native-neural.json")
    prior_subset = [r for r in published_prior["cells"] if r["method"] != "DOPE"]
    if sorted(old_cells, key=lambda r: digest(r)) != sorted(
        prior_subset, key=lambda r: digest(r)
    ):
        raise ValueError("prior comparator report differs from immutable metrics")
    fit_attempts = []
    fit_index = {}
    worker_hashes = {}
    source_manifests = {}
    for dataset in DATASETS:
        path = FIT / dataset / "round.lock.json"
        parent = read(path)
        sealed(parent)
        if (
            sha256(path) != PINS[dataset]
            or parent["source_sha256"] != sha256(FIT / "source/round.py")
            or parent["production_ptf_v1"] is not None
            or parent["new_tuning_trials"] != 0
            or parent["fixed_fit_seeds_new"] != [11, 37, 53, 71]
            or len(parent["jobs"]) != 12
            or sha256(Path(parent["prior_fixed_round_path"]))
            != parent["prior_fixed_round_sha256"]
            or parent["prior_published_report_sha256"]
            != sha256(HERE / "results/pilot24-native-neural.json")
            or lock["parents"][dataset]["sha256"] != sha256(path)
        ):
            raise ValueError("SDV fixed replication lineage changed")
        for name, h in parent["source_files"].items():
            if sha256(Path(parent["package_root"]) / "research/benchmark" / name) != h:
                raise ValueError("author adapter package changed")
        for name, h in parent["dependency_files"].items():
            if sha256(ROOT.parent / "neural-native-v1/deps" / name) != h:
                raise ValueError("author dependency source changed")
        source_manifests[dataset] = {
            "source_files": parent["source_files"],
            "dependency_files_digest": digest(parent["dependency_files"]),
            "environment": parent["environment"],
            "round_lock_sha256": sha256(path),
            "prior_round_sha256": parent["prior_fixed_round_sha256"],
        }
        worker = Path(parent["worker_root"]) / dataset
        if (worker / "test.csv").exists() or "evaluator" in worker.resolve().parts:
            raise ValueError("worker includes tests")
        worker_hashes[dataset] = {
            n: sha256(worker / n)
            for n in ("train.csv", "validation.csv", "projection.json")
        }
        for source_path in (path, Path(parent["prior_fixed_round_path"])):
            prior = read(source_path)
            for job in prior["jobs"]:
                if job["dataset"] != dataset:
                    continue
                if (
                    any(
                        worker_hashes[dataset][f"{n}.csv"] != job[f"{n}_sha256"]
                        for n in ("train", "validation")
                    )
                    or worker_hashes[dataset]["projection.json"]
                    != job["projection_sha256"]
                ):
                    raise ValueError("fixed input hashes changed")
                rows = fit_row(source_path, job)
                fit_attempts.extend(rows)
                key = (dataset, job["method"], job["track"], job["fit_seed"])
                if key in fit_index:
                    raise ValueError("duplicate fixed fit identity")
                fit_index[key] = rows[-1]
    if len(fit_index) != 30:
        raise ValueError("five-fit comparator matrix incomplete")
    cells = []
    sample_receipts = []
    for job in lock["jobs"]:
        path = EVAL / "cells" / digest(job) / "attempt-0001.json"
        if not path.exists():
            raise ValueError("streaming sample cells remain pending")
        receipt = read(path)
        sealed(receipt)
        evidence(path.parent / path.stem, receipt)
        fit = job["fit_job"]
        row = fit_index[fit["dataset"], fit["method"], fit["track"], fit["fit_seed"]]
        if (
            receipt["job"] != job
            or receipt["round_lock_sha256"] != sha256(eval_path)
            or receipt["fit_receipt_sha256"] != row["fit_receipt_sha256"]
            or receipt["fit_status"] != row["status"]
            or receipt["artifact_bytes"] != row["artifact_bytes"]
        ):
            raise ValueError("streamed sample lineage changed")
        metric_size = job["size_multiplier"] in (1, 4)
        sample_receipts.append(
            {
                "job": job,
                "status": receipt["status"],
                "receipt_path": str(path),
                "receipt_sha256": sha256(path),
                "sample_sha256": receipt["sample_sha256"],
                "metric_sha256": receipt["metric_sha256"],
            }
        )
        if receipt["status"] == "ok":
            if (
                sha256(path.parent / path.stem / "sample.csv")
                != receipt["sample_sha256"]
                or receipt["repeat_sha256"] != receipt["sample_sha256"]
            ):
                raise ValueError("sample replay differs")
            if metric_size:
                metric_path = path.parent / path.stem / "metrics.json"
                if (
                    sha256(metric_path) != receipt["metric_sha256"]
                    or receipt["metric_replay_status"] != "exact"
                ):
                    raise ValueError("metric replay differs")
                metric = read(metric_path)
                cells.append(measured_cell(job, receipt, path, metric))
        elif metric_size:
            if row["status"] == "ok" and receipt["status"] == "fit_unavailable":
                raise ValueError("successful fit mislabeled unavailable")
            cells.append(
                {
                    "dataset": fit["dataset"],
                    "method": fit["method"],
                    "configuration": fit["track"],
                    "fit_seed": fit["fit_seed"],
                    "sample_seed": job["sample_seed"],
                    "size_multiplier": job["size_multiplier"],
                    "status": receipt["status"],
                    "artifact_bytes": row["artifact_bytes"],
                    "within_l3_bytes": row["within_l3_bytes"],
                    "release_safe": None,
                    "fit_receipt_sha256": row["fit_receipt_sha256"],
                    "sample_receipt_path": str(path),
                    "sample_receipt_sha256": sha256(path),
                    "metric_receipt_sha256": None,
                    "metric_replay_status": None,
                    "sample_sha256": receipt["sample_sha256"],
                    "utility": None,
                    "source_unavailable": False,
                    "contributes_dope_win": False,
                }
            )
    # The fifth fit seed is reused only through the already verified fixed rounds.
    for group in ("Adult", "regression"):
        old_lock_path = ROOT / "sdv-final-metrics-v1" / group / "round.lock.json"
        old_lock = read(old_lock_path)
        for job in old_lock["jobs"]:
            if job["dataset"] not in DATASETS:
                continue
            path = old_lock_path.parent / "cells" / f"{digest(job)}.json"
            r = read(path)
            fit = fit_index[job["dataset"], job["method"], job["track"], 23]
            enriched = {
                "fit_job": {
                    "dataset": job["dataset"],
                    "method": job["method"],
                    "track": job["track"],
                    "fit_seed": 23,
                },
                "row_count": job["row_count"],
                "sample_seed": job["sample_seed"],
                "size_multiplier": job["size_multiplier"],
            }
            adapted = {
                "artifact_bytes": r["artifact_bytes"],
                "fit_receipt_sha256": job["fit_receipt_sha256"],
                "metric_sha256": sha256(path),
                "metric_replay_status": "prior_immutable_receipt_verified",
                "sample_sha256": r["sample_sha256"],
            }
            if (
                fit["status"] != "ok"
                or fit["fit_receipt_sha256"] != job["fit_receipt_sha256"]
            ):
                raise ValueError("prior fit metric identity differs")
            cells.append(measured_cell(enriched, adapted, path, r["metrics"]))
    for key, row in fit_index.items():
        if row["fit_seed"] != 23 or row["status"] == "ok":
            continue
        for size in (1, 4):
            for seed in SAMPLE_SEEDS:
                cells.append(
                    {
                        "dataset": row["dataset"],
                        "method": row["method"],
                        "configuration": row["configuration"],
                        "fit_seed": 23,
                        "sample_seed": seed,
                        "size_multiplier": size,
                        "status": "fit_unavailable",
                        "artifact_bytes": row["artifact_bytes"],
                        "within_l3_bytes": row["within_l3_bytes"],
                        "release_safe": None,
                        "fit_receipt_sha256": row["fit_receipt_sha256"],
                        "sample_receipt_path": None,
                        "sample_receipt_sha256": None,
                        "metric_receipt_sha256": None,
                        "metric_replay_status": None,
                        "sample_sha256": None,
                        "utility": None,
                        "source_unavailable": False,
                        "contributes_dope_win": False,
                    }
                )
    for row in dope["cells"]:
        if row["size_multiplier"] not in (1, 4):
            continue
        metric_path = Path(row["metric_receipt_path"])
        metric = read(metric_path)
        if sha256(metric_path) != row["metric_receipt_sha256"]:
            raise ValueError("DOPE metric changed")
        copy = {
            k: row[k]
            for k in (
                "dataset",
                "fit_seed",
                "sample_seed",
                "size_multiplier",
                "status",
                "artifact_bytes",
                "within_l3_bytes",
                "release_safe",
                "sample_receipt_path",
                "sample_receipt_sha256",
                "metric_receipt_sha256",
                "metric_replay_status",
                "sample_sha256",
                "c2st_auc",
            )
        }
        fit = next(
            f
            for f in dope["fit_attempts"]
            if f["dataset"] == row["dataset"] and f["fit_seed"] == row["fit_seed"]
        )
        copy["fit_receipt_sha256"] = fit["fit_receipt_sha256"]
        copy.update(
            {
                "method": "DOPE",
                "configuration": "q8_selected",
                "utility": metric["utility"],
                "null_loss": metric["null_loss"],
                "copy_counts": metric["copy_counts"],
                "real_vs_real_control_counts": metric["real_vs_real_control_counts"],
                "synthetic_rows": metric["rows"]["synthetic"],
                "validation_rows": metric["rows"]["validation"],
                "train_rows": metric["rows"]["train"],
                "marginal_ks_mean": metric["marginal_ks_mean"],
                "pair_correlation_fidelity": metric["pair_correlation_fidelity"],
            }
        )
        cells.append(copy)

    prior_sample_receipts = prior_samples(fit_index)
    all_sample_status_counts = sample_matrix_counts(
        sample_receipts, prior_sample_receipts
    )

    # Shared configurations count once; explicit per-dataset sets reconcile all cells.
    def configs(dataset):
        return [("DOPE", "q8_selected"), *COMPARATOR_CONFIGS[dataset]]

    expected = {
        (d, m, c, f, s, z)
        for d in DATASETS
        for m, c in configs(d)
        for f in FIT_SEEDS
        for s in SAMPLE_SEEDS
        for z in (1, 4)
    }
    keys = [
        (
            r["dataset"],
            r["method"],
            r["configuration"],
            r["fit_seed"],
            r["sample_seed"],
            r["size_multiplier"],
        )
        for r in cells
    ]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("complete matched metric matrix differs")
    return {
        "format": "dope-native-neural-fivefit-matched-validation",
        "version": 1,
        "scope": "training_derived_validation_only",
        "official_tests_opened": False,
        "production_certified": False,
        "mfs_v2": None,
        "ptf_v1": None,
        "source_sha256": sha256(Path(__file__)),
        "input_reports": {
            "dope_q8_sha256": sha256(HERE / "results/pilot24-dope-q8-fivefit.json"),
            "prior_native_neural_sha256": sha256(
                HERE / "results/pilot24-native-neural.json"
            ),
        },
        "locks": PINS,
        "helper_source_sha256": {
            name: sha256(HERE / name)
            for name in (
                "pilot24_neural_comparison.py",
                "publish_dope_q8_fivefit.py",
                "pilot_metrics.py",
                "score.py",
                "manifest.py",
            )
        },
        "source_manifests": source_manifests,
        "worker_hashes": worker_hashes,
        "native_selections": [r for r in selections if r["dataset"] in DATASETS],
        "fit_attempts": fit_attempts,
        "fit_status_counts": dict(Counter(r["status"] for r in fit_attempts)),
        "new_sample_receipts": sample_receipts,
        "prior_sample_receipts": prior_sample_receipts,
        "comparator_sample_status_counts": all_sample_status_counts,
        "new_sample_status_counts": dict(Counter(r["status"] for r in sample_receipts)),
        "dope_fit_attempts": dope["fit_attempts"],
        "dope_preflight": {
            "events": dope["preflight_events"],
            "failure_receipt_sha256": dope["preflight_failure_receipt_sha256"],
            "ledger_sha256": dope["preflight_ledger_sha256"],
        },
        "summaries": summarize(cells),
        "cells": sorted(
            cells,
            key=lambda r: (
                r["dataset"],
                r["method"],
                r["configuration"],
                r["fit_seed"],
                r["sample_seed"],
                r["size_multiplier"],
            ),
        ),
        "fit_seeds": list(FIT_SEEDS),
        "sample_seeds": list(SAMPLE_SEEDS),
        "sizes_sampled": [1, 2, 4, 8],
        "sizes_measured": [1, 4],
        "cost": {
            "comparator_fixed_fit_seconds": sum(r["fit_seconds"] for r in fit_attempts),
            "comparator_fixed_total_attempt_seconds": sum(
                r["total_attempt_seconds"] for r in fit_attempts
            ),
            "comparator_new_sample_elapsed_seconds": sum(
                read(Path(r["receipt_path"]))["elapsed_seconds"]
                for r in sample_receipts
            ),
            "dope_fixed_fit_cost": dope["fit_cost"],
            "dope_prior_pilot_fit_r_and_d": dope_research_cost(),
            "native_tuning_r_and_d_wall_seconds": sum(
                r["tuning_wall_seconds"] for r in selections if r["dataset"] in DATASETS
            ),
        },
        "notes": [
            "All outcomes use the same grouped training-derived validation inputs; official tests remain sealed.",
            "CTGAN and TVAE reference: xu2019modeling in the corrected local DOPE_SYNTH_REFERENCES.bib; exact author-library sources and native objectives remain pinned in SDV_METHOD_AUDIT.md and prior locks.",
            "Native-selected/default configurations are fixed before these replications. No new tuning or common-outcome selection occurred.",
            "Default-and-tuned configurations share the same physical fit and are never charged twice.",
            "Failed executable fits remain failures, are not source-unavailable, and confer no DOPE win.",
            "PTF-v1, MFS-v2, release-safe status, and paper superiority remain null/unestablished.",
            "DOPE q8 is the selected compact compiler candidate. GPU placement does not establish that every final artifact is neural.",
            "These two datasets do not establish public-core paired significance or product coverage.",
        ],
    }


def summarize(cells):
    """Require every planned seed before emitting a complete summary."""
    keys = [
        (
            r["dataset"],
            r["method"],
            r["configuration"],
            r["fit_seed"],
            r["sample_seed"],
            r["size_multiplier"],
        )
        for r in cells
    ]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate matched metric cell")
    groups = sorted({key[:3] + (key[5],) for key in keys})
    fit_summaries = []
    group_summaries = []
    for dataset, method, configuration, size in groups:
        rows = [
            r
            for r in cells
            if (r["dataset"], r["method"], r["configuration"], r["size_multiplier"])
            == (dataset, method, configuration, size)
        ]
        if {(r["fit_seed"], r["sample_seed"]) for r in rows} != {
            (f, s) for f in FIT_SEEDS for s in SAMPLE_SEEDS
        }:
            raise ValueError("matched summary seed matrix incomplete")
        start = len(fit_summaries)
        for fit in FIT_SEEDS:
            samples = [r for r in rows if r["fit_seed"] == fit]
            good = [r for r in samples if r["status"] == "ok"]
            for auditor in AUDITORS:
                values = [
                    r["utility"][auditor]["retention"]
                    for r in good
                    if r["utility"][auditor]["informative"] is True
                    and r["utility"][auditor]["retention"] is not None
                ]
                if any(not math.isfinite(x) for x in values):
                    raise ValueError("nonfinite matched retention")
                complete = len(values) == len(SAMPLE_SEEDS)
                fit_summaries.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "configuration": configuration,
                        "size_multiplier": size,
                        "fit_seed": fit,
                        "auditor": auditor,
                        "planned_sample_seeds": len(SAMPLE_SEEDS),
                        "measured_sample_seeds": len(good),
                        "informative_sample_seeds": len(values),
                        "sample_status_counts": dict(
                            Counter(r["status"] for r in samples)
                        ),
                        "complete_median_retention": (
                            median(values) if complete else None
                        ),
                        "partial_median_retention": median(values) if values else None,
                        "artifact_bytes": good[0]["artifact_bytes"] if good else None,
                        "release_safe": None,
                    }
                )
        fits = fit_summaries[start:]
        for auditor in AUDITORS:
            values = [
                r["complete_median_retention"]
                for r in fits
                if r["auditor"] == auditor
                and r["complete_median_retention"] is not None
            ]
            complete = len(values) == len(FIT_SEEDS)
            group_summaries.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "configuration": configuration,
                    "size_multiplier": size,
                    "auditor": auditor,
                    "planned_fit_seeds": len(FIT_SEEDS),
                    "complete_fit_seeds": len(values),
                    "cell_status_counts": dict(Counter(r["status"] for r in rows)),
                    "complete_median_of_fit_medians": (
                        median(values) if complete else None
                    ),
                    "partial_median_of_complete_fit_medians": (
                        median(values) if values else None
                    ),
                    "complete_fit_min_retention": min(values) if complete else None,
                    "complete_fit_max_retention": max(values) if complete else None,
                    "release_safe": None,
                }
            )
    return {"per_fit": fit_summaries, "per_group": group_summaries}


def number(value):
    return f"{value:.4f}" if value is not None else "unavailable"


def render(report):
    """Regenerate tables from committed, rights-safe results alone."""
    if report["comparator_sample_status_counts"] != sample_matrix_counts(
        report["new_sample_receipts"], report["prior_sample_receipts"]
    ):
        raise ValueError("matched sample counts differ from receipts")
    if report["summaries"] != summarize(report["cells"]):
        raise ValueError("matched summary differs from cells")
    buffer = io.StringIO()
    fields = (
        "dataset",
        "method",
        "configuration",
        "fit_seed",
        "sample_seed",
        "size_multiplier",
        "status",
        "artifact_bytes",
        "within_l3_bytes",
        "catboost_retention",
        "linear_retention",
        "mlp_retention",
        "exact_copies",
        "near_copies",
        "real_control_exact",
        "real_control_near",
        "c2st_auc",
        "sample_receipt_sha256",
        "metric_receipt_sha256",
    )
    writer = csv.DictWriter(
        buffer, fieldnames=fields, lineterminator="\n", extrasaction="ignore"
    )
    writer.writeheader()
    for row in report["cells"]:
        flat = dict(row)
        for auditor in AUDITORS:
            flat[f"{auditor}_retention"] = (
                row["utility"][auditor]["retention"] if row["status"] == "ok" else None
            )
        if row["status"] == "ok":
            flat.update(
                {
                    "exact_copies": row["copy_counts"]["exact"],
                    "near_copies": row["copy_counts"]["near"],
                    "real_control_exact": row["real_vs_real_control_counts"]["exact"],
                    "real_control_near": row["real_vs_real_control_counts"]["near"],
                }
            )
        writer.writerow(flat)
    lines = [
        "# DOPE, CTGAN and TVAE: five-fit matched pilot validation",
        "",
        "Adult and California use the same grouped training-derived fit/validation inputs for all methods. Official tests remain sealed. CTGAN and TVAE retain their author defaults and configurations previously selected only by SDMetrics ML efficacy (binary F1 or regression R²). The fixed replication adds no tuning. Native KPI values are reported separately and are never ranked against each other.",
        "",
        "CTGAN/TVAE citation key: `xu2019modeling` from the corrected local bibliography. Exact author-library sources, licenses, native metric implementation and configuration grids are recorded in [SDV_METHOD_AUDIT.md](../SDV_METHOD_AUDIT.md).",
        "",
        "Five fit seeds (11, 23, 37, 53, 71), three sample seeds (101, 211, 307), and n/2n/4n/8n generation sizes. Shared utility is measured at n and 4n by CatBoost, linear/logistic and MLP auditors. Fit23 reuses the verified prior fixed fit and all 60 successful size/sample receipts; its 12 unavailable default samples remain explicit. The complete comparator schedule accounts for all 360 sample identities. Default and tuned configurations that are identical share one physical fit.",
        "",
        "## Common validation utility",
        "",
        "Each fit value is the median of three informative sample-seed retentions; each group value is the median of all five complete fit values. Incomplete groups have a null complete estimate. Failures never become a DOPE win.",
        "",
        "| Dataset | Method | Configuration | Size | Complete fits | CatBoost | Linear/logistic | MLP |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    summary = report["summaries"]["per_group"]
    keys = sorted(
        {
            (r["dataset"], r["method"], r["configuration"], r["size_multiplier"])
            for r in summary
        }
    )
    for key in keys:
        rows = {
            r["auditor"]: r
            for r in summary
            if (r["dataset"], r["method"], r["configuration"], r["size_multiplier"])
            == key
        }
        lines.append(
            f"| {key[0]} | {key[1]} | {key[2]} | {key[3]}n | {rows['catboost']['complete_fit_seeds']}/5 | "
            + " | ".join(
                number(rows[a]["complete_median_of_fit_medians"]) for a in AUDITORS
            )
            + " |"
        )
    lines += [
        "",
        "## Per-fit CatBoost outcomes",
        "",
        "| Dataset | Method | Configuration | Fit seed | n | 4n | Charged bytes |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    perfit = report["summaries"]["per_fit"]
    for key in sorted(
        {(r["dataset"], r["method"], r["configuration"], r["fit_seed"]) for r in perfit}
    ):
        rows = {
            r["size_multiplier"]: r
            for r in perfit
            if (r["dataset"], r["method"], r["configuration"], r["fit_seed"]) == key
            and r["auditor"] == "catboost"
        }
        b = rows[1]["artifact_bytes"]
        lines.append(
            f'| {key[0]} | {key[1]} | {key[2]} | {key[3]} | {number(rows[1]["complete_median_retention"])} | {number(rows[4]["complete_median_retention"])} | {b if b is not None else "unavailable"} |'
        )
    lines += [
        "",
        "## Native validation objectives (separate values)",
        "",
        "| Dataset | Method | Native objective | Default KPI | Selected KPI | Trial | Tuning attempts |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for r in report["native_selections"]:
        lines.append(
            f'| {r["dataset"]} | {r["method"]} | {r["objective"]} | {number(r["default_native_value"])} | {number(r["selected_native_value"])} | {r["selected_trial"]} | {r["tuning_attempts"]} |'
        )
    lines += [
        "",
        "## Failures and cost",
        "",
        "The DOPE replication also retains three preflight launch failures before a GPU fit or metric started: a missing research package path, a receipt glob that matched host inventory JSON, and a relative invocation path under scratch. Their phases, source identities and immutable ledger hashes remain in `dope_preflight`; no extra fit or successful metric is inferred.",
        "",
        f'Comparator fixed-fit attempt statuses: {json.dumps(report["fit_status_counts"],sort_keys=True)}. Complete 360-cell comparator sample statuses: {json.dumps(report["comparator_sample_status_counts"],sort_keys=True)}. New streaming sample statuses: {json.dumps(report["new_sample_status_counts"],sort_keys=True)}. Every failed attempt and its elapsed time remain in the JSON. No source-unavailable status or utility value is inferred for executable failures.',
        "",
        f'Summed comparator fixed-fit time: {report["cost"]["comparator_fixed_fit_seconds"]:.2f}s. Summed complete attempt time: {report["cost"]["comparator_fixed_total_attempt_seconds"]:.2f}s. Prior native tuning is separately reported at {report["cost"]["native_tuning_r_and_d_wall_seconds"]:.2f}s. DOPE fixed q8 replication costs {report["cost"]["dope_fixed_fit_cost"]["total_elapsed_seconds"]:.2f}s; {report["cost"]["dope_prior_pilot_fit_r_and_d"]["fit_attempts"]} prior pilot research fits add {report["cost"]["dope_prior_pilot_fit_r_and_d"]["total_elapsed_seconds"]:.2f}s separately. That pilot subtotal includes News follow-ups and excludes earlier S3 discovery/confirmation and CPU development: total campaign R&D is not yet completely accounted and is not claimed equal between methods. Fixed-fit costs include the prior seed23 once, with shared configurations counted once. Sample elapsed time includes auditor evaluation/replays and is reported separately.',
        "",
        "| Dataset | Method | Configuration | Fit seed | Host | Fit status | Fit seconds | Peak GPU MiB |",
        "|---|---|---|---:|---|---|---:|---:|",
    ]
    for r in report["fit_attempts"]:
        lines.append(
            f'| {r["dataset"]} | {r["method"]} | {r["configuration"]} | {r["fit_seed"]} | {r["host"]} | {r["status"]} | {r["fit_seconds"]:.2f} | {r["peak_device_used_mib"]} |'
        )
    lines += [
        "",
        "## Copy controls and artifact limits",
        "",
        "| Dataset | Method | Configuration | Size | Exact-copy cells | Synthetic near-match rate | Real control near-match rate |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for key in keys:
        rows = [
            r
            for r in report["cells"]
            if (r["dataset"], r["method"], r["configuration"], r["size_multiplier"])
            == key
            and r["status"] == "ok"
        ]
        if not rows:
            continue
        lines.append(
            f'| {key[0]} | {key[1]} | {key[2]} | {key[3]}n | {sum(r["copy_counts"]["exact"]>0 for r in rows)}/{len(rows)} | {median(r["copy_counts"]["near"]/r["synthetic_rows"] for r in rows):.6f} | {median(r["real_vs_real_control_counts"]["near"]/r["validation_rows"] for r in rows):.6f} |'
        )
    lines += [
        "",
        "Near counts use the fixed normalized RMS1e-3 screen and real-vs-real controls. These screens are not a complete privacy attack profile. All learned model, preprocessor, configuration and projection bytes are charged. L3 byte eligibility is reported separately from release safety; baseline artifacts above10,240 bytes remain visible as unconstrained quality.",
        "",
        "MFS-v2, PTF-v1, release-safe status and production certification remain null/unestablished. These two validation datasets do not establish the preregistered public-core paired claim or product coverage. All per-cell auditor losses, null losses, copy controls, C2ST/fidelity screens and immutable evidence hashes are retained in JSON.",
        "",
    ]
    return buffer.getvalue(), "\n".join(lines)


def main():
    from jsonschema import Draft202012Validator

    parser = argparse.ArgumentParser()
    parser.add_argument("--from-json", action="store_true")
    args = parser.parse_args()
    result = HERE / "results/pilot24-native-neural-fivefit.json"
    report = read(result) if args.from_json else build()
    schema = read(result.with_suffix(".schema.json"))
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(report)
    sealed(report)
    if (
        report["source_sha256"] != sha256(Path(__file__))
        or report["production_certified"] is not False
    ):
        raise ValueError("matched publication source or claim changed")
    if not args.from_json:
        result.write_text(
            json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n"
        )
    csv_text, markdown = render(report)
    result.with_suffix(".csv").write_text(csv_text)
    result.with_suffix(".md").write_text(markdown)
    print(
        json.dumps(
            {
                "cells": len(report["cells"]),
                "fixed_fits": report["fit_status_counts"],
                "new_samples": report["new_sample_status_counts"],
            }
        )
    )


if __name__ == "__main__":
    main()
