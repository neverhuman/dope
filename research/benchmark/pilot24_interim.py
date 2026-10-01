"""Rights-safe, validation-only reconciliation of the completed compact pilot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import statistics
from pathlib import Path


ROOT = Path("/mnt/fast-scratch/dope-benchmark/pilot-24h")
DATASETS = ("Adult", "California", "News")
SEEDS = (11, 23)
SAMPLE_SEEDS = (101, 211)
SIZES = (1, 4)
METHODS = ("Chow-Liu", "independent_marginals")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def within(root: Path, directory: str, raw: str) -> Path:
    path = Path(raw).resolve()
    allowed = (root / directory).resolve()
    if (not path.is_relative_to(allowed) or "evaluator" in path.parts
            or path.name == "test.csv"):
        raise ValueError("pilot receipt path leaves training-only scratch")
    return path


def measure(metric: dict) -> dict:
    if metric["mfs_v2"] is not None or metric["gate_profile_complete"] is not False:
        raise ValueError("pilot metric has a claimed gated score")
    catboost = metric["utility"]["catboost"]
    return {"retention": catboost["retention"], "null_loss": metric["null_loss"],
            "trtr_loss": catboost["trtr_loss"], "tstr_loss": catboost["tstr_loss"],
            "informative": catboost["informative"],
            "c2st_auc": metric["c2st_auc"],
            "exact_row_copies": metric["copy_counts"]["exact"],
            "real_vs_real_exact": metric["real_vs_real_control_counts"]["exact"]}


def collect_dope(root: Path) -> tuple[dict, list[dict]]:
    samples = root / "gpu-v3/validation-samples"
    packed = {}
    for path in (root / "projection-refine-v1/cells").glob("*/receipt.json"):
        receipt = read(path)
        job = receipt["job"]
        key = job["dataset"], job["fit_seed"]
        if (key in packed or receipt["status"] != "l3_byte_eligible"
                or receipt["restricted_projection_map"] is not True
                or receipt["production_certified"] is not False
                or receipt["ptf_v1"] is not None or receipt["mfs_v2"] is not None
                or sha(within(root, "gpu-v3/fits", job["fit_receipt_path"]))
                != job["fit_receipt_sha256"]):
            raise ValueError("invalid packed DOPE receipt")
        packed[key] = (receipt["artifact_bytes"], sha(path))
    if set(packed) != {(dataset, seed) for dataset in DATASETS for seed in SEEDS}:
        raise ValueError("packed DOPE evidence incomplete")
    cells = {}
    rows = []
    for dataset in DATASETS:
        for seed in SEEDS:
            for sample_seed in SAMPLE_SEEDS:
                for size in SIZES:
                    directory = samples / dataset / f"fit-{seed}" / f"sample-{sample_seed}-size-{size}"
                    receipt_path, metric_path = directory / "sample-receipt.json", directory / "validation-metrics.json"
                    receipt, metric = read(receipt_path), read(metric_path)
                    key = dataset, seed, sample_seed, size
                    if (key in cells or receipt["dataset"] != dataset
                            or receipt["fit_seed"] != seed
                            or receipt["sample_seed"] != sample_seed
                            or receipt["size_multiplier"] != size
                            or receipt["metric_sha256"] != sha(metric_path)
                            or receipt["fit_receipt_sha256"]
                            != sha(within(root, "gpu-v3/fits", receipt["fit_receipt_path"]))
                            or receipt["validation_only"] is not True
                            or receipt["official_tests_opened"] is not False
                            or receipt["ptf_v1"] is not None or receipt["mfs_v2"] is not None):
                        raise ValueError("invalid DOPE sample lineage")
                    raw_bytes = receipt["artifact_bytes"]
                    packed_bytes, pack_sha = packed[dataset, seed]
                    if packed_bytes > 10_240:
                        raise ValueError("packed DOPE artifact exceeds L3")
                    value = measure(metric)
                    cells[key] = value
                    rows.append({"dataset": dataset, "method": "dope", "configuration": "q8_symbolic",
                                 "fit_seed": seed, "sample_seed": sample_seed, "size": size,
                                 "artifact_bytes_raw": raw_bytes,
                                 "artifact_bytes_research_packed": packed_bytes,
                                 "restricted_projection_receipt_sha256": pack_sha,
                                 "metric_receipt_sha256": sha(metric_path),
                                 "sample_receipt_sha256": sha(receipt_path), **value})
    return cells, rows


def collect_compact(root: Path) -> tuple[dict, list[dict], list[dict]]:
    runner = root / "runner-v3"
    matrix_path = runner / "compact-full-matrix.lock.json"
    matrix = read(matrix_path)
    if len(matrix["jobs"]) != 24:
        raise ValueError("compact matrix incomplete")
    expected = {(d, m, k, s) for d in DATASETS for m in METHODS
                for k in ("default", "tuned") for s in SEEDS}
    jobs = {}
    for entry in matrix["jobs"]:
        key = entry["dataset"], entry["method"], entry["kind"], entry["fit_seed"]
        job_path = within(root, "runner-v3", entry["path"])
        if key in jobs or sha(job_path) != entry["sha256"]:
            raise ValueError("compact job changed")
        name = f"{entry['dataset']}-{entry['method']}-{entry['fit_seed']}-{entry['kind']}"
        dispatch = runner / "dispatch" / f"{name}.receipt.json"
        log = runner / "dispatch" / f"{name}.log"
        receipt = read(dispatch)
        if (receipt["job"] != entry or receipt["matrix_sha256"] != sha(matrix_path)
                or receipt["log_sha256"] != sha(log)
                or receipt["status"] not in ("ok", "verified_prior_success")):
            raise ValueError("compact dispatch changed")
        jobs[key] = entry
    if set(jobs) != expected:
        raise ValueError("compact dispatch set differs from frozen matrix")
    cells = {}
    rows = []
    for path in (runner / "validation-metrics").glob("*/*.json"):
        report = read(path)
        key = (report["dataset"], report["method"], report["kind"],
               report["fit_seed"], report["sample_seed"], report["size_multiplier"])
        job_key = key[:4]
        if (key in cells or job_key not in jobs
                or report["job_sha256"] != jobs[job_key]["sha256"]
                or report["fit_receipt_sha256"]
                != sha(within(root, "runner-v3/results", report["fit_receipt_path"]))
                or report["sample_receipt_sha256"]
                != sha(within(root, "runner-v3/results", report["sample_receipt_path"]))
                or report["validation_only"] is not True
                or report["official_tests_opened"] is not False
                or report["ptf_v1"] is not None or report["mfs_v2"] is not None):
            raise ValueError("invalid compact metric lineage")
        value = measure(report["metrics"])
        cells[key] = value
        rows.append({"dataset": key[0], "method": key[1], "configuration": key[2],
                     "fit_seed": key[3], "sample_seed": key[4], "size": key[5],
                     "artifact_bytes_raw": report["artifact_bytes"],
                     "metric_receipt_sha256": sha(path),
                     "fit_receipt_sha256": report["fit_receipt_sha256"],
                     "sample_receipt_sha256": report["sample_receipt_sha256"], **value})
    required = {(d, m, k, fs, ss, size) for d in DATASETS for m in METHODS
                for k in ("default", "tuned") for fs in SEEDS
                for ss in SAMPLE_SEEDS for size in SIZES}
    if set(cells) != required:
        raise ValueError("compact validation metric set incomplete")
    native = []
    methods = read(root / "package-v3/research/benchmark/methods.lock.json")["methods"]
    for dataset in DATASETS:
        for method in METHODS:
            path = root / "compact-native-v3/results" / dataset / method / "selection.json"
            selected = read(path)
            objective = methods[method]["native_objective"]
            default = methods[method]["default_config"]
            choices = [trial for trial in selected["trials"] if trial["config"] == default]
            if (selected["test_opened"] is not False or selected["objective"] != objective
                    or selected["selected_config"] != selected["trials"][selected["selected_trial_index"]]["config"]
                    or len(choices) != 1 or choices[0]["status"] != "ok"):
                raise ValueError("compact native objective changed")
            native.append({"dataset": dataset, "method": method,
                           "objective": objective["name"], "direction": objective["direction"],
                           "default_value": choices[0]["native_kpi"],
                           "tuned_value": selected["trials"][selected["selected_trial_index"]]["native_kpi"],
                           "selection_receipt_sha256": sha(path)})
    return cells, rows, native


def collect_research_candidates(root: Path) -> tuple[list[dict], list[dict]]:
    cells = []
    failures = []
    for family, directory, fits_expected in (("symbolic_refinement", "gpu-refine-v5", 6),
                                              ("gpu_neural", "gpu-neural-v1", 4)):
        stage = root / directory
        lock_path = stage / "evaluation.lock.json"
        lock = read(lock_path)
        if len(lock["fits"]) != fits_expected:
            raise ValueError("research candidate fit set incomplete")
        if family == "gpu_neural":
            if len(lock["failures"]) != 8:
                raise ValueError("neural failure set incomplete")
            for entry in lock["failures"]:
                fit_path = within(root, "gpu-neural-v1/fits", entry["fit_receipt_path"])
                fit = read(fit_path)
                if (sha(fit_path) != entry["fit_receipt_sha256"]
                        or fit["status"] != "failed"
                        or fit["failure_code"] != "byte_cap"
                        or fit["encoded_candidate_bytes"] != entry["encoded_candidate_bytes"]
                        or fit["ptf_v1"] is not None):
                    raise ValueError("neural failure receipt changed")
                failures.append({"dataset": entry["job"]["dataset"],
                                 "candidate": entry["job"]["candidate"],
                                 "target_weight": entry["job"]["target_weight"],
                                 "failure_code": fit["failure_code"],
                                 "encoded_candidate_bytes": fit["encoded_candidate_bytes"],
                                 "fit_receipt_sha256": sha(fit_path)})
        for entry in lock["fits"]:
            fit_path = within(root, f"{directory}/fits", entry["fit_receipt_path"])
            fit = read(fit_path)
            job = entry["job"]
            if (sha(fit_path) != entry["fit_receipt_sha256"]
                    or fit["status"] != "ok"
                    or fit["ptf_v1"] is not None):
                raise ValueError("research fit receipt changed")
            key = f"{job['dataset']}-{job['candidate']}"
            if family == "gpu_neural":
                key += f"-tw{job['target_weight']:g}"
            for size in SIZES:
                output = stage / "evaluation" / key / f"size-{size}"
                receipt_path = output / "receipt.json"
                metric_path = output / "validation-metrics.json"
                receipt, metric = read(receipt_path), read(metric_path)
                if (receipt["fit_receipt_sha256"] != entry["fit_receipt_sha256"]
                        or receipt["evaluation_lock_sha256"] != sha(lock_path)
                        or receipt["metric_sha256"] != sha(metric_path)
                        or receipt["job"] != job
                        or receipt["size_multiplier"] != size
                        or receipt["sample_sha256"] != receipt["deterministic_repeat_sha256"]
                        or receipt["validation_only"] is not True
                        or receipt["official_tests_opened"] is not False
                        or receipt["ptf_v1"] is not None or receipt["mfs_v2"] is not None):
                    raise ValueError("research metric lineage changed")
                cells.append({"family": family, "dataset": job["dataset"],
                              "candidate": job["candidate"], "target_weight": job["target_weight"],
                              "fit_seed": job["seed"], "sample_seed": receipt["sample_seed"],
                              "size": size, "artifact_bytes_raw": fit["artifact_bytes"],
                              "artifact_bytes_research_packed": receipt["artifact_bytes"],
                              "research_l3_byte_eligible": receipt["l3_eligible_by_bytes"],
                              "fit_receipt_sha256": sha(fit_path),
                              "sample_receipt_sha256": sha(receipt_path),
                              "metric_receipt_sha256": sha(metric_path), **measure(metric)})
    if len(cells) != 20 or len(failures) != 8:
        raise ValueError("research outcome set incomplete")
    return cells, failures


def paired_summary(dope: dict, compact: dict) -> list[dict]:
    result = []
    for dataset in DATASETS:
        for method in METHODS:
            for kind in ("default", "tuned"):
                for size in SIZES:
                    pairs = [(dope[dataset, fs, ss, size]["retention"],
                              compact[dataset, method, kind, fs, ss, size]["retention"])
                             for fs in SEEDS for ss in SAMPLE_SEEDS]
                    if any(a is None or b is None for a, b in pairs):
                        raise ValueError("pilot retention is missing")
                    delta = [a - b for a, b in pairs]
                    result.append({"dataset": dataset, "baseline": method,
                                   "baseline_configuration": kind, "size": size,
                                   "paired_replicates": len(pairs),
                                   "dope_median_retention": statistics.median(a for a, _ in pairs),
                                   "baseline_median_retention": statistics.median(b for _, b in pairs),
                                   "median_paired_difference": statistics.median(delta),
                                   "min_paired_difference": min(delta),
                                   "max_paired_difference": max(delta)})
    return result


def build(root: Path = ROOT) -> dict:
    dope, dope_rows = collect_dope(root)
    compact, compact_rows, native = collect_compact(root)
    research, failures = collect_research_candidates(root)
    return {"format": "dope-24h-validation-interim", "version": 1,
            "scope": "training_derived_validation_only", "official_tests_opened": False,
            "production_certified": False, "ptf_v1": None, "mfs_v2": None,
            "method_note": "q8 DOPE is a symbolic compiler candidate, not a neural GPU-trained model",
            "artifact_note": "DOPE packed projection maps are restricted scratch research artifacts; public sidecar safety is unverified",
            "news_note": "News TRTR and null losses are close; retention differences have a small denominator",
            "source_sha256": sha(Path(__file__)),
            "locks": {name: sha(root / name) for name in (
                "start-v2.json", "gpu-v3/gpu-validation-v3.lock.json",
                "runner-v3/compact-full-matrix.lock.json",
                "runner-v3/compact-validation-v3.lock.json",
                "projection-refine-v1/round.lock.json",
                "gpu-refine-v5/evaluation.lock.json",
                "gpu-neural-v1/evaluation.lock.json")},
            "counts": {"dope_metric_receipts": len(dope_rows),
                       "compact_metric_receipts": len(compact_rows),
                       "paired_comparisons": len(METHODS) * 2 * len(DATASETS) * len(SIZES),
                       "research_candidate_metrics": len(research),
                       "neural_byte_failures": len(failures)},
            "native_validation_kpi": native,
            "cells": sorted(dope_rows + compact_rows,
                            key=lambda row: (row["dataset"], row["method"], row["configuration"],
                                             row["fit_seed"], row["sample_seed"], row["size"])),
            "paired": paired_summary(dope, compact),
            "research_candidates": research,
            "neural_byte_failures": failures}


def render_table(report: dict) -> str:
    lines = ["# Adult / California / News validation interim", "",
             "Training-derived validation only. The official test partitions remain sealed. "
             "PTF-v1 and MFS-v2 are null because the product and privacy gates are incomplete.",
             "", "Median CatBoost null-normalized TSTR/TRTR retention over two fit and two "
             "sample seeds per dataset. Native tuning used each baseline's own held-out "
             "mean log density. The DOPE q8 rows are symbolic compiler results; the separate "
             "GPU-trained neural round is still a research candidate.", "",
             "| Dataset | Method | Configuration | n | 4n | Charged bytes |",
             "| --- | --- | --- | ---: | ---: | ---: |"]
    cells = report["cells"]
    for dataset in DATASETS:
        for method, kind in (("dope", "q8_symbolic"),
                             ("Chow-Liu", "default"), ("Chow-Liu", "tuned"),
                             ("independent_marginals", "default"),
                             ("independent_marginals", "tuned")):
            selected = [row for row in cells if row["dataset"] == dataset
                        and row["method"] == method and row["configuration"] == kind]
            values = [statistics.median(row["retention"] for row in selected
                                        if row["size"] == size) for size in SIZES]
            raw = sorted({row["artifact_bytes_raw"] for row in selected})
            byte_note = f"{raw[0]}–{raw[-1]} raw" if len(raw) > 1 else str(raw[0])
            if method == "dope":
                packed = sorted({row["artifact_bytes_research_packed"] for row in selected})
                byte_note += f"; {packed[0]}–{packed[-1]} restricted packed"
            lines.append(f"| {dataset} | {method} | {kind} | {values[0]:.4f} | "
                         f"{values[1]:.4f} | {byte_note} |")
    lines += ["", "The packed DOPE projection maps remain restricted because they contain "
              "source names and extrema. Most baseline artifacts exceed the 10,240-byte L3 cap. "
              "News has a small TRTR-to-null improvement, so its retention ratio is unstable.",
              "", "## Baseline native validation KPI", "",
              "Each value is comparable only with the same method and dataset. Values from "
              "different methods are not ranked.", "",
              "| Dataset | Method | Default mean log density | Tuned mean log density |",
              "| --- | --- | ---: | ---: |"]
    for row in report["native_validation_kpi"]:
        lines.append(f"| {row['dataset']} | {row['method']} | "
                     f"{row['default_value']:.4f} | {row['tuned_value']:.4f} |")
    lines += ["", "## One-seed architecture research", "",
              "These are validation measurements at fit seed 11 and sample seed 101. "
              "They are not the paired four-replicate comparison above.", "",
              "| Dataset | Family | Candidate | Target weight | n | 4n | Restricted packed bytes |",
              "| --- | --- | --- | ---: | ---: | ---: | ---: |"]
    candidates = report["research_candidates"]
    for dataset in DATASETS:
        keys = sorted({(row["candidate"], row["target_weight"]) for row in candidates
                       if row["dataset"] == dataset})
        for candidate, weight in keys:
            selected = [row for row in candidates if row["dataset"] == dataset
                        and row["candidate"] == candidate and row["target_weight"] == weight]
            outcomes = {row["size"]: row["retention"] for row in selected}
            lines.append(f"| {dataset} | {selected[0]['family']} | {candidate} | {weight:g} | "
                         f"{outcomes[1]:.4f} | {outcomes[4]:.4f} | "
                         f"{selected[0]['artifact_bytes_research_packed']} |")
    lines += ["", "Eight additional GPU-trained neural fits on Adult and News failed "
              "the 10,240-byte cap before sampling. Their encoded candidate sizes and "
              "receipt hashes are in the JSON. California's best GPU-trained neural "
              "candidate trails the symbolic candidates on validation utility. "
              "Changing target weight in the q8 symbolic family left its model artifact "
              "unchanged; those refits count as research cost without a selection gain."]
    lines += ["", "The machine-readable JSON contains all 120 cell metrics, 24 paired "
              "comparisons, 20 architecture metrics, eight neural failures, byte charges, "
              "and hashes of immutable scratch receipts.", ""]
    return "\n".join(lines)


def render_paired_csv(report: dict) -> str:
    output = io.StringIO()
    rows = report["paired"]
    fields = ("dataset", "baseline", "baseline_configuration", "size", "paired_replicates",
              "dope_median_retention", "baseline_median_retention",
              "median_paired_difference", "min_paired_difference", "max_paired_difference")
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--table", type=Path)
    parser.add_argument("--paired-csv", type=Path)
    args = parser.parse_args()
    report = build(args.root)
    encoded = json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if args.output:
        with args.output.open("x") as stream:
            stream.write(encoded)
    else:
        print(encoded, end="")
    for path, content in ((args.table, render_table(report)),
                          (args.paired_csv, render_paired_csv(report))):
        if path:
            with path.open("x") as stream:
                stream.write(content)


if __name__ == "__main__":
    main()
