"""Regenerate rights-safe metric cells and tables from an immutable receipt lock."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import statistics


def verified(path, ref):
    data = Path(path).read_bytes()
    if len(data) != ref["bytes"] or hashlib.sha256(data).hexdigest() != ref["sha256"]:
        raise ValueError("frozen_receipt_changed")
    return json.loads(data)


def value(metric):
    return metric["value"] if metric["status"] == "ok" else None


def flatten(cell):
    m = cell["metrics"]
    row = {key: cell[key] for key in ("dataset", "method", "fit_seed", "sample_seed", "row_multiplier",
                                     "config_sha256", "selection_binding", "charged_artifact_bytes")}
    row.update(marginal_error_mean=statistics.mean(x["value"] for x in m["fidelity"]["marginals"]),
               pairwise_pearson_difference_mean=(value(m["fidelity"]["pairwise_pearson_difference"]) or {}).get("mean"),
               contingency_tv_mean=(value(m["fidelity"]["categorical_contingency_tv"]) or {}).get("mean"),
               alpha_precision=(value(m["alpha_beta"]) or {}).get("alpha_precision"),
               beta_recall=(value(m["alpha_beta"]) or {}).get("beta_recall"),
               c2st_catboost_auc=value(m["detection"]["catboost"]),
               c2st_logistic_auc=value(m["detection"]["logistic"]),
               dcr_fit_median=m["privacy"]["dcr_fit"]["median"],
               dcr_validation_median=m["privacy"]["dcr_validation"]["median"],
               nndr_fit_median=(value(m["privacy"]["nndr_fit"]) or {}).get("median"),
               distance_mia_auc=value(m["privacy"]["distance_mia"]),
               domias_kde_auc=value(m["privacy"]["domias_kde"]),
               null_loss=m["utility"]["null_loss"])
    for name, auditor in m["utility"]["auditors"].items():
        for key in ("trtr_loss", "tstr_loss", "retention"):
            row[name + "_" + key] = auditor.get(key)
    return row


def build(lock_ref, receipt_dir, manifest_ref, execution_ref):
    lock = verified(Path(receipt_dir) / Path(lock_ref["path"]).name, lock_ref)
    manifest = verified(Path(receipt_dir) / Path(manifest_ref["path"]).name, manifest_ref)
    execution = verified(Path(receipt_dir) / Path(execution_ref["path"]).name, execution_ref)
    if execution["receipt_lock_ref"] != lock_ref or execution["metric_receipts"] != lock["count"]:
        raise ValueError("execution_receipt_lock_binding_mismatch")
    if execution["source_refs"]["original_metric_input_manifest"]["sha256"] != manifest_ref["sha256"]:
        raise ValueError("execution_input_manifest_binding_mismatch")
    original_jobs = {job["job_sha256"]: job for job in manifest["jobs"]}
    if len(original_jobs) != len(manifest["jobs"]) or len(original_jobs) != lock["count"]:
        raise ValueError("input_matrix_count_mismatch")
    if lock.get("actual_complete") is not True or lock.get("official_tests_opened") is not False:
        raise ValueError("complete_validation_lock_required")
    if lock["count"] != len(lock["receipts"]):
        raise ValueError("receipt_count_mismatch")
    cells = []
    seen = set()
    source_sets = set()
    for ref in lock["receipts"]:
        receipt = verified(Path(receipt_dir) / Path(ref["path"]).name, ref)
        if receipt["job_sha256"] in seen or receipt.get("status") != "ok":
            raise ValueError("invalid_or_duplicate_metric_cell")
        canonical = json.dumps(receipt["job_identity"], sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        if hashlib.sha256(canonical).hexdigest() != receipt["job_sha256"]:
            raise ValueError("metric_job_identity_mismatch")
        original = original_jobs.get(receipt["job_sha256"])
        if original is None or receipt["job_identity"] != original["job_identity"]:
            raise ValueError("metric_job_not_in_frozen_input_matrix")
        for key in ("dataset", "fit_seed", "sample_seed", "row_multiplier", "config_sha256", "selection_binding", "charged_artifact_bytes"):
            if receipt[key] != original["evaluator_input"][key]:
                raise ValueError("metric_cell_identity_changed")
        for key, input_ref in receipt["input_refs"].items():
            if input_ref != original["evaluator_input"][key]:
                raise ValueError("metric_input_binding_changed")
        seen.add(receipt["job_sha256"])
        m = receipt["metrics"]
        if any(receipt.get(key) is not None or m.get(key) is not None
               for key in ("mfs_v2", "ptf_v1", "release_safe_l3", "superiority")):
            raise ValueError("gated_score_must_be_null")
        if receipt.get("official_tests_opened") is not False or m.get("official_tests_opened") is not False:
            raise ValueError("official_tests_must_remain_sealed")
        source_sets.add(tuple(sorted((key, x["sha256"]) for key, x in receipt["source_refs"].items())))
        cell = {key: receipt[key] for key in ("job_sha256", "dataset", "method", "fit_seed", "sample_seed",
                "row_multiplier", "config_sha256", "selection_binding", "physical_sampling_backend",
                "charged_artifact_bytes", "rows", "metrics", "metric_seconds", "peak_rss_bytes",
                "dependencies", "warning_count", "original_generation_deadline_unix_seconds")}
        cell.update(receipt_ref=ref, input_hashes={key: x["sha256"] for key, x in receipt["input_refs"].items()},
                    official_tests_opened=False, mfs_v2=None, ptf_v1=None, release_safe_l3=None, superiority=None)
        cells.append(cell)
    if len(source_sets) != 1:
        raise ValueError("mixed_metric_implementations")
    cells.sort(key=lambda r: (r["method"], r["dataset"], r["config_sha256"], r["row_multiplier"], r["sample_seed"]))
    flat = [flatten(cell) for cell in cells]
    summary = []
    for method, selection, n in sorted({(x["method"], x["selection_binding"], x["row_multiplier"]) for x in flat}):
        group = [x for x in flat if (x["method"], x["selection_binding"], x["row_multiplier"]) == (method, selection, n)]
        lineages = sorted({x["dataset"] for x in group})
        if any(len([x for x in group if x["dataset"] == dataset]) != 3 or
               {x["sample_seed"] for x in group if x["dataset"] == dataset} != {101, 211, 307} or
               len({(x["fit_seed"], x["config_sha256"]) for x in group if x["dataset"] == dataset}) != 1
               for dataset in lineages):
            raise ValueError("incomplete_three_sample_seed_group")
        row = dict(method=method, selection_binding=selection,
                   configuration_hashes=sorted({x["config_sha256"] for x in group}),
                   row_multiplier=n, datasets=len(lineages), cells=len(group),
                   aggregation="mean_of_three_sample_seeds_then_median_of_datasets")
        for metric in flat[0]:
            if metric in ("dataset", "method", "fit_seed", "sample_seed", "row_multiplier", "config_sha256", "selection_binding"):
                continue
            means = []
            unavailable = 0
            for dataset in lineages:
                values = [x[metric] for x in group if x["dataset"] == dataset]
                if any(v is None for v in values):
                    unavailable += 1
                else:
                    means.append(statistics.mean(values))
            row[metric] = dict(median=statistics.median(means) if means else None,
                               measured_datasets=len(means), unavailable_datasets=unavailable)
        summary.append(row)
    return dict(format="dope-retained-validation-panel", version=1, receipt_lock_ref=lock_ref, input_manifest_ref=manifest_ref,
                execution_ref=execution_ref, executed_driver_ref=execution["source_refs"]["measurement_driver"],
                runtime_inventory_ref=execution["source_refs"]["runtime_inventory"],
                source_refs=lock["source_refs"], cells=cells, summary=summary,
                metric_rows=flat, sample_cells=len(cells), lineages=len({x["dataset"] for x in cells}),
                selection_use=False, comparison_scope="training_derived_validation_only",
                production_certified=False, official_tests_opened=False, mfs_v2=None, ptf_v1=None,
                release_safe_l3=None, superiority=None)


def render(panel):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(panel["metric_rows"][0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(panel["metric_rows"])
    body = {key: val for key, val in panel.items() if key != "metric_rows"}
    return {"panel.json": (json.dumps(body, sort_keys=True, indent=2, allow_nan=False) + "\n").encode(),
            "figure-kpis.csv": stream.getvalue().encode()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt-lock", type=Path, required=True)
    parser.add_argument("--receipt-lock-sha256", required=True)
    parser.add_argument("--receipt-lock-bytes", type=int, required=True)
    parser.add_argument("--receipt-dir", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--input-manifest-sha256", required=True)
    parser.add_argument("--input-manifest-bytes", type=int, required=True)
    parser.add_argument("--execution-record", type=Path, required=True)
    parser.add_argument("--execution-record-sha256", required=True)
    parser.add_argument("--execution-record-bytes", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    lock_ref = dict(path=str(args.receipt_lock), sha256=args.receipt_lock_sha256, bytes=args.receipt_lock_bytes)
    manifest_ref = dict(path=str(args.input_manifest), sha256=args.input_manifest_sha256, bytes=args.input_manifest_bytes)
    execution_ref = dict(path=str(args.execution_record), sha256=args.execution_record_sha256, bytes=args.execution_record_bytes)
    panel = build(lock_ref, args.receipt_dir, manifest_ref, execution_ref)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, data in render(panel).items():
        (args.output / name).write_bytes(data)


if __name__ == "__main__":
    main()
