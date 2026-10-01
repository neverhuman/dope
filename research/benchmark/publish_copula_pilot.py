"""Reconcile the GaussianCopula 24-hour validation matrix and paired DOPE readout."""

from __future__ import annotations

import argparse
import csv
import io
import json
import statistics
from pathlib import Path

from .score import sha256


HERE = Path(__file__).parent
ROOT = Path("/mnt/fast-scratch/dope-benchmark/pilot-24h")
RUN = ROOT / "copula-matrix-v1"
RESULT = HERE / "results/pilot24-copula-matched.json"


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def checked(path: Path, digest: str) -> dict:
    if sha256(path) != digest:
        raise ValueError("GaussianCopula pilot evidence hash changed")
    return read(path)


def build() -> dict:
    matrix_path = RUN / "matrix.lock.json"
    parent_path = RUN / "validation.lock.json"
    validation_path = RUN / "validation-v2.lock.json"
    matrix = read(matrix_path)
    parent = read(parent_path)
    validation = read(validation_path)
    replay_lock_path = RUN / "metric-replay-v1/round.lock.json"
    replay_manifest_path = RUN / "metric-replay-v1/manifest.json"
    replay_lock = read(replay_lock_path)
    replay_manifest = read(replay_manifest_path)
    if (len(matrix["jobs"]) != 12
            or validation["matrix_sha256"] != sha256(matrix_path)
            or validation["parent_validation_lock_sha256"] != sha256(parent_path)
            or validation["metric_sha256"] != parent["metric_sha256"]
            or validation["script_sha256"] != sha256(RUN / "validate_v2.py")
            or validation["official_tests_opened"] is not False
            or validation["mfs_v2"] is not None or validation["ptf_v1"] is not None):
        raise ValueError("GaussianCopula pilot matrix or validator changed")
    if (replay_lock["source_sha256"] != sha256(
            HERE / "pilot24_copula_metric_replay.py")
            or replay_lock["validation_v2_lock_sha256"] != sha256(validation_path)
            or replay_lock["comparison_policy"]
            != "exact_metric_payload_except_elapsed_metric_seconds"
            or replay_manifest["round_lock_sha256"] != sha256(replay_lock_path)
            or replay_manifest["source_sha256"] != replay_lock["source_sha256"]
            or replay_manifest["exact_cells"] != 24
            or replay_manifest["official_tests_opened"] is not False
            or replay_manifest["mfs_v2"] is not None
            or replay_manifest["ptf_v1"] is not None):
        raise ValueError("GaussianCopula metric replay is incomplete")
    replay_by_metric = {}
    for entry in replay_manifest["cells"]:
        path = RUN / "metric-replay-v1/cells" / f"{entry['cell_digest']}.json"
        replay = checked(path, entry["receipt_sha256"])
        if (replay["status"] != "exact"
                or replay["round_lock_sha256"] != sha256(replay_lock_path)
                or replay["metric_sha256"] != entry["metric_sha256"]
                or replay["official_tests_opened"] is not False
                or replay["mfs_v2"] is not None or replay["ptf_v1"] is not None
                or entry["metric_sha256"] in replay_by_metric):
            raise ValueError("GaussianCopula metric replay receipt changed")
        replay_by_metric[entry["metric_sha256"]] = (path, replay)
    if len(replay_by_metric) != 24:
        raise ValueError("GaussianCopula replay metric matrix is incomplete")
    rows = []
    failures = []
    native = []
    for entry in matrix["jobs"]:
        job_path = Path(entry["job_path"])
        job = checked(job_path, entry["job_sha256"])
        worker = Path(job["worker_dir"])
        worker_manifest = read(worker / "worker-manifest.json")
        if (worker_manifest["dataset_id"] != entry["dataset"]
                or (worker / "test.csv").exists()
                or any(sha256(worker / f"{part}.csv")
                       != worker_manifest["projected_hashes"][part]
                       for part in ("train", "validation"))):
            raise ValueError("GaussianCopula worker partition changed")
        dispatch_path = RUN / "dispatch" / f"{entry['name']}.receipt.json"
        dispatch = read(dispatch_path)
        if (dispatch["job"] != entry or dispatch["matrix_sha256"] != sha256(matrix_path)
                or dispatch["validation_only"] is not True
                or dispatch["mfs_v2"] is not None
                or dispatch["log_sha256"] != sha256(RUN / "dispatch" / f"{entry['name']}.log")):
            raise ValueError("GaussianCopula dispatch changed")
        attempt_matches = []
        for path in (RUN / "results").glob("*/fit-attempts/attempt-*.json"):
            attempt = read(path)
            identity = attempt["fit_identity"]
            if (identity["dataset"] == entry["dataset"]
                    and identity["method"] == "GaussianCopula"
                    and identity["fit_seed"] == entry["seed"]
                    and identity["configuration"]["kind"] == entry["kind"]):
                attempt_matches.append((path, attempt))
        if len(attempt_matches) != 1:
            raise ValueError("GaussianCopula fit attempt missing or ambiguous")
        attempt_path, attempt = attempt_matches[0]
        fit_path = attempt_path.parents[1] / "fit-receipt.json"
        failure_path = RUN / "validation-failures-v2" / f"{entry['name']}.json"
        if attempt["status"] != "ok":
            failure = read(failure_path)
            if (fit_path.exists() or failure["stage"] != "fit"
                    or failure["fit_attempt_sha256"] != sha256(attempt_path)
                    or failure["validation_lock_sha256"] != sha256(validation_path)
                    or failure["official_tests_opened"] is not False
                    or failure["mfs_v2"] is not None
                    or failure["ptf_v1"] is not None):
                raise ValueError("GaussianCopula fit failure changed")
            failures.append({"dataset": entry["dataset"], "kind": entry["kind"],
                             "fit_seed": entry["seed"], "stage": "fit",
                             "status": attempt["status"],
                             "failure_receipt_sha256": sha256(failure_path)})
            continue
        fit = read(fit_path)
        if (fit != attempt or fit["fit_key"] != fit_path.parent.name
                or sum(item["bytes"] for item in fit["artifact_inventory"])
                != fit["artifact_bytes"]):
            raise ValueError("GaussianCopula fit or artifact inventory changed")
        for item in fit["artifact_inventory"]:
            if sha256(fit_path.parent / "artifact" / item["path"]) != item["sha256"]:
                raise ValueError("GaussianCopula artifact changed")
        sample_attempts = {}
        for path in (fit_path.parent / "sample-attempts").glob("*/attempt-0001.json"):
            receipt = read(path)
            key = (receipt["sample_seed"], receipt["row_count"])
            if (receipt["fit_key"] != fit["fit_key"]
                    or receipt["run_key"] != path.parent.name
                    or key in sample_attempts):
                raise ValueError("GaussianCopula sample attempt changed")
            sample_attempts[key] = (path, receipt)
        expected = {(seed, worker_manifest["train_rows"] * size)
                    for seed in job["sample_seeds"]
                    for size in job["size_multipliers"]}
        if set(sample_attempts) != expected:
            raise ValueError("GaussianCopula sample attempt matrix incomplete")
        failed_samples = [(path, receipt) for path, receipt in sample_attempts.values()
                          if receipt["status"] != "ok"]
        if failed_samples:
            failure = read(failure_path)
            if (failure["stage"] != "sample"
                    or failure["fit_receipt_sha256"] != sha256(fit_path)
                    or failure["validation_lock_sha256"] != sha256(validation_path)
                    or sorted(item["sample_attempt_sha256"]
                              for item in failure["failed_samples"])
                    != sorted(sha256(path) for path, _ in failed_samples)
                    or failure["official_tests_opened"] is not False
                    or failure["mfs_v2"] is not None or failure["ptf_v1"] is not None):
                raise ValueError("GaussianCopula sample failure changed")
            failures.append({"dataset": entry["dataset"], "kind": entry["kind"],
                             "fit_seed": entry["seed"], "stage": "sample",
                             "status": "timeout_or_failure",
                             "failed_sample_attempts": len(failed_samples),
                             "failure_receipt_sha256": sha256(failure_path)})
        elif failure_path.exists():
            raise ValueError("successful GaussianCopula fit has failure receipt")
        for seed in job["sample_seeds"]:
            for size in validation["metric_multipliers"]:
                key = (seed, worker_manifest["train_rows"] * size)
                _, sample_attempt = sample_attempts[key]
                if sample_attempt["status"] != "ok":
                    continue
                sample_path = fit_path.parent / f"{sample_attempt['run_key']}.receipt.json"
                if read(sample_path) != sample_attempt:
                    raise ValueError("GaussianCopula sample success changed")
                sample_csv = sample_path.with_suffix("").with_suffix(".csv")
                if sha256(sample_csv) != sample_attempt["sample_sha256"]:
                    raise ValueError("GaussianCopula sample changed")
                metric_path = RUN / "validation-metrics" / fit["fit_key"] / f"{sample_attempt['run_key']}.json"
                metric = read(metric_path)
                metric_sha = sha256(metric_path)
                replay_path, replay = replay_by_metric[metric_sha]
                if (replay["cell"]["metric_path"] != str(metric_path)
                        or replay["metric_payload"] != {
                            key: value for key, value in metric["metrics"].items()
                            if key != "metric_seconds"}):
                    raise ValueError("GaussianCopula metric replay payload changed")
                if (metric["fit_receipt_sha256"] != sha256(fit_path)
                        or metric["sample_receipt_sha256"] != sha256(sample_path)
                        or metric["metric_source_sha256"] != validation["metric_sha256"]
                        or metric["validation_lock_sha256"] not in
                        (sha256(parent_path), sha256(validation_path))
                        or metric["dataset"] != entry["dataset"]
                        or metric["kind"] != entry["kind"]
                        or metric["fit_seed"] != entry["seed"]
                        or metric["sample_seed"] != seed
                        or metric["size_multiplier"] != size
                        or metric["artifact_bytes"] != fit["artifact_bytes"]
                        or metric["official_tests_opened"] is not False
                        or metric["mfs_v2"] is not None
                        or metric["ptf_v1"] is not None
                        or metric["metrics"]["mfs_v2"] is not None):
                    raise ValueError("GaussianCopula validation metric changed")
                rows.append({"dataset": entry["dataset"], "method": "GaussianCopula",
                             "kind": entry["kind"], "fit_seed": entry["seed"],
                             "sample_seed": seed, "size_multiplier": size,
                             "catboost_retention": metric["metrics"]["utility"]["catboost"].get("retention"),
                             "c2st_auc": metric["metrics"]["c2st_auc"],
                             "exact_row_copies": metric["metrics"]["copy_counts"]["exact"],
                             "artifact_bytes": fit["artifact_bytes"],
                             "within_l3_bytes": fit["artifact_bytes"] <= 10_240,
                             "metric_receipt_sha256": sha256(metric_path),
                             "metric_replay_receipt_sha256": sha256(replay_path),
                             "fit_receipt_sha256": sha256(fit_path),
                             "sample_receipt_sha256": sha256(sample_path)})
        if entry["kind"] == "tuned" and entry["seed"] == 23:
            selection_path = Path(job["configuration"]["selection_path"])
            selection = checked(selection_path, job["configuration"]["selection_sha256"])
            trial = selection["trials"][selection["selected_trial_index"]]
            if (selection["dataset"] != entry["dataset"]
                    or selection["method"] != "GaussianCopula"
                    or selection["objective"]["name"] != "mean_log_density"
                    or selection["selected_config"] != job["configuration"]["values"]
                    or trial["status"] != "ok"
                    or selection["test_opened"] is not False):
                raise ValueError("GaussianCopula native selection changed")
            for item in selection["trials"]:
                checked(Path(item["attempt_receipt_path"]), item["attempt_receipt_sha256"])
            native.append({"dataset": entry["dataset"],
                           "objective": "mean_log_density", "direction": "maximize",
                           "selected_value": trial["native_kpi"],
                           "selected_trial_index": selection["selected_trial_index"],
                           "selection_receipt_sha256": sha256(selection_path)})
    if (len(rows) != 24 or len(failures) != 6 or len(native) != 3):
        raise ValueError("GaussianCopula validation matrix is incomplete")
    neural_path = HERE / "results/pilot24-native-neural.json"
    neural = read(neural_path)
    if (neural["official_tests_opened"] is not False
            or neural["mfs_v2"] is not None or neural["ptf_v1"] is not None):
        raise ValueError("DOPE comparator report is not validation-only")
    dope = {(row["dataset"], row["sample_seed"], row["size_multiplier"]): row
            for row in neural["cells"]
            if row["method"] == "DOPE" and row["configuration"] == "q8_symbolic"
            and row["fit_seed"] == 23 and row["sample_seed"] in (101, 211)
            and row["size_multiplier"] in (1, 4)}
    copula = {(row["dataset"], row["sample_seed"], row["size_multiplier"]): row
              for row in rows if row["kind"] == "tuned" and row["fit_seed"] == 23}
    if len(dope) != 12 or set(dope) != set(copula):
        raise ValueError("DOPE and GaussianCopula paired validation cells differ")
    pairs = []
    for key in sorted(dope):
        a, b = dope[key], copula[key]
        left, right = a["catboost_retention"], b["catboost_retention"]
        pairs.append({"dataset": key[0], "sample_seed": key[1],
                      "size_multiplier": key[2],
                      "dope_catboost_retention": left,
                      "copula_catboost_retention": right,
                      "difference_dope_minus_copula":
                      left - right if left is not None and right is not None else None,
                      "dope_artifact_bytes": a["artifact_bytes"],
                      "copula_artifact_bytes": b["artifact_bytes"],
                      "dope_within_l3": a["within_l3_bytes"],
                      "copula_within_l3": b["within_l3_bytes"],
                      "dope_metric_receipt_sha256": a["metric_receipt_sha256"],
                      "copula_metric_receipt_sha256": b["metric_receipt_sha256"]})
    summary = []
    for dataset in ("Adult", "California", "News"):
        subset = [row for row in pairs if row["dataset"] == dataset]
        differences = [row["difference_dope_minus_copula"] for row in subset
                       if row["difference_dope_minus_copula"] is not None]
        summary.append({"dataset": dataset, "paired_cells": len(subset),
                        "median_dope_minus_copula": statistics.median(differences)
                        if differences else None})
    return {"format": "dope-pilot24-copula-matched-validation", "version": 1,
            "scope": "training_derived_validation_only",
            "source_sha256": sha256(Path(__file__)),
            "locks": {"matrix": sha256(matrix_path), "validation_v1": sha256(parent_path),
                      "validation_v2": sha256(validation_path),
                      "metric_replay": sha256(replay_lock_path),
                      "metric_replay_manifest": sha256(replay_manifest_path),
                      "dope_neural_report": sha256(neural_path)},
            "native_validation_kpi": native, "copula_cells": rows,
            "failed_cells": failures, "paired_dope_q8_vs_tuned_copula": pairs,
            "paired_descriptive_summary": summary,
            "notes": ["Three datasets and one paired fit seed are descriptive only.",
                      "Failed default or sample cells remain visible and have no score.",
                      "All 24 Copula metrics replayed exactly except elapsed timing.",
                      "Native mean log density is used only for GaussianCopula tuning.",
                      "No official test rows opened; MFS-v2 and PTF-v1 remain null."],
            "official_tests_opened": False, "mfs_v2": None, "ptf_v1": None,
            "production_certified": False}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-json", action="store_true",
                        help="regenerate CSV and Markdown from the committed JSON")
    args = parser.parse_args()
    report = read(RESULT) if args.from_json else build()
    if (report["format"] != "dope-pilot24-copula-matched-validation"
            or report["source_sha256"] != sha256(Path(__file__))
            or report["official_tests_opened"] is not False
            or report["mfs_v2"] is not None or report["ptf_v1"] is not None):
        raise ValueError("Copula published report identity changed")
    if not args.from_json:
        RESULT.write_text(json.dumps(report, sort_keys=True, indent=2,
                                     allow_nan=False) + "\n")
    table = io.StringIO()
    columns = ("dataset", "sample_seed", "size_multiplier",
               "dope_catboost_retention", "copula_catboost_retention",
               "difference_dope_minus_copula", "dope_artifact_bytes",
               "copula_artifact_bytes", "dope_within_l3", "copula_within_l3")
    writer = csv.DictWriter(table, fieldnames=columns, lineterminator="\n",
                            extrasaction="ignore")
    writer.writeheader()
    writer.writerows(report["paired_dope_q8_vs_tuned_copula"])
    RESULT.with_suffix(".csv").write_text(table.getvalue())
    lines = ["# DOPE versus native-tuned GaussianCopula: validation pilot", "",
             "Same Adult, California, and News training-derived validation partitions. "
             "GaussianCopula was selected only by its own held-out mean log density. "
             "The common outcome below is null-normalized CatBoost TSTR–TRTR retention "
             "over fit seed 23, sample seeds 101/211, and sample sizes n/4n.", "",
             "| Dataset | DOPE median | Copula median | Median paired difference "
             "(DOPE − Copula) | DOPE bytes / L3 | Copula bytes / L3 |",
             "|---|---:|---:|---:|---:|---:|"]
    for row in report["paired_descriptive_summary"]:
        dataset = row["dataset"]
        pair = [item for item in report["paired_dope_q8_vs_tuned_copula"]
                if item["dataset"] == dataset]
        dope = statistics.median(item["dope_catboost_retention"] for item in pair)
        copula = statistics.median(item["copula_catboost_retention"] for item in pair)
        sample = pair[0]
        lines.append(f"| {dataset} | {dope:.3f} | {copula:.3f} | "
                     f"{row['median_dope_minus_copula']:+.3f} | "
                     f"{sample['dope_artifact_bytes']:,} / "
                     f"{'yes' if sample['dope_within_l3'] else 'no'} | "
                     f"{sample['copula_artifact_bytes']:,} / "
                     f"{'yes' if sample['copula_within_l3'] else 'no'} |")
    lines += ["", "| Dataset | Copula selected native mean log density | "
              "Selected native trial |", "|---|---:|---:|"]
    for row in report["native_validation_kpi"]:
        lines.append(f"| {row['dataset']} | {row['selected_value']:.6f} | "
                     f"{row['selected_trial_index']} |")
    lines += ["", "Six default fit/sample cells failed under their frozen time caps "
              "and remain explicit null outcomes. Artifact bytes include the fitted "
              "generator and projection map. All 24 scored Copula metrics replayed "
              "exactly from their verified samples, excluding elapsed timing. "
              "Native KPI values are used only within "
              "GaussianCopula datasets. This three-dataset, one-fit-seed panel is "
              "descriptive and does not establish superiority or production certification. "
              "Official tests stayed sealed; MFS-v2 and PTF-v1 are null.", ""]
    RESULT.with_suffix(".md").write_text("\n".join(lines))
    print(json.dumps(report["paired_descriptive_summary"], sort_keys=True))


if __name__ == "__main__":
    main()
