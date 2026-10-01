"""Publish the losslessly packed News q10 validation comparison."""

from __future__ import annotations

import argparse
import csv
import io
import json
import statistics
from pathlib import Path

from . import pilot24_neural_comparison
from .refine_projection import unpacked_projection
from .score import artifact_inventory, sha256


HERE = Path(__file__).parent
ROOT = Path("/mnt/fast-scratch/dope-benchmark/pilot-24h")
ROUND = ROOT / "dope-news-q10-packed-v2"
RESULT = HERE / "results/pilot24-news-q10-packed.json"


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def build() -> dict:
    prior = ROOT / "dope-news-q10-packed-v1"
    prior_failure = read(prior / "failure-receipt.json")
    if (prior_failure["format"] != "dope-news-q10-packed-refinement-failure"
            or prior_failure["status"] != "failed"
            or prior_failure["stage"] != "sample_check_receipt_write"
            or prior_failure["error_type"] != "FileNotFoundError"
            or prior_failure["round_lock_sha256"] != sha256(prior / "round.lock.json")
            or prior_failure["source_sha256"] != sha256(
                HERE / "refine_news_q10_projection.py")
            or prior_failure["run_log_sha256"] != sha256(prior / "run.log")
            or prior_failure["official_tests_opened"] is not False
            or prior_failure["mfs_v2"] is not None
            or prior_failure["ptf_v1"] is not None
            or any(sha256(prior / "artifact" / name) != expected
                   for name, expected in
                   prior_failure["partial_artifact_sha256"].items())):
        raise ValueError("News q10 first refinement failure changed")
    lock_path = ROUND / "round.lock.json"
    receipt_path = ROUND / "receipt.json"
    lock = read(lock_path)
    receipt = read(receipt_path)
    parent = ROOT / "dope-news-q10-v1"
    parent_round = read(parent / "round.lock.json")
    parent_samples = read(parent / "sample.lock.json")
    artifact = ROUND / "artifact"
    model = artifact / "model.dpk"
    packed = artifact / "projection.compact.json"
    worker = ROOT / "prepared-v2/worker/News"
    original_projection = worker / "projection.json"
    if (lock["format"] != "dope-news-q10-lossless-projection-refinement"
            or lock["version"] != 2
            or lock["source_sha256"] != sha256(
                HERE / "refine_news_q10_projection_v2.py")
            or lock["codec_source_sha256"] != sha256(
                HERE / "refine_projection.py")
            or lock["prior_failure_receipt_sha256"] != sha256(
                prior / "failure-receipt.json")
            or lock["parent_sample_round_sha256"] != sha256(
                parent / "sample.lock.json")
            or lock["parent_round_sha256"] != sha256(
                parent / "round.lock.json")
            or parent_round["source_sha256"] != sha256(parent / "fit.py")
            or parent_samples["source_sha256"] != sha256(parent / "sample.py")
            or lock["gpu_binary_sha256"] != sha256(Path(lock["gpu_binary_path"]))
            or lock["fit_receipt_sha256"] != sha256(
                Path(read(parent / "fit.json")["fit_receipt_path"]))
            or lock["model_sha256"] != sha256(model)
            or lock["projection_sha256"] != sha256(original_projection)
            or unpacked_projection(packed.read_bytes()) != read(original_projection)
            or (worker / "test.csv").exists()
            or lock["official_tests_opened"] is not False
            or lock["mfs_v2"] is not None or lock["ptf_v1"] is not None):
        raise ValueError("News q10 packed source or artifact changed")
    inventory, charged = artifact_inventory(
        artifact, ["model.dpk", "projection.compact.json"])
    if (receipt["format"] != "dope-news-q10-packed-projection-receipt"
            or receipt["version"] != 2
            or receipt["round_lock_sha256"] != sha256(lock_path)
            or receipt["source_sha256"] != lock["source_sha256"]
            or receipt["codec_source_sha256"] != lock["codec_source_sha256"]
            or receipt["artifact_inventory"] != inventory
            or receipt["artifact_bytes"] != charged
            or receipt["within_l3_bytes"] != (charged <= 10_240)
            or receipt["sample_checks_exact"] != 12
            or len(receipt["sample_checks"]) != 12
            or receipt["official_tests_opened"] is not False
            or receipt["mfs_v2"] is not None
            or receipt["ptf_v1"] is not None
            or receipt["production_certified"] is not False):
        raise ValueError("News q10 packed refinement receipt changed")
    checks = {}
    for cell in lock["sample_cells"]:
        from .manifest import digest
        path = ROUND / "sample-checks" / f"{digest(cell)}.json"
        check = read(path)
        job = cell["job"]
        key = (job["sample_seed"], job["size_multiplier"])
        if (check["cell"] != cell
                or check["round_sha256"] != sha256(lock_path)
                or check["status"] != "exact"
                or check["sample_sha256"] != cell["sample_sha256"]
                or check["stdout_sha256"] != sha256(path.with_suffix(".stdout"))
                or check["stderr_sha256"] != sha256(path.with_suffix(".stderr"))
                or check["official_tests_opened"] is not False
                or check["mfs_v2"] is not None or check["ptf_v1"] is not None
                or key in checks):
            raise ValueError("News q10 sample replay changed")
        checks[key] = sha256(path)
    if (len(checks) != 12
            or {(row["sample_seed"], row["size_multiplier"],
                 row["sample_check_sha256"])
                for row in receipt["sample_checks"]}
            != {(seed, size, digest_value)
                for (seed, size), digest_value in checks.items()}):
        raise ValueError("News q10 complete sample replay matrix changed")
    neural_path = HERE / "results/pilot24-native-neural.json"
    neural = read(neural_path)
    if (pilot24_neural_comparison.build() != neural
            or neural["official_tests_opened"] is not False
            or neural["mfs_v2"] is not None or neural["ptf_v1"] is not None):
        raise ValueError("matched neural validation report changed")
    selected = [row for row in neural["cells"]
                if row["dataset"] == "News" and row["fit_seed"] == 23
                and row["sample_seed"] in (101, 211, 307)
                and row["size_multiplier"] in (1, 4)
                and (row["method"], row["configuration"]) in
                (("DOPE", "q10_overcap_quality"), ("CTGAN", "tuned"),
                 ("TVAE", "default_and_tuned"))]
    by_method = {}
    for row in selected:
        key = (row["sample_seed"], row["size_multiplier"])
        bucket = by_method.setdefault(row["method"], {})
        if key in bucket:
            raise ValueError("duplicate matched News validation cell")
        bucket[key] = row
    expected = {(seed, size) for seed in (101, 211, 307) for size in (1, 4)}
    if (set(by_method) != {"DOPE", "CTGAN", "TVAE"}
            or any(set(rows) != expected for rows in by_method.values())
            or any(row["artifact_bytes"] != 13_713
                   for row in by_method["DOPE"].values())):
        raise ValueError("News paired comparator panel is incomplete")
    pairs = []
    for method in ("CTGAN", "TVAE"):
        for seed, size in sorted(expected):
            dope = by_method["DOPE"][(seed, size)]
            other = by_method[method][(seed, size)]
            a, b = dope["catboost_retention"], other["catboost_retention"]
            pairs.append({"comparator": method, "sample_seed": seed,
                          "size_multiplier": size,
                          "dope_retention": a, "comparator_retention": b,
                          "difference_dope_minus_comparator": a - b,
                          "dope_packed_artifact_bytes": charged,
                          "comparator_artifact_bytes": other["artifact_bytes"],
                          "dope_metric_receipt_sha256": dope["metric_receipt_sha256"],
                          "comparator_metric_receipt_sha256": other["metric_receipt_sha256"],
                          "dope_sample_check_sha256": checks[(seed, size)]})
    medians = []
    for size in (1, 4):
        for method in ("DOPE", "CTGAN", "TVAE"):
            values = [row["catboost_retention"] for (seed, multiplier), row in
                      by_method[method].items() if multiplier == size]
            artifact_bytes = (charged if method == "DOPE" else
                              next(iter(by_method[method].values()))["artifact_bytes"])
            medians.append({"method": method, "size_multiplier": size,
                            "median_catboost_retention": statistics.median(values),
                            "artifact_bytes": artifact_bytes,
                            "within_l3_bytes": artifact_bytes <= 10_240})
    native = [row for row in neural["native_validation_kpi"]
              if row["dataset"] == "News" and row["method"] in ("CTGAN", "TVAE")]
    if len(native) != 2:
        raise ValueError("News author KPI selections missing")
    return {"format": "dope-pilot24-news-q10-packed-validation-comparison",
            "version": 1, "scope": "training_derived_validation_only",
            "source_sha256": sha256(Path(__file__)),
            "locks": {"failed_v1_receipt": sha256(prior / "failure-receipt.json"),
                      "packed_v2_round": sha256(lock_path),
                      "packed_v2_receipt": sha256(receipt_path),
                      "neural_comparison": sha256(neural_path)},
            "dataset": "News", "fit_seed": 23,
            "sample_seeds": [101, 211, 307], "size_multipliers": [1, 4],
            "l3_byte_cap": 10_240, "original_q10_artifact_bytes": 13_713,
            "packed_q10_artifact_bytes": charged,
            "packed_q10_within_l3_bytes": charged <= 10_240,
            "packed_sample_checks_exact": 12,
            "first_attempt_status": "failed_receipt_io",
            "native_validation_kpi": native,
            "paired_cells": pairs, "summary": medians,
            "notes": ["DOPE q10 was trained on a GPU and fixed as an unconstrained quality candidate in earlier validation research; this is a lossless artifact-only packing refinement.",
                      "The packed projection map is restricted scratch material; public sidecar safety and release gates remain unverified.",
                      "CTGAN and TVAE were tuned on their own frozen author-library synthetic-ML efficacy objective.",
                      "Three sample seeds on one fit seed and one dataset are descriptive only; News retention has a small real-over-null denominator.",
                      "The failed v1 receipt-write attempt remains visible and contributes no score.",
                      "Official tests stayed sealed; MFS-v2 and PTF-v1 are null."],
            "official_tests_opened": False, "mfs_v2": None,
            "ptf_v1": None, "production_certified": False}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-json", action="store_true")
    args = parser.parse_args()
    report = read(RESULT) if args.from_json else build()
    if (report["format"] != "dope-pilot24-news-q10-packed-validation-comparison"
            or report["source_sha256"] != sha256(Path(__file__))
            or report["official_tests_opened"] is not False
            or report["mfs_v2"] is not None or report["ptf_v1"] is not None):
        raise ValueError("News packed validation result changed")
    if not args.from_json:
        RESULT.write_text(json.dumps(report, sort_keys=True, indent=2,
                                     allow_nan=False) + "\n")
    table = io.StringIO()
    writer = csv.DictWriter(table, fieldnames=("method", "size_multiplier",
                            "median_catboost_retention", "artifact_bytes",
                            "within_l3_bytes"), lineterminator="\n")
    writer.writeheader()
    writer.writerows(report["summary"])
    RESULT.with_suffix(".csv").write_text(table.getvalue())
    lines = ["# News validation: packed DOPE q10 against native-tuned CTGAN and TVAE", "",
             "The lossless packed projection and GPU-trained q10 model total "
             f"{report['packed_q10_artifact_bytes']:,} bytes, under the 10,240-byte "
             "L3 artifact limit. All 12 n/2n/4n/8n sample hashes replayed exactly. "
             "This remains validation research: the packed projection is restricted, "
             "the official tests are sealed, and MFS-v2/PTF-v1 are null.", "",
             "| Method | n median retention | 4n median retention | Artifact bytes | L3 bytes |",
             "|---|---:|---:|---:|---:|"]
    for method in ("DOPE", "CTGAN", "TVAE"):
        rows = [row for row in report["summary"] if row["method"] == method]
        one = next(row for row in rows if row["size_multiplier"] == 1)
        four = next(row for row in rows if row["size_multiplier"] == 4)
        lines.append(f"| {method} | {one['median_catboost_retention']:.4f} | "
                     f"{four['median_catboost_retention']:.4f} | "
                     f"{one['artifact_bytes']:,} | "
                     f"{'yes' if one['within_l3_bytes'] else 'no'} |")
    lines += ["", "| Comparator | Median paired DOPE difference at n | At 4n |",
              "|---|---:|---:|"]
    for method in ("CTGAN", "TVAE"):
        values = {size: statistics.median(
            row["difference_dope_minus_comparator"] for row in report["paired_cells"]
            if row["comparator"] == method and row["size_multiplier"] == size)
            for size in (1, 4)}
        lines.append(f"| {method} | {values[1]:+.4f} | {values[4]:+.4f} |")
    lines += ["", "Values are null-normalized CatBoost TSTR–TRTR retention on the "
              "same training-derived News validation rows, fit seed 23 and sample "
              "seeds 101/211/307. Native author KPI values are recorded in JSON "
              "and used only within each method's tuning search. The v1 failed "
              "receipt attempt is retained. There is no paired superiority or "
              "production certification claim.", ""]
    RESULT.with_suffix(".md").write_text("\n".join(lines))
    print(json.dumps({"packed_bytes": report["packed_q10_artifact_bytes"],
                      "paired_cells": len(report["paired_cells"])}, sort_keys=True))


if __name__ == "__main__":
    main()
