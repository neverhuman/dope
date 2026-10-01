"""Reconcile the frozen synthpop CART pilot sample schedule without publishing rows."""

from __future__ import annotations

import argparse
import csv
import io
import json
from collections import Counter
from pathlib import Path

from .manifest import digest
from .score import sha256


HERE = Path(__file__).parent
ROOT = Path("/mnt/fast-scratch/dope-benchmark/pilot-24h")
RUN = ROOT / "synthpop-cart-final-samples-v1"
LOCK = RUN / "round.lock.json"
SNAP = ROOT.parent / "source-snapshots"
PRIOR = HERE / "results/pilot24-synthpop-matched.json"
RESULT = HERE / "results/pilot24-synthpop-sample-matrix.json"
DATASETS = ("Adult", "California", "News")
SEEDS = (101, 211, 307)
SIZES = (1, 2, 4, 8)
CAP = 10_240


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def build() -> dict:
    lock = read(LOCK)
    paired = read(PRIOR)
    if (lock["format"] != "dope-synthpop-cart-pilot-final-sample-round"
            or lock["source_sha256"]
            != sha256(SNAP / "synthpop_cart_pilot_final_sample.py")
            or lock["sample_engine_source_sha256"]
            != sha256(SNAP / "synthpop_cart_pilot_sample.py")
            or lock["adapter_source_sha256"]
            != sha256(HERE / "synthpop_cart.R")
            or lock["final_fit_round_sha256"]
            != sha256(ROOT / "synthpop-cart-final-v1/round.lock.json")
            or lock["sample_seeds"] != list(SEEDS)
            or lock["sample_size_multipliers"] != list(SIZES)
            or lock["sample_timeout_seconds"] != 900
            or len(lock["jobs"]) != 72 or lock["failed_fit_cells"]
            or lock["official_tests_opened"] is not False
            or lock["mfs_v2"] is not None or lock["ptf_v1"] is not None
            or paired["official_tests_opened"] is not False
            or paired["mfs_v2"] is not None or paired["ptf_v1"] is not None):
        raise ValueError("synthpop sample round or paired panel changed")
    paired_receipts = {
        (row["dataset"], row["configuration"], row["sample_seed"],
         row["size_multiplier"]): row["sample_receipt_sha256"]
        for row in paired["cells"]
    }
    if len(paired_receipts) != 36:
        raise ValueError("synthpop paired n/4n cells incomplete")
    if len(list((RUN / "jobs").glob("*/attempt-0001/receipt.json"))) != 72:
        raise ValueError("synthpop sample receipt count changed")

    rows = []
    checked_fits = {}
    checked_workers = set()
    for job in lock["jobs"]:
        dataset, kind, seed, size = (job["dataset"], job["kind"],
                                     job["sample_seed"],
                                     job["size_multiplier"])
        worker = ROOT / "prepared-v2/worker" / dataset
        if dataset not in checked_workers:
            manifest = read(worker / "worker-manifest.json")
            if ((worker / "test.csv").exists()
                    or manifest["dataset_id"] != dataset
                    or sha256(worker / "train.csv")
                    != manifest["projected_hashes"]["train"]
                    or sha256(worker / "validation.csv")
                    != manifest["projected_hashes"]["validation"]):
                raise ValueError("synthpop worker input or test seal changed")
            checked_workers.add(dataset)
        selection = read(ROOT / "synthpop-cart-native-v1"
                         / f"{dataset}-selection.json")
        fit_path = Path(job["fit_receipt_path"])
        if fit_path not in checked_fits:
            fit = read(fit_path)
            model = Path(job["model_path"])
            projection = fit_path.parent / "artifact/projection.json"
            if (sha256(fit_path) != job["fit_receipt_sha256"]
                    or fit["format"]
                    != "dope-synthpop-cart-pilot-final-fit-attempt"
                    or fit["status"] != "ok"
                    or fit["job"]["dataset"] != dataset
                    or fit["job"]["fit_seed"] != 23
                    or fit["artifact_bytes"]
                    != model.stat().st_size + projection.stat().st_size
                    or fit["artifact_bytes"] <= CAP
                    or fit["evidence_files"]["artifact/model.rds"]
                    != sha256(model)
                    or fit["evidence_files"]["artifact/model.rds"]
                    != job["model_sha256"]
                    or fit["evidence_files"]["artifact/projection.json"]
                    != sha256(projection)
                    or fit["evidence_files"]["artifact/projection.json"]
                    != job["projection_sha256"]
                    or fit["official_tests_opened"] is not False
                    or fit["mfs_v2"] is not None
                    or fit["ptf_v1"] is not None):
                raise ValueError("synthpop fitted artifact changed")
            checked_fits[fit_path] = fit["artifact_bytes"]
        receipt_path = RUN / "jobs" / digest(job) / "attempt-0001/receipt.json"
        receipt = read(receipt_path)
        attempt = receipt_path.parent
        if (dataset not in DATASETS or kind not in ("default", "tuned")
                or seed not in SEEDS or size not in SIZES
                or job["fit_seed"] != 23
                or job["row_count"]
                != read(worker / "worker-manifest.json")["train_rows"] * size
                or job["train_sha256"] != sha256(worker / "train.csv")
                or job["validation_sha256"] != sha256(worker / "validation.csv")
                or job["fit_trial"] != (0 if kind == "default"
                    else selection["selected_trial"])
                or job["selection_sha256"] != sha256(
                    ROOT / "synthpop-cart-native-v1" / f"{dataset}-selection.json")
                or receipt["format"]
                != "dope-synthpop-cart-pilot-sample-attempt"
                or receipt["job"] != job
                or receipt["round_sha256"] != sha256(LOCK)
                or receipt["status"] != "ok"
                or receipt["cpu_affinity"] != lock["cpu_slot"]
                or not 0 <= receipt["wall_seconds"] <= 910
                or receipt["official_tests_opened"] is not False
                or receipt["mfs_v2"] is not None
                or receipt["ptf_v1"] is not None):
            raise ValueError("synthpop sample identity or status changed")
        for name, expected in receipt["evidence_files"].items():
            if sha256(attempt / name) != expected:
                raise ValueError("synthpop sample evidence changed")
        sample = attempt / "sample.csv"
        if (receipt["sample_sha256"] != sha256(sample)
                or receipt["evidence_files"].get("sample.csv")
                != receipt["sample_sha256"]):
            raise ValueError("synthpop sample changed")
        key = (dataset, kind, seed, size)
        receipt_hash = sha256(receipt_path)
        if size in (1, 4) and paired_receipts.get(key) != receipt_hash:
            raise ValueError("synthpop paired sample receipt changed")
        rows.append({"dataset": dataset, "configuration": kind,
                     "fit_seed": 23, "sample_seed": seed,
                     "size_multiplier": size, "row_count": job["row_count"],
                     "status": receipt["status"],
                     "sample_sha256": receipt["sample_sha256"],
                     "sample_receipt_sha256": receipt_hash,
                     "fit_receipt_sha256": job["fit_receipt_sha256"],
                     "artifact_bytes": checked_fits[fit_path],
                     "within_l3_bytes": False,
                     "wall_seconds": receipt["wall_seconds"],
                     "contributes_dope_win": False})
    expected = {(dataset, kind, seed, size) for dataset in DATASETS
                for kind in ("default", "tuned") for seed in SEEDS
                for size in SIZES}
    if ({(row["dataset"], row["configuration"], row["sample_seed"],
          row["size_multiplier"]) for row in rows} != expected):
        raise ValueError("synthpop 72-cell sample matrix incomplete")
    rows.sort(key=lambda row: (row["dataset"], row["configuration"],
                               row["size_multiplier"], row["sample_seed"]))
    counts = Counter(row["status"] for row in rows)
    return {"format": "dope-pilot24-synthpop-sample-matrix", "version": 1,
            "scope": "training_derived_validation_only",
            "source_sha256": sha256(Path(__file__)),
            "sample_round_sha256": sha256(LOCK),
            "prior_paired_report_sha256": sha256(PRIOR),
            "sample_source_sha256": lock["source_sha256"],
            "sample_engine_source_sha256": lock["sample_engine_source_sha256"],
            "l3_byte_cap": CAP, "cells": rows,
            "status_counts": {"ok": counts["ok"]},
            "notes": ["All 72 frozen synthpop CART sample jobs at n/2n/4n/8n are accounted for. The prior n/4n paired report remains the only shared metric panel.",
                      "Native tuning minimized validation CART pMSE. Every fitted synthpop artifact exceeds L3, so no cell contributes a release-safe DOPE win. Official tests and release gates remain closed."],
            "official_tests_opened": False, "mfs_v2": None,
            "ptf_v1": None, "production_certified": False}


def render(report: dict) -> tuple[str, str]:
    table = io.StringIO()
    fields = ("dataset", "configuration", "fit_seed", "sample_seed",
              "size_multiplier", "status", "artifact_bytes",
              "wall_seconds", "sample_receipt_sha256")
    writer = csv.DictWriter(table, fieldnames=fields, lineterminator="\n",
                            extrasaction="ignore")
    writer.writeheader()
    writer.writerows(report["cells"])
    lines = ["# synthpop CART pilot: full sample-size schedule", "",
             f"The frozen 72-cell n/2n/4n/8n schedule has "
             f"{report['status_counts']['ok']} successful samples. "
             "Every attempt and fitted artifact is hash-verified in the JSON "
             "receipt matrix.", "",
             "Native tuning minimized validation CART pMSE. The paired shared "
             "validation metrics cover n and 4n only. No 2n/8n utility or "
             "release score is inferred. Every fitted synthpop CART artifact "
             "exceeds 10,240 bytes. Official tests remain sealed; MFS-v2, "
             "PTF-v1 and certification are null.", ""]
    return table.getvalue(), "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-json", action="store_true")
    args = parser.parse_args()
    report = read(RESULT) if args.from_json else build()
    if (report["format"] != "dope-pilot24-synthpop-sample-matrix"
            or report["source_sha256"] != sha256(Path(__file__))
            or report["official_tests_opened"] is not False
            or report["mfs_v2"] is not None or report["ptf_v1"] is not None):
        raise ValueError("synthpop sample report changed")
    if not args.from_json:
        RESULT.write_text(json.dumps(report, sort_keys=True, indent=2,
                                     allow_nan=False) + "\n")
    csv_text, markdown = render(report)
    RESULT.with_suffix(".csv").write_text(csv_text)
    RESULT.with_suffix(".md").write_text(markdown)
    print(json.dumps(report["status_counts"], sort_keys=True))


if __name__ == "__main__":
    main()
