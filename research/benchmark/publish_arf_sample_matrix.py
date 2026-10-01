"""Reconcile the frozen ARF pilot sample grid without publishing samples."""

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
RUN = ROOT / "arf-final-samples-v1"
LOCK = RUN / "round.lock.json"
SNAP = ROOT.parent / "source-snapshots"
PRIOR = HERE / "results/pilot24-arf-matched.json"
RESULT = HERE / "results/pilot24-arf-sample-matrix.json"
DATASETS = ("Adult", "California", "News")
SEEDS = (101, 211, 307)
SIZES = (1, 2, 4, 8)
CAP = 10_240


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def build() -> dict:
    lock = read(LOCK)
    paired = read(PRIOR)
    if (lock["format"] != "dope-arf-pilot-fixed-seed-sample-round"
            or lock["source_sha256"] != sha256(SNAP / "arf_final_sample_v1.py")
            or lock["fit_source_sha256"] != sha256(SNAP / "arf_final_fit_v1.py")
            or lock["adapter_source_sha256"]
            != sha256(HERE / "arf_native.py")
            or lock["fit_round_sha256"]
            != sha256(ROOT / "arf-final-v1/round.lock.json")
            or lock["sample_seeds"] != list(SEEDS)
            or lock["sample_size_multipliers"] != list(SIZES)
            or lock["sample_timeout_seconds"] != 900
            or len(lock["jobs"]) != 48
            or len(lock["unavailable_default_cells"]) != 2
            or lock["failed_fit_cells"]
            or lock["official_tests_opened"] is not False
            or lock["mfs_v2"] is not None or lock["ptf_v1"] is not None
            or paired["official_tests_opened"] is not False
            or paired["mfs_v2"] is not None or paired["ptf_v1"] is not None):
        raise ValueError("ARF sample round or prior paired panel changed")
    paired_receipts = {
        (row["dataset"], row["configuration"], row["sample_seed"],
         row["size_multiplier"]): row["sample_receipt_sha256"]
        for row in paired["cells"] + paired["failed_sample_cells"]
    }
    if len(paired_receipts) != 24:
        raise ValueError("ARF paired n/4n cells incomplete")
    rows = []
    for job in lock["jobs"]:
        dataset, kind, seed, size = (job["dataset"], job["kind"],
                                     job["sample_seed"],
                                     job["size_multiplier"])
        receipt_path = RUN / "jobs" / digest(job) / "attempt-0001/receipt.json"
        receipt = read(receipt_path)
        fit_path = Path(job["fit_receipt_path"])
        artifact = Path(job["artifact_path"])
        inventory = [{"path": part["path"],
                      "bytes": (artifact / part["path"]).stat().st_size,
                      "sha256": sha256(artifact / part["path"])}
                     for part in job["artifact_inventory"]]
        if (dataset not in DATASETS or kind not in ("default", "tuned")
                or seed not in SEEDS or size not in SIZES
                or job["fit_seed"] != 23
                or (ROOT / "prepared-v2/worker" / dataset / "test.csv").exists()
                or sha256(fit_path) != job["fit_receipt_sha256"]
                or read(fit_path)["status"] != "ok"
                or inventory != job["artifact_inventory"]
                or sum(part["bytes"] for part in inventory)
                != job["artifact_bytes"]
                or job["artifact_bytes"] <= CAP
                or receipt["format"]
                != "dope-arf-pilot-fixed-seed-sample-attempt"
                or receipt["job"] != job
                or receipt["round_sha256"] != sha256(LOCK)
                or receipt["status"] not in ("ok", "timeout")
                or receipt["artifact_bytes"] != job["artifact_bytes"]
                or receipt["cpu_affinity"] != lock["cpu_slot"]
                or receipt["wall_seconds"] > 910
                or receipt["official_tests_opened"] is not False
                or receipt["mfs_v2"] is not None
                or receipt["ptf_v1"] is not None):
            raise ValueError("ARF sample identity, charge, or status changed")
        attempt = receipt_path.parent
        for name, expected in receipt["evidence_files"].items():
            if sha256(attempt / name) != expected:
                raise ValueError("ARF sample evidence changed")
        sample = attempt / "sample.csv"
        if ((receipt["status"] == "ok" and
             (not sample.is_file()
              or sha256(sample) != receipt["sample_sha256"]))
                or (receipt["sample_sha256"] is not None
                    and sha256(sample) != receipt["sample_sha256"])):
            raise ValueError("ARF sample hash changed")
        key = (dataset, kind, seed, size)
        receipt_hash = sha256(receipt_path)
        if size in (1, 4) and paired_receipts.get(key) != receipt_hash:
            raise ValueError("ARF paired sample receipt changed")
        rows.append({"dataset": dataset, "configuration": kind,
                     "fit_seed": 23, "sample_seed": seed,
                     "size_multiplier": size, "row_count": job["row_count"],
                     "status": receipt["status"],
                     "sample_sha256": receipt["sample_sha256"],
                     "sample_receipt_sha256": receipt_hash,
                     "fit_receipt_sha256": job["fit_receipt_sha256"],
                     "artifact_bytes": job["artifact_bytes"],
                     "within_l3_bytes": False,
                     "wall_seconds": receipt["wall_seconds"],
                     "contributes_dope_win": False})
    expected = {(dataset, kind, seed, size)
                for dataset, kind in (("Adult", "tuned"),
                                      ("California", "default"),
                                      ("California", "tuned"),
                                      ("News", "tuned"))
                for seed in SEEDS for size in SIZES}
    if ({(r["dataset"], r["configuration"], r["sample_seed"],
          r["size_multiplier"]) for r in rows} != expected):
        raise ValueError("ARF 48-cell sample matrix incomplete")
    rows.sort(key=lambda r: (r["dataset"], r["configuration"],
                             r["size_multiplier"], r["sample_seed"]))
    counts = Counter(r["status"] for r in rows)
    if counts["ok"] + counts["timeout"] != 48:
        raise ValueError("ARF sample status accounting incomplete")
    unavailable = []
    for item in lock["unavailable_default_cells"]:
        path = ROOT / "arf-native-combined-v1" / f"{item['dataset']}-selection.json"
        if (item["dataset"] not in ("Adult", "News")
                or item["status"] != "timeout"
                or sha256(path) != item["selection_receipt_sha256"]):
            raise ValueError("ARF unavailable default receipt changed")
        unavailable.append({**item, "contributes_dope_win": False})
    return {"format": "dope-pilot24-arf-sample-matrix", "version": 1,
            "scope": "training_derived_validation_only",
            "source_sha256": sha256(Path(__file__)),
            "sample_round_sha256": sha256(LOCK),
            "prior_paired_report_sha256": sha256(PRIOR),
            "sample_source_sha256": lock["source_sha256"],
            "fit_source_sha256": lock["fit_source_sha256"],
            "l3_byte_cap": CAP,
            "cells": rows,
            "status_counts": {"ok": counts["ok"],
                              "timeout": counts["timeout"]},
            "unavailable_default_cells": unavailable,
            "notes": ["All 48 frozen ARF sample jobs at n/2n/4n/8n are accounted for, including timeouts. The previously published n/4n paired result remains the only common metric panel.",
                      "All fitted ARF artifacts exceed L3; failed or unavailable cells contribute no DOPE win. Official tests and release gates remain closed."],
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
    counts = report["status_counts"]
    lines = ["# ARF pilot: full sample-size schedule", "",
             f"The frozen 48-cell n/2n/4n/8n schedule has {counts['ok']} "
             f"successful samples and {counts['timeout']} timeouts. "
             "Every attempt and fitted artifact is hash-verified in the JSON "
             "receipt matrix. Adult and News author defaults remain "
             "unavailable after native fit timeouts.", "",
             "The paired shared validation metrics cover n and 4n only. "
             "No 2n/8n utility or release score is inferred. Every ARF "
             "artifact exceeds 10,240 bytes. Official tests remain sealed; "
             "MFS-v2, PTF-v1 and certification are null.", ""]
    return table.getvalue(), "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-json", action="store_true")
    args = parser.parse_args()
    report = read(RESULT) if args.from_json else build()
    if (report["format"] != "dope-pilot24-arf-sample-matrix"
            or report["source_sha256"] != sha256(Path(__file__))
            or report["official_tests_opened"] is not False
            or report["mfs_v2"] is not None or report["ptf_v1"] is not None):
        raise ValueError("ARF sample report changed")
    if not args.from_json:
        RESULT.write_text(json.dumps(report, sort_keys=True, indent=2,
                                     allow_nan=False) + "\n")
    csv_text, markdown = render(report)
    RESULT.with_suffix(".csv").write_text(csv_text)
    RESULT.with_suffix(".md").write_text(markdown)
    print(json.dumps(report["status_counts"], sort_keys=True))


if __name__ == "__main__":
    main()
