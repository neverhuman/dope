"""Coordinator-only preparation of the rights-cleared S3 regression panel.

The official S3 test remains in the evaluator directory. No metric or model
selection runs here; this step checks source integrity and freezes partitions.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .fetch_jope import CATALOG_SHA256, fetch
from .manifest import digest, prepare
from .score import sha256


ROOT = Path("/mnt/fast-scratch/dope-benchmark")


def write_once(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w") as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def selected_catalog(catalog: Path) -> tuple[list[dict], int]:
    if sha256(catalog) != CATALOG_SHA256:
        raise ValueError("catalog digest mismatch")
    all_rows = [json.loads(line) for line in catalog.open()]
    selected = [row for row in all_rows
                if row.get("origin") == "real" and row.get("task") == "regression"
                and row.get("provenance", {}).get("platform") == "pmlb"
                and row.get("provenance", {}).get("license", {}).get("status") == "recorded"
                and row["provenance"]["license"].get("spdx") == "MIT"
                and row["provenance"]["license"].get("evidence_url")]
    if len(all_rows) != 3859 or len(selected) != 104:
        raise ValueError("rights-cleared catalog population changed")
    return sorted(selected, key=lambda row: row["dataset_hash"]), len(all_rows) - len(selected)


def prepare_entry(row: dict, scratch: Path, profile: str | None) -> dict:
    dataset_id = row["dataset_hash"]
    source = fetch(dataset_id, scratch, profile)
    columns = row["files"]["train"]["columns"]
    if (row["target"] != {"index": -1, "name": "target", "type": "numeric"}
            or row["files"]["test"]["columns"] != columns
            or row["files"]["train"]["has_header"]
            or row["files"]["test"]["has_header"]):
        raise ValueError("incompatible target or file schema")
    entry = {
        "id": dataset_id, "panel": "s3_matched_regression",
        "source": row["provenance"].get("url"),
        "source_identity": f"pmlb:{row['provenance']['dataset_id']}",
        "license": source["license"], "task": "regression",
        "target": f"c{columns - 1}", "header": False,
        "raw": {name: source["files"][name]["path"] for name in ("train", "test")},
        "official_split_id": dataset_id,
        "official_split_source": "jope.dataset-catalog/v1",
        "source_object": {
            "catalog_sha256": CATALOG_SHA256,
            "catalog_entry_sha256": digest(row),
            "manifest_sha256": source["manifest_sha256"],
            "canonical_sha256": row["canonical_sha256"],
            "blob_sha256": {name: row["files"][name]["sha256"]
                            for name in ("train", "test", "meta")},
        },
    }
    manifest = prepare(entry, scratch / "s3-v1/prepared")
    counts = manifest["split"]["rows"]
    if counts["train"] < 30 or counts["validation"] < 10 or counts["test"] < 10:
        raise ValueError("insufficient usable rows")
    if manifest["split"]["kind"] != "official_test_grouped_training_80_20":
        raise ValueError("official test partition was not preserved")
    return {"status": "prepared", "id": dataset_id, "manifest": manifest,
            "evaluator_manifest_sha256": sha256(scratch / "s3-v1/prepared/evaluator" /
                                                dataset_id / "manifest.json")}


def verify_prepared(receipt: dict, scratch: Path) -> None:
    dataset_id = receipt["id"]
    worker = scratch / "s3-v1/prepared/worker" / dataset_id
    evaluator = scratch / "s3-v1/prepared/evaluator" / dataset_id
    manifest = receipt["manifest"]
    if (sha256(evaluator / "manifest.json") != receipt["evaluator_manifest_sha256"]
            or json.loads((evaluator / "manifest.json").read_text()) != manifest
            or (worker / "test.csv").exists()
            or sha256(worker / "projection.json") != manifest["projection_sha256"]
            or sha256(worker / "row-group-assignments.json") != manifest["row_group_assignments_sha256"]
            or any(sha256((evaluator if name == "test" else worker) / f"{name}.csv") != value
                   for name, value in manifest["projected_files"].items())):
        raise ValueError("prepared S3 manifest or sealed partition changed")


def run(scratch: Path, profile: str | None) -> dict:
    if not scratch.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError("S3 preparation must stay on benchmark scratch")
    catalog = scratch / "catalog/v1/catalog.jsonl"
    if not catalog.exists():
        from .fetch_jope import fetch_object
        fetch_object("catalog/v1/catalog.jsonl", catalog, profile)
    selected, excluded_unknown_rights = selected_catalog(catalog)
    receipt_root = scratch / "s3-v1/receipts"
    records = []
    for row in selected:
        dataset_id = row["dataset_hash"]
        path = receipt_root / f"{dataset_id}.json"
        if path.exists():
            receipt = json.loads(path.read_text())
            if receipt.get("catalog_entry_sha256") != digest(row):
                raise ValueError("existing receipt differs from frozen catalog")
        else:
            try:
                receipt = prepare_entry(row, scratch, profile)
            except (ValueError, OSError, KeyError) as error:
                receipt = {"status": "excluded", "id": dataset_id,
                           "reason_type": type(error).__name__, "reason": str(error)}
            receipt["catalog_entry_sha256"] = digest(row)
            write_once(path, receipt)
        if receipt["status"] == "prepared":
            verify_prepared(receipt, scratch)
        records.append({"id": dataset_id, "status": receipt["status"],
                        "receipt_sha256": sha256(path),
                        **({"reason": receipt["reason"]} if receipt["status"] == "excluded" else {})})
    summary = {"format": "dope-benchmark-s3-preparation", "version": 1,
               "catalog_sha256": CATALOG_SHA256, "eligible_catalog_entries": len(selected),
               "unknown_rights_excluded": excluded_unknown_rights,
               "prepared": sum(row["status"] == "prepared" for row in records),
               "excluded_after_checks": sum(row["status"] == "excluded" for row in records),
               "records": records}
    summary_path = scratch / "s3-v1/preparation-summary.json"
    if summary_path.exists():
        if json.loads(summary_path.read_text()) != summary:
            raise ValueError("existing S3 preparation summary differs")
    else:
        write_once(summary_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scratch", type=Path, default=ROOT)
    parser.add_argument("--profile", help="optional AWS profile; otherwise use the standard AWS credential chain")
    args = parser.parse_args()
    summary = run(args.scratch, args.profile)
    print(json.dumps({key: summary[key] for key in
                      ("eligible_catalog_entries", "unknown_rights_excluded",
                       "prepared", "excluded_after_checks")}, sort_keys=True))


if __name__ == "__main__":
    main()
