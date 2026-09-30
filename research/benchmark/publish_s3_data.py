"""Export a rights-safe, hash-bound S3 panel data lock from sealed scratch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .fetch_jope import CATALOG_SHA256
from .prepare_s3 import selected_catalog, verify_prepared
from .score import sha256


def build(scratch: Path) -> dict:
    catalog = scratch / "catalog/v1/catalog.jsonl"
    selected, unknown = selected_catalog(catalog)
    summary_path = scratch / "s3-v1/preparation-summary.json"
    summary = json.loads(summary_path.read_text())
    if (summary["catalog_sha256"] != CATALOG_SHA256 or len(summary["records"]) != 104
            or summary["prepared"] != 100 or summary["excluded_after_checks"] != 4):
        raise ValueError("S3 preparation summary is incomplete")
    by_id = {record["id"]: record for record in summary["records"]}
    entries = []
    for row in selected:
        dataset_id = row["dataset_hash"]
        record = by_id[dataset_id]
        receipt_path = scratch / "s3-v1/receipts" / f"{dataset_id}.json"
        if sha256(receipt_path) != record["receipt_sha256"]:
            raise ValueError("S3 preparation receipt changed")
        receipt = json.loads(receipt_path.read_text())
        source = scratch / "datasets" / dataset_id
        source_manifest_sha256 = sha256(source / "manifest.json")
        expected_manifest = receipt.get("manifest", {}).get("source_object", {}).get("manifest_sha256")
        if ((expected_manifest is not None and source_manifest_sha256 != expected_manifest)
                or any(sha256(source / ("meta.json" if name == "meta" else f"{name}.csv")) !=
                       item["sha256"] for name, item in row["files"].items())):
            raise ValueError("S3 source object changed")
        entry = {"id": dataset_id, "status": record["status"],
                 "catalog_entry_sha256": receipt["catalog_entry_sha256"],
                 "source_manifest_sha256": source_manifest_sha256,
                 "object_sha256": {name: item["sha256"] for name, item in row["files"].items()},
                 "license": row["provenance"]["license"],
                 "receipt_sha256": record["receipt_sha256"]}
        if record["status"] == "prepared":
            verify_prepared(receipt, scratch)
            manifest = receipt["manifest"]
            entry.update({"source_row_hash": manifest["source_row_hash"],
                          "split": manifest["split"],
                          "projection_sha256": manifest["projection_sha256"],
                          "row_group_assignments_sha256": manifest["row_group_assignments_sha256"],
                          "projected_files": manifest["projected_files"],
                          "evaluator_manifest_sha256": receipt["evaluator_manifest_sha256"]})
        else:
            entry["exclusion_reason"] = record["reason"]
        entries.append(entry)
    return {"format": "dope-benchmark-s3-data-lock", "version": 1,
            "panel": "s3_matched_regression", "catalog_sha256": CATALOG_SHA256,
            "scratch_summary_sha256": sha256(summary_path),
            "eligible_entries": 104, "unknown_rights_excluded": unknown,
            "prepared": 100, "excluded_after_checks": 4,
            "final_evaluation_authorized": False, "entries": entries}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scratch", type=Path, default=Path("/mnt/fast-scratch/dope-benchmark"))
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("results") /
                        "s3-data.lock.json")
    args = parser.parse_args()
    result = build(args.scratch)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(result, sort_keys=True, indent=2) + "\n"
    if args.output.exists():
        if args.output.read_text() != encoded:
            raise ValueError("existing S3 data lock differs from scratch")
    else:
        args.output.write_text(encoded)
    print(json.dumps({"output": str(args.output), "sha256": sha256(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
