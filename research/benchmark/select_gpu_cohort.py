"""Freeze disjoint size/width-stratified S3 discovery and confirmation cohorts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .score import sha256


def select(summary_path: Path, receipt_root: Path, worker_root: Path) -> dict:
    summary = json.loads(summary_path.read_text())
    rows = []
    for record in summary["records"]:
        if record["status"] != "prepared":
            continue
        path = receipt_root / f"{record['id']}.json"
        if sha256(path) != record["receipt_sha256"]:
            raise ValueError("S3 preparation receipt changed")
        receipt = json.loads(path.read_text())
        manifest = receipt["manifest"]
        projection = worker_root / record["id"] / "projection.json"
        if sha256(projection) != manifest["projection_sha256"]:
            raise ValueError("S3 projection changed")
        rows.append({"id": record["id"], "train_rows": manifest["split"]["rows"]["train"],
                     "projected_features": json.loads(projection.read_text())["output_features"],
                     "train_sha256": manifest["projected_files"]["train"],
                     "validation_sha256": manifest["projected_files"]["validation"]})
    if len(rows) != 100:
        raise ValueError("S3 candidate population is not the frozen 100")
    size_order = sorted(rows, key=lambda row: (row["train_rows"], row["id"]))
    width_order = sorted(rows, key=lambda row: (row["projected_features"], row["id"]))
    size_bin = {row["id"]: index * 2 // len(rows) for index, row in enumerate(size_order)}
    width_bin = {row["id"]: index * 3 // len(rows) for index, row in enumerate(width_order)}
    strata = {}
    for row in rows:
        key = f"size{size_bin[row['id']]}-width{width_bin[row['id']]}"
        strata.setdefault(key, []).append(row)
    discovery, confirmation = [], []
    for key, members in sorted(strata.items()):
        ordered = sorted(members, key=lambda row: hashlib.sha256(row["id"].encode()).hexdigest())
        if len(ordered) < 2:
            raise ValueError("discovery/confirmation stratum has fewer than two datasets")
        discovery.append({"stratum": key, **ordered[0]})
        confirmation.append({"stratum": key, **ordered[1]})
    return {"format": "dope-benchmark-gpu-cohort-lock", "version": 1,
            "source_summary_sha256": sha256(summary_path),
            "selection_rule": "size and projected-feature rank bins (2 by 3); SHA-256(id) first for discovery, second for confirmation",
            "discovery": discovery, "confirmation": confirmation,
            "validation_only": True}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument("receipt_root", type=Path)
    parser.add_argument("worker_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = select(args.summary, args.receipt_root, args.worker_root)
    with args.output.open("x") as stream:
        json.dump(result, stream, sort_keys=True, indent=2)
        stream.write("\n")
    print(json.dumps({"discovery": len(result["discovery"]),
                      "confirmation": len(result["confirmation"]),
                      "sha256": sha256(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
