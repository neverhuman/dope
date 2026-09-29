"""Fail-closed admission check before a public test evaluator may run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .score import validate_contract


LOCK_NAMES = ("methods.lock.json", "datasets.lock.json", "budget.lock.json",
              "method-dataset-matrix.lock.json", "evaluator.lock.json")


def assess(repo_root: Path, lock_root: Path) -> dict:
    blockers = []
    try:
        validate_contract(repo_root)
    except (AssertionError, ValueError, FileNotFoundError):
        blockers.append("production_kpi_contract_mismatch")
    locks = {}
    for name in LOCK_NAMES:
        path = lock_root / name
        try:
            locks[name] = json.loads(path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            blockers.append(f"{name}:missing_or_invalid")
    for name, lock in locks.items():
        if lock.get("complete") is not True or lock.get("frozen_for_final_evaluation") is not True:
            blockers.append(f"{name}:not_frozen")
    if "methods.lock.json" in locks:
        methods = locks["methods.lock.json"].get("methods", {})
        if not methods or any(row.get("status") not in ("locked", "unavailable") or
                (row.get("status") == "locked" and not all(row.get(key) is not None
                 for key in ("source_sha256", "adapter_sha256", "default_config",
                             "tuning_search_space", "dependency_or_container_digest", "license")))
                or (row.get("status") == "unavailable" and not row.get("note"))
                for row in methods.values()):
            blockers.append("methods.lock.json:source_or_config_gap")
    if "datasets.lock.json" in locks:
        datasets = locks["datasets.lock.json"].get("datasets", [])
        if not datasets or any(row.get("license", {}).get("status") != "recorded"
                or not row.get("source_row_hash") or not row.get("split", {}).get("hashes")
                for row in datasets):
            blockers.append("datasets.lock.json:manifest_or_rights_gap")
    if "budget.lock.json" in locks:
        budget = locks["budget.lock.json"]
        if budget.get("campaign_ceiling_days") != 14 or not budget.get("pilot_receipts_sha256"):
            blockers.append("budget.lock.json:pilot_or_ceiling_gap")
    if "evaluator.lock.json" in locks and not locks["evaluator.lock.json"].get("metric_implementation_sha256"):
        blockers.append("evaluator.lock.json:metric_implementation_gap")
    if "method-dataset-matrix.lock.json" in locks:
        cells = locks["method-dataset-matrix.lock.json"].get("cells", [])
        methods = locks.get("methods.lock.json", {}).get("methods", {})
        if not cells or any(not all(key in row for key in
                ("dataset", "method", "panel", "track", "tier", "applicable"))
                or (row["applicable"] and methods.get(row["method"], {}).get("status") != "locked")
                or (not row["applicable"] and not row.get("exclusion_reason"))
                for row in cells):
            blockers.append("method-dataset-matrix.lock.json:cell_gap")
    return {"format": "dope-benchmark-admission", "version": 1,
            "admitted": not blockers, "blockers": sorted(blockers)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("repo_root", type=Path)
    parser.add_argument("lock_root", type=Path)
    args = parser.parse_args()
    result = assess(args.repo_root, args.lock_root)
    print(json.dumps(result, sort_keys=True))
    if not result["admitted"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
