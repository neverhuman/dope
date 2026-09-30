"""Fail-closed admission check before a public test evaluator may run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .fetch_jope import LIMIT
from .score import CONTRACT, validate_contract


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
                             "tuning_search_space", "dependency_or_container_digest", "license",
                             "license_evidence", "fit_command", "sampling_command")))
                or (row.get("status") == "unavailable" and not row.get("note"))
                for row in methods.values()):
            blockers.append("methods.lock.json:source_or_config_gap")
    if "datasets.lock.json" in locks:
        datasets = locks["datasets.lock.json"].get("datasets", [])
        if (not datasets or len({row.get("id") for row in datasets}) != len(datasets)
                or any(row.get("license", {}).get("status") != "recorded"
                or not row.get("source_row_hash") or not row.get("projection_sha256")
                or set(row.get("split", {}).get("hashes", {})) != {"train", "validation", "test"}
                or set(row.get("projected_files", {})) != {"train", "validation", "test"}
                for row in datasets)):
            blockers.append("datasets.lock.json:manifest_or_rights_gap")
    if "budget.lock.json" in locks:
        budget = locks["budget.lock.json"]
        if (budget.get("reporting_milestone_days") != 14
                or budget.get("scratch_ceiling_bytes") != LIMIT
                or budget.get("tuning_trials_per_method_dataset") != 8
                or budget.get("tuning_wall_time_hours_per_method_dataset") != 12
                or not budget.get("pilot_receipts_sha256")):
            blockers.append("budget.lock.json:pilot_or_budget_gap")
    if "evaluator.lock.json" in locks and not locks["evaluator.lock.json"].get("metric_implementation_sha256"):
        blockers.append("evaluator.lock.json:metric_implementation_gap")
    if "method-dataset-matrix.lock.json" in locks:
        cells = locks["method-dataset-matrix.lock.json"].get("cells", [])
        methods = locks.get("methods.lock.json", {}).get("methods", {})
        datasets = {row.get("id") for row in locks.get("datasets.lock.json", {}).get("datasets", [])}
        applicable = [row for row in cells if row.get("applicable")]
        if (not cells or len({row.get("id") for row in applicable}) != len(applicable)
                or any(not all(key in row for key in
                ("dataset", "method", "panel", "track", "tier", "applicable"))
                or row.get("dataset") not in datasets or row.get("method") not in methods
                or (row["applicable"] and methods.get(row["method"], {}).get("status") != "locked")
                or (row["applicable"] and not all(key in row for key in
                    ("id", "fit_seed", "sample_seeds", "size_multipliers", "worker_dir",
                     "scratch_reservation_bytes", "memory_reservation_bytes", "requires_gpu",
                     "gpu_vram_mib", "timeout_seconds", "runtime_python", "configuration")))
                or (row["applicable"] and
                    (row.get("fit_seed") not in CONTRACT["fit_seeds"]
                     or row.get("sample_seeds") != CONTRACT["sample_seeds"]
                     or row.get("size_multipliers") != CONTRACT["size_multipliers"]
                     or not isinstance(row.get("configuration"), dict)
                     or row["configuration"].get("kind") not in ("default", "tuned")
                     or (methods.get(row["method"], {}).get("group") == "dp"
                         and row.get("dp_epsilon") not in (1, 4, 10))
                     or (methods.get(row["method"], {}).get("group") != "dp"
                         and "dp_epsilon" in row)))
                or (not row["applicable"] and not row.get("exclusion_reason"))
                for row in cells)):
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
