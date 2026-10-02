"""Fail-closed admission check before a public test evaluator may run."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from .fetch_jope import LIMIT
from .manifest import digest
from .runner import check_numpy_runtime, resolve_configuration
from .score import CONTRACT, validate_contract


LOCK_NAMES = ("methods.lock.json", "datasets.lock.json", "budget.lock.json",
              "method-dataset-matrix.lock.json", "evaluator.lock.json")
SCRATCH_ROOT = Path("/mnt/fast-scratch/dope-benchmark")


def frozen_set_digest(locks: dict) -> str:
    """Bind all five payloads without a circular digest of their own binding."""
    return digest({name: digest({key: value for key, value in locks[name].items()
                                if key != "freeze_set_sha256"})
                   for name in LOCK_NAMES})


def hash_value(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def final_set_blockers(locks: dict) -> list[str]:
    blockers = []
    binding = frozen_set_digest(locks)
    if any(row.get("freeze_set_sha256") != binding for row in locks.values()):
        blockers.append("final_lock_set:hash_binding_gap")
    methods = locks["methods.lock.json"]["methods"]
    datasets = {row["id"]: row for row in locks["datasets.lock.json"]["datasets"]}
    metric_hash = locks["evaluator.lock.json"].get("metric_implementation_sha256")
    valid_hashes = hash_value(metric_hash) and hash_value(locks["budget.lock.json"].get("pilot_receipts_sha256"))
    for row in methods.values():
        if row["status"] == "locked":
            valid_hashes = valid_hashes and all(hash_value(row.get(key)) for key in ("source_sha256", "adapter_sha256"))
            dependency = row.get("dependency_or_container_digest", "")
            valid_hashes = valid_hashes and isinstance(dependency, str) and hash_value(dependency.removeprefix("sha256:"))
            objective = row["native_objective"]
            if objective["status"] == "locked":
                valid_hashes = valid_hashes and hash_value(objective.get("implementation_sha256"))
    for row in datasets.values():
        valid_hashes = valid_hashes and all(hash_value(row.get(key)) for key in ("source_row_hash", "projection_sha256"))
        valid_hashes = valid_hashes and all(hash_value(value) for value in row["split"]["hashes"].values())
        valid_hashes = valid_hashes and all(hash_value(value) for value in row["projected_files"].values())
    if not valid_hashes:
        blockers.append("final_lock_set:invalid_sha256")
    for method, row in methods.items():
        if method in ("independent_marginals", "Chow-Liu") and row["status"] == "locked":
            try:
                check_numpy_runtime(row, final=True)
            except (KeyError, TypeError, ValueError, OSError):
                blockers.append("methods.lock.json:compact_runtime_evidence_gap")
    cells = locks["method-dataset-matrix.lock.json"]["cells"]
    if {(row["method"], row["dataset"]) for row in cells} != {(method, dataset) for method in methods for dataset in datasets}:
        blockers.append("method-dataset-matrix.lock.json:pair_coverage_gap")
    groups = defaultdict(list)
    lineages_match = True
    for cell in cells:
        if not cell["applicable"]:
            continue
        method = methods[cell["method"]]
        dataset = datasets[cell["dataset"]]
        expected = {"method_source_sha256": method["source_sha256"],
                    "adapter_sha256": method["adapter_sha256"],
                    "dependency_or_container_digest": method["dependency_or_container_digest"],
                    "projection_sha256": dataset["projection_sha256"],
                    "train_sha256": dataset["projected_files"]["train"],
                    "validation_sha256": dataset["projected_files"]["validation"],
                    "evaluator_sha256": metric_hash}
        lineages_match = lineages_match and all(cell.get(key) == value for key, value in expected.items())
        key = (cell["method"], cell["dataset"], cell["panel"], cell["track"], cell["tier"], cell.get("dp_epsilon"))
        groups[key].append(cell)
    if not lineages_match:
        blockers.append("method-dataset-matrix.lock.json:lineage_binding_gap")
    complete = True
    for key, rows in groups.items():
        method = methods[key[0]]
        kinds = ("default", "tuned") if method["native_objective"]["status"] == "locked" else ("default",)
        observed = [(row["configuration"]["kind"], row["fit_seed"]) for row in rows]
        expected = {(kind, seed) for kind in kinds for seed in CONTRACT["fit_seeds"]}
        complete = complete and len(observed) == len(expected) and set(observed) == expected
        for kind in kinds:
            # Each selected configuration and selection receipt is fixed across fit seeds.
            complete = complete and len({digest(row["configuration"]) for row in rows
                                         if row["configuration"]["kind"] == kind}) == 1
        if method.get("group") == "dp":
            epsilons = {other[5] for other in groups if other[:5] == key[:5]}
            complete = complete and epsilons == {1, 4, 10}
    if not complete:
        blockers.append("method-dataset-matrix.lock.json:fit_schedule_gap")
    selections_verified = True
    for key, rows in groups.items():
        dataset = datasets[key[1]]
        if type(dataset.get("train_rows")) is not int or dataset["train_rows"] <= 0:
            selections_verified = False
            continue
        # Reuse the runner's native-winner, attempt, byte and validation-lineage
        # checks before authorizing evaluator access. These read selection and
        # model receipts only; they never read a test partition or train a model.
        unique = {(row["configuration"]["kind"], digest(row["configuration"])): row for row in rows}
        for row in unique.values():
            try:
                resolve_configuration({"final": True, "method": key[0],
                                       "configuration": row["configuration"],
                                       **({"dp_epsilon": key[5]} if key[5] is not None else {})},
                                      methods[key[0]], key[1], dataset["train_rows"], SCRATCH_ROOT,
                                      dataset["projected_files"]["validation"])
            except (KeyError, TypeError, ValueError, OSError):
                selections_verified = False
    if not selections_verified:
        blockers.append("method-dataset-matrix.lock.json:native_selection_evidence_gap")
    return blockers


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
            value = json.loads(path.read_text())
            if not isinstance(value, dict):
                raise ValueError("lock is not an object")
            locks[name] = value
        except (FileNotFoundError, json.JSONDecodeError, ValueError):
            blockers.append(f"{name}:missing_or_invalid")
    for name, lock in locks.items():
        if lock.get("complete") is not True or lock.get("frozen_for_final_evaluation") is not True:
            blockers.append(f"{name}:not_frozen")
    if "methods.lock.json" in locks:
        methods = locks["methods.lock.json"].get("methods", {})
        def objective_ready(row: dict) -> bool:
            objective = row.get("native_objective")
            return isinstance(objective, dict) and (
                (objective.get("status") == "locked"
                 and objective.get("name")
                 and objective.get("direction") in ("maximize", "minimize")
                 and objective.get("implementation_sha256")
                 and objective.get("tie_breaks") ==
                 ["artifact_bytes_ascending", "config_sha256_ascending"])
                or (objective.get("status") == "inapplicable" and objective.get("reason")))
        if not methods or any(row.get("status") not in ("locked", "unavailable") or
                (row.get("status") == "locked" and not all(row.get(key) is not None
                 for key in ("source_sha256", "adapter_sha256", "default_config",
                             "tuning_search_space", "dependency_or_container_digest", "license",
                             "license_evidence", "fit_command", "sampling_command")))
                or (row.get("status") == "locked" and not objective_ready(row))
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
                or type(row.get("applicable")) is not bool
                or row.get("dataset") not in datasets or row.get("method") not in methods
                or (row["applicable"] and row.get("track") != "common_numeric")
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
    if len(locks) == len(LOCK_NAMES):
        try:
            blockers.extend(final_set_blockers(locks))
        except (KeyError, TypeError, ValueError):
            blockers.append("final_lock_set:malformed_binding_or_schedule")
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
