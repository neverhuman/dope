"""Fail-closed, generator-only MFS-v2 gate evaluation.

Metric producers must be pinned separately. This module consumes their measured
evidence and never substitutes a missing attack or KPI with a favorable value.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent
CONTRACT = json.loads((HERE / "contract.json").read_text())
GATES = CONTRACT["generator_gates"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_contract(repo_root: Path) -> None:
    actual = sha256(repo_root / "production/kpi-contract.json")
    if actual != CONTRACT["production_kpi_sha256"]:
        raise ValueError("production KPI contract digest changed")
    production = json.loads((repo_root / "production/kpi-contract.json").read_text())
    expected = production["master_fitness"]
    if CONTRACT["mfs_v2"] != {
        "epsilon": expected["epsilon"],
        "privacy_soft_weight": expected["privacy_soft_weight"],
        "weights": expected["weights"],
    }:
        raise ValueError("MFS-v2 scalar differs from production")
    for key in (
        "ptf_v1_min", "feature_importance_spearman_min",
        "feature_importance_top_k_jaccard_min",
        "feature_importance_min_informative_features", "joint_fidelity_min",
        "driver_agreement_min", "calibration_degradation_max",
        "query_p95_normalized_error_max", "rare_tail_subgroup_retention_min",
        "type_i_error_max", "nominal_95_coverage_min",
        "nominal_95_coverage_max", "membership_auc_max",
        "attribute_inference_advantage_max",
    ):
        if GATES[key] != production["release_gates"][key]:
            raise ValueError(f"generator threshold differs from production: {key}")
    if CONTRACT["tier_byte_limits"] != production["tier_byte_limits"]:
        raise ValueError("tier byte limits differ from production")


def _number(value: object) -> bool:
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def _bounded(value: object, bound: float, direction: str) -> bool:
    return _number(value) and (value >= bound if direction == "min" else value <= bound)


def artifact_inventory(root: Path, relative_files: list[str]) -> tuple[list[dict], int]:
    root = root.resolve()
    if not relative_files or len(relative_files) != len(set(relative_files)):
        raise ValueError("artifact inventory absent or duplicated")
    if not root.is_dir():
        raise ValueError("artifact directory missing")
    actual = []
    for path in root.rglob("*"):
        if path.is_symlink() or (not path.is_file() and not path.is_dir()):
            raise ValueError("artifact contains a symlink or unsupported entry")
        if path.is_file():
            actual.append(path.relative_to(root).as_posix())
    if sorted(actual) != sorted(relative_files):
        raise ValueError("artifact file list does not reconcile with directory")
    inventory = []
    for name in sorted(relative_files):
        path = root / name
        if Path(name).is_absolute() or not path.resolve().is_relative_to(root):
            raise ValueError("artifact path escapes root")
        if path.is_symlink() or not path.is_file():
            raise ValueError("artifact file missing or symlinked")
        inventory.append({"path": name, "bytes": path.stat().st_size, "sha256": sha256(path)})
    return inventory, sum(item["bytes"] for item in inventory)


def evaluate(evidence: dict, artifact_root: Path, artifact_files: list[str], tier: str) -> dict:
    if tier not in CONTRACT["tier_byte_limits"]:
        raise ValueError("unknown tier")
    try:
        inventory, size = artifact_inventory(artifact_root, artifact_files)
    except ValueError:
        inventory, size = [], None
    checks = {}
    for key in (
        "ptf_v1_min", "feature_importance_spearman_min",
        "feature_importance_top_k_jaccard_min", "joint_fidelity_min",
        "driver_agreement_min", "rare_tail_subgroup_retention_min",
        "nominal_95_coverage_min",
    ):
        checks[key] = _bounded(evidence.get(key), GATES[key], "min")
    for key in (
        "calibration_degradation_max", "query_p95_normalized_error_max",
        "type_i_error_max", "nominal_95_coverage_max",
        "membership_auc_max", "attribute_inference_advantage_max",
        "exact_copies_max", "near_copies_max",
    ):
        checks[key] = _bounded(evidence.get(key), GATES[key], "max")
    # Applicability is itself evidence. Inapplicable FI needs a measured count.
    count = evidence.get("informative_feature_count")
    checks["feature_importance_applicability"] = isinstance(count, int) and not isinstance(count, bool) and count >= 0
    checks["feature_importance_complete"] = evidence.get("feature_importance_complete") is True
    if checks["feature_importance_applicability"] and count < GATES["feature_importance_min_informative_features"]:
        checks["feature_importance_spearman_min"] = True
        checks["feature_importance_top_k_jaccard_min"] = True
    checks["artifact_inventory"] = size is not None and evidence.get("artifact_bytes") == size
    checks["tier_bytes"] = size is not None and size <= CONTRACT["tier_byte_limits"][tier]
    checks["source_rows_absent"] = evidence.get("source_rows_required") is False
    checks["artifact_sampling_verified"] = evidence.get("artifact_sampling_verified") is True
    checks["privacy_attack_complete"] = evidence.get("privacy_attack_complete") is True
    checks["real_vs_real_control_complete"] = evidence.get("real_vs_real_control_complete") is True
    checks["required_size_cells_complete"] = evidence.get("required_size_cells_complete") is True
    checks["metric_implementations_locked"] = evidence.get("metric_implementations_locked") is True
    vector = evidence.get("mfs_components")
    if not isinstance(vector, dict):
        vector = {}
    weights = CONTRACT["mfs_v2"]["weights"]
    checks["mfs_components_complete"] = all(
        _number(vector.get(key)) and 0 <= vector[key] <= 1 for key in weights
    )
    eligible = all(checks.values())
    score = None
    if eligible:
        score = 100 * math.exp(sum(
            weight * math.log(CONTRACT["mfs_v2"]["epsilon"] + vector[key])
            for key, weight in weights.items()
        ))
    return {
        "format": "dope-generator-gate-report", "version": 1,
        "tier": tier, "eligible": eligible, "score": score,
        "gates": checks, "failed_gates": [key for key, passed in checks.items() if not passed],
        "mfs_components": vector, "raw_metric_vector": evidence.get("raw_metric_vector", {}),
        "artifact_bytes": size, "artifact_inventory": inventory,
        "shared_pretrained_bytes": evidence.get("shared_pretrained_bytes"),
        "runtime_bytes": evidence.get("runtime_bytes"),
        "production_release_gates": evidence.get("production_release_gates"),
    }
