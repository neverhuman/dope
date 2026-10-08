"""Fail-closed MFS-v3 gate evaluation.

This module is separate from ``score.py``. Published receipts pin the bytes of
that v2 scorer, and a v3 function must not move those pins.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from research.benchmark.representation import (
    HEADLINE_ENCODER,
    NORMALIZER,
    representation_closeness,
    tabular_transfer_ok,
)
from research.benchmark.score import _bounded, _number


V3_PATH = Path(__file__).resolve().parents[2] / "production" / "kpi-contract-v3.json"
V3_WEIGHTS = {
    "utility_transfer": 0.35,
    "representation_closeness": 0.15,
    "driver_fidelity": 0.15,
    "distribution_fidelity": 0.15,
    "structure_fidelity": 0.1,
    "coverage_realism": 0.05,
    "compactness": 0.05,
}


def _v3_contract() -> dict:
    return json.loads(V3_PATH.read_text())


def evaluate_v3(evidence: dict) -> dict:
    """Score MFS-v3 from measured gates. Missing or refused evidence stays null.

    Representation closeness is computed here. A closeness number already sitting
    on the evidence is ignored.
    """
    contract = _v3_contract()["master_fitness"]
    weights = contract["weights"]
    frozen = (
        contract["version"] == 3
        and contract["scalarization"] == "geometric_mean"
        and contract["privacy_soft_weight"] == 0.0
        and contract["epsilon"] == 1e-6
        and weights == V3_WEIGHTS
    )
    shared_map = evidence.get("normalizer") == NORMALIZER
    closeness = None
    if shared_map and all(
        _number(evidence.get(key)) for key in ("distance", "d_null", "d_match", "gap_mitra", "gap_tabicl")
    ):
        closeness = representation_closeness(
            float(evidence["distance"]),
            float(evidence["d_null"]),
            float(evidence["d_match"]),
            float(evidence["gap_mitra"]),
            float(evidence["gap_tabicl"]),
        )
    auditors = evidence.get("utility_auditors")
    if not isinstance(auditors, list):
        auditors = []
    tabular_ok = tabular_transfer_ok(
        str(evidence.get("normalizer") or ""),
        str(evidence.get("utility_protocol") or ""),
        [str(name) for name in auditors],
    )
    vector = evidence.get("mfs_components")
    if not isinstance(vector, dict):
        vector = {}
    else:
        vector = dict(vector)
    if not tabular_ok:
        vector["utility_transfer"] = None
    checks = {
        "exact_row_match": evidence.get("exact_row_matches") == 0,
        "near_copy": evidence.get("near_copy_ok") is True,
        "cleartext_absent": evidence.get("cleartext_absent") is True,
        "clean_encoder": evidence.get("encoder") == HEADLINE_ENCODER,
        "membership_auc": _bounded(evidence.get("membership_auc"), 0.55, "max"),
        "attribute_inference": _bounded(evidence.get("attribute_inference_advantage"), 0.05, "max"),
        "tier_bytes": _number(evidence.get("artifact_bytes")) and evidence["artifact_bytes"] <= 10240,
        "shared_normalizer": shared_map,
        "tabular_transfer": tabular_ok,
        "representation_closeness": closeness is not None,
        "frozen_contract": frozen,
    }
    component_keys = [key for key in V3_WEIGHTS if key != "representation_closeness"]
    components_ok = all(_number(vector.get(key)) and 0 <= vector[key] <= 1 for key in component_keys)
    eligible = frozen and components_ok and all(checks.values()) and closeness is not None
    score = None
    if eligible:
        measured = dict(vector)
        measured["representation_closeness"] = closeness
        total = sum(V3_WEIGHTS.values())
        score = 100 * math.exp(
            sum(weight * math.log(1e-6 + measured[key]) for key, weight in V3_WEIGHTS.items()) / total
        )
    return {
        "format": "dope-mfs-v3-report",
        "version": 3,
        "eligible": eligible,
        "score": score,
        "representation_closeness": closeness,
        "gates": checks,
        "failed_gates": [key for key, passed in checks.items() if not passed],
    }
