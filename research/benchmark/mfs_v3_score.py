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
PREREQUISITES = (
    "real_vs_real_control_complete",
    "required_size_cells_complete",
    "artifact_sampling_verified",
    "metric_implementations_locked",
)


def _v3_contract() -> dict:
    return json.loads(V3_PATH.read_text())


def evaluate_v3(evidence: dict) -> dict:
    """Score MFS-v3 from measured gates. Missing or refused evidence stays null.

    Representation closeness is computed here. A closeness number already sitting
    on the evidence is ignored.
    """
    document = _v3_contract()
    contract = document["master_fitness"]
    weights = contract["weights"]
    frozen = (
        contract["version"] == 3
        and contract["scalarization"] == "geometric_mean"
        and contract["privacy_soft_weight"] == 0.0
        and contract["epsilon"] == 1e-6
        and contract["epsilon_rule"] == "max_component_epsilon"
        and contract["prerequisites"] == list(PREREQUISITES)
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
        "exact_row_match": type(evidence.get("exact_row_matches")) is int and evidence["exact_row_matches"] == 0,
        "near_copy": evidence.get("near_copy_ok") is True,
        "cleartext_absent": evidence.get("cleartext_absent") is True,
        "clean_encoder": evidence.get("encoder") == HEADLINE_ENCODER,
        "membership_auc": _bounded(evidence.get("membership_auc"), 0.55, "max") and evidence["membership_auc"] >= 0,
        "attribute_inference": _bounded(evidence.get("attribute_inference_advantage"), 0.05, "max") and evidence["attribute_inference_advantage"] >= 0,
        "tier_bytes": type(evidence.get("artifact_bytes")) is int and 0 < evidence["artifact_bytes"] <= 10240,
        "shared_normalizer": shared_map,
        "tabular_transfer": tabular_ok,
        "representation_closeness": closeness is not None,
        "frozen_contract": frozen,
    }
    checks.update({key: evidence.get(key) is True for key in PREREQUISITES})
    component_keys = [key for key in V3_WEIGHTS if key != "representation_closeness"]
    components_ok = all(_number(vector.get(key)) and 0 <= vector[key] <= 1 for key in component_keys)
    checks["mfs_components_complete"] = components_ok
    eligible = frozen and components_ok and all(checks.values()) and closeness is not None
    score = None
    if eligible:
        measured = dict(vector)
        measured["representation_closeness"] = closeness
        total = sum(V3_WEIGHTS.values())
        score = min(100.0, max(0.0, 100 * math.exp(
            sum(weight * math.log(max(contract["epsilon"], measured[key])) for key, weight in V3_WEIGHTS.items()) / total
        )))
    return {
        "format": "dope-mfs-v3-report",
        "version": 3,
        "eligible": eligible,
        "score": score,
        "representation_closeness": closeness,
        "gates": checks,
        "failed_gates": [key for key, passed in checks.items() if not passed],
    }


V4_PATH = Path(__file__).resolve().parents[2] / "production" / "kpi-contract-v4.json"
V4_WEIGHTS = {
    "utility_transfer": 0.30,
    "driver_fidelity": 0.20,
    "representation_closeness": 0.15,
    "knowledge_compression": 0.15,
    "distribution_fidelity": 0.10,
    "structure_fidelity": 0.05,
    "coverage_realism": 0.05,
}
V4_PREREQUISITES = (
    "real_vs_real_control_complete",
    "required_size_cells_complete",
    "artifact_sampling_verified",
    "metric_implementations_locked",
    "control_contrastive_baselines_complete",
    "model_driver_attribution_measured",
)
V4_TABULAR_AUDITORS = ("kumo_tabular_l", "catboost", "mitra_v2", "tabicl2")


def evaluate_v4(evidence: dict) -> dict:
    """Score MFS-v4 from raw measured observations under the frozen v4 contract."""
    from research.transfer.objectives import (
        control_contrastive_lift,
        driver_attribution_agreement,
        knowledge_compression_metrics,
    )

    document = json.loads(V4_PATH.read_text())
    contract = document["master_fitness"]
    weights = contract["weights"]
    frozen = (
        contract["version"] == 4
        and contract["scalarization"] == "geometric_mean"
        and contract["privacy_soft_weight"] == 0.0
        and contract["epsilon"] == 1e-6
        and contract["epsilon_rule"] == "max_component_epsilon"
        and contract.get("score_range") == [0, 100]
        and contract["prerequisites"] == list(V4_PREREQUISITES)
        and weights == V4_WEIGHTS
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
    if not isinstance(auditors, list) or not all(isinstance(name, str) for name in auditors):
        auditors = []
    tabular_ok = (
        evidence.get("normalizer") == NORMALIZER
        and evidence.get("utility_protocol") == "train_on_synthetic_score_on_real"
        and len(auditors) == len(V4_TABULAR_AUDITORS)
        and all(name in auditors for name in V4_TABULAR_AUDITORS)
    )
    baseline_lift = None
    if evidence.get("control_contrastive_baselines_complete") is True:
        try:
            baseline_lift = control_contrastive_lift(
                evidence["retention"],
                evidence["predictor_only_retention"],
                evidence["independent_marginals_retention"],
                evidence.get("real_bootstrap_4n_retention"),
            )
        except (KeyError, TypeError, ValueError):
            baseline_lift = None
    lift = baseline_lift if tabular_ok else None
    utility_transfer = (
        lift["contrastive_utility_score"]
        if lift is not None and lift["positive_predictor_lift"] and lift["positive_dependence_lift"]
        else None
    )
    driver_attr = None
    real_cols = evidence.get("real_columns")
    real_imp = evidence.get("real_importances")
    if (
        evidence.get("model_driver_attribution_measured") is True
        and type(real_cols) is int
        and isinstance(real_imp, (list, tuple))
        and real_cols >= len(real_imp)
    ):
        try:
            driver_attr = driver_attribution_agreement(
                real_imp,
                evidence["synthetic_importances"],
                evidence.get("real_signs"),
                evidence.get("synthetic_signs"),
            )
        except (KeyError, TypeError, ValueError):
            driver_attr = None
    driver_fidelity = driver_attr["composite_driver_fidelity"] if driver_attr is not None else None
    compression = None
    try:
        compression = knowledge_compression_metrics(
            evidence.get("artifact_bytes"),
            evidence.get("encoded_numbers"),
            evidence.get("real_rows"),
            evidence.get("real_columns"),
            10240,
            evidence.get("retention"),
        )
    except (TypeError, ValueError):
        compression = None
    knowledge_compression = (
        compression["knowledge_compression_score"]
        if compression is not None and compression["knowledge_compression_score"] > 0.0
        else None
    )
    vector = evidence.get("mfs_components")
    vector = dict(vector) if isinstance(vector, dict) else {}
    distribution_fidelity = (
        float(vector["distribution_fidelity"])
        if _number(vector.get("distribution_fidelity")) and 0 <= vector["distribution_fidelity"] <= 1
        else None
    )
    structure_fidelity = (
        float(vector["structure_fidelity"])
        if _number(vector.get("structure_fidelity")) and 0 <= vector["structure_fidelity"] <= 1
        else None
    )
    coverage_realism = (
        float(vector["coverage_realism"])
        if _number(vector.get("coverage_realism")) and 0 <= vector["coverage_realism"] <= 1
        else None
    )
    components_ok = (
        distribution_fidelity is not None
        and structure_fidelity is not None
        and coverage_realism is not None
    )
    named_terms = (
        ("utility_transfer", utility_transfer),
        ("driver_fidelity", driver_fidelity),
        ("representation_closeness", closeness),
        ("knowledge_compression", knowledge_compression),
        ("distribution_fidelity", distribution_fidelity),
        ("structure_fidelity", structure_fidelity),
        ("coverage_realism", coverage_realism),
    )
    unavailable_terms = [name for name, val in named_terms if val is None]
    checks = {
        "exact_row_match": type(evidence.get("exact_row_matches")) is int and evidence["exact_row_matches"] == 0,
        "near_copy": evidence.get("near_copy_ok") is True,
        "cleartext_absent": evidence.get("cleartext_absent") is True,
        "clean_encoder": evidence.get("encoder") == HEADLINE_ENCODER,
        "membership_auc": _bounded(evidence.get("membership_auc"), 0.55, "max") and evidence["membership_auc"] >= 0,
        "attribute_inference": _bounded(evidence.get("attribute_inference_advantage"), 0.05, "max") and evidence["attribute_inference_advantage"] >= 0,
        "tier_bytes": type(evidence.get("artifact_bytes")) is int and 0 < evidence["artifact_bytes"] <= 10240,
        "encoded_numbers": type(evidence.get("encoded_numbers")) is int and evidence["encoded_numbers"] >= 1 and compression is not None,
        "positive_predictor_lift": lift is not None and lift["positive_predictor_lift"] is True,
        "positive_dependence_lift": lift is not None and lift["positive_dependence_lift"] is True,
        "shared_normalizer": shared_map,
        "tabular_transfer": tabular_ok,
        "representation_closeness": closeness is not None,
        "frozen_contract": frozen,
        "mfs_components_complete": components_ok,
    }
    checks.update({key: evidence.get(key) is True for key in V4_PREREQUISITES})
    checks["control_contrastive_baselines_complete"] = (
        checks["control_contrastive_baselines_complete"] and baseline_lift is not None
    )
    checks["model_driver_attribution_measured"] = (
        checks["model_driver_attribution_measured"] and driver_attr is not None
    )
    eligible = (
        frozen
        and components_ok
        and all(checks.values())
        and not unavailable_terms
    )
    score = None
    if eligible:
        measured = dict(named_terms)
        total = sum(V4_WEIGHTS.values())
        score = min(100.0, max(0.0, 100.0 * math.exp(
            sum(weight * math.log(max(contract["epsilon"], measured[key])) for key, weight in V4_WEIGHTS.items()) / total
        )))
    return {
        "format": "dope-mfs-v4-report",
        "version": 4,
        "eligible": eligible,
        "score": score,
        "utility_transfer": utility_transfer,
        "driver_fidelity": driver_fidelity,
        "representation_closeness": closeness,
        "knowledge_compression": knowledge_compression,
        "compression_metrics": compression,
        "control_contrastive_lift": lift,
        "driver_attribution": driver_attr,
        "unavailable_terms": unavailable_terms,
        "gates": checks,
        "failed_gates": [key for key, passed in checks.items() if not passed],
    }
