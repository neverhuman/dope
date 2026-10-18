"""Calibrated empirical membership measurements with separate attack stages.

Low-FPR evaluation follows https://arxiv.org/abs/2112.03570. Supplied score
identities are commitments, not verified custody or attack qualification.
Aggregate evidence grants neither formal privacy nor publication authority.
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import json
import math
from pathlib import Path
import re

from scipy.stats import binomtest


class PrivacyError(ValueError):
    """Stable codes omit source values, scores, row identities and paths."""


def require(condition, code):
    if not condition:
        raise PrivacyError(code)


def number(value):
    return ((type(value) is int and abs(value) <= 2**53 - 1)
            or (type(value) is float and math.isfinite(value)))


def is_pin(value):
    return type(value) is str and re.fullmatch(r"[a-f0-9]{64}", value) is not None


def pin(value):
    try:
        data = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    except (TypeError, ValueError, UnicodeError, RecursionError):
        raise PrivacyError("invalid_identity") from None
    return hashlib.sha256(data).hexdigest()


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate_json_key")
        result[key] = value
    return result


def _decode(data):
    require(type(data) is bytes and len(data) <= 1_000_000, "invalid_evidence")
    try:
        value = json.loads(data, object_pairs_hook=_pairs,
                           parse_constant=lambda _: require(False, "nonfinite_json_number"))
        require(type(value) is dict, "invalid_evidence")
        return value
    except (ValueError, UnicodeError, RecursionError) as error:
        if isinstance(error, PrivacyError):
            raise
        raise PrivacyError("invalid_evidence") from None


def _policy_file_hash():
    try:
        data = Path(__file__).with_name("policy.json").read_bytes()
    except OSError:
        raise PrivacyError("policy_unavailable") from None
    document = _decode(data)
    expected = {"format", "version", "qualification", "default_target_fpr", "threshold_comparison",
                "calibration", "multiplicity", "auc_uncertainty", "clustered_or_repeated_release_uncertainty",
                "shadow_attacker", "attribute_attacker", "distance_control", "empirical_pass_margins",
                "formal_dp", "legal_route", "release_authorized"}
    require(set(document) == expected and document["format"] == "dope-empirical-disclosure-policy"
            and type(document["version"]) is int and document["version"] == 1, "invalid_policy")
    require(document["qualification"] == "unresolved" and document["empirical_pass_margins"] is None
            and document["formal_dp"] == document["legal_route"] == "unverified"
            and document["release_authorized"] is False, "invalid_policy")
    require(document["default_target_fpr"] == .01 and document["threshold_comparison"] == "greater_than_or_equal"
            and document["calibration"] == "independent_attack_development_only"
            and document["multiplicity"] == "bonferroni_six_one_sided_events_per_declared_comparison"
            and document["auc_uncertainty"] == "bounded_differences_per_independent_member_and_nonmember"
            and document["clustered_or_repeated_release_uncertainty"] == "unmeasured"
            and document["shadow_attacker"] == "regularized_logistic_baseline_v1"
            and document["attribute_attacker"] == "matched_reference_count_ridge_v1"
            and document["distance_control"] == "type_aware_matched_nearest_v1", "invalid_policy")
    return hashlib.sha256(data).hexdigest()


class EmpiricalPrivacyStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    INCONCLUSIVE = "inconclusive"
    UNMEASURED = "unmeasured"


@dataclass(frozen=True, slots=True, repr=False)
class DisclosurePolicy:
    target_fpr: float = .01
    alpha: float = .05
    comparisons: int = 1
    attacker_prior: float = .5

    def __post_init__(self):
        require(number(self.target_fpr) and 0 < self.target_fpr < 1, "invalid_fpr")
        require(number(self.alpha) and 1e-12 <= self.alpha <= .5, "invalid_alpha")
        require(type(self.comparisons) is int and 1 <= self.comparisons <= 100_000, "invalid_multiplicity")
        require(number(self.attacker_prior) and 0 < self.attacker_prior < 1, "invalid_prior")

    @property
    def sha256(self):
        return pin({"format": "dope-disclosure-policy-v1", "target_fpr": self.target_fpr,
                    "alpha": self.alpha, "comparisons": self.comparisons, "attacker_prior": self.attacker_prior,
                    "policy_file": _policy_file_hash()})


@dataclass(frozen=True, slots=True, repr=False)
class DisclosureThreatModel:
    access: str
    protected_unit: str
    artifact_inventory_sha256: str
    decoder_sha256: str
    ledger_sha256: str
    auxiliary_sha256: str
    sample_budget: int
    reference_rows: int
    repeated_releases: int = 1
    source_derived_lengths_sha256: str | None = None
    source_derived_statuses_sha256: str | None = None
    source_derived_metrics_sha256: str | None = None

    def __post_init__(self):
        require(type(self.access) is str and self.access in {"row_only", "query_access", "white_box"}, "invalid_access")
        require(type(self.protected_unit) is str and self.protected_unit in {"row", "entity"}, "invalid_protected_unit")
        require(all(is_pin(x) for x in (self.artifact_inventory_sha256, self.decoder_sha256,
                                       self.ledger_sha256, self.auxiliary_sha256)), "invalid_threat_identity")
        require(type(self.sample_budget) is int and 1 <= self.sample_budget <= 100_000, "invalid_sample_budget")
        require(type(self.reference_rows) is int and 1 <= self.reference_rows <= 100_000, "invalid_reference_count")
        require(type(self.repeated_releases) is int and 1 <= self.repeated_releases <= 100_000, "invalid_release_count")
        require(all(x is None or is_pin(x) for x in (self.source_derived_lengths_sha256,
                    self.source_derived_statuses_sha256, self.source_derived_metrics_sha256)), "invalid_threat_identity")

    @property
    def sha256(self):
        return pin(asdict(self))


@dataclass(frozen=True, slots=True, repr=False)
class ScoreView:
    members: tuple
    nonmembers: tuple
    member_ids: tuple
    nonmember_ids: tuple
    units: tuple
    role: str
    recipe_sha256: str
    input_sha256: str

    def __post_init__(self):
        require(is_pin(self.input_sha256), "invalid_input_identity")
        require(type(self.role) is str and self.role in {"attack_development", "attack_evaluation"}, "invalid_role")
        require(is_pin(self.recipe_sha256), "invalid_recipe")
        for scores, ids in ((self.members, self.member_ids), (self.nonmembers, self.nonmember_ids)):
            require(type(scores) is tuple and type(ids) is tuple and len(scores) == len(ids), "invalid_scores")
            require(all(number(x) for x in scores), "invalid_scores")
        ids = self.member_ids + self.nonmember_ids
        require(type(self.units) is tuple and len(self.units) == len(ids), "invalid_units")
        require(all(is_pin(x) for x in ids + self.units), "invalid_ids")
        require(0 < len(ids) <= 100_000 and len(set(ids)) == len(ids), "invalid_observation_ids")
        require(not set(self.units[:len(self.members)]).intersection(self.units[len(self.members):]), "unit_class_overlap")

    @property
    def sha256(self):
        return pin(asdict(self))

    @property
    def independent(self):
        return len(set(self.units)) == len(self.units)


def auc(members, nonmembers):
    require(type(members) is tuple and type(nonmembers) is tuple, "invalid_scores")
    require(bool(members) and bool(nonmembers), "missing_class")
    require(all(number(x) for x in members + nonmembers), "invalid_scores")
    ordered = sorted(nonmembers)
    wins = sum(bisect_left(ordered, x) + (bisect_right(ordered, x) - bisect_left(ordered, x)) / 2 for x in members)
    return wins / (len(members) * len(nonmembers))


def rate_bounds(successes, total, alpha):
    """Exact one-sided Clopper--Pearson endpoints, each with error alpha."""
    require(type(successes) is int and type(total) is int and 0 <= successes <= total and total > 0, "invalid_counts")
    require(number(alpha) and 0 < alpha < 1, "invalid_alpha")
    lo = binomtest(successes, total, alternative="greater").proportion_ci(1 - alpha, method="exact").low
    hi = binomtest(successes, total, alternative="less").proportion_ci(1 - alpha, method="exact").high
    return float(lo), float(hi)


@dataclass(frozen=True, slots=True, repr=False)
class AttackEvidence:
    _json: bytes

    def __post_init__(self):
        data = _decode(self._json)
        expected = {"format", "version", "status", "reasons", "attack", "calibration_sha256", "recipe_sha256",
                    "policy_sha256", "evaluation_sha256", "threat_sha256", "ledger_sha256", "artifact_inventory_sha256",
                    "raw_auc", "oriented_auc", "reversible_diagnostic_auc", "auc_bounds", "tpr", "fpr", "advantage",
                    "tpr_bounds", "fpr_bounds", "advantage_bounds", "low_fpr_target", "low_fpr_resolved",
                    "true_positives", "false_positives", "members", "nonmembers", "independent_units",
                    "attacker_prior", "confidence", "comparisons", "formal_dp", "legal_route", "release_authorized"}
        require(set(data) in (expected, expected | {"release_profile"})
                and data["format"] == "dope-privacy-evidence" and type(data["version"]) is int
                and data["version"] == 1 and data["attack"] == "calibrated_membership", "invalid_evidence")
        require(data["status"] == "inconclusive" and data["formal_dp"] == data["legal_route"] == "unverified"
                and data["release_authorized"] is False, "invalid_evidence")
        require(all(is_pin(v) for k, v in data.items() if k.endswith("sha256")), "invalid_evidence")
        allowed_reasons = {"custody_controls_and_policy_unqualified", "low_fpr_unresolved",
                           "cluster_or_release_inference_required", "access_threat_unmeasured",
                           "release_channels_unaccounted"}
        require(type(data["reasons"]) is list and bool(data["reasons"]) and all(type(x) is str and x in allowed_reasons
                for x in data["reasons"]) and len(set(data["reasons"])) == len(data["reasons"]), "invalid_evidence")
        for key in ("members", "nonmembers", "independent_units", "comparisons"):
            require(type(data[key]) is int and 1 <= data[key] <= 100_000, "invalid_evidence")
        for key, count in (("true_positives", "members"), ("false_positives", "nonmembers")):
            require(type(data[key]) is int and 0 <= data[key] <= data[count], "invalid_evidence")
        for key in ("raw_auc", "oriented_auc", "reversible_diagnostic_auc", "tpr", "fpr", "low_fpr_target",
                    "attacker_prior", "confidence"):
            require(number(data[key]) and 0 <= data[key] <= 1, "invalid_evidence")
        require(number(data["advantage"]) and -1 <= data["advantage"] <= 1, "invalid_evidence")
        require(type(data["low_fpr_resolved"]) is bool, "invalid_evidence")
        require(data["tpr"] == data["true_positives"] / data["members"]
                and data["fpr"] == data["false_positives"] / data["nonmembers"]
                and data["advantage"] == data["tpr"] - data["fpr"], "invalid_evidence")
        for key in ("auc_bounds", "tpr_bounds", "fpr_bounds", "advantage_bounds"):
            value = data[key]
            require(value is None or (type(value) is list and len(value) == 2 and all(number(x) for x in value)
                                     and (-1 if key == "advantage_bounds" else 0) <= value[0] <= value[1] <= 1), "invalid_evidence")
        intervals = [data[k] for k in ("auc_bounds", "tpr_bounds", "fpr_bounds", "advantage_bounds")]
        require(all(x is None for x in intervals) or all(x is not None for x in intervals), "invalid_evidence")
        resolved = data["fpr_bounds"] is not None and data["fpr_bounds"][1] <= data["low_fpr_target"]
        require(data["low_fpr_resolved"] == resolved, "invalid_evidence")
        if "release_profile" in data:
            from .release_profile import ReleaseProfile, local_empirical_profile
            profile = ReleaseProfile._from_value(data["release_profile"])
            require(profile.value == local_empirical_profile(), "invalid_evidence")

    @property
    def value(self):
        return _decode(self._json)

    @property
    def status(self):
        return EmpiricalPrivacyStatus(self.value["status"])

    @property
    def sha256(self):
        return hashlib.sha256(self._json).hexdigest()


@dataclass(frozen=True, slots=True, repr=False)
class AttackCalibration:
    direction: int
    threshold: float
    development: ScoreView
    recipe_sha256: str
    policy: DisclosurePolicy
    policy_sha256: str

    def __post_init__(self):
        require(type(self.direction) is int and self.direction in (-1, 1), "invalid_direction")
        require(number(self.threshold), "invalid_threshold")
        require(type(self.development) is ScoreView and self.development.role == "attack_development", "development_required")
        require(type(self.policy) is DisclosurePolicy and self.policy_sha256 == self.policy.sha256, "policy_identity_mismatch")
        require(is_pin(self.recipe_sha256) and self.recipe_sha256 == self.development.recipe_sha256, "recipe_identity_mismatch")

    @classmethod
    def freeze(cls, view, policy):
        require(type(view) is ScoreView and view.role == "attack_development", "development_required")
        require(type(policy) is DisclosurePolicy, "invalid_policy")
        direction = -1 if auc(view.members, view.nonmembers) < .5 else 1
        negative = sorted((direction * x for x in view.nonmembers), reverse=True)
        allowed = math.floor(policy.target_fpr * len(negative))
        threshold = math.nextafter(negative[allowed], math.inf)
        require(math.isfinite(threshold), "threshold_not_representable")
        return cls(direction, threshold, view, view.recipe_sha256, policy, policy.sha256)

    @property
    def sha256(self):
        return pin({"format": "dope-attack-calibration-v1", "direction": self.direction,
                    "threshold": self.threshold, "development": self.development.sha256,
                    "recipe": self.recipe_sha256, "policy": self.policy_sha256})

    def evaluate(self, view, threat):
        require(type(view) is ScoreView and view.role == "attack_evaluation", "evaluation_required")
        require(type(threat) is DisclosureThreatModel, "invalid_threat")
        require(view.recipe_sha256 == self.recipe_sha256, "recipe_identity_mismatch")
        require(self.policy.sha256 == self.policy_sha256, "policy_identity_mismatch")
        require(not set(self.development.member_ids + self.development.nonmember_ids).intersection(
            view.member_ids + view.nonmember_ids), "calibration_observation_overlap")
        require(not set(self.development.units).intersection(view.units), "calibration_unit_overlap")
        raw = auc(view.members, view.nonmembers)
        oriented = raw if self.direction == 1 else 1 - raw
        tp = sum(self.direction * x >= self.threshold for x in view.members)
        fp = sum(self.direction * x >= self.threshold for x in view.nonmembers)
        tpr, fpr = tp / len(view.members), fp / len(view.nonmembers)
        tpr_ci = fpr_ci = auc_ci = advantage_ci = None
        reasons = ["custody_controls_and_policy_unqualified"]
        if any(x is None for x in (threat.source_derived_lengths_sha256,
                                  threat.source_derived_statuses_sha256, threat.source_derived_metrics_sha256)):
            reasons.append("release_channels_unaccounted")
        if view.independent and threat.repeated_releases == 1:
            event_alpha = self.policy.alpha / (6 * self.policy.comparisons)
            tpr_ci = rate_bounds(tp, len(view.members), event_alpha)
            fpr_ci = rate_bounds(fp, len(view.nonmembers), event_alpha)
            # AUC pairs share examples. Bound per independent row, never per pair.
            radius = math.sqrt(.5 * (1 / len(view.members) + 1 / len(view.nonmembers)) * math.log(1 / event_alpha))
            auc_ci = (max(0., oriented - radius), min(1., oriented + radius))
            advantage_ci = (tpr_ci[0] - fpr_ci[1], tpr_ci[1] - fpr_ci[0])
            if fpr_ci[1] > self.policy.target_fpr:
                reasons.append("low_fpr_unresolved")
        else:
            reasons.append("cluster_or_release_inference_required")
        if threat.access != "row_only":
            reasons.append("access_threat_unmeasured")
        result = {"format": "dope-privacy-evidence", "version": 1, "status": "inconclusive", "reasons": reasons,
                  "attack": "calibrated_membership", "calibration_sha256": self.sha256,
                  "recipe_sha256": self.recipe_sha256, "policy_sha256": self.policy_sha256,
                  "evaluation_sha256": view.sha256, "threat_sha256": threat.sha256,
                  "ledger_sha256": threat.ledger_sha256, "artifact_inventory_sha256": threat.artifact_inventory_sha256,
                  "raw_auc": raw, "oriented_auc": oriented, "reversible_diagnostic_auc": max(raw, 1 - raw),
                  "auc_bounds": auc_ci, "tpr": tpr, "fpr": fpr, "advantage": tpr - fpr,
                  "tpr_bounds": tpr_ci, "fpr_bounds": fpr_ci, "advantage_bounds": advantage_ci,
                  "low_fpr_target": self.policy.target_fpr,
                  "low_fpr_resolved": fpr_ci is not None and fpr_ci[1] <= self.policy.target_fpr,
                  "true_positives": tp, "false_positives": fp,
                  "members": len(view.members), "nonmembers": len(view.nonmembers),
                  "independent_units": len(set(view.units)), "attacker_prior": self.policy.attacker_prior,
                  "confidence": 1 - self.policy.alpha, "comparisons": self.policy.comparisons,
                  "formal_dp": "unverified", "legal_route": "unverified", "release_authorized": False}
        from .release_profile import local_empirical_profile
        result["release_profile"] = local_empirical_profile()
        return AttackEvidence(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def uncalibrated(view):
    require(type(view) is ScoreView, "invalid_scores")
    raw = auc(view.members, view.nonmembers)
    from .release_profile import local_empirical_profile
    return {"status": "inconclusive", "reason": "direction_evidence_missing", "raw_auc": raw,
            "reversible_diagnostic_auc": max(raw, 1 - raw), "formal_dp": "unverified", "release_authorized": False,
            "release_profile": local_empirical_profile()}
