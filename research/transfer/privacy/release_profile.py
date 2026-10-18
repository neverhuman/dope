"""Bounded mirror of Rust release profiles; no local result grants authority.

The verifier is supplied by the trusted evidence/rights-owner boundary, never
by decoded JSON. It must resolve receipts, power, scope and composition. The
default verifier rejects every positive claim. Product authority stays in Rust.
"""
from __future__ import annotations

from dataclasses import dataclass

from .attacks import _decode, is_pin, number, pin, require, PrivacyError


AXES = {
    "utility": ("utility_qualification_state", "passed", "failed"),
    "empirical_privacy": ("empirical_privacy_state", "screened", "failed"),
    "formal_dp": ("formal_dp_state", "verified", "failed"),
    "legal_release": ("legal_release_state", "authorized", "denied"),
}
ACCOUNTING = {"artifact_inventory_sha256", "release_history_sha256", "source_derived_lengths_sha256",
              "source_derived_statuses_sha256", "source_derived_metrics_sha256", "release_count"}
DETAILS = {"utility": set(), "empirical_privacy": {"threat_model_sha256", "attack_power_sha256"},
           "formal_dp": {"mechanism_sha256", "accountant_sha256", "proof_sha256", "epsilon", "delta"},
           "legal_release": {"rights_owner_sha256"}}
SURFACES = {"cli", "service", "ui", "report", "export"}


def deny_evidence(_):
    return False


def local_empirical_profile():
    return ReleaseProfile.local_research().begin("empirical_privacy").empirical_inconclusive().value


def _accounting(value):
    require(type(value) is dict and set(value) == ACCOUNTING, "invalid_release_profile")
    require(type(value["release_count"]) is int and 1 <= value["release_count"] <= 100_000,
            "invalid_release_profile")
    require(all(is_pin(v) for k, v in value.items() if k != "release_count"), "invalid_release_profile")


def _evidence(value):
    require(type(value) is dict and type(value.get("axis")) is str and value["axis"] in AXES,
            "invalid_release_profile")
    require(set(value) == {"axis", "receipt_sha256", "accounting_sha256"} | DETAILS[value["axis"]],
            "invalid_release_profile")
    require(all(is_pin(v) for k, v in value.items() if k not in {"axis", "epsilon", "delta"}),
            "invalid_release_profile")
    if value["axis"] == "formal_dp":
        require(number(value["epsilon"]) and value["epsilon"] > 0
                and number(value["delta"]) and 0 <= value["delta"] < 1, "invalid_release_profile")


def _validate(value):
    fields = {x[0] for x in AXES.values()}
    require(set(value) == {"format", "version", "accounting", "evidence"} | fields, "invalid_release_profile")
    require(value["format"] == "dope-release-profile" and type(value["version"]) is int
            and value["version"] == 1, "invalid_release_profile")
    for axis, (field, positive, negative) in AXES.items():
        allowed = {"unassessed", "pending", positive, negative}
        if axis == "empirical_privacy":
            allowed.add("inconclusive")
        require(type(value[field]) is str and value[field] in allowed, "invalid_release_profile")
    if value["accounting"] is not None:
        _accounting(value["accounting"])
    require(type(value["evidence"]) is list and len(value["evidence"]) <= 4, "invalid_release_profile")
    axes = set()
    for evidence in value["evidence"]:
        _evidence(evidence)
        require(evidence["axis"] not in axes and value["accounting"] is not None, "invalid_release_profile")
        axes.add(evidence["axis"])
        require(evidence["accounting_sha256"] == pin(value["accounting"]), "invalid_release_profile")
    require(axes == {axis for axis, (field, positive, _) in AXES.items() if value[field] == positive},
            "invalid_release_profile")


@dataclass(frozen=True, slots=True, repr=False)
class ReleaseProfile:
    _json: bytes

    def __post_init__(self):
        require(type(self._json) is bytes and len(self._json) <= 65_536, "invalid_release_profile")
        try:
            _validate(_decode(self._json))
        except PrivacyError:
            raise PrivacyError("invalid_release_profile") from None

    @classmethod
    def _from_value(cls, value):
        import json
        try:
            data = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        except (TypeError, ValueError, UnicodeError, RecursionError):
            raise PrivacyError("invalid_release_profile") from None
        return cls(data)

    @classmethod
    def local_research(cls, accounting=None):
        value = {field: "unassessed" for field, _, _ in AXES.values()}
        return cls._from_value({"format": "dope-release-profile", "version": 1, **value,
                                "accounting": accounting, "evidence": []})

    @property
    def value(self):
        # A fresh copy prevents decoded dictionaries from changing an assessment.
        return _decode(self._json)

    def begin(self, axis):
        require(type(axis) is str and axis in AXES, "invalid_release_profile")
        value = self.value
        value[AXES[axis][0]] = "pending"
        value["evidence"] = [x for x in value["evidence"] if x["axis"] != axis]
        return self._from_value(value)

    def complete(self, evidence, verifier=deny_evidence):
        _evidence(evidence)
        value = self.value
        field, positive, _ = AXES[evidence["axis"]]
        require(value[field] == "pending", "invalid_release_profile")
        value[field] = positive
        value["evidence"].append(evidence)
        result = self._from_value(value)
        result.validate(verifier)
        return result

    def reject(self, axis):
        require(type(axis) is str and axis in AXES, "invalid_release_profile")
        value = self.value
        field, _, negative = AXES[axis]
        require(value[field] == "pending", "invalid_release_profile")
        value[field] = negative
        return self._from_value(value)

    def empirical_inconclusive(self):
        value = self.value
        require(value["empirical_privacy_state"] == "pending", "invalid_release_profile")
        value["empirical_privacy_state"] = "inconclusive"
        return self._from_value(value)

    def validate(self, verifier=deny_evidence):
        for evidence in self.value["evidence"]:
            try:
                valid = verifier(evidence) is True
            except Exception:
                raise PrivacyError("release_evidence_unverified") from None
            require(valid, "release_evidence_unverified")

    def view(self, surface, require_formal_dp=False, verifier=deny_evidence):
        require(type(surface) is str and surface in SURFACES and type(require_formal_dp) is bool,
                "invalid_release_profile")
        self.validate(verifier)
        value = self.value
        authorized = (value["utility_qualification_state"] == "passed"
                      and value["empirical_privacy_state"] == "screened"
                      and value["legal_release_state"] == "authorized"
                      and (value["formal_dp_state"] == "verified"
                           or (not require_formal_dp and value["formal_dp_state"] == "unassessed")))
        require(surface != "export" or authorized, "release_authorization_required")
        return {"profile": value, "release_authorized": authorized,
                "label": "scoped_release" if authorized else "local_research"}

    def check_claim(self, label, epsilon=None, verifier=deny_evidence):
        view = self.view("report", epsilon is not None, verifier)
        value = self.value
        require(epsilon is None or (number(epsilon) and any(e["axis"] == "formal_dp" and e["epsilon"] == epsilon
                                                          for e in value["evidence"])), "invalid_release_profile")
        labels = {"local_research": True, "scoped_release": view["release_authorized"]}
        labels.update({label: value[AXES[axis][0]] == AXES[axis][1] for axis, label in (
            ("utility", "utility_qualified"), ("empirical_privacy", "empirical_screened"),
            ("formal_dp", "formal_dp_verified"), ("legal_release", "organization_authorized"))})
        require(type(label) is str and labels.get(label) is True, "invalid_release_profile")
