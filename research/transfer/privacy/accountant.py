"""Complete-ledger DP accounting; default-deny proof audit, no release authority.

Per-step costs are proof obligations, not inferred guarantees of a noise label.
The trusted audit adapter must authenticate their derivation and all private
access, including earlier releases and adaptive choices. This module reads no
source data and cannot discover private accesses omitted by an external caller.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from jsonschema import Draft202012Validator

from .attacks import PrivacyError, number, require


STAGES = frozenset(("domain_discovery", "binning", "winsorization_clipping_bounds",
                    "transform_fitting", "structure_selection", "parameter_fitting", "candidate_selection",
                    "source_derived_lengths", "source_derived_statuses", "source_derived_metrics",
                    "post_summary_private_reads"))


def deny_audit(_):
    return False


def _snapshot(value):
    try:
        data = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        require(len(data) <= 8_000_000, "invalid_dp_profile")
        return json.loads(data), hashlib.sha256(data).hexdigest()
    except (ValueError, TypeError, UnicodeError, RecursionError):
        raise PrivacyError("invalid_dp_profile") from None


def _profile(value):
    # Snapshot before validation; neither a callback nor a caller can mutate it.
    value, digest = _snapshot(value)
    try:
        schema_bytes = (Path(__file__).resolve().parents[3] / "schemas/dp-profile-v1.schema.json").read_bytes()
        validator = Draft202012Validator(json.loads(schema_bytes))
        require(validator.is_valid(value), "invalid_dp_profile")
        source_digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    except (OSError, ValueError, TypeError, RecursionError):
        raise PrivacyError("invalid_dp_profile") from None
    return value, digest, source_digest, hashlib.sha256(schema_bytes).hexdigest()


def _up(value):
    return math.nextafter(value, math.inf) if value else 0.


def _sum(values):
    return _up(math.fsum(values))


def _compose(profile, costs):
    method = profile["composition"]["accountant"]
    delta = profile["budget"]["delta"]
    if not costs:
        return 0., 0.
    if method == "pld":
        # Conservative sequential envelope of independently proved PLD bounds;
        # no convolution, sampling amplification, or independence assumption.
        require(all(set(c) == {"epsilon", "delta"} for c in costs), "invalid_dp_cost")
        return _sum(c["epsilon"] for c in costs), _sum(c["delta"] for c in costs)
    require(delta > 0, "dp_conversion_delta_required")
    log_delta = _up(-math.log(delta))
    if method == "zcdp":
        require(all(set(c) == {"rho"} for c in costs), "invalid_dp_cost")
        rho = _sum(c["rho"] for c in costs)
        return _up(rho + _up(2 * _up(math.sqrt(_up(rho * log_delta))))), delta
    orders = profile["composition"]["rdp_orders"]
    require(all(set(c) == {"rdp"} and len(c["rdp"]) == len(orders)
                and any(x > 0 for x in c["rdp"]) for c in costs), "invalid_dp_cost")
    # Mironov, Proposition 1 (adaptive addition) and Proposition 3 (conversion).
    epsilon = min(_up(_sum(c["rdp"][i] for c in costs) + _up(log_delta / (order - 1)))
                  for i, order in enumerate(orders))
    return epsilon, delta


def account(profile, *, expected_adjacency, independent_audit_passed=False,
            empirical_screened_only=False, audit_verifier=deny_audit):
    """Assess one complete profile/history, never incrementally reset a budget.

    expected_adjacency is the protected unit contract from the trusted caller.
    independent_audit_passed must be exactly True AND the trusted verifier must
    authenticate a signoff for the supplied digest context. JSON flags or an
    empirical screen alone cannot establish formal_dp_verified. A fixture audit
    callback exercises control flow only; it grants no product authority.
    """
    require(type(independent_audit_passed) is bool and type(empirical_screened_only) is bool,
            "invalid_dp_assessment")
    profile, digest, source_digest, schema_digest = _profile(profile)
    expected, _ = _snapshot(expected_adjacency)
    require(type(expected) is dict and set(expected) == {"unit", "relation", "max_rows_per_unit"}
            and type(expected["unit"]) is str and expected["unit"] in {"row", "entity", "user_group"}
            and type(expected["relation"]) is str and expected["relation"] in {"add_remove", "replace_one"}
            and type(expected["max_rows_per_unit"]) is int and 1 <= expected["max_rows_per_unit"] <= 1_000_000
            and (expected["unit"] != "row" or expected["max_rows_per_unit"] == 1), "invalid_dp_adjacency")
    reasons = set()
    steps = profile["steps"]
    if profile["adjacency"] != expected or any(s["adjacency"] != expected for s in steps):
        reasons.add("adjacency_mismatch")
    if {s["stage"] for s in steps} != STAGES:
        reasons.add("incomplete_private_read_scope")
    if any(s["access"] == "unaccounted_private" or
           (s["access"] == "accounted_dp" and s["cost"] is None) for s in steps):
        reasons.add("unaccounted_private_read")
    if profile["bounds"]["provenance"] == "accounted_dp" and not any(
            s["stage"] == "winsorization_clipping_bounds" and s["access"] == "accounted_dp"
            and s["cost"] is not None for s in steps):
        reasons.add("unaccounted_private_bounds")
    costs = [s["cost"] for s in steps if s["access"] == "accounted_dp" and s["cost"] is not None]
    epsilon = delta = None
    try:
        epsilon, delta = _compose(profile, costs)
        require(number(epsilon) and number(delta), "invalid_dp_cost")
    except PrivacyError as error:
        reasons.add(str(error))
    except (ArithmeticError, ValueError):
        reasons.add("invalid_dp_cost")
    composition_passed = (epsilon is not None and delta is not None
                          and math.isfinite(epsilon) and math.isfinite(delta)
                          and epsilon <= profile["budget"]["epsilon"] and delta <= profile["budget"]["delta"])
    if not composition_passed:
        reasons.add("dp_budget_exceeded" if epsilon is not None else "dp_composition_unavailable")
    # No callback is consulted after a private-read, scope or arithmetic refusal.
    audit_passed = False
    if not reasons and not empirical_screened_only and independent_audit_passed:
        if profile["independent_cryptographic_audit"] == "independently_audited":
            context = {"profile_sha256": digest, "accountant_sha256": source_digest,
                       "schema_sha256": schema_digest, "epsilon": epsilon, "delta": delta}
            try:
                audit_passed = audit_verifier(context) is True
            except Exception:
                audit_passed = False
    not_dp = bool(reasons)
    if empirical_screened_only:
        reasons.add("empirical_screening_is_not_proof")
    if not audit_passed:
        reasons.add("independent_audit_required")
    verified = not not_dp and audit_passed and not empirical_screened_only
    status = ("not_dp" if not_dp else "verified" if verified else
              "empirical_screened_only" if empirical_screened_only else "unverified_design")
    return {"formal_dp_verified": verified, "formal_dp_status": status,
            "composition_passed": composition_passed, "independent_audit_passed": audit_passed,
            "epsilon": epsilon if epsilon is not None and math.isfinite(epsilon) else None,
            "delta": delta if delta is not None and math.isfinite(delta) else None,
            "profile_sha256": digest, "reasons": sorted(reasons), "release_authorized": False}
