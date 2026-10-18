"""Preregistered prior-decoder admission and deterministic byte-accounting pilot.

The pilot validates accounting arithmetic only. It does not load a vendor model,
fit a decoder, admit license rights, or provide scientific transfer evidence.
"""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
PLAN_PATH = Path(__file__).with_name("prior_decoder_study.json")
LOCK_PATH = ROOT / "research/transfer/locks/prior-decoder.json"
PILOT_PATH = Path(__file__).with_name("prior_decoder_pilot.json")
ROLES = {"shared_learned_state", "sampling_payload", "transform", "dictionary", "runtime"}


class StudyError(ValueError):
    """Stable, source-free refusal for an invalid study plan or fixture."""


def _require(condition, code):
    if not condition:
        raise StudyError(code)


def _int(value):
    return type(value) is int and value >= 0


def closure_bytes(components, manifest_bytes, consumers):
    """Return cold and amortized full sampler closure including runtime bytes."""
    _require(_int(manifest_bytes) and type(consumers) is int and consumers >= 1, "invalid_accounting_input")
    _require(type(components) is list and bool(components), "invalid_accounting_input")
    cold = manifest_bytes
    shared = 0
    for component in components:
        _require(type(component) is dict and set(component) == {"role", "bytes"}, "invalid_component")
        _require(component["role"] in ROLES and _int(component["bytes"]), "invalid_component")
        cold += component["bytes"]
        if component["role"] == "shared_learned_state":
            shared += component["bytes"]
    amortized = cold - shared + (shared + consumers - 1) // consumers
    return {"cold_bytes": cold, "amortized_bytes": amortized, "shared_bytes": shared}


def validate_plan(plan, lock):
    _require(plan.get("format") == "dope-prior-relative-decoder-study-v1" and plan.get("version") == 1,
             "invalid_plan")
    _require(plan.get("state") == "preregistered_blocked", "study_must_remain_blocked")
    _require(plan["teacher"].get("role") == "distillation_teacher_only" and
             plan["teacher"].get("may_score_its_own_decoder") is False, "teacher_role_mismatch")
    _require(plan["model_holdouts"]["excluded_from_confirmation"] ==
             ["TabICLv2 teacher", "decoder student"], "confirmatory_model_overlap")
    _require(plan["execution"] == {"vendor_study_started": False, "fixture_pilot_status": "pending"},
             "execution_state_mismatch")
    _require(plan["cost_cap"]["fixture_pilot"] == {
        "usd": 0, "cpu_wall_seconds": 10, "gpu_seconds": 0, "external_download_bytes": 0, "runs": 1
    }, "fixture_cost_cap_mismatch")
    _require(plan["cost_cap"]["vendor_study"]["status"] == "not_admitted" and
             plan["cost_cap"]["vendor_study"]["runs"] == 0, "vendor_execution_not_blocked")
    caps = plan["byte_comparison"]
    limit = caps["cold_closure_limit_bytes"]
    manifest = caps["manifest_ceiling_bytes"]
    _require(_int(limit) and _int(manifest), "invalid_byte_cap")
    candidates = plan["candidates"]
    _require([item["id"] for item in candidates] == ["prior_distilled_decoder", "no_prior_decoder"],
             "candidate_set_mismatch")
    summaries = {}
    for candidate in candidates:
        components = candidate["components"]
        _require(all(item.get("role") in ROLES and _int(item.get("ceiling_bytes")) for item in components),
                 "invalid_candidate_closure")
        ceiling = sum(item["ceiling_bytes"] for item in components) + manifest
        _require(ceiling == candidate["closure_ceiling_bytes"] == limit, "candidate_ceiling_mismatch")
        summaries[candidate["id"]] = ceiling
    amortizations = caps["amortization_counts"]
    _require(amortizations == [1, 16, 128], "amortization_schedule_mismatch")
    _require(caps["candidate_byte_ceiling_equal_at_each_amortization_count"] is True, "unequal_candidate_caps")
    rights = lock["teacher"]
    for field in ("code_license_review", "checkpoint_license_review", "distillation_permission",
                  "output_redistribution_permission", "pretraining_source_independence"):
        _require(rights[field] == "unresolved", "admission_blocker_dropped")
    _require(lock["status"] == "unqualified" and lock["decoder"]["confirmatory_auditor_eligible"] is False,
             "unqualified_lock_changed")
    return summaries


def deterministic_fixture_pilot():
    """Exercise complete closure accounting with synthetic byte quantities."""
    manifest_bytes = 1
    candidates = {
        "prior_distilled_decoder": [
            {"role": "shared_learned_state", "bytes": 8},
            {"role": "sampling_payload", "bytes": 12},
            {"role": "transform", "bytes": 4},
            {"role": "runtime", "bytes": 8},
        ],
        "no_prior_decoder": [
            {"role": "sampling_payload", "bytes": 16},
            {"role": "transform", "bytes": 4},
            {"role": "runtime", "bytes": 12},
        ],
    }
    result = {}
    for name, components in candidates.items():
        result[name] = {count: closure_bytes(components, manifest_bytes, count)
                        for count in (1, 16, 128)}
    _require(result["prior_distilled_decoder"][1]["cold_bytes"] == 33 and
             result["no_prior_decoder"][1]["cold_bytes"] == 33, "cold_closure_fixture_mismatch")
    _require(result["prior_distilled_decoder"][16]["amortized_bytes"] == 26 and
             result["no_prior_decoder"][16]["amortized_bytes"] == 33, "amortized_closure_fixture_mismatch")
    _require(result["prior_distilled_decoder"][128]["amortized_bytes"] == 26, "large_amortization_fixture_mismatch")
    return result


def run_fixture_pilot():
    plan = json.loads(PLAN_PATH.read_bytes())
    lock = json.loads(LOCK_PATH.read_bytes())
    plan_digest = hashlib.sha256(PLAN_PATH.read_bytes()).hexdigest()
    _require(lock["study_manifest_sha256"] == plan_digest, "study_manifest_pin_mismatch")
    caps = validate_plan(plan, lock)
    fixture = deterministic_fixture_pilot()
    pilot_digest = hashlib.sha256(PILOT_PATH.read_bytes()).hexdigest()
    _require(lock["fixture_pilot"]["receipt_sha256"] == pilot_digest, "pilot_receipt_pin_mismatch")
    pilot = json.loads(PILOT_PATH.read_bytes())
    normalized_fixture = json.loads(json.dumps(fixture))
    _require(pilot["status"] == "passed" and pilot["vendor_model_loaded"] is False and
             pilot["external_spend_usd"] == 0 and pilot["synthetic_fixture_accounting"] == normalized_fixture,
             "pilot_receipt_mismatch")
    return {
        "status": "fixture_pilot_passed_vendor_study_blocked",
        "study_manifest_sha256": plan_digest,
        "pilot_receipt_sha256": pilot_digest,
        "candidate_closure_ceilings": caps,
        "synthetic_fixture_accounting": fixture,
        "vendor_execution": False,
        "scientific_result": "unmeasured",
        "admission_blockers": len(plan["admission_blockers"]),
    }


if __name__ == "__main__":
    print(json.dumps(run_fixture_pilot(), sort_keys=True, separators=(",", ":")))
