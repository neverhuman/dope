"""Pure full-pilot contract checks, without scientific execution authority.

These immutable metadata contracts do not load sources, models or evidence files.
Gate declarations still need owner-authenticated evidence resolution, trusted
signatures and actual runtime validation before a future pilot runner can launch.
Historical planning/native/confirmation APIs retain their existing semantics.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re

from research.transfer.campaign_runner import STAGES
from research.transfer.evidence import EvidenceError, canonical_json, parse_json

MAX_BYTES = 1024 * 1024
MAX_INT = 2**53 - 1
PLAN_FORMAT = "dope-pilot-execution-contract"
ADMISSION_FORMAT = "dope-pilot-admission-contract"
GATES = ("observational_source_rights", "mechanism_source_rights",
         "model_rights_and_native_parity", "source_ownership_and_collision_review",
         "resource_budget_and_owner_admission", "protected_owner_admission",
         "modern_panel_and_scheduler_admission")
PIN_FIELDS = ("source_commit", "source_tree", "environment_sha256", "policy_sha256",
              "schedule_sha256", "operation_dag_sha256")
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_SEED = re.compile(r"0x[0-9a-f]{16}\Z")
_ID = re.compile(r"[a-z][a-z0-9_]{0,63}\Z")
_STREAMS = ("fit", "sample_primary", "sample_4n", "context", "learner")


class PilotContractError(ValueError):
    """Static codes only: never include source values, headers or paths."""


def _require(ok, code):
    if not ok:
        raise PilotContractError(code)


def _fields(value, fields, code):
    _require(type(value) is dict and set(value) == set(fields), code)


def _integer(value, minimum=1):
    return type(value) is int and minimum <= value <= MAX_INT


def _identifier(value):
    return type(value) is str and _ID.fullmatch(value) is not None


def _hash(value):
    return type(value) is str and _HASH.fullmatch(value) is not None


def _capture(value):
    try:
        encoded = canonical_json(value)
        _require(len(encoded) <= MAX_BYTES, "pilot_contract_size_limit")
        return encoded, parse_json(encoded)
    except PilotContractError:
        raise
    except (EvidenceError, TypeError, ValueError, UnicodeError):
        raise PilotContractError("pilot_contract_invalid") from None


def seed_uint64(value, *, master=False):
    _require(type(value) is str and _SEED.fullmatch(value) is not None,
             "pilot_seed_invalid")
    result = int(value[2:], 16)
    _require(not master or result <= MAX_INT, "pilot_master_seed_out_of_range")
    return result


def _pins(value):
    _fields(value, PIN_FIELDS, "pilot_pins_invalid")
    _require(all(type(value[k]) is str and _COMMIT.fullmatch(value[k])
                 for k in ("source_commit", "source_tree")), "pilot_pins_invalid")
    _require(all(_hash(value[k]) for k in PIN_FIELDS[2:]), "pilot_pins_invalid")


def _unique_ids(values, code):
    _require(type(values) is list and 0 < len(values) <= 128
             and all(_identifier(v) for v in values)
             and len(values) == len(set(values)), code)


def _validate_plan(value):
    _fields(value, ("format", "version", "operation", "campaign_id", "pins", "tasks",
                    "methods", "byte_caps", "learners", "contexts", "seeds", "stages",
                    "sample_policies", "budget"), "pilot_plan_invalid")
    _require(value["format"] == PLAN_FORMAT and type(value["version"]) is int
             and value["version"] == 1 and value["operation"] == "development_pilot",
             "pilot_operation_mismatch")
    _require(_hash(value["campaign_id"]), "pilot_campaign_invalid")
    _pins(value["pins"])
    tasks = value["tasks"]
    _require(type(tasks) is list and len(tasks) == 2, "pilot_task_roles_invalid")
    roles, identities = set(), set()
    for task in tasks:
        _fields(task, ("task_id", "role", "fit_rows", "primary_n", "input_sha256"),
                "pilot_task_invalid")
        _require(_hash(task["task_id"]) and _hash(task["input_sha256"])
                 and _integer(task["fit_rows"]) and _integer(task["primary_n"])
                 and task["primary_n"] <= MAX_INT // 4, "pilot_task_invalid")
        _require(task["role"] in ("observational", "adverse_mechanism"),
                 "pilot_task_roles_invalid")
        roles.add(task["role"])
        identities.add(task["task_id"])
    _require(len(identities) == 2 and len(roles) == 2, "pilot_task_roles_invalid")
    _require(value["methods"] == ["dope", "gaussian_copula", "compressed_real_rows"],
             "pilot_controls_incomplete")
    caps = value["byte_caps"]
    _require(type(caps) is list and len(caps) == 2
             and all(_integer(x) for x in caps) and caps[0] < caps[1],
             "pilot_byte_caps_invalid")
    _unique_ids(value["learners"], "pilot_learners_invalid")
    contexts = value["contexts"]
    _require(type(contexts) is list and 0 < len(contexts) <= 16, "pilot_contexts_invalid")
    names = []
    for context in contexts:
        _fields(context, ("context_id", "rows"), "pilot_contexts_invalid")
        _require(_identifier(context["context_id"]) and _integer(context["rows"])
                 and all(context["rows"] <= min(task["fit_rows"], task["primary_n"]) for task in tasks),
                 "pilot_contexts_invalid")
        names.append(context["context_id"])
    _require(len(names) == len(set(names)), "pilot_contexts_invalid")
    _fields(value["seeds"], ("encoding", "master", "substreams"), "pilot_seeds_invalid")
    _require(value["seeds"]["encoding"] == "explicit-hex-uint64-v1", "pilot_seeds_invalid")
    seed_uint64(value["seeds"]["master"], master=True)
    streams = value["seeds"]["substreams"]
    _fields(streams, _STREAMS, "pilot_seeds_invalid")
    values = [seed_uint64(streams[k]) for k in _STREAMS]
    _require(len(set(values)) == len(values), "pilot_seed_alias")
    _require(value["stages"] == list(STAGES), "pilot_stages_incomplete")
    _require(value["sample_policies"] == ["primary_n", "sensitivity_4n"],
             "pilot_sample_policies_invalid")
    budget = value["budget"]
    _fields(budget, ("wall_ms", "cpu_ms", "peak_ram_bytes", "disk_bytes", "max_attempts"),
            "pilot_budget_invalid")
    _require(all(_integer(x) for x in budget.values()) and budget["max_attempts"] <= 16,
             "pilot_budget_invalid")


@dataclass(frozen=True)
class PilotExecutionPlan:
    _encoded: bytes

    @classmethod
    def from_document(cls, document):
        encoded, value = _capture(document)
        _validate_plan(value)
        return cls(encoded)

    @classmethod
    def from_bytes(cls, encoded):
        _require(type(encoded) is bytes and len(encoded) <= MAX_BYTES,
                 "pilot_contract_size_limit")
        try:
            return cls.from_document(parse_json(encoded))
        except EvidenceError:
            raise PilotContractError("pilot_contract_invalid") from None

    def document(self):
        return parse_json(self._encoded)

    @property
    def sha256(self):
        return hashlib.sha256(self._encoded).hexdigest()

    def summary(self):
        value = PilotExecutionPlan.from_bytes(self._encoded).document()
        base = len(value["tasks"]) * len(value["methods"]) * len(value["byte_caps"])
        return {"base_cells": base, "sampling_cells": base * 2,
                "learner_context_coordinates": base * 2 * len(value["learners"]) * len(value["contexts"]),
                "physical_operations": None, "execution_authorized": False,
                "scientific_acceptance": False, "confirmation_allowed": False}


def inspect_admission_contract(plan, document, *, expected_pins, required_learners):
    """Check metadata only; accepted gate declarations are not trusted grants.

    The owner must independently resolve and authenticate the referenced evidence,
    signatures and live runtime. This API never returns an execution capability.
    """
    _require(type(plan) is PilotExecutionPlan, "pilot_plan_invalid")
    plan = PilotExecutionPlan.from_bytes(plan._encoded)
    _, value = _capture(document)
    _fields(value, ("format", "version", "operation", "plan_sha256", "pins", "gates"),
            "pilot_admission_invalid")
    _require(value["format"] == ADMISSION_FORMAT and type(value["version"]) is int
             and value["version"] == 1 and value["operation"] == "development_pilot",
             "pilot_operation_mismatch")
    _require(value["plan_sha256"] == plan.sha256, "pilot_plan_binding_mismatch")
    _pins(value["pins"])
    _pins(expected_pins)
    _require(value["pins"] == plan.document()["pins"] == expected_pins,
             "pilot_runtime_binding_mismatch")
    _unique_ids(required_learners, "pilot_required_roster_invalid")
    _require(set(plan.document()["learners"]) == set(required_learners),
             "pilot_required_roster_mismatch")
    _fields(value["gates"], GATES, "pilot_gates_incomplete")
    missing = []
    for gate_id in GATES:
        gate = value["gates"][gate_id]
        _fields(gate, ("status", "evidence_sha256"), "pilot_gate_invalid")
        _require(gate["status"] in ("accepted_for_operation", "missing", "rejected", "scope_mismatch"),
                 "pilot_gate_invalid")
        _require(gate["evidence_sha256"] is None or _hash(gate["evidence_sha256"]),
                 "pilot_gate_invalid")
        if gate["status"] == "accepted_for_operation":
            _require(_hash(gate["evidence_sha256"]), "pilot_gate_evidence_missing")
        else:
            missing.append({"gate_id": gate_id, "status": gate["status"]})
    return {"status": "metadata_complete_requires_owner_verification" if not missing else "blocked",
            "plan_sha256": plan.sha256, "missing_gates": missing,
            "evidence_authenticated": False, "runtime_observed": False,
            "execution_authorized": False, "scientific_acceptance": False,
            "confirmation_allowed": False, "release_authorized": False}
