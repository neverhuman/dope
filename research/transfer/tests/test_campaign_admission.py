"""Public metadata-only contracts: these fixtures never admit a real campaign."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import unittest
from unittest import mock

import jsonschema

from research.transfer.campaign_admission import (
    ADMISSION_FORMAT, GATES, PLAN_FORMAT, PIN_FIELDS, PilotContractError,
    PilotExecutionPlan, inspect_admission_contract, seed_uint64,
)
from research.transfer.campaign_runner import STAGES

ROOT = Path(__file__).resolve().parents[3]


def fixture():
    return {"format": PLAN_FORMAT, "version": 1, "operation": "development_pilot",
            "campaign_id": "a" * 64,
            "pins": {k: "1" * (40 if k in ("source_commit", "source_tree") else 64)
                     for k in PIN_FIELDS},
            "tasks": [{"task_id": "2" * 64, "role": "observational", "fit_rows": 512,
                       "primary_n": 256, "input_sha256": "3" * 64},
                      {"task_id": "4" * 64, "role": "adverse_mechanism", "fit_rows": 384,
                       "primary_n": 128, "input_sha256": "5" * 64}],
            "methods": ["dope", "gaussian_copula", "compressed_real_rows"],
            "byte_caps": [4096, 32768],
            "learners": ["kumo_tabular", "tabicl", "mitra", "gradient_boosting", "neural"],
            "contexts": [{"context_id": "standard", "rows": 128},
                         {"context_id": "stratified_anchor", "rows": 64}],
            "seeds": {"encoding": "explicit-hex-uint64-v1", "master": "0x000d0beef2026100",
                      "substreams": {k: f"0x{3672295587602689 + i:016x}" for i, k in enumerate(
                          ("fit", "sample_primary", "sample_4n", "context", "learner"))}},
            "stages": list(STAGES), "sample_policies": ["primary_n", "sensitivity_4n"],
            "budget": {"wall_ms": 600000, "cpu_ms": 600000, "peak_ram_bytes": 1 << 30,
                       "disk_bytes": 1 << 24, "max_attempts": 1}}


def declaration(plan):
    return {"format": ADMISSION_FORMAT, "version": 1, "operation": "development_pilot",
            "plan_sha256": plan.sha256, "pins": plan.document()["pins"],
            "gates": {g: {"status": "accepted_for_operation", "evidence_sha256": "6" * 64}
                      for g in GATES}}


def inspect(plan, document, **kwargs):
    return inspect_admission_contract(plan, document,
            expected_pins=kwargs.get("expected_pins", plan.document()["pins"]),
            required_learners=kwargs.get("required_learners", fixture()["learners"]))


class PilotAdmissionContractTests(unittest.TestCase):
    def test_schema_valid_two_task_contract_has_explicit_240_denominator(self):
        doc = fixture()
        plan = PilotExecutionPlan.from_document(doc)
        for name, value in (("pilot-execution-v1", doc),
                            ("pilot-admission-v1", declaration(plan))):
            schema = json.loads((ROOT / "schemas" / (name + ".schema.json")).read_text())
            jsonschema.Draft202012Validator.check_schema(schema)
            jsonschema.Draft202012Validator(schema).validate(value)
        summary = plan.summary()
        self.assertEqual((summary["base_cells"], summary["sampling_cells"],
                          summary["learner_context_coordinates"]), (12, 24, 240))
        self.assertIsNone(summary["physical_operations"])
        self.assertFalse(summary["execution_authorized"])

    def test_self_asserted_accepted_gates_never_authorize_execution(self):
        plan = PilotExecutionPlan.from_document(fixture())
        report = inspect(plan, declaration(plan))
        self.assertEqual(report["status"], "metadata_complete_requires_owner_verification")
        for name in ("evidence_authenticated", "runtime_observed", "execution_authorized",
                     "scientific_acceptance", "confirmation_allowed", "release_authorized"):
            self.assertIs(report[name], False)

    def test_accepted_contract_is_immutable_despite_caller_mutation(self):
        doc = fixture()
        plan = PilotExecutionPlan.from_document(doc)
        before = plan.sha256
        doc["byte_caps"][0] = 1
        projection = plan.document()
        projection["tasks"][0]["primary_n"] = 1
        self.assertEqual(plan.sha256, before)
        self.assertEqual(plan.document()["byte_caps"], [4096, 32768])

    def test_invalid_hex_seed_placeholders_and_unsafe_master_refused(self):
        for value in ("0x000d0pe202610190", "0x000d0beef202610", "000d0beef20261000",
                      "0x000D0BEEF2026100", 1, True, "0x10000000000000000"):
            with self.subTest(value=value), self.assertRaisesRegex(PilotContractError, "pilot_seed_invalid"):
                seed_uint64(value)
        self.assertEqual(seed_uint64("0xffffffffffffffff"), 2**64 - 1)
        self.assertEqual(seed_uint64("0x001fffffffffffff", master=True), 2**53 - 1)
        with self.assertRaisesRegex(PilotContractError, "pilot_master_seed_out_of_range"):
            seed_uint64("0x0020000000000000", master=True)

    def test_required_controls_stages_and_sample_policies_cannot_be_omitted(self):
        for key in ("methods", "stages", "sample_policies"):
            doc = fixture()
            doc[key].pop()
            with self.subTest(key=key), self.assertRaises(PilotContractError):
                PilotExecutionPlan.from_document(doc)

    def test_context_cannot_exceed_actual_task_fit_or_generated_rows(self):
        doc = fixture()
        doc["contexts"][0]["rows"] = 129
        with self.assertRaisesRegex(PilotContractError, "pilot_contexts_invalid"):
            PilotExecutionPlan.from_document(doc)

    def test_each_runtime_pin_and_budget_change_breaks_existing_admission(self):
        original = PilotExecutionPlan.from_document(fixture())
        admission = declaration(original)
        for key in PIN_FIELDS:
            changed = fixture()
            changed["pins"][key] = "7" * len(changed["pins"][key])
            with self.subTest(key=key), self.assertRaisesRegex(PilotContractError, "pilot_plan_binding_mismatch"):
                inspect(PilotExecutionPlan.from_document(changed), admission)
        changed = fixture()
        changed["budget"]["wall_ms"] += 1
        with self.assertRaisesRegex(PilotContractError, "pilot_plan_binding_mismatch"):
            inspect(PilotExecutionPlan.from_document(changed), admission)

    def test_plan_or_caller_cannot_shrink_required_model_panel(self):
        doc = fixture()
        doc["learners"].pop()
        plan = PilotExecutionPlan.from_document(doc)
        with self.assertRaisesRegex(PilotContractError, "pilot_required_roster_mismatch"):
            inspect(plan, declaration(plan))

    def test_observed_runtime_and_admission_pins_must_agree(self):
        plan = PilotExecutionPlan.from_document(fixture())
        expected = copy.deepcopy(plan.document()["pins"])
        expected["environment_sha256"] = "8" * 64
        with self.assertRaisesRegex(PilotContractError, "pilot_runtime_binding_mismatch"):
            inspect(plan, declaration(plan), expected_pins=expected)

    def test_gate_denials_keep_precise_static_missing_items(self):
        plan = PilotExecutionPlan.from_document(fixture())
        doc = declaration(plan)
        for g, status in zip(GATES[:3], ("missing", "rejected", "scope_mismatch")):
            doc["gates"][g] = {"status": status, "evidence_sha256": None}
        result = inspect(plan, doc)
        self.assertEqual(result["status"], "blocked")
        self.assertEqual([g["gate_id"] for g in result["missing_gates"]], list(GATES[:3]))
        doc = declaration(plan)
        del doc["gates"][GATES[-1]]
        with self.assertRaisesRegex(PilotContractError, "pilot_gates_incomplete"):
            inspect(plan, doc)

    def test_accepted_gate_requires_retained_evidence_identity(self):
        plan = PilotExecutionPlan.from_document(fixture())
        doc = declaration(plan)
        doc["gates"][GATES[0]]["evidence_sha256"] = None
        with self.assertRaisesRegex(PilotContractError, "pilot_gate_evidence_missing"):
            inspect(plan, doc)

    def test_native_or_confirmation_grants_cannot_be_replayed_as_pilot(self):
        plan = PilotExecutionPlan.from_document(fixture())
        for scope in ("native_engineering_stages_only", "confirmation"):
            doc = declaration(plan)
            doc["operation"] = scope
            with self.subTest(scope=scope), self.assertRaisesRegex(PilotContractError, "pilot_operation_mismatch"):
                inspect(plan, doc)

    def test_malformed_metadata_and_source_values_do_not_escape_errors(self):
        for bad in (b'{"format":1,"format":2}', b'{"x":NaN}', b"not-json"):
            with self.assertRaises(PilotContractError):
                PilotExecutionPlan.from_bytes(bad)
        doc = fixture()
        doc["seeds"]["master"] = "PRIVATE_SOURCE_HEADER_95"
        with self.assertRaises(PilotContractError) as caught:
            PilotExecutionPlan.from_document(doc)
        self.assertNotIn("PRIVATE_SOURCE_HEADER", str(caught.exception))

    def test_no_files_network_processes_or_models_are_accessed(self):
        plan = PilotExecutionPlan.from_document(fixture())
        doc = declaration(plan)
        with mock.patch("builtins.open", side_effect=AssertionError("file IO")), \
             mock.patch("os.open", side_effect=AssertionError("file IO")), \
             mock.patch("subprocess.Popen", side_effect=AssertionError("process")), \
             mock.patch("socket.socket", side_effect=AssertionError("network")):
            self.assertFalse(inspect(plan, doc)["execution_authorized"])

    def test_false_counts_duplicate_ids_and_invalid_numeric_limits_refused(self):
        for key, value in (("wall_ms", True), ("cpu_ms", 0), ("max_attempts", 17),
                           ("disk_bytes", 2**53), ("peak_ram_bytes", 1.5)):
            doc = fixture()
            doc["budget"][key] = value
            with self.subTest(key=key), self.assertRaises(PilotContractError):
                PilotExecutionPlan.from_document(doc)
        for change in (lambda d: d["tasks"][1].update(role="observational"),
                       lambda d: d["tasks"][1].update(task_id=d["tasks"][0]["task_id"]),
                       lambda d: d["seeds"]["substreams"].update(fit=d["seeds"]["substreams"]["context"]),
                       lambda d: d["byte_caps"].reverse()):
            doc = fixture()
            change(doc)
            with self.assertRaises(PilotContractError):
                PilotExecutionPlan.from_document(doc)


if __name__ == "__main__":
    unittest.main()
