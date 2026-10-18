"""E04 precommit integrity and sealed-exposure denial fixtures."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import unittest
import tempfile

from research.transfer.evidence import canonical_json
from research.transfer.precommit import (
    BYTE_GRID,
    FORMAT,
    PROTOCOL,
    expected_pins,
    PrecommitError,
    load_precommit,
    sealed_access_allowed,
    validate_blocked_precommit,
)

ROOT = Path(__file__).resolve().parents[3]


def bind(document):
    result = copy.deepcopy(document)
    result.pop("binding_sha256", None)
    result["binding_sha256"] = hashlib.sha256(canonical_json(result)).hexdigest()
    return result


class PrecommitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Build a blocked synthetic packet from tracked inputs. A lane-local
        # ignored campaign packet must never be a fresh-checkout prerequisite.
        protocol = json.loads((ROOT / PROTOCOL).read_text())
        pins = expected_pins(ROOT)
        document = bind({
            "format": FORMAT, "version": 1,
            "campaign_id": "synthetic-precommit-fixture",
            "admission_state": "blocked", "sealed_access_allowed": False,
            "protocol_path": PROTOCOL, "protocol_sha256": pins[PROTOCOL],
            "evidence_pins": pins,
            "design": {"fixture_only": True, "rule": protocol["design_rule"]},
            "source_roles": [], "model_roles": [], "prepared_maps": [],
            "candidate_budget_slots": [
                {"budget_bytes": budget,
                 "candidate_id": f"development-selected-at-{budget}-bytes",
                 "artifact_sha256": None, "recipe_sha256": None,
                 "eligible_for_minimum_tdl_reporting": False}
                for budget in BYTE_GRID
            ],
            "full_comparison_roster": [], "roster_complete": False,
            "thresholds": protocol["analysis_policy"],
            "sample_schedule": protocol["seed_schedule"],
            "privacy_outcomes": protocol["analysis_policy"]["privacy_outcomes"],
            "budget_and_power": protocol["budget_admission"],
            "retry_exclusion": protocol["retry_exclusion_policy"],
            "review_signature": None, "custodian_attestation": None,
            "admission_blockers": ["synthetic_fixture_only", "roster_missing",
                                   "independent_review_and_custody_missing"],
        })
        directory = tempfile.TemporaryDirectory(prefix="precommit-fixture-", dir=ROOT / "target")
        cls.addClassCleanup(directory.cleanup)
        path = Path(directory.name) / "PRECOMMIT.json"
        path.write_bytes(canonical_json(document))
        cls.document, cls.blockers = load_precommit(path, ROOT)

    def test_all_budget_slots_are_enumerated_and_confirmation_is_denied(self):
        self.assertEqual([slot["budget_bytes"] for slot in self.document["candidate_budget_slots"]], list(BYTE_GRID))
        self.assertEqual(len(self.document["candidate_budget_slots"]), 10)
        self.assertTrue(self.blockers)
        self.assertFalse(sealed_access_allowed(self.document, ROOT))
        self.assertFalse(self.document["sealed_access_allowed"])
        self.assertIn("research/transfer/precommit.py", self.document["evidence_pins"])

    def test_incomplete_source_model_and_candidate_roster_cannot_be_promoted(self):
        for mutation in (
            lambda doc: doc.update(admission_state="admitted", sealed_access_allowed=True),
            lambda doc: doc["candidate_budget_slots"][0].update(artifact_sha256="a" * 64),
            lambda doc: doc.update(full_comparison_roster=[{"source": "unbound"}], roster_complete=True),
            lambda doc: doc.update(review_signature={"signature": "unverified"}),
            lambda doc: doc.update(custodian_attestation={"identity": "unverified"}),
        ):
            changed = copy.deepcopy(self.document)
            mutation(changed)
            with self.assertRaises(PrecommitError):
                sealed_access_allowed(bind(changed), ROOT)

    def test_evidence_and_protocol_hash_drift_refuse(self):
        changed = copy.deepcopy(self.document)
        changed["evidence_pins"]["research/transfer/evidence.py"] = "0" * 64
        with self.assertRaisesRegex(PrecommitError, "precommit_evidence_pin_mismatch"):
            validate_blocked_precommit(bind(changed), ROOT)
        changed = copy.deepcopy(self.document)
        changed["protocol_sha256"] = "0" * 64
        with self.assertRaisesRegex(PrecommitError, "precommit_protocol_pin_mismatch"):
            validate_blocked_precommit(bind(changed), ROOT)

    def test_budget_roster_omission_and_binding_drift_refuse(self):
        changed = copy.deepcopy(self.document)
        changed["candidate_budget_slots"].pop()
        with self.assertRaisesRegex(PrecommitError, "precommit_budget_roster_incomplete"):
            validate_blocked_precommit(bind(changed), ROOT)
        changed = copy.deepcopy(self.document)
        changed["candidate_budget_slots"][0]["budget_bytes"] = BYTE_GRID[1]
        with self.assertRaisesRegex(PrecommitError, "precommit_candidate_slot_invalid"):
            validate_blocked_precommit(bind(changed), ROOT)
        changed = copy.deepcopy(self.document)
        changed["binding_sha256"] = "f" * 64
        with self.assertRaisesRegex(PrecommitError, "precommit_binding_mismatch"):
            validate_blocked_precommit(changed, ROOT)

    def test_campaign_schema_binds_the_optional_precommit_document(self):
        schema = json.loads((ROOT / "schemas/campaign-v1.schema.json").read_text())
        self.assertEqual(schema["properties"]["precommit"]["$ref"], "#/$defs/confirmatory_precommit")
        required = schema["$defs"]["confirmatory_precommit"]["required"]
        self.assertIn("candidate_budget_slots", required)
        self.assertIn("review_signature", required)
        self.assertIn("binding_sha256", required)


if __name__ == "__main__":
    unittest.main()
