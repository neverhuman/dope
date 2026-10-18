"""E04 precommit integrity and sealed-exposure denial fixtures."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import unittest

from research.transfer.evidence import canonical_json
from research.transfer.precommit import (
    BYTE_GRID,
    PrecommitError,
    load_precommit,
    sealed_access_allowed,
    validate_blocked_precommit,
)

ROOT = Path(__file__).resolve().parents[3]
PRECOMMIT = ROOT / "target/confirmatory-e04/PRECOMMIT.json"


def bind(document):
    result = copy.deepcopy(document)
    result.pop("binding_sha256", None)
    result["binding_sha256"] = hashlib.sha256(canonical_json(result)).hexdigest()
    return result


class PrecommitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.document, cls.blockers = load_precommit(PRECOMMIT, ROOT)

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
