"""Validate real Rust CLI snapshot fixtures and refuse incompatible wire shapes."""
import copy
import json
from pathlib import Path
import unittest

from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[3]


class LandmarkEncodingSchemaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        schema = json.loads((ROOT / "schemas/landmark-encoding-receipt-v1.schema.json").read_text())
        Draft202012Validator.check_schema(schema)
        cls.validator = Draft202012Validator(schema)
        cls.receipts = [json.loads(line) for line in (ROOT / "engine/dope-cli/tests/fixtures/landmark-encoding-v1.jsonl").read_text().splitlines()]

    def test_actual_rust_progressive_snapshots(self):
        self.assertEqual(len(self.receipts), 4)
        for receipt in self.receipts:
            self.validator.validate(receipt)
            count = sum(receipt["mask"])
            self.assertEqual(receipt["status"], "none" if count == 0 else "full" if count == 23 else {"partial": count})

    def test_required_identity_and_evidence_cannot_be_omitted(self):
        for key in self.receipts[0]:
            with self.subTest(key=key):
                changed = copy.deepcopy(self.receipts[0])
                del changed[key]
                self.assertFalse(self.validator.is_valid(changed))

    def test_partial_status_must_equal_observed_mask_count(self):
        for count in range(1, 23):
            with self.subTest(count=count):
                changed = copy.deepcopy(self.receipts[0])
                changed["mask"] = [True] * count + [False] * (23 - count)
                changed["status"] = {"partial": count}
                self.validator.validate(changed)
                changed["status"] = {"partial": count % 22 + 1}
                self.assertFalse(self.validator.is_valid(changed))

    def test_wrong_layout_mask_status_or_hash_is_rejected(self):
        cases = [("schema", "jope.landmark-encoding/v1"), ("mask", [False] * 22),
                 ("mask", [0] * 23), ("status", "partial"), ("status", {"partial": 23}),
                 ("status", "full"), ("source_digest", "zero"), ("weights_sha256", "invalid"),
                 ("subsample_n", 1), ("version", -1), ("features629", [0.0] * 629)]
        for key, value in cases:
            with self.subTest(key=key, value=str(value)[:40]):
                changed = copy.deepcopy(self.receipts[0])
                changed[key] = value
                self.assertFalse(self.validator.is_valid(changed))
        changed = copy.deepcopy(self.receipts[0])
        changed["embedding_identity"]["dimensions"] = 512
        self.assertFalse(self.validator.is_valid(changed))


if __name__ == "__main__":
    unittest.main()
