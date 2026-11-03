"""Data-free conformance tests for the offline NC1 apply oracle."""
import json
import math
import os
from pathlib import Path
import unittest

from research.levels.nc1_reference import Nc1Error, _normalize, apply_private_card


class Nc1ReferenceTests(unittest.TestCase):
    def test_unclipped_full_span_and_constant(self):
        self.assertEqual(_normalize(12, {"lo": 0, "hi": 4}), 3)
        self.assertEqual(_normalize(9, {"lo": 7, "hi": 7}), 0)
        self.assertEqual(_normalize(0, {"lo": -float.fromhex("0x1.fffffffffffffp+1023"), "hi": float.fromhex("0x1.fffffffffffffp+1023")}), 0.5)

    def test_nonfinite_safe_refusal(self):
        for value in [math.inf, math.nan, "CANARY_VALUE"]:
            with self.assertRaisesRegex(Nc1Error, "^invalid_numeric$"):
                _normalize(value, {"lo": 0, "hi": 1})

    def test_malformed_card_refuses_without_echo(self):
        with self.assertRaisesRegex(Nc1Error, "^invalid_private_card$"):
            apply_private_card({"CANARY": "private"}, {"x": [0]})

    def test_rust_export_conformance(self):
        location = os.environ.get("DOPE_NC1_CONFORMANCE_OUT")
        if not location:
            self.skipTest("explicit Rust-export conformance fixture not supplied")
        root = Path(location)
        card = json.loads((root / "card.json").read_text())
        expected = json.loads((root / "matrix.json").read_text())
        columns = {"CANARY_HEADER": [10.0, 20.0, None, 100.0],
                   "CANARY_CATEGORY": ["__other__", "CANARY_LEVEL", None, "unseen"],
                   "target": [100.0, 200.0, 300.0, 500.0]}
        actual = apply_private_card(card, columns)
        self.assertEqual(actual["x"], expected["x"])
        self.assertEqual(actual["y"], expected["y"])
        del columns["target"]
        self.assertIsNone(apply_private_card(card, columns, with_target=False)["y"])


if __name__ == "__main__":
    unittest.main()
