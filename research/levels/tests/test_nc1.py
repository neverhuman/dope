"""Data-free conformance tests for the offline NC1 apply oracle."""
import json
import hashlib
import math
import os
from pathlib import Path
import unittest
from unittest import mock

from research.levels.nc1_reference import Nc1Error, _normalize, apply_private_card


ENVIRONMENT='DOPE_NC1_CONFORMANCE_OUT'
FIXTURE_FOLDER='nc1-rust-v1'
FIXTURE_KIND='nc1'
EXTERNAL_CASE={'id': 'external', 'card': 'card.json', 'matrix': 'matrix.json', 'input': None}

def _fixture_root():
    override = os.environ.get(ENVIRONMENT)
    root = Path(override) if override else Path(__file__).resolve().parent / "fixtures" / FIXTURE_FOLDER
    if not root.is_dir():
        raise FileNotFoundError("required Rust conformance fixture directory missing")
    return root


def _fixture_cases(root):
    manifest_path = root / "MANIFEST.json"
    if not manifest_path.is_file():
        if os.environ.get(ENVIRONMENT):
            # Explicit external Rust-export probe; producer provenance belongs
            # to the caller's run receipt. Missing required files still fail.
            return [EXTERNAL_CASE]
        raise FileNotFoundError("required Rust conformance manifest missing")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "dope.reference-fixtures/v1" or manifest.get("kind") != FIXTURE_KIND:
        raise AssertionError("invalid conformance manifest")
    source_root = Path(os.environ.get("DOPE_CONFORMANCE_SOURCE_ROOT", Path(__file__).resolve().parents[3]))
    for relative, expected in manifest["producer"]["rust_source_files"].items():
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise AssertionError("invalid conformance source path")
        if hashlib.sha256((source_root / path).read_bytes()).hexdigest() != expected:
            raise AssertionError("Rust conformance producer source drift; regenerate explicitly")
    for relative, expected in manifest["files"].items():
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise AssertionError("invalid conformance artifact path")
        if hashlib.sha256((root / path).read_bytes()).hexdigest() != expected:
            raise AssertionError("Rust conformance artifact drift")
    return manifest["cases"]




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
        root = _fixture_root()
        cases = _fixture_cases(root)
        self.assertTrue(cases)
        for case in cases:
            with self.subTest(case=case["id"]):
                card = json.loads((root / case["card"]).read_text())
                expected = json.loads((root / case["matrix"]).read_text())
                columns = json.loads((root / case["input"]).read_text()) if case["input"] else {
                    "CANARY_HEADER": [10.0, 20.0, None, 100.0],
                    "CANARY_CATEGORY": ["__other__", "CANARY_LEVEL", None, "unseen"],
                    "target": [100.0, 200.0, 300.0, 500.0]}
                actual = apply_private_card(card, columns)
                self.assertEqual(actual["x"], expected["x"])
                self.assertEqual(actual["y"], expected["y"])
                del columns[card["names"][card["declared"]["target"]["column"]]]
                self.assertIsNone(apply_private_card(card, columns, with_target=False)["y"])

    def test_malformed_identity_group_and_private_fields_refuse(self):
        root = _fixture_root()
        case = _fixture_cases(root)[0]
        card = json.loads((root / case["card"]).read_text())
        columns = json.loads((root / case["input"]).read_text()) if case["input"] else {
            "CANARY_HEADER": [10.0, 20.0, None, 100.0],
            "CANARY_CATEGORY": ["__other__", "CANARY_LEVEL", None, "unseen"],
            "target": [100.0, 200.0, 300.0, 500.0]}
        mutations = []
        malformed = json.loads(json.dumps(card))
        malformed["source_sha256"] = "CANARY_not_a_digest"
        mutations.append(malformed)
        malformed = json.loads(json.dumps(card))
        malformed["public"]["groups"][0]["width"] += 1
        mutations.append(malformed)
        malformed = json.loads(json.dumps(card))
        malformed["private_unknown_field"] = "CANARY"
        mutations.append(malformed)
        for malformed in mutations:
            with self.assertRaisesRegex(Nc1Error, "^invalid_private_card$"):
                apply_private_card(malformed, columns)

    def test_required_fixture_missing_is_failure_not_skip(self):
        missing = Path(__file__).resolve().parent / "fixtures" / "__missing_conformance_directory__"
        with mock.patch.dict(os.environ, {ENVIRONMENT: str(missing)}):
            with self.assertRaises(FileNotFoundError):
                _fixture_root()
        with mock.patch.dict(os.environ, {ENVIRONMENT: ""}):
            with self.assertRaises(FileNotFoundError):
                _fixture_cases(missing)



if __name__ == "__main__":
    unittest.main()
