"""Data-free independent DML4 graph and Rust-export conformance."""
import copy
import json
import struct
import zlib
import hashlib
import math
import os
from pathlib import Path
import unittest
from unittest import mock
from research.levels.v4_wire_reference import Dml4Error, basis, decode, sample


ENVIRONMENT='DOPE_DML4_CONFORMANCE_OUT'
FIXTURE_FOLDER='dml4-rust-v1'
FIXTURE_KIND='dml4'
EXTERNAL_CASE={'id': 'external', 'packet': 'packet.dml4', 'json': 'packet.json', 'samples': 'samples.json', 'input': None}

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




class V4WireReferenceTests(unittest.TestCase):
    def test_independent_gate_basis(self):
        gates = dict(zip(("linear", "asinh", "periodic", "hinge", "pair", "row_mean"), (.1, .2, .3, .4, .5, .6)))
        actual = basis([.25, .75], gates)
        expected = [.025, .075, .2 * math.asinh(.25), .2 * math.asinh(.75), .3, -.3, 0, .1, .5 * .25 * .75, .3]
        for a, b in zip(actual, expected):
            self.assertAlmostEqual(a, b, places=14)

    def test_envelope_refusal(self):
        for data in [b"", b"DML3", b"DML4\x01" + bytes(8), bytes(65537)]:
            with self.assertRaises(Dml4Error):
                decode(data)

    def test_nonfinite_basis_refuses(self):
        with self.assertRaises(Dml4Error):
            basis([math.inf], dict.fromkeys(("linear", "asinh", "periodic", "hinge", "pair", "row_mean"), 1))

    def test_rust_export_fixed_seed_parity(self):
        root = _fixture_root()
        cases = _fixture_cases(root)
        self.assertTrue(cases)
        for case in cases:
            with self.subTest(case=case["id"]):
                packet = decode((root / case["packet"]).read_bytes())
                self.assertEqual(packet, json.loads((root / case["json"]).read_text()))
                expected = json.loads((root / case["samples"]).read_text())
                inputs = json.loads((root / case["input"]).read_text()) if case["input"] else {"rows": 4, "seed": 42}
                actual = sample(packet, inputs["rows"], inputs["seed"])
                self.assertEqual(len(actual["x"]), len(expected["x"]))
                self.assertEqual(len(actual["y"]), len(expected["y"]))
                for row, expected_row in zip(actual["x"], expected["x"]):
                    self.assertEqual(row, expected_row)
                for a, b in zip(actual["y"], expected["y"]):
                    self.assertAlmostEqual(a, b, places=13)
                corrupted = bytearray((root / case["packet"]).read_bytes())
                corrupted[-1] ^= 1
                with self.assertRaises(Dml4Error):
                    decode(bytes(corrupted))

    def test_nested_identity_shape_and_private_metadata_refuse_before_sampling(self):
        root = _fixture_root()
        case = _fixture_cases(root)[0]
        valid = decode((root / case["packet"]).read_bytes())
        def wire(packet):
            body = json.dumps(packet, separators=(",", ":")).encode()
            return b"DML4\x01" + struct.pack("<II", len(body), zlib.crc32(body)) + body
        mutations = []
        for key, value in [("schema_sha256", "0"*64), ("groups", [])]:
            bad=copy.deepcopy(valid);bad[key]=value;mutations.append(bad)
        bad=copy.deepcopy(valid);bad["identity"]["weights_sha256"]="PRIVATE_CANARY";mutations.append(bad)
        bad=copy.deepcopy(valid);bad["schema"]["groups"][0]["id"]=17;mutations.append(bad)
        bad=copy.deepcopy(valid);bad["target"]["coefficients"]=[];mutations.append(bad)
        bad=copy.deepcopy(valid);bad["groups"][0]["PRIVATE_CANARY"]="PRIVATE_VALUE";mutations.append(bad)
        bad=copy.deepcopy(valid);bad["gates"]["linear"]=True;mutations.append(bad)
        for packet in mutations:
            with self.subTest(packet_keys=tuple(packet)):
                with self.assertRaises(Dml4Error) as caught:decode(wire(packet))
                self.assertNotIn("PRIVATE",str(caught.exception))
                with self.assertRaises(Dml4Error):sample(packet,1,0)
        body=json.dumps(valid,separators=(",", ":")).encode()
        body=body[:-1]+b',"contract":"dope.dml4-reference/v1"}'
        with self.assertRaises(Dml4Error):
            decode(b"DML4\x01"+struct.pack("<II",len(body),zlib.crc32(body))+body)

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
