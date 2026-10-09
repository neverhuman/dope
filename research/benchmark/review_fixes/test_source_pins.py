"""Source-pin fixtures only: opaque JSON, no CSVs, partitions or live registry."""

import hashlib
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from research.benchmark.review_fixes import privacy_fidelity as privacy
from research.benchmark.review_fixes import sealed_once as sealed
from research.benchmark.review_fixes import source_pins

DENSITY = "density-matched-population-validation.json"
PANEL = "review-fixes-receipts-v1/panel.json"
LEDGERS = ("s3-lineage-record.json", DENSITY,
           "sdv-matched-population-validation.json",
           "arf-matched-population-validation.json",
           "s3-matched-forest-confirmation-validation.json")


class PublishedSourcePinTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="source-pin-toy-")
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.cell = {
            "dataset": "toy-only", "method": "DOPE",
            "configuration": "features12_steps2048", "fit_seed": 11,
            "size_multiplier": 1, "sample_seed": 101,
            "status": "fit_unavailable", "metric_receipt": {"sample_sha256": "a" * 64},
        }
        documents = {name: {"cells": []} for name in LEDGERS}
        documents[DENSITY] = {"cells": [self.cell]}
        documents["s3-lineage-record.json"] = {"rows": [{
            "dataset": "toy-only", "display_name": "toy-only",
        }]}
        self.raw = {name: json.dumps(document, sort_keys=True).encode()
                    for name, document in documents.items()}
        self.frozen = {name: hashlib.sha256(raw).hexdigest()
                       for name, raw in self.raw.items()}
        self.raw[PANEL] = json.dumps({"sources": self.frozen}, sort_keys=True).encode()
        self.frozen[PANEL] = hashlib.sha256(self.raw[PANEL]).hexdigest()
        for name, raw in self.raw.items():
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        # Expectations are captured once from the untouched toy fixture. Neither
        # tested producer accepts an expectation from a mutable JSON input.
        self.addCleanup(patch.stopall)
        patch.object(source_pins, "PUBLISHED_SOURCE_SHA256", self.frozen).start()
        patch.object(sealed, "PUBLISHED_SOURCE_SHA256", self.frozen, create=True).start()
        patch.object(sealed, "RESULTS", self.root).start()
        patch.object(privacy, "PUBLISHED", [(DENSITY, "DOPE", "features12_steps2048")]).start()
        patch.object(privacy, "_resolve_sample_csv", return_value="/toy/not-opened.csv").start()

    def rewrite_ledger_and_panel(self):
        changed = {"cells": [{**self.cell, "metric_receipt": {"sample_sha256": "b" * 64}}]}
        raw = json.dumps(changed, sort_keys=True).encode()
        (self.root / DENSITY).write_bytes(raw)
        pins = {name: digest for name, digest in self.frozen.items() if name != PANEL}
        pins[DENSITY] = hashlib.sha256(raw).hexdigest()
        (self.root / PANEL).write_text(json.dumps({"sources": pins}, sort_keys=True))

    def test_positive_unchanged_frozen_toy_bytes(self):
        self.assertEqual(source_pins.authenticated_published_bytes(self.root, self.raw), self.raw)
        self.assertEqual(sealed._published_samples()[0]["expected_sha256"], "a" * 64)
        self.assertEqual(privacy._index(self.root)[0]["expected_sha256"], "a" * 64)

    def test_mutually_rewritten_panel_and_ledger_refused(self):
        self.rewrite_ledger_and_panel()
        with self.assertRaises(ValueError):
            sealed._assert_ledger_pins()

    def test_privacy_mutually_rewritten_ledger_refused_before_resolving_csv(self):
        self.rewrite_ledger_and_panel()
        with patch.object(privacy, "_resolve_sample_csv", side_effect=AssertionError("resolve before auth")) as resolve:
            with self.assertRaises(ValueError):
                privacy._index(self.root)
        resolve.assert_not_called()

    def test_panel_substitution_refused_before_decode(self):
        (self.root / PANEL).write_bytes(b"opaque unauthenticated replacement")
        with patch.object(sealed.json, "loads", side_effect=AssertionError("decode before auth")) as decode:
            with self.assertRaises(ValueError):
                sealed._assert_ledger_pins()
        decode.assert_not_called()

    def test_ledger_substitution_refused_before_decode(self):
        (self.root / DENSITY).write_bytes(b"opaque unauthenticated replacement")
        with patch.object(privacy.json, "loads", side_effect=AssertionError("decode before auth")) as decode:
            with self.assertRaises(ValueError):
                privacy._index(self.root)
        decode.assert_not_called()

    def test_sealed_parser_uses_captured_authenticated_bytes(self):
        verify = sealed._assert_ledger_pins

        def capture_then_replace():
            captured = verify()
            self.rewrite_ledger_and_panel()
            return captured

        with patch.object(sealed, "_assert_ledger_pins", side_effect=capture_then_replace):
            samples = sealed._published_samples()
        self.assertEqual(samples[0]["expected_sha256"], "a" * 64)

    def test_privacy_parser_does_not_reopen_checked_paths(self):
        with patch.object(Path, "read_text", side_effect=AssertionError("mutable path reopened")):
            jobs = privacy._index(self.root)
        self.assertEqual(jobs[0]["expected_sha256"], "a" * 64)

    def test_first_lineage_decode_is_authenticated_before_sealed_admission(self):
        # main is the captured source under complete mocks. No actual host,
        # registry, sample resolver, gate or official-test read is permitted.
        (self.root / "s3-lineage-record.json").write_bytes(b"opaque replacement")
        registry = self.root / "toy-registry-never-opened"
        argv = ["sealed-toy", "--predeclare", str(self.root / "never-read"),
                "--expect-sha256", "0" * 64, "--expect-git-blob", "0" * 40,
                "--controls", str(self.root / "never-read-controls"),
                "--out", str(registry), "--host", sealed.APPROVED_HOST]
        # predeclare_v2 makes the v1 run refuse first; this test checks the guards behind that refusal.
        with (
            patch.object(sealed, "_refuse_superseded", return_value=None),
            patch.object(sys, "argv", argv),
            patch.object(sealed.os, "uname", return_value=types.SimpleNamespace(nodename=sealed.APPROVED_HOST)),
            patch.dict(os.environ, {"DOPE_RF_UNSEAL": "1"}),
            patch.object(sealed, "REGISTRY", registry),
            patch.object(sealed, "registry_mode", return_value="fresh") as mode,
            patch.object(sealed, "_preflight", side_effect=AssertionError("preflight before auth")) as preflight,
            patch.object(sealed, "_gate", side_effect=AssertionError("gate before auth")) as gate,
            patch.object(sealed, "_read_numeric", side_effect=AssertionError("raw read forbidden")) as rows,
            patch.object(sealed.json, "loads", side_effect=AssertionError("decode before auth")) as decode,
        ):
            with self.assertRaises(ValueError):
                sealed.main()
        mode.assert_called_once()
        preflight.assert_not_called()
        gate.assert_not_called()
        rows.assert_not_called()
        decode.assert_not_called()
        self.assertFalse(registry.exists())

    def test_unknown_source_refused_before_path_read(self):
        with patch.object(Path, "read_bytes", side_effect=AssertionError("unregistered path read")) as read:
            with self.assertRaises(ValueError):
                source_pins.authenticated_published_bytes(self.root, ["unknown.json"])
        read.assert_not_called()

    def test_missing_pin_refused_before_path_read(self):
        with (
            patch.object(source_pins, "PUBLISHED_SOURCE_SHA256", {}),
            patch.object(Path, "read_bytes", side_effect=AssertionError("unpinned path read")) as read,
        ):
            with self.assertRaises(ValueError):
                source_pins.authenticated_published_bytes(self.root, [DENSITY])
        read.assert_not_called()

    def test_missing_source_refused(self):
        (self.root / DENSITY).unlink()
        with self.assertRaises(FileNotFoundError):
            source_pins.authenticated_published_bytes(self.root, [DENSITY])

    def test_symlinked_source_refused(self):
        target = self.root / "opaque-copy.json"
        target.write_bytes(self.raw[DENSITY])
        path = self.root / DENSITY
        path.unlink()
        path.symlink_to(target)
        with self.assertRaises(ValueError):
            source_pins.authenticated_published_bytes(self.root, [DENSITY])


if __name__ == "__main__":
    unittest.main()
