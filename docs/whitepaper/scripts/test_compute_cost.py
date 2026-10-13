"""The compute-cost reducer reads only its pinned bundle and ledgers, and reproduces the committed ledger."""

import builtins
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import compute_cost as C
from pinned_inputs import PinRefused, Registry

LOCK = C.REPO / "ops" / "ci" / "paper-inputs" / "inputs.lock.json"


class ComputeCostPinTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "generated"
        self.root.mkdir()
        self.raw = (C.GENERATED / C.BUNDLE).read_bytes()
        self.bundle_registry = Registry(self.root, C.BUNDLE_PINS.files)

    def tearDown(self):
        self.tmp.cleanup()

    def test_the_committed_cost_ledger_is_reproduced_from_the_pinned_inputs(self):
        out = Path(self.tmp.name) / "compute-cost.json"
        self.assertEqual(C.build(out=out), 0)
        self.assertEqual(out.read_bytes(), C.OUT.read_bytes())

    def test_the_bundle_pin_is_the_locked_static_input(self):
        locked = json.loads(LOCK.read_text())["files"][C.BUNDLE]
        self.assertEqual(C.BUNDLE_PINS.files[C.BUNDLE], (locked["bytes"], locked["sha256"]))
        self.assertEqual(set(C.LEDGER_PINS.files), {C.FOREST_LEDGER, C.ARF_LEDGER})
        self.assertEqual((C.BUNDLE_PINS.root, C.LEDGER_PINS.root), (C.GENERATED, C.RESULTS))

    def test_a_substituted_bundle_refuses_before_decoding_in_load_and_build(self):
        changed = self.raw.replace(b'"version": 1', b'"version": 2', 1)
        self.assertEqual(len(changed), len(self.raw))
        (self.root / C.BUNDLE).write_bytes(changed)
        out = Path(self.tmp.name) / "compute-cost.json"
        with patch.object(C.json, "loads", side_effect=AssertionError("decoded")):
            with self.assertRaisesRegex(PinRefused, "digest differs"):
                C._load_bundle(self.bundle_registry)
            with self.assertRaisesRegex(PinRefused, "digest differs"):
                C.build(registry=self.bundle_registry, out=out)
        self.assertFalse(out.exists())

    def test_a_missing_bundle_is_an_error_and_a_symlinked_bundle_refuses(self):
        with self.assertRaisesRegex(FileNotFoundError, "paper-prepare"):
            C._load_bundle(self.bundle_registry)
        (self.root / C.BUNDLE).symlink_to(C.GENERATED / C.BUNDLE)
        with self.assertRaisesRegex(PinRefused, "symlink"):
            C._load_bundle(self.bundle_registry)

    def test_a_results_other_sibling_or_unregistered_ledger_refuses(self):
        sibling = Path(self.tmp.name) / "results-other"
        sibling.mkdir()
        (sibling / C.FOREST_LEDGER).write_bytes(b"{}")
        with self.assertRaisesRegex(PinRefused, "length differs"):
            C._ledger(C.FOREST_LEDGER, Registry(sibling, C.LEDGER_PINS.files))
        for name in ("../results-other/" + C.FOREST_LEDGER, "density-matched-population-validation.json"):
            with self.subTest(name), self.assertRaises(PinRefused):
                C._ledger(name)

    def test_an_absent_arf_ledger_stays_a_note_and_a_present_one_is_authenticated(self):
        empty = Path(self.tmp.name) / "results"
        empty.mkdir()
        with patch.object(C, "LEDGER_PINS", Registry(empty, C.LEDGER_PINS.files)):
            self.assertEqual(C._arf()["notes"], ["population native ledger is absent"])
        shutil.copy(C.RESULTS / C.FOREST_LEDGER, empty / C.ARF_LEDGER)
        with patch.object(C, "LEDGER_PINS", Registry(empty, C.LEDGER_PINS.files)), \
                self.assertRaisesRegex(PinRefused, "length differs"):
            C._arf()

    def test_a_redirected_results_directory_refuses_instead_of_reading_as_absent(self):
        empty = Path(self.tmp.name) / "empty"
        empty.mkdir()
        link = Path(self.tmp.name) / "results-link"
        link.symlink_to(empty, target_is_directory=True)
        with patch.object(C, "LEDGER_PINS", Registry(link, C.LEDGER_PINS.files)), \
                self.assertRaisesRegex(PinRefused, "symlink"):
            C._arf()

    def test_parsing_uses_the_authenticated_bytes_and_never_rereads_a_path(self):
        def refuse(*_args, **_kwargs):
            raise AssertionError("second read of a path")

        with patch.object(Path, "read_text", refuse), patch.object(Path, "read_bytes", refuse), \
                patch.object(builtins, "open", refuse):
            bundle = C._load_bundle()
            forest = C._ledger(C.FOREST_LEDGER)
        self.assertEqual(bundle["format"], "dope-paper-compute-cost-receipts")
        self.assertIn("cost", forest)

    def test_no_scratch_receipt_is_read(self):
        source = Path(C.__file__).read_text()
        self.assertNotIn("/mnt/fast-scratch", source)
        self.assertFalse(hasattr(C, "SCRATCH"))


if __name__ == "__main__":
    unittest.main()
