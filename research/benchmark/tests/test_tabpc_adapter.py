"""Sealed-input and pickle-integrity checks without optional TabPC libraries."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from research.benchmark import tabpc_adapter as adapter
from research.benchmark.score import artifact_inventory, sha256


class TabPCBoundaryTests(unittest.TestCase):
    def setUp(self):
        target = Path(__file__).parents[3] / "target"
        target.mkdir(exist_ok=True)
        self.scratch = tempfile.TemporaryDirectory(dir=target, prefix="tabpc-boundary-")
        self.root = Path(self.scratch.name)
        self.patcher = patch.object(adapter, "SCRATCH", self.root)
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()
        self.scratch.cleanup()

    def test_sealed_test_is_rejected_before_read(self):
        worker = self.root / "evaluator"
        worker.mkdir()
        table = worker / "test.csv"
        table.write_text("0,0\n1,1\n")
        with patch("pandas.read_csv", side_effect=AssertionError("opened sealed rows")):
            with self.assertRaisesRegex(ValueError, "verified training partition"):
                adapter.load_table(table, sha256(table))

    def test_worker_containing_tests_and_changed_hash_fail_closed(self):
        table = self.root / "train.csv"
        table.write_text("0,0\n1,1\n")
        expected = sha256(table)
        with self.assertRaises(ValueError):
            adapter.load_table(table, "0" * 64)
        (self.root / "test.csv").touch()
        with self.assertRaises(ValueError):
            adapter.load_table(table, expected)

    def test_symlink_and_outside_scratch_fail_closed(self):
        table = self.root / "train.csv"
        table.touch()
        link = self.root / "validation.csv"
        link.symlink_to(table)
        with self.assertRaises(ValueError):
            adapter.scratch_path(link)
        with self.assertRaises(ValueError):
            adapter.scratch_path(self.root.parent / "other.csv")

    def test_extra_and_changed_artifacts_fail_before_author_pickle_import(self):
        artifact = self.root / "artifact"
        artifact.mkdir()
        weights = artifact / "weights.pkl"
        weights.write_bytes(b"untrusted pickle must never be loaded")
        inventory, _ = artifact_inventory(artifact, ["weights.pkl"])
        with patch.object(adapter, "check_runtime", return_value={}):
            weights.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "artifact hash changed"):
                adapter.load_artifact(artifact, inventory, self.root / "runtime.json", "0" * 64)
            weights.write_bytes(b"untrusted pickle must never be loaded")
            (artifact / "extra.pkl").touch()
            with self.assertRaisesRegex(ValueError, "file list does not reconcile"):
                adapter.load_artifact(artifact, inventory, self.root / "runtime.json", "0" * 64)

    def test_changed_runtime_source_is_rejected_before_import(self):
        source = self.root / "author.py"
        source.write_text("raise AssertionError('must never import')\n")
        lock = self.root / "runtime.json"
        lock.write_text(json.dumps({
            "format": "dope-tabpc-author-runtime", "version": 1,
            "adapter_sha256": sha256(Path(adapter.__file__)),
            "patch_sha256": sha256(Path(adapter.__file__).with_name("tabpc-cpu-sampling.patch")),
            "environment": {}, "audit_root": str(self.root),
            "source_files": {"author.py": "0" * 64}}))
        with self.assertRaisesRegex(ValueError, "source or runtime evidence changed"):
            adapter.check_runtime(lock, sha256(lock))


if __name__ == "__main__":
    unittest.main()
