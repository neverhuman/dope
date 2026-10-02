"""Integrity failures must precede optional library initialization/unpickling."""

import builtins
import json
import tempfile
import traceback
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from research.benchmark import forestdiffusion_adapter as adapter


class ForestBoundaryTests(unittest.TestCase):
    def setUp(self):
        target = Path(__file__).parents[3] / "target"
        target.mkdir(exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(dir=target, prefix="forest-boundary-")
        self.root = Path(self.directory.name)
        self.scratch = patch.object(adapter, "SCRATCH", self.root)
        self.scratch.start()

    def tearDown(self):
        self.scratch.stop()
        self.directory.cleanup()

    def mismatched_runtime(self, field):
        dependency = self.root / "initializer.py"
        dependency.write_text("raise AssertionError('initializer ran')\n")
        lock = self.root / "runtime.json"
        value = {
            "format": "dope-forestdiffusion-author-runtime", "version": 1,
            "adapter_sha256": adapter.sha(adapter.__file__), "environment": {},
            "package_root": str(self.root), "dependency_root": str(self.root),
            "author_package_files": {}, "dependency_files": {},
        }
        value[field] = {dependency.name: "0" * 64}
        lock.write_text(json.dumps(value))
        return {"runtime_lock": str(lock), "runtime_lock_sha256": adapter.sha(lock)}

    def test_all_entries_reject_changed_dependencies_before_initialization(self):
        original_import = builtins.__import__
        for operation in ("fit", "sample", "native"):
            for field in ("dependency_files", "author_package_files"):
                with self.subTest(operation=operation, field=field):
                    request = self.mismatched_runtime(field)
                    initialized = []

                    def guarded_import(name, *args, **kwargs):
                        if name.split(".")[0] in ("numpy", "pandas", "sklearn", "xgboost", "ForestDiffusion"):
                            initialized.append(name)
                            raise AssertionError("optional library initialized")
                        return original_import(name, *args, **kwargs)

                    with patch("builtins.__import__", side_effect=guarded_import):
                        with self.assertRaisesRegex(ValueError, "source or dependency changed"):
                            getattr(adapter, operation)(request)
                    self.assertEqual(initialized, [])

    def test_changed_and_extra_artifact_files_precede_pickle_execution(self):
        root = self.root / "artifact"
        root.mkdir()
        weights = root / "sampler.pkl"
        weights.write_bytes(b"must never execute")
        request = {"runtime_lock": "unused", "runtime_lock_sha256": "0" * 64,
                   "artifact": str(root), "expected_inventory": adapter.inventory(root)}
        with patch.object(adapter, "runtime"), patch.object(adapter.pickle, "load") as load:
            weights.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "inventory changed"):
                adapter.load_model(request)
            weights.write_bytes(b"must never execute")
            (root / "extra.pkl").touch()
            with self.assertRaisesRegex(ValueError, "inventory changed"):
                adapter.load_model(request)
            load.assert_not_called()

    def test_retained_training_containers_fail_closed(self):
        model = SimpleNamespace(X1=SimpleNamespace(size=0), mask_y={}, X_covs=None, label_y=None)
        adapter.check_rows(model)
        for field, value in (("X1", SimpleNamespace(size=1)), ("mask_y", {0: True}),
                             ("X_covs", object()), ("label_y", SimpleNamespace(size=1))):
            with self.subTest(field=field):
                changed = SimpleNamespace(**{**vars(model), field: value})
                with self.assertRaisesRegex(ValueError, "retains training rows"):
                    adapter.check_rows(changed)

    def test_symlinks_and_outside_paths_are_rejected(self):
        original = self.root / "weights"
        original.touch()
        link = self.root / "alias"
        link.symlink_to(original)
        with self.assertRaises(ValueError):
            adapter.safe(link)
        with self.assertRaises(ValueError):
            adapter.safe(self.root.parent / "outside")

    def test_official_test_and_workers_with_tests_are_rejected_before_read(self):
        import pandas

        path = self.root / "train.csv"
        path.write_text("0,0\n1,1\n")
        with patch.object(pandas, "read_csv", side_effect=AssertionError("test rows opened")):
            with self.assertRaisesRegex(ValueError, "verified training-derived"):
                adapter.table(path, "0" * 64)
            (self.root / "test.csv").touch()
            with self.assertRaisesRegex(ValueError, "verified training-derived"):
                adapter.table(path, adapter.sha(path))
            with self.assertRaisesRegex(ValueError, "verified training-derived"):
                adapter.table(self.root / "test.csv", adapter.sha(self.root / "test.csv"))

    def test_invalid_numeric_input_does_not_echo_source(self):
        path = self.root / "train.csv"
        marker = "private_forest_numeric_marker"
        path.write_text(f"{marker},0\n0,1\n")
        try:
            adapter.table(path, adapter.sha(path))
        except ValueError as error:
            self.assertEqual(str(error), "invalid common-numeric input")
            self.assertNotIn(marker, "".join(traceback.format_exception(error)))
        else:
            self.fail("malformed numeric input accepted")


if __name__ == "__main__":
    unittest.main()
