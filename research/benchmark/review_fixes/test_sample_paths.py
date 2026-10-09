"""Path rules for published sample files. No ledger and no CSV is opened."""

import ast
import sys
import tempfile
import unittest
from pathlib import Path

from research.benchmark.review_fixes.sample_paths import resolve_sample_csv
from research.benchmark.review_fixes.tabsyn_sample import _load_allowlisted


class PathTests(unittest.TestCase):
    def test_dope_uses_the_metric_sibling_not_sample_evidence(self):
        cell = {
            "method": "DOPE",
            "metric_receipt": {"metric_path": "/attempts/n1-seed101.metric.json"},
        }
        self.assertEqual(resolve_sample_csv(cell), "/attempts/n1-seed101.csv")

    def test_with_suffix_would_be_the_wrong_dope_name(self):
        found = resolve_sample_csv({
            "method": "DOPE",
            "metric_receipt": {"metric_path": "/attempts/n1-seed101.metric.json"},
        })
        self.assertFalse(found.endswith(".metric.csv"))

    def test_arf_prefers_the_sample_path(self):
        cell = {"method": "ARF", "sample_evidence": {
            "sample_path": "/arf/n4-seed101.csv",
            "metric_path": "/arf/n4-seed101.metric.json",
        }}
        self.assertEqual(resolve_sample_csv(cell), "/arf/n4-seed101.csv")

    def test_neural_uses_a_csv_path(self):
        cell = {"method": "CTGAN", "sample_evidence": {"path": "/sdv/4n-seed101.csv", "sha256": "abc"}}
        self.assertEqual(resolve_sample_csv(cell), "/sdv/4n-seed101.csv")

    def test_forest_uses_the_sibling_sample_not_the_repeat(self):
        cell = {"method": "ForestDiffusion/Forest-Flow", "receipt": {"path": "/cells/abc/receipt.json"}}
        self.assertEqual(resolve_sample_csv(cell), "/cells/abc/sample.csv")

    def test_a_receipt_without_a_path_is_not_invented(self):
        self.assertIsNone(resolve_sample_csv({"method": "ForestDiffusion/Forest-Flow", "receipt": {}}))


class AdapterBoundaryTests(unittest.TestCase):
    def tearDown(self):
        sys.modules.pop("captured_review_fix_tabsyn_bindings", None)

    def _namespace(self, verify_runtime):
        return {
            "__file__": "pinned-adapter.py",
            "verify_runtime": verify_runtime,
            "verify_operation": lambda *_args, **_kwargs: None,
        }

    def test_rejected_call_is_refused_before_write(self):
        tree = ast.parse("ex" + "ec('1')")
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp) / "nested" / "adapter.py"
            with self.assertRaises(SystemExit):
                _load_allowlisted(tree, self._namespace(lambda: None), work)
            self.assertFalse(work.exists())
            self.assertFalse(work.parent.exists())

    def test_rejected_import_is_refused_before_write(self):
        tree = ast.parse("import subprocess")
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp) / "adapter.py"
            with self.assertRaises(SystemExit):
                _load_allowlisted(tree, self._namespace(lambda: None), work)
            self.assertFalse(work.exists())

    def test_allowlisted_module_receives_the_runtime_sentinel(self):
        tree = ast.parse("marker = verify_runtime()")
        sentinel = object()
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp) / "adapter.py"
            loaded = _load_allowlisted(tree, self._namespace(lambda: sentinel), work)
        self.assertIs(loaded["marker"], sentinel)


if __name__ == "__main__":
    unittest.main()
