"""Path rules for published sample files. No ledger and no CSV is opened."""

import ast
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from research.benchmark.review_fixes.sample_paths import resolve_sample_csv
from research.benchmark.review_fixes.tabsyn_sample import _load_allowlisted, _reject_adapter


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

    def test_aliased_sink_is_rejected_before_write(self):
        tree = ast.parse("from builtins import " + "exec as run\nrun('1')")
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp) / "adapter.py"
            with self.assertRaises(SystemExit):
                _load_allowlisted(tree, self._namespace(lambda: None), work)
            self.assertFalse(work.exists())

    def test_substituted_staging_bytes_are_rejected(self):
        tree = ast.parse("VALUE = 1\n")
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp) / "adapter.py"
            original = importlib.util.spec_from_file_location

            def swap(name, path, *args, **kwargs):
                Path(path).write_text("VALUE = 99\n")
                return original(name, path, *args, **kwargs)

            with patch.object(importlib.util, "spec_from_file_location", swap):
                with self.assertRaises(SystemExit):
                    _load_allowlisted(tree, self._namespace(lambda: None), work)

    def test_symlink_staging_path_is_rejected(self):
        tree = ast.parse("VALUE = 1\n")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            victim = root / "victim.py"
            victim.write_text("VALUE = 99\n")
            work = root / "adapter.py"
            work.symlink_to(victim.name)
            with self.assertRaises(SystemExit):
                _load_allowlisted(tree, self._namespace(lambda: None), work)
            self.assertEqual(victim.read_text(), "VALUE = 99\n")

    def test_assigned_sink_alias_is_rejected_before_write(self):
        tree = ast.parse("run = eval\nrun('1')\n")
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp) / "adapter.py"
            with self.assertRaises(SystemExit):
                _load_allowlisted(tree, self._namespace(lambda: None), work)
            self.assertFalse(work.exists())

    def test_method_call_and_import_alias_still_load(self):
        tree = ast.parse(
            "class Model:\n"
            "    def eval(self):\n"
            "        return 7\n"
            "from pathlib import Path as path\n"
            "marker = Model().eval()\n"
            "located = path('.')\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp) / "adapter.py"
            loaded = _load_allowlisted(tree, self._namespace(lambda: None), work)
        self.assertEqual(loaded["marker"], 7)
        self.assertEqual(loaded["located"], Path("."))

    def test_pinned_adapter_passes_the_call_boundary(self):
        source = Path(__file__).resolve().parents[1] / "tabsyn_adapter.py"
        _reject_adapter(ast.parse(source.read_text()))


if __name__ == "__main__":
    unittest.main()
