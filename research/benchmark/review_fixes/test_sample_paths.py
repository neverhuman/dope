"""Path rules for published sample files, and the TabSyn sampler's adapter and failure boundaries.

No ledger, artifact, GPU, or vendor runtime is opened; sampling is a fake that
returns a small matrix or raises, and CSVs are written only inside a test-owned folder.
"""

import ast
import importlib.util
import os
import py_compile
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from research.benchmark.review_fixes.sample_paths import resolve_sample_csv
from research.benchmark.review_fixes import tabsyn_sample
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

    def test_builtin_module_alias_is_rejected_before_write(self):
        tree = ast.parse("import builtins as b\nb.ex" + "ec('1')")
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp) / "adapter.py"
            with self.assertRaises(SystemExit):
                _load_allowlisted(tree, self._namespace(lambda: None), work)
            self.assertFalse(work.exists())

    def test_assigned_builtin_alias_is_rejected_before_write(self):
        tree = ast.parse("alias = builtins\nalias.ex" + "ec('1')")
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp) / "adapter.py"
            with self.assertRaises(SystemExit):
                _load_allowlisted(tree, self._namespace(lambda: None), work)
            self.assertFalse(work.exists())

    def test_late_staging_change_executes_only_captured_payload(self):
        tree = ast.parse("VALUE = 1\n")
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp) / "adapter.py"
            original = importlib.util.module_from_spec

            def swap(spec):
                Path(spec.origin).write_text("VALUE = 99\n")
                return original(spec)

            with patch.object(importlib.util, "module_from_spec", swap):
                loaded = _load_allowlisted(tree, self._namespace(lambda: None), work)
            self.assertEqual(loaded["VALUE"], 1)

    def test_timestamp_valid_cache_cannot_replace_captured_payload(self):
        tree = ast.parse("VALUE = 1\n")
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp) / "adapter.py"
            prefix = "from captured_review_fix_tabsyn_bindings import verify_runtime, verify_operation\n"
            work.write_text(prefix + "VALUE = 9\n")
            stamp = int(time.time())
            os.utime(work, (stamp, stamp))
            py_compile.compile(str(work), doraise=True)
            original_fsync = os.fsync

            def align_mtime(descriptor):
                original_fsync(descriptor)
                os.utime(work, (stamp, stamp))

            with patch.object(os, "fsync", align_mtime):
                loaded = _load_allowlisted(tree, self._namespace(lambda: None), work)
            self.assertEqual(loaded["VALUE"], 1)

    def test_pinned_adapter_passes_the_call_boundary(self):
        source = Path(__file__).resolve().parents[1] / "tabsyn_adapter.py"
        _reject_adapter(ast.parse(source.read_text()))



class SampleFailureBoundaryTests(unittest.TestCase):
    """One sampled cell: known failures get a fixed code, anything else propagates."""

    SENTINEL = "zq-source-header-extremum-7781"

    def setUp(self):
        import numpy as np

        self.np = np
        self.folder = tempfile.TemporaryDirectory()
        self.final = Path(self.folder.name) / "sample-101-1n.csv"
        self.matrix = np.array([[0.5, 1.0, 2.0], [1.5, 3.0, 4.0]])

    def tearDown(self):
        self.folder.cleanup()

    def cell(self, sample, rows=2):
        return tabsyn_sample._sample_cell(sample, self.final, rows, "artifact", rows, 101, ["inventory"],
                                          device="cuda:0", admission={"grant": "review-fixes"})

    def assert_nothing_published(self):
        self.assertFalse(self.final.exists())
        self.assertFalse(self.final.with_suffix(".csv.tmp").exists())

    def raising(self, error):
        def sample(*_args, **_kwargs):
            raise error
        return sample

    def test_a_sampled_matrix_is_published_target_last(self):
        calls = []

        def sample(*args, **kwargs):
            calls.append((args, kwargs))
            return self.matrix

        self.assertEqual(self.cell(sample), ("ok", None))
        self.assertEqual(calls, [(("artifact", 2, 101, ["inventory"]),
                                  {"device": "cuda:0", "admission": {"grant": "review-fixes"}})])
        self.assertEqual(self.final.read_text().splitlines(),
                         ["f0,f1,target", "1.0,2.0,0.5", "3.0,4.0,1.5"])
        self.assertFalse(self.final.with_suffix(".csv.tmp").exists())

    def test_known_failures_are_recorded_as_fixed_codes(self):
        class OutOfMemoryError(RuntimeError):
            """Stands in for torch.cuda.OutOfMemoryError, a RuntimeError subclass."""

        cases = (
            (ValueError("TabSyn artifact custody differs " + self.SENTINEL), "sample_contract_refused"),
            (TimeoutError("whole fit deadline exceeded " + self.SENTINEL), "sample_deadline_exceeded"),
            (RuntimeError("sample admission does not match the artifact " + self.SENTINEL), "sample_runtime_failed"),
            (OutOfMemoryError("CUDA out of memory " + self.SENTINEL), "sample_runtime_failed"),
            (FileNotFoundError(self.SENTINEL), "sample_io_failed"),
        )
        for error, code in cases:
            with self.subTest(type(error).__name__):
                status, reason = self.cell(self.raising(error))
                self.assertEqual((status, reason), ("failed", code))
                self.assertNotIn(self.SENTINEL, reason)
                self.assert_nothing_published()

    def test_unexpected_errors_propagate_and_publish_nothing(self):
        for error in (KeyError(self.SENTINEL), TypeError(self.SENTINEL), AttributeError(self.SENTINEL),
                      NameError(self.SENTINEL), ZeroDivisionError(self.SENTINEL)):
            with self.subTest(type(error).__name__), self.assertRaises(type(error)):
                self.cell(self.raising(error))
            self.assert_nothing_published()

    def test_a_malformed_matrix_is_a_contract_refusal(self):
        np = self.np
        for matrix in ([[0.5, 1.0], [1.5, 3.0]], np.zeros((3, 3)), np.zeros((2, 1)), np.zeros(6)):
            with self.subTest(repr(matrix)[:40]):
                self.assertEqual(self.cell(lambda *_a, value=matrix, **_k: value), ("failed", "sample_contract_refused"))
                self.assert_nothing_published()

    def test_a_failed_rename_leaves_no_partial_file(self):
        with patch.object(Path, "replace", side_effect=PermissionError(self.SENTINEL)):
            self.assertEqual(self.cell(lambda *_a, **_k: self.matrix), ("failed", "sample_io_failed"))
        self.assert_nothing_published()

    def test_an_unexpected_write_error_propagates_and_removes_the_staging_file(self):
        def broken_writer(_handle):
            class Writer:
                def writerow(self, _row):
                    pass

                def writerows(self, _rows):
                    raise TypeError("unexpected writer state")
            return Writer()

        with patch.object(tabsyn_sample.csv, "writer", broken_writer), self.assertRaises(TypeError):
            self.cell(lambda *_a, **_k: self.matrix)
        self.assert_nothing_published()

    def test_a_deadline_is_not_reported_as_generic_io(self):
        self.assertEqual(tabsyn_sample.sample_failure_reason(TimeoutError()), "sample_deadline_exceeded")
        self.assertEqual(tabsyn_sample.sample_failure_reason(OSError()), "sample_io_failed")

    def test_the_sampler_has_no_catch_all_handler(self):
        """Every handler names explicit classes; sample_lineage delegates each cell to _sample_cell."""
        tree = ast.parse(Path(tabsyn_sample.__file__).read_text())
        broad = {"Exception", "BaseException"}
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler):
                self.assertIsNotNone(node.type, f"bare except at line {node.lineno}")
                names = node.type.elts if isinstance(node.type, ast.Tuple) else [node.type]
                for name in names:
                    self.assertFalse(isinstance(name, ast.Name) and name.id in broad, f"broad except at line {node.lineno}")
        lineage = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "sample_lineage")
        self.assertFalse(any(isinstance(node, ast.Try) for node in ast.walk(lineage)))
        self.assertTrue(any(isinstance(node, ast.Call) and getattr(node.func, "id", None) == "_sample_cell"
                            for node in ast.walk(lineage)))

if __name__ == "__main__":
    unittest.main()
