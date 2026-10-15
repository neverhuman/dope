"""Bounded public-demo source/split/report regressions; no model campaign."""
import importlib.util
from pathlib import Path
import unittest
import tempfile
import json
from unittest.mock import patch
import builtins

ROOT=Path(__file__).resolve().parents[3]
spec=importlib.util.spec_from_file_location("public_airfoil_demo",ROOT/"examples/public-regression/run.py")
demo=importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)


class PublicDemoTests(unittest.TestCase):
    def rows(self):
        return [[float(frequency),float(group),0.1,float(group%3+20),0.01,100.0+group+frequency]
                for group in range(20) for frequency in range(20)]

    def test_physical_groups_do_not_cross_and_query_labels_cannot_choose_split(self):
        rows=self.rows();fit,query,manifest=demo.grouped_split(rows,1729)
        fit_groups={tuple(rows[i][j] for j in [1,2,3,4]) for i in fit}
        query_groups={tuple(rows[i][j] for j in [1,2,3,4]) for i in query}
        self.assertFalse(fit_groups&query_groups)
        changed=[row[:] for row in rows]
        for i in query:changed[i][5]=-999999.0
        self.assertEqual(demo.grouped_split(changed,1729),(fit,query,manifest))
        self.assertEqual([rows[i] for i in fit],[changed[i] for i in fit])

    def test_source_drift_refuses_before_runtime_or_output(self):
        original_import=builtins.__import__
        def forbid_learner_import(name,*args,**kwargs):
            if name=="numpy":raise AssertionError("learner runtime loaded before source admission")
            return original_import(name,*args,**kwargs)
        with patch("builtins.__import__",side_effect=forbid_learner_import):
            with self.assertRaisesRegex(ValueError,"public_source_digest_mismatch"):
                demo.run(Path("missing-native"),b"not-the-pinned-public-source",Path("target/never-created-demo-test"),1729)

    def test_unbounded_retention_and_html_do_not_invent_qualification(self):
        losses=demo.losses([10.0,20.0],[12.0,18.0])
        self.assertEqual(losses,{"mse_db_squared":4.0,"rmse_db":2.0,"mae_db":2.0})
        report=demo.render_report({"retention":-2.0,"privacy":"unmeasured","text":"<source>"})
        self.assertIn("-2.0",report);self.assertIn("unmeasured",report)
        self.assertIn("&lt;source&gt;",report);self.assertNotIn("<source>",report)

    def test_failed_steps_keep_terminal_cost_without_private_diagnostics(self):
        (ROOT/"target").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT/"target",prefix="public-demo-fixture-") as directory:
            output=Path(directory);attempts=[]
            def fail():raise ValueError("private fixture value")
            with self.assertRaises(ValueError):demo.measured_step(output,attempts,"native_fit",fail)
            records=[json.loads(line) for line in (output/"attempts.jsonl").read_text().splitlines()]
            self.assertEqual([r["state"] for r in records],["started","failed"])
            self.assertGreaterEqual(records[1]["elapsed_seconds"],0)
            self.assertEqual(attempts,[records[1]])
            self.assertNotIn("private fixture value",(output/"attempts.jsonl").read_text())

    def test_fractional_frequency_refuses_before_serializing_a_native_integer(self):
        (ROOT/"target").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT/"target",prefix="public-demo-frequency-") as directory:
            with self.assertRaisesRegex(ValueError,"public_frequency_type_mismatch"):
                demo.write_fit_csv(Path(directory)/"fit.csv",[[1.5,0.0,0.1,20.0,0.01,100.0]],[0])

    def test_native_diagnostics_keep_hash_cost_code_and_exclude_raw_private_text(self):
        (ROOT/"target").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT/"target",prefix="public-demo-command-") as directory:
            path=Path(directory)/"diagnostic.json"
            stderr=b"private header value: integer_precision_or_type_mismatch"
            demo.write_native_diagnostic(path,"compile-raw",1,b"",stderr,0.25,"failed")
            record=json.loads(path.read_text())
            self.assertEqual(record["static_error_codes"],["integer_precision_or_type_mismatch"])
            self.assertEqual(record["stderr_sha256"],demo.digest(stderr))
            self.assertEqual(record["elapsed_seconds"],0.25)
            self.assertNotIn("private header value",path.read_text())


if __name__=="__main__":unittest.main()
