"""Replay-layout and comparison checks for rescore_cells. No benchmark data is read."""

import json
import math
import tempfile
import unittest
from pathlib import Path

from research.benchmark.review_fixes import rescore_cells as R


class RescoreTests(unittest.TestCase):
    def test_layout_finds_sample_cells_and_ignores_others(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fit = root / "aa00000000000001" / "fit-23"
            fit.mkdir(parents=True)
            (fit / "sample-101-4n.csv").write_text("0,1\n")
            (fit / "sample-101-4n.repeat.csv").write_text("0,1\n")
            (fit / "notes.txt").write_text("x")
            found = R.cells(root)
            self.assertEqual(len(found), 1)
            self.assertEqual((found[0]["fit_seed"], found[0]["sample_seed"], found[0]["size"]), (23, 101, 4))

    def test_comparison_is_exact_and_reports_the_largest_gap(self):
        fresh = {"catboost": {"retention": 0.5}, "linear": {"retention": None}, "mlp": {"retention": 0.25}}
        self.assertEqual(R.compare({"catboost": 0.5, "linear": None, "mlp": 0.25}, fresh), (True, 0.0))
        equal, worst = R.compare({"catboost": 0.5, "linear": None, "mlp": 0.5}, fresh)
        self.assertFalse(equal)
        self.assertAlmostEqual(worst, 0.25)
        self.assertEqual(R.compare(None, fresh), (False, math.inf))

    def test_stored_utility_reads_both_record_shapes(self):
        with tempfile.TemporaryDirectory() as tmp:
            old = Path(tmp) / "old.json"
            old.write_text(json.dumps({"status": "ok", "metrics": {"utility": {"catboost": {"retention": 0.9}}}}))
            new = Path(tmp) / "new.json"
            new.write_text(json.dumps({"status": "ok", "utility": {"mlp": {"retention": 1.1}}}))
            self.assertEqual(R._stored_utility(old)["catboost"], 0.9)
            self.assertEqual(R._stored_utility(new)["mlp"], 1.1)
            self.assertIsNone(R._stored_utility(Path(tmp) / "absent.json"))


if __name__ == "__main__":
    unittest.main()
