"""Table 19 (review RMSE) is in original target units, and its caption says how they are formed."""

import ast
import json
import math
import unittest
from pathlib import Path

from research.benchmark.review_fixes import receipt_panel as panel
from research.benchmark.review_fixes.receipt_tables import rmse_table

HERE = Path(__file__).resolve().parent


class RmseUnitTests(unittest.TestCase):
    def test_lineage_rmse_rescales_the_projected_loss_by_the_training_target_range(self):
        task = {"task": "regression", "span": 10.0}
        losses = [0.04, 0.01, 0.09]
        self.assertEqual(panel.lineage_error(losses, task, "original"), {"kind": "rmse", "value": 2.0})
        self.assertTrue(math.isclose(panel.lineage_error(losses, task, "projected")["value"], 0.2))

    def test_every_rmse_the_panel_reports_requests_original_units(self):
        tree = ast.parse((HERE / "receipt_panel.py").read_text())
        kinds = [call.args[2].value for call in ast.walk(tree)
                 if isinstance(call, ast.Call) and getattr(call.func, "id", None) == "lineage_error"]
        self.assertEqual(kinds, ["original", "original"])
        declared = json.loads((HERE / "predeclare.json").read_text())["original_unit_rmse"]["regression"]
        self.assertIn("times (target_max - target_min)", declared)

    def test_the_caption_names_original_units_and_how_they_are_formed(self):
        row = {"method": "DOPE", "configuration": "native_selected", "size": 4, "auditor": "catboost",
               "n_regression": 3, "synthetic_rmse_median": 1.5, "real_rmse_median": 1.0}
        text = rmse_table({"rmse": [row]})
        self.assertIn("original target units", text)
        self.assertIn("times that split's target range", text)
        self.assertNotIn("RMSE is on each lineage's target min--max projected", text)


if __name__ == "__main__":
    unittest.main()
