import json
import math
import tempfile
import unittest
from pathlib import Path

from research.benchmark.native_objective import score


ROOT = Path(__file__).resolve().parents[3]


class NativeObjectiveTests(unittest.TestCase):
    def test_marginal_density_uses_validation_only(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "target") as directory:
            root = Path(directory)
            (root / "model.json").write_text(json.dumps({"bins": 2, "columns": [
                {"kind": "histogram", "p": [1.0, 0.0]},
                {"kind": "binary", "p": 0.75}]}))
            valid = root / "validation.csv"
            valid.write_text("0.1,1\n0.4,0\n")
            result = score("independent_marginals", root, valid)
            self.assertAlmostEqual(result["value"], math.log(2) +
                                   (math.log(0.75) + math.log(0.25)) / 2)
            self.assertEqual(result["partition"], "validation")

    def test_chow_likelihood_includes_continuous_bin_width(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "target") as directory:
            root = Path(directory)
            (root / "model.json").write_text(json.dumps({"cards": [4, 2],
                "root_probs": [0.25] * 4,
                "edges": [{"parent": 0, "child": 1,
                           "conditional": [[0.8, 0.2]] * 4}]}))
            valid = root / "validation.csv"
            valid.write_text("0.1,1\n")
            result = score("Chow-Liu", root, valid)
            self.assertAlmostEqual(result["value"], math.log(0.2))


if __name__ == "__main__":
    unittest.main()
