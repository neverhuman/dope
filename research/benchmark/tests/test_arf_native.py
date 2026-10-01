"""Checks the frozen ARF native density formula on a simple leaf mixture."""

import math
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.stats import norm

from research.benchmark.arf_native import heldout_mean_log_density, prepare_frames
from research.benchmark.arf_select import winner
from research.benchmark.arf_short_source import SOURCE_SHA256, materialize


class OneLeafForest:
    def apply(self, frame):
        return np.zeros((len(frame), 1), dtype=int)


class ArfNativeTests(unittest.TestCase):
    def test_continuous_leaf_density_matches_product_normal(self):
        model = SimpleNamespace(
            orig_colnames=["c0", "c1"], num_trees=1,
            factor_cols=pd.Series([False, False], index=["c0", "c1"]),
            levels={}, clf=OneLeafForest(),
            bnds=pd.DataFrame([(0, 0, "c0", 1.0), (0, 0, "c1", 1.0)],
                               columns=["tree", "nodeid", "variable", "cvg"]),
            params=pd.DataFrame([(0, 0, "c0", 0.5, 0.2, -math.inf, math.inf),
                                 (0, 0, "c1", 0.4, 0.1, -math.inf, math.inf)],
                                columns=["tree", "nodeid", "variable", "mean",
                                         "sd", "min", "max"]),
            class_probs=pd.DataFrame(columns=["tree", "nodeid", "variable",
                                              "value", "prob"]),
        )
        validation = pd.DataFrame({"c0": [0.5, 0.3], "c1": [0.4, 0.5]})
        expected = np.mean(norm.logpdf(validation["c0"], 0.5, 0.2)
                           + norm.logpdf(validation["c1"], 0.4, 0.1))
        self.assertAlmostEqual(heldout_mean_log_density(model, validation), expected)

    def test_training_observed_binary_columns_are_categorical(self):
        fit, validation, categories = prepare_frames(
            np.array([[0.0, 0.1], [1.0, 0.2], [0.0, 0.3]]),
            np.array([[1.0, 0.4]]))
        self.assertEqual(categories, ["c0"])
        self.assertEqual(str(fit["c0"].dtype), "category")
        self.assertEqual(str(validation["c0"].dtype), "category")

    def test_selection_uses_native_density_then_charged_bytes(self):
        trials = [{"status": "ok", "native_kpi": value,
                   "artifact_bytes": size,
                   "job": {"trial": trial, "config": {"num_trees": trial}}}
                  for trial, value, size in ((0, -2.0, 100),
                                             (1, -1.0, 200),
                                             (2, -1.0, 150))]
        self.assertEqual(winner(trials)["job"]["trial"], 2)
        self.assertIsNone(winner([{"status": "timeout"}]))

    def test_short_grid_patch_reconstructs_frozen_source(self):
        output = Path(__file__).parents[3] / "target/arf-short-source-test.py"
        self.assertEqual(materialize(output), SOURCE_SHA256)
        self.assertEqual(materialize(output), SOURCE_SHA256)


if __name__ == "__main__":
    unittest.main()
