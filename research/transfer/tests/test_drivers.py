import unittest

import numpy as np

from research.transfer.drivers import (
    historical_target_correlation, importance_agreement, ordinal_association,
    pearson, permutation_importance,
)


class DriverTests(unittest.TestCase):
    def test_xor_importance_group_permutation_and_undefined_agreement(self):
        rng = np.random.default_rng(0)
        x1 = rng.integers(0, 2, 240)
        x2 = rng.integers(0, 2, 240)
        target = np.bitwise_xor(x1, x2).astype(float)
        self.assertLess(abs(pearson(x1, target)), 0.15)
        self.assertLess(abs(pearson(x2, target)), 0.15)
        columns = (tuple(x1), tuple(x2))

        def loss(values):
            pred = np.logical_xor(np.asarray(values[0]) > 0, np.asarray(values[1]) > 0).astype(float)
            return float(np.mean((pred - target) ** 2))

        report = permutation_importance(loss, columns, {"x1": (0,), "x2": (1,), "pair": (0, 1)},
                                        learner_id="xor-learner", query_population="fixture-rows", seed=3, repeats=8)
        self.assertGreater(report["importance"]["x1"], 0.2)
        self.assertGreater(report["importance"]["x2"], 0.2)
        self.assertGreater(report["importance"]["pair"], 0.2)
        self.assertEqual(report["learner_id"], "xor-learner")
        self.assertEqual(report["query_population"], "fixture-rows")
        proxy = historical_target_correlation(x1, target)
        self.assertEqual(proxy["path"], "historical-proxy-v0")
        self.assertIsNone(proxy["learner_id"])
        flat = {"importance": {"x1": 0.0, "x2": 0.0, "pair": 0.0}, "learner_id": "constant", "query_population": "fixture-rows"}
        agreement = importance_agreement(report, flat)
        self.assertIn(agreement["status"], ("low_signal", "tied"))
        self.assertIsNone(agreement["agreement"])
        with self.assertRaises(ValueError):
            ordinal_association((0, 1, 2), (1, 0, 1), scale="nominal")
        color = ((1, 0, 1, 0), (0, 1, 0, 1))

        def one_hot_loss(values):
            return float(sum(left + right for left, right in zip(values[0], values[1])))

        grouped = permutation_importance(one_hot_loss, color, {"color": (0, 1)},
                                         learner_id="group-learner", query_population="fixture-rows", seed=1, repeats=4)
        self.assertAlmostEqual(grouped["importance"]["color"], 0.0)


if __name__ == "__main__":
    unittest.main()
