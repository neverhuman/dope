"""Checks for the pre-declared review-fix statistics. No ledger is required."""

import unittest

import numpy as np

from research.benchmark.review_fixes.stats import (
    cluster_of,
    delta_method_se,
    family_of,
    tost_mean,
)


class FamilyTests(unittest.TestCase):
    def test_prefixes(self):
        self.assertEqual(family_of("feynman_I_12_1"), "feynman")
        self.assertEqual(family_of("strogatz_vdp2"), "strogatz")
        self.assertEqual(family_of("588_fri_c4_1000_100"), "fri")
        self.assertEqual(family_of("bng_pbc"), "bng")
        self.assertEqual(family_of("215_2dplanes"), "other")

    def test_other_lineages_are_not_one_cluster(self):
        self.assertNotEqual(cluster_of("215_2dplanes", "aa"), cluster_of("adult", "bb"))
        self.assertEqual(cluster_of("feynman_a", "aa"), cluster_of("feynman_b", "bb"))


class TostTests(unittest.TestCase):
    def test_exact_zero_is_equivalent_at_the_declared_margin(self):
        result = tost_mean(np.zeros(40))
        self.assertTrue(result["equivalent"])
        self.assertLess(result["p"], 0.05)

    def test_a_shift_of_one_tenth_is_not_equivalent(self):
        result = tost_mean(np.full(40, 0.1))
        self.assertFalse(result["equivalent"])

    def test_margin_is_the_predeclared_two_hundredths(self):
        self.assertEqual(tost_mean([0.0, 0.0])["margin"], 0.02)


class DeltaTests(unittest.TestCase):
    def test_se_blows_up_as_the_gap_shrinks(self):
        wide = delta_method_se([1.0, 1.2, 0.8], null_loss=2.0, trtr_loss=0.0)
        tight = delta_method_se([1.0, 1.2, 0.8], null_loss=1.01, trtr_loss=1.0)
        self.assertLess(wide, tight)

    def test_zero_gap_is_undefined(self):
        self.assertIsNone(delta_method_se([1.0, 1.1], 1.0, 1.0))


if __name__ == "__main__":
    unittest.main()
