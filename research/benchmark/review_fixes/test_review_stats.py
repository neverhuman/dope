"""Checks for the pre-declared review-fix statistics. No ledger is required."""

import unittest
import hashlib
import json
from pathlib import Path

import numpy as np

from research.benchmark.review_fixes.stats import (
    cluster_of,
    delta_method_se,
    family_of,
    median_ci,
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


class MedianTests(unittest.TestCase):
    def test_cluster_ids_must_match_values(self):
        with self.assertRaises(ValueError):
            median_ci([1.0, 2.0, 1000.0], "length", ["a", "b"])


class DeltaTests(unittest.TestCase):
    def test_se_blows_up_as_the_gap_shrinks(self):
        wide = delta_method_se([1.0, 1.2, 0.8], null_loss=2.0, trtr_loss=0.0)
        tight = delta_method_se([1.0, 1.2, 0.8], null_loss=1.01, trtr_loss=1.0)
        self.assertLess(wide, tight)

    def test_zero_gap_is_undefined(self):
        self.assertIsNone(delta_method_se([1.0, 1.1], 1.0, 1.0))


class TableTests(unittest.TestCase):
    def test_column_spec_must_match_the_header_and_every_row(self):
        from research.benchmark.review_fixes.review_tex import longtable

        with self.assertRaises(ValueError):
            longtable("Caption. No Holm family.", "A & B", ["1 \\\\"], "ll")
        with self.assertRaises(ValueError):
            longtable("Caption without the family.", "A & B", ["1 & 2 \\\\"], "lr")
        body = longtable("Caption. No Holm family.", "A & B", ["1 & 2 \\\\"], "lr")
        self.assertIn("\\begin{longtable}{lr}\n\\caption{Caption. No Holm family.}", body)


class HistoricalBootstrapConformance(unittest.TestCase):
    def test_singleton_fast_path_preserves_frozen_intervals_and_rng_state(self):
        from research.benchmark.review_fixes import stats
        path = Path(__file__).with_name('stats-hier-conformance.json')
        raw = path.read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(),
                         'f548379b386b9151e8854d351ac5d4aefec3bb9899b2fd956b66de66ae06da9e')
        fixture = json.loads(raw)
        self.assertEqual(stats.DRAWS, fixture['draws'])
        for case in fixture['cases']:
            with self.subTest(case=case['name']):
                rng = stats.rng_for('hier-conformance|' + case['name'])
                result = stats._hier_median_ci(np.asarray(case['values']), case['clusters'], rng)
                self.assertEqual(result, case['result'])
                state = json.dumps(rng.bit_generator.state, sort_keys=True, separators=(',', ':')).encode()
                self.assertEqual(hashlib.sha256(state).hexdigest(), case['rng_state_sha256'])

    def test_numpy_singleton_draw_consumes_no_random_state(self):
        from research.benchmark.review_fixes import stats
        rng = stats.rng_for('singleton-state-conformance')
        before = json.dumps(rng.bit_generator.state, sort_keys=True)
        self.assertTrue(np.array_equal(rng.integers(0, 1, size=37), np.zeros(37)))
        self.assertEqual(json.dumps(rng.bit_generator.state, sort_keys=True), before)


if __name__ == "__main__":
    unittest.main()
