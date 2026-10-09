"""predeclare_v2 statistics on small fixtures. No benchmark data is read."""

import unittest

from research.benchmark.review_fixes import stats_v2 as S
from research.benchmark.review_fixes.stats import tost_mean


def fits(dataset, values, cluster=None, seeds=(11, 23, 37, 53, 71)):
    return [{"dataset": dataset, "cluster": cluster or "lineage:" + dataset,
             "fit_seed": seed, "value": value} for seed, value in zip(seeds, values)]


class HodgesLehmannTests(unittest.TestCase):
    def test_walsh_median_on_a_known_triple(self):
        self.assertEqual(S.hodges_lehmann([1.0, 2.0, 3.0]), 2.0)

    def test_empty_and_nonfinite_values(self):
        self.assertIsNone(S.hodges_lehmann([]))
        self.assertEqual(S.hodges_lehmann([float("nan"), 4.0]), 4.0)


class OutlierEquivalenceTests(unittest.TestCase):
    def test_mean_tost_fails_where_the_robust_estimand_is_inside_the_margin(self):
        items_left, items_right = [], []
        for index in range(30):
            dataset = f"{index:016x}"
            diff = 50.0 if index == 0 else 0.001 * (index % 3)
            items_left += fits(dataset, [0.9 + diff] * 3, seeds=(11, 23, 37))
            items_right += fits(dataset, [0.9] * 3, seeds=(11, 23, 37))
        paired = S.nested_paired(items_left, items_right, "toy-outlier", draws=400)
        self.assertFalse(tost_mean(paired["differences"])["equivalent"])
        self.assertLess(abs(paired["hl"]), 0.01)
        self.assertTrue(S.equivalence_hl(paired)["equivalent"])
        self.assertTrue(S.equivalence_hl(paired)["non_inferior"])


class NestedBootstrapTests(unittest.TestCase):
    def test_single_lineage_does_not_identify_an_interval(self):
        result = S.nested_level(fits("aa00000000000001", [0.1, 0.2, 0.9]), "toy-single", draws=50)
        self.assertEqual(result["n"], 1)
        self.assertIsNone(result["lo"])
        self.assertEqual(result["ci_status"], "not_identified_single_lineage")

    def test_min_fits_drops_thin_lineages(self):
        items = fits("aa00000000000001", [0.5, 0.6, 0.7]) + fits("aa00000000000002", [0.4, 0.5])
        self.assertEqual(S.nested_level(items, "toy-min", min_fits=3, draws=50)["n"], 1)
        self.assertEqual(S.nested_level(items, "toy-min", min_fits=1, draws=50)["n"], 2)

    def test_same_label_gives_the_same_interval(self):
        items = []
        for index in range(12):
            items += fits(f"{index:016x}", [0.5 + 0.01 * index, 0.6, 0.55], cluster="fri" if index < 4 else None)
        first = S.nested_level(items, "toy-repeat", draws=300)
        second = S.nested_level(items, "toy-repeat", draws=300)
        self.assertEqual((first["lo"], first["hi"]), (second["lo"], second["hi"]))
        self.assertLessEqual(first["lo"], first["median"])
        self.assertLessEqual(first["median"], first["hi"])

    def test_duplicates_and_cluster_conflicts_refuse(self):
        with self.assertRaises(ValueError):
            S.lineage_table(fits("aa00000000000001", [0.1, 0.2], seeds=(11, 11)))
        mixed = fits("aa00000000000001", [0.1], cluster="fri") + fits(
            "aa00000000000001", [0.2], cluster="bng", seeds=(23,))
        with self.assertRaises(ValueError):
            S.lineage_table(mixed)

    def test_paired_uses_only_shared_lineages(self):
        left = fits("aa00000000000001", [0.9, 0.9, 0.9]) + fits("aa00000000000002", [0.8, 0.8, 0.8])
        right = fits("aa00000000000002", [0.5, 0.5, 0.5])
        paired = S.nested_paired(left, right, "toy-shared", draws=50)
        self.assertEqual(paired["n"], 1)
        self.assertAlmostEqual(paired["median"], 0.3)
        self.assertEqual(paired["ci_status"], "not_identified_single_lineage")


class YuenAndSeedTests(unittest.TestCase):
    def test_yuen_trimmed_mean_ignores_the_tails(self):
        result = S.yuen_tost([0.001] * 18 + [-40.0, 40.0], margin=0.02)
        self.assertAlmostEqual(result["trimmed_mean"], 0.001)
        self.assertTrue(result["equivalent"])

    def test_seed_rank_counts_strict_maxima_only(self):
        rows = []
        for index in range(10):
            dataset = f"{index:016x}"
            values = {11: 0.9, 23: 0.8, 37: 0.7, 53: 0.6, 71: 0.5} if index < 4 else \
                {11: 0.5, 23: 0.9, 37: 0.7, 53: 0.6, 71: 0.8}
            rows += [{"dataset": dataset, "fit_seed": seed, "value": value} for seed, value in values.items()]
        rows += [{"dataset": "ffffffffffffffff", "fit_seed": 11, "value": 1.0}]
        result = S.seed_rank_test(rows)
        self.assertEqual((result["n"], result["seed11_top"]), (10, 4))
        self.assertGreater(result["p_one_sided"], 0.0)


if __name__ == "__main__":
    unittest.main()
