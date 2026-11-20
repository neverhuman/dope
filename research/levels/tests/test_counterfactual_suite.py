"""Unit tests for the 7-case Spec 3 section 6.4 counterfactual sensitivity suite."""
import unittest

from research.levels.counterfactual_suite import (
    CASE_NAMES,
    SCHEMA,
    CounterfactualSuiteError,
    evaluate_counterfactual_case,
    evaluate_counterfactual_suite,
)


class CounterfactualSuiteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.suite = evaluate_counterfactual_suite(seed=20261120, rows=384)

    def test_suite_schema_and_determinism(self):
        self.assertEqual(self.suite["schema"], SCHEMA)
        self.assertEqual(self.suite["case_count"], 7)
        self.assertEqual(tuple(self.suite["cases"].keys()), CASE_NAMES)
        replay = evaluate_counterfactual_case("interaction_vs_shuffled_target", seed=20261120, rows=384)
        self.assertEqual(replay, self.suite["cases"]["interaction_vs_shuffled_target"])
        with self.assertRaises(CounterfactualSuiteError) as err:
            evaluate_counterfactual_case("unknown_private_case")
        self.assertEqual(str(err.exception), "counterfactual_unknown_case")

    def test_case_1_interaction_vs_shuffled_target(self):
        rec = self.suite["cases"]["interaction_vs_shuffled_target"]
        self.assertEqual(rec["x_max_abs_diff"], 0.0)
        self.assertEqual(rec["y_sorted_max_abs_diff"], 0.0)
        self.assertGreater(rec["full629_l2"], 0.05)
        self.assertGreater(rec["transfer512_l2"], 0.05)
        self.assertGreater(rec["transfer512_cosine_distance"], 1e-4)

    def test_case_2_linear_vs_xor_product(self):
        rec = self.suite["cases"]["linear_vs_xor_product"]
        self.assertGreater(rec["arm_a_max_univariate_corr"], 0.75)
        self.assertLess(rec["arm_b_max_univariate_corr"], 1e-10)
        self.assertGreater(rec["arm_b_product_corr"], 0.95)
        self.assertGreater(rec["full629_l2"], 0.05)
        self.assertGreater(rec["transfer512_l2"], 0.05)
        self.assertGreater(rec["transfer512_cosine_distance"], 1e-4)

    def test_case_3_homoscedastic_vs_rare_heteroscedastic(self):
        rec = self.suite["cases"]["homoscedastic_vs_rare_heteroscedastic"]
        self.assertLess(rec["mean_abs_diff"], 1e-12)
        self.assertLess(rec["std_abs_diff"], 1e-12)
        self.assertGreater(rec["full629_l2"], 0.05)
        self.assertGreater(rec["transfer512_l2"], 0.02)
        self.assertGreater(rec["transfer512_cosine_distance"], 1e-5)

    def test_case_4_smooth_trend_vs_high_freq_periodic(self):
        rec = self.suite["cases"]["smooth_trend_vs_high_freq_periodic"]
        self.assertLess(rec["var_abs_diff"], 1e-12)
        self.assertGreater(rec["full629_l2"], 0.05)
        self.assertGreater(rec["transfer512_l2"], 0.05)
        self.assertGreater(rec["transfer512_cosine_distance"], 1e-4)

    def test_case_5_task_budget_and_metric_shift(self):
        rec = self.suite["cases"]["task_budget_and_metric_shift"]
        self.assertEqual(rec["full629_l2"], 0.0)
        self.assertGreater(rec["transfer512_l2"], 0.5)
        self.assertGreater(rec["transfer512_cosine_distance"], 1e-3)

    def test_case_6_informative_vs_ignorable_missingness(self):
        rec = self.suite["cases"]["informative_vs_ignorable_missingness"]
        self.assertEqual(rec["missing_rate_a"], 0.25)
        self.assertEqual(rec["missing_rate_b"], 0.25)
        self.assertGreater(rec["full629_l2"], 0.05)
        self.assertGreater(rec["transfer512_l2"], 0.05)
        self.assertGreater(rec["transfer512_cosine_distance"], 1e-4)

    def test_case_7_wide_columns_beyond_64_and_reordering(self):
        rec = self.suite["cases"]["wide_columns_beyond_64_and_reordering"]
        self.assertEqual(rec["total_columns"], 72)
        self.assertEqual(rec["signal_columns"], (66, 67, 68, 69))
        self.assertEqual(rec["truncated64_l2"], 0.0)
        self.assertGreater(rec["wide_readout_l2"], 0.05)
        self.assertGreater(rec["wide_readout_cosine_distance"], 1e-4)
        self.assertLess(rec["reordering_l2"], 1e-5)
        self.assertLess(rec["reordering_cosine_distance"], 1e-9)


if __name__ == "__main__":
    unittest.main()
