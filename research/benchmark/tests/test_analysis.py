import unittest

from research.benchmark.analysis import compare


class AnalysisTests(unittest.TestCase):
    @staticmethod
    def all_positive_pairs(dataset_count, methods=("independent_marginals",)):
        names = ("dope", *methods)
        matrix = [{"dataset": str(index), "method": name, "panel": "public_core",
                   "track": "common_numeric", "tier": "l3", "applicable": True}
                  for index in range(dataset_count) for name in names]
        reports = [{"dataset": str(index), "method": name, "status": "ok",
                    "eligible": name == "dope", "score": 80 if name == "dope" else None}
                   for index in range(dataset_count) for name in names]
        lock = {"methods": {name: {"group": "compact", "status": "locked"}
                            for name in methods}}
        return compare(matrix, reports, lock)

    def test_positive_interval_without_sign_evidence_is_inconclusive(self):
        result = self.all_positive_pairs(2)
        row = result["comparators"][0]
        self.assertEqual(row["gate_pass_familywise_95_ci"], (1, 1))
        self.assertEqual(row["gate_pass_holm_adjusted_p"], 0.25)
        self.assertFalse(row["gate_pass_superiority"])
        self.assertFalse(result["overall_superiority"])

    def test_holm_correction_is_required_for_each_comparator(self):
        result = self.all_positive_pairs(5, ("independent_marginals", "ChowLiu"))
        for row in result["comparators"]:
            self.assertEqual(row["gate_pass_one_sided_sign_p"], 0.03125)
            self.assertEqual(row["gate_pass_holm_adjusted_p"], 0.0625)
            self.assertFalse(row["gate_pass_superiority"])
        self.assertFalse(result["overall_superiority"])

    def test_corrected_sign_evidence_and_positive_interval_allow_claim(self):
        result = self.all_positive_pairs(6, ("independent_marginals", "ChowLiu"))
        for row in result["comparators"]:
            self.assertEqual(row["gate_pass_holm_adjusted_p"], 0.03125)
            self.assertTrue(row["gate_pass_superiority"])
        self.assertTrue(result["overall_superiority"])

    def test_unavailable_method_is_not_a_win(self):
        matrix = [{"dataset": "a", "method": name, "panel": "public_core",
                   "track": "common_numeric", "tier": "l3", "applicable": True}
                  for name in ("dope", "GEM-T")]
        lock = {"methods": {"GEM-T": {"group": "compact", "status": "availability_pending"}}}
        result = compare(matrix, [], lock)
        self.assertFalse(result["overall_superiority"])
        self.assertEqual(result["comparators"], [])

    def test_missing_run_counts_as_failed_and_remains_visible(self):
        matrix = [{"dataset": "a", "method": name, "panel": "public_core",
                   "track": "common_numeric", "tier": "l3", "applicable": True}
                  for name in ("dope", "independent_marginals")]
        lock = {"methods": {"independent_marginals": {"group": "compact", "status": "locked"}}}
        report = [{"dataset": "a", "method": "dope", "status": "ok", "eligible": True, "score": 80}]
        result = compare(matrix, report, lock)["comparators"][0]
        self.assertEqual(result["gate_pass_difference"], 1)
        self.assertEqual(result["failures"]["baseline_missing_or_timeout"], 1)
        self.assertFalse(result["gate_pass_superiority"])
        self.assertEqual(result["mfs_v2_conclusion"], "inconclusive")


if __name__ == "__main__":
    unittest.main()
