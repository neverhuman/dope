import unittest

from research.benchmark.analysis import compare


class AnalysisTests(unittest.TestCase):
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
