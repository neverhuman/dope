"""Path rules for published sample files. No ledger and no CSV is opened."""

import unittest

from research.benchmark.review_fixes.sample_paths import resolve_sample_csv


class PathTests(unittest.TestCase):
    def test_dope_uses_the_metric_sibling_not_sample_evidence(self):
        cell = {
            "method": "DOPE",
            "metric_receipt": {"metric_path": "/attempts/n1-seed101.metric.json"},
        }
        self.assertEqual(resolve_sample_csv(cell), "/attempts/n1-seed101.csv")

    def test_with_suffix_would_be_the_wrong_dope_name(self):
        found = resolve_sample_csv({
            "method": "DOPE",
            "metric_receipt": {"metric_path": "/attempts/n1-seed101.metric.json"},
        })
        self.assertFalse(found.endswith(".metric.csv"))

    def test_arf_prefers_the_sample_path(self):
        cell = {"method": "ARF", "sample_evidence": {
            "sample_path": "/arf/n4-seed101.csv",
            "metric_path": "/arf/n4-seed101.metric.json",
        }}
        self.assertEqual(resolve_sample_csv(cell), "/arf/n4-seed101.csv")

    def test_neural_uses_a_csv_path(self):
        cell = {"method": "CTGAN", "sample_evidence": {"path": "/sdv/4n-seed101.csv", "sha256": "abc"}}
        self.assertEqual(resolve_sample_csv(cell), "/sdv/4n-seed101.csv")

    def test_forest_uses_the_sibling_sample_not_the_repeat(self):
        cell = {"method": "ForestDiffusion/Forest-Flow", "receipt": {"path": "/cells/abc/receipt.json"}}
        self.assertEqual(resolve_sample_csv(cell), "/cells/abc/sample.csv")

    def test_a_receipt_without_a_path_is_not_invented(self):
        self.assertIsNone(resolve_sample_csv({"method": "ForestDiffusion/Forest-Flow", "receipt": {}}))


if __name__ == "__main__":
    unittest.main()
