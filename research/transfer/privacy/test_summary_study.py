"""Deterministic interface fixtures; these do not establish DP or release authority."""
import unittest

from research.transfer.privacy.summary_study import (
    SummaryStudyError,
    canonical_summary,
    summary_digest,
)


SUMMARY = {"format": "bounded-entity-dp-summary-v1",
           "statistics": {"feature_mean": [0.25, 0.75], "feature_count": [12.0]}}
ACCOUNTING = {"formal_dp_status": "verified", "independent_review": "passed",
              "unit": "entity", "adjacency": "add_remove",
              "mechanism_sha256": "a" * 64, "epsilon": 1.0, "delta": 1e-6}


class SummaryStudyTests(unittest.TestCase):
    def test_only_bounded_summary_shape_crosses_the_pure_postprocess_boundary(self):
        raw = canonical_summary(SUMMARY, accounting=ACCOUNTING)
        self.assertIn(b'"format":"bounded-entity-dp-summary-v1"', raw)
        self.assertEqual(summary_digest(SUMMARY, accounting=ACCOUNTING),
                         summary_digest(SUMMARY, accounting=ACCOUNTING))

    def test_default_deny_or_mismatched_accounting_refuses(self):
        for change in ({"formal_dp_status": "unverified_design"},
                       {"independent_review": "pending"}, {"unit": "row"},
                       {"adjacency": "replace_one"}, {"epsilon": -1.0}):
            bad = ACCOUNTING | change
            with self.subTest(change=change), self.assertRaisesRegex(
                    SummaryStudyError, "^summary_accounting_not_admitted$"):
                canonical_summary(SUMMARY, accounting=bad)

    def test_row_payloads_and_unbounded_shapes_are_rejected(self):
        for bad in ({"rows": [[1, 2]], "format": "bounded-entity-dp-summary-v1",
                     "statistics": {"feature": [1.0]}},
                    {"format": "bounded-entity-dp-summary-v1",
                     "statistics": {"feature": [float("nan")]}}):
            with self.subTest(bad=bad), self.assertRaisesRegex(
                    SummaryStudyError, "^invalid_summary$"):
                canonical_summary(bad, accounting=ACCOUNTING)


if __name__ == "__main__":
    unittest.main()
