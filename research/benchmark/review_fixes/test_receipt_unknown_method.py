"""Rubric 1d: duplicate identities and unknown methods raise in the review receipt reducer."""
import sys
import unittest
from pathlib import Path

from research.benchmark.review_fixes.receipt_panel import KNOWN_METHODS, reduce_cells

SCRIPTS = Path(__file__).resolve().parents[3] / "docs/whitepaper/scripts"


def cell(method="DOPE", configuration="features12_steps2048", dataset="a", size=4, seed=101):
    return {"method": method, "configuration": configuration, "dataset": dataset, "size_multiplier": size,
            "sample_seed": seed, "fit_seed": 11, "status": "ok"}


class ReceiptReducerFailsClosedTests(unittest.TestCase):
    def test_an_unknown_method_in_the_ledger_is_refused_not_skipped(self):
        with self.assertRaisesRegex(ValueError, "unknown method 'Renamed'"):
            reduce_cells([cell(), cell(method="Renamed")], "DOPE", "features12_steps2048")

    def test_an_unknown_requested_method_is_refused(self):
        with self.assertRaisesRegex(ValueError, "unknown method 'Nope'"):
            reduce_cells([cell()], "Nope", "x")

    def test_a_duplicate_identity_is_refused(self):
        with self.assertRaisesRegex(ValueError, "duplicate cell"):
            reduce_cells([cell(), cell()], "DOPE", "features12_steps2048")

    def test_known_methods_cover_the_pinned_receipt_ledgers(self):
        self.assertEqual(len(KNOWN_METHODS), 9)
        self.assertIn("ForestDiffusion/Forest-Flow", KNOWN_METHODS)
        self.assertIn("TabSyn", KNOWN_METHODS)

    def test_the_tabsyn_writers_reduce_the_pinned_ledgers(self):
        # review_wave2 and review_tabsyn reduce the authenticated TabSyn cells; the guard must accept them.
        sys.path.insert(0, str(SCRIPTS))
        try:
            import review_wave2
        finally:
            sys.path.remove(str(SCRIPTS))
        payload = review_wave2.payload_from_inputs(review_wave2.authenticated_inputs())
        self.assertEqual(len(payload["tabsyn_teaser"]), 3)


if __name__ == "__main__":
    unittest.main()
