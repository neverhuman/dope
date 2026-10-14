"""Rubric 1d: duplicate identities and unknown methods raise in the review receipt reducer."""
import unittest

from research.benchmark.review_fixes.receipt_panel import KNOWN_METHODS, reduce_cells


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
        self.assertEqual(len(KNOWN_METHODS), 8)
        self.assertIn("ForestDiffusion/Forest-Flow", KNOWN_METHODS)


if __name__ == "__main__":
    unittest.main()
