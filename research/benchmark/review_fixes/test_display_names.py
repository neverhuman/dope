"""Printed table names: identifiers map to reader-facing names by exact match; everything else passes through."""
import unittest

from research.benchmark.review_fixes import receipt_panel
from research.benchmark.review_fixes.display_names import DISPLAY, display


class DisplayNameTests(unittest.TestCase):
    def test_known_identifiers_print_readable_names(self):
        self.assertEqual(display("GaussianCopula"), "Gaussian copula")
        self.assertEqual(display("independent_marginals"), "Independent marginals")
        self.assertEqual(display("ForestDiffusion/Forest-Flow"), "Forest-Flow")
        self.assertEqual(display("native_selected"), "native selected")
        self.assertEqual(display("catboost"), "CatBoost")

    def test_unknown_values_and_dataset_names_pass_through_unchanged(self):
        for value in ("feynman_I_12_1", "strogatz_vdp1", "DOPE", "ARF", "TVAE", "matched density", 11):
            self.assertEqual(display(value), str(value))

    def test_tex_name_escapes_after_mapping_and_keeps_raw_identifiers_escaped(self):
        self.assertEqual(receipt_panel.tex_name("independent_marginals"), "Independent marginals")
        self.assertEqual(receipt_panel.tex_name("feynman_I_12_1"), r"feynman\_I\_12\_1")
        self.assertEqual(receipt_panel.tex_name("A&B"), r"A\&B")

    def test_every_printed_name_is_distinct_so_no_two_rows_merge(self):
        self.assertEqual(len(set(DISPLAY.values())), len(DISPLAY))
        self.assertTrue(set(DISPLAY.values()).isdisjoint(DISPLAY))


if __name__ == "__main__":
    unittest.main()
