"""Persistent scalar-only singleton-CI negatives; no data/process/registry reads."""
import math
import unittest
from pathlib import Path
from unittest.mock import patch

from docs.whitepaper.scripts import review_fit_seeds as FIT


def completed_fits(dataset, values):
    return [{"dataset": dataset, "cluster": "lineage:" + dataset,
             "fit_seed": fit_seed, "retention": value}
            for fit_seed, value in zip(FIT.FIT_SEEDS, values)]


class SingletonCITests(unittest.TestCase):
    def test_single_lineage_keeps_descriptive_point_and_actual_fit_count(self):
        for values, expected in (((.1, .2, .8, .9), .5), ((.2, .8), .5), ((.3,), .3)):
            with self.subTest(values=values):
                result = FIT._nested_summary(completed_fits("aa00000000000001", values), "toy-singleton")
                self.assertEqual((result["n"], result["n_fits"], result["median"]), (1, len(values), expected))
                self.assertEqual(result["estimand"], FIT.ESTIMAND)
                self.assertEqual(result["ci_status"], "not_identified_single_lineage")
                for key in ("lo", "hi", "iid_lo", "iid_hi"):
                    self.assertIsNone(result[key])

    def test_null_cohort_interval_refuses_before_any_formatting(self):
        block = FIT._nested_summary(completed_fits("aa00000000000001", (.1, .2, .8, .9)), "toy-singleton")
        with patch.object(FIT, "ci") as ci, patch.object(FIT, "fmt") as fmt:
            with self.assertRaisesRegex(SystemExit, "refusing table publication"):
                FIT.render({"blocks": [block]})
            ci.assert_not_called()
            fmt.assert_not_called()

    def test_emit_refuses_before_any_publication_write(self):
        block = FIT._nested_summary(completed_fits("aa00000000000001", (.1, .2, .8, .9)), "toy-singleton")
        with patch.object(FIT, "reduce_panel", return_value={"blocks": [block]}), patch.object(Path, "mkdir") as mkdir, patch.object(Path, "write_text") as write, patch.object(FIT, "write_tex") as tex:
            with self.assertRaisesRegex(SystemExit, "refusing table publication"):
                FIT.emit({"cells": []})
            mkdir.assert_not_called()
            write.assert_not_called()
            tex.assert_not_called()

    def test_multiple_lineages_still_bootstrap_without_inventing_fit_seeds(self):
        items = completed_fits("aa00000000000001", (.1, .2, .3, .4))
        items += completed_fits("bb00000000000002", (.6, .7, .8, .9))
        with patch.object(FIT, "DRAWS", 64), patch.dict(FIT.median_ci.__globals__, {"DRAWS": 64}):
            result = FIT._nested_summary(items, "toy-multiple-lineages")
        self.assertEqual((result["n"], result["n_fits"]), (2, 8))
        self.assertAlmostEqual(result["median"], .5)
        self.assertTrue(math.isfinite(result["lo"]) and math.isfinite(result["hi"]))
        self.assertLessEqual(result["lo"], result["hi"])
        self.assertEqual(FIT.FIT_SEEDS, (23, 37, 53, 71))

    def test_empty_summary_remains_null_and_cannot_print_a_placeholder(self):
        block = FIT._nested_summary([], "toy-empty")
        self.assertEqual((block["n"], block["n_fits"]), (0, 0))
        self.assertIsNone(block["median"])
        self.assertIsNone(block["lo"])
        self.assertIsNone(block["hi"])
        with self.assertRaisesRegex(SystemExit, "refusing table publication"):
            FIT.render({"blocks": [block]})


if __name__ == "__main__":
    unittest.main()
