"""v2 source-family sensitivity on synthetic fixtures. No benchmark data is read."""

import unittest

from research.benchmark.review_fixes import bind_v2_cells as B
from research.benchmark.review_fixes import v2_family as F
from research.benchmark.review_fixes import v2_panel as P
from research.benchmark.review_fixes.test_v2_panel import CONTRACT, ledger, raw

DRAWS = 200
FAMILY_NAMES = ("feynman_I_6_2", "strogatz_bacres1", "fri_c0_100_5", "bng_echomonths", "529_pollen")


def names_for(cells, pattern=FAMILY_NAMES):
    datasets = sorted({cell["dataset"] for cell in cells})
    return {dataset: f"{pattern[index % len(pattern)]}_{index}" for index, dataset in enumerate(datasets)}


class FamilyTests(unittest.TestCase):
    def setUp(self):
        self.cells = ledger(n_lineages=20)
        self.names = names_for(self.cells)
        self.family = F.reduce(self.cells, self.names, draws=DRAWS, jobs=1, predeclare=b"{}")

    def test_scopes_partition_the_lineages_without_a_remainder_family(self):
        levels = [row for row in self.family["levels"] if row["size"] == 4 and row["auditor"] == "catboost"]
        self.assertEqual([row["scope"] for row in levels], list(F.SCOPES))
        self.assertNotIn("other", self.family["scopes"])
        self.assertEqual({row["scope"]: row["n"] for row in levels},
                         {"excluding_simulated": 4, "feynman": 4, "strogatz": 4, "fri": 4, "bng": 4})
        self.assertEqual(sum(row["n"] for row in levels), 20)

    def test_excluding_simulated_keeps_only_unprefixed_lineages(self):
        values = P.fit_values(self.cells, self.names)
        kept = F.scoped(values[(P.HEADLINE, 4, "catboost")], "excluding_simulated")
        self.assertTrue(kept)
        self.assertTrue(all(self.names[item["dataset"]].startswith("529_pollen") for item in kept))
        self.assertTrue(F.in_scope("feynman", "feynman"))
        self.assertFalse(F.in_scope("feynman", "fri"))
        self.assertFalse(F.in_scope("excluding_simulated", "bng"))
        with self.assertRaises(ValueError):
            F.in_scope("other", "other")

    def test_rows_are_a_sensitivity_and_fit_seeds_stay_nested(self):
        contrasts = self.family["contrasts"]
        self.assertTrue(contrasts)
        self.assertTrue(all(row["holm_family"] is None and "holm_p" not in row for row in contrasts))
        self.assertEqual({(row["method"], row["configuration"]) for row in contrasts},
                         {("TabSyn", "scaled_200_vae_1000_diffusion")})
        level = next(row for row in self.family["levels"]
                     if (row["scope"], row["size"], row["auditor"]) == ("feynman", 4, "catboost"))
        self.assertEqual((level["n"], level["n_fits"]), (4, 12))
        self.assertIs(self.family["official_tests_opened"], False)

    def test_below_six_pairs_is_not_identified(self):
        for row in self.family["contrasts"]:
            self.assertEqual(row["ci_status"], "not_identified_below_minimum_pairs")
            self.assertIsNone(row["wilcoxon_p"])
        wide = F.reduce(self.cells, names_for(self.cells, ("529_pollen",)), draws=DRAWS, jobs=1, predeclare=b"{}")
        row = next(row for row in wide["contrasts"] if row["size"] == 4 and row["auditor"] == "catboost")
        self.assertEqual(row["n"], 20)
        self.assertNotIn("ci_status", row)

    def test_values_do_not_depend_on_workers(self):
        again = F.reduce(self.cells, self.names, draws=DRAWS, jobs=2, predeclare=b"{}")
        self.assertEqual(P.canonical(again), P.canonical(self.family))

    def test_controls_and_undeclared_comparators_are_never_compared(self):
        cells = list(self.cells)
        dataset = sorted({cell["dataset"] for cell in cells})[0]
        for seed in (101, 211, 307):
            for size in (1, 4):
                cells.append(B.public_cell(raw("predictor_only", "control", dataset, 11, seed, size, 0.5),
                                           CONTRACT, "t"))
        family = F.reduce(cells, self.names, draws=DRAWS, jobs=1, predeclare=b"{}")
        self.assertNotIn("predictor_only", {row["method"] for row in family["contrasts"]})


if __name__ == "__main__":
    unittest.main()
