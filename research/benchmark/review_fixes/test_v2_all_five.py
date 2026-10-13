"""v2 all-five-fits sensitivity on synthetic fixtures. No benchmark data is read."""

import unittest

from research.benchmark.review_fixes import bind_v2_cells as B
from research.benchmark.review_fixes import v2_all_five as F
from research.benchmark.review_fixes import v2_panel as P
from research.benchmark.review_fixes.test_v2_panel import CONTRACT, raw

DRAWS = 200
TABSYN = ("TabSyn", "scaled_200_vae_1000_diffusion")


def ledger(n_lineages=10):
    """DOPE has all five fits everywhere; TabSyn misses seed 71 on the first three lineages."""
    cells = []
    for index in range(n_lineages):
        dataset = f"{index:016x}"
        for arm, seeds in ((("DOPE", "headline"), F.FIT_SEEDS),
                           (TABSYN, F.FIT_SEEDS[:4] if index < 3 else F.FIT_SEEDS)):
            for fit_seed in seeds:
                for sample_seed in (101, 211, 307):
                    for size in (1, 4):
                        cells.append(B.public_cell(raw(arm[0], arm[1], dataset, fit_seed, sample_seed, size,
                                                       0.9 if arm[0] == "DOPE" else 0.8), CONTRACT, "t"))
        for sample_seed in (101, 211, 307):
            for size in (1, 4):
                cells.append(B.public_cell(raw("GaussianCopula", "native_selected", dataset, None, sample_seed,
                                               size, 0.5), CONTRACT, "t"))
    return cells


class AllFiveTests(unittest.TestCase):
    def setUp(self):
        self.result = F.reduce(ledger(), {}, draws=DRAWS, jobs=1, predeclare=b"{}")

    def rows(self, section, method):
        return [row for row in self.result[section]
                if row["method"] == method and row["size"] == 4 and row["auditor"] == "catboost"]

    def test_only_lineages_with_all_five_fits_enter(self):
        values = P.fit_values(ledger(), {})
        kept = F.all_five(values[(TABSYN, 4, "catboost")])
        self.assertEqual(len({item["dataset"] for item in kept}), 7)
        self.assertEqual(len(kept), 35)
        self.assertEqual((self.rows("levels", "DOPE")[0]["n"], self.rows("levels", "DOPE")[0]["n_fits"]), (10, 50))
        self.assertEqual((self.rows("levels", "TabSyn")[0]["n"], self.rows("levels", "TabSyn")[0]["n_fits"]), (7, 35))

    def test_contrasts_pair_lineages_where_both_arms_have_all_five(self):
        row = self.rows("contrasts", "TabSyn")[0]
        self.assertEqual(row["n"], 7)
        self.assertEqual(row["verdict"], "above")
        self.assertIsNone(row["holm_family"])
        self.assertNotIn("holm_p", row)

    def test_one_seed_arms_are_never_rows(self):
        methods = {row["method"] for section in ("levels", "contrasts") for row in self.result[section]}
        self.assertEqual(methods, {"DOPE", "TabSyn"})
        self.assertEqual(F.arms_of(P.fit_values(ledger(), {})), sorted([("DOPE", "headline"), TABSYN]))

    def test_values_do_not_depend_on_workers(self):
        again = F.reduce(ledger(), {}, draws=DRAWS, jobs=2, predeclare=b"{}")
        self.assertEqual(P.canonical(again), P.canonical(self.result))
        self.assertIs(self.result["official_tests_opened"], False)
        self.assertEqual(self.result["fit_seeds"], [11, 23, 37, 53, 71])


if __name__ == "__main__":
    unittest.main()
