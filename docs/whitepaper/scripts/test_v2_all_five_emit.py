"""The v2 all-five emitter on a synthetic file, and its committed outputs."""

import json
import re
import unittest
from pathlib import Path

import review_v2_all_five as F

HERE = Path(__file__).resolve().parent


def row(method, configuration, **extra):
    return {"method": method, "configuration": configuration, "size": 4, "auditor": "catboost", **extra}


DATA = {
    "levels": [row("DOPE", "headline", n=9, n_fits=45, median=0.9, lo=0.8, hi=0.95),
               row("TabSyn", "scaled_200_vae_1000_diffusion", n=8, n_fits=40, median=0.88, lo=0.7, hi=0.9)],
    "contrasts": [row("TabSyn", "scaled_200_vae_1000_diffusion", n=8, hl=0.01, hl_lo=-0.02, hl_hi=0.05,
                      verdict="inseparable", wins=4, ties=0, losses=4),
                  row("ARF", "author_default", n=5, hl=0.1, hl_lo=0.0, hl_hi=0.2, verdict="above", wins=5, ties=0,
                      losses=0, ci_status="not_identified_below_minimum_pairs")],
}


class AllFiveEmitTests(unittest.TestCase):
    def test_strongest_macros_follow_the_computed_strongest(self):
        text = F.macros(DATA, {"method": "TabSyn", "configuration": "scaled_200_vae_1000_diffusion"})
        self.assertIn("\\newcommand{\\VAllFiveStrongN}{8}", text)
        self.assertIn("\\newcommand{\\VAllFiveStrongVerdict}{cannot be separated from}", text)
        self.assertIn("\\newcommand{\\VAllFiveDopeCbFourN}{9}", text)
        names = re.findall(r"\\newcommand\{\\([^}]*)\}", text)
        self.assertTrue(names and all(name.isalpha() for name in names) and len(names) == len(set(names)))

    def test_a_strongest_outside_the_five_seed_arms_prints_dashes(self):
        text = F.macros(DATA, {"method": "GaussianCopula", "configuration": "native_selected"})
        self.assertIn("\\newcommand{\\VAllFiveStrongDiff}{" + F.E.DASH + "}", text)
        self.assertIn("\\newcommand{\\VAllFiveDopeCbFour}{0.900}", text)

    def test_table_marks_rows_below_the_minimum_and_has_no_holm_column(self):
        table = F.table(DATA)
        self.assertIn("\\label{tab:v2-all-five}", table)
        self.assertIn("not identified", table)
        self.assertNotIn("Holm $p$", table)
        self.assertIn("outside every Holm family", table)

    def test_committed_outputs_match_the_committed_file(self):
        data = json.loads(F.ALL_FIVE.read_text())
        panel = json.loads(F.E.PANEL.read_text())
        for name, text in F.outputs(data, panel).items():
            self.assertEqual((HERE.parent / "generated" / name).read_text(), text, name)


if __name__ == "__main__":
    unittest.main()
