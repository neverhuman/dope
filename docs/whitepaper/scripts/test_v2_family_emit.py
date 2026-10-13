"""The v2 family emitter on a synthetic family file, and its committed outputs."""

import json
import re
import unittest
from pathlib import Path

import review_v2_family as F

HERE = Path(__file__).resolve().parent


def contrast(scope, method, configuration, hl, lo, hi, n=10, verdict="inseparable", **extra):
    return {"scope": scope, "method": method, "configuration": configuration, "size": 4, "auditor": "catboost",
            "n": n, "hl": hl, "hl_lo": lo, "hl_hi": hi, "verdict": verdict, "wins": 5, "ties": 0, "losses": 5,
            **extra}


FAMILY = {
    "scopes": ["excluding_simulated", "feynman"],
    "levels": [{"scope": scope, "method": "DOPE", "configuration": "headline", "size": 4, "auditor": "catboost",
                "n": 10, "median": 0.9, "lo": 0.8, "hi": 0.95} for scope in ("excluding_simulated", "feynman")],
    "contrasts": [
        contrast("excluding_simulated", "TabSyn", "scaled_200_vae_1000_diffusion", -0.03, -0.09, 0.02),
        contrast("excluding_simulated", "ARF", "author_default", 0.02, 0.01, 0.05, verdict="above"),
        contrast("feynman", "ARF", "author_default", 0.04, 0.0, 0.08, n=4,
                 ci_status="not_identified_below_minimum_pairs"),
    ],
}


class FamilyEmitTests(unittest.TestCase):
    def test_strongest_macros_follow_the_computed_strongest(self):
        text = F.macros(FAMILY, {"method": "ARF", "configuration": "author_default"})
        self.assertIn("\\newcommand{\\VExclStrongDiff}{0.020}", text)
        self.assertIn("\\newcommand{\\VExclStrongVerdict}{is above}", text)
        self.assertIn("\\newcommand{\\VExclTabSynCbFourVerdict}{cannot be separated from}", text)
        names = re.findall(r"\\newcommand\{\\([^}]*)\}", text)
        self.assertTrue(names and all(name.isalpha() for name in names))
        self.assertEqual(len(names), len(set(names)))

    def test_missing_strongest_prints_dashes(self):
        text = F.macros(FAMILY, None)
        self.assertIn("\\newcommand{\\VExclStrongDiff}{" + F.E.DASH + "}", text)

    def test_table_marks_rows_below_the_minimum_and_has_no_holm_column(self):
        table = F.table(FAMILY)
        self.assertIn("\\label{tab:v2-family}", table)
        self.assertIn("not identified", table)
        self.assertNotIn("Holm $p$", table)
        self.assertIn("outside every Holm family", table)

    def test_committed_outputs_match_the_committed_family_file(self):
        family = json.loads(F.FAMILY.read_text())
        panel = json.loads(F.E.PANEL.read_text())
        for name, text in F.outputs(family, panel).items():
            self.assertEqual((HERE.parent / "generated" / name).read_text(), text, name)


if __name__ == "__main__":
    unittest.main()
