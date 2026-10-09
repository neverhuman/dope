"""The cost macros must not rebuild the repeated 'not measured and not measured' sentence."""

from __future__ import annotations

import unittest
from pathlib import Path

import paper_emit

MANUSCRIPT = Path(__file__).resolve().parents[1] / "dope-mfs.tex"


class PaperEmitLintTest(unittest.TestCase):
    def test_cost_phrases_differ(self):
        self.assertNotEqual(paper_emit.TABSYN_COST, paper_emit.TABDDPM_COST)
        self.assertIn("not measured", paper_emit.TABSYN_COST)
        self.assertIn("not measured", paper_emit.TABDDPM_COST)
        self.assertNotIn("not measured and not measured", paper_emit.TABSYN_COST)
        self.assertNotIn("not measured and not measured", paper_emit.TABDDPM_COST)

    def test_identical_adjacent_macros_fail(self):
        lines = [
            r"\newcommand{\TabSynPanel}{not measured}",
            r"\newcommand{\TabDDPMPanel}{not measured}",
        ]
        manuscript = r"TabSyn and TabDDPM costs are \TabSynPanel\ and \TabDDPMPanel."
        with self.assertRaises(ValueError):
            paper_emit.assert_no_adjacent_repeat(lines, manuscript)

    def test_repeated_phrase_inside_one_macro_fails(self):
        lines = [r"\newcommand{\TabSynPanel}{not measured and not measured}"]
        with self.assertRaises(ValueError):
            paper_emit.assert_no_adjacent_repeat(lines, "")

    def test_distinct_phrases_pass(self):
        lines = [
            rf"\newcommand{{\TabSynPanel}}{{{paper_emit.TABSYN_COST}}}",
            rf"\newcommand{{\TabDDPMPanel}}{{{paper_emit.TABDDPM_COST}}}",
        ]
        manuscript = "TabSyn cost is \\TabSynPanel. TabDDPM cost is \\TabDDPMPanel."
        paper_emit.assert_no_adjacent_repeat(lines, manuscript)

    def test_arf_macros_follow_the_review_panel(self):
        rows = paper_emit.arf_author_default_rows()
        cat = rows["catboost"]
        linear = rows["linear"]
        self.assertLess(cat["lo"], cat["median"])
        self.assertLess(cat["median"], cat["hi"])
        self.assertLess(linear["lo"], 0)
        self.assertGreater(linear["hi"], 0)
        lines = paper_emit.arf_paired_commands(lambda name, body: f"{name}={body}")
        self.assertIn(f"DiffArfCb={float(cat['median']):.4f}", lines)
        self.assertIn(f"LoDiffArfLin={float(linear['lo']):.4f}", lines)
        self.assertIn(f"NDiffArfCb={int(cat['n'])}", lines)
        self.assertNotIn(" not ", " ".join(lines))

    def test_committed_manuscript_has_no_repeated_phrase(self):
        text = MANUSCRIPT.read_text()
        self.assertNotIn("not measured and not measured", text)
        self.assertNotIn(r"\TabSynPanel\ and \TabDDPMPanel", text)
        paper_emit.assert_no_adjacent_repeat(
            [
                rf"\newcommand{{\TabSynPanel}}{{{paper_emit.TABSYN_COST}}}",
                rf"\newcommand{{\TabDDPMPanel}}{{{paper_emit.TABDDPM_COST}}}",
            ],
            text,
        )


if __name__ == "__main__":
    unittest.main()
