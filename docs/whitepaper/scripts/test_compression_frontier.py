"""The compression frontier and quickstart blocks are rendered, never typed.

Fast checks on committed inputs; `compression_frontier.py --check` (run in
ops/ci/cpu.sh) recomputes the statistics from the pinned ledgers.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))

import compression_frontier_render as R  # noqa: E402
import quickstart_block as Q  # noqa: E402
import readme_blocks as B  # noqa: E402


class CompressionFrontierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = json.loads((REPO / R.DATA_JSON).read_text())
        cls.readme = (REPO / "README.md").read_text()

    def test_committed_text_outputs_equal_their_rendering(self):
        for name, text in R.text_outputs(self.data).items():
            with self.subTest(name=name):
                self.assertEqual((REPO / name).read_text(), text)

    def test_figures_exist(self):
        for name in (R.FIGURE_PDF, R.FIGURE_PNG):
            self.assertTrue((REPO / name).is_file(), name)

    def test_evidence_label_and_shared_denominator(self):
        self.assertEqual(self.data["evidence_status"], "selection_exposed_validation")
        self.assertFalse(self.data["official_tests_opened"])
        full = [r for r in self.data["rows"] if r["role"] == "generator" and r["lineages"] >= R.FULL_PANEL_MIN]
        self.assertEqual({r["lineages"] for r in full}, {self.data["lineages_all_auditors_informative"]})

    def test_frontier_and_ranking_follow_from_the_rows(self):
        rows = [r for r in self.data["rows"] if r["role"] == "generator" and r["lift_per_kib"]]
        ranked = sorted(rows, key=lambda r: -r["lift_per_kib"])
        self.assertEqual(self.data["lift_ranking"], [r["display"] for r in ranked])
        frontier = [r for r in self.data["rows"] if r["display"] in self.data["frontier"]]
        sizes = [r["bytes"]["median"] for r in frontier]
        levels = [r["worst_median"] for r in frontier]
        self.assertEqual(sizes, sorted(sizes))
        self.assertEqual(levels, sorted(levels))

    def test_readme_carries_both_generated_blocks(self):
        self.assertEqual(B.block_failures(self.readme, REPO), [])
        self.assertEqual(B.block_failures(self.readme.replace("22,301", "22,302"), REPO),
                         ["README: generated compression block drift"])

    def test_generated_tex_is_authenticated(self):
        verified, failures = B.verified_tex(REPO)
        self.assertEqual(failures, [])
        self.assertEqual(verified, {"paper/generated/compression-macros.tex",
                                    "paper/generated/compression-table.tex", Q.MACROS, Q.ABLATION_TEX})

    def test_quickstart_block_matches_the_committed_run(self):
        report = Q.load_report()
        self.assertEqual(report["evidence_status"], "local_engineering_demo")
        self.assertEqual(Q.failures(self.readme), [])
        self.assertEqual((REPO / Q.MACROS).read_text(), Q.tex_macros(report, Q.load_ablation()))

    def test_stripping_leaves_no_generated_marker(self):
        stripped = B.strip_blocks(self.readme)
        for marker in (R.README_BEGIN, R.README_END, Q.BEGIN, Q.END):
            self.assertNotIn(marker, stripped)


if __name__ == "__main__":
    unittest.main()
