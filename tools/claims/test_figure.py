"""The hero figure reads its threshold from the results file and refuses values outside its axes."""
import csv
import io
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.claims import render  # noqa: E402


def rows_to_csv(rows: list[dict]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode()


class HeroFigureTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "paper/generated/headline-results.json").read_text())
        self.csv = (ROOT / "paper/generated/hero-frontier.csv").read_bytes()
        self.rows = list(csv.DictReader(io.StringIO(self.csv.decode())))

    def test_committed_points_and_threshold_fit_the_axes(self):
        self.assertEqual(render.figure_bound_failures(self.rows, self.data["thresholds"]), [])
        self.assertEqual(render.threshold_label(self.data["thresholds"]), "Proposed T90; driver gate unmeasured")

    def test_threshold_label_and_hero_visual_qa_contract(self):
        moved = dict(self.data["thresholds"], transfer=0.95)
        self.assertEqual(render.threshold_label(moved), "Proposed T95; driver gate unmeasured")
        svg = render.figure_outputs(self.csv, moved)["paper/generated/hero-frontier.svg"].decode()
        self.assertIn("More predictive gain. About one-third the state.", svg)
        self.assertIn("DOPE (1,844 B)", svg)
        self.assertIn("Gaussian copula (5,464 B)", svg)
        for pct in ("86.3%", "74.7%", "87.1%", "75.4%", "85.8%", "71.3%"):
            self.assertIn(pct, svg)
        self.assertNotIn("Proposed T90", svg)
        self.assertNotIn("Proposed T95", svg)
        self.assertIn("Cross-model feature-driver preservation: not yet measured", svg)

    def test_hero_csv_matches_six_plotted_foundation_rows_and_legacy_catboost_is_separate(self):
        from tools.claims import foundation
        f_summary = foundation.load_foundation_summary(ROOT)
        six_bars = foundation.hero_six_bar_rows(f_summary)
        self.assertEqual(len(self.rows), 6)
        self.assertEqual(render.hero_csv_agreement_failures(self.rows, six_bars), [])
        self.assertEqual([int(r["artifact_bytes"]) for r in self.rows], [1844, 5464, 1844, 5464, 1844, 5464])
        legacy_rows = list(csv.DictReader(io.StringIO((ROOT / "paper/generated/legacy-v2-catboost.csv").read_text())))
        self.assertEqual(len(legacy_rows), 11)
        self.assertEqual(legacy_rows[0]["cohort_id"], "legacy-v2-catboost-99")
        tampered = [dict(r) for r in self.rows]
        tampered[0]["artifact_bytes"] = "1895"
        with self.assertRaisesRegex(ValueError, "CSV disagreement"):
            render.figure_outputs(rows_to_csv(tampered), self.data["thresholds"])

    def test_point_outside_either_axis_is_refused(self):
        for key, value in (("artifact_bytes", "5e9"), ("artifact_bytes", "500"), ("retention", "-0.6"),
                           ("interval_hi", "1.3"), ("interval_lo", "-0.5"), ("retention", "nan")):
            with self.subTest(key=key, value=value):
                rows = [dict(row) for row in self.rows]
                rows[0][key] = value
                with self.assertRaisesRegex(ValueError, "out-of-axis"):
                    render.figure_outputs(rows_to_csv(rows), self.data["thresholds"])

    def test_retention_outside_its_interval_is_refused(self):
        rows = [dict(row) for row in self.rows]
        rows[0]["interval_hi"] = str(float(rows[0]["retention"]) - 0.01)
        self.assertTrue(any("outside its own interval" in f
                            for f in render.figure_bound_failures(rows, self.data["thresholds"])))

    def test_threshold_outside_the_axes_is_refused(self):
        for transfer in (1.2, -1.0, float("nan"), True):
            with self.subTest(transfer=transfer):
                thresholds = dict(self.data["thresholds"], transfer=transfer)
                self.assertTrue(render.figure_bound_failures(self.rows, thresholds))


    def test_interval_level_is_read_from_the_predeclaration(self):
        predeclare = json.loads((ROOT / "research/benchmark/review_fixes/predeclare_v2.json").read_text())
        self.assertIn(f"percentile {render.interval_level()}% interval", predeclare["estimands"]["uncertainty"])
        status = json.loads((ROOT / "evidence/status.json").read_text())
        card = render.text_outputs(self.data, status, 90)["paper/generated/readme-result-card.md"].decode()
        self.assertIn("[historical 90% interval]", card)
        svg = render.figure_outputs(self.csv, self.data["thresholds"], 90)["paper/generated/hero-frontier.svg"].decode()
        self.assertIn("historical 90% interval", svg)
        self.assertNotIn("95%", (ROOT / "tools/claims/render.py").read_text(), "renderer types the interval level")


if __name__ == "__main__":
    unittest.main()
