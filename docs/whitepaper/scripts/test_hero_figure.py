"""Page-1 hero readability floor (ledger C053) and the rows it may leave to Table 1."""
import copy
import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import figure_style
import v2_figures

PAPER = Path(__file__).resolve().parents[1]


class HeroFigureTests(unittest.TestCase):
    def setUp(self):
        self.headline = json.loads(v2_figures.HEADLINE.read_text())

    def draw(self, headline, drawer=None):
        captured = {}
        original = v2_figures._save

        def save(figure, path, dpi):
            figure.canvas.draw()
            # The tight box covers only what is drawn, so out-of-view tick labels do not count.
            tight = figure.get_tightbbox(figure.canvas.get_renderer())
            width, height = figure.get_size_inches()
            captured["size"] = (width, height)
            captured["texts"] = v2_figures.text_sizes(figure)
            captured["outside"] = [edge for edge, cut in (("left", tight.x0 < 0), ("bottom", tight.y0 < 0),
                                                          ("right", tight.x1 > width), ("top", tight.y1 > height))
                                   if cut]
            # Point labels and control labels are the axes' own texts; their boxes must not touch.
            renderer = figure.canvas.get_renderer()
            boxes = [(label.get_text(), label.get_window_extent(renderer)) for axis in figure.axes
                     for label in axis.texts if label.get_visible() and label.get_text().strip()]
            captured["overlaps"] = [(a, b) for i, (a, box) in enumerate(boxes) for b, other in boxes[i + 1:]
                                    if box.overlaps(other)]
            # Tick labels drawn inside each axis's view must not run into each other either.
            ticks = []
            for axis in figure.axes:
                low, high = axis.get_xlim()
                drawn = [(label.get_text(), label.get_window_extent(renderer)) for tick, label in
                         zip(axis.get_xticks(), axis.get_xticklabels()) if min(low, high) <= tick <= max(low, high)]
                ticks += [(a, b) for i, (a, box) in enumerate(drawn) for b, other in drawn[i + 1:] if box.overlaps(other)]
            captured["tick_overlaps"] = ticks
            return original(figure, path, dpi)

        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(v2_figures, "_save", save):
            (drawer or v2_figures.draw_hero)(headline, Path(tmp) / "figure.pdf")
        return captured

    def test_every_hero_label_meets_the_floor_inside_the_canvas_at_text_width(self):
        drawn = self.draw(self.headline)
        self.assertAlmostEqual(drawn["size"][0], figure_style.TEASER_WIDTH_IN)
        small = [(text, size) for text, size in drawn["texts"] if size < v2_figures.HERO_MIN_POINTS]
        self.assertEqual(small, [])
        self.assertEqual(drawn["outside"], [])
        self.assertEqual(drawn["overlaps"], [])
        self.assertEqual(drawn["tick_overlaps"], [])
        self.assertGreaterEqual(len(drawn["texts"]), 18)

    def test_no_hero_value_label_is_needed_because_every_point_sits_inside_the_window(self):
        """A value printed at the window edge would cover part of its interval, so the window holds every point."""
        left, right = v2_figures.TEASER_X_LIM
        for arm in self.headline["arms"]:
            if arm["code"] in v2_figures.HERO_OMITTED_ROWS or arm not in v2_figures._rows(self.headline):
                continue
            for auditor, _ in v2_figures.AUDITORS:
                row = arm["contrasts"][auditor]
                if row is not None:
                    self.assertTrue(left <= row["hl"] <= right, (arm["code"], auditor, row["hl"]))

    def test_results_pareto_keeps_the_floor_without_overlapping_labels(self):
        """The paper's Pareto figure prints at the text width; the README PNG keeps its own drawing."""
        drawn = self.draw(self.headline, lambda headline, path: v2_figures.draw_pareto(
            headline, path, floor=v2_figures.HERO_MIN_POINTS))
        self.assertAlmostEqual(drawn["size"][0], figure_style.TEASER_WIDTH_IN)
        self.assertEqual([(text, size) for text, size in drawn["texts"] if size < v2_figures.HERO_MIN_POINTS], [])
        self.assertEqual(drawn["overlaps"], [])
        self.assertEqual(drawn["outside"], [])

    def test_manuscript_places_the_hero_at_its_drawn_width(self):
        """Scale 1: the text width equals the drawing width and the include adds no height cap."""
        tex = (PAPER / "dope-mfs.tex").read_text()
        self.assertIn("\\includegraphics[width=\\linewidth]{figures/teaser.pdf}", tex)
        style = (PAPER / "iclr2027_conference.sty").read_text()
        width = re.search(r"\\textwidth\s+([0-9.]+)\s*true in", style)
        self.assertIsNotNone(width)
        self.assertAlmostEqual(float(width.group(1)), figure_style.TEASER_WIDTH_IN)

    def test_a_row_may_leave_the_hero_only_while_every_interval_clears_the_window(self):
        for code in v2_figures.HERO_OMITTED_ROWS:
            changed = copy.deepcopy(self.headline)
            arm = next(arm for arm in changed["arms"] if arm["code"] == code)
            arm["contrasts"]["mlp"]["hl_lo"] = v2_figures.TEASER_X_LIM[1]
            with self.assertRaisesRegex(ValueError, "omitted from the hero"):
                self.draw(changed)


if __name__ == "__main__":
    unittest.main()
