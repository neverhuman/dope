"""The denominator table prints the level n beside the complete-loss path."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from research.benchmark.review_fixes import denominator_view as view
from research.benchmark.review_fixes import receipt_panel as panel
from research.benchmark.review_fixes.review_tex import longtable
from research.benchmark.review_fixes.source_pins import PUBLISHED_SOURCE_SHA256

PANEL_NAME = "review-fixes-receipts-v1/panel.json"
FOREST = "ForestDiffusion/Forest-Flow"


def _retention(method, size, n, auditor="catboost"):
    return {"method": method, "configuration": "native_selected", "size": size, "auditor": auditor, "n": n}


def _denominator(method, size, informative, noninformative, undefined, incomplete, auditor="catboost"):
    return {
        "method": method, "configuration": "native_selected", "size": size, "auditor": auditor,
        "n_informative": informative, "n_noninformative": noninformative, "n_undefined": undefined,
        "n_capped": 0, "n_incomplete": incomplete,
        "sensitivity_n": informative + noninformative,
        "sensitivity_median": 0.5 if informative + noninformative else None,
    }


def _fixture():
    return {
        "retention": [_retention(FOREST, 4, 6), _retention("ARF", 4, 99), _retention(FOREST, 4, 5, "linear")],
        "denominator": [
            _denominator(FOREST, 4, 0, 0, 0, 6),
            _denominator("ARF", 4, 99, 1, 0, 0),
            _denominator(FOREST, 4, 0, 0, 0, 6, "linear"),
        ],
    }


class DenominatorViewTests(unittest.TestCase):
    def test_level_n_replaces_the_empty_loss_path(self):
        rows = {row["method"]: row for row in view.denominator_rows(_fixture())}
        self.assertEqual(set(rows), {FOREST, "ARF"})
        self.assertEqual((rows[FOREST]["informative"], rows[FOREST]["complete_loss_rows"]), (6, 0))
        self.assertEqual((rows["ARF"]["informative"], rows["ARF"]["complete_loss_rows"]), (99, 100))
        linear = view.denominator_rows(_fixture(), "linear")
        self.assertEqual([row["informative"] for row in linear], [5])
        lines = view.denominator_lines(_fixture(), panel.fmt, panel.tex_name)
        self.assertEqual(lines[0], "ForestDiffusion/Forest-Flow & native\\_selected & 4$n$ & 6 & 0 & 0 & 0 & 0 & 0 & --- \\\\")
        table = longtable("Denominators. No Holm family.", view.HEADER, lines, view.ALIGN)
        self.assertIn("Informative & \\shortstack[r]{Complete-\\\\loss rows} & \\shortstack[r]{Non-\\\\informative}", table)
        self.assertEqual(panel._nfields(view.HEADER), len(view.ALIGN))

    def test_inconsistent_panels_are_refused(self):
        missing = _fixture()
        missing["retention"].pop(0)
        with self.assertRaisesRegex(ValueError, "no retention row"):
            view.denominator_rows(missing)
        larger = _fixture()
        larger["denominator"][1]["n_informative"] = 100
        larger["denominator"][1]["sensitivity_n"] = 101
        with self.assertRaisesRegex(ValueError, "exceeds the level count"):
            view.denominator_rows(larger)
        drifted = _fixture()
        drifted["denominator"][1]["sensitivity_n"] = 7
        with self.assertRaisesRegex(ValueError, "sensitivity n"):
            view.denominator_rows(drifted)
        duplicate = _fixture()
        duplicate["retention"].append(dict(duplicate["retention"][0]))
        with self.assertRaisesRegex(ValueError, "duplicate retention row"):
            view.denominator_rows(duplicate)

    def test_committed_panel_renders_forest_flow_informative_counts(self):
        path = panel.RESULTS / PANEL_NAME
        raw = path.read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), PUBLISHED_SOURCE_SHA256[PANEL_NAME])
        payload = json.loads(raw)
        levels = view.level_counts(payload["retention"])
        rows = view.denominator_rows(payload)
        self.assertEqual(len(rows), 18)
        for row in rows:
            self.assertEqual(row["informative"], levels[(row["method"], row["configuration"], row["size"], "catboost")])
        forest = [row for row in rows if row["method"] == FOREST]
        self.assertEqual([(row["size"], row["informative"], row["complete_loss_rows"]) for row in forest],
                         [(1, 6, 0), (4, 6, 0)])
        # Where every lineage has complete losses, both paths agree.
        for row in rows:
            if row["n_incomplete"] == 0:
                self.assertEqual(row["informative"], row["n_informative"])

    def test_emit_matches_the_committed_tables(self):
        generated = panel.REPO / "docs" / "whitepaper" / "generated"
        target = panel.REPO / "target"
        target.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=target) as directory:
            out = Path(directory)
            with patch.object(panel, "GENERATED", out), patch.object(panel, "_figure", lambda points: None):
                panel.emit_tex(json.loads((panel.RESULTS / PANEL_NAME).read_text()))
            denominator = (out / "review-denominator.tex").read_text()
            self.assertIn("ForestDiffusion/Forest-Flow & native\\_selected & 4$n$ & 6 & 0 &", denominator)
            self.assertIn("DOPE & features12\\_steps2048 & 4$n$ & 97 & 98 &", denominator)
            for written in sorted(out.glob("review-*.tex")):
                if written.name != "review-denominator.tex":
                    self.assertEqual(written.read_bytes(), (generated / written.name).read_bytes(), written.name)


if __name__ == "__main__":
    unittest.main()
