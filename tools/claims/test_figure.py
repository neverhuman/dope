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

    def test_svg_internal_names_are_portable_but_geometry_remains_authenticated(self):
        from tools.claims.figure_drawing import canonical_svg
        original = (b'<svg><defs><clipPath id="p123"><rect width="12.000001"/></clipPath>'
                    b'<path id="m123" d="M 1 2 L 3 4"/></defs><g clip-path="url(#p123)">'
                    b'<use xlink:href="#m123"/></g></svg>')
        renamed = original.replace(b"p123", b"p456").replace(b"m123", b"m456")
        self.assertEqual(canonical_svg(original), canonical_svg(renamed))
        changed_geometry = renamed.replace(b"12.000001", b"12.000002")
        self.assertNotEqual(canonical_svg(original), canonical_svg(changed_geometry))
        with self.assertRaisesRegex(ValueError, "duplicate identifier"):
            canonical_svg(b'<svg><g id="a"/><g id="a"/></svg>')
        with self.assertRaisesRegex(ValueError, "unresolved internal reference"):
            canonical_svg(b'<svg><use xlink:href="#absent"/></svg>')

    def test_png_encoding_is_stable_without_losing_a_single_rgba_bit(self):
        from PIL import Image
        from tools.claims.figure_drawing import canonical_png
        image = Image.new("RGBA", (2, 2))
        image.putdata([(0, 0, 0, 0), (255, 2, 3, 255),
                       (4, 5, 6, 128), (255, 255, 255, 255)])
        def encoded(level):
            stream = io.BytesIO()
            image.save(stream, format="PNG", compress_level=level)
            return stream.getvalue()
        first = canonical_png(encoded(0))
        self.assertEqual(first, canonical_png(encoded(9)))
        with Image.open(io.BytesIO(first)) as decoded:
            self.assertEqual(decoded.size, image.size)
            self.assertEqual(decoded.tobytes(), image.tobytes())
        image.putpixel((1, 0), (254, 2, 3, 255))
        self.assertNotEqual(first, canonical_png(encoded(9)))

    def test_committed_points_and_threshold_fit_the_axes(self):
        self.assertEqual(render.figure_bound_failures(self.rows, self.data["thresholds"]), [])
        self.assertEqual(render.threshold_label(self.data["thresholds"]), "Proposed T90; driver gate unmeasured")

    def test_threshold_label_and_hero_visual_qa_contract(self):
        moved = dict(self.data["thresholds"], transfer=0.95)
        self.assertEqual(render.threshold_label(moved), "Proposed T95; driver gate unmeasured")
        svg = render.figure_outputs(self.csv, moved)["paper/generated/hero-frontier.svg"].decode()
        self.assertIn("Preserve predictive lift. Charge every byte.", svg)
        for text in ("DOPE default", "DOPE research", "TabSyn", "ARF default", "ARF selected",
                     "Copula", "Chow-Liu", "Independent", "separate 97-lineage cohort", "TDL and cold closure unmeasured"):
            self.assertIn(text, svg)
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


    def test_interval_level_is_read_from_the_predeclaration_and_refuses_label_only_changes(self):
        from tools.claims import foundation
        predeclare = json.loads((ROOT / "research/benchmark/review_fixes/predeclare_v2.json").read_text())
        self.assertIn(f"percentile {render.interval_level()}% interval", predeclare["estimands"]["uncertainty"])
        status = json.loads((ROOT / "evidence/status.json").read_text())
        for row in self.data["results"]:
            if row.get("retention"):
                meta = render.record_interval_metadata(row)
                self.assertEqual(meta["level_pct"], 95)
                self.assertEqual(meta["quantiles"], [0.025, 0.975])
                self.assertEqual(meta["estimand"], "lineage_median_null_normalized_retention")
        with self.assertRaisesRegex(ValueError, "interval_level_mismatch"):
            render.text_outputs(self.data, status, 90)
        with self.assertRaisesRegex(ValueError, "interval_level_mismatch"):
            render.figure_outputs(self.csv, self.data["thresholds"], 90)

        f95 = foundation.load_foundation_summary(ROOT, interval_level_pct=95)
        f90 = foundation.load_foundation_summary(ROOT, interval_level_pct=90)
        self.assertEqual(f90["interval_metadata"]["level_pct"], 90)
        self.assertEqual(f90["interval_metadata"]["quantiles"], [0.05, 0.95])
        self.assertEqual(f90["interval_metadata"]["method"], "nonparametric_lineage_percentile_bootstrap")
        self.assertEqual(f90["interval_metadata"]["resampling_unit"], "lineage")
        self.assertEqual(f90["interval_metadata"]["seed"], 20261005)
        self.assertEqual(f90["interval_metadata"]["cohort_id"], "mfs-v3-foundation-97")
        for aud_id in ("kumo_tabular_l", "tabicl2", "mitra_v2"):
            m95 = f95["auditors"][aud_id]["methods"]["DOPE"]
            m90 = f90["auditors"][aud_id]["methods"]["DOPE"]
            self.assertEqual(m90["median"], m95["median"])
            self.assertGreater(m90["lo"], m95["lo"])
            self.assertLess(m90["hi"], m95["hi"])

        data90 = json.loads(json.dumps(self.data))
        for row in data90["results"]:
            if row.get("retention"):
                row["retention"]["interval_metadata"] = {
                    "level_pct": 90,
                    "quantiles": [0.05, 0.95],
                    "method": "nested_cluster_lineage_fit_seed_bootstrap",
                    "estimand": "lineage_median_null_normalized_retention",
                    "resampling_unit": "cluster_lineage_fit_seed",
                    "cohort_id": row["cohort"]["id"],
                }
        out90 = render.text_outputs(data90, status, 90, foundation_summary=f90)
        card = out90["paper/generated/readme-result-card.md"].decode()
        self.assertIn("[historical 90% interval]", card)
        kumo90 = f90["auditors"]["kumo_tabular_l"]["methods"]["DOPE"]
        self.assertIn(f"[{100.0 * kumo90['lo']:.1f}%, {100.0 * kumo90['hi']:.1f}%]", card)
        svg = render.figure_outputs(
            out90["paper/generated/hero-frontier.csv"], data90["thresholds"], 90, foundation_summary=f90
        )["paper/generated/hero-frontier.svg"].decode()
        self.assertIn("historical 90% interval", svg)
        self.assertNotIn("95%", (ROOT / "tools/claims/render.py").read_text(), "renderer types the interval level")

    def test_compression_panel_keeps_defaults_and_refuses_mixed_cohorts(self):
        source = (ROOT / "paper/generated/legacy-v2-catboost.csv").read_bytes()
        rows = list(csv.DictReader(io.StringIO(source.decode())))
        selected = render.compression_rows(source)
        self.assertEqual({row["id"] for row in selected},
                         {"Dope", "DopeDef", "TabSyn", "Arf", "ArfNat", "Gauss", "Chow", "Ind"})
        for mutation in ("cohort", "duplicate", "missing", "nonfinite"):
            with self.subTest(mutation=mutation):
                bad = [dict(row) for row in rows]
                if mutation == "cohort":
                    bad[0]["cohort_id"] = "legacy-v2-catboost-6"
                elif mutation == "duplicate":
                    bad.append(dict(bad[0]))
                elif mutation == "missing":
                    bad = [row for row in bad if row["id"] != "DopeDef"]
                else:
                    bad[0]["retention"] = "nan"
                with self.assertRaises(ValueError):
                    render.compression_rows(rows_to_csv(bad))


if __name__ == "__main__":
    unittest.main()
