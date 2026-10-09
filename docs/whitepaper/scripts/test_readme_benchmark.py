"""The README benchmark is the density table, and the PNG uses those bars."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import compute_panel
import readme_benchmark as bench

REPO = Path(__file__).resolve().parents[3]
TOY = REPO / "examples" / "readme-regression"


class ReadmeBenchmark(unittest.TestCase):
    def test_readme_block_matches_the_density_table(self):
        tex = bench.DENSITY.read_text()
        readme = bench.README.read_text()
        self.assertEqual(bench.readme_block(readme), bench.expected_readme_block(tex))
        self.assertIn("docs/whitepaper/dope-mfs.pdf", readme)
        self.assertIn("docs/readme/retention-bars.png", readme)
        self.assertIn("fit seed $11$", (REPO / "docs/whitepaper/dope-mfs.tex").read_text())

    def test_png_source_matches_density_table(self):
        tex = bench.DENSITY.read_text()
        stats = json.loads(bench.PANEL.read_text())
        pairs = stats["blocks"]["density"]["catboost"]["pairs"]
        labels = {
            "GaussianCopula": "Gaussian copula",
            "Chow-Liu": "Chow--Liu",
            "independent_marginals": "Indep.\\ marginals",
        }
        for key, label in labels.items():
            pair = pairs[key]
            dope = (
                f"{compute_panel.sig3(pair['dope_median'])} "
                f"[{compute_panel.sig3(pair['dope_lo'])}, {compute_panel.sig3(pair['dope_hi'])}]"
            )
            other = (
                f"{compute_panel.sig3(pair['other_median'])} "
                f"[{compute_panel.sig3(pair['other_lo'])}, {compute_panel.sig3(pair['other_hi'])}]"
            )
            self.assertIn(f"CatBoost & {label} & {pair['n']} & {dope} & {other}", tex)
        self.assertTrue(bench.PNG.is_file(), "committed README figure is missing")
        with tempfile.TemporaryDirectory() as tmp:
            fresh = bench.write_png(Path(tmp) / "retention-bars.png")
            self.assertEqual(fresh.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            self.assertGreater(fresh.stat().st_size, 1000)

    def test_toy_tables_match_the_documented_formula(self):
        spec = importlib.util.spec_from_file_location(
            "readme_tables", TOY / "make_tables.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            module.write(tmp)
            self.assertEqual(
                (Path(tmp) / "train.csv").read_bytes(),
                (TOY / "train.csv").read_bytes(),
            )
            self.assertEqual(
                (Path(tmp) / "test.csv").read_bytes(),
                (TOY / "test.csv").read_bytes(),
            )
        train = (TOY / "train.csv").read_text().splitlines()
        self.assertEqual(len(train), 400)
        self.assertEqual(len(train[0].split(",")), 4)
        self.assertFalse(train[0].startswith("x"))

    def test_inspect_transcript_matches_the_readme(self):
        transcript = (TOY / "INSPECT.txt").read_text()
        readme = (REPO / "README.md").read_text()
        self.assertIn('"artifact_bytes": 196', transcript)
        self.assertIn('"rows_fitted": 400', transcript)
        self.assertIn('"compliant": false', transcript)
        self.assertIn("`artifact_bytes` 196", readme)
        self.assertIn("`compliant` is false", readme)
