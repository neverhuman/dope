"""The README benchmark block and figures come from v2-headline.json, like the paper."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import readme_benchmark as bench

REPO = Path(__file__).resolve().parents[3]
TOY = REPO / "examples" / "readme-regression"


class ReadmeBenchmark(unittest.TestCase):
    def test_readme_block_matches_the_v2_headline(self):
        headline = json.loads(bench.HEADLINE.read_text())
        readme = bench.README.read_text()
        self.assertEqual(bench.readme_block(readme), bench.expected_readme_block(headline))
        self.assertIn("docs/whitepaper/dope-mfs.pdf", readme)
        for path in bench.PNGS:
            self.assertIn("docs/readme/" + path.name, readme)
            self.assertTrue(path.is_file(), f"committed README figure {path.name} is missing")
            self.assertEqual(path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

    def test_block_names_the_computed_strongest_comparator(self):
        headline = json.loads(bench.HEADLINE.read_text())
        key = (headline["strongest"]["method"], headline["strongest"]["configuration"])
        strongest = next(arm for arm in headline["arms"] if (arm["method"], arm["configuration"]) == key)
        block = bench.benchmark_block(headline)
        self.assertIn(f"The strongest full-panel comparator is {strongest['display']}.", block)
        self.assertNotIn("D.O.P.E.", bench.README.read_text())

    def test_splice_refuses_duplicate_markers(self):
        with self.assertRaises(ValueError):
            bench.splice_readme(bench.BEGIN + bench.BEGIN + bench.END, "x")

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
