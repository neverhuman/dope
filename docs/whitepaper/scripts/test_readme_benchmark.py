"""The README benchmark block and figures come from v2-headline.json, like the paper."""

import copy
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

    def test_counts_and_cap_come_from_their_sources(self):
        headline = json.loads(bench.HEADLINE.read_text())
        predeclare = json.loads(bench.PREDECLARE.read_text())
        contract = json.loads(bench.CONTRACT.read_text())
        facts = bench.load_facts(headline)
        self.assertEqual(facts["panel"], next(iter(predeclare["methods"]["primary"].values()))["lineages"])
        self.assertEqual(facts["fit_seeds"], bench.COUNT_WORDS[len(predeclare["dope"]["fit_seeds"])])
        self.assertIn(f"percentile {facts['level']}% interval", predeclare["estimands"]["uncertainty"])
        self.assertEqual(facts["cap"], contract["tier_byte_limits"]["l3"])
        block = bench.benchmark_block(headline)
        self.assertIn(f"{facts['panel']} PMLB regression tables", block)
        self.assertIn(f"DOPE uses {facts['fit_seeds']} fit seeds", block)
        self.assertIn(f"{facts['level']}% nested bootstrap intervals", block)
        self.assertIn(f"within the {facts['cap']:,}-byte release cap", block)
        source = Path(bench.__file__).read_text()
        for literal in ("10240", "10,240", "100 PMLB", "five fit seeds", "95%"):
            self.assertNotIn(literal, source, f"renderer types {literal!r} instead of reading it")

    def test_a_drifted_source_breaks_the_readme_check(self):
        headline = json.loads(bench.HEADLINE.read_text())
        predeclare = json.loads(bench.PREDECLARE.read_text())
        contract = json.loads(bench.CONTRACT.read_text())
        committed = bench.readme_block(bench.README.read_text())
        self.assertEqual(committed, bench.expected_readme_block(headline))

        def rendered(h, p, c):
            return bench.BEGIN + "\n" + bench.benchmark_block(h, bench.panel_facts(h, p, c)).rstrip() + "\n" + bench.END

        h2, p2 = copy.deepcopy(headline), copy.deepcopy(predeclare)
        for method in p2["methods"]["primary"].values():
            method["lineages"] = 101
        next(arm for arm in h2["arms"] if arm["code"] == "Dope")["bytes"]["n"] = 101
        p3 = copy.deepcopy(predeclare)
        p3["dope"]["fit_seeds"] = p3["dope"]["fit_seeds"][:4]
        p4 = copy.deepcopy(predeclare)
        p4["estimands"]["uncertainty"] = p4["estimands"]["uncertainty"].replace("percentile 95% interval",
                                                                                "percentile 90% interval")
        c2 = copy.deepcopy(contract)
        c2["tier_byte_limits"]["l3"] = 32768
        for label, h, p, c, text in (("panel", h2, p2, contract, "101 PMLB"), ("seeds", headline, p3, contract, "four fit seeds"),
                                     ("level", headline, p4, contract, "90% nested"), ("cap", headline, predeclare, c2, "32,768-byte")):
            with self.subTest(label=label):
                block = rendered(h, p, c)
                self.assertIn(text, block)
                self.assertNotEqual(block, committed)

    def test_disagreeing_sources_are_refused(self):
        headline = json.loads(bench.HEADLINE.read_text())
        predeclare = json.loads(bench.PREDECLARE.read_text())
        contract = json.loads(bench.CONTRACT.read_text())
        split = copy.deepcopy(predeclare)
        next(iter(split["methods"]["primary"].values()))["lineages"] = 99
        mismatch = copy.deepcopy(headline)
        next(arm for arm in mismatch["arms"] if arm["code"] == "Dope")["bytes"]["n"] = 98
        unstated = copy.deepcopy(predeclare)
        unstated["estimands"]["uncertainty"] = "Nested bootstrap."
        for h, p in ((headline, split), (mismatch, predeclare), (headline, unstated)):
            with self.assertRaises(ValueError):
                bench.panel_facts(h, p, contract)

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
        self.assertIn('"artifact_bytes": 151', transcript)
        self.assertIn('"rows_fitted": 400', transcript)
        self.assertIn('"dependence": "chow_liu"', transcript)
        self.assertIn('"target": "sparse_linear"', transcript)
        self.assertIn('"compliant": false', transcript)
        self.assertIn("`artifact_bytes` 151", readme)
        self.assertIn("dependence `chow_liu`", readme)
        self.assertIn("target `sparse_linear`", readme)
        self.assertIn("`compliant` is false", readme)
