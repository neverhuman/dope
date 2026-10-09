"""v2 binder and panel on synthetic fixtures. No benchmark data is read."""

import json
import tempfile
import unittest
from pathlib import Path

from research.benchmark.review_fixes import bind_v2_cells as B
from research.benchmark.review_fixes import v2_panel as P

CONTRACT = B._contract()


def raw(method, configuration, dataset, fit_seed, sample_seed, size, retention, informative=True):
    utility = {name: {"trtr_loss": 0.1, "tstr_loss": 0.2, "informative": informative,
                      "retention": retention if informative else None} for name in P.AUDITORS}
    return {"status": "ok", "method": method, "configuration": configuration, "dataset": dataset,
            "fit_seed": fit_seed, "sample_seed": sample_seed, "size": size, "official_tests_opened": False,
            "synthetic_sha256": "a" * 64,
            "report": {"implementation_sha256": CONTRACT["implementation_sha256"],
                       "dependencies": CONTRACT["dependencies"], "null_loss": 1.0, "utility": utility}}


def ledger(n_lineages=8, dope=0.9, other=0.8):
    cells = []
    for index in range(n_lineages):
        dataset = f"{index:016x}"
        for fit_seed in (11, 23, 37):
            for seed in (101, 211, 307):
                for size in (1, 4):
                    cells.append(B.public_cell(raw("DOPE", "headline", dataset, fit_seed, seed, size,
                                                   dope + 0.001 * index), CONTRACT, "t"))
        for seed in (101, 211, 307):
            for size in (1, 4):
                cells.append(B.public_cell(raw("TabSyn", "scaled_200_vae_1000_diffusion", dataset, 11, seed,
                                               size, other), CONTRACT, "t"))
    return cells


class BinderTests(unittest.TestCase):
    def test_refusals(self):
        good = raw("DOPE", "headline", "aa00000000000001", 11, 101, 4, 0.9)
        self.assertEqual(B.public_cell(good, CONTRACT, "t")["configuration"], "headline")
        for change in ({"official_tests_opened": True}, {"method": "Mystery"}, {"dataset": "x"},
                       {"size": 2}, {"report": {**good["report"], "implementation_sha256": "0" * 64}},
                       {"report": {**good["report"], "dependencies": {"numpy": "2.0"}}}):
            with self.subTest(change=sorted(change)):
                with self.assertRaises(ValueError):
                    B.public_cell({**good, **change}, CONTRACT, "t")

    def test_duplicates_refuse_and_ledger_is_canonical(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "cells"
            for name in ("a", "b"):
                path = root / name / "sample-101-4n.metric.json"
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps(raw("DOPE", "headline", "aa00000000000001", 11, 101, 4, 0.9)))
            with self.assertRaises(ValueError):
                B.collect([(root, "t", None)], CONTRACT)
            only = B.collect([(root / "a", "t", None)], CONTRACT)
            meta = B.write_ledger(only, Path(tmp) / "out" / "scalar-cells.jsonl", CONTRACT)
            self.assertEqual(meta["cells"], 1)
            override = B.collect([(root / "b", "t", "headline_bnew")], CONTRACT)
            self.assertEqual(override[0]["configuration"], "headline_bnew")


class FitLedgerTests(unittest.TestCase):
    def test_charged_bytes_need_the_model_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            cell = Path(tmp) / "aa00000000000001" / "headline" / "fit-11"
            cell.mkdir(parents=True)
            record = {"status": "ok", "dataset": "aa00000000000001", "fit_seed": 11, "arm": "headline",
                      "artifact_bytes": 5, "projection_bytes": 7, "official_tests_opened": False}
            (cell / "fit.json").write_text(json.dumps(record))
            with self.assertRaises(ValueError):
                B.collect_fits([(Path(tmp), "t", None)])
            (cell / "model.dpk").write_bytes(b"12345")
            fits = B.collect_fits([(Path(tmp), "t", None)])
            self.assertEqual(fits[0]["charged_bytes"], 12)
            self.assertEqual(fits[0]["configuration"], "headline")


class PanelTests(unittest.TestCase):
    def test_reduce_pairs_headline_against_comparators(self):
        panel = P.reduce(ledger(), {}, draws=200)
        row = [item for item in panel["contrasts"] if item["method"] == "TabSyn"
               and item["size"] == 4 and item["auditor"] == "catboost"][0]
        self.assertEqual(row["n"], 8)
        self.assertGreater(row["hl"], 0)
        self.assertEqual(row["verdict"], "above")
        self.assertEqual(row["family"], "primary")
        self.assertEqual(row["holm_family_n"], 12)
        self.assertIsNone(panel["strongest"])  # fewer than 90 paired lineages

    def test_incomplete_sample_triples_drop_the_fit(self):
        cells = [cell for cell in ledger(n_lineages=6) if not (cell["method"] == "TabSyn" and cell["sample_seed"] == 307)]
        panel = P.reduce(cells, {}, draws=50)
        self.assertFalse(any(item["method"] == "TabSyn" for item in panel["contrasts"]))

    def test_below_minimum_pairs_has_no_test(self):
        panel = P.reduce(ledger(n_lineages=4), {}, draws=50)
        row = [item for item in panel["contrasts"] if item["method"] == "TabSyn"][0]
        self.assertIsNone(row["wilcoxon_p"])
        self.assertEqual(row["ci_status"], "not_identified_below_minimum_pairs")


if __name__ == "__main__":
    unittest.main()
