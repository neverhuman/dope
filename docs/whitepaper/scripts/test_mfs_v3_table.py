"""The MFS-v3 density table is recomputed from lineage retentions."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import mfs_v3_table
from research.benchmark.mfs_v3_panel import KUMO_ESTIMATORS, REFUSED_ENCODERS, VECTOR_PIN
from research.benchmark.representation import TABULAR_AUDITORS


def _lineage(method, dataset, shift):
    retention = {}
    for index, name in enumerate(TABULAR_AUDITORS):
        retention[name] = 0.8 - shift - 0.05 * index + (0.02 if method == "DOPE" else 0.0)
        retention[name] += 0.01 * (sum(ord(char) for char in dataset) % 5)
    return {
        "counts_as_dope_win": False,
        "dataset": dataset,
        "failed_gates": ["cleartext_absent"],
        "method": method,
        "official_tests_opened": False,
        "retention": retention,
        "score": None,
        "superiority": None,
    }


def _receipt(lineages, scored=0):
    return {
        "counts_as_dope_win": False,
        "format": "dope-mfs-v3-density-panel",
        "lineages": lineages,
        "official_tests_opened": False,
        "scored_lineages": scored,
        "superiority": None,
        "version": 1,
    }


class MfsV3TableTest(unittest.TestCase):
    def test_holm_family_is_the_three_comparators(self):
        lineages = []
        for method in mfs_v3_table.METHODS:
            for dataset in ("aa", "bb", "cc", "dd"):
                lineages.append(_lineage(method, dataset, 0.1 if method != "DOPE" else 0.0))
        measured = mfs_v3_table.measure(_receipt(lineages))
        for auditor in TABULAR_AUDITORS:
            pairs = measured["block"][auditor]["pairs"]
            self.assertEqual(set(pairs), set(mfs_v3_table.COMPARATORS))
            raw = [pairs[name]["wilcoxon_p"] for name in mfs_v3_table.COMPARATORS]
            adjusted = mfs_v3_table.compute_panel.holm(raw)
            for name, value in zip(mfs_v3_table.COMPARATORS, adjusted):
                self.assertEqual(pairs[name]["holm_p"], value)
            self.assertGreater(pairs["GaussianCopula"]["wins"], 0)
        text = mfs_v3_table.table_tex(measured)
        self.assertNotIn("CatBoost", text)
        self.assertNotIn("sensitivity", text)
        for refused in REFUSED_ENCODERS:
            self.assertNotIn(refused, text)
        note = mfs_v3_table.scalar_tex(measured)
        self.assertIn(f"{KUMO_ESTIMATORS} inverse-transformed", note)
        self.assertIn("scores 0 of 16 method--lineage cells (4 lineages", note)
        self.assertIn("Holm family: 3 tests per auditor", text)
        for name in TABULAR_AUDITORS:
            for part in VECTOR_PIN[name].split("_"):
                self.assertIn(part, note)

    def test_a_win_claim_or_refused_encoder_is_rejected(self):
        lineages = [_lineage("DOPE", "aa", 0.0)]
        receipt = _receipt(lineages)
        receipt["counts_as_dope_win"] = True
        with self.assertRaises(ValueError):
            mfs_v3_table.require_receipt(receipt)
        receipt["counts_as_dope_win"] = False
        lineages[0]["retention"]["foundation"] = 0.2
        with self.assertRaises(ValueError):
            mfs_v3_table.require_receipt(receipt)

    def test_duplicate_unknown_and_incomplete_matrix_raise(self):
        import copy
        rows = [_lineage(method, dataset, 0.1) for method in mfs_v3_table.METHODS for dataset in ("aa", "bb")]
        for mutate in (lambda r: r.append(copy.deepcopy(r[0])),
                       lambda r: r[0].update(method="unknown"),
                       lambda r: r.pop()):
            bad = copy.deepcopy(rows)
            mutate(bad)
            with self.assertRaises(ValueError):
                mfs_v3_table.measure(_receipt(bad))

    def test_real_receipt_matches_its_lineage_retentions(self):
        path = mfs_v3_table.RECEIPT
        self.assertTrue(path.is_file(), "density receipt is missing")
        receipt = json.loads(path.read_text())
        measured = mfs_v3_table.measure(receipt)
        self.assertEqual(measured["scored"], 0)
        self.assertEqual(measured["method_lineage_cells"], 388)
        self.assertEqual(measured["cleartext_unscanned"], 388)
        self.assertEqual(measured["n_lineages"], 97)
        self.assertEqual(measured["n_methods"], 4)
        for auditor in TABULAR_AUDITORS:
            for comparator in mfs_v3_table.COMPARATORS:
                pair = measured["block"][auditor]["pairs"][comparator]
                self.assertEqual(pair["n"], 97)
                self.assertIsNotNone(pair["holm_p"])
        manuscript = Path(mfs_v3_table.REPO) / "docs" / "whitepaper" / "dope-mfs.tex"
        text = manuscript.read_text()
        self.assertNotIn("are sensitivity checks", text)
        self.assertIn("were not run", text)
        self.assertIn("averaged into representation closeness", text)
        self.assertIn("applied to absolute correlations with the target", text)
        self.assertIn("permutation-importance auditor", text)
        self.assertIn("not averaged into the bound", text)
        self.assertNotIn("Driver fidelity, distribution fidelity", text)
        self.assertNotIn("no clean-license tabular transfer has been measured", text)
        self.assertIn("\\fittowidthfile{generated/mfs-v3-density-table.tex}", text)
        self.assertIn("\\input{generated/mfs-v3-scalar.tex}", text)
        self.assertIn("Downstream Objective-Preserving Encoding", text)
        self.assertNotIn("10{,}240-Byte Tabular Generator", text)
        self.assertNotIn("10240-Byte Tabular Generator", text)
        self.assertIn("BEGIN JOURNAL AVAILABILITY", text)
        self.assertIn("https://github.com/neverhuman/dope", text)
        self.assertIn("not a utility score", text)


if __name__ == "__main__":
    unittest.main()
