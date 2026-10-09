"""The MFS-v3 density table is recomputed from lineage retentions."""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import check_paper
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
        # The scalar note is prose only. The vector pins live in the captioned float.
        for environment in ("tabular", "center", "table"):
            self.assertNotIn("\\begin{" + environment, note)
        layers = mfs_v3_table.layers_tex()
        for name in TABULAR_AUDITORS:
            for part in VECTOR_PIN[name].split("_"):
                self.assertIn(part, layers)
                self.assertNotIn(part, note)

    def test_layer_table_is_a_captioned_float_with_no_holm_family(self):
        layers = mfs_v3_table.layers_tex()
        self.assertTrue(layers.startswith("\\begin{table}"))
        self.assertTrue(layers.rstrip().endswith("\\end{table}"))
        begin = layers.index("\\begin{tabular}")
        self.assertLess(layers.index("\\caption{"), begin)
        self.assertLess(begin, layers.index("\\end{tabular}"))
        caption = check_paper._caption_bodies(layers)
        self.assertEqual(len(caption), 1)
        self.assertTrue(caption[0].endswith("No Holm family."))
        self.assertEqual(check_paper.table_family_failures(layers), [])
        self.assertEqual(layers.count("\\begin{tabular}"), 1)
        for name in TABULAR_AUDITORS:
            self.assertIn(mfs_v3_table.AUDITOR_TEX[name] + " & ", layers)

    def test_check_refuses_a_missing_or_stale_layer_table(self):
        lineages = [
            _lineage(method, dataset, 0.1 if method != "DOPE" else 0.0)
            for method in mfs_v3_table.METHODS
            for dataset in ("aa", "bb", "cc")
        ]
        measured = mfs_v3_table.measure(_receipt(lineages))

        def fake_draw(_measured, bar_path, paired_path):
            bar_path.parent.mkdir(parents=True, exist_ok=True)
            bar_path.write_bytes(b"bars")
            paired_path.write_bytes(b"paired")

        target = Path(mfs_v3_table.REPO) / "target"
        target.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=target) as directory:
            root = Path(directory)
            layers = root / "generated" / mfs_v3_table.LAYERS_NAME
            with patch.object(mfs_v3_table, "GENERATED", root / "generated"), \
                    patch.object(mfs_v3_table, "FIGURES", root / "figures"), \
                    patch.object(mfs_v3_table, "draw_figures", fake_draw), \
                    redirect_stdout(io.StringIO()):
                mfs_v3_table.write_outputs(measured)
                self.assertEqual(layers.read_text(), mfs_v3_table.layers_tex())
                scalar = (root / "generated" / mfs_v3_table.SCALAR_NAME).read_text()
                self.assertNotIn("tabular", scalar)
                mfs_v3_table.check_outputs(measured)
                layers.write_text(layers.read_text().replace(" No Holm family.", ""))
                with self.assertRaisesRegex(SystemExit, "layer table"):
                    mfs_v3_table.check_outputs(measured)
                layers.unlink()
                with self.assertRaisesRegex(SystemExit, "layer table"):
                    mfs_v3_table.check_outputs(measured)

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
        receipt = mfs_v3_table.load_receipt()
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
        supplement = manuscript.with_name("supplement.tex").read_text()
        package_text = text + supplement
        self.assertNotIn("are sensitivity checks", text)
        self.assertIn("pre-registered run", text)
        self.assertIn("Neither is averaged into representation closeness", package_text)
        self.assertIn("applied to absolute correlations with the target", package_text)
        self.assertIn("permutation-importance auditor", package_text)
        self.assertIn("Their retentions stay out of that bound", package_text)
        self.assertNotIn("Driver fidelity, distribution fidelity", text)
        self.assertNotIn("no clean-license tabular transfer has been measured", text)
        self.assertIn("\\input{generated/mfs-v3-density-table.tex}", supplement)
        self.assertIn("\\input{generated/mfs-v3-scalar.tex}", supplement)
        self.assertIn("\\input{generated/mfs-v3-layers.tex}", supplement)
        self.assertIn("Downstream Objective-Preserving Encoding", text)
        self.assertNotIn("10{,}240-Byte Tabular Generator", text)
        self.assertNotIn("10240-Byte Tabular Generator", text)
        self.assertIn("BEGIN JOURNAL AVAILABILITY", text)
        self.assertIn("https://github.com/neverhuman/dope", text)
        self.assertIn("not a utility score", text)

    def test_substituted_receipt_byte_refuses_before_a_table_write(self):
        panel = mfs_v3_table.compute_panel
        original = panel.RESULTS / mfs_v3_table.RECEIPT.name
        raw = bytearray(original.read_bytes())
        raw[0] ^= 0x01
        with tempfile.TemporaryDirectory(dir=panel.REPO / "target") as directory:
            root = Path(directory)
            (root / mfs_v3_table.RECEIPT.name).write_bytes(raw)
            generated = root / "generated"
            figures = root / "figures"
            generated.mkdir()
            figures.mkdir()
            with patch.object(panel, "RESULTS", root), \
                    patch.object(mfs_v3_table, "GENERATED", generated), \
                    patch.object(mfs_v3_table, "FIGURES", figures), \
                    patch("mfs_v3_table.sys.argv", ["mfs_v3_table.py"]):
                with self.assertRaisesRegex(ValueError, "digest mismatch"):
                    mfs_v3_table.main()
            self.assertEqual(list(generated.iterdir()), [])
            self.assertEqual(list(figures.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
