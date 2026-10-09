"""Fail-closed identities and pre-decode authentication for paper reductions."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import compute_panel as panel


class PanelInputTests(unittest.TestCase):
    def test_every_registered_input_authenticates_and_rejects_drift(self):
        for name in panel.INPUT_SHA256:
            with self.subTest(name=name):
                original = panel.RESULTS / name
                with tempfile.TemporaryDirectory(dir=panel.REPO / "target") as directory:
                    path = Path(directory) / name
                    path.write_bytes(original.read_bytes() + b" ")
                    with patch.object(panel, "RESULTS", Path(directory)):
                        with self.assertRaisesRegex(ValueError, "digest mismatch"):
                            panel.load(name)
                self.assertIsInstance(panel.load(name), dict)

    def test_every_transitive_hash_inventory_input_rejects_drift(self):
        original_read = Path.read_bytes
        panel.authenticate_hash_inventory()
        for relative in panel.HASH_INPUT_SHA256:
            target = panel.REPO / relative
            def read(path):
                raw = original_read(path)
                return raw + b" " if path == target else raw
            with self.subTest(relative=relative), patch.object(Path, "read_bytes", read):
                with self.assertRaisesRegex(ValueError, "digest mismatch"):
                    panel.authenticate_hash_inventory()

    def test_unregistered_missing_and_symlinked_inputs_fail(self):
        with self.assertRaisesRegex(ValueError, "unregistered"):
            panel.load("../unregistered.json")
        with tempfile.TemporaryDirectory(dir=panel.REPO / "target") as directory:
            root = Path(directory)
            with patch.object(panel, "RESULTS", root):
                with self.assertRaises(FileNotFoundError):
                    panel.load("s3-data.lock.json")
                (root / "s3-data.lock.json").symlink_to(__file__)
                with self.assertRaisesRegex(ValueError, "symlinked"):
                    panel.load("s3-data.lock.json")

    def test_summary_duplicate_and_unknown_raise_even_for_blank_rows(self):
        row = {"method": "DOPE", "dataset": "aa", "configuration": "frozen", "size_multiplier": 4,
               "fit_seed": 11, "utility": {}}
        for rows in ([row, copy.deepcopy(row)], [dict(row, method="unknown")]):
            with self.assertRaises(ValueError):
                panel.summary_map(rows, "DOPE", "frozen", 4, "linear")

    def test_duplicate_sample_cell_raises(self):
        row = {"method": "DOPE", "dataset": "aa", "configuration": "frozen", "size_multiplier": 4,
               "fit_seed": 11, "sample_seed": 101}
        with self.assertRaises(ValueError):
            panel.group_cells([row, copy.deepcopy(row)], "DOPE", "frozen", 4)

    def test_distinct_fit_seeds_cannot_overwrite_the_same_output_key(self):
        row = {"method": "DOPE", "dataset": "aa", "configuration": "frozen", "size_multiplier": 4,
               "fit_seed": 11, "utility": {}}
        rows = [row, dict(row, fit_seed=23)]
        with self.assertRaises(ValueError):
            panel.summary_map(rows, "DOPE", "frozen", 4, "linear")
        sampled = [dict(item, sample_seed=101) for item in rows]
        with self.assertRaises(ValueError):
            panel.group_cells(sampled, "DOPE", "frozen", 4)

    def test_missing_matched_lineage_raises_before_statistics(self):
        record = {"rows": [{"dataset": "aa"}],
                  "comparators": {name: {} for name in panel.DENSITY_COMPARATORS}}
        with self.assertRaisesRegex(ValueError, "missing comparator lineage"):
            panel.marginal_from_rows(record, None)


if __name__ == "__main__":
    unittest.main()
