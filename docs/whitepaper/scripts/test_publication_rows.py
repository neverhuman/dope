"""Publication rows come from a receipt or stay the words not measured."""

import json
import tempfile
import unittest
from pathlib import Path

import publication_rows as rows


FIXTURE_MEDIAN = 0.321


def _report(method, format_name, median=FIXTURE_MEDIAN):
    return {
        "format": format_name,
        "version": 1,
        "cells": [{
            "dataset": "aa",
            "method": method,
            "configuration": "author_default",
            "size_multiplier": 4,
            "status": "ok",
            "charged_artifact_bytes": 100,
            "counts_as_dope_win": False,
            "mfs_v2": None,
            "ptf_v1": None,
            "release_safe": None,
            "superiority": None,
        }],
        "paired_descriptive": [{
            "configuration": "features12_steps2048",
            "reference_method": method,
            "reference_configuration": "author_default",
            "size_multiplier": 4,
            "auditor": "catboost",
            "paired_complete_informative_lineages": 17,
            "dope_median_retention": 0.5,
            "reference_median_retention": median,
            "median_paired_difference": 0.179,
            "superiority": None,
        }],
    }


class PublicationRows(unittest.TestCase):
    def test_missing_receipt_is_not_measured(self):
        with tempfile.TemporaryDirectory() as tmp:
            text = rows.render(Path(tmp))
        self.assertEqual(text.count("not measured"), 12)
        self.assertNotIn(str(FIXTURE_MEDIAN), text)
        self.assertNotIn("0.940", text)
        self.assertFalse(rows.RESULTS.joinpath("tabddpm-matched-population-validation.json").is_file())

    def test_fixture_median_is_copied_from_the_receipt(self):
        source = Path(rows.__file__).read_text()
        self.assertNotIn(str(FIXTURE_MEDIAN), source)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for display, filename, method, format_name in rows.PUBLICATIONS:
                if display != "TabDDPM":
                    continue
                (root / filename).write_text(json.dumps(_report(method, format_name)))
            text = rows.render(root)
            self.assertEqual(rows.phrase("TabDDPM", root), "0.321")
            self.assertEqual(rows.phrase("TabSyn", root), "not measured")
        self.assertIn("TabDDPM, author\\_default & 17 & 0.500 & 0.321 & 0.179", text)
        self.assertIn("TabSyn & not measured", text)
        self.assertIn("Forest-Flow & not measured", text)

    def test_a_dope_win_flag_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            display, filename, method, format_name = rows.PUBLICATIONS[0]
            document = _report(method, format_name)
            document["cells"][0]["counts_as_dope_win"] = True
            (root / filename).write_text(json.dumps(document))
            with self.assertRaises(ValueError):
                rows.phrase(display, root)

    def test_superiority_must_stay_null(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            display, filename, method, format_name = rows.PUBLICATIONS[0]
            document = _report(method, format_name)
            document["paired_descriptive"][0]["superiority"] = 0.01
            (root / filename).write_text(json.dumps(document))
            with self.assertRaises(ValueError):
                rows.rows_for(display, root)

    def test_wrong_method_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            display, filename, method, format_name = rows.PUBLICATIONS[1]
            document = _report("TabDDPM", format_name)
            (root / filename).write_text(json.dumps(document))
            with self.assertRaises(ValueError):
                rows.load_publication(root / filename, method, format_name)

    def test_filename_cannot_escape_the_results_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                rows.locate(Path(tmp), "../secret.json")

    def test_symlink_out_of_the_results_directory_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            results = root / "results"
            results.mkdir()
            outside = root / "outside.json"
            outside.write_text("{}")
            link = results / "tabddpm-matched-population-validation.json"
            link.symlink_to(outside)
            with self.assertRaises(ValueError):
                rows.phrase("TabDDPM", results)

    def test_render_does_not_create_a_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            before = set(root.iterdir())
            rows.render(root)
            self.assertEqual(set(root.iterdir()), before)


if __name__ == "__main__":
    unittest.main()
