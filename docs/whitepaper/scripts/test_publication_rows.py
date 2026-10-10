"""Publication rows come from a receipt or stay the words not measured."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import publication_rows as rows
from pinned_inputs import PinRefused


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


def _pins(root):
    """The fixture declaration: every regular file in the fixture root with its length and digest."""
    return {path.name: (path.stat().st_size, hashlib.sha256(path.read_bytes()).hexdigest())
            for path in Path(root).iterdir() if path.is_file() and not path.is_symlink()}


class PublicationRows(unittest.TestCase):
    def test_missing_receipt_is_not_measured(self):
        with tempfile.TemporaryDirectory() as tmp:
            text = rows.render(Path(tmp), {})
        self.assertNotIn("not measured", text)
        self.assertIn("Publication & Lineages", text)
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
            pins = _pins(root)
            text = rows.render(root, pins)
            self.assertEqual(rows.phrase("TabDDPM", root, pins), "0.321")
            self.assertEqual(rows.phrase("TabSyn", root, pins), "not measured")
        self.assertIn("TabDDPM, author\\_default & 17 & 0.500 & 0.321 & 0.179", text)
        self.assertNotIn("TabSyn &", text)
        self.assertNotIn("Forest-Flow &", text)
        self.assertNotIn("ARF &", text)
        self.assertNotIn("not measured", text)

    def test_non_displayed_profile_is_omitted(self):
        display, filename, method, format_name = next(
            item for item in rows.PUBLICATIONS if item[0] == "ARF"
        )
        document = _report(method, format_name, median=0.321)
        extra = dict(document["paired_descriptive"][0])
        extra["configuration"] = "features12_steps8192"
        extra["reference_median_retention"] = 0.111
        extra["dope_median_retention"] = 0.222
        document["paired_descriptive"].append(extra)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / filename).write_text(json.dumps(document))
            text = rows.render(root, _pins(root))
            self.assertEqual(rows.phrase(display, root, _pins(root)), "0.321")
        self.assertNotIn("0.111", text)
        self.assertNotIn("0.222", text)
        self.assertNotIn("8192", text)
        self.assertIn("ARF, author\\_default & 17 & 0.500 & 0.321 & 0.179", text)

    def test_a_dope_win_flag_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            display, filename, method, format_name = rows.PUBLICATIONS[0]
            document = _report(method, format_name)
            document["cells"][0]["counts_as_dope_win"] = True
            (root / filename).write_text(json.dumps(document))
            with self.assertRaises(ValueError):
                rows.phrase(display, root, _pins(root))

    def test_superiority_must_stay_null(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            display, filename, method, format_name = rows.PUBLICATIONS[0]
            document = _report(method, format_name)
            document["paired_descriptive"][0]["superiority"] = 0.01
            (root / filename).write_text(json.dumps(document))
            with self.assertRaises(ValueError):
                rows.rows_for(display, root, _pins(root))

    def test_wrong_method_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            display, filename, method, format_name = rows.PUBLICATIONS[1]
            document = _report("TabDDPM", format_name)
            (root / filename).write_text(json.dumps(document))
            with self.assertRaises(ValueError):
                rows.load_publication(root / filename, method, format_name, _pins(root))

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
            matching = {link.name: (2, hashlib.sha256(b"{}").hexdigest())}
            with self.assertRaisesRegex(PinRefused, "symlink"):
                rows.phrase("TabDDPM", results, matching)

    def test_render_does_not_create_a_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            before = set(root.iterdir())
            rows.render(root, {})
            self.assertEqual(set(root.iterdir()), before)

    def test_committed_rows_are_reproduced_from_the_pinned_receipt(self):
        committed = (rows.OUT).read_text()
        self.assertEqual(rows.render(), committed)
        self.assertEqual(set(rows.PINS), {"arf-matched-population-validation.json"})

    def test_present_but_unregistered_receipt_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _display, filename, method, format_name = rows.PUBLICATIONS[0]
            (root / filename).write_text(json.dumps(_report(method, format_name)))
            with self.assertRaisesRegex(PinRefused, "not registered"):
                rows.render(root, {})

    def test_a_registered_receipt_that_is_missing_is_refused_not_unmeasured(self):
        with tempfile.TemporaryDirectory() as tmp:
            _display, filename, _method, _format_name = rows.PUBLICATIONS[0]
            with self.assertRaisesRegex(PinRefused, "registered publication is missing"):
                rows.render(Path(tmp), {filename: (10, "0" * 64)})
            with self.assertRaisesRegex(PinRefused, "registered publication is missing"):
                rows.render(Path(tmp))

    def test_direct_load_refuses_a_wrong_digest_before_decoding(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _display, filename, method, format_name = rows.PUBLICATIONS[0]
            (root / filename).write_text(json.dumps(_report(method, format_name)))
            length, _digest = _pins(root)[filename]
            with patch.object(rows.json, "loads", side_effect=AssertionError("decoded")), \
                    self.assertRaisesRegex(PinRefused, "digest differs"):
                rows.load_publication(root / filename, method, format_name, {filename: (length, "0" * 64)})
            with self.assertRaisesRegex(PinRefused, "not registered"):
                rows.load_publication(root / filename, method, format_name)

    def test_a_results_directory_reached_through_a_symlink_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            real, link = Path(tmp) / "real", Path(tmp) / "results"
            real.mkdir()
            _display, filename, method, format_name = rows.PUBLICATIONS[0]
            (real / filename).write_text(json.dumps(_report(method, format_name)))
            link.symlink_to(real, target_is_directory=True)
            with self.assertRaisesRegex(PinRefused, "through a symlink"):
                rows.render(link, _pins(real))


if __name__ == "__main__":
    unittest.main()
