"""Mocked ingestion guards. No benchmark rows, processes, or registry access."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from docs.whitepaper.scripts import review_controls as controls
from docs.whitepaper.scripts import review_privacy as privacy
from research.benchmark.review_fixes import privacy_fidelity as producer


class ControlFamilyTests(unittest.TestCase):
    def test_missing_p_value_keeps_planned_family_metadata(self):
        record = {"rows": [{"dataset": "aa", "display_name": "fixture"}]}
        mapped = {auditor: {"aa": 0.0} for auditor in controls.AUDITORS}
        dope = {size: mapped for size in (1, 4)}
        drafted = []

        def paired(*_args, **_kwargs):
            item = {
                "n": 1,
                "median": 0.0,
                "lo": 0.0,
                "hi": 0.0,
                "mean": 0.0,
                "wins": 0,
                "ties": 1,
                "losses": 0,
                "tost": {"equivalent": False},
                "wilcoxon_p": None if not drafted else 0.01,
            }
            drafted.append(item)
            return item

        def load(name):
            return (record if name == "s3-lineage-record.json" else {"cells": []}), "fixture"

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with (
                patch.object(controls, "RESULTS", root),
                patch.object(controls, "GENERATED", root),
                patch.object(controls.sys, "argv", ["review_controls", "--cells", "fixture"]),
                patch.object(controls, "_cells", return_value=[{}]),
                patch.object(controls, "_load", side_effect=load),
                patch.object(controls, "primary_map", return_value=dope),
                patch.object(controls, "_median_map", return_value=mapped),
                patch.object(controls, "median_ci", return_value={"n": 1, "median": 0.0, "lo": 0.0, "hi": 0.0}),
                patch.object(controls, "_paired", side_effect=paired),
                patch.object(controls, "_split_rows", return_value=[]),
                patch.object(controls, "_sha", return_value="a" * 64),
                patch.object(controls, "render_controls", return_value=""),
                patch.object(controls, "write_tex"),
            ):
                controls.main()
            written = json.loads((root / "review-fixes-controls-v1" / "panel.json").read_text())
        rows = written["paired"]
        self.assertEqual([row["holm_family_n"] for row in rows if row["size"] == 1], [3, 3, 3])
        self.assertEqual([row["holm_family_n"] for row in rows if row["size"] == 4], [6] * 6)
        self.assertIsNone(rows[0]["holm_p"])
        self.assertAlmostEqual(rows[1]["holm_p"], 0.03)


class SamplePinTests(unittest.TestCase):
    def test_index_propagates_dope_and_sdv_sample_pins(self):
        cells = [
            {
                "dataset": "aa",
                "method": "DOPE",
                "configuration": "features12_steps2048",
                "fit_seed": 11,
                "size_multiplier": 1,
                "sample_seed": 101,
                "metric_receipt": {"sample_sha256": "a" * 64},
            },
            {
                "dataset": "aa",
                "method": "CTGAN",
                "configuration": "native_selected",
                "fit_seed": 11,
                "size_multiplier": 1,
                "sample_seed": 101,
                "sample_evidence": {"path": "/fixture/sample.csv", "sha256": "b" * 64},
            },
        ]
        with (
            patch.object(producer, "PUBLISHED", [
                ("fixture.json", "DOPE", "features12_steps2048"),
                ("fixture.json", "CTGAN", "native_selected"),
            ]),
            patch.object(Path, "read_text", return_value=json.dumps({"cells": cells})),
            patch.object(producer, "_resolve_sample_csv", return_value="/fixture/sample.csv"),
        ):
            jobs = producer._index(Path("fixture-results"))
        self.assertEqual([job["expected_sha256"] for job in jobs], ["a" * 64, "b" * 64])

    def test_receipt_json_sha_is_not_a_sample_pin(self):
        cell = {"method": "ForestDiffusion/Forest-Flow", "receipt": {"sha256": "a" * 64}}
        self.assertIsNone(producer._expected_sample_sha(cell, "/fixture/sample.csv"))

    def test_missing_or_mismatched_pin_prevents_table_reads(self):
        for expected in (None, "a" * 64):
            with self.subTest(expected=expected):
                job = {
                    "dataset": "aa",
                    "method": "DOPE",
                    "configuration": "features12_steps2048",
                    "size": 1,
                    "sample_seed": 101,
                    "csv": "/fixture/sample.csv",
                    "out": "/fixture/cell.json",
                    "expected_sha256": expected,
                }
                with (
                    patch.object(Path, "mkdir"),
                    patch.object(Path, "exists", return_value=False),
                    patch.object(Path, "is_file", return_value=True),
                    patch.object(producer, "_sha", return_value="b" * 64),
                    patch.object(producer, "_write") as write,
                    patch.object(producer, "_read_table") as rows,
                    patch.object(producer, "_wait_for_memory") as wait,
                ):
                    result = producer._score_one(job)
                self.assertEqual(result["status"], "unavailable")
                self.assertEqual(write.call_args.args[1]["status"], "unavailable")
                rows.assert_not_called()
                wait.assert_not_called()

    def test_prior_success_requires_explicit_pin_revalidation(self):
        job = {
            "dataset": "aa",
            "method": "DOPE",
            "configuration": "features12_steps2048",
            "size": 1,
            "sample_seed": 101,
            "expected_sha256": "a" * 64,
        }
        prior = {**job, "status": "ok", "synthetic_sha256": "a" * 64}
        prior.pop("expected_sha256")
        self.assertIn("revalidation", producer._existing_problem(prior, job))
        self.assertIsNone(producer._existing_problem({**prior, "expected_sha256": "a" * 64}, job))


class ClassSizeTests(unittest.TestCase):
    def test_finite_c2st_requires_exact_integer_declared_class_size(self):
        cell = {
            "rows": {"validation": 18, "synthetic": 80},
            "c2st_catboost_auc": 0.6,
            "c2st_rows_per_class": 18,
        }
        self.assertEqual(privacy._c2st_rows(cell), 18)
        for count in (None, True, 18.5, 19):
            with self.subTest(count=count), self.assertRaisesRegex(ValueError, "class-n"):
                privacy._c2st_rows({**cell, "c2st_rows_per_class": count})
        self.assertIsNone(privacy._c2st_rows({**cell, "c2st_catboost_auc": None, "c2st_rows_per_class": None}))

    def test_cross_method_disagreement_refuses_before_reduction(self):
        cells = [
            {
                "dataset": "aa",
                "method": method,
                "configuration": "fixture",
                "status": "ok",
                "size": 1,
                "sample_seed": 101,
                "rows": {"fit": 20, "synthetic": 20, "validation": count},
                "c2st_catboost_auc": 0.6,
                "c2st_rows_per_class": count,
            }
            for method, count in (("DOPE", 18), ("GaussianCopula", 19))
        ]
        record = {"rows": [{"dataset": "aa", "display_name": "fixture"}]}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with (
                patch.object(privacy, "RESULTS", root),
                patch.object(privacy, "GENERATED", root),
                patch.object(privacy.sys, "argv", ["review_privacy", "--cells", "fixture"]),
                patch.object(privacy, "_cells", return_value=cells),
                patch.object(privacy, "_load", return_value=(record, "fixture")),
                patch.object(privacy, "median_ci", side_effect=AssertionError("reduction before class-n validation")) as reduce,
                self.assertRaisesRegex(ValueError, "differs across methods"),
            ):
                privacy.main()
        reduce.assert_not_called()


if __name__ == "__main__":
    unittest.main()
