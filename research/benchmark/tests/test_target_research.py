"""Incomplete and noninformative research evidence cannot yield a full estimate."""

import copy
import json
import unittest
from pathlib import Path

from research.benchmark import publish_target_research as publisher


class TargetResearchTests(unittest.TestCase):
    def cells(self):
        return [
            dict(
                dataset=d,
                profile=p,
                fit_seed=f,
                sample_seed=s,
                size_multiplier=z,
                status="ok",
                metrics={
                    "utility": {
                        a: {"retention": 0.8, "informative": True}
                        for a in publisher.AUDITORS
                    }
                },
            )
            for d in publisher.DATASETS
            for p in publisher.PROFILES
            for f in publisher.FIT_SEEDS
            for s in publisher.SAMPLE_SEEDS
            for z in publisher.SIZES
        ]

    def test_missing_duplicate_or_wrong_seed_rejected(self):
        rows = self.cells()
        wrong_seed = copy.deepcopy(rows)
        wrong_seed[0]["fit_seed"] = 37
        for changed in (rows[:-1], rows + [copy.deepcopy(rows[0])], wrong_seed):
            with self.assertRaises(ValueError):
                publisher.summarize(changed)

    def test_failed_sample_and_low_signal_prevent_full_estimate(self):
        for reason in ("failed", "low_signal"):
            rows = self.cells()
            if reason == "failed":
                rows[0].update(status="metric_timeout", metrics=None)
            else:
                rows[0]["metrics"]["utility"]["catboost"]["informative"] = False
            result = publisher.summarize(rows)
            affected = next(
                r
                for r in result
                if r["dataset"] == "Adult"
                and r["profile"] == publisher.PROFILES[0]
                and r["size_multiplier"] == 1
                and r["auditor"] == "catboost"
            )
            self.assertEqual(affected["complete_fit_seeds"], 1)
            self.assertIsNone(affected["median_of_fit_medians"])

    def test_nonfinite_retention_rejected(self):
        rows = self.cells()
        rows[0]["metrics"]["utility"]["catboost"]["retention"] = float("nan")
        with self.assertRaises(ValueError):
            publisher.summarize(rows)

    def test_committed_matrix_schema_and_tables(self):
        from jsonschema import Draft202012Validator

        path = (
            Path(publisher.__file__).with_name("results")
            / "pilot24-target-gpu-research.json"
        )
        report = json.loads(path.read_text())
        schema = json.loads(path.with_suffix(".schema.json").read_text())
        validator = Draft202012Validator(schema)
        validator.validate(report)
        csv_text, markdown = publisher.render(report)
        self.assertEqual(csv_text, path.with_suffix(".csv").read_text())
        self.assertEqual(markdown, path.with_suffix(".md").read_text())
        self.assertEqual(len(report["raw_cells"]), 144)
        self.assertEqual(len(report["packed_cells"]), 48)
        self.assertEqual(len(report["fit_attempts"]), 24)
        for field, value in (
            ("mfs_v2", 0.99),
            ("ptf_v1", 0.99),
            ("official_tests_opened", True),
            ("production_certified", True),
        ):
            changed = copy.deepcopy(report)
            changed[field] = value
            with self.assertRaises(ValueError):
                publisher.render(changed)
            self.assertFalse(validator.is_valid(changed))
        changed = copy.deepcopy(report)
        changed["packed_cells"].pop()
        self.assertFalse(validator.is_valid(changed))
        self.assertTrue(
            all(r["contributes_dope_win"] is False for r in report["raw_cells"])
        )


if __name__ == "__main__":
    unittest.main()
