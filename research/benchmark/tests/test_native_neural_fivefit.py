"""Incomplete seed evidence cannot produce a complete matched estimate."""

import copy
import unittest
from research.benchmark import publish_native_neural_fivefit as publisher


class MatchedFiveFitTests(unittest.TestCase):
    def cells(self):
        return [
            {
                "dataset": "Adult",
                "method": "CTGAN",
                "configuration": "default",
                "fit_seed": fit,
                "sample_seed": sample,
                "size_multiplier": size,
                "status": "ok",
                "artifact_bytes": 40000,
                "utility": {
                    a: {"retention": 0.8, "informative": True}
                    for a in publisher.AUDITORS
                },
            }
            for fit in publisher.FIT_SEEDS
            for sample in publisher.SAMPLE_SEEDS
            for size in (1, 4)
        ]

    def test_failed_fit_is_visible_without_complete_estimate(self):
        rows = self.cells()
        for row in rows:
            if row["fit_seed"] == 11:
                row.update(status="fit_unavailable", utility=None)
        result = publisher.summarize(rows)
        for row in result["per_group"]:
            self.assertIsNone(row["complete_median_of_fit_medians"])
            self.assertEqual(row["complete_fit_seeds"], 4)
            self.assertEqual(row["cell_status_counts"]["fit_unavailable"], 3)
            self.assertEqual(row["partial_median_of_complete_fit_medians"], 0.8)

    def test_noninformative_sample_prevents_complete_estimate(self):
        rows = self.cells()
        rows[0]["utility"]["catboost"]["informative"] = False
        result = publisher.summarize(rows)
        affected = next(
            r
            for r in result["per_group"]
            if r["auditor"] == "catboost" and r["size_multiplier"] == 1
        )
        self.assertIsNone(affected["complete_median_of_fit_medians"])
        self.assertEqual(affected["complete_fit_seeds"], 4)

    def test_missing_or_duplicate_cell_is_rejected(self):
        rows = self.cells()
        for changed in (rows[:-1], rows + [copy.deepcopy(rows[0])]):
            with self.assertRaises(ValueError):
                publisher.summarize(changed)

    def test_publication_cannot_accept_gated_scores(self):
        for changed in (
            {"official_tests_opened": True, "mfs_v2": None, "ptf_v1": None},
            {"official_tests_opened": False, "mfs_v2": 0.99, "ptf_v1": None},
            {"official_tests_opened": False, "mfs_v2": None, "ptf_v1": 0.99},
        ):
            with self.assertRaises(ValueError):
                publisher.sealed(changed)

    def test_wider_sample_schedule_requires_every_identity_once(self):
        new, prior = [], []
        for dataset in publisher.DATASETS:
            for method, track in publisher.COMPARATOR_CONFIGS[dataset]:
                for fit_seed in publisher.FIT_SEEDS:
                    fit = dict(
                        dataset=dataset, method=method, track=track, fit_seed=fit_seed
                    )
                    for sample_seed in publisher.SAMPLE_SEEDS:
                        for size in publisher.SIZES:
                            job = dict(sample_seed=sample_seed, size_multiplier=size)
                            if fit_seed == 23:
                                job.update(fit)
                            else:
                                job["fit_job"] = fit
                            row = {
                                "job": job,
                                "status": (
                                    "fit_unavailable"
                                    if (dataset, method, track)
                                    == ("Adult", "CTGAN", "default")
                                    else "ok"
                                ),
                            }
                            (prior if fit_seed == 23 else new).append(row)
        self.assertEqual(
            publisher.sample_matrix_counts(new, prior),
            {"ok": 300, "fit_unavailable": 60},
        )
        for changed in (prior[:-1], prior + [copy.deepcopy(prior[0])]):
            with self.assertRaises(ValueError):
                publisher.sample_matrix_counts(new, changed)


if __name__ == "__main__":
    unittest.main()
