"""Native selection and complete TabPC evidence stay separate from utility."""

import copy
import unittest

from research.benchmark import publish_tabpc_native as publisher


class TabPCPublicationTests(unittest.TestCase):
    def native_trials(self):
        return [
            {
                "job": {
                    "trial": i,
                    "config": {"num_units": i + 1},
                    "validation_sha256": "f" * 64,
                },
                "status": "ok",
                "native_kpi": {
                    "direction": "minimize",
                    "objective": "author_transformed_validation_mean_nll",
                    "partition": "validation",
                    "preprocessing_fit_partition": "train_only",
                    "validation_sha256": "f" * 64,
                    "value": 10.0 + i,
                },
                "artifact_bytes": 30000,
                "common_retention": i / 7,
            }
            for i in range(8)
        ]

    def cells(self):
        return [
            {
                "dataset": d,
                "configuration": c,
                "sample_seed": s,
                "size_multiplier": z,
                "status": "ok",
                "artifact_bytes": 30000,
                "metrics": {
                    "utility": {"catboost": {"retention": 0.8, "informative": True}}
                },
            }
            for d in publisher.DATASETS
            for c in ("default", "tuned")
            for s in publisher.SEEDS
            for z in (1, 4)
        ]

    def test_native_likelihood_selects_against_common_utility(self):
        rows = self.native_trials()
        self.assertEqual(publisher.native_winner(rows)["job"]["trial"], 0)
        rows[1]["native_kpi"]["value"] = rows[0]["native_kpi"]["value"]
        rows[1]["artifact_bytes"] = 20000
        self.assertEqual(publisher.native_winner(rows)["job"]["trial"], 1)
        for row in rows:
            row["status"] = "failed"
        self.assertIsNone(publisher.native_winner(rows))

    def test_native_trial_loss_or_changed_validation_rejected(self):
        rows = self.native_trials()
        with self.assertRaises(ValueError):
            publisher.native_winner(rows[:-1])
        rows[0]["native_kpi"]["validation_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            publisher.native_winner(rows)

    def test_incomplete_or_noninformative_group_has_null_estimate(self):
        for status in ("fit_unavailable", "ok"):
            rows = self.cells()
            rows[0]["status"] = status
            rows[0]["metrics"]["utility"]["catboost"]["informative"] = False
            affected = publisher.summaries(rows)[0]
            self.assertIsNone(affected["median_catboost_retention"])
            self.assertEqual(affected["complete_informative_samples"], 2)
        with self.assertRaises(ValueError):
            publisher.summaries(rows[:-1])

    def test_wider_schedule_rejects_missing_and_duplicate_samples(self):
        rows = [
            {
                "job": {
                    "dataset": d,
                    "config_tracks": [c],
                    "sample_seed": s,
                    "size_multiplier": z,
                },
                "status": "ok",
            }
            for d in ("California", "News")
            for c in ("default", "tuned")
            for s in publisher.SEEDS
            for z in publisher.SIZES
        ]
        self.assertEqual(publisher.sample_counts(rows), {"ok": 48})
        for changed in (rows[:-1], rows + [copy.deepcopy(rows[0])]):
            with self.assertRaises(ValueError):
                publisher.sample_counts(changed)


if __name__ == "__main__":
    unittest.main()
