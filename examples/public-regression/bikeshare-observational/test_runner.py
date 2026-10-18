"""Source admission and entity/time leakage fixtures for the observational example."""
import json
from pathlib import Path
import unittest

import run

HERE = Path(__file__).resolve().parent


class BikeObservationFixtures(unittest.TestCase):
    def test_public_source_manifest_has_explicit_license_and_exact_pins(self):
        manifest = json.loads((HERE / "source-manifest.json").read_text())
        self.assertEqual(manifest["format"], "dope-public-observation-source-v1")
        self.assertEqual([item["uci_dataset_id"] for item in manifest["sources"]], [275, 560])
        for source in manifest["sources"]:
            self.assertEqual(source["license"], "CC-BY-4.0")
            for field in ("archive_sha256", "member_sha256"):
                self.assertRegex(source[field], r"^[0-9a-f]{64}$")
            self.assertTrue(source["citation"])
            self.assertTrue(source["operational_population"])
            key = "capital" if source["uci_dataset_id"] == 275 else "seoul"
            self.assertEqual(source["archive_sha256"], run.ARCHIVES[key][1])
            self.assertEqual(source["member"], run.ARCHIVES[key][2])
            self.assertEqual(source["member_sha256"], run.ARCHIVES[key][3])
            self.assertEqual(source["archive_bytes"], run.SOURCE_SIZES[key]["archive"])
            self.assertEqual(source["member_bytes"], run.SOURCE_SIZES[key]["member"])

    def test_protocol_keeps_target_and_source_identity_out_of_predictors(self):
        protocol = json.loads((HERE / "protocol.json").read_text())
        self.assertEqual(protocol["status"], "development_extension_frozen_before_comparator_execution")
        self.assertIn("previously used in an unqualified DOPE-only observational run", protocol["exposure_disclosure"])
        self.assertEqual(protocol["entity_and_time_policy"]["entity"], "service date; all hourly rows for one date stay in one partition")
        self.assertEqual(protocol["prepared_features"], list(run.FEATURES))
        self.assertNotIn("rental_count", protocol["prepared_features"])
        for forbidden in ("cnt", "casual", "registered", "instant", "service_date"):
            self.assertNotIn(forbidden, protocol["prepared_features"])
        self.assertEqual(protocol["selection"]["rows_consumed_for_selection"], 0)

    def test_ordered_day_partitions_are_disjoint_and_keep_whole_entities(self):
        rows = []
        for day in ("2017-01-01", "2017-01-02", "2017-01-03", "2017-01-04", "2017-01-05"):
            rows.extend([{"entity": day, "x": [1.0], "y": 2.0},
                         {"entity": day, "x": [2.0], "y": 3.0}])
        fit, select, test = run.split_dates(rows, (.4, .2, .4))
        groups = [{row["entity"] for row in part} for part in (fit, select, test)]
        self.assertEqual(groups, [{"2017-01-01", "2017-01-02"}, {"2017-01-03"}, {"2017-01-04", "2017-01-05"}])
        self.assertFalse(groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
        for part in (fit, select, test):
            self.assertEqual(len(part), 2 * len({row["entity"] for row in part}))

    def test_split_refuses_invalid_fractions_or_empty_partitions(self):
        rows = [{"entity": "one"}, {"entity": "two"}]
        for proportions in ((.5, .4), (.5, .5, 0.0), (1.0, 0.0)):
            with self.subTest(proportions=proportions), self.assertRaises(ValueError):
                run.split_dates(rows, proportions)

    def test_two_caps_and_fixed_fit_only_recipe_are_frozen(self):
        protocol = json.loads((HERE / "protocol.json").read_text())
        self.assertEqual(protocol["complete_artifact_byte_caps"], [16384, 32768])
        self.assertEqual(protocol["sample_sizes"], ["n=number of Seoul fit rows", "4n"])
        self.assertEqual(protocol["control_arms"]["gaussian_copula"]["artifact_byte_caps"], [16384, 32768])
        self.assertEqual(protocol["control_arms"]["compressed_row_subsample"]["sample_sizes"], ["n", "4n"])
        self.assertEqual(protocol["source_holdout"]["fit_source"], "Capital Bikeshare, Washington DC, UCI dataset 275")
        self.assertEqual(protocol["source_holdout"]["confirmation_source"], "Seoul Bike Sharing System, UCI dataset 560")


    def test_comparator_artifacts_are_deterministic_bounded_and_fit_only(self):
        import numpy as np
        rows=np.asarray([[i/200.0, (i%7)/6.0, (i*i%13)/12.0] for i in range(200)],dtype=float)
        for cap in (16384,32768):
            copula=run.gaussian_copula_artifact(rows,cap)
            self.assertLessEqual(len(copula),cap)
            self.assertEqual(copula,run.gaussian_copula_artifact(rows,cap))
            sample=run.gaussian_copula_sample(copula,32)
            self.assertEqual(sample.shape,(32,3))
            self.assertTrue(np.isfinite(sample).all())
            compressed=run.compressed_row_artifact(rows,cap)
            self.assertLessEqual(len(compressed),cap)
            self.assertEqual(compressed,run.compressed_row_artifact(rows,cap))
            copied=run.compressed_row_sample(compressed,32)
            self.assertEqual(copied.shape,(32,3))
            self.assertTrue(all(any(np.array_equal(row,fit) for fit in rows) for row in copied))

    def test_cluster_bootstrap_is_seeded_and_keeps_day_blocks(self):
        null = [5.0, 5.0, 9.0, 9.0]
        real = [1.0, 1.0, 2.0, 2.0]
        days = ["day-a", "day-a", "day-b", "day-b"]
        first = run.gain_lcb(null, real, days, replicates=100)
        second = run.gain_lcb(null, real, days, replicates=100)
        self.assertEqual(first, second)
        self.assertGreater(first, 0)


if __name__ == "__main__":
    unittest.main()
