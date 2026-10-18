"""The E03 scientific pilot stays fail-closed until its external gates clear."""

from __future__ import annotations

import json
import unittest
from pathlib import Path


MANIFEST = Path(__file__).parents[1] / "manifests" / "e03-pilot.json"


class E03PilotManifestTests(unittest.TestCase):
    def test_prelaunch_block_is_explicit_and_panel_cannot_pass(self):
        manifest = json.loads(MANIFEST.read_text())
        self.assertEqual(manifest["format"], "dope-scientific-pilot-manifest-v1")
        self.assertEqual(manifest["status"], "blocked_prelaunch")
        self.assertFalse(manifest["execution_authorized"])
        self.assertIsNone(manifest["launch_command"])
        self.assertFalse(manifest["scientific_admission"])
        self.assertFalse(manifest["panel_pass"])
        self.assertTrue(manifest["preflight_must_pass_before_native_launch"])
        self.assertTrue(all(value == "pending" for value in manifest["gates"].values()))

    def test_smoke_preserves_both_required_comparators_and_full_trace(self):
        manifest = json.loads(MANIFEST.read_text())
        smoke = manifest["smoke"]
        self.assertEqual(smoke["observational_tasks"], 1)
        self.assertEqual(smoke["adverse_mechanism_tasks"], 1)
        self.assertEqual(smoke["generators"], ["dope", "gaussian_copula_control"])
        self.assertEqual(smoke["budgets"], ["small", "large"])
        self.assertEqual(smoke["required_non_generator_control"], "compressed_real_row_reference")
        self.assertEqual(smoke["required_trace"], [
            "prepare", "compile", "reload", "source_free_sample",
            "real_null_synthetic_models", "driver_analysis", "rust_receipt", "report",
        ])

    def test_price_is_not_invented_without_observed_runtime(self):
        manifest = json.loads(MANIFEST.read_text())
        price = manifest["quick12_price"]
        self.assertEqual((price["tasks"], price["budgets_per_task"], price["fits_per_cell"]), (12, 3, 1))
        self.assertIsNone(price["observed_time_seconds"])
        self.assertIsNone(price["uncertainty_and_driver_privacy_cost"])
        self.assertEqual(price["status"], "not_estimable_before_admitted_smoke")


if __name__ == "__main__":
    unittest.main()
