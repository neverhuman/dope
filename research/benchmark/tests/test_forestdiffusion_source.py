"""Source receipts are a pilot lock and cannot admit or certify a full matrix."""

import copy
import json
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator

from research.benchmark.publish_forestdiffusion_source import native_mean
from research.benchmark.score import sha256

ROOT = Path(__file__).parents[1]


class ForestSourceTests(unittest.TestCase):
    def test_pinned_package_and_pilot_scope(self):
        lock = json.loads((ROOT / "forestdiffusion-source.lock.json").read_text())
        schema = json.loads((ROOT / "forestdiffusion-source.lock.schema.json").read_text())
        validator = Draft202012Validator(schema)
        validator.validate(lock)
        self.assertEqual(lock["adapter_source_sha256"], sha256(ROOT / "forestdiffusion_adapter.py"))
        self.assertEqual(lock["license_scope"], "original Python package only")
        self.assertEqual(len(lock["author_package_files"]), 7)
        self.assertEqual(lock["generated_contract_costs"]["fits"], 4)
        self.assertFalse(lock["independent_generator_implementation"])
        self.assertFalse(lock["historical_aggregate_resource_admission_complete"])
        for key, bad in (("full_matrix_admission", True), ("official_tests_opened", True),
                         ("mfs_v2", .99), ("ptf_v1", .99), ("release_safe", .99),
                         ("reported_experiment_reproductions_claimed", 2),
                         ("total_trial_cap_per_dataset", 9)):
            changed = copy.deepcopy(lock)
            changed[key] = bad
            self.assertFalse(validator.is_valid(changed), key)

    def test_native_mean_uses_all_twenty_scores_without_common_kpi(self):
        scores = {name: [.1, .2, .3, .4, .5] for name in
                  ("linear", "adaboost", "random_forest", "xgboost")}
        kpi = {"direction": "maximize", "shared_kpi_used_for_selection": False,
               "auditor_fit_seeds": list(range(5)), "auditor_seed_scores": scores, "value": .3}
        native_mean(kpi)
        for patch in ({"value": .9}, {"shared_kpi_used_for_selection": True},
                      {"auditor_fit_seeds": [0]}, {"direction": "minimize"}):
            with self.assertRaises(ValueError):
                native_mean({**kpi, **patch})


if __name__ == "__main__":
    unittest.main()
