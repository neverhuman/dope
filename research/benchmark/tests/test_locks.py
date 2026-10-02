import json
import hashlib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class LockTests(unittest.TestCase):
    def test_method_inventory_and_incomplete_final_lock(self):
        lock = json.loads((ROOT / "methods.lock.json").read_text())
        self.assertFalse(lock["complete"])
        self.assertEqual(len(lock["methods"]), 29)
        adapter_hash = hashlib.sha256((ROOT / "adapters.py").read_bytes()).hexdigest()
        for name, method in lock["methods"].items():
            if method["status"] == "locked":
                for key in ("upstream_url", "source_sha256", "license", "license_evidence",
                            "dependency_or_container_digest", "default_config",
                            "tuning_search_space", "fit_command", "sampling_command"):
                    self.assertIsNotNone(method.get(key), (name, key))
                expected = (hashlib.sha256((ROOT / "sdv_adapter.py").read_bytes()).hexdigest()
                            if name in ("CTGAN", "TVAE") else adapter_hash)
                self.assertEqual(method["adapter_sha256"], expected, name)

    def test_pilot_manifest_has_three_distinct_rights_recorded_sources(self):
        lock = json.loads((ROOT / "pilot-datasets.lock.json").read_text())
        self.assertEqual([row["id"] for row in lock["datasets"]],
                         ["Adult", "California", "News"])
        self.assertTrue(all(row["license"]["status"] == "recorded" for row in lock["datasets"]))
        self.assertEqual(len({row["source_row_hash"] for row in lock["datasets"]}), 3)
        self.assertTrue(all(row["transformation_sha256"] and row["source_archive_sha256"]
                            for row in lock["datasets"]))

    def test_forest_pilot_lock_is_bound_without_full_admission(self):
        methods = json.loads((ROOT / "methods.lock.json").read_text())
        method = methods["methods"]["ForestDiffusion/Forest-Flow"]
        source = ROOT / "forestdiffusion-source.lock.json"
        self.assertEqual(method["status"], "pilot_locked")
        self.assertEqual(method["source_lock_sha256"], hashlib.sha256(source.read_bytes()).hexdigest())
        lock = json.loads(source.read_text())
        self.assertEqual(method["adapter_sha256"], lock["adapter_source_sha256"])
        self.assertEqual(method["dependency_or_container_digest"], lock["gpu_runtime"]["sha256"])
        objective = method["native_objective"]
        self.assertEqual(objective["status"], "locked")
        self.assertEqual(objective["implementation_sha256"],
                         hashlib.sha256((ROOT / "forestdiffusion_adapter.py").read_bytes()).hexdigest())
        for key in ("classification", "regression", "direction", "sha256", "url"):
            self.assertEqual(objective[key], lock["native_objective"][key])
        self.assertEqual(objective["source_sha256"], objective["sha256"])
        self.assertEqual(objective["auditor_fit_seeds"], list(range(5)))
        self.assertEqual(objective["native_sample_seed"], 101)
        self.assertFalse(objective["native_values_cross_method_ranking"])
        self.assertFalse(objective["shared_kpi_used_for_selection"])
        self.assertEqual(objective["tie_breaks"],
                         ["artifact_bytes_ascending", "config_sha256_ascending"])
        for evidence in method["pilot_evidence"]:
            path = ROOT.parents[1] / evidence["path"]
            self.assertEqual(evidence["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertFalse(methods["complete"])
        self.assertFalse(methods["frozen_for_final_evaluation"])
        self.assertFalse(lock["full_matrix_admission"])


if __name__ == "__main__":
    unittest.main()
