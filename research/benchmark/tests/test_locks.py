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
                self.assertEqual(method["adapter_sha256"], adapter_hash, name)

    def test_pilot_manifest_has_three_distinct_rights_recorded_sources(self):
        lock = json.loads((ROOT / "pilot-datasets.lock.json").read_text())
        self.assertEqual([row["id"] for row in lock["datasets"]],
                         ["Adult", "California", "News"])
        self.assertTrue(all(row["license"]["status"] == "recorded" for row in lock["datasets"]))
        self.assertEqual(len({row["source_row_hash"] for row in lock["datasets"]}), 3)
        self.assertTrue(all(row["transformation_sha256"] and row["source_archive_sha256"]
                            for row in lock["datasets"]))


if __name__ == "__main__":
    unittest.main()
