import json
import tempfile
import unittest
from pathlib import Path

from research.benchmark.admission import assess


ROOT = Path(__file__).resolve().parents[3]


class AdmissionTests(unittest.TestCase):
    def test_missing_final_locks_keep_test_closed(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "target") as directory:
            result = assess(ROOT, Path(directory))
            self.assertFalse(result["admitted"])
            self.assertIn("budget.lock.json:missing_or_invalid", result["blockers"])

    def test_incomplete_method_lock_keeps_test_closed(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "target") as directory:
            lock_root = Path(directory)
            (lock_root / "methods.lock.json").write_text(json.dumps({
                "complete": False, "frozen_for_final_evaluation": False,
                "methods": {"dope": {"status": "locked"}}}))
            result = assess(ROOT, lock_root)
            self.assertIn("methods.lock.json:not_frozen", result["blockers"])
            self.assertIn("methods.lock.json:source_or_config_gap", result["blockers"])

    def test_complete_locks_admit_only_locked_applicable_cells(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "target") as directory:
            lock_root = Path(directory)
            locks = {
                "methods.lock.json": {"methods": {"dope": {
                    "status": "locked", "source_sha256": "source", "adapter_sha256": "adapter",
                    "default_config": {}, "tuning_search_space": {},
                    "dependency_or_container_digest": "dependency", "license": "MIT"}}},
                "datasets.lock.json": {"datasets": [{"license": {"status": "recorded"},
                    "source_row_hash": "rows", "split": {"hashes": {"train": "hash"}}}]},
                "budget.lock.json": {"campaign_ceiling_days": 14, "pilot_receipts_sha256": "receipt"},
                "evaluator.lock.json": {"metric_implementation_sha256": "metric"},
                "method-dataset-matrix.lock.json": {"cells": [{
                    "dataset": "toy", "method": "dope", "panel": "public_core",
                    "track": "common_numeric", "tier": "l3", "applicable": True}]},
            }
            for name, body in locks.items():
                body.update({"complete": True, "frozen_for_final_evaluation": True})
                (lock_root / name).write_text(json.dumps(body))
            self.assertTrue(assess(ROOT, lock_root)["admitted"])
            locks["method-dataset-matrix.lock.json"]["cells"][0]["method"] = "unavailable"
            (lock_root / "method-dataset-matrix.lock.json").write_text(
                json.dumps(locks["method-dataset-matrix.lock.json"]))
            self.assertIn("method-dataset-matrix.lock.json:cell_gap",
                          assess(ROOT, lock_root)["blockers"])


if __name__ == "__main__":
    unittest.main()
