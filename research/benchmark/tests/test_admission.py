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
                    "dependency_or_container_digest": "dependency", "license": "MIT",
                    "license_evidence": "https://example.com/license",
                    "native_objective": {"status": "locked", "name": "validation_ptf_v1",
                        "direction": "maximize", "implementation_sha256": "metric",
                        "tie_breaks": ["artifact_bytes_ascending", "config_sha256_ascending"]},
                    "fit_command": "fit", "sampling_command": "sample"}}},
                "datasets.lock.json": {"datasets": [{"id": "toy",
                    "license": {"status": "recorded"}, "source_row_hash": "rows",
                    "projection_sha256": "projection",
                    "split": {"hashes": {"train": "a", "validation": "b", "test": "c"}},
                    "projected_files": {"train": "a", "validation": "b", "test": "c"}}]},
                "budget.lock.json": {"reporting_milestone_days": 14,
                                     "scratch_ceiling_bytes": 200_000_000_000,
                                     "tuning_trials_per_method_dataset": 8,
                                     "tuning_wall_time_hours_per_method_dataset": 12,
                                     "pilot_receipts_sha256": "receipt"},
                "evaluator.lock.json": {"metric_implementation_sha256": "metric"},
                "method-dataset-matrix.lock.json": {"cells": [{
                    "id": "toy-dope-default-11", "dataset": "toy", "method": "dope",
                    "panel": "public_core", "track": "common_numeric", "tier": "l3",
                    "applicable": True, "fit_seed": 11, "sample_seeds": [101, 211, 307],
                    "size_multipliers": [1, 2, 4, 8], "worker_dir": "/mnt/fast-scratch/dope-benchmark/toy",
                    "scratch_reservation_bytes": 1000, "memory_reservation_bytes": 1000,
                    "requires_gpu": False, "gpu_vram_mib": 0,
                    "timeout_seconds": 600, "runtime_python": "/usr/bin/python3",
                    "configuration": {"kind": "default"}}]},
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
