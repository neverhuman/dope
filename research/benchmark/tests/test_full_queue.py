import tempfile
import unittest
from pathlib import Path

from research.benchmark.full_queue import (available_slot, cpu_slots, latest_attempt,
                                           verify_success, write_once)
from research.benchmark.score import artifact_inventory, sha256


ROOT = Path(__file__).resolve().parents[3]


class FullQueueTests(unittest.TestCase):
    def snapshot(self):
        return {"allowed_cpus": list(range(64)), "load_average": [8.0, 8.0, 8.0],
                "memory": {"MemAvailable": 40_000_000_000},
                "gpus": [{"index": 0, "memory_free_mib": 23000}],
                "active_gpu_processes": []}

    def cell(self, gpu=False):
        return {"memory_reservation_bytes": 8_000_000_000,
                "requires_gpu": gpu, "gpu_vram_mib": 12000}

    def test_four_disjoint_cpu_slots(self):
        slots = cpu_slots(self.snapshot())
        self.assertEqual(len(slots), 4)
        self.assertEqual(len(set().union(*map(set, slots))), 64)
        self.assertEqual(available_slot(self.snapshot(), {slots[0], slots[1]},
                                        self.cell()), slots[2])

    def test_gpu_and_memory_admission_respect_other_work(self):
        snapshot = self.snapshot()
        snapshot["active_gpu_processes"] = ["other-owner"]
        self.assertIsNone(available_slot(snapshot, set(), self.cell(gpu=True)))
        self.assertIsNotNone(available_slot(snapshot, set(), self.cell()))
        snapshot["active_gpu_processes"] = []
        snapshot["memory"]["MemAvailable"] = 5_000_000_000
        self.assertIsNone(available_slot(snapshot, set(), self.cell()))
        snapshot["memory"]["MemAvailable"] = 40_000_000_000
        snapshot["gpus"][0]["memory_free_mib"] = 12000
        self.assertIsNone(available_slot(snapshot, set(), self.cell(gpu=True)))

    def test_attempt_receipts_are_immutable_and_latest_is_used_for_resume(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "target") as directory:
            attempts = Path(directory)
            first = attempts / "attempt-0001.json"
            second = attempts / "attempt-0002.json"
            write_once(first, {"status": "failed"})
            write_once(second, {"status": "ok"})
            self.assertEqual(latest_attempt(attempts)["status"], "ok")
            with self.assertRaises(FileExistsError):
                write_once(first, {"status": "ok"})

    def test_success_resume_rechecks_cross_host_artifact_and_sample_hashes(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "target") as directory:
            root = Path(directory)
            fit_dir = root / "results" / "fit-key"
            artifact = fit_dir / "artifact"
            artifact.mkdir(parents=True)
            (artifact / "model.bin").write_bytes(b"artifact")
            inventory, _ = artifact_inventory(artifact, ["model.bin"])
            write_once(fit_dir / "fit-receipt.json", {"status": "ok",
                                                     "artifact_files": ["model.bin"],
                                                     "artifact_inventory": inventory})
            sample = fit_dir / "sample-key.csv"
            sample.write_text("0,1\n")
            child = {"status": "ok", "fit_key": "fit-key", "run_key": "sample-key",
                     "sample_sha256": sha256(sample)}
            write_once(fit_dir / "sample-key.receipt.json", child)
            invent = root / "inventories" / "cell-attempt-0001.json"
            log = root / "logs" / "cell-attempt-0001.log"
            invent.parent.mkdir()
            log.parent.mkdir()
            invent.write_text("{}")
            log.write_text("completed")
            receipt = {"status": "ok", "cell_id": "cell", "attempt": 1,
                       "inventory_sha256": sha256(invent), "log_sha256": sha256(log),
                       "runner_receipts": [child]}
            verify_success(receipt, root)
            sample.write_text("1,0\n")
            with self.assertRaisesRegex(ValueError, "sample or receipt changed"):
                verify_success(receipt, root)


if __name__ == "__main__":
    unittest.main()
