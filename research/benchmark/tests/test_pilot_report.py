import json
import tempfile
import unittest
from pathlib import Path

from research.benchmark import pilot_queue, pilot_report


class PilotReportTests(unittest.TestCase):
    def test_repair_preserves_failed_attempt_and_counts_cell_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "receipts").mkdir()
            (root / "repair-receipts").mkdir()
            for dataset in pilot_queue.DATASETS:
                for method in pilot_queue.ROSTER:
                    for seed in pilot_queue.FIT_SEEDS:
                        name = f"{dataset}-{method}-{seed}"
                        receipt = {"dataset": dataset, "method": method,
                                   "fit_seed": seed, "host": "xbabe2",
                                   "status": "failed" if name == "Adult-GaussianCopula-11" else "blocked",
                                   "elapsed_seconds": 2.0}
                        (root / "receipts" / f"{name}.json").write_text(json.dumps(receipt))
            repaired = {"dataset": "Adult", "method": "GaussianCopula",
                        "fit_seed": 11, "host": "xbabe2", "status": "ok",
                        "elapsed_seconds": 5.0}
            (root / "logs").mkdir()
            (root / "logs" / "Adult-GaussianCopula-11.log").write_text(
                "ValueError: adapter source digest changed\n")
            (root / "repair-receipts" / "Adult-GaussianCopula-11-attempt2.json").write_text(
                json.dumps(repaired))
            jobs, ledger = pilot_report.reconcile_jobs(root)
            self.assertEqual(len(jobs), 42)
            self.assertEqual(next(job for job in jobs if job["cell"] == "Adult-GaussianCopula-11")["status"], "ok")
            self.assertEqual(next(job for job in jobs if job["cell"] == "Adult-GaussianCopula-11")["attempts"], 2)
            self.assertEqual(next(item for item in ledger if item["cell"] == "Adult-GaussianCopula-11")["status"], "failed")
            (root / "logs" / "Adult-GaussianCopula-11.log").write_text("unrelated failure\n")
            with self.assertRaisesRegex(ValueError, "not the registered package-path failure"):
                pilot_report.reconcile_jobs(root)


if __name__ == "__main__":
    unittest.main()
