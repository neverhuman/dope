import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from research.benchmark import adapters
from research.benchmark import runner
from research.benchmark.manifest import prepare
from research.benchmark.runner import run
from research.benchmark.score import sha256


ROOT = Path(__file__).resolve().parents[3]


class RunnerTests(unittest.TestCase):
    def test_scratch_reservation_counts_existing_files(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "target") as directory:
            root = Path(directory)
            (root / "existing.bin").write_bytes(b"x" * 200)
            with patch.object(runner, "LIMIT", 1_000_000_250):
                with self.assertRaisesRegex(ValueError, "scratch ceiling"):
                    runner.reserve_scratch(root, 2, 2)

    def test_resumed_job_reproduces_receipts(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "target") as directory:
            root = Path(directory)
            source = root / "source.csv"
            with source.open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["x", "y"])
                writer.writerows([[i / 40, i % 2] for i in range(40)])
            entry = {"id": "toy", "panel": "public_core", "source": "local-fixture",
                     "source_identity": "fixture-toy", "license": {"status": "recorded",
                     "spdx": "CC0-1.0", "evidence_url": "https://example.com/license"},
                     "task": "binary", "target": "y", "raw": str(source)}
            prepare(entry, root / "prepared")
            methods = {"methods": {"independent_marginals": {"status": "locked",
                       "adapter": "independent_marginals", "source_sha256": sha256(Path(adapters.__file__)),
                       "adapter_sha256": sha256(Path(adapters.__file__)),
                       "dependency_or_container_digest": "numpy==" + __import__("numpy").__version__,
                       "default_config": {"bins": 16}}}}
            job = {"method": "independent_marginals", "fit_seed": 11,
                   "track": "common_numeric", "worker_dir": str(root / "prepared/worker/toy")}
            first = run(job, methods, root / "results")
            second = run(job, methods, root / "results")
            self.assertEqual(first, second)
            self.assertEqual(len(first), 12)
            self.assertTrue(all(item["status"] == "ok" and item["artifact_sampling_verified"]
                                for item in first))
            fit_receipt = next((root / "results").glob("*/fit-receipt.json"))
            self.assertEqual(set(__import__("json").loads(fit_receipt.read_text())["artifact_files"]),
                             {"model.json", "projection.json"})


if __name__ == "__main__":
    unittest.main()
