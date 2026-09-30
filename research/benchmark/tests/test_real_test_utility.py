import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from research.benchmark import real_test_utility
from research.benchmark.real_test_utility import evaluate, retention
from research.benchmark.score import sha256


ROOT = Path(__file__).resolve().parents[3]


class RealTestUtilityTests(unittest.TestCase):
    def test_null_normalized_retention_and_low_signal(self):
        measured = retention(1.0, 0.5, 0.75)
        self.assertEqual(measured["retention"], 0.5)
        self.assertTrue(measured["informative"])
        low_signal = retention(1.0, 0.999, 1.005)
        self.assertIsNone(low_signal["retention"])
        self.assertTrue(low_signal["low_signal_noninferior"])

    def test_test_partition_is_not_opened_before_admission(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "target") as directory:
            missing = Path(directory) / "missing"
            with self.assertRaisesRegex(ValueError, "admission blocked"):
                evaluate(ROOT, Path(directory), missing, missing, {}, missing / "out.json")

    def test_real_test_cells_use_sample_receipts_and_resumable_output(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "target") as directory:
            root = Path(directory)
            worker, evaluator = root / "worker", root / "evaluator"
            worker.mkdir()
            evaluator.mkdir()
            (worker / "projection.json").write_text(json.dumps({"task": "binary"}))
            def write_rows(path, count):
                with path.open("w") as stream:
                    for index in range(count):
                        stream.write(f"{index % 10 / 10},{int(index % 10 >= 5)}\n")
            write_rows(worker / "train.csv", 30)
            write_rows(evaluator / "test.csv", 20)
            projection_hash = sha256(worker / "projection.json")
            (worker / "worker-manifest.json").write_text(json.dumps({
                "dataset_id": "toy", "projection_sha256": projection_hash,
                "train_rows": 30}))
            (evaluator / "manifest.json").write_text(json.dumps({
                "id": "toy", "projection_sha256": projection_hash,
                "projected_files": {"train": sha256(worker / "train.csv"),
                                    "test": sha256(evaluator / "test.csv")}}))
            samples = {}
            for multiplier in (1, 4):
                sample = root / f"sample-{multiplier}.csv"
                receipt = root / f"receipt-{multiplier}.json"
                write_rows(sample, 30 * multiplier)
                receipt.write_text(json.dumps({"status": "ok", "fit_key": "same-fit",
                                               "sample_seed": 101, "row_count": 30 * multiplier,
                                               "sample_sha256": sha256(sample)}))
                samples[multiplier] = sample, receipt
            output = root / "utility.json"
            with patch.object(real_test_utility, "assess", return_value={"admitted": True}), \
                 patch.object(real_test_utility, "ROOT", root):
                first = evaluate(ROOT, root, worker, evaluator, samples, output)
                second = evaluate(ROOT, root, worker, evaluator, samples, output)
            self.assertEqual(first, second)
            self.assertEqual(first["fit_key"], "same-fit")
            self.assertEqual(set(first["auditors"]), {"1", "4"})
            self.assertEqual(first["test_sha256"], sha256(evaluator / "test.csv"))


if __name__ == "__main__":
    unittest.main()
