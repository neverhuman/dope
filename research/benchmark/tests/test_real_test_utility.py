import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from research.benchmark import real_test_utility
from research.benchmark.manifest import digest
from research.benchmark.real_test_utility import evaluate, retention
from research.benchmark.score import artifact_inventory, sha256


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
                "train_rows": 30, "split_hashes": {"train": "train-split", "validation": "valid-split"}}))
            manifest = {
                "id": "toy", "projection_sha256": projection_hash,
                "source_row_hash": "source-rows",
                "split": {"hashes": {"train": "train-split", "validation": "valid-split",
                                     "test": "test-split"}},
                "projected_files": {"train": sha256(worker / "train.csv"),
                                    "test": sha256(evaluator / "test.csv")}}
            (evaluator / "manifest.json").write_text(json.dumps(manifest))
            (root / "datasets.lock.json").write_text(json.dumps({"datasets": [manifest]}))
            fit_identity = {"dataset": "toy", "projection": projection_hash,
                            "split": {"train": "train-split", "validation": "valid-split"},
                            "sample_seeds": [101, 211, 307],
                            "size_multipliers": [1, 2, 4, 8]}
            fit_key = digest(fit_identity)
            fit_dir = root / fit_key
            artifact = fit_dir / "artifact"
            artifact.mkdir(parents=True)
            (artifact / "model.bin").write_bytes(b"fitted")
            inventory, artifact_bytes = artifact_inventory(artifact, ["model.bin"])
            (fit_dir / "fit-receipt.json").write_text(json.dumps({
                "status": "ok", "fit_key": fit_key, "fit_identity": fit_identity,
                "artifact_files": ["model.bin"], "artifact_inventory": inventory,
                "artifact_bytes": artifact_bytes}))
            samples = {}
            for multiplier in (1, 4):
                run_key = digest({**fit_identity, "sample_seed": 101,
                                  "row_count": 30 * multiplier})
                sample = fit_dir / f"{run_key}.csv"
                receipt = fit_dir / f"{run_key}.receipt.json"
                write_rows(sample, 30 * multiplier)
                receipt.write_text(json.dumps({"status": "ok", "fit_key": fit_key,
                                               "run_key": run_key,
                                               "sample_seed": 101, "row_count": 30 * multiplier,
                                               "artifact_sampling_verified": True,
                                               "sample_sha256": sha256(sample)}))
                samples[multiplier] = sample, receipt
            output = root / "utility.json"
            with patch.object(real_test_utility, "assess", return_value={"admitted": True}), \
                 patch.object(real_test_utility, "ROOT", root):
                altered = dict(manifest)
                altered["source_row_hash"] = "other-source"
                (root / "datasets.lock.json").write_text(json.dumps({"datasets": [altered]}))
                with self.assertRaisesRegex(ValueError, "frozen dataset lock"):
                    evaluate(ROOT, root, worker, evaluator, samples, output)
                (root / "datasets.lock.json").write_text(json.dumps({"datasets": [manifest]}))
                first = evaluate(ROOT, root, worker, evaluator, samples, output)
                second = evaluate(ROOT, root, worker, evaluator, samples, output)
                receipt_path = samples[4][1]
                changed = json.loads(receipt_path.read_text())
                changed["run_key"] = "wrong-run"
                receipt_path.write_text(json.dumps(changed))
                with self.assertRaisesRegex(ValueError, "frozen fit and artifact"):
                    evaluate(ROOT, root, worker, evaluator, samples, output)
            self.assertEqual(first, second)
            self.assertEqual(first["fit_key"], fit_key)
            self.assertEqual(set(first["auditors"]), {"1", "4"})
            self.assertEqual(first["test_sha256"], sha256(evaluator / "test.csv"))


if __name__ == "__main__":
    unittest.main()
