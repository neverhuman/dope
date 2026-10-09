import json
import os
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from research.benchmark import pilot_metrics, real_test_utility
from research.benchmark.manifest import digest
from research.benchmark.real_test_utility import evaluate, retention
from research.benchmark.score import artifact_inventory, sha256


ROOT = Path(__file__).resolve().parents[3]


@contextmanager
def generated_artifact_directory(path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


class RealTestUtilityTests(unittest.TestCase):
    def test_catboost_utility_bounds_importance_and_prediction_threads(self):
        import math
        import numpy as np
        from catboost import CatBoostClassifier, CatBoostRegressor
        x = np.arange(30, dtype=float).reshape(-1, 1) / 30
        for task, cls in (("binary", CatBoostClassifier), ("regression", CatBoostRegressor)):
            with self.subTest(task=task):
                importance = cls.get_feature_importance
                prediction_name = "predict_proba" if task == "binary" else "predict"
                predict = getattr(cls, prediction_name)
                calls = {"importance": [], "prediction": []}

                def checked_importance(model, *args, **kwargs):
                    self.assertEqual(kwargs.get("thread_count"), 4)
                    self.assertEqual(model.get_param("thread_count"), 4)
                    calls["importance"].append(kwargs["thread_count"])
                    return importance(model, *args, **kwargs)

                def checked_prediction(model, *args, **kwargs):
                    self.assertEqual(kwargs.get("thread_count"), 4)
                    calls["prediction"].append(kwargs["thread_count"])
                    return predict(model, *args, **kwargs)

                y = (x[:, 0] >= .5).astype(int) if task == "binary" else x[:, 0] ** 2
                with patch.object(cls, "get_feature_importance", checked_importance), \
                        patch.object(cls, prediction_name, checked_prediction):
                    model = pilot_metrics._model("catboost", task, 1729).fit(x, y)
                    loss = pilot_metrics._loss(model, x, y, task)
                self.assertTrue(math.isfinite(loss))
                self.assertEqual(calls, {"importance": [4], "prediction": [4]})
                self.assertEqual({k: model.get_param(k) for k in
                                  ("iterations", "depth", "learning_rate", "random_seed")},
                                 {"iterations": 100, "depth": 6, "learning_rate": .05,
                                  "random_seed": 1729})

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
            with generated_artifact_directory(root), \
                 patch.object(real_test_utility, "assess", return_value={"admitted": True}), \
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
