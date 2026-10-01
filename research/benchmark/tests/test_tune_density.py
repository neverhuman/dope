import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from research.benchmark import adapters, runner, tune_density
from research.benchmark.score import sha256


ROOT = Path(__file__).resolve().parents[3]
METHODS = ROOT / "research/benchmark/methods.lock.json"


class TuneDensityTests(unittest.TestCase):
    def test_cpu_affinity_stays_inside_host_allocation(self):
        with patch.object(tune_density.os, "sched_getaffinity", return_value={2, 4, 6, 8}):
            self.assertEqual(tune_density.cpu_affinity(), (2, 4, 6, 8))
        with patch.object(tune_density.os, "sched_getaffinity", return_value=set(range(40))):
            self.assertEqual(tune_density.cpu_affinity(), tuple(range(16, 32)))

    def test_frozen_training_validation_identity_and_native_winner(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "target") as directory:
            root = Path(directory)
            worker = root / "worker"
            worker.mkdir()
            (worker / "train.csv").write_text(
                "0.1,0.2\n0.2,0.3\n0.3,0.4\n0.4,0.5\n0.5,0.6\n0.6,0.7\n")
            (worker / "validation.csv").write_text("0.15,0.25\n0.45,0.55\n")
            (worker / "projection.json").write_text(json.dumps({"task": "regression"}))
            manifest = {"dataset_id": "fixture", "projected_hashes": {
                name: sha256(worker / f"{name}.csv") for name in ("train", "validation")},
                "projection_sha256": sha256(worker / "projection.json")}
            (worker / "worker-manifest.json").write_text(json.dumps(manifest))
            entry = json.loads(METHODS.read_text())["methods"]["independent_marginals"]
            job = {"dataset": "fixture", "method": "independent_marginals",
                   "stage": "discovery", "configurations": tune_density.candidates(entry),
                   "fit_seed": 11, "budget_seconds": tune_density.BUDGET_SECONDS,
                   "train_sha256": manifest["projected_hashes"]["train"],
                   "validation_sha256": manifest["projected_hashes"]["validation"],
                   "projection_sha256": manifest["projection_sha256"]}
            lock = {"format": "dope-benchmark-compact-native-round",
                    "method_lock_sha256": sha256(METHODS),
                    "adapter_sha256": sha256(Path(adapters.__file__)),
                    "tuner_sha256": sha256(Path(tune_density.__file__)), "jobs": [job]}
            lock_path = root / "round.json"
            lock_path.write_text(json.dumps(lock))
            with patch.object(tune_density, "SCRATCH", root):
                result = tune_density.tune(worker, "independent_marginals", METHODS,
                                           lock_path, root / "results")
                self.assertEqual(tune_density.tune(worker, "independent_marginals",
                                                   METHODS, lock_path, root / "results"), result)
                selection = json.loads(Path(result["selection_path"]).read_text())
                self.assertEqual(len(selection["trials"]), 3)
                self.assertTrue(all(trial["status"] == "ok"
                                    for trial in selection["trials"]))
                config, _ = runner.resolve_configuration(
                    {"final": True, "method": "independent_marginals",
                     "configuration": {"kind": "tuned", "values": selection["selected_config"],
                                       "selection_path": result["selection_path"],
                                       "selection_sha256": result["selection_sha256"]}},
                    entry, "fixture", 6, root)
                self.assertEqual(config, selection["selected_config"])
                (worker / "validation.csv").write_text("0.9,0.9\n")
                with self.assertRaisesRegex(ValueError, "worker partition digest mismatch"):
                    tune_density.tune(worker, "independent_marginals", METHODS,
                                      lock_path, root / "results")


if __name__ == "__main__":
    unittest.main()
