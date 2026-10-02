import csv
import json
import sys
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
    def test_compact_final_runtime_digest_is_verified_before_initialization(self):
        import builtins
        import importlib.metadata
        runtime = ROOT / "research/benchmark/numpy-runtime.lock.json"
        lock = json.loads(runtime.read_text())
        import jsonschema
        schema = json.loads(runtime.with_name("numpy-runtime.schema.json").read_text())
        jsonschema.Draft202012Validator.check_schema(schema)
        jsonschema.Draft202012Validator(schema).validate(lock)
        entry = {"dependency_or_container_digest": sha256(runtime),
                 "dependency_versions": lock["versions"]}
        original_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name == "numpy" or name.startswith("numpy."):
                raise AssertionError("runtime verifier initialized NumPy")
            return original_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", guarded_import):
            runner.check_numpy_runtime(entry, final=True)
            with self.assertRaisesRegex(ValueError, "digest changed"):
                runner.check_numpy_runtime({"dependency_or_container_digest": "numpy==1.26.4"}, final=True)
            with patch.object(importlib.metadata, "version", return_value="unexpected"):
                with self.assertRaisesRegex(ValueError, "version changed"):
                    runner.check_numpy_runtime(entry, final=True)
            first = lock["files"][0]["path"]
            installed_file = Path(importlib.metadata.distribution("numpy").locate_file(first))
            original_hash = runner.sha256
            with patch.object(runner, "sha256", side_effect=lambda path:
                              "0" * 64 if Path(path) == installed_file else original_hash(path)):
                with self.assertRaisesRegex(ValueError, "source changed"):
                    runner.check_numpy_runtime(entry, final=True)

    def test_native_worker_is_killed_at_hard_deadline(self):
        with self.assertRaises(TimeoutError):
            runner._run_worker([sys.executable, "-c", "import time; time.sleep(10)"],
                               {}, 1)

    def test_pilot_locked_method_cannot_enter_final_job(self):
        methods = {"methods": {"AIM": {"status": "pilot_locked", "adapter": "AIM"}}}
        with self.assertRaisesRegex(ValueError, "source/config locked"):
            run({"method": "AIM", "fit_seed": 11, "track": "common_numeric"},
                methods, ROOT / "target" / "unused-aim-test-results")

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
            fit_receipt.rename(fit_receipt.with_suffix(".saved"))
            sample_receipt = fit_receipt.parent / f"{first[0]['run_key']}.receipt.json"
            sample_receipt.rename(sample_receipt.with_suffix(".saved"))
            self.assertEqual(run(job, methods, root / "results"), first)
            self.assertTrue(fit_receipt.exists())
            self.assertTrue(sample_receipt.exists())

    def test_frozen_tuned_configuration_selects_native_validation_winner(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "target") as directory:
            root = Path(directory)
            selected = {"bins": 32}
            objective = {"status": "locked", "name": "mean_log_density",
                         "direction": "maximize", "implementation_sha256": "metric",
                         "tie_breaks": ["artifact_bytes_ascending", "config_sha256_ascending"]}
            trial_root = root / "trial-00"
            artifact = trial_root / "artifact"
            artifact.mkdir(parents=True)
            (artifact / "model.json").write_text("{}")
            (artifact / "projection.json").write_text("{}")
            from research.benchmark.score import artifact_inventory
            inventory, charged = artifact_inventory(artifact, ["model.json", "projection.json"])
            metric = trial_root / "native-metric.json"
            metric.write_text(json.dumps({"method": "independent_marginals",
                                          "partition": "validation", "objective": "mean_log_density",
                                          "implementation_sha256": "metric", "value": 1.0,
                                          "validation_sha256": "validation",
                                          "artifact_sha256": sha256(artifact / "model.json")}))
            attempt = trial_root / "attempt.json"
            trial = {"status": "ok", "wall_seconds": 1, "config": selected,
                     "native_kpi": 1.0, "artifact_bytes": charged,
                     "artifact_inventory": inventory,
                     "artifact_sha256": sha256(artifact / "model.json"),
                     "metric_receipt_path": str(metric),
                     "metric_receipt_sha256": sha256(metric),
                     "identity": {"dataset": "toy", "method": "independent_marginals",
                                  "round_sha256": "round", "method_source_sha256": "source",
                                  "validation_sha256": "validation", "config": selected}}
            attempt.write_text(json.dumps(trial))
            trial.update({"attempt_receipt_path": str(attempt),
                          "attempt_receipt_sha256": sha256(attempt)})
            receipt = root / "selection.json"
            evidence = {"format": "dope-benchmark-validation-selection",
                        "partition": "validation", "validation_sha256": "validation",
                        "test_opened": False, "dataset": "toy",
                        "round_sha256": "round",
                        "method": "independent_marginals", "objective": objective,
                        "selected_config": selected, "selected_trial_index": 0,
                        "trials": [trial]}
            receipt.write_text(json.dumps(evidence))
            job = {"final": True, "method": "independent_marginals",
                   "configuration": {"kind": "tuned", "values": selected,
                                     "selection_path": str(receipt),
                                     "selection_sha256": sha256(receipt)}}
            entry = {"group": "compact", "default_config": {"bins": 16},
                     "source_sha256": "source",
                     "tuning_search_space": {"bins": [8, 16, 32]},
                     "native_objective": objective}
            config, identity = runner.resolve_configuration(job, entry, "toy", 100, root, "validation")
            self.assertEqual(config, selected)
            self.assertEqual(identity["selection_sha256"], sha256(receipt))
            evidence["selected_trial_index"] = 1
            receipt.write_text(json.dumps(evidence))
            job["configuration"]["selection_sha256"] = sha256(receipt)
            with self.assertRaisesRegex(ValueError, "native KPI winner"):
                runner.resolve_configuration(job, entry, "toy", 100, root, "validation")
            evidence["selected_trial_index"] = 0
            evidence["trials"][0]["attempt_receipt_sha256"] = "tampered"
            receipt.write_text(json.dumps(evidence))
            job["configuration"]["selection_sha256"] = sha256(receipt)
            with self.assertRaisesRegex(ValueError, "attempt receipt"):
                runner.resolve_configuration(job, entry, "toy", 100, root, "validation")
            evidence["trials"][0]["attempt_receipt_sha256"] = sha256(attempt)
            (artifact / "model.json").write_text('{"changed":true}')
            receipt.write_text(json.dumps(evidence))
            job["configuration"]["selection_sha256"] = sha256(receipt)
            with self.assertRaisesRegex(ValueError, "native KPI receipt"):
                runner.resolve_configuration(job, entry, "toy", 100, root, "validation")
            (artifact / "model.json").write_text("{}")
            evidence["selected_trial_index"] = 0
            evidence["trials"].clear()
            receipt.write_text(json.dumps(evidence))
            job["configuration"]["selection_sha256"] = sha256(receipt)
            with self.assertRaisesRegex(ValueError, "invalid validation selection"):
                runner.resolve_configuration(job, entry, "toy", 100, root, "validation")
            job["configuration"]["values"] = {"bins": 64}
            with self.assertRaisesRegex(ValueError, "outside locked search space"):
                runner.resolve_configuration(job, entry, "toy", 100, root, "validation")

    def test_dp_budget_is_part_of_configuration_identity(self):
        entry = {"group": "dp", "default_config": {"epsilon": 1.0, "bins": 8},
                 "tuning_search_space": {"epsilon": [1.0, 4.0, 10.0], "bins": [8]}}
        job = {"final": True, "configuration": {"kind": "default"}, "dp_epsilon": 4}
        config, identity = runner.resolve_configuration(job, entry, "toy", 100, ROOT / "target")
        self.assertEqual(config["epsilon"], 4)
        self.assertEqual(identity["dp_budget"], {"epsilon": 4, "delta": 1e-5})
        job["dp_epsilon"] = 2
        with self.assertRaisesRegex(ValueError, "outside frozen budget"):
            runner.resolve_configuration(job, entry, "toy", 100, ROOT / "target")

    def test_failed_fit_and_sample_attempts_can_retry_without_losing_receipts(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "target") as directory:
            root = Path(directory)
            source = root / "source.csv"
            with source.open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["x", "y"])
                writer.writerows([[index / 40, index % 2] for index in range(40)])
            prepare({"id": "toy", "panel": "public_core", "source": "fixture",
                     "source_identity": "fixture-toy", "task": "binary", "target": "y",
                     "raw": str(source), "license": {"status": "recorded", "spdx": "CC0-1.0",
                     "evidence_url": "https://example.com/license"}}, root / "prepared")
            methods = {"methods": {"independent_marginals": {
                "status": "locked", "adapter": "independent_marginals",
                "source_sha256": sha256(Path(adapters.__file__)),
                "adapter_sha256": sha256(Path(adapters.__file__)),
                "dependency_or_container_digest": "numpy==" + __import__("numpy").__version__,
                "default_config": {"bins": 16}}}}
            job = {"method": "independent_marginals", "fit_seed": 11,
                   "track": "common_numeric", "worker_dir": str(root / "prepared/worker/toy"),
                   "sample_seeds": [101], "size_multipliers": [1]}
            calls = {"fit": 0, "sample": 0}

            def adapter(request, _seconds):
                action = request["action"]
                calls[action] += 1
                if calls[action] == 1:
                    raise runner.AdapterFailure("FixtureFailure")
                if action == "fit":
                    (Path(request["artifact_dir"]) / "model.json").write_text("{}")
                    return {"files": ["model.json"]}
                Path(request["output"]).write_text("0,1\n" * request["row_count"])
                return {}

            with patch.object(runner, "call_adapter", side_effect=adapter):
                first = run(job, methods, root / "results")
                second = run(job, methods, root / "results")
                third = run(job, methods, root / "results")
            self.assertEqual([first[0]["status"], second[0]["status"], third[0]["status"]],
                             ["failed", "failed", "ok"])
            self.assertEqual(calls, {"fit": 2, "sample": 3})
            fit_dir = next((root / "results").iterdir())
            self.assertEqual(len(list((fit_dir / "fit-attempts").glob("attempt-*.json"))), 2)
            self.assertEqual(len(list((fit_dir / "sample-attempts").glob("*/attempt-*.json"))), 2)
            self.assertEqual(json.loads((fit_dir / "fit-receipt.json").read_text())["status"], "ok")


if __name__ == "__main__":
    unittest.main()
