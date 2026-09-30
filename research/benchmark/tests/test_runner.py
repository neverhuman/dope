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

    def test_frozen_tuned_configuration_requires_eight_validation_trials(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "target") as directory:
            root = Path(directory)
            selected = {"bins": 32}
            receipt = root / "selection.json"
            evidence = {"format": "dope-benchmark-validation-selection",
                        "partition": "validation", "dataset": "toy",
                        "method": "independent_marginals", "selected_config": selected,
                        "trials": [{"status": "ok", "wall_seconds": 1} for _ in range(8)]}
            receipt.write_text(json.dumps(evidence))
            job = {"final": True, "method": "independent_marginals",
                   "configuration": {"kind": "tuned", "values": selected,
                                     "selection_path": str(receipt),
                                     "selection_sha256": sha256(receipt)}}
            entry = {"group": "compact", "default_config": {"bins": 16},
                     "tuning_search_space": {"bins": [8, 16, 32]}}
            config, identity = runner.resolve_configuration(job, entry, "toy", 100, root)
            self.assertEqual(config, selected)
            self.assertEqual(identity["selection_sha256"], sha256(receipt))
            evidence["trials"].pop()
            receipt.write_text(json.dumps(evidence))
            job["configuration"]["selection_sha256"] = sha256(receipt)
            with self.assertRaisesRegex(ValueError, "invalid validation selection"):
                runner.resolve_configuration(job, entry, "toy", 100, root)
            job["configuration"]["values"] = {"bins": 64}
            with self.assertRaisesRegex(ValueError, "outside locked search space"):
                runner.resolve_configuration(job, entry, "toy", 100, root)

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
