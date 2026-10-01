import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from research.benchmark import gpu_probe, sdv_adapter, sdv_round
from research.benchmark.publish_sdv_validation import winner
from research.benchmark.score import sha256


class SdvRoundTests(unittest.TestCase):
    def test_native_selection_ignores_common_metric_and_failed_trials(self):
        native_best = {"status": "ok", "native_kpi": {"value": 0.8}, "artifact_bytes": 100000,
                       "job": {"config": {"batch_size": 500}}, "catboost_retention": 0.1}
        common_best = {"status": "ok", "native_kpi": {"value": 0.6}, "artifact_bytes": 1000,
                       "job": {"config": {"batch_size": 100}}, "catboost_retention": 0.9}
        failed = {"status": "timeout", "native_kpi": {"value": 1.0}}
        self.assertIs(winner([common_best, failed, native_best]), native_best)
        self.assertIsNone(winner([failed]))

    def test_evaluator_input_is_rejected_before_read(self):
        for path in (Path("test.csv"), Path("evaluator/dataset/renamed.csv")):
            with self.assertRaisesRegex(ValueError, "sealed test"):
                sdv_adapter.load_table(path)

    def test_attempt_hash_verification_detects_altered_metric(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metric = root / "native.receipt.json"
            metric.write_text('{"value": 0.1}')
            receipt = {"status": "failed", "evidence_files": {metric.name: sha256(metric)}}
            sdv_round.verify_attempt(root, receipt)
            metric.write_text('{"value": 0.9}')
            with self.assertRaisesRegex(ValueError, "evidence changed"):
                sdv_round.verify_attempt(root, receipt)

    def test_timeout_preserves_receipt_and_terminates_process_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            prefix = Path(tmp) / "fit"
            class Child:
                pid = 12345
                returncode = None
                def poll(self):
                    return None
                def wait(self, timeout=None):
                    self.returncode = -15
            with patch.object(sdv_round.subprocess, "Popen", return_value=Child()), \
                    patch.object(sdv_round.os, "killpg") as kill, \
                    patch.object(sdv_round.time, "sleep"):
                with self.assertRaisesRegex(RuntimeError, "timeout"):
                    sdv_round.process({"action": "fit"}, prefix, 0)
                kill.assert_called_once_with(12345, sdv_round.signal.SIGTERM)
            receipt = json.loads(prefix.with_suffix(".receipt.json").read_text())
            self.assertEqual(receipt["status"], "timeout")
            self.assertEqual(receipt["exit_code"], -15)

    def test_dope_probe_records_timeout_and_monitor_failure(self):
        for reason, clock_values in (("timeout", [0, 0, 601, 601, 602]),
                                     ("resource_monitor_failed", [0, 0, 2, 2, 2])):
            with self.subTest(reason=reason), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                worker = root / "worker"
                worker.mkdir()
                for part in ("train", "validation"):
                    (worker / f"{part}.csv").write_text("0.1,0.2\n")
                (worker / "projection.json").write_text('{"task":"regression"}')
                manifest = {"dataset_id": "fixture", "split_hashes": {},
                            "projection_sha256": sha256(worker / "projection.json"),
                            "projected_hashes": {p: sha256(worker / f"{p}.csv") for p in ("train", "validation")}}
                (worker / "worker-manifest.json").write_text(json.dumps(manifest))
                binary = root / "binary"
                binary.write_bytes(b"fixture")
                class Child:
                    pid = 23456
                    def poll(self):
                        return None
                    def wait(self):
                        return -9
                gpu = {"index": 0, "active_processes": False, "free_mib": 24000,
                       "used_mib": 0, "power_watts": 30}
                samples = [gpu, OSError("monitor unavailable") if reason == "resource_monitor_failed" else gpu]
                with patch.object(gpu_probe, "ROOT", root), patch.object(gpu_probe, "used_bytes", return_value=0), \
                        patch.object(gpu_probe, "gpu_snapshot", side_effect=samples), \
                        patch.object(gpu_probe.os, "sched_getaffinity", return_value=set(range(16))), \
                        patch.object(gpu_probe.time, "monotonic", side_effect=clock_values), \
                        patch.object(gpu_probe.time, "sleep"), \
                        patch.object(gpu_probe.subprocess, "Popen", return_value=Child()) as popen, \
                        patch.object(gpu_probe.os, "killpg") as kill:
                    result = gpu_probe.run(worker, binary, "micro_tvae_4_16", 11, 2.0, 0.0, root / "fits")
                self.assertEqual(result["status"], reason)
                self.assertEqual(popen.call_args.args[0][:3], ["timeout", "--signal=KILL", "600"])
                kill.assert_called_once_with(23456, gpu_probe.signal.SIGKILL)
                receipt = json.loads(Path(result["receipt"]).read_text())
                self.assertIsNone(receipt["ptf_v1"])
                self.assertEqual(receipt["exit_code"], -9)


if __name__ == "__main__":
    unittest.main()
