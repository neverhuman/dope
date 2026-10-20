"""Signed engineering bridge fixtures; native evidence is an explicit separate entry.

Run a real offline kernel fixture after building the exact source:
  python3 -B -m research.transfer.tests.test_campaign_execution \
    --native-fixture target/debug/dope-kernel --receipt target/native-fixture.json
This fixture is synthetic and cannot qualify E03, official models or a release.
"""
from __future__ import annotations

import base64
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from research.transfer import campaign_execution as execution
from research.transfer.campaign_artifacts import create_private, directory_pin, file_hash, read_regular
from research.transfer.campaign_cli import main
from research.transfer.campaign_native import invoke
from research.transfer.campaign_runner import CampaignRunSpec, CampaignRunnerError, FORMAT, StageJournal
from research.transfer.evidence import canonical_json, digest

GIB = 1 << 30


def fixture(directory, binary=None, stage_wall_ms=10000):
    """Only synthetic private rows; the signing key is local test material."""
    root = Path(directory).absolute()
    source_binary = Path(binary).absolute() if binary else None
    binary = root / "dope-kernel"
    if source_binary is not None:
        # Cargo intentionally hardlinks its output into deps/. The test takes a
        # private single-link snapshot; production admission never relaxes it.
        assert not source_binary.is_symlink() and source_binary.stat().st_size <= 256 << 20
        create_private(binary, source_binary.read_bytes(), 0o500)
    else:
        create_private(binary, b"negative-unit-fixture-not-a-native-kernel", 0o500)
    csv, schema = root / "SECRET_fit.csv", root / "SECRET_schema.json"
    create_private(csv, ("SECRET_y,SECRET_x\n" + "".join(
        f"{100 + 200 * i / 63},{10 + 20 * i / 63}\n" for i in range(64))).encode())
    create_private(schema, canonical_json({"format": "dope-table-schema", "version": 1,
        "columns": [{"id": "c0001", "source_name": "SECRET_x", "role": "feature", "dtype": "continuous"},
                    {"id": "c0002", "source_name": "SECRET_y", "role": "target", "dtype": "continuous"}]}))
    value = {"format": execution.FORMAT, "version": 1, "cell_id": "5" * 64,
        "binary": str(binary), "binary_sha256": file_hash(binary, 256 << 20),
        "csv": str(csv), "csv_sha256": file_hash(csv), "schema": str(schema),
        "schema_sha256": file_hash(schema), "fit_seed": 1729, "draw_seed": 1730,
        "rows": 16, "byte_budget": 32768, "deadline_seconds": 1,
        "stage_wall_ms": {stage: stage_wall_ms for stage in execution.NATIVE_STAGES}}
    plan = execution.NativeExecutionPlan.from_document(value)
    spec = CampaignRunSpec.from_document({"format": FORMAT, "version": 1, "phase": "development",
        "source_commit": "1" * 40, "source_tree": "2" * 40, "environment_sha256": "3" * 64,
        "protocol_sha256": "4" * 64, "roster_sha256": "9" * 64,
        "cells": [{"cell_id": value["cell_id"], "input_sha256": plan.input_sha256,
                   "command_sha256": plan.sha256}],
        "budget": {"wall_ms": stage_wall_ms * 4, "peak_ram_bytes": 2 * GIB}, "max_attempts": 1})
    evidence = root / "evidence"
    evidence.mkdir(mode=0o700)
    key, public = root / "test-private.pem", root / "test-public.pem"
    subprocess.run(["openssl", "genpkey", "-algorithm", "Ed25519", "-out", str(key)],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["openssl", "pkey", "-in", str(key), "-pubout", "-out", str(public)],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    payload = {"scope": execution.SCOPE, "phase": "development", "scientific_admission": False,
        "run_spec_sha256": spec.sha256, "execution_plan_sha256": plan.sha256,
        "rights_admission_sha256": "a" * 64, "resource_lease_sha256": "b" * 64,
        "evidence_directory_sha256": directory_pin(evidence)}
    message, signature, admission = root / "message", root / "signature", root / "admission.json"
    create_private(message, canonical_json(payload))
    subprocess.run(["openssl", "pkeyutl", "-sign", "-inkey", str(key), "-rawin",
                    "-in", str(message), "-out", str(signature)], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    create_private(admission, canonical_json({"payload": payload,
        "signature_ed25519_base64": base64.b64encode(signature.read_bytes()).decode()}))
    return spec, plan, evidence, {"admission": admission, "public_key": public,
                                  "trusted_key_sha256": file_hash(public)}


def simulated_kernel(plan):
    """Negative/control-flow fixture only, never real execution evidence."""
    def call(argv, root, label, _wall, _ram):
        if label == "compile":
            create_private(root / "model.raw", b"unit-artifact")
        document = {"doctor": {"name": "dope-kernel", "implementation": "native_rust",
                        "gpu_training": False, "executable_sha256": plan.document()["binary_sha256"]},
                    "compile": {"unit_fixture": True},
                    "reload": {"accounting": {"complete_sampling_bytes": 13}},
                    "sample": {"sample_sha256": "c" * 64, "filesystem_network_exec": "denied",
                        "inherited_source_fds": "closed", "rows": 16, "seed": 1730}}[label]
        create_private(root / (label + ".stdout"), canonical_json(document))
        create_private(root / (label + ".stderr"), b"")
        return {"status": "completed", "wall_ms": 1, "peak_ram_bytes": 1,
                "returncode": 0, "stdout_sha256": file_hash(root / (label + ".stdout")),
                "stderr_sha256": file_hash(root / (label + ".stderr"))}
    return call


@contextlib.contextmanager
def simulated_execution(plan, call=None):
    """Control-flow fixtures use only simulated worker time/RSS.

    Host fsync latency and lifetime RSS from earlier tests are not measurements
    of a simulated kernel. Real supervisor/native fixtures keep the OS clock
    and usage accounting, and simulated overrun outcomes still exercise limits.
    """
    with mock.patch.object(execution, "invoke", side_effect=call or simulated_kernel(plan)), \
         mock.patch.object(execution.time, "monotonic", return_value=0.0), \
         mock.patch.object(execution.resource, "getrusage", return_value=mock.Mock(ru_maxrss=0)):
        yield


class NativeExecutionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.spec, self.plan, self.root, self.auth = fixture(self.temporary.name)

    def execute(self):
        return execution.execute_native(self.spec, self.plan, self.root, **self.auth)

    def test_exact_plan_binding_and_facade_rejected(self):
        self.plan.verify_binding(self.spec)
        for field, value in [("binary", "/tmp/dope"), ("rows", True),
                             ("deadline_seconds", 0), ("schema", "../SECRET")]:
            changed = self.plan.document()
            changed[field] = value
            with self.assertRaises(CampaignRunnerError):
                execution.NativeExecutionPlan.from_document(changed)
        document = self.spec.document()
        document["cells"][0]["input_sha256"] = "0" * 64
        with self.assertRaises(CampaignRunnerError):
            self.plan.verify_binding(CampaignRunSpec.from_document(document))

    def test_admission_wrong_key_signature_scope_and_input_pins_fail_closed(self):
        original = self.auth["admission"].read_bytes()
        for mutate in [lambda e: e["payload"].update(scope="confirmation"),
                       lambda e: e["payload"].update(scientific_admission=True),
                       lambda e: e["payload"].update(execution_plan_sha256="0" * 64),
                       lambda e: e.update(signature_ed25519_base64="not-base64")]:
            envelope = json.loads(original)
            mutate(envelope)
            self.auth["admission"].write_bytes(canonical_json(envelope))
            with mock.patch.object(execution, "invoke") as worker:
                with self.assertRaises(CampaignRunnerError):
                    self.execute()
                worker.assert_not_called()
            self.assertEqual(list(self.root.iterdir()), [])
        self.auth["admission"].write_bytes(original)
        self.auth["trusted_key_sha256"] = "0" * 64
        with self.assertRaises(CampaignRunnerError):
            self.execute()
        self.assertEqual(list(self.root.iterdir()), [])

    def test_authorization_cannot_be_replayed_in_a_new_evidence_directory(self):
        alternate = Path(self.temporary.name) / "alternate"
        alternate.mkdir(mode=0o700)
        with mock.patch.object(execution, "invoke") as worker:
            with self.assertRaises(CampaignRunnerError):
                execution.execute_native(self.spec, self.plan, alternate, **self.auth)
            worker.assert_not_called()
        self.assertEqual(list(alternate.iterdir()), [])
        self.assertEqual(list(self.root.iterdir()), [])

    def test_artifact_fsync_failure_is_not_success_or_an_automatic_retry(self):
        with simulated_execution(self.plan), \
             mock.patch.object(execution, "sync_artifact",
                               side_effect=CampaignRunnerError("native_artifact_sync_failed")):
            self.assertEqual(self.execute()["status"], "failed")
        with StageJournal(self.root, self.spec, read_only=True) as journal:
            self.assertEqual(journal.charged, 20000)
            self.assertEqual(journal.state[("5" * 64, "compile")]["status"], "failed")
        with self.assertRaises(CampaignRunnerError):
            self.execute()

    def test_private_snapshot_refuses_symlink_fifo_hardlink_oversize(self):
        parent = Path(self.temporary.name)
        target = parent / "SECRET_payload"
        target.write_bytes(b"SECRET_value")
        alias = parent / "alias"
        alias.symlink_to(target)
        with self.assertRaises(CampaignRunnerError):
            read_regular(alias)
        alias.unlink()
        os.mkfifo(alias)
        with self.assertRaises(CampaignRunnerError):
            read_regular(alias)
        alias.unlink()
        os.link(target, alias)
        with self.assertRaises(CampaignRunnerError):
            read_regular(alias)
        alias.unlink()
        with self.assertRaises(CampaignRunnerError):
            read_regular(target, 1)

    def test_changed_source_input_has_no_native_side_effect_and_cannot_reset(self):
        Path(self.plan.document()["csv"]).write_bytes(b"SECRET_changed")
        with mock.patch.object(execution, "invoke") as worker:
            result = self.execute()
            worker.assert_not_called()
        self.assertEqual(result["status"], "failed")
        with StageJournal(self.root, self.spec, read_only=True) as journal:
            self.assertEqual(journal.charged, 10000)
        with self.assertRaises(CampaignRunnerError):
            self.execute()

    def test_original_input_changes_do_not_replace_private_snapshot(self):
        simulated = simulated_kernel(self.plan)
        def call(*args):
            value = simulated(*args)
            if args[2] == "doctor":
                Path(self.plan.document()["csv"]).write_bytes(b"SECRET_changed")
            return value
        with simulated_execution(self.plan, call):
            self.assertEqual(self.execute()["status"], "ok")
        self.assertEqual(file_hash(self.root / "fit.csv"), self.plan.document()["csv_sha256"])

    def test_artifact_changed_between_compile_and_reload_is_refused(self):
        actual_hash = execution.file_hash
        def checked_hash(path, *args):
            if Path(path).name == "dope-kernel" and (self.root / "model.raw").exists():
                (self.root / "model.raw").write_bytes(b"changed-artifact")
            return actual_hash(path, *args)
        with simulated_execution(self.plan), \
             mock.patch.object(execution, "file_hash", side_effect=checked_hash):
            self.assertEqual(self.execute()["status"], "failed")
        receipt = json.loads((self.root / "native-execution.json").read_bytes())
        self.assertEqual([row["stage"] for row in receipt["stages"]], ["prepare", "compile", "reload"])
        self.assertEqual(receipt["native_processes_executed"], 2)

    def test_journal_write_failure_escapes_without_double_finish(self):
        with simulated_execution(self.plan), \
             mock.patch.object(StageJournal, "finish", side_effect=CampaignRunnerError("journal_write_failed")) as finish:
            with self.assertRaisesRegex(CampaignRunnerError, "journal_write_failed"):
                self.execute()
            self.assertEqual(finish.call_count, 1)
        self.assertFalse((self.root / "native-execution.json").exists())
        with StageJournal(self.root, self.spec, read_only=True) as journal:
            self.assertEqual(journal.public_summary()["pending_stages"], 1)

    def test_budget_overrun_and_native_failure_cause_are_preserved(self):
        def call(*_args):
            return {"status": "timeout", "wall_ms": 10001, "peak_ram_bytes": 1, "returncode": -9}
        with simulated_execution(self.plan, call):
            self.assertEqual(self.execute()["status"], "failed")
        row = json.loads((self.root / "native-execution.json").read_bytes())["stages"][0]
        self.assertEqual(row["native_status"], "timeout")
        self.assertEqual(row["status"], "over_budget")
        with StageJournal(self.root, self.spec, read_only=True) as journal:
            self.assertTrue(journal.over_budget)

    def test_redacted_cli_and_unfinished_pipeline_qualification(self):
        manifest, plan = Path(self.temporary.name) / "spec.json", Path(self.temporary.name) / "plan.json"
        manifest.write_bytes(self.spec._encoded)
        plan.write_bytes(self.plan._encoded)
        argv = ["native", "--manifest", str(manifest), "--execution-plan", str(plan),
                "--evidence-dir", str(self.root), "--admission", str(self.auth["admission"]),
                "--public-key", str(self.auth["public_key"]),
                "--trusted-key-sha256", self.auth["trusted_key_sha256"]]
        with simulated_execution(self.plan), \
             contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(main(argv), 0)
        self.assertNotIn("SECRET", output.getvalue())
        summary = json.loads(output.getvalue())
        self.assertFalse(summary["pipeline_complete"])
        self.assertEqual(summary["scientific_qualification"], "unmeasured")
        self.assertEqual(summary["release_authorization"], "not_authorized")
        for phase in ["pilot", "confirm", "native"]:
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(main([phase, "--manifest", "/SECRET/missing"]), 2)
            self.assertNotIn("SECRET", output.getvalue())

    def test_process_identity_race_finishes_failed_stage_and_reaps_owned_child(self):
        import psutil
        children = []
        actual_popen = subprocess.Popen
        def start_child(*args, **kwargs):
            child = actual_popen(*args, **kwargs)
            children.append(child)
            return child
        def invoke_with_race(_argv, root, label, wall, ram):
            # Real owned child, synthetic process-inspection failure; no kernel claim.
            with mock.patch("research.transfer.campaign_native.subprocess.Popen", side_effect=start_child):
                return invoke([sys.executable, "-c", "import time; time.sleep(5)"], root, label, wall, ram)
        with mock.patch("psutil.Process", side_effect=psutil.NoSuchProcess(0)) as inspect, \
             mock.patch.object(execution, "invoke", side_effect=invoke_with_race):
            self.assertEqual(self.execute()["status"], "failed")
        self.assertEqual(inspect.call_count, 1)
        self.assertEqual(len(children), 1)
        self.assertIsNotNone(children[0].poll())
        with StageJournal(self.root, self.spec, read_only=True) as journal:
            self.assertEqual(journal.charged, 10000)
            self.assertEqual(journal.state[("5" * 64, "prepare")]["status"], "failed")
            self.assertEqual(journal.public_summary()["pending_stages"], 0)
        with self.assertRaises(CampaignRunnerError):
            self.execute()

    def test_simulated_control_flow_is_independent_of_prior_host_usage(self):
        with mock.patch.object(execution.time, "monotonic", side_effect=iter(range(0, 10000, 11)).__next__), \
             mock.patch.object(execution.resource, "getrusage", return_value=mock.Mock(ru_maxrss=3 * (1 << 20))), \
             simulated_execution(self.plan):
            self.assertEqual(self.execute()["status"], "ok")
        receipt = json.loads((self.root / "native-execution.json").read_bytes())
        self.assertEqual([row["stage"] for row in receipt["stages"]], list(execution.NATIVE_STAGES))
        self.assertEqual(receipt["native_processes_executed"], 4)
        self.assertTrue(all(row["wall_ms"] == 1 for row in receipt["stages"]))
        self.assertTrue(all(row["peak_ram_bytes"] == 1 for row in receipt["stages"]))

    def test_simulated_reported_ram_overrun_still_refuses(self):
        simulated = simulated_kernel(self.plan)
        ram_limit = self.spec.document()["budget"]["peak_ram_bytes"]
        def call(*args):
            result = simulated(*args)
            return {**result, "peak_ram_bytes": ram_limit + 1}
        with simulated_execution(self.plan, call):
            self.assertEqual(self.execute()["status"], "failed")
        receipt = json.loads((self.root / "native-execution.json").read_bytes())
        row = receipt["stages"][0]
        self.assertEqual(row["native_status"], "completed")
        self.assertEqual(row["status"], "over_budget")
        self.assertEqual(row["peak_ram_bytes"], ram_limit + 1)
        with StageJournal(self.root, self.spec, read_only=True) as journal:
            self.assertTrue(journal.over_budget)
        with self.assertRaises(CampaignRunnerError):
            self.execute()

    def test_real_supervisor_kills_own_timeout_and_returns_observed_usage(self):
        result = invoke([sys.executable, "-c", "import time; time.sleep(5)"],
                        self.root, "timeout-fixture", 80, GIB)
        self.assertIn(result["status"], ["timeout", "over_budget"])
        self.assertIsNotNone(result["returncode"])
        self.assertGreater(result["wall_ms"], 0)
        self.assertGreaterEqual(result["peak_ram_bytes"], 0)


def native_fixture(binary, receipt_path):
    """Real kernel execution, with distinct synthetic engineering authority."""
    with tempfile.TemporaryDirectory(prefix="dope-native-fixture-") as directory:
        spec, plan, root, auth = fixture(directory, binary, stage_wall_ms=30000)
        result = execution.execute_native(spec, plan, root, **auth)
        receipt = json.loads((root / "native-execution.json").read_bytes())
        if result["status"] != "ok":
            raise AssertionError(canonical_json(receipt).decode())
        assert result["native_processes_executed"] == 4
        assert len(receipt["stages"]) == 4
        assert receipt["stages"][-1]["output"]["filesystem_network_exec"] == "denied"
        assert receipt["stages"][2]["artifact_sha256"] == receipt["stages"][3]["artifact_sha256"]
        create_private(receipt_path, canonical_json({"result": result, "receipt": receipt,
            "module_file": execution.__file__, "module_sha256": file_hash(execution.__file__),
            "fixture_kind": "synthetic_engineering_only", "binary_sha256": plan.document()["binary_sha256"]}))
        print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    if len(sys.argv) == 5 and sys.argv[1] == "--native-fixture" and sys.argv[3] == "--receipt":
        native_fixture(sys.argv[2], sys.argv[4])
    else:
        unittest.main()
