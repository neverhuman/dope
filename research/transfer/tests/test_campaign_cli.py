"""CLI output must remain redacted and protected work must remain denied."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from research.transfer.campaign_cli import main
from research.transfer.campaign_runner import CampaignRunSpec, JOURNAL, NativeDopeCommandPlan, StageJournal
from research.transfer.tests.test_campaign_runner import native_commands, specification


class CampaignCliTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.manifest = self.root / "SECRET_SOURCE.json"
        self.manifest.write_text(json.dumps(specification()))

    def run_cli(self, *argv):
        out, error = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(error):
            code = main(list(argv))
        self.assertEqual(error.getvalue(), "")
        self.assertNotIn("SECRET_SOURCE", out.getvalue())
        return code, json.loads(out.getvalue())

    def test_plan_is_read_only_and_reports_no_native_execution(self):
        before = self.manifest.read_bytes()
        code, result = self.run_cli("plan", "--manifest", str(self.manifest))
        self.assertEqual(code, 0)
        self.assertEqual(result["cells"], 1)
        self.assertFalse(result["worker_executed"])
        self.assertEqual(self.manifest.read_bytes(), before)
        self.assertFalse((self.root / JOURNAL).exists())

    def test_pilot_and_confirm_deny_before_opening_manifest_or_output(self):
        for command in ("pilot", "confirm"):
            code, result = self.run_cli(command, "--manifest", "/SECRET_SOURCE/missing",
                                        "--evidence-dir", str(self.root / "must-not-exist"))
            self.assertEqual(code, 2)
            self.assertEqual(result["reason"], "native_execution_admission_not_connected")
        self.assertFalse((self.root / "must-not-exist").exists())

    def test_arguments_io_and_manifest_errors_do_not_echo_inputs(self):
        cases = [("plan", "--manifest", "/SECRET_SOURCE/missing"),
                 ("plan", "--manifest", str(self.manifest), "--SECRET_SOURCE"),
                 ("SECRET_SOURCE", "--manifest", str(self.manifest))]
        for args in cases:
            self.assertEqual(self.run_cli(*args)[0], 2)
        self.manifest.write_text('{"SECRET_SOURCE": NaN}')
        self.assertEqual(self.run_cli("plan", "--manifest", str(self.manifest))[0], 2)

    def test_fifo_manifest_refuses_without_waiting_for_a_writer(self):
        fifo = self.root / "SECRET_SOURCE.pipe"
        os.mkfifo(fifo, 0o600)
        result = subprocess.run([sys.executable, "-B", "-m", "research.transfer.campaign_cli",
                                 "plan", "--manifest", str(fifo)],
                                capture_output=True, text=True, timeout=5)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(json.loads(result.stdout)["reason"], "run_spec_unavailable")
        self.assertEqual(result.stderr, "")
        self.assertNotIn("SECRET_SOURCE", result.stdout)

    def test_inspect_preserves_pending_reservation_without_replay(self):
        spec = CampaignRunSpec.from_document(specification())
        with StageJournal(self.root, spec, create=True) as journal:
            journal.begin("5" * 64, "prepare", 100)
        before = (self.root / JOURNAL).read_bytes()
        code, result = self.run_cli("inspect", "--manifest", str(self.manifest),
                                    "--evidence-dir", str(self.root))
        self.assertEqual(code, 0)
        self.assertEqual(result["pending_stages"], 1)
        self.assertEqual(result["charged_wall_ms"], 100)
        self.assertEqual((self.root / JOURNAL).read_bytes(), before)

    def test_inspect_missing_directory_and_symlink_manifest_do_not_create_or_follow(self):
        self.assertEqual(self.run_cli("inspect", "--manifest", str(self.manifest),
                                     "--evidence-dir", str(self.root / "absent"))[0], 2)
        alias = self.root / "alias.json"
        alias.symlink_to(self.manifest)
        self.assertEqual(self.run_cli("plan", "--manifest", str(alias))[0], 2)
        self.assertFalse((self.root / "absent").exists())

    def install_commands(self):
        value = native_commands()
        plan = NativeDopeCommandPlan.from_document(value)
        spec = specification()
        spec["cells"][0]["command_sha256"] = plan.sha256
        self.manifest.write_text(json.dumps(spec))
        commands = self.root / "SECRET_SOURCE.commands.json"
        commands.write_text(json.dumps(value))
        return commands

    def test_plan_checks_native_command_binding_without_executing_or_creating_outputs(self):
        commands = self.install_commands()
        original = self.manifest.read_bytes(), commands.read_bytes()
        code, result = self.run_cli("plan", "--manifest", str(self.manifest), "--commands", str(commands))
        self.assertEqual(code, 0)
        self.assertEqual(result["native_commands"]["command_count"], 2)
        self.assertEqual(result["native_commands"]["execution_admission"], "not_connected")
        self.assertFalse(result["native_commands"]["worker_executed"])
        self.assertFalse(result["native_commands"]["source_free_execution_verified"])
        self.assertEqual((self.manifest.read_bytes(), commands.read_bytes()), original)
        self.assertFalse((self.root / JOURNAL).exists())

    def test_changed_native_plan_refuses_and_inspect_does_not_consume_commands(self):
        commands = self.install_commands()
        changed = native_commands()
        changed["draw_seed"] += 1
        commands.write_text(json.dumps(changed))
        code, result = self.run_cli("plan", "--manifest", str(self.manifest), "--commands", str(commands))
        self.assertEqual(code, 2)
        self.assertEqual(result["reason"], "command_plan_binding_mismatch")
        code, result = self.run_cli("inspect", "--manifest", "/SECRET_SOURCE/missing",
                                    "--commands", "/SECRET_SOURCE/missing")
        self.assertEqual(code, 2)
        self.assertEqual(result["reason"], "command_plan_requires_plan")
        code, result = self.run_cli("plan", "--manifest", str(self.manifest), "--commands", "")
        self.assertEqual(code, 2)
        self.assertEqual(result["reason"], "run_spec_unavailable")
        code, result = self.run_cli("inspect", "--manifest", "/SECRET_SOURCE/missing", "--commands", "")
        self.assertEqual(code, 2)
        self.assertEqual(result["reason"], "command_plan_requires_plan")

    def test_native_command_fifo_refuses_and_pilot_stays_denied_before_input_reads(self):
        self.install_commands()
        fifo = self.root / "SECRET_SOURCE.commands.pipe"
        os.mkfifo(fifo, 0o600)
        result = subprocess.run([sys.executable, "-B", "-m", "research.transfer.campaign_cli",
                                 "plan", "--manifest", str(self.manifest), "--commands", str(fifo)],
                                capture_output=True, text=True, timeout=5)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(json.loads(result.stdout)["reason"], "run_spec_unavailable")
        self.assertNotIn("SECRET_SOURCE", result.stdout + result.stderr)
        for command in ("pilot", "confirm"):
            code, output = self.run_cli(command, "--manifest", "/SECRET_SOURCE/missing",
                                        "--commands", str(fifo))
            self.assertEqual(code, 2)
            self.assertEqual(output["reason"], "native_execution_admission_not_connected")
