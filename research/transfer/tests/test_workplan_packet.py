"""Portable refusal fixtures; these never supply canonical packet qualification."""
import json
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from research.transfer import workplan_packet as packet

ROOT = Path(__file__).resolve().parents[3]


class PacketBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / "target")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.env = mock.patch.dict(os.environ, {}, clear=False)
        self.env.start()
        self.addCleanup(self.env.stop)
        for name in ("DOPE_PACKET_DIR", "DOPE_PACKET_MODE", "DOPE_REQUIRE_PACKET"):
            os.environ.pop(name, None)

    def fixture(self):
        for name in ("tasks.json", "progress.json"):
            (self.root / name).write_text("{}")
        (self.root / "validate_plan.py").write_text("print('fixture validator only')")
        return self.root

    def unit_fixture_pins(self):
        # Failure-path fixtures only; these never qualify canonical acceptance.
        values = {n: hashlib.sha256((self.root / n).read_bytes()).hexdigest()
                  for n in packet.EXPECTED_INPUT_PINS}
        return mock.patch.dict(packet.EXPECTED_INPUT_PINS, values)

    def command(self, *args, **env):
        return subprocess.run([sys.executable, "-B", "-m", "research.transfer.workplan_packet", *args],
                              cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT), **env},
                              capture_output=True, text=True, timeout=10)

    def test_absence_is_not_run_without_any_checkout_probe(self):
        with mock.patch.object(Path, "is_dir", side_effect=AssertionError("implicit probe")):
            self.assertIsNone(packet.resolve_packet_dir())
        result = self.command()
        self.assertEqual(result.returncode, 0)
        report = json.loads(result.stdout)
        self.assertEqual((report["status"], report["reason"]), ("not_run", packet.NOT_RUN))
        self.assertFalse(report["release_authorized"])
        self.assertEqual(report["scientific_qualification"], "not_established")

    def test_require_absent_refuses_in_cli_and_environment(self):
        for args, env in [(("--require-packet",), {}), ((), {"DOPE_REQUIRE_PACKET": "1"}),
                          (("--mode", "qualify"), {})]:
            with self.subTest(args=args, env=env):
                result = self.command(*args, **env)
                self.assertEqual(result.returncode, 2)
                self.assertEqual(json.loads(result.stdout)["status"], "not_run")
                self.assertEqual(json.loads(result.stdout)["reason"], "PACKET_REQUIRED_NOT_QUALIFIED")

    def test_explicit_unit_mode_cannot_qualify_even_when_packet_is_present(self):
        self.fixture()
        with mock.patch.dict(os.environ, {"DOPE_PACKET_DIR": str(self.root)}):
            with mock.patch.object(Path, "is_dir", side_effect=AssertionError("unit read")):
                self.assertIsNone(packet.resolve_packet_dir(mode="unit"))
            report = json.loads(self.command("--mode", "unit").stdout)
        self.assertEqual(report["status"], "not_run")
        self.assertNotIn("input_hashes", report)

    def test_required_unit_conflict_and_unknown_mode_refuse(self):
        with mock.patch.dict(os.environ, {"DOPE_REQUIRE_PACKET": "1"}):
            with self.assertRaisesRegex(packet.PacketError, "PACKET_MODE_CONFLICT"):
                packet.resolve_packet_dir(mode="unit")
        with self.assertRaisesRegex(packet.PacketError, "PACKET_MODE_INVALID"):
            packet.resolve_packet_dir(mode="not-a-mode")
        with mock.patch.dict(os.environ, {"DOPE_REQUIRE_PACKET": "yes"}):
            with self.assertRaisesRegex(packet.PacketError, "PACKET_REQUIRE_FLAG_INVALID"):
                packet.resolve_packet_dir()

    def test_missing_incomplete_relative_and_denied_explicit_input_refuse(self):
        for root in [str(self.root / "missing-sensitive-path"), str(self.root), "relative"]:
            with self.subTest(root=root), self.assertRaises(packet.PacketError):
                packet.resolve_packet_dir(root)
        self.fixture()
        with mock.patch.object(Path, "is_dir", side_effect=PermissionError("private-value")):
            with self.assertRaisesRegex(packet.PacketError, "^PACKET_INPUT_UNAVAILABLE$"):
                packet.resolve_packet_dir(str(self.root))

    def test_refusal_does_not_echo_input_path_or_values(self):
        marker = "source-header-secret-12345"
        result = self.command("--require-packet", DOPE_PACKET_DIR=str(self.root / marker))
        self.assertEqual(result.returncode, 2)
        self.assertNotIn(marker, result.stdout + result.stderr)
        self.assertEqual(json.loads(result.stdout)["status"], "not_run")

    def test_valid_explicit_directory_is_selected_without_foreign_fallback(self):
        self.fixture()
        self.assertEqual(packet.resolve_packet_dir(str(self.root)), self.root)
        with mock.patch.dict(os.environ, {"DOPE_PACKET_DIR": str(self.root)}):
            self.assertEqual(packet.resolve_packet_dir(), self.root)

    def test_packet_root_file_and_parent_symlinks_refuse(self):
        self.fixture()
        alias = self.root / "alias"
        alias.symlink_to(self.root, target_is_directory=True)
        with self.assertRaises(packet.PacketError):
            packet.resolve_packet_dir(str(alias))
        (self.root / "tasks.json").unlink()
        (self.root / "tasks.json").symlink_to(self.root / "progress.json")
        with self.assertRaises(packet.PacketError):
            packet.resolve_packet_dir(str(self.root))

    def test_fifo_refuses_before_open_without_waiting(self):
        self.fixture()
        (self.root / "tasks.json").unlink()
        os.mkfifo(self.root / "tasks.json")
        result = self.command("--require-packet", DOPE_PACKET_DIR=str(self.root))
        self.assertEqual(result.returncode, 2)
        self.assertEqual(json.loads(result.stdout)["reason"], "PACKET_FILE_INVALID")

    def test_replacement_between_stat_and_open_refuses(self):
        self.fixture()
        native_open = os.open
        def replace(path, flags, *args, **kwargs):
            Path(path).unlink()
            Path(path).symlink_to(self.root / "progress.json")
            return native_open(path, flags, *args, **kwargs)
        with mock.patch.object(os, "open", side_effect=replace):
            with self.assertRaises(packet.PacketError):
                packet._regular_bytes(self.root, "tasks.json")

    def test_oversized_and_duplicate_json_refuse_without_values(self):
        self.fixture()
        with (self.root / "tasks.json").open("wb") as stream:
            stream.truncate(8 * 1024 * 1024 + 1)
        with self.assertRaisesRegex(packet.PacketError, "PACKET_FILE_INVALID"):
            packet.resolve_packet_dir(str(self.root))
        for data in [b'{"secret":1,"secret":2}', b'{"value":NaN}']:
            with self.assertRaisesRegex(packet.PacketError, "PACKET_JSON_INVALID"):
                packet._json_bytes(data)

    def test_schema_and_boolean_count_drift_refuse(self):
        self.fixture()
        for data in [[], {}, {"tasks": []}]:
            (self.root / "tasks.json").write_text(json.dumps(data))
            with self.unit_fixture_pins(), self.assertRaises(packet.PacketError):
                packet.assert_canonical_acceptance_truth(self.root)
        tasks = [{"id": f"T{i}", "status": "done"} for i in range(65)]
        (self.root / "tasks.json").write_text(json.dumps({"tasks": tasks, "summary_counts": {
            "done": 61, "blocked": 4, "in_progress": False, "todo": 0}}))
        (self.root / "progress.json").write_text(json.dumps({"counts": {}, "status_summary": {}}))
        with self.unit_fixture_pins(), self.assertRaisesRegex(packet.PacketError, "PACKET_REPORTED_COUNTS_CHANGED"):
            packet.assert_canonical_acceptance_truth(self.root)

    def test_fixture_validator_success_never_qualifies_canonical_acceptance(self):
        self.fixture()
        result = self.command("--require-packet", DOPE_PACKET_DIR=str(self.root))
        self.assertEqual(result.returncode, 2)
        self.assertEqual(json.loads(result.stdout)["status"], "not_run")

    def test_validator_failure_and_input_mutation_refuse(self):
        self.fixture()
        with mock.patch.dict(os.environ, {"DOPE_PACKET_DIR": str(self.root)}):
            (self.root / "validate_plan.py").write_text("raise RuntimeError('negative validator fixture')")
            with self.unit_fixture_pins(), self.assertRaisesRegex(packet.PacketError, "PACKET_VALIDATOR_FAILED"):
                packet.run_validate_plan()
            (self.root / "validate_plan.py").write_text("from pathlib import Path;Path('tasks.json').write_text('{}changed')")
            with self.unit_fixture_pins(), self.assertRaisesRegex(packet.PacketError, "PACKET_CHANGED_DURING_VALIDATION"):
                packet.run_validate_plan()

    def test_unpinned_validator_is_refused_before_execution(self):
        self.fixture()
        (self.root / "validate_plan.py").write_text("from pathlib import Path;Path('must-not-execute').touch()")
        with mock.patch.dict(os.environ, {"DOPE_PACKET_DIR": str(self.root)}):
            with self.assertRaisesRegex(packet.PacketError, "PACKET_SNAPSHOT_OR_VALIDATOR_PIN_CHANGED"):
                packet.run_validate_plan()
        self.assertFalse((self.root / "must-not-execute").exists())

    def test_matching_counters_cannot_substitute_for_admitted_input_bytes(self):
        self.fixture()
        ids = ["W01"] + [f"T{i}" for i in range(60)]
        tasks = [{"id": i, "status": "done"} for i in ids]
        tasks += [{"id": i, "status": "blocked"} for i in ["E03", "E04", "E05", "E06"]]
        counts = {"done": 61, "blocked": 4, "in_progress": 0}
        (self.root / "tasks.json").write_text(json.dumps({"tasks": tasks, "summary_counts": {**counts, "todo": 0}}))
        progress = {**counts, "total": 65, "backlog": 0, "ready": 0, "review": 0}
        (self.root / "progress.json").write_text(json.dumps({"counts": progress, "status_summary": progress,
            "blocked_tasks": ["E03", "E04", "E05", "E06"]}))
        result = self.command("--require-packet", DOPE_PACKET_DIR=str(self.root))
        self.assertEqual(result.returncode, 2)
        self.assertEqual(json.loads(result.stdout)["reason"], "PACKET_SNAPSHOT_PIN_CHANGED")

    def test_unit_acceptance_asserts_not_qualified_without_disabling_case(self):
        result = subprocess.run([sys.executable, "-B", "-m", "unittest",
            "research.transfer.tests.test_workplan_acceptance.TestP04WorkplanDashboard", "-v"],
            cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT), "DOPE_PACKET_MODE": "unit"},
            capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0)
        self.assertNotIn("skipped", result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["status"], "not_run")
        self.assertEqual(report["reason"], packet.NOT_RUN)
        self.assertFalse(report["release_authorized"])
        self.assertNotIn("input_hashes", report)

    def test_no_packet_validator_and_dashboard_are_explicitly_not_run(self):
        self.assertEqual(packet.run_validate_plan(), packet.NOT_RUN)
        self.assertEqual(packet.check_dashboard_200(), "DOPE_DASHBOARD_NOT_RUN_NOT_QUALIFIED")


if __name__ == "__main__":
    unittest.main()
