"""Crash, binding, privacy and ownership checks for development stage bookkeeping."""
import hashlib
import os
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest
from unittest import mock

from research.transfer.campaign_runner import (
    CampaignRunSpec, CampaignRunnerError, COMMAND_FORMAT, FORMAT, JOURNAL,
    NativeDopeCommandPlan, STAGES, StageJournal,
)
from research.transfer.evidence import canonical_json, digest


def specification():
    return {"format": FORMAT, "version": 1, "phase": "development",
            "source_commit": "1" * 40, "source_tree": "2" * 40,
            "environment_sha256": "3" * 64, "protocol_sha256": "4" * 64, "roster_sha256": "9" * 64,
            "cells": [{"cell_id": "5" * 64, "input_sha256": "6" * 64, "command_sha256": "7" * 64}],
            "budget": {"wall_ms": 1000, "peak_ram_bytes": 1024}, "max_attempts": 2}


class CampaignRunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.spec = CampaignRunSpec.from_document(specification())
        self.cell = "5" * 64

    def journal(self, **kwargs):
        return StageJournal(self.root, self.spec, **kwargs)

    def complete(self, journal, stage="prepare"):
        journal.finish(self.cell, stage, "completed", observed_wall_ms=5,
                       peak_ram_bytes=512, output_sha256="8" * 64)

    def test_source_and_nested_input_mutation_cannot_change_existing_spec(self):
        value = specification()
        spec = CampaignRunSpec.from_document(value)
        original = spec.sha256
        value["cells"][0]["command_sha256"] = "a" * 64
        self.assertEqual(spec.sha256, original)
        self.assertNotEqual(CampaignRunSpec.from_document(value).sha256, original)
        returned = spec.document()
        returned["budget"]["wall_ms"] = 1
        self.assertEqual(spec.document()["budget"]["wall_ms"], 1000)

    def test_malformed_or_protected_spec_refuses_without_echo(self):
        for change in [dict(phase="confirmation"), dict(source_commit="SECRET_SOURCE"),
                       dict(max_attempts=True), dict(version=True), dict(source_values="SECRET_SOURCE")]:
            value = {**specification(), **change}
            with self.assertRaises(CampaignRunnerError) as error:
                CampaignRunSpec.from_document(value)
            self.assertNotIn("SECRET_SOURCE", str(error.exception))
        for data in [b'{"format": 1, "format": 2}', b'{"budget":NaN}', b'{}']:
            with self.assertRaises(CampaignRunnerError):
                CampaignRunSpec.from_bytes(data)

    def test_validation_uses_captured_bytes_when_caller_mutates_during_capture(self):
        value = specification()
        value["phase"] = "confirmation"

        def mutate_after_capture(document):
            encoded = canonical_json(document)
            document["phase"] = "development"
            return encoded

        with mock.patch("research.transfer.campaign_runner.canonical_json", mutate_after_capture):
            with self.assertRaises(CampaignRunnerError):
                CampaignRunSpec.from_document(value)

    def test_read_only_fifo_refuses_without_waiting_for_a_writer(self):
        os.mkfifo(self.root / JOURNAL, 0o600)
        manifest = self.root / "spec.json"
        manifest.write_bytes(canonical_json(specification()))
        code = """
import sys
from pathlib import Path
from research.transfer.campaign_runner import CampaignRunSpec, CampaignRunnerError, StageJournal
spec = CampaignRunSpec.from_bytes(Path(sys.argv[2]).read_bytes())
observed = "unexpectedly_opened"
try:
    with StageJournal(sys.argv[1], spec, read_only=True):
        pass
except CampaignRunnerError as error:
    observed = str(error)
print(observed)
"""
        result = subprocess.run([sys.executable, "-B", "-c", code, str(self.root), str(manifest)],
                                capture_output=True, timeout=5)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, b"unsafe_stage_journal\n")

    def test_direct_constructor_cannot_bypass_development_validation_at_journal(self):
        value = specification()
        value["phase"] = "confirmation"
        forged = CampaignRunSpec(canonical_json(value))
        with self.assertRaises(CampaignRunnerError):
            StageJournal(self.root, forged, create=True)
        self.assertFalse((self.root / JOURNAL).exists())

    def test_existing_scheduler_roster_must_match_exactly(self):
        from research.transfer.scheduler import CampaignSchedule
        dimensions = {k: [0] for k in ("task", "method", "budget", "fitseed", "drawseed",
                                        "auditor", "auditorseed", "endpoint")}
        schedule = CampaignSchedule(self.root / "schedule.json", dimensions, 4, ["full"])
        value = specification()
        value["roster_sha256"] = schedule.document()["roster_sha256"]
        value["cells"] = [{"cell_id": cell["cell_id"], "input_sha256": "6" * 64,
                           "command_sha256": "7" * 64} for cell in schedule.roster]
        CampaignRunSpec.from_document(value).verify_schedule(schedule)
        value["cells"].pop()
        with self.assertRaises(CampaignRunnerError):
            CampaignRunSpec.from_document(value).verify_schedule(schedule)

    def test_duplicate_cells_and_nonfinite_budgets_refuse(self):
        value = specification()
        value["cells"] *= 2
        with self.assertRaises(CampaignRunnerError):
            CampaignRunSpec.from_document(value)
        value = specification()
        value["budget"]["wall_ms"] = float("inf")
        with self.assertRaises(CampaignRunnerError):
            CampaignRunSpec.from_document(value)

    def test_durable_begin_survives_reload_and_is_not_automatically_replayed(self):
        with mock.patch("research.transfer.campaign_runner.os.fsync", wraps=os.fsync) as sync:
            with self.journal(create=True) as journal:
                journal.begin(self.cell, "prepare", 100)
            self.assertGreaterEqual(sync.call_count, 3)
        before = (self.root / JOURNAL).read_bytes()
        with self.journal() as journal:
            self.assertEqual(journal.public_summary()["pending_stages"], 1)
            self.assertEqual(journal.charged, 100)
            with self.assertRaises(CampaignRunnerError):
                journal.begin(self.cell, "prepare", 100, retry=True)
        self.assertEqual((self.root / JOURNAL).read_bytes(), before)

    def test_failed_fsync_closes_writer_and_requires_reconciliation(self):
        with self.journal(create=True) as journal:
            with mock.patch("research.transfer.campaign_runner.os.fsync", side_effect=OSError()):
                with self.assertRaises(CampaignRunnerError):
                    journal.begin(self.cell, "prepare", 100)
            self.assertIsNone(journal.fd)
            with self.assertRaises(CampaignRunnerError):
                journal.begin(self.cell, "prepare", 100)
        with self.journal() as journal:
            self.assertEqual(journal.public_summary()["pending_stages"], 1)
            self.assertEqual(journal.charged, 100)

    def test_interrupted_reservation_remains_charged_before_explicit_retry(self):
        with self.journal(create=True) as journal:
            journal.begin(self.cell, "prepare", 100)
            journal.finish(self.cell, "prepare", "interrupted")
        with self.journal() as journal:
            with self.assertRaises(CampaignRunnerError):
                journal.begin(self.cell, "prepare", 100)
            journal.begin(self.cell, "prepare", 100, retry=True)
            journal.finish(self.cell, "prepare", "failed")
            self.assertEqual(journal.charged, 200)
            with self.assertRaises(CampaignRunnerError):
                journal.begin(self.cell, "prepare", 100, retry=True)

    def test_order_and_duplicate_completion_refuse_without_mutating_bytes(self):
        with self.journal(create=True) as journal:
            for stage in ("compile", "unknown"):
                with self.assertRaises(CampaignRunnerError):
                    journal.begin(self.cell, stage, 10)
            journal.begin(self.cell, "prepare", 10)
            self.complete(journal)
            before = (self.root / JOURNAL).read_bytes()
            with self.assertRaises(CampaignRunnerError):
                self.complete(journal)
            with self.assertRaises(CampaignRunnerError):
                journal.begin(self.cell, "prepare", 10)
            self.assertEqual((self.root / JOURNAL).read_bytes(), before)

    def test_budget_prevents_new_work_and_overruns_are_terminal(self):
        with self.journal(create=True) as journal:
            with self.assertRaises(CampaignRunnerError):
                journal.begin(self.cell, "prepare", 1001)
            journal.begin(self.cell, "prepare", 10)
            with self.assertRaises(CampaignRunnerError):
                journal.finish(self.cell, "prepare", "completed", observed_wall_ms=11,
                               peak_ram_bytes=512, output_sha256="8" * 64)
            journal.finish(self.cell, "prepare", "over_budget", observed_wall_ms=11,
                           peak_ram_bytes=512)
            with self.assertRaises(CampaignRunnerError):
                journal.begin(self.cell, "prepare", 10, retry=True)
        with self.journal() as journal:
            self.assertTrue(journal.public_summary()["resource_overrun"])

    def test_complete_stage_requires_observations_and_output_binding(self):
        with self.journal(create=True) as journal:
            journal.begin(self.cell, "prepare", 10)
            with self.assertRaises(CampaignRunnerError):
                journal.finish(self.cell, "prepare", "completed", output_sha256="8" * 64)
            with self.assertRaises(CampaignRunnerError):
                journal.finish(self.cell, "prepare", "completed", observed_wall_ms=True,
                               peak_ram_bytes=512, output_sha256="8" * 64)
            self.complete(journal)

    def test_completed_bookkeeping_does_not_become_native_or_scientific_evidence(self):
        with self.journal(create=True) as journal:
            for stage in STAGES:
                journal.begin(self.cell, stage, 10)
                self.complete(journal, stage)
            summary = journal.public_summary()
            self.assertEqual(summary["completed_stages"], len(STAGES))
            self.assertFalse(summary["worker_executed"])
            self.assertEqual(summary["scientific_qualification"], "unmeasured")
            self.assertEqual(summary["release_authorization"], "not_authorized")
            self.assertNotIn(self.cell, str(summary))

    def test_corruption_or_partial_record_is_not_repaired_on_load(self):
        with self.journal(create=True):
            pass
        path = self.root / JOURNAL
        for suffix in [b'{"private":"SECRET_SOURCE"}', b'{}\n']:
            original = path.read_bytes()
            path.write_bytes(original + suffix)
            with self.assertRaises(CampaignRunnerError):
                with self.journal():
                    pass
            self.assertEqual(path.read_bytes(), original + suffix)
            path.write_bytes(original)

    def test_hash_valid_illegal_transition_is_rejected(self):
        with self.journal(create=True) as journal:
            previous = journal.records[-1]["sha256"]
        body = {"type": "finish", "plan_sha256": self.spec.sha256, "sequence": 1,
                "previous_sha256": previous, "data": {"cell_id": self.cell, "stage": "prepare",
                    "attempt": 1, "status": "completed", "observed_wall_ms": 0,
                    "peak_ram_bytes": 0, "output_sha256": "8" * 64}}
        with (self.root / JOURNAL).open("ab") as stream:
            stream.write(canonical_json({**body, "sha256": digest(body)}) + b"\n")
        with self.assertRaises(CampaignRunnerError):
            with self.journal():
                pass

    def test_foreign_plan_and_concurrent_writer_are_denied(self):
        with self.journal(create=True):
            with self.assertRaises(CampaignRunnerError):
                with self.journal():
                    pass
        changed = specification()
        changed["source_commit"] = "a" * 40
        with self.assertRaises(CampaignRunnerError):
            with StageJournal(self.root, CampaignRunSpec.from_document(changed)):
                pass

    def test_private_single_link_regular_evidence_is_required(self):
        with self.journal(create=True):
            pass
        path = self.root / JOURNAL
        path.chmod(0o644)
        with self.assertRaises(CampaignRunnerError):
            with self.journal():
                pass
        path.chmod(0o600)
        alias = self.root / "alias"
        os.link(path, alias)
        with self.assertRaises(CampaignRunnerError):
            with self.journal():
                pass
        alias.unlink()
        path.rename(alias)
        path.symlink_to(alias)
        before = alias.read_bytes()
        with self.assertRaises(CampaignRunnerError):
            with self.journal():
                pass
        self.assertEqual(alias.read_bytes(), before)

    def test_read_only_inspection_does_not_write_or_create_a_journal(self):
        with self.assertRaises(CampaignRunnerError):
            with self.journal(read_only=True):
                pass
        self.assertFalse((self.root / JOURNAL).exists())
        with self.journal(create=True):
            pass
        path = self.root / JOURNAL
        before = (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns)
        with self.journal(read_only=True) as journal:
            with self.assertRaises(CampaignRunnerError):
                journal.begin(self.cell, "prepare", 1)
        self.assertEqual((hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns), before)


def native_commands():
    return {"format": COMMAND_FORMAT, "version": 1, "cell_id": "5" * 64,
            "input_sha256": "6" * 64, "binary": "/runtime/dope", "binary_sha256": "8" * 64,
            "csv": "/private/SECRET_SOURCE.csv", "schema": "/private/SECRET_SOURCE.schema",
            "artifact": "/output/model.raw", "fit_seed": 17, "draw_seed": 29,
            "rows": 7, "byte_budget": 4096, "deadline_seconds": 1}


class NativeCommandPlanTests(unittest.TestCase):
    def bound_spec(self, plan):
        value = specification()
        value["cells"][0]["command_sha256"] = plan.sha256
        return CampaignRunSpec.from_document(value)

    def test_native_arguments_keep_sampling_source_paths_out_and_never_launch(self):
        with mock.patch("os.open", side_effect=AssertionError("unexpected IO")), \
                mock.patch("subprocess.Popen", side_effect=AssertionError("unexpected launch")):
            plan = NativeDopeCommandPlan.from_document(native_commands())
            plan.verify_binding(self.bound_spec(plan))
            compile_args, sample_args = plan.compile_argv(), plan.sample_argv()
        self.assertIsInstance(compile_args, tuple)
        self.assertEqual(compile_args[1], "compile-raw")
        self.assertIn("--max-complete-bytes", compile_args)
        self.assertEqual(compile_args[-2:], ("--deadline-seconds", "1"))
        self.assertEqual(sample_args, ("/runtime/dope", "sample-raw-isolated", "--artifact",
                                     "/output/model.raw", "--rows", "7", "--seed", "29"))
        self.assertNotIn(native_commands()["csv"], sample_args)
        self.assertNotIn(native_commands()["schema"], sample_args)
        self.assertFalse(plan.public_summary()["native_binary_verified"])
        self.assertFalse(plan.public_summary()["source_free_execution_verified"])

    def test_shell_characters_in_paths_remain_literal_tokens_and_summaries_redact(self):
        value = native_commands()
        value["csv"] = "/private/SECRET_SOURCE;$(touch injected).csv"
        plan = NativeDopeCommandPlan.from_document(value)
        self.assertEqual(plan.compile_argv()[3], value["csv"])
        self.assertNotIn("SECRET_SOURCE", str(plan.public_summary()))
        self.assertFalse(plan.public_summary()["worker_executed"])

    def test_command_source_and_limits_tampering_breaks_the_existing_cell_binding(self):
        plan = NativeDopeCommandPlan.from_document(native_commands())
        spec = self.bound_spec(plan)
        for field, value in [("csv", "/private/other.csv"), ("binary_sha256", "a" * 64),
                             ("fit_seed", 18), ("draw_seed", 30), ("rows", 8),
                             ("byte_budget", 8192), ("input_sha256", "b" * 64)]:
            with self.subTest(field=field):
                changed = NativeDopeCommandPlan.from_document({**native_commands(), field: value})
                with self.assertRaisesRegex(CampaignRunnerError, "command_plan_binding_mismatch"):
                    changed.verify_binding(spec)
        value = specification()
        value["cells"][0]["command_sha256"] = plan.sha256
        value["cells"][0]["input_sha256"] = "b" * 64
        with self.assertRaisesRegex(CampaignRunnerError, "command_plan_binding_mismatch"):
            plan.verify_binding(CampaignRunSpec.from_document(value))

    def test_cell_and_compile_deadline_must_fit_the_exact_run_spec(self):
        plan = NativeDopeCommandPlan.from_document(native_commands())
        value = self.bound_spec(plan).document()
        value["cells"][0]["cell_id"] = "a" * 64
        with self.assertRaisesRegex(CampaignRunnerError, "command_plan_cell_mismatch"):
            plan.verify_binding(CampaignRunSpec.from_document(value))
        value = self.bound_spec(plan).document()
        value["budget"]["wall_ms"] = 999
        with self.assertRaisesRegex(CampaignRunnerError, "command_plan_budget_mismatch"):
            plan.verify_binding(CampaignRunSpec.from_document(value))

    def test_schema_paths_interpreters_and_malformed_values_refuse_without_echo(self):
        changes = [{"argv": ["sh", "-c", "SECRET_SOURCE"]}, {"version": True},
                   {"binary": "/bin/sh"}, {"binary": "/usr/bin/python3"},
                   {"artifact": native_commands()["csv"]}, {"csv": "relative/SECRET_SOURCE"},
                   {"csv": "/private/../SECRET_SOURCE"}, {"csv": "/private//SECRET_SOURCE"},
                   {"csv": "/private/SECRET_SOURCE\nnext"}, {"csv": "/private/\ud800"},
                   {"rows": True}, {"fit_seed": -1}, {"deadline_seconds": 61},
                   {"byte_budget": 0}, {"binary_sha256": "SECRET_SOURCE"}]
        for change in changes:
            with self.subTest(field=list(change)):
                with self.assertRaises(CampaignRunnerError) as error:
                    NativeDopeCommandPlan.from_document({**native_commands(), **change})
                self.assertNotIn("SECRET_SOURCE", str(error.exception))
        for encoded in [b'{"format":1,"format":2}', b'{"rows":NaN}', b'{}']:
            with self.assertRaises(CampaignRunnerError):
                NativeDopeCommandPlan.from_bytes(encoded)

    def test_original_objects_and_protected_specs_cannot_bypass_validation(self):
        value = native_commands()
        plan = NativeDopeCommandPlan.from_document(value)
        before = plan.sha256
        value["rows"] = 100
        plan.document()["rows"] = 200
        self.assertEqual(plan.sha256, before)
        forged = NativeDopeCommandPlan(b"{}")
        for method in [forged.compile_argv, forged.sample_argv, forged.public_summary]:
            with self.assertRaises(CampaignRunnerError):
                method()
        protected = self.bound_spec(plan).document()
        protected["phase"] = "confirmation"
        with self.assertRaises(CampaignRunnerError):
            plan.verify_binding(CampaignRunSpec(canonical_json(protected)))
