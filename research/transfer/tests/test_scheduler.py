"""Per-source identities, query charges, and a durable interrupted attempt."""

import copy
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest import mock
from pathlib import Path
import tempfile
import unittest

from research.transfer.evidence import canonical_json, digest, parse_json
from research.transfer.scheduler import CampaignSchedule, DurableSchedule, ExecutionError, FORMAT


ROOT = Path(__file__).resolve().parents[3]


def workspace_temp():
    parent = ROOT / "target"
    parent.mkdir(parents=True, exist_ok=True)
    return tempfile.TemporaryDirectory(dir=parent)


def sources():
    return [
        {"source_id": "source-a", "identity_sha256": digest("source-a")},
        {"source_id": "source-b", "identity_sha256": digest("source-b")},
    ]


class SchedulerTests(unittest.TestCase):
    def test_schema_names_the_execution_plan(self):
        schema = json.loads((ROOT / "schemas/transfer-execution-plan-v1.schema.json").read_text())
        self.assertEqual(schema["$id"], "urn:dope:transfer-execution-plan:v1")
        self.assertEqual(schema["properties"]["format"]["const"], FORMAT)
        self.assertEqual(schema["properties"]["version"]["const"], 1)
        self.assertEqual(schema["required"], ["format", "version", "sources", "attempts"])
        self.assertIs(schema["additionalProperties"], False)

    def test_one_source_cannot_carry_two_identity_hashes(self):
        other = digest("other-identity")
        with workspace_temp() as directory:
            with self.assertRaises(ExecutionError) as raised:
                DurableSchedule(Path(directory) / "plan.json", [
                    {"source_id": "source-a", "identity_sha256": digest("source-a")},
                    {"source_id": "source-a", "identity_sha256": other},
                ])
        self.assertEqual(str(raised.exception), "source_identity_mismatch")
        self.assertNotIn(other, str(raised.exception))

    def test_query_charges_stay_on_one_source_and_survive_interrupt(self):
        with workspace_temp() as directory:
            path = Path(directory) / "plan.json"
            schedule = DurableSchedule(path, sources())
            schedule.charge("attempt-1", "source-a", "query-1", 4)
            schedule.charge("attempt-2", "source-a", "query-1", 1)
            with self.assertRaises(ExecutionError) as raised:
                schedule.charge("attempt-3", "source-b", "query-1", 9)
            self.assertEqual(str(raised.exception), "query_charge_source_mismatch")
            self.assertNotIn("query-1", str(raised.exception))
            self.assertEqual(schedule.attempts[0]["charged_queries"], 4)
            self.assertEqual(schedule.interrupt("attempt-1"), 4)
            self.assertEqual(schedule.attempts[0]["status"], "interrupted")
            self.assertEqual(schedule.attempts[0]["charged_queries"], 4)
            loaded = DurableSchedule.load(path)
            self.assertEqual(parse_json(path.read_bytes()), schedule.document())
            self.assertEqual(path.read_bytes(), canonical_json(loaded.document()))
            self.assertEqual(loaded.attempts[0]["status"], "interrupted")
            self.assertEqual(loaded.attempts[0]["charged_queries"], 4)
            self.assertEqual(loaded.attempts[0]["source_id"], "source-a")
            self.assertEqual(loaded.attempts[1]["charged_queries"], 1)
            self.assertEqual(loaded.sources[0]["identity_sha256"], digest("source-a"))
            self.assertEqual(loaded.sources[1]["identity_sha256"], digest("source-b"))


class CampaignScheduleDurabilityTests(unittest.TestCase):
    @staticmethod
    def dimensions():
        return {key: [key + "-fixture"] for key in (
            "task", "method", "budget", "fitseed", "drawseed", "auditor", "auditorseed", "endpoint")}

    def schedule(self, path, caps=None):
        return CampaignSchedule(path, self.dimensions(), 4, ["full"], resource_caps=caps)

    @staticmethod
    def attempt(schedule, identity="one", cell=0, status="timeout", **changes):
        item = {
            "attempt_id": "attempt-" + identity, "run_id": "run-" + identity,
            "cell_id": schedule.roster[cell]["cell_id"], "parent_attempt_id": None,
            "original_failure": None, "retry_authorized": False, "status": status,
            "requested_rows": 4, "produced_rows": 4, "provided_rows": 4, "consumed_rows": 4,
            "wall_ms": 3, "cpu_ms": 2, "gpu_ms": 0, "residency_ms": 1,
            "peak_ram_bytes": 4, "peak_vram_bytes": 0, "disk_bytes": 2,
            "cost_basis": {key: key + "-exclusive" for key in ("wall", "cpu", "gpu", "residency", "disk")},
            "run_pins": {"parent_hashes": [digest("public-fixture-parent")],
                         "toolchain_sha256": digest("toolchain"),
                         "command_sha256": digest("command"), "budget_sha256": digest("budget")},
        }
        item.update(changes)
        return item

    def test_existing_campaign_constructor_preserves_all_charged_bytes(self):
        with workspace_temp() as directory:
            path = Path(directory) / "campaign.json"
            schedule = self.schedule(path)
            schedule.append_attempt(self.attempt(schedule))
            before = path.read_bytes()
            with self.assertRaisesRegex(ExecutionError, "^execution_ledger_exists$"):
                self.schedule(path)
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(len(CampaignSchedule.load(path).attempts), 1)

    def test_existing_legacy_constructor_preserves_query_charges(self):
        with workspace_temp() as directory:
            path = Path(directory) / "plan.json"
            schedule = DurableSchedule(path, sources())
            schedule.charge("attempt-1", "source-a", "query-1", 4)
            before = path.read_bytes()
            with self.assertRaisesRegex(ExecutionError, "^execution_ledger_exists$"):
                DurableSchedule(path, sources())
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(DurableSchedule.load(path).attempts[0]["charged_queries"], 4)

    def test_initialization_refuses_sentinels_and_symlinks_without_modification(self):
        for factory in (self.schedule, lambda p: DurableSchedule(p, sources())):
            for kind in ("regular", "symlink", "dangling"):
                with self.subTest(kind=kind, factory=factory), workspace_temp() as directory:
                    root = Path(directory)
                    path, target = root / "ledger.json", root / "prior.json"
                    if kind == "regular":
                        path.write_bytes(b"preserve-public-sentinel")
                    elif kind == "symlink":
                        target.write_bytes(b"preserve-public-target")
                        path.symlink_to(target)
                    else:
                        path.symlink_to(target)
                    before = path.read_bytes() if kind != "dangling" else None
                    with self.assertRaisesRegex(ExecutionError, "^execution_ledger_exists$"):
                        factory(path)
                    self.assertEqual(path.is_symlink(), kind != "regular")
                    if before is not None:
                        self.assertEqual(path.read_bytes(), before)
                    else:
                        self.assertFalse(target.exists())

    def test_competing_constructors_have_exactly_one_exclusive_winner(self):
        with workspace_temp() as directory:
            path = Path(directory) / "ledger.json"
            barrier = threading.Barrier(2)

            def create():
                barrier.wait(timeout=5)
                try:
                    return self.schedule(path)
                except ExecutionError as exc:
                    return str(exc)

            with ThreadPoolExecutor(max_workers=2) as pool:
                results = list(pool.map(lambda _: create(), range(2)))
            winners = [x for x in results if isinstance(x, CampaignSchedule)]
            self.assertEqual(len(winners), 1)
            self.assertEqual([x for x in results if isinstance(x, str)], ["execution_ledger_exists"])
            self.assertEqual(path.read_bytes(), canonical_json(winners[0].document()))

    def test_each_observed_resource_overrun_is_saved_before_refusal(self):
        for field in ("wall_ms", "cpu_ms", "gpu_ms", "residency_ms", "peak_ram_bytes",
                      "peak_vram_bytes", "disk_bytes"):
            with self.subTest(field=field), workspace_temp() as directory:
                path = Path(directory) / "campaign.json"
                schedule = self.schedule(path, {field: 10})
                attempt = self.attempt(schedule, **{field: 11})
                with self.assertRaisesRegex(ExecutionError, "^resource_cap_exceeded$"):
                    schedule.append_attempt(attempt)
                self.assertEqual(schedule.attempts, [attempt])
                before = path.read_bytes()
                self.assertEqual(parse_json(before)["attempts"], [attempt])
                loaded = CampaignSchedule.load(path)
                self.assertEqual(loaded.document(), schedule.document())
                with self.assertRaisesRegex(ExecutionError, "^resource_cap_exceeded$"):
                    loaded.append_attempt(self.attempt(loaded, "two", cell=1))
                retry = self.attempt(loaded, "retry", parent_attempt_id=attempt["attempt_id"],
                                     original_failure="timeout", retry_authorized=True)
                with self.assertRaisesRegex(ExecutionError, "^resource_cap_exceeded$"):
                    loaded.append_attempt(retry)
                self.assertEqual(path.read_bytes(), before)
                self.assertFalse(loaded.complete(accepted=lambda _: True))

    def test_completed_overrun_cannot_be_reported_as_success(self):
        with workspace_temp() as directory:
            path = Path(directory) / "campaign.json"
            schedule = self.schedule(path, {"wall_ms": 10})
            document = schedule.document()
            document["attempts"] = [self.attempt(schedule, "one", status="completed", wall_ms=11),
                                    self.attempt(schedule, "two", cell=1, status="completed")]
            path.write_bytes(canonical_json(document))
            before = path.read_bytes()
            loaded = CampaignSchedule.load(path)
            self.assertEqual(len(loaded.attempts), len(loaded.roster))
            self.assertFalse(loaded.complete())
            self.assertFalse(loaded.complete(accepted=lambda _: True))
            self.assertEqual(path.read_bytes(), before)

    def test_invalid_or_duplicate_attempt_does_not_mutate_evidence(self):
        with workspace_temp() as directory:
            path = Path(directory) / "campaign.json"
            schedule = self.schedule(path, {"wall_ms": 10})
            schedule.append_attempt(self.attempt(schedule))
            before, retained = path.read_bytes(), copy.deepcopy(schedule.attempts)
            invalid = self.attempt(schedule, "two", cell=1, wall_ms=-1)
            duplicate = self.attempt(schedule, wall_ms=11)
            for item, code in ((invalid, "invalid_attempt"), (duplicate, "duplicate_attempt")):
                with self.subTest(code=code), self.assertRaisesRegex(ExecutionError, "^" + code + "$"):
                    schedule.append_attempt(item)
                self.assertEqual(path.read_bytes(), before)
                self.assertEqual(schedule.attempts, retained)

    def test_failed_persistence_fences_writer_until_explicit_reload(self):
        import research.transfer.scheduler as module
        for after_replace in (False, True):
            with self.subTest(after_replace=after_replace), workspace_temp() as directory:
                path = Path(directory) / "campaign.json"
                schedule = self.schedule(path)
                schedule.append_attempt(self.attempt(schedule))
                before = path.read_bytes()
                original_write = module._atomic_write

                def fail_write(*args):
                    if after_replace:
                        original_write(*args)
                    raise OSError("public-fixture-io-failure")

                with mock.patch.object(module, "_atomic_write", side_effect=fail_write):
                    with self.assertRaisesRegex(ExecutionError, "^ledger_write_uncertain$"):
                        schedule.append_attempt(self.attempt(schedule, "two", cell=1))
                uncertain = path.read_bytes()
                self.assertEqual(uncertain != before, after_replace)
                with self.assertRaisesRegex(ExecutionError, "^ledger_write_uncertain$"):
                    schedule.append_attempt(self.attempt(schedule, "three", cell=1))
                self.assertEqual(path.read_bytes(), uncertain)
                loaded = CampaignSchedule.load(path)
                self.assertEqual(len(loaded.attempts), 2 if after_replace else 1)
                self.assertEqual(path.read_bytes(), uncertain)

    def test_caller_cannot_change_retained_attempt_pins_or_cost_basis(self):
        with workspace_temp() as directory:
            path = Path(directory) / "campaign.json"
            schedule = self.schedule(path)
            item = self.attempt(schedule)
            schedule.append_attempt(item)
            before = path.read_bytes()
            item["run_pins"]["parent_hashes"].append(digest("caller-change"))
            item["cost_basis"]["wall"] = "changed"
            self.assertEqual(schedule.document(), parse_json(before))
            self.assertEqual(path.read_bytes(), before)

    def test_partial_initialization_is_preserved_and_never_overwritten(self):
        import research.transfer.scheduler as module
        with workspace_temp() as directory:
            path = Path(directory) / "campaign.json"
            original_write = module.os.write

            def partial_then_fail(fd, payload):
                original_write(fd, payload[:10])
                raise OSError("public-fixture-partial-write")

            with mock.patch.object(module.os, "write", side_effect=partial_then_fail):
                with self.assertRaisesRegex(ExecutionError, "^execution_ledger_write_uncertain$"):
                    self.schedule(path)
            before = path.read_bytes()
            self.assertEqual(len(before), 10)
            with self.assertRaisesRegex(ExecutionError, "^execution_ledger_exists$"):
                self.schedule(path)
            self.assertEqual(path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
