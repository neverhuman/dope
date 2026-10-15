"""Per-source identities, query charges, and a durable interrupted attempt."""

import json
from pathlib import Path
import tempfile
import unittest

from research.transfer.evidence import canonical_json, digest, parse_json
from research.transfer.scheduler import DurableSchedule, ExecutionError, FORMAT


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


if __name__ == "__main__":
    unittest.main()
