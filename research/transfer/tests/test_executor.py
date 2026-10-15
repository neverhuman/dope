"""A selection-family learner is never reported held out."""

from pathlib import Path
import tempfile
import unittest

from research.transfer.evidence import digest
from research.transfer.executor import ResearchExecutor
from research.transfer.scheduler import DurableSchedule, ExecutionError


ROOT = Path(__file__).resolve().parents[3]


def roles():
    return {
        "format": "dope-learner-roles",
        "version": 2,
        "selection_families": ["linear", "tree"],
        "evaluation_required_families": ["linear", "tree", "kumo"],
        "held_out_families": ["kumo"],
    }


class ExecutorTests(unittest.TestCase):
    def test_selection_family_learner_is_never_held_out(self):
        parent = ROOT / "target"
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as directory:
            schedule = DurableSchedule(Path(directory) / "plan.json", [
                {"source_id": "source-a", "identity_sha256": digest("source-a")},
            ])
            executor = ResearchExecutor(schedule)
        report = executor.report_family_exposure(roles(), "tree", "tree-a")
        self.assertEqual(report["learner_id"], "tree-a")
        self.assertFalse(report["held_out"])
        self.assertEqual(report["reason"], "selection_family")
        with self.assertRaises(ExecutionError) as raised:
            executor.report_family_exposure(roles(), "linear", "linear-a", claimed_held_out=True)
        self.assertEqual(str(raised.exception), "selection_not_held_out")
        self.assertNotIn("linear-a", str(raised.exception))
        held = executor.report_family_exposure(roles(), "kumo", "kumo-a", claimed_held_out=True)
        self.assertTrue(held["held_out"])
        self.assertEqual(held["reason"], "held_out_family")


if __name__ == "__main__":
    unittest.main()
