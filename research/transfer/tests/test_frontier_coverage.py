"""Minimum-byte and fixed-roster regressions, sharing the Rust golden corpus."""
from dataclasses import replace
import copy
import itertools
import json
from pathlib import Path
import unittest

from research.transfer.frontier_coverage import ArtifactTrial, minimum_qualified_bytes, roster_coverage
from research.transfer.losses import PairedLosses, TrainingScale
from research.transfer.metrics import CellResult, Margin, TaskEvaluation, observed_coverage, qualified_coverage
from research.transfer.tdl import frontier, golden


ROOT = Path(__file__).resolve().parents[3] / "tests/golden/frontier"


def fixture(name):
    return json.loads((ROOT / f"v2_{name}.json").read_text())


def scheduled(task_id):
    pairs = PairedLosses(("q",), ("entity",), (1.,), (1.,), (101.,))
    cell = CellResult("measured", pairs, TrainingScale(100.), Margin(0., 0.01))
    return TaskEvaluation("family", task_id, ("auditor",), {"auditor": cell}, 900, 1000, True)


class FrontierCoverageTests(unittest.TestCase):
    def test_shared_rust_python_golden_values(self):
        paths = sorted(ROOT.glob("v2_*.json"))
        self.assertGreaterEqual(len(paths), 3)
        for path in paths:
            case = json.loads(path.read_text())
            with self.subTest(case=case["case_id"]):
                self.assertTrue(golden.close(golden.run_case(case), case["expected"]))

    def test_minimum_900_800_is_independent_of_discovery_order(self):
        case = fixture("minimum_900_800")
        for cells in itertools.permutations(case["input"]["cells"]):
            data = dict(case["input"], cells=list(cells))
            self.assertEqual(frontier.evaluate(data), case["expected"])
        self.assertEqual(case["expected"]["minimum_qualified_bytes"], [800])
        self.assertEqual([row["coverage"] for row in case["expected"]["frontier"]], [0., 1., 1., 1.])

    def test_later_oom_failure_or_timeout_retains_900_byte_artifact(self):
        case = fixture("later_failure")
        for status in ("oom", "failed", "crashed", "technical_failure", "timeout", "timed_out",
                       "incomplete", "unqualified", "unsupported", "unevaluated", "unresolved"):
            data = copy.deepcopy(case["input"])
            data["cells"][1]["status"] = status
            with self.subTest(status=status):
                self.assertEqual(frontier.evaluate(data), case["expected"])
        strict = fixture("monotone_required")
        self.assertEqual(frontier.evaluate(strict["input"]), strict["expected"])

    def test_roster_denominator_cannot_drop_omitted_or_terminal_tasks(self):
        case = fixture("full_roster")
        for omitted in (False, True):
            data = copy.deepcopy(case["input"])
            if omitted:
                data["cells"] = data["cells"][:1]
            self.assertEqual(frontier.evaluate(data), case["expected"])
        for rows in ([], [{"unexpected": "PRIVATE_SOURCE"}]):
            data = dict(case["input"], cells=rows)
            result = frontier.evaluate(data)
            self.assertNotIn("PRIVATE_SOURCE", json.dumps(result))
            if not rows:
                self.assertEqual(result["admitted_tasks"], 5)
                self.assertTrue(all(row["coverage"] == 0 for row in result["frontier"]))

    def test_search_allowance_does_not_replace_charged_bytes(self):
        case = fixture("larger_search_cap")
        result = frontier.evaluate(case["input"])
        self.assertEqual(result["minimum_qualified_bytes"], [800])
        self.assertEqual([row["coverage"] for row in result["frontier"]], [0., 1., 1.])
        self.assertFalse(result["global_optimum_claimed"])

    def test_multiple_attempts_at_same_cap_do_not_invent_a_regime_decline(self):
        data = copy.deepcopy(fixture("minimum_900_800")["input"])
        data["require_monotone"] = True
        failed = dict(data["cells"][0], candidate_id="failed", artifact_bytes=None, status="oom")
        data["cells"].append(failed)
        result = frontier.evaluate(data)
        self.assertEqual(result["minimum_qualified_bytes"], [800])
        self.assertEqual(result["non_monotone_tasks"], 0)

    def test_typed_frontier_api_uses_the_full_roster(self):
        trials = [ArtifactTrial(0, 1000, 900, True), ArtifactTrial(0, 1000, 800, True),
                  ArtifactTrial(1, 1200, None, False)]
        self.assertEqual(minimum_qualified_bytes(trials, 0, 900), 800)
        self.assertIsNone(minimum_qualified_bytes(trials, 0, 799))
        result = roster_coverage(5, trials, 1000)
        self.assertEqual(result["minimum_bytes"], [800, None, None, None, None])
        self.assertEqual((result["qualified_tasks"], result["admitted_tasks"], result["coverage"]), (1, 5, 0.2))
        self.assertEqual(roster_coverage(5, [], 1000)["coverage"], 0.)
        for count in (0, 1, True):
            with self.subTest(count=count), self.assertRaises(ValueError):
                roster_coverage(count, trials, 1000)

    def test_typed_evidence_validation(self):
        for args in ((0, 100, None, True), (0, 100, 101, True), (0, 0, None, False),
                     (0, 100, 0, True), (True, 100, 90, True), (0, 100, 90, 1)):
            with self.subTest(args=args), self.assertRaisesRegex(ValueError, "^invalid_frontier_evidence$"):
                ArtifactTrial(*args)
        for cap in (-1, True, 1.5):
            with self.assertRaises(ValueError):
                minimum_qualified_bytes([], 0, cap)

    def test_legacy_cell_coverage_can_bind_an_explicit_roster(self):
        roster = (scheduled("a"), scheduled("b"), scheduled("c"))
        result = observed_coverage([roster[0]], admitted_tasks=roster)
        self.assertEqual((result["tasks"], result["scheduled_cells"]), (3, 3))
        self.assertAlmostEqual(result["coverage"], 1 / 3)
        self.assertEqual(result["cell_counts"]["unresolved"], 2)
        self.assertEqual(observed_coverage([], admitted_tasks=roster)["coverage"], 0.)
        self.assertEqual(qualified_coverage([roster[0]], admitted_tasks=roster)["coverage"], 0.)
        failed = replace(roster[1], cells={"auditor": CellResult("technical_failure")})
        self.assertAlmostEqual(observed_coverage([roster[0], failed], admitted_tasks=roster)["coverage"], 1 / 3)

    def test_explicit_roster_refuses_schedule_changes(self):
        roster = (scheduled("a"), scheduled("b"))
        for tasks, declared in (([scheduled("unknown")], roster), ([roster[0]] * 2, roster),
                                ([roster[0]], [roster[0]] * 2), ([], []),
                                ([replace(roster[0], byte_budget=2000)], roster)):
            with self.subTest(tasks=len(tasks)), self.assertRaises(ValueError):
                observed_coverage(tasks, admitted_tasks=declared)

    def test_wire_validation_and_redaction(self):
        base = fixture("minimum_900_800")["input"]
        mutations = [
            ("protocol_version", None, "invalid_frontier_version"),
            ("protocol_version", True, "invalid_frontier_version"),
            ("protocol_version", 2., "invalid_frontier_version"),
            ("admitted_tasks", [], "invalid_roster"),
            ("admitted_tasks", ["a", "a"], "duplicate_task"),
            ("admitted_tasks", ["a", ["PRIVATE_SOURCE"]], "invalid_roster"),
            ("byte_grid", [1000, 1000], "duplicate_budget"),
            ("byte_grid", [True], "invalid_count"),
            ("byte_grid", [1 << 53], "invalid_count"),
            ("require_monotone", "PRIVATE_SOURCE", "invalid_monotone_policy"),
            ("method_id", None, "invalid_frontier_identity"),
            ("cells", None, "invalid_cells"),
        ]
        for key, value, reason in mutations:
            with self.subTest(key=key, value_type=type(value).__name__):
                result = frontier.evaluate(dict(base, **{key: value}))
                self.assertEqual(result, {"status": "failed", "reason": reason})
                self.assertNotIn("PRIVATE_SOURCE", json.dumps(result))
        for key, value, reason in [
            ("artifact_bytes", 1001, "artifact_exceeds_budget"),
            ("artifact_bytes", None, "missing_qualified_artifact"),
            ("search_cap_bytes", 0, "invalid_count"),
            ("candidate_id", None, "invalid_candidate_identity"),
            ("status", "PRIVATE_SOURCE", "invalid_cell_status"),
            ("status", {}, "invalid_cell_status"),
            ("task_id", "PRIVATE_SOURCE", "unscheduled_task"),
            ("seed_id", "PRIVATE_SOURCE", "frontier_identity_mismatch"),
        ]:
            data = copy.deepcopy(base)
            data["cells"][0][key] = value
            with self.subTest(cell_field=key):
                self.assertEqual(frontier.evaluate(data), {"status": "failed", "reason": reason})
        duplicate = dict(base, cells=[base["cells"][0]] * 2)
        self.assertEqual(frontier.evaluate(duplicate)["reason"], "duplicate_candidate")

    def test_historical_frontier_cases_preserve_v1_results(self):
        for path in sorted(ROOT.glob("*.json")):
            if path.name.startswith("v2_"):
                continue
            case = json.loads(path.read_text())
            self.assertTrue(golden.close(frontier.evaluate(case["input"]), case["expected"]))


if __name__ == "__main__":
    unittest.main()
