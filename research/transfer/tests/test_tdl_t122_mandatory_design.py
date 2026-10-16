"""T-122 B2/B5: observations cannot declare their own evaluation design."""
import copy
from pathlib import Path
import unittest

from research.transfer.tdl import aggregate, frontier
from research.transfer.tdl.golden import close, run_case, strict_load
from research.transfer.tests.test_tdl_t122_regressions import declared_frontier, declared_panel

GOLDEN = Path(__file__).resolve().parents[3] / "tests/golden"


class T122MandatoryRosterTests(unittest.TestCase):
    def assert_incomplete(self, data):
        report = aggregate.evaluate(data)
        self.assertEqual(report["status"], "incomplete")
        for field in ("robust_transfer", "mean_transfer", "lcb"):
            self.assertIsNone(report.get(field), field)

    def test_absent_auditor_roster_cannot_certify_a_stronger_remaining_panel(self):
        data = declared_panel()
        baseline = aggregate.evaluate(data)
        self.assertEqual(baseline["status"], "complete")
        self.assertAlmostEqual(baseline["robust_transfer"], .725, delta=1e-12)
        del data["required_auditors"]
        for drop_weak in (False, True):
            with self.subTest(drop_weak=drop_weak):
                candidate = copy.deepcopy(data)
                if drop_weak:
                    candidate["rows"] = [r for r in candidate["rows"] if r["auditor"] != "lgbm"]
                self.assert_incomplete(candidate)

    def test_absent_seed_roster_cannot_certify_only_the_stronger_seeds(self):
        data = declared_panel(cb=.95, lgbm=.95, en=.95)
        for row in data["rows"]:
            if row["seed"] == 101:
                row["transfer"] = .20
        baseline = aggregate.evaluate(data)
        self.assertEqual(baseline["status"], "complete")
        self.assertAlmostEqual(baseline["robust_transfer"], .70, delta=1e-12)
        del data["required_seeds"]
        for drop_weak in (False, True):
            with self.subTest(drop_weak=drop_weak):
                candidate = copy.deepcopy(data)
                if drop_weak:
                    candidate["rows"] = [r for r in candidate["rows"] if r["seed"] != 101]
                self.assert_incomplete(candidate)

    def test_absent_both_rosters_cannot_be_inferred_from_remaining_observations(self):
        data = declared_panel()
        del data["required_auditors"]
        del data["required_seeds"]
        data["rows"] = [r for r in data["rows"] if r["auditor"] != "lgbm"]
        self.assert_incomplete(data)

    def test_null_roster_is_also_an_absent_declaration(self):
        for field in ("required_auditors", "required_seeds"):
            with self.subTest(field=field):
                data = declared_panel()
                data[field] = None
                self.assert_incomplete(data)


class T122MandatoryGridTests(unittest.TestCase):
    def assert_incomplete(self, data):
        report = frontier.evaluate(data)
        self.assertEqual(report["status"], "incomplete")
        for field in ("tdl_bytes", "tdl_budget_bytes", "tdl_interval_bytes"):
            self.assertIsNone(report.get(field), field)
        self.assertFalse(report.get("frontier"))

    def test_absent_grid_cannot_be_inferred_from_cell_budgets(self):
        data = declared_frontier()
        self.assertEqual(frontier.evaluate(data)["status"], "qualified")
        del data["byte_grid"]
        self.assert_incomplete(data)

    def test_absent_grid_cannot_be_inferred_from_artifact_sizes(self):
        data = declared_frontier()
        del data["byte_grid"]
        for cell in data["cells"]:
            del cell["budget_bytes"]
        self.assert_incomplete(data)

    def test_omitting_failed_budget_and_grid_cannot_qualify_remaining_cells(self):
        data = declared_frontier()
        data["cells"][0]["status"] = "failed"
        self.assertEqual(frontier.evaluate(data)["status"], "failed")
        data["cells"] = data["cells"][1:]
        self.assertEqual(frontier.evaluate(data)["status"], "incomplete")
        del data["byte_grid"]
        self.assert_incomplete(data)

    def test_null_grid_is_also_an_absent_declaration(self):
        data = declared_frontier()
        data["byte_grid"] = None
        self.assert_incomplete(data)


class T122MandatoryDesignGoldenTests(unittest.TestCase):
    def test_shared_missing_design_goldens(self):
        paths = sorted(GOLDEN.glob("*/t122d_*.json"))
        self.assertGreaterEqual(len(paths), 8, "all mandatory-design regression cases must be shared")
        for path in paths:
            case = strict_load(path.read_text())
            with self.subTest(case=case["case_id"]):
                self.assertIs(case["citable"], False)
                self.assertEqual(case["expected"]["status"], "incomplete")
                actual = run_case(case)
                self.assertTrue(close(actual, case["expected"]),
                                f"{case['case_id']}: actual={actual!r}; expected={case['expected']!r}")
