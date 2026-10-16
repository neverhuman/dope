"""T-122 blockers 2/4/5/8: fail closed on the declared evaluation design.

These regressions intentionally fail on ffa61ee. L0 owns the metric fixes;
this file changes no implementation and never marks a missing row as a win.
"""
import copy
import json
from pathlib import Path
import unittest

from research.transfer.tdl import aggregate, drivers, frontier, transfer
from research.transfer.tdl.common import MetricError, count, number
from research.transfer.tdl.golden import close, run_case, strict_load

ROOT = Path(__file__).resolve().parents[3]
GOLDEN = ROOT / "tests/golden"


def declared_panel(cb=.95, lgbm=.50, en=.90):
    seeds = [101, 211, 307]
    return {
        "required_families": ["tree", "linear"],
        "required_auditors": {"tree": ["cb", "lgbm"], "linear": ["en"]},
        "required_seeds": seeds,
        "bootstrap_replicates": 9,
        "bootstrap_seed": 1729,
        "alpha": .05,
        "rows": [
            {"source_family": "source1", "dataset": "dataset1", "seed": seed,
             "family": family, "auditor": auditor, "transfer": value}
            for seed in seeds
            for family, auditor, value in (("tree", "cb", cb), ("tree", "lgbm", lgbm), ("linear", "en", en))
        ],
    }


def declared_frontier():
    budgets = [512, 1024, 2048]
    return {
        "byte_grid": budgets,
        "required_families": ["tree", "linear"],
        "required_gates": ["closure", "privacy"],
        "transfer_threshold": .90,
        "driver_threshold": .80,
        "cells": [
            {"budget_bytes": budget, "artifact_bytes": size, "status": "complete",
             "families": ["tree", "linear"], "gates": {"closure": True, "privacy": True},
             "lcb_transfer": score, "driver_fidelity": .95}
            for budget, size, score in zip(budgets, [256, 768, 1536], [.85, .95, .96])
        ],
    }


class T122DeclaredRosterTests(unittest.TestCase):
    def test_complete_declared_panel_retains_the_weak_auditor(self):
        report = aggregate.evaluate(declared_panel())
        self.assertEqual(report["status"], "complete")
        self.assertAlmostEqual(report["family_means"]["tree"], .725, delta=1e-12)
        self.assertAlmostEqual(report["robust_transfer"], .725, delta=1e-12)

    def test_missing_declared_auditor_in_one_seed_is_incomplete(self):
        data = declared_panel()
        data["rows"] = [row for row in data["rows"] if not (row["auditor"] == "lgbm" and row["seed"] == 211)]
        self.assertEqual(aggregate.evaluate(data)["status"], "incomplete")

    def test_omitting_weak_auditor_from_every_seed_cannot_qualify(self):
        data = declared_panel()
        data["rows"] = [row for row in data["rows"] if row["auditor"] != "lgbm"]
        self.assertEqual(aggregate.evaluate(data)["status"], "incomplete")

    def test_missing_whole_declared_seed_is_incomplete(self):
        data = declared_panel()
        data["rows"] = [row for row in data["rows"] if row["seed"] != 211]
        self.assertEqual(aggregate.evaluate(data)["status"], "incomplete")

    def test_explicit_null_observation_remains_incomplete(self):
        data = declared_panel()
        data["rows"][0]["transfer"] = None
        self.assertEqual(aggregate.evaluate(data)["status"], "incomplete")


class T122TransferClippingTests(unittest.TestCase):
    def test_above_one_is_clipped_before_family_mean_and_bootstrap(self):
        report = aggregate.evaluate(declared_panel(cb=1.4, lgbm=.5, en=1.))
        self.assertEqual(report["status"], "complete")
        for value in (report["family_means"]["tree"], report["robust_transfer"], report["lcb"]):
            self.assertAlmostEqual(value, .75, delta=1e-12)
            self.assertLess(value, .90)

    def test_clipping_for_aggregation_preserves_raw_transfer_receipt(self):
        measurement = transfer.evaluate({"null_losses": [10.], "real_losses": [5.],
                                         "synthetic_losses": [3.], "real_gain_lcb": 4.,
                                         "constant_target": False})
        self.assertEqual(measurement["raw_transfer"], 1.4)
        self.assertEqual(measurement["clipped_transfer"], 1.)
        report = aggregate.evaluate(declared_panel(cb=measurement["raw_transfer"], lgbm=.5, en=1.))
        self.assertAlmostEqual(report["family_means"]["tree"], .75, delta=1e-12)


class T122DeclaredByteGridTests(unittest.TestCase):
    def test_omitted_grid_budget_does_not_disappear_from_the_design(self):
        for omitted in (512, 1024, 2048):
            with self.subTest(budget=omitted):
                data = declared_frontier()
                data["cells"] = [cell for cell in data["cells"] if cell["budget_bytes"] != omitted]
                self.assertEqual(frontier.evaluate(data)["status"], "incomplete")

    def test_artifact_exceeding_its_declared_budget_fails(self):
        data = declared_frontier()
        data["cells"][0]["artifact_bytes"] = data["cells"][0]["budget_bytes"] + 1
        report = frontier.evaluate(data)
        self.assertEqual(report["status"], "failed")
        self.assertIsNone(report.get("tdl_bytes"))

    def test_same_artifact_size_at_distinct_budgets_is_allowed(self):
        data = declared_frontier()
        for cell in data["cells"]:
            cell["artifact_bytes"] = 256
        report = frontier.evaluate(data)
        self.assertEqual(report["status"], "qualified")
        self.assertEqual([cell["budget_bytes"] for cell in report["frontier"]], data["byte_grid"])

    def test_largest_tested_size_is_the_declared_grid_endpoint(self):
        report = frontier.evaluate(declared_frontier())
        self.assertEqual(report["status"], "qualified")
        self.assertEqual(report["largest_tested_bytes"], 2048)
        self.assertEqual([cell["budget_bytes"] for cell in report["frontier"]], [512, 1024, 2048])

    def test_selected_budget_and_interval_follow_the_declared_grid(self):
        data = declared_frontier()
        report = frontier.evaluate(data)
        self.assertEqual(report["tdl_budget_bytes"], 1024)
        self.assertEqual(report["tdl_interval_bytes"], [512, 1024])
        data["cells"][0]["lcb_transfer"] = .95
        report = frontier.evaluate(data)
        self.assertEqual(report["tdl_budget_bytes"], 512)
        self.assertEqual(report["tdl_interval_bytes"], [0, 512])

    def test_all_candidates_failing_a_mandatory_gate_are_unqualified(self):
        for gate in ("closure", "privacy"):
            with self.subTest(gate=gate):
                data = declared_frontier()
                for cell in data["cells"]:
                    cell["gates"][gate] = False
                report = frontier.evaluate(data)
                self.assertEqual(report["status"], "unqualified")
                self.assertIsNone(report.get("tdl_bytes"))

    def test_missing_family_manifest_is_incomplete(self):
        for manifest in ("omitted", None, []):
            with self.subTest(families=manifest):
                data = declared_frontier()
                if manifest == "omitted":
                    del data["cells"][0]["families"]
                else:
                    data["cells"][0]["families"] = manifest
                self.assertEqual(frontier.evaluate(data)["status"], "incomplete")

    def test_null_threshold_options_match_absent_options(self):
        data = declared_frontier()
        for key in ("transfer_threshold", "driver_threshold"):
            data.pop(key)
        expected = frontier.evaluate(data)
        for key in ("transfer_threshold", "driver_threshold"):
            with self.subTest(option=key):
                explicit_null = copy.deepcopy(data)
                explicit_null[key] = None
                self.assertEqual(frontier.evaluate(explicit_null), expected)


class T122RustParityTests(unittest.TestCase):
    def test_new_component_goldens_match_exact_rust_states_and_booleans(self):
        paths = sorted(GOLDEN.glob("*/t122_*.json"))
        self.assertGreaterEqual(len(paths), 18)
        for path in paths:
            case = strict_load(path.read_text())
            with self.subTest(case=case["case_id"]):
                self.assertIs(case["citable"], False)
                actual = run_case(case)
                self.assertTrue(close(actual, case["expected"]),
                                f"{case['case_id']}: actual={actual!r}; expected={case['expected']!r}")

    def test_strict_json_matches_rust_rejection_goldens(self):
        suite = json.loads((GOLDEN / "t122-json-admission.json").read_text())
        self.assertIs(suite["citable"], False)
        for case in suite["cases"]:
            with self.subTest(case=case["case_id"]):
                if "expected_error" in case:
                    with self.assertRaises(MetricError) as caught:
                        strict_load(case["input_json"])
                    self.assertEqual(str(caught.exception), case["expected_error"])
                else:
                    actual = strict_load(case["input_json"])
                    self.assertEqual(actual, case["expected_value"])
                    self.assertIs(type(actual["value"]), int)

    def test_direct_number_admission_rejects_unsafe_integer_magnitudes(self):
        maximum = 2**53 - 1
        for accepted in (-maximum, maximum):
            self.assertEqual(number(accepted), float(accepted))
        for rejected in (-(maximum + 1), maximum + 1, 2**64 - 1):
            with self.subTest(value=rejected):
                with self.assertRaisesRegex(MetricError, "^nonfinite_or_invalid_number$"):
                    number(rejected)

    def test_count_admission_retains_exact_safe_integer_boundary(self):
        self.assertEqual(count(2**53 - 1), 2**53 - 1)
        for rejected in (2**53, 2**63, 2**64 - 1):
            with self.subTest(value=rejected):
                with self.assertRaisesRegex(MetricError, "^invalid_count$"):
                    count(rejected)

    def test_mass_threshold_uses_rust_binary64_accumulation(self):
        # The one-ULP difference is below the float tolerance but flips the
        # certificate. Exact boolean and boundary assertions catch it.
        case = strict_load((GOLDEN / "drivers/t122_sequential_sum_threshold.json").read_text())
        report = drivers.evaluate(case["input"])
        self.assertEqual(report["driver_fidelity"], .7846876807403124)
        self.assertIs(report["driver_pass"], True)
