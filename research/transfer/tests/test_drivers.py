import copy
import json
from pathlib import Path
import unittest

import numpy as np
from jsonschema import Draft202012Validator

from research.transfer.drivers import (
    historical_target_correlation, importance_agreement, ordinal_association,
    pearson, permutation_importance,
)
from research.transfer.tdl.drivers import evaluate
from research.transfer.tdl.golden import close, run_case, strict_load

ROOT = Path(__file__).resolve().parents[3]


def fixture(name):
    return strict_load((ROOT / f"tests/golden/drivers/v2/{name}.json").read_text())


class DriverTests(unittest.TestCase):
    def test_paired_d80_shared_failure_corpus_and_result_schema(self):
        schema = json.loads((ROOT / "schemas/driver-result-v1.schema.json").read_text())
        Draft202012Validator.check_schema(schema)
        validator = Draft202012Validator(schema)
        paths = sorted((ROOT / "tests/golden/drivers/v2").glob("v2_*.json"))
        self.assertGreaterEqual(len(paths), 34)
        for path in paths:
            case = strict_load(path.read_text())
            with self.subTest(case=case["case_id"]):
                self.assertIs(case["citable"], False)
                self.assertEqual(case["schema_version"], 2)
                with self.assertRaisesRegex(ValueError, "^invalid_golden_schema$"):
                    run_case(case)
                actual = evaluate(case["input"])
                self.assertTrue(close(actual, case["expected"]), actual)
                validator.validate(actual)
                self.assertFalse(validator.is_valid({**actual, "private_header": 197}))
                if actual["status"] == "complete":
                    changed = copy.deepcopy(actual)
                    changed["components"]["mass"]["lcbs"]["private_extremum"] = 197
                    self.assertFalse(validator.is_valid(changed))
                else:
                    self.assertFalse(validator.is_valid({**actual, "driver_pass": True}))

    def test_explicit_protocol_dispatch_preserves_every_v1_fixture(self):
        for path in (ROOT / "tests/golden/drivers").glob("*.json"):
            case = strict_load(path.read_text())
            if path.name.startswith("v2_"):
                continue
            with self.subTest(case=case["case_id"]):
                self.assertTrue(close(evaluate(case["input"]), case["expected"]))
                self.assertTrue(close(evaluate({**case["input"], "protocol_version": "v1"}), case["expected"]))
        for version in (None, False, 2, "v3", "private_header_197"):
            self.assertEqual(evaluate({"protocol_version": version}),
                             {"status": "failed", "reason": "unsupported_driver_protocol"})

    def test_d80_requires_all_bound_coordinates_and_explicit_positive_floors(self):
        data = fixture("v2_additive")["input"]
        for coordinate in data["inference"]["bounds"]:
            changed = copy.deepcopy(data)
            del changed["inference"]["bounds"][coordinate]
            self.assertEqual(evaluate(changed)["status"], "incomplete")
        for key in ("stability_floor", "importance_floor", "synthetic_strength_floor"):
            for value in (0, -1, True, float("inf"), float("nan"), "private_header_197"):
                changed = {**data, key: value}
                out = evaluate(changed)
                self.assertEqual(out["status"], "failed")
                self.assertNotIn("private_header_197", json.dumps(out))
            changed = dict(data)
            del changed[key]
            self.assertEqual(evaluate(changed)["status"], "incomplete")

    def test_adversarial_evidence_cannot_echo_private_values(self):
        data = fixture("v2_additive")["input"]
        for state in (False, 2, [], {}):
            self.assertEqual(evaluate({**data, "perturbation_status": state})["reason"], "invalid_perturbation_status")
        for key in ("query_sha256", "groups_sha256", "perturbation_sha256", "prepared_view_sha256",
                    "real_importances", "synthetic_importances", "real_control_importances"):
            changed = {**data, key: "CANARY_PRIVATE_HEADER_extremum_197"}
            result = evaluate(changed)
            self.assertEqual(result["status"], "failed")
            self.assertNotIn("CANARY", json.dumps(result))

    def test_additive_u_shape_and_grouped_proxy_mechanisms(self):
        x = np.tile(np.arange(-3.0, 4.0), 24)
        nuisance = np.tile(np.arange(7.0), 24)
        for name, target, prediction in (
            ("additive", 2 * x, lambda v: 2 * np.asarray(v[0])),
            ("u_shape", x ** 2, lambda v: np.asarray(v[0]) ** 2),
        ):
            with self.subTest(mechanism=name):
                def loss(values):
                    return float(np.mean((prediction(values) - target) ** 2))
                report = permutation_importance(loss, (tuple(x), tuple(nuisance)), {"g0": (0,), "g1": (1,)},
                                                learner_id="fixture", query_population="fixture", seed=17, repeats=16)
                self.assertGreater(report["importance"]["g0"], 1.0)
                self.assertEqual(report["importance"]["g1"], 0.0)
        self.assertAlmostEqual(pearson(x, x ** 2), 0.0)
        # The two learners use distinct proxies within the same logical group.
        reports = []
        for index in (0, 1):
            def loss(values):
                return float(np.mean((np.asarray(values[index]) - x) ** 2))
            reports.append(permutation_importance(loss, (tuple(x), tuple(x)), {"g0": (0, 1)},
                                                  learner_id="fixture", query_population="fixture", seed=19, repeats=16))
        self.assertEqual(reports[0]["importance"], reports[1]["importance"])

    def test_xor_importance_group_permutation_and_undefined_agreement(self):
        rng = np.random.default_rng(0)
        x1 = rng.integers(0, 2, 240)
        x2 = rng.integers(0, 2, 240)
        target = np.bitwise_xor(x1, x2).astype(float)
        self.assertLess(abs(pearson(x1, target)), 0.15)
        self.assertLess(abs(pearson(x2, target)), 0.15)
        columns = (tuple(x1), tuple(x2))

        def loss(values):
            pred = np.logical_xor(np.asarray(values[0]) > 0, np.asarray(values[1]) > 0).astype(float)
            return float(np.mean((pred - target) ** 2))

        report = permutation_importance(loss, columns, {"x1": (0,), "x2": (1,), "pair": (0, 1)},
                                        learner_id="xor-learner", query_population="fixture-rows", seed=3, repeats=8)
        self.assertGreater(report["importance"]["x1"], 0.2)
        self.assertGreater(report["importance"]["x2"], 0.2)
        self.assertGreater(report["importance"]["pair"], 0.2)
        self.assertEqual(report["learner_id"], "xor-learner")
        self.assertEqual(report["query_population"], "fixture-rows")
        proxy = historical_target_correlation(x1, target)
        self.assertEqual(proxy["path"], "historical-proxy-v0")
        self.assertIsNone(proxy["learner_id"])
        flat = {"importance": {"x1": 0.0, "x2": 0.0, "pair": 0.0}, "learner_id": "constant", "query_population": "fixture-rows"}
        agreement = importance_agreement(report, flat)
        self.assertIn(agreement["status"], ("low_signal", "tied"))
        self.assertIsNone(agreement["agreement"])
        with self.assertRaises(ValueError):
            ordinal_association((0, 1, 2), (1, 0, 1), scale="nominal")
        color = ((1, 0, 1, 0), (0, 1, 0, 1))

        def one_hot_loss(values):
            return float(sum(left + right for left, right in zip(values[0], values[1])))

        grouped = permutation_importance(one_hot_loss, color, {"color": (0, 1)},
                                         learner_id="group-learner", query_population="fixture-rows", seed=1, repeats=4)
        self.assertAlmostEqual(grouped["importance"]["color"], 0.0)

    def test_driver_agreement_refuses_cross_population_proxy_booleans_and_overflow(self):
        with self.assertRaises(ValueError):
            pearson([1e200, 2e200], [1e200, 3e200])
        with self.assertRaises(ValueError):
            pearson([True, False, True], [1.0, 0.0, 1.0])
        with self.assertRaises(ValueError):
            pearson([[1.0, 2.0], [3.0, 4.0]], [[1.0, 2.0], [3.0, 4.0]])
        with self.assertRaises(ValueError):
            permutation_importance(
                lambda cols: 0.1,
                ((1.0, 2.0, 3.0), (4.0, 5.0, 6.0)),
                {},
                learner_id="l1",
                query_population="holdout",
                seed=0,
            )
        with self.assertRaises(ValueError):
            permutation_importance(
                lambda cols: True,
                ((1.0, 2.0, 3.0), (4.0, 5.0, 6.0)),
                {"g": (0,)},
                learner_id="l1",
                query_population="holdout",
                seed=0,
            )
        valid_left = {
            "path": "drivers-v1",
            "evidence_kind": "permutation_importance",
            "learner_id": "catboost",
            "query_population": "real-holdout-v1",
            "importance": {"f0": 0.1, "f1": 0.1, "f2": 0.5},
        }
        valid_right = {
            "path": "drivers-v1",
            "evidence_kind": "permutation_importance",
            "learner_id": "catboost-syn",
            "query_population": "real-holdout-v1",
            "importance": {"f0": 0.2, "f1": 0.2, "f2": 0.8},
        }
        agreed = importance_agreement(valid_left, valid_right)
        self.assertEqual(agreed["status"], "defined")
        self.assertAlmostEqual(agreed["agreement"], 1.0)
        with self.assertRaises(ValueError):
            importance_agreement(valid_left, {**valid_right, "query_population": "train-fit-v1"})
        with self.assertRaises(ValueError):
            importance_agreement(valid_left, {**valid_right, "path": "historical-proxy-v0"})
        with self.assertRaises(ValueError):
            importance_agreement(valid_left, {**valid_right, "evidence_kind": "absolute_target_correlation"})
        with self.assertRaises(ValueError):
            importance_agreement(valid_left, {**valid_right, "learner_id": ""})
        with self.assertRaises(ValueError):
            importance_agreement(
                valid_left,
                {**valid_right, "importance": {"f0": True, "f1": 0.2, "f2": 0.8}},
            )
        with self.assertRaises(ValueError):
            importance_agreement(valid_left, valid_right, low_signal=-1e-8)


if __name__ == "__main__":
    unittest.main()
