import copy
import json
from pathlib import Path
import unittest

from research.transfer.tdl.drivers import (
    INFORMATIVE_RANK_PROTOCOL, evaluate, rank_agreement, rank_agreement_informative,
)

ROOT = Path(__file__).resolve().parents[3]


class DriverRankSuccessorTests(unittest.TestCase):
    def test_shared_analytic_cases(self):
        cases = json.loads((ROOT / "tests/golden/drivers/informative-v1/rank-cases.json").read_text())
        for case in cases:
            with self.subTest(case=case["name"]):
                self.assertEqual(rank_agreement_informative(case["real"], case["synthetic"]), case["expected"])

    def test_legacy_joint_ties_and_v2_fixture_are_unchanged(self):
        self.assertEqual(rank_agreement([2., 2.], [2., 2.]), 1.)
        case = json.loads((ROOT / "tests/golden/drivers/v2/v2_additive.json").read_text())
        self.assertEqual(evaluate(case["input"]), case["expected"])

    def test_group_input_and_c02_bounds_are_mandatory(self):
        case = json.loads((ROOT / "tests/golden/drivers/v2/v2_additive.json").read_text())
        d = {**case["input"], "protocol_version": INFORMATIVE_RANK_PROTOCOL,
             "importance_unit": "declared_group",
             "group_ids": list(range(len(case["input"]["real_importances"])))}
        out = evaluate(d)
        self.assertEqual(out["status"], "complete")
        self.assertEqual(out["format"], "driver-result-informative-v1")
        for mutation, reason in [({"group_ids": [0] * len(d["group_ids"])}, "invalid_driver_groups"),
                                 ({"importance_unit": "columns"}, "grouped_importance_required"),
                                 ({"inference": None}, "missing_simultaneous_bounds")]:
            result = evaluate({**d, **mutation})
            self.assertEqual(result["reason"], reason)
        bad = copy.deepcopy(d)
        bad["inference"]["bound_method"] = "ratio_only"
        self.assertEqual(evaluate(bad)["reason"], "non_simultaneous_driver_bounds")

    def test_successor_all_real_ties_returns_null_and_not_pass(self):
        case = json.loads((ROOT / "tests/golden/drivers/v2/v2_additive.json").read_text())
        d = copy.deepcopy(case["input"])
        d.update(protocol_version=INFORMATIVE_RANK_PROTOCOL, importance_unit="declared_group",
                 group_ids=[0, 1], required_components=["mass", "rank"],
                 real_importances=[1., 1.], synthetic_importances=[1., 1.],
                 real_control_importances=[1., 1.])
        # Mass is processed before rank; replace its bound estimates for these arms.
        for endpoint, estimate in {"margin": .2, "stability": 1., "real_strength": 2.,
                                   "control_strength": 2., "synthetic_strength": 1.}.items():
            d["inference"]["bounds"]["mass." + endpoint] = {"estimate": estimate, "lower": min(0., estimate)}
        out = evaluate(d)
        self.assertEqual(out["status"], "uninformative")
        self.assertIsNone(out["driver_fidelity"])
        self.assertIsNone(out["driver_pass"])
        self.assertIsNone(out["rank_diagnostics"]["real_synthetic"]["value"])

    def test_adversarial_rank_input_has_only_safe_reason(self):
        for a, b in [([], []), ([1], [1,2]), ([float("nan")], [1]),
                     ([-1], [1]), ([True], [1]), (["CANARY_PRIVATE"], [1])]:
            result = rank_agreement_informative(a, b)
            self.assertEqual(result["status"], "failed")
            self.assertNotIn("CANARY", json.dumps(result))
        self.assertEqual(evaluate({"protocol_version": "private_bad"}),
                         {"status": "failed", "reason": "unsupported_driver_protocol"})


if __name__ == "__main__":
    unittest.main()
