import json
import math
from pathlib import Path
import unittest

from research.transfer.losses import (
    ClassificationContract, DegenerateScale, LossSpace, PairedLosses, TrainingScale, training_variance,
)
from research.transfer.metrics import (
    BoundEvidence, CellResult, Margin, TaskEvaluation, harm_summary, loss_summary,
    observed_coverage, qualified_coverage, weighted_cvar,
)


GOLDEN = json.loads((Path(__file__).resolve().parents[3] / "schemas/metric-golden-v1.json").read_text())


def pairs(real=1., synthetic=1., null=101.):
    return PairedLosses(("q",), ("entity",), (real,), (synthetic,), (null,))


def measured(synthetic=1., bound=None):
    return CellResult("measured", pairs(synthetic=synthetic), TrainingScale(100.), Margin(0., 0.01), bound)


def task(name="t", family="f", cells=None, artifact_bytes=100):
    return TaskEvaluation(family, name, ("a/n", "b/n", "a/4n", "b/4n"),
                          cells or {k: measured() for k in ("a/n", "b/n", "a/4n", "b/4n")},
                          artifact_bytes, 100, True)


class MetricsTests(unittest.TestCase):
    def test_golden_values(self):
        for case in GOLDEN["cases"]:
            with self.subTest(case=case["name"]):
                scale = TrainingScale(case["scale"], DegenerateScale.CONSTANT_OR_NEAR_CONSTANT if case["scale"] is None else None)
                result = loss_summary(pairs(case["real"], case["synthetic"], case["null"]), scale)
                for key in ("gap", "normalized_gap", "retention", "ratio"):
                    if case[key] is None:
                        self.assertIsNone(result[key])
                    else:
                        self.assertAlmostEqual(result[key], case[key])
                self.assertEqual(result["ratio_status"], case["ratio_status"])
                json.dumps(result, allow_nan=False)

    def test_rank_reversal_is_scored_in_raw_units(self):
        case = GOLDEN["rank_reversal"]
        p = PairedLosses.regression(("a", "b", "c"), ("a", "b", "c"), case["targets"],
                                   case["real_predictions"], case["synthetic_predictions"], [0., 0., 0.])
        self.assertAlmostEqual(p.means[0], case["raw_real_mse"])
        self.assertAlmostEqual(p.means[1], case["raw_synthetic_mse"])
        # A fit-only piecewise rank map reverses the raw-unit preference.
        def rank(x):
            return x if x <= 1 else 1 + (x - 1) / 999
        rank_real = sum((rank(y) - rank(r)) ** 2 for y, r in zip(case["targets"], case["real_predictions"]))
        rank_syn = sum((rank(y) - rank(s)) ** 2 for y, s in zip(case["targets"], case["synthetic_predictions"]))
        self.assertGreater(rank_real, rank_syn)
        # A global rank transform of predictions alone can even hide this harm.
        self.assertEqual(sorted(range(3), key=case["synthetic_predictions"].__getitem__), [0, 1, 2])
        with self.assertRaises(ValueError):
            PairedLosses.regression(("q",), ("q",), (1.,), (1.,), (1.,), (1.,), loss_space="rank_mse")

    def test_affine_units_and_error_summary(self):
        y, r, s = [0., 1., 4.], [0., 2., 3.], [1., 3., 6.]
        original = PairedLosses.regression(("a", "b", "c"), ("a", "b", "c"), y, r, s, [1., 1., 1.])
        transform = lambda values: [7 * v + 30 for v in values]
        changed = PairedLosses.regression(("a", "b", "c"), ("a", "b", "c"), transform(y), transform(r), transform(s), transform([1., 1., 1.]))
        a = loss_summary(original, training_variance(y))
        b = loss_summary(changed, training_variance(transform(y)))
        self.assertAlmostEqual(a["normalized_gap"], b["normalized_gap"])
        self.assertAlmostEqual(49 * a["gap"], b["gap"])
        self.assertAlmostEqual(original.regression_errors()["real"]["mae"], 2/3)

    def test_degenerate_and_nonfinite_evidence(self):
        scale = training_variance([5., 5.])
        self.assertIsNone(scale.value)
        self.assertIsNone(training_variance([0., 1e-6], minimum_variance=1e-10).value)
        with self.assertRaises(ValueError):
            Margin(0., 0.1).tolerance(1., scale)
        self.assertEqual(Margin(0., 0., 0.5).tolerance(1., scale), 0.5)
        for value in (math.nan, math.inf, -math.inf, True, -1.):
            with self.subTest(value_type=type(value).__name__), self.assertRaises(ValueError):
                pairs(real=value)
        with self.assertRaises(ValueError):
            PairedLosses(("q", "q"), ("e", "e"), (1., 1.), (2., 2.), (3., 3.))
        self.assertNotIn("entity", repr(pairs()))

    def test_each_required_auditor_and_size_can_fail_task(self):
        for key in task().required_cells:
            cells = {k: measured(11. if k == key else 1.) for k in task().required_cells}
            self.assertEqual(observed_coverage([task(cells=cells)])["coverage"], 0.)
        self.assertEqual(observed_coverage([task(artifact_bytes=101)])["coverage"], 0.)
        self.assertEqual(observed_coverage([task(artifact_bytes=1)])["coverage"], 1.)

    def test_observed_supported_and_verifier_binding(self):
        b = BoundEvidence(-0.1, "a/n", "protocol", "receipt")
        cells = {k: measured(bound=b) for k in task().required_cells}
        t = task(cells=cells)
        self.assertEqual(observed_coverage([t])["coverage"], 1.)
        self.assertEqual(qualified_coverage([t])["coverage"], 0.)
        self.assertEqual(qualified_coverage([t], verify_bound=lambda t, k, b: b.contrast_id == k)["coverage"], 0.)
        cells = {k: measured(bound=BoundEvidence(-0.1, k, "protocol", "receipt")) for k in t.required_cells}
        self.assertEqual(qualified_coverage([task(cells=cells)], verify_bound=lambda t, k, b: b.contrast_id == k)["coverage"], 1.)
        cells["b/n"] = measured(bound=BoundEvidence(0.1, "b/n", "protocol", "receipt"))
        self.assertEqual(qualified_coverage([task(cells=cells)], verify_bound=lambda *args: True)["coverage"], 0.)

    def test_reliability_denominator_and_family_weights(self):
        cells = {"a/n": CellResult("technical_failure"), "b/n": CellResult("unsupported")}
        report = observed_coverage([task(cells=cells)])
        self.assertEqual(report["scheduled_cells"], 4)
        self.assertEqual(report["cell_counts"]["unresolved"], 2)
        self.assertEqual(report["upper"], 0.)
        incomplete = task(cells={"a/n": measured()})
        report = observed_coverage([incomplete])
        self.assertEqual((report["lower"], report["upper"]), (0., 1.))
        harmful = task("bad", "other", {k: measured(11.) for k in task().required_cells})
        tasks = [task(f"variant-{i}") for i in range(10)] + [harmful]
        self.assertEqual(observed_coverage(tasks)["coverage"], 0.5)
        with self.assertRaises(ValueError):
            CellResult("technical_failure", pairs())

    def test_weighted_tail_fraction_and_harm(self):
        case = GOLDEN["weighted_tail"]
        self.assertAlmostEqual(weighted_cvar(case["values"], case["weights"], alpha=case["alpha"]), case["cvar"])
        self.assertAlmostEqual(weighted_cvar([2., 4.], [1., 1.], alpha=0), 3.)
        for weights in ([0., 0.], [1., -1.], [1., math.inf]):
            with self.assertRaises(ValueError):
                weighted_cvar([1., 2.], weights)
        report = harm_summary([task(cells={k: measured(11.) for k in task().required_cells})], catastrophic_threshold=0.05)
        self.assertEqual(report["worst_required_learner_gap"], 0.1)
        self.assertEqual(report["catastrophic_rate"], 1.)
        self.assertAlmostEqual(report["positive_cvar90"], 0.1)

    def test_separate_classification_contract(self):
        case = GOLDEN["classification"]
        for space, key in ((LossSpace.LOG_LOSS, "log_loss"), (LossSpace.BRIER, "brier_sum_over_classes")):
            contract = ClassificationContract(tuple(case["classes"]), space, case["probability_floor"])
            self.assertAlmostEqual(contract.losses(case["labels"], case["probabilities"])[0], case[key])
            p = contract.paired(("q",), ("q",), case["labels"], case["probabilities"], case["probabilities"], [[0.5, 0.5]])
            self.assertEqual(p.loss_space, space)
            with self.assertRaises(ValueError):
                contract.losses(["unknown"], [[0.5, 0.5]])
            with self.assertRaises(ValueError):
                contract.losses(["yes"], [[0.7, 0.7]])


if __name__ == "__main__":
    unittest.main()
