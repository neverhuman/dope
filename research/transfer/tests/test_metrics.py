import hashlib
import json
import math
from pathlib import Path
import unittest

from research.transfer.evidence import digest
from research.transfer.losses import (
    ClassificationContract, DegenerateScale, LossSpace, PairedLosses, TrainingScale, training_variance,
)
from research.transfer.metrics import (
    PAIRED_MEAN_CODE_DIGEST, BoundEvidence, CellResult, FitArtifact, Margin, TaskEvaluation,
    authenticate_bound, bound_binding_record, bound_qualification, harm_summary, loss_summary, observed_coverage,
    qualified_coverage, query_z_mean, tail_regression, weighted_cvar,
)


GOLDEN = json.loads((Path(__file__).resolve().parents[3] / "schemas/metric-golden-v1.json").read_text())


def pairs(real=1., synthetic=1., null=101.):
    return PairedLosses(("q",), ("entity",), (real,), (synthetic,), (null,))


def measured(synthetic=1., bound=None):
    return CellResult("measured", pairs(synthetic=synthetic), TrainingScale(100.), Margin(0., 0.01), bound)


def task(name="t", family="f", cells=None, artifact_bytes=100, artifacts=()):
    return TaskEvaluation(family, name, ("a/n", "b/n", "a/4n", "b/4n"),
                          cells or {k: measured() for k in ("a/n", "b/n", "a/4n", "b/4n")},
                          artifact_bytes, 100, True, artifacts)


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
        with self.assertRaises(ValueError):
            TrainingScale(None, "unknown")

    def test_observed_supported_and_verifier_binding(self):
        b = BoundEvidence(-0.1, "a/n", "protocol", "receipt")
        cells = {k: measured(bound=b) for k in task().required_cells}
        t = task(cells=cells)
        self.assertEqual(observed_coverage([t])["coverage"], 1.)
        self.assertEqual(qualified_coverage([t])["coverage"], 0.)
        self.assertEqual(qualified_coverage([t], verify_bound=lambda t, k, b: b.contrast_id == k)["coverage"], 0.)
        cells = {k: measured(bound=BoundEvidence(-0.1, k, "protocol", "receipt")) for k in t.required_cells}
        point_only = qualified_coverage([task(cells=cells)], verify_bound=lambda t, k, b: b.contrast_id == k)
        self.assertEqual(point_only["coverage"], 0.)
        self.assertEqual(point_only["cell_counts"]["observed_only"], 4)
        self.assertEqual(point_only["cell_counts"]["qualified"], 0)
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
        with self.assertRaises(ValueError) as failed_cell:
            CellResult("technical_failure", pairs())
        self.assertEqual(str(failed_cell.exception), "failed cells require absent loss fields")

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

    def test_coverage_truth_table_separates_completion_harm_and_artifacts(self):
        outside = task(cells={k: measured(11.) for k in task().required_cells})
        report = observed_coverage([outside])
        self.assertEqual(report["cell_counts"]["observed_outside"], 4)
        self.assertEqual(report["cell_counts"]["harmful"], 0)
        self.assertEqual(report["coverage"], 0.)
        self.assertLessEqual(report["lower"], report["upper"])
        self.assertEqual(report["meaning"], "completion")
        self.assertIsNone(report["confidence_interval"])
        bound = BoundEvidence(0.2, "a/n", "protocol", "receipt", lower_z=0.1)
        artifact = FitArtifact("a/n", "fit-0", 10)
        harmful = task(cells={"a/n": measured(11., bound)}, artifacts=(artifact,))
        analysis = {
            "learner_id": "a", "endpoint": "z", "schedule_id": "schedule-1",
            "code_digest": digest({"code": "owner-analysis-v1"}), "rng_id": "rng-1",
            "draws": 20, "multiplicity": "cell",
        }
        record = bound_binding_record(harmful, "a/n", bound, analysis)
        expected = {**record, "trust": "owner", "binding_digest": digest(record)}
        legacy = {
            "pin_source": "external", "task_id": "t", "contrast_id": "a/n",
            "protocol_id": "protocol", "receipt_id": "receipt", "upper_z": 0.2, "lower_z": 0.1,
        }
        self.assertEqual(qualified_coverage(
            [harmful], verify_bound=lambda task, key, bound: authenticate_bound(task, key, bound, expected),
        )["cell_counts"]["harmful"], 1)
        self.assertFalse(authenticate_bound(harmful, "a/n", bound, legacy))
        self.assertFalse(authenticate_bound(harmful, "a/n", BoundEvidence(0.4, "a/n", "protocol", "receipt", 0.1), expected))
        swapped = task(name="other", cells={"a/n": measured(11., bound)}, artifacts=(artifact,))
        self.assertFalse(authenticate_bound(swapped, "a/n", bound, expected))
        shifted = task(cells={"a/n": measured(12., bound)}, artifacts=(artifact,))
        self.assertFalse(authenticate_bound(shifted, "a/n", bound, expected))
        self.assertFalse(authenticate_bound(harmful, "a/n", bound, {**expected, "pin_source": "candidate"}))
        self.assertFalse(authenticate_bound(harmful, "a/n", bound, {**expected, "rng_id": "other-rng"}))
        self.assertFalse(authenticate_bound(harmful, "a/n", bound, {**expected, "contrast_id": ""}))
        recomputed_bound = BoundEvidence(query_z_mean(harmful.cells["a/n"]), "a/n", "protocol", "receipt")
        recomputed_task = task(cells={"a/n": measured(11., recomputed_bound)}, artifacts=(artifact,))
        recomputed_analysis = {
            **analysis, "code_digest": PAIRED_MEAN_CODE_DIGEST,
            "rng_id": "deterministic-paired-mean", "draws": 1,
        }
        recomputed_record = bound_binding_record(recomputed_task, "a/n", recomputed_bound, recomputed_analysis)
        recomputed = {**recomputed_record, "trust": "recomputed", "binding_digest": digest(recomputed_record)}
        self.assertTrue(authenticate_bound(recomputed_task, "a/n", recomputed_bound, recomputed))
        forged = BoundEvidence(recomputed_bound.upper_z + 1, "a/n", "protocol", "receipt")
        self.assertFalse(authenticate_bound(recomputed_task, "a/n", forged, recomputed))
        keys = task().required_cells
        inventory = tuple(FitArtifact(key, "fit-0", 10) for key in keys)
        hidden = inventory[:-1] + (FitArtifact(keys[-1], "fit-0", 1000),)
        self.assertEqual(observed_coverage([task(artifact_bytes=10, artifacts=hidden)])["coverage"], 0.)
        self.assertEqual(observed_coverage([task(artifact_bytes=10, artifacts=inventory[:-1])])["coverage"], 0.)
        cells = {key: measured(0. if key != "b/4n" else 3.) for key in keys}
        tails = tail_regression([task(cells=cells)], catastrophic_threshold=0.05)
        self.assertLess(tails["mean_gap"], 0.)
        self.assertGreater(tails["positive_p95"], 0.)
        self.assertTrue(tails["mean_hides_tail"])
        self.assertEqual(tails["tail_claim"], "disclosed")

    def test_uncalibrated_point_mean_stays_observed_only(self):
        cell_id = "a/n"
        cell = CellResult("measured", pairs(1., 1.), TrainingScale(100.), Margin(0., 0.01),
                          BoundEvidence(-1., cell_id, "protocol", "receipt"))
        self.assertEqual(query_z_mean(cell), -1.)
        artifact = FitArtifact(cell_id, "fit-0", 10)
        evaluation = TaskEvaluation("f", "t", (cell_id,), {cell_id: cell}, 10, 100, True, (artifact,))
        analysis = {
            "learner_id": "a", "endpoint": "z", "schedule_id": "schedule-1",
            "code_digest": PAIRED_MEAN_CODE_DIGEST, "rng_id": "deterministic-paired-mean",
            "draws": 1, "multiplicity": "cell",
        }
        counted_id = bound_binding_record(evaluation, cell_id, cell.bound, analysis)
        self.assertEqual(counted_id["version"], 1)
        self.assertNotIn("artifact_commitment", counted_id)
        self.assertEqual(counted_id["artifact_id"], digest({
            "cell_id": cell_id, "fit_id": "fit-0", "artifact_bytes": 10,
        }))
        record = bound_binding_record(evaluation, cell_id, cell.bound, analysis, version=2)
        self.assertIsNone(record["artifact_id"])
        self.assertEqual(record["artifact_commitment"], "unverified_byte_count")
        self.assertNotEqual(record["artifact_id"], counted_id["artifact_id"])
        expected = {**record, "trust": "recomputed", "binding_digest": digest(record)}
        self.assertTrue(authenticate_bound(evaluation, cell_id, cell.bound, expected))
        decision = bound_qualification(evaluation, cell_id, cell.bound, expected)
        self.assertTrue(decision["authenticated"])
        self.assertFalse(decision["qualifies"])
        self.assertEqual(decision["disposition"], "observed_only")
        self.assertEqual(decision["calibration"], "unmeasured")
        report = qualified_coverage(
            [evaluation], verify_bound=lambda task, key, bound: authenticate_bound(task, key, bound, expected),
        )
        self.assertEqual(report["coverage"], 0.)
        self.assertEqual(report["cell_counts"]["qualified"], 0)
        self.assertEqual(report["cell_counts"]["observed_only"], 1)
        self.assertIsNone(report["confidence_interval"])
        counted = qualified_coverage(
            [evaluation], verify_bound=lambda task, key, bound: bound_qualification(task, key, bound, expected),
        )
        self.assertEqual(counted["cell_counts"]["qualified"], 0)
        content = b"fit-body"
        committed = FitArtifact(cell_id, "fit-0", len(content), content)
        interval = BoundEvidence(-0.2, cell_id, "protocol", "receipt", lower_z=-0.5)
        owner_cell = CellResult("measured", pairs(1., 1.), TrainingScale(100.), Margin(0., 0.01), interval)
        owner_task = TaskEvaluation("f", "t", (cell_id,), {cell_id: owner_cell}, len(content), 100, True, (committed,))
        owner_analysis = {
            "learner_id": "a", "endpoint": "z", "schedule_id": "schedule-1",
            "code_digest": digest({"code": "owner-analysis-v1"}), "rng_id": "rng-1",
            "draws": 1, "multiplicity": "cell",
        }
        owner_record = bound_binding_record(owner_task, cell_id, interval, owner_analysis, version=2)
        self.assertEqual(owner_record["artifact_id"], hashlib.sha256(content).hexdigest())
        self.assertEqual(owner_record["artifact_commitment"], "content_sha256")
        self.assertNotIn("calibration", owner_record)
        owner = {**owner_record, "trust": "owner", "binding_digest": digest(owner_record)}
        unqualified = bound_qualification(owner_task, cell_id, interval, owner)
        self.assertTrue(unqualified["authenticated"])
        self.assertFalse(unqualified["qualifies"])
        self.assertEqual(unqualified["disposition"], "observed_only")
        self.assertEqual(unqualified["calibration"], "unmeasured")
        covered = qualified_coverage(
            [owner_task], verify_bound=lambda task, key, bound: bound_qualification(task, key, bound, owner),
        )
        self.assertEqual(covered["cell_counts"]["qualified"], 0)
        self.assertEqual(covered["cell_counts"]["observed_only"], 1)
        self.assertEqual(covered["coverage"], 0.)
        self.assertIsNone(covered["confidence_interval"])
        for label, extra in (
            ("absent", {}),
            ("null", {"calibration": None}),
            ("unmeasured", {"calibration": "unmeasured"}),
            ("asserted", {"calibration": "measured"}),
        ):
            with self.subTest(calibration=label):
                forged = {**owner, **extra}
                decision = bound_qualification(owner_task, cell_id, interval, forged)
                self.assertFalse(decision["qualifies"])
                self.assertEqual(decision["calibration"], "unmeasured")
                if extra:
                    self.assertFalse(decision["authenticated"])
                    self.assertEqual(decision["disposition"], "unauthenticated")
        forged_flag = {"authenticated": True, "qualifies": True, "calibration": "measured"}
        flagged = qualified_coverage([owner_task], verify_bound=lambda *args: forged_flag)
        self.assertEqual(flagged["cell_counts"]["qualified"], 0)
        self.assertEqual(flagged["coverage"], 0.)
        point_with_content = TaskEvaluation(
            "f", "t", (cell_id,), {cell_id: cell}, len(content), 100, True, (committed,),
        )
        content_record = bound_binding_record(point_with_content, cell_id, cell.bound, analysis, version=2)
        content_expected = {**content_record, "trust": "recomputed", "binding_digest": digest(content_record)}
        self.assertEqual(content_record["artifact_commitment"], "content_sha256")
        self.assertFalse(bound_qualification(point_with_content, cell_id, cell.bound, content_expected)["qualifies"])

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
            self.assertTrue(math.isfinite(contract.losses(["yes"], [[1., 0.]])[0]))

    def test_package_entrypoints_are_distinct(self):
        import research.transfer as transfer
        import research.transfer.stress as stress
        self.assertEqual(transfer.__all__, ["BoundEvidence", "qualified_coverage"])
        self.assertEqual(stress.__all__, ["product_target", "sign_gate_target"])
        self.assertIs(transfer.qualified_coverage, qualified_coverage)
        self.assertTrue(callable(stress.product_target))
        self.assertNotEqual(transfer.__all__, stress.__all__)


if __name__ == "__main__":
    unittest.main()
