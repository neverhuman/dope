import hashlib
import json
import math
from pathlib import Path
import unittest

from research.transfer.calibration_receipt import development_protocol, verify_calibration_receipt
from research.transfer.evidence import digest
from research.transfer.losses import (
    ClassificationContract, DegenerateScale, LossSpace, PairedLosses, TrainingScale, training_variance,
)
from research.transfer.metrics import (
    PAIRED_MEAN_CODE_DIGEST, BoundEvidence, CellResult, FitArtifact, Margin, TaskEvaluation,
    authenticate_bound, bound_binding_record, bound_qualification, control_contrastive_lift,
    driver_attribution_agreement, harm_summary, knowledge_compression_metrics, loss_summary,
    observed_coverage, qualified_coverage, query_z_mean, standardize_numeric_table_01,
    tail_regression, weighted_cvar,
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

    def test_mfs_v4_golden_values_and_standard_01_preprocessing(self):
        v4 = GOLDEN["mfs_v4"]
        ccl_expected = v4["control_contrastive_lift"]
        ccl = control_contrastive_lift(
            ccl_expected["retention"],
            ccl_expected["predictor_only_retention"],
            ccl_expected["independent_marginals_retention"],
            ccl_expected["real_bootstrap_4n_retention"],
        )
        for key in (
            "lift_over_predictor_only",
            "lift_over_independent_marginals",
            "lift_vs_real_bootstrap_4n",
            "contrastive_utility_score",
        ):
            self.assertAlmostEqual(ccl[key], ccl_expected[key])
        self.assertEqual(ccl["positive_predictor_lift"], ccl_expected["positive_predictor_lift"])
        self.assertEqual(ccl["positive_dependence_lift"], ccl_expected["positive_dependence_lift"])

        drv_expected = v4["driver_attribution"]
        drv = driver_attribution_agreement(
            drv_expected["real_importances"],
            drv_expected["synthetic_importances"],
            drv_expected["real_signs"],
            drv_expected["synthetic_signs"],
        )
        self.assertEqual(drv["k"], drv_expected["k"])
        for key in (
            "ndcg_at_k",
            "top_k_jaccard",
            "top_k_overlap",
            "l1_attribution_agreement",
            "sign_agreement",
            "composite_driver_fidelity",
        ):
            self.assertAlmostEqual(drv[key], drv_expected[key])

        kc_expected = v4["knowledge_compression"]
        kc = knowledge_compression_metrics(
            kc_expected["artifact_bytes"],
            kc_expected["encoded_numbers"],
            kc_expected["real_rows"],
            kc_expected["real_columns"],
            kc_expected["byte_limit"],
            kc_expected["retained_utility"],
        )
        for key in (
            "bits_per_original_cell",
            "numbers_per_original_cell",
            "bpc_score",
            "parameter_sparsity_score",
            "log_byte_score",
            "knowledge_compression_score",
            "utility_per_kilobyte",
            "utility_per_encoded_number",
        ):
            self.assertAlmostEqual(kc[key], kc_expected[key])

        fit_table = [["cardiology", 10.0, 100.0], ["oncology", 20.0, 200.0], ["neurology", 30.0, 300.0]]
        eval_table = [["cardiology", 15.0, 250.0], ["pediatrics", 50.0, 50.0]]
        for mode in ("zscore_std_01", "minmax_01", "midrank_quantile_01"):
            standardized = standardize_numeric_table_01(
                fit_table,
                eval_table,
                column_kinds=("categorical", "numeric", "numeric"),
                numeric_mode=mode,
            )
            self.assertEqual(standardized["columns"], 3)
            self.assertFalse(standardized["cleartext_headers_retained"])
            self.assertFalse(standardized["raw_extrema_retained"])
            serialized = json.dumps(standardized, allow_nan=False)
            self.assertNotIn("cardiology", serialized)
            self.assertNotIn("pediatrics", serialized)
            for row in standardized["rows"]:
                self.assertTrue(all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in row))
            large_finite = standardize_numeric_table_01(
                [[-1e200, 1.0], [1e200, 2.0]],
                [[0.0, 1.5]],
                column_kinds=("numeric", "numeric"),
                numeric_mode=mode,
            )
            self.assertAlmostEqual(large_finite["rows"][0][0], 0.5)
        for mode in ("minmax_01", "midrank_quantile_01"):
            with self.assertRaises(ValueError):
                standardize_numeric_table_01(
                    [[-1e308, 1.0], [1e308, 2.0]],
                    [[0.0, 1.5]],
                    column_kinds=("numeric", "numeric"),
                    numeric_mode=mode,
                )
        with self.assertRaises(ValueError):
            standardize_numeric_table_01(
                [[1e308, 1.0], [1.5e308, 2.0]],
                [[-1e308, 1.5]],
                column_kinds=("numeric", "numeric"),
                numeric_mode="zscore_std_01",
            )
        for bad_cat in ("", None, 123, True):
            with self.assertRaises(ValueError):
                standardize_numeric_table_01(
                    [[bad_cat, 10.0, 100.0]],
                    eval_table,
                    column_kinds=("categorical", "numeric", "numeric"),
                )
        with self.assertRaises(ValueError):
            standardize_numeric_table_01(
                ["ab"],
                column_kinds=("categorical", "categorical"),
            )
        with self.assertRaises(ValueError):
            driver_attribution_agreement(
                [1e307] * 500,
                [1e307] * 500,
                [1] * 500,
                [1] * 500,
            )
        with self.assertRaises(ValueError):
            driver_attribution_agreement(
                [0.4, 0.3, 0.2, 0.1],
                [0.4, 0.3, 0.2, 0.1],
                [1, -1],
                [1, -1],
            )
        with self.assertRaises(ValueError):
            driver_attribution_agreement(
                {"f0": 0.4, "f1": 0.6},
                [0.4, 0.6],
                [1, -1],
                [1, -1],
            )
        with self.assertRaises(ValueError):
            driver_attribution_agreement(
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [1, -1, 1, -1],
                [1, -1, 1, -1],
            )
        with self.assertRaises(ValueError):
            driver_attribution_agreement(
                [0.6, 0.4, 0.0, 0.0],
                [0.6, 0.4, 0.0, 0.0],
                [0, 1, 0, 0],
                [0, 1, 0, 0],
            )
        disjoint = driver_attribution_agreement(
            [0.6, 0.4, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.5, 0.5, 0.0],
            [1, -1, 0, 0, 0],
            [1, -1, 1, -1, 0],
        )
        self.assertEqual(disjoint["k"], 2)
        self.assertEqual(disjoint["top_k_jaccard"], 0.0)
        self.assertEqual(disjoint["top_k_overlap"], 0.0)
        self.assertEqual(disjoint["ndcg_at_k"], 0.0)
        self.assertEqual(disjoint["sign_agreement"], 0.0)
        self.assertEqual(disjoint["composite_driver_fidelity"], 0.0)
        with self.assertRaises(ValueError):
            driver_attribution_agreement(
                [0.6, 0.4, 0.0, 0.0],
                [0.0, 0.0, 0.5, 0.5],
                [1, -1, 0, 0],
                [0, 0, 0, -1],
            )
        with self.assertRaises(ValueError):
            driver_attribution_agreement(
                [0.6, 0.4, 0.0, 0.0],
                [0.6, 0.4, 0.0, 0.0],
                [1, -1, 1, 0],
                [1, -1, 1, 0],
            )
        self.assertEqual(
            knowledge_compression_metrics(1536, 48, 1000, 16, 10240, 0.0)["knowledge_compression_score"],
            0.0,
        )
        self.assertEqual(
            knowledge_compression_metrics(12288, 48, 1000, 16, 10240, 0.95)["knowledge_compression_score"],
            0.0,
        )

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
        uncalibrated_harm = qualified_coverage(
            [harmful], verify_bound=lambda task, key, bound: authenticate_bound(task, key, bound, expected),
        )
        self.assertTrue(authenticate_bound(harmful, "a/n", bound, expected))
        self.assertEqual(uncalibrated_harm["cell_counts"]["harmful"], 0)
        self.assertEqual(uncalibrated_harm["cell_counts"]["qualified"], 0)
        self.assertEqual(uncalibrated_harm["cell_counts"]["observed_outside"], 1)
        self.assertEqual(uncalibrated_harm["scheduled_cells"], 4)
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
        self.assertEqual(flagged["cell_counts"]["harmful"], 0)
        self.assertEqual(flagged["coverage"], 0.)
        point_with_content = TaskEvaluation(
            "f", "t", (cell_id,), {cell_id: cell}, len(content), 100, True, (committed,),
        )
        content_record = bound_binding_record(point_with_content, cell_id, cell.bound, analysis, version=2)
        content_expected = {**content_record, "trust": "recomputed", "binding_digest": digest(content_record)}
        self.assertEqual(content_record["artifact_commitment"], "content_sha256")
        self.assertFalse(bound_qualification(point_with_content, cell_id, cell.bound, content_expected)["qualifies"])

    def test_uncalibrated_bounds_are_not_inferential(self):
        def owner_case(synthetic, upper, lower):
            cell_id = "a/n"
            bound = BoundEvidence(upper, cell_id, "protocol", "receipt", lower_z=lower)
            cell = CellResult("measured", pairs(synthetic=synthetic), TrainingScale(100.), Margin(0., 0.01), bound)
            artifact = FitArtifact(cell_id, "fit-0", 10)
            evaluation = TaskEvaluation("f", "t", (cell_id,), {cell_id: cell}, 10, 100, True, (artifact,))
            analysis = {
                "learner_id": "a", "endpoint": "z", "schedule_id": "schedule-1",
                "code_digest": digest({"code": "owner-analysis-v1"}), "rng_id": "rng-1",
                "draws": 1, "multiplicity": "cell",
            }
            record = bound_binding_record(evaluation, cell_id, bound, analysis)
            expected = {**record, "trust": "owner", "binding_digest": digest(record)}
            return evaluation, expected

        def counts(evaluation, verifier):
            report = qualified_coverage([evaluation], verify_bound=verifier)
            return report["cell_counts"], report

        positive, positive_expected = owner_case(11., 0.2, 0.1)
        self.assertGreater(positive.cells["a/n"].z, 0)
        self.assertTrue(authenticate_bound(positive, "a/n", positive.cells["a/n"].bound, positive_expected))
        positive_counts, positive_report = counts(
            positive, lambda task, key, bound: authenticate_bound(task, key, bound, positive_expected),
        )
        self.assertEqual(positive_counts["observed_outside"], 1)
        self.assertEqual(positive_counts["harmful"], 0)
        self.assertEqual(positive_counts["qualified"], 0)
        self.assertEqual(positive_report["coverage"], 0.)
        self.assertEqual(positive_report["scheduled_cells"], 1)
        self.assertIsNone(positive_report["confidence_interval"])
        inside, inside_expected = owner_case(1., 0.2, 0.1)
        self.assertLessEqual(inside.cells["a/n"].z, 0)
        inside_counts, inside_qualified = counts(
            inside, lambda task, key, bound: authenticate_bound(task, key, bound, inside_expected),
        )
        self.assertEqual(inside_counts["observed_only"], 1)
        self.assertEqual(inside_counts["harmful"], 0)
        self.assertEqual(inside_counts["qualified"], 0)
        self.assertEqual(inside_qualified["coverage"], 0.)
        self.assertEqual(observed_coverage([inside])["coverage"], 1.)
        benefit, benefit_expected = owner_case(1., -0.2, -0.5)
        benefit_counts, benefit_report = counts(
            benefit, lambda task, key, bound: bound_qualification(task, key, bound, benefit_expected),
        )
        self.assertTrue(bound_qualification(benefit, "a/n", benefit.cells["a/n"].bound, benefit_expected)["authenticated"])
        self.assertEqual(benefit_counts["observed_only"], 1)
        self.assertEqual(benefit_counts["qualified"], 0)
        self.assertEqual(benefit_counts["harmful"], 0)
        self.assertEqual(benefit_report["coverage"], 0.)
        outside_benefit, _ = owner_case(11., -0.05, -0.2)
        outside_counts, _ = counts(outside_benefit, lambda *args: True)
        self.assertEqual(outside_counts["observed_outside"], 1)
        self.assertEqual(outside_counts["qualified"], 0)
        self.assertEqual(outside_counts["harmful"], 0)
        forged = {
            "authenticated": True, "qualifies": True, "calibration": "measured",
            "harmful": True, "uncertainty": "present",
        }
        forged_counts, _ = counts(positive, lambda *args: forged)
        self.assertEqual(forged_counts["harmful"], 0)
        self.assertEqual(forged_counts["qualified"], 0)
        self.assertEqual(forged_counts["observed_outside"], 1)
        protocol = development_protocol()
        receipt = {
            "format": "dope-calibration-receipt", "version": 1,
            "protocol_id": protocol["protocol_id"], "protocol_digest": digest(protocol),
            "repetitions": protocol["development_validation"]["repetitions"],
            "bootstrap_draws": protocol["development_validation"]["bootstrap_draws"],
            "frozen_for_confirmation": protocol["frozen_for_confirmation"],
            "manifest_digest": None, "tolerances_frozen_before_outcomes": False,
            "mcse_reported": False, "wilson_reported": False,
        }
        receipt["binding_digest"] = digest(receipt)
        decision = verify_calibration_receipt(receipt)
        self.assertIs(decision["authenticates"], True)
        self.assertIs(decision["n13b"], False)
        self.assertIs(decision["qualifies"], False)
        self.assertEqual(decision["calibration"], "unmeasured")
        with_receipt = {**forged, "calibration_receipt": receipt}
        receipt_counts, _ = counts(positive, lambda *args: with_receipt)
        self.assertEqual(receipt_counts["harmful"], 0)
        self.assertEqual(receipt_counts["qualified"], 0)
        self.assertEqual(receipt_counts["observed_only"], 0)
        self.assertEqual(receipt_counts["observed_outside"], 1)
        broken_counts, _ = counts(positive, lambda *args: {**forged, "calibration_receipt": {"version": 1}})
        self.assertEqual(broken_counts["harmful"], 0)
        self.assertEqual(broken_counts["qualified"], 0)
        self.assertEqual(broken_counts["observed_outside"], 1)
        tampered = {**positive_expected, "binding_digest": "b" * 64}
        tampered_counts, _ = counts(
            positive, lambda task, key, bound: authenticate_bound(task, key, bound, tampered),
        )
        self.assertFalse(authenticate_bound(positive, "a/n", positive.cells["a/n"].bound, tampered))
        self.assertEqual(tampered_counts["harmful"], 0)
        self.assertEqual(tampered_counts["qualified"], 0)
        self.assertEqual(tampered_counts["observed_outside"], 1)

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
