"""S01 deterministic synthetic fixtures; no protected sources or release authority."""
from dataclasses import replace
import json
import unittest
from unittest.mock import patch

from research.transfer.privacy import attack_harness as harness
from research.transfer.privacy.attacks import PrivacyError, pin, run_whitebox_disclosure
from research.transfer.privacy.controls import DistanceRecipe, RowView


def view(rows, prefix, role):
    ids = tuple(pin([prefix, i]) for i in range(len(rows)))
    return RowView(tuple(rows), ids, ids, role)


def bundle(rows=(), *, tables=(), endpoints=(), marginals=None, correlations=(), vine=()):
    return harness.WhiteboxBundle(rows, marginals or (((0., 1.),),) * 3,
                                  correlations, vine, tables, endpoints)


def inputs(n=200, leaky=False):
    reference = view(((0., 0., 0.),), "reference", "auxiliary_train")
    development = view(((2., 2., 2.), (4., 4., 4.)), "development", "attack_development")
    evaluation = view(((2., 2., 2.),) * n + ((4., 4., 4.),) * n, "evaluation", "attack_evaluation")
    model = bundle(((2., 2., 2.),)) if leaky else bundle(((0., 0., 0.),))
    return (model, reference, development, (True, False), evaluation, (True,) * n + (False,) * n,
            DistanceRecipe(("numeric",) * 3, (1.,) * 3))


def policy(**changes):
    return harness.HarnessPolicy(**({"target_fpr": .05, "control_trials": 200, "maximum_tpr": .1} | changes))


def run(args=None, **changes):
    return run_whitebox_disclosure(*(inputs() if args is None else args), qid_splits=((0,), (1,)), sensitive=2,
                                  policy=policy(**changes))


class WhiteboxHarnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.clean = run()
        cls.leaky = run(inputs(leaky=True))

    def test_every_probe_detects_every_independently_planted_control(self):
        self.assertEqual(set(self.clean["controls"]), set(harness.CONTROLS))
        for control, probes in self.clean["controls"].items():
            self.assertEqual(set(probes), set(harness.PROBES))
            for probe, result in probes.items():
                with self.subTest(control=control, probe=probe):
                    self.assertTrue(result["detected"])
                    self.assertGreater(result["tpr_lower"], .8)
                    self.assertLessEqual(result["fpr_upper"], .05)
                    self.assertEqual((result["tpr"], result["fpr"]), (1., 0.))

    def test_clean_measurement_can_only_be_empirically_screened_without_authority(self):
        self.assertEqual(self.clean["status"], "empirical_screened_only")
        self.assertEqual(set(self.clean["attacks"]), set(harness.PROBES))
        self.assertEqual(self.clean["formal_dp_status"], "not_dp")
        self.assertEqual(self.clean["legal_release_status"], "unauthorized")
        self.assertFalse(self.clean["release_authorized"])
        self.assertEqual(self.clean["custody_hold"], "D18_HOLD")
        self.assertEqual(self.clean["release_profile"]["empirical_privacy_state"], "inconclusive")
        self.assertEqual(self.clean["release_profile"]["formal_dp_state"], "unassessed")
        self.assertEqual(self.clean["release_profile"]["legal_release_state"], "unassessed")

    def test_actual_copies_are_detected_and_duplicate_targets_are_not_singletons(self):
        self.assertEqual(self.leaky["status"], "disclosure_detected")
        for probe in ("membership_distance", "membership_likelihood_ratio", "membership_parameter_inspection"):
            self.assertEqual(self.leaky["attacks"][probe]["tpr"], 1.)
        self.assertEqual(self.leaky["attacks"]["singling_out"]["tpr"], 0.)

    def test_disabled_probe_fails_calibration_and_never_scores_target_evaluation(self):
        original = harness._scores
        calls = []
        def broken(model, reference, rows, *args):
            calls.append(rows)
            result = original(model, reference, rows, *args)
            result["membership_distance"] = [0.] * len(rows)
            return result
        arguments = inputs()
        with patch.object(harness, "_scores", side_effect=broken):
            result = run(arguments)
        self.assertEqual(result["status"], "calibration_failed")
        self.assertEqual(result["attacks"], {})
        self.assertFalse(any(rows is arguments[4].rows for rows in calls))

    def test_tiny_controls_and_tiny_evaluation_never_screen(self):
        controls = run(control_trials=2)
        self.assertEqual(controls["status"], "inconclusive_underpowered")
        self.assertEqual(controls["attacks"], {})
        final = run(inputs(n=2))
        self.assertEqual(final["status"], "inconclusive_underpowered")
        self.assertTrue(final["attacks"])

    def test_no_unqualified_default_margin_can_screen(self):
        result = run(maximum_tpr=None)
        self.assertEqual(result["status"], "inconclusive_unqualified")

    def test_evaluation_labels_never_recalibrate_thresholds_or_control_results(self):
        args = list(inputs(leaky=True))
        args[5] = tuple(not member for member in args[5])
        result = run(tuple(args))
        self.assertEqual(result["controls"], self.leaky["controls"])
        self.assertEqual(result["attacks"]["membership_distance"]["tpr"], 0.)
        self.assertEqual(result["status"], "inconclusive_underpowered")

    def test_source_values_extrema_paths_and_row_ids_never_appear_in_reports(self):
        args = list(inputs())
        args[0] = bundle(((12345.67891, 23456.78912, 34567.89123),))
        result = run(tuple(args), control_trials=2)
        public = json.dumps(result)
        for value in ("12345.67891", "23456.78912", "34567.89123", args[4].ids[0], "/home/", "private-header"):
            self.assertNotIn(value, public)
            self.assertNotIn(value, repr(args[0]))
        with self.assertRaisesRegex(PrivacyError, "^harness_input_invalid$"):
            run_whitebox_disclosure(**{"private-header": "/home/private/values.csv"})

    def test_roles_unit_overlap_and_clustered_observations_refuse_before_controls(self):
        good = inputs()
        modifications = [(2, replace(good[2], role="attack_evaluation")),
                         (4, replace(good[4], units=(good[2].units[0],) + good[4].units[1:])),
                         (4, replace(good[4], units=(pin("same-unit"),) * len(good[4].rows)))]
        for index, changed in modifications:
            args = list(good)
            args[index] = changed
            with patch.object(harness, "_controls") as controls, self.assertRaises(PrivacyError):
                run(tuple(args))
            controls.assert_not_called()

    def test_qi_splits_sensitive_column_and_work_limits_fail_closed(self):
        for splits, sensitive in ((((0,), (0,)), 2), (((0,), (2,)), 2), (((0,), (1,)), True)):
            with self.assertRaises(PrivacyError):
                run_whitebox_disclosure(*inputs(), qid_splits=splits, sensitive=sensitive, policy=policy())
        args = list(inputs())
        args[-1] = replace(args[-1], max_pairs=1)
        with self.assertRaisesRegex(PrivacyError, "^harness_pair_budget$"):
            run(tuple(args))

    def test_invalid_whitebox_parameters_and_policy_errors_do_not_echo_inputs(self):
        for bad in (True, float("nan"), float("inf"), "/home/private/raw.csv", "private-header"):
            with self.assertRaisesRegex(PrivacyError, "^whitebox_rows_invalid$"):
                bundle(((bad, 0., 0.),))
        for changes in ({"maximum_tpr": True}, {"control_trials": True}, {"detection_floor": 1.},
                        {"maximum_tpr": .9}, {"match_radius": 0.}):
            with self.assertRaises(PrivacyError):
                policy(**changes)
        with self.assertRaisesRegex(PrivacyError, "^whitebox_correlation_invalid$"):
            bundle(correlations=((1., 1., 0.), (1., 1., 0.), (0., 0., 1.)))

    def test_no_sampled_rows_required_to_expose_lookup_and_endpoint_leaks(self):
        point = (2., 2., 2.)
        models = (bundle(tables=(((0, 1, 2), ((point, 1.),)),)),
                  bundle(endpoints=((0., 2.),) * 3))
        reference = harness._reference(((0., 0., 0.),))
        recipe = inputs()[-1]
        for model in models:
            scores = harness._scores(model, reference, (point, (4., 4., 4.)), recipe, ((0,), (1,)), 2, .01)
            for probe in harness.PROBES:
                with self.subTest(probe=probe):
                    self.assertGreater(scores[probe][0], scores[probe][1])

    def test_gaussian_copula_and_pair_correlations_change_sensitive_prediction(self):
        marginal = ((-1., .5), (1., .5))
        row = (1., 0., 99999.12345)  # Hidden truth must never affect the guess.
        recipe = inputs()[-1]
        for use_vine in (False, True):
            guesses = []
            for rho in (-.9, .9):
                matrix = ((1., 0., rho), (0., 1., 0.), (rho, 0., 1.))
                model = bundle(marginals=(marginal,) * 3, correlations=() if use_vine else matrix,
                               vine=((0, 2, rho),) if use_vine else ())
                guesses.append(harness._predict(model, row, (0, 1), 2, recipe, .01))
            self.assertEqual(guesses, [-1., 1.])

    def test_marginals_and_partial_interaction_cells_are_consumed_without_headers(self):
        marginals = (((-1., .2), (1., .8)),) * 3
        model = bundle(marginals=marginals, tables=(((0, 2), (((1., -1.), 1.),)),))
        recipe = inputs()[-1]
        self.assertEqual(harness._predict(model, (1., 1., 0.), (0, 1), 2, recipe, .01), -1.)
        self.assertIn((1., 1., -1.), harness._candidates(model))

    def test_nominal_sensitive_categories_use_equality_and_not_numeric_radius(self):
        point = (1., 2., 7.)
        reference = harness._reference(((0., 0., 8.),))
        model = bundle((point,))
        recipe = DistanceRecipe(("nominal",) * 3, (1., 1., 100.))
        scores = harness._scores(model, reference, (point,), recipe, ((0,), (1,)), 2, .01)
        self.assertEqual(scores["attribute_inference"], [1.])

    def test_bounds_cover_all_declared_probe_control_comparisons(self):
        event_alpha = .05 / (3 * len(harness.PROBES) * (len(harness.CONTROLS) + 1))
        expected = harness.rate_bounds(0, 200, event_alpha)[1]
        self.assertAlmostEqual(self.clean["attacks"]["membership_distance"]["fpr_upper"], expected)


if __name__ == "__main__":
    unittest.main()
