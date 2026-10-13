"""Synthetic attacks and adversarial evidence/role regressions; no campaign inputs."""
from dataclasses import FrozenInstanceError, replace
import hashlib
import json
import math
from pathlib import Path
import unittest
from unittest.mock import patch

from research.transfer.privacy.attacks import (AttackCalibration, AttackEvidence, DisclosurePolicy,
    DisclosureThreatModel, EmpiricalPrivacyStatus, PrivacyError, ScoreView, auc, rate_bounds, uncalibrated)
from research.transfer.privacy.controls import (DistanceRecipe, FeatureView, RowView, ShadowMembershipAttack,
    attribute_increment, attribute_inference, control_check, holdout_nndr, membership_features, run_shadow_mia)


def h(value):
    return hashlib.sha256(str(value).encode()).hexdigest()


def view(members, nonmembers, prefix, role="attack_evaluation", units=None, recipe=None):
    ids = tuple(h((prefix, i)) for i in range(len(members) + len(nonmembers)))
    return ScoreView(tuple(members), tuple(nonmembers), ids[:len(members)], ids[len(members):],
                     ids if units is None else units, role, recipe or h("recipe"), h((prefix, "inputs")))


def threat(**updates):
    args = dict(access="row_only", protected_unit="row", artifact_inventory_sha256=h("inventory"),
                decoder_sha256=h("decoder"), ledger_sha256=h("ledger"), auxiliary_sha256=h("aux"),
                sample_budget=100, reference_rows=100)
    return DisclosureThreatModel(**(args | updates))


def calibration(policy=None):
    return AttackCalibration.freeze(view((2., 3.), (0., 0.), "dev", "attack_development"), policy or DisclosurePolicy())


def features(values, prefix, role):
    ids = tuple(h((prefix, i)) for i in range(len(values)))
    return FeatureView(tuple((float(x),) for x in values), ids, ids, role, h("distance"), h(prefix))


def rows(values, prefix, role):
    values = tuple(values)
    ids = tuple(h((prefix, i)) for i in range(len(values)))
    return RowView(tuple(tuple(x) for x in values), ids, ids, role)


class CalibrationTests(unittest.TestCase):
    def test_auc_ties_and_reversible_point_one_are_descriptive(self):
        self.assertEqual(auc((1., 1.), (1., 1.)), .5)
        result = uncalibrated(view((0.,), (-1.,) + (1.,) * 9, "uncalibrated"))
        self.assertEqual((result["raw_auc"], result["reversible_diagnostic_auc"], result["status"]), (.1, .9, "inconclusive"))

    def test_direction_and_policy_are_frozen_before_evaluation(self):
        dev = view((-2., -1.), (1., 2.), "reverse-dev", "attack_development")
        cal = AttackCalibration.freeze(dev, DisclosurePolicy())
        before = cal.sha256
        aligned = cal.evaluate(view((-2.,), (2.,), "aligned"), threat()).value
        flipped = cal.evaluate(view((2.,), (-2.,), "flipped"), threat()).value
        self.assertEqual((cal.direction, aligned["oriented_auc"], flipped["oriented_auc"]), (-1, 1., 0.))
        self.assertEqual(cal.sha256, before)
        with self.assertRaises((FrozenInstanceError, AttributeError)):
            cal.direction = 1
        self.assertFalse(hasattr(cal, "__dict__"))
        with self.assertRaises((FrozenInstanceError, AttributeError)):
            cal.policy.comparisons = 0

    def test_exact_zero_fp_resolution_and_multiplicity(self):
        self.assertAlmostEqual(rate_bounds(0, 2995, .05)[1], -math.expm1(math.log(.05) / 2995))
        self.assertGreater(rate_bounds(0, 2994, .05)[1], .001)
        self.assertLessEqual(rate_bounds(0, 2995, .05)[1], .001)
        data = view((2.,) * 1000, (0.,) * 1000, "enough")
        one = calibration().evaluate(data, threat()).value
        many = calibration(DisclosurePolicy(comparisons=100)).evaluate(data, threat()).value
        self.assertTrue(one["low_fpr_resolved"])
        self.assertGreater(many["fpr_bounds"][1], one["fpr_bounds"][1])
        self.assertLess(many["auc_bounds"][0], one["auc_bounds"][0])
        expected = math.sqrt(.5 * (1/1000 + 1/1000) * math.log(6/.05))
        self.assertAlmostEqual(one["auc_bounds"][0], 1 - expected)

    def test_small_zero_fp_cohort_does_not_resolve_one_percent(self):
        result = calibration().evaluate(view((2.,) * 5, (0.,) * 10, "small"), threat()).value
        self.assertEqual(result["fpr"], 0.)
        self.assertFalse(result["low_fpr_resolved"])
        self.assertIn("low_fpr_unresolved", result["reasons"])

    def test_ties_are_not_split_to_satisfy_false_positive_target(self):
        dev = view((1.,) * 20, (0.,) * 20, "ties-dev", "attack_development")
        cal = AttackCalibration.freeze(dev, DisclosurePolicy(target_fpr=.1))
        self.assertGreater(cal.threshold, 0.)
        self.assertEqual(cal.evaluate(view((1.,), (0.,), "ties"), threat()).value["fpr"], 0.)

    def test_calibration_observation_unit_and_recipe_overlap_refuse(self):
        cal = calibration()
        dev = cal.development
        with self.assertRaisesRegex(PrivacyError, "calibration_observation_overlap"):
            cal.evaluate(replace(dev, role="attack_evaluation"), threat())
        with self.assertRaisesRegex(PrivacyError, "calibration_unit_overlap"):
            cal.evaluate(view((2., 2.), (0., 0.), "same-entities", units=dev.units), threat())
        with self.assertRaisesRegex(PrivacyError, "recipe_identity_mismatch"):
            cal.evaluate(view((2.,), (0.,), "changed-recipe", recipe=h("different")), threat())

    def test_clustered_entities_and_repeated_releases_have_no_independent_bounds(self):
        clustered = view((2., 2.), (0., 0.), "clustered", units=(h("m"),) * 2 + (h("n"),) * 2)
        result = calibration().evaluate(clustered, threat(protected_unit="entity")).value
        self.assertEqual(result["independent_units"], 2)
        self.assertIsNone(result["auc_bounds"])
        self.assertIsNone(result["fpr_bounds"])
        self.assertIn("cluster_or_release_inference_required", result["reasons"])
        repeated = calibration().evaluate(view((2.,), (0.,), "repeat"), threat(repeated_releases=2)).value
        self.assertIsNone(repeated["tpr_bounds"])

    def test_whitebox_and_query_threats_are_not_claimed_by_row_scores(self):
        for access in ("white_box", "query_access"):
            result = calibration().evaluate(view((2.,), (0.,), access), threat(access=access)).value
            self.assertIn("access_threat_unmeasured", result["reasons"])
            self.assertFalse(result["release_authorized"])

    def test_no_empirical_result_grants_formal_legal_or_release_status(self):
        evidence = calibration().evaluate(view((2.,) * 1000, (0.,) * 1000, "perfect"), threat())
        self.assertEqual(evidence.status, EmpiricalPrivacyStatus.INCONCLUSIVE)
        self.assertEqual(evidence.value["formal_dp"], "unverified")
        self.assertEqual(evidence.value["legal_route"], "unverified")
        self.assertFalse(evidence.value["release_authorized"])
        changed = evidence.value
        changed["release_authorized"] = True
        self.assertFalse(evidence.value["release_authorized"])

    def test_evidence_closed_decoding_refuses_types_authority_and_inconsistent_counts(self):
        good = calibration().evaluate(view((2.,), (0.,), "closed"), threat()).value
        mutations = [{"extra": 1}, {"version": True}, {"status": "passed"}, {"release_authorized": True},
                     {"formal_dp": "verified"}, {"tpr": True}, {"members": True}, {"members": 0},
                     {"tpr": .3}, {"fpr_bounds": [1., 0.]}, {"raw_auc": float("nan")},
                     {"low_fpr_resolved": True}, {"reasons": ["private source text"]}]
        for mutation in mutations:
            with self.subTest(keys=list(mutation)), self.assertRaises(PrivacyError):
                AttackEvidence(json.dumps(good | mutation).encode())
        with self.assertRaisesRegex(PrivacyError, "duplicate_json_key"):
            AttackEvidence(b'{"version":1,"version":1}')

    def test_errors_and_aggregates_omit_raw_scores_and_observation_identifiers(self):
        private = view((12345.67,), (9876.54,), "private")
        evidence = calibration().evaluate(private, threat())
        for secret in ("12345.67", "9876.54", private.member_ids[0]):
            self.assertNotIn(secret, str(evidence.value))
            self.assertNotIn(secret, repr(private))
        with self.assertRaises(PrivacyError) as error:
            view(("private-field",), (0.,), "invalid")
        self.assertEqual(str(error.exception), "invalid_scores")

    def test_types_nonfinite_empty_classes_and_final_role_refuse(self):
        for bad in (True, float("nan"), float("inf"), "private-field"):
            with self.assertRaises(PrivacyError):
                view((bad,), (0.,), "bad-type")
        with self.assertRaisesRegex(PrivacyError, "missing_class"):
            uncalibrated(view((), (0.,), "empty"))
        with self.assertRaisesRegex(PrivacyError, "development_required"):
            AttackCalibration.freeze(view((1.,), (0.,), "final"), DisclosurePolicy())
        for args in ({"comparisons": True}, {"alpha": 1e-320}, {"target_fpr": True}):
            with self.assertRaises(PrivacyError):
                DisclosurePolicy(**args)
        with self.assertRaises(PrivacyError):
            rate_bounds(True, 10, .05)
        with self.assertRaises(PrivacyError):
            threat(sample_budget=True)

    def test_policy_byte_drift_invalidates_frozen_calibration(self):
        cal = calibration()
        with patch("research.transfer.privacy.attacks._policy_file_hash", return_value=h("changed-policy")):
            with self.assertRaisesRegex(PrivacyError, "policy_identity_mismatch"):
                cal.evaluate(view((2.,), (0.,), "drift"), threat())

    def test_schema_and_policy_declare_unqualified_aggregate_boundary(self):
        root = Path(__file__).resolve().parents[3]
        schema = json.loads((root / "schemas/privacy-evidence-v1.schema.json").read_text())
        evidence = calibration().evaluate(view((2.,), (0.,), "schema"), threat()).value
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(set(schema["required"]), set(evidence))
        policy = json.loads((root / "research/transfer/privacy/policy.json").read_text())
        self.assertIsNone(policy["empirical_pass_margins"])
        self.assertFalse(policy["release_authorized"])


class ShadowAttackTests(unittest.TestCase):
    def setUp(self):
        self.training = features((0., 2.) * 40, "shadow", "attack_training")
        self.training_labels = (False, True) * 40
        self.development = features((0., 2.) * 20, "calibration", "attack_development")
        self.development_labels = (False, True) * 20
        self.evaluation = features((0., 2.) * 1000, "final", "attack_evaluation")
        self.evaluation_labels = (False, True) * 1000

    def test_learned_shadow_mia_detects_planted_signal_at_frozen_one_percent_fpr(self):
        evidence = run_shadow_mia(self.training, self.training_labels, self.development, self.development_labels,
                                 self.evaluation, self.evaluation_labels, h("shadow-recipe"), DisclosurePolicy(), threat())
        value = evidence.value
        self.assertEqual((value["raw_auc"], value["tpr"], value["fpr"]), (1., 1., 0.))
        self.assertEqual(value["low_fpr_target"], .01)
        self.assertTrue(value["low_fpr_resolved"])
        self.assertEqual(value["status"], "inconclusive")

    def test_final_label_reversal_never_refits_shadow_model_or_direction(self):
        args = (self.training, self.training_labels, self.development, self.development_labels, self.evaluation)
        positive = run_shadow_mia(*args, self.evaluation_labels, h("recipe"), DisclosurePolicy(), threat()).value
        negative = run_shadow_mia(*args, tuple(not x for x in self.evaluation_labels), h("recipe"), DisclosurePolicy(), threat()).value
        self.assertEqual(positive["calibration_sha256"], negative["calibration_sha256"])
        self.assertEqual(negative["oriented_auc"], 0.)

    def test_shadow_overlap_and_final_training_role_are_rejected_before_fit(self):
        overlapped = replace(self.evaluation, ids=self.development.ids + self.evaluation.ids[40:])
        cases = [(overlapped, self.training), (self.evaluation, replace(self.training, role="attack_evaluation")),
                 (self.evaluation, replace(self.training, units=self.evaluation.units[:80]))]
        for evaluation, training in cases:
            with patch.object(ShadowMembershipAttack, "fit") as fitter, self.assertRaises(PrivacyError):
                run_shadow_mia(training, self.training_labels, self.development, self.development_labels,
                               evaluation, self.evaluation_labels, h("recipe"), DisclosurePolicy(), threat())
            fitter.assert_not_called()

    def test_shadow_state_is_immutable_and_changed_recipe_changes_identity(self):
        a = ShadowMembershipAttack.fit(self.training, self.training_labels, h("recipe-a"))
        b = ShadowMembershipAttack.fit(self.training, self.training_labels, h("recipe-b"))
        self.assertNotEqual(a.sha256, b.sha256)
        self.assertFalse(hasattr(a, "__dict__"))
        with self.assertRaises((FrozenInstanceError, AttributeError)):
            a.coefficients = (0.,)
        with self.assertRaisesRegex(PrivacyError, "invalid_shadow_state"):
            replace(a, scale=(0.,))
        with self.assertRaisesRegex(PrivacyError, "feature_recipe_mismatch"):
            a.scores(replace(self.evaluation, feature_recipe_sha256=h("changed-feature-recipe")))

    def test_shadow_fit_labels_and_width_are_exact(self):
        for labels in (self.training_labels[:-1], (0, 1) * 40, (True,) * 80):
            with self.assertRaises(PrivacyError):
                ShadowMembershipAttack.fit(self.training, labels, h("recipe"))
        with self.assertRaises(PrivacyError):
            FeatureView(((True,),), (h("i"),), (h("u"),), "attack_training", h("f"), h("input"))

    def test_row_samples_drive_shadow_training_calibration_and_final_scoring(self):
        recipe = DistanceRecipe(("numeric",), (1.,))
        batches = []
        for offset, role in ((0, "attack_training"), (80, "attack_development"), (160, "attack_evaluation")):
            query = rows(((float(offset + i),) for i in (*range(10), *range(20, 30))), f"query-{offset}", role)
            released = rows(((float(offset + i),) for i in range(10)), f"sample-{offset}", "released_rows")
            auxiliary = rows(((float(offset + 100 + i),) for i in range(10)), f"aux-{offset}", "auxiliary_train")
            batches.append(membership_features(query, released, auxiliary, recipe, 10, 10))
        labels = (True,) * 10 + (False,) * 10
        evidence = run_shadow_mia(batches[0], labels, batches[1], labels, batches[2], labels,
                                 h("sample-attack"), DisclosurePolicy(), threat(sample_budget=10, reference_rows=10))
        self.assertEqual(evidence.value["oriented_auc"], 1.)
        self.assertFalse(evidence.value["low_fpr_resolved"])

    def test_constant_shadow_features_cannot_establish_attack_power(self):
        train = replace(self.training, rows=((0.,),) * len(self.training.rows))
        dev = replace(self.development, rows=((0.,),) * len(self.development.rows))
        final = replace(self.evaluation, rows=((0.,),) * len(self.evaluation.rows))
        evidence = run_shadow_mia(train, self.training_labels, dev, self.development_labels,
                                 final, self.evaluation_labels, h("weak"), DisclosurePolicy(), threat()).value
        self.assertEqual(evidence["raw_auc"], .5)
        self.assertFalse(control_check(evidence, evidence, .9, .1)["positive_detected"])


class ControlTests(unittest.TestCase):
    def test_planted_copy_weak_and_population_controls_need_power_and_matching(self):
        cal = calibration()
        positive = cal.evaluate(view((2.,) * 1000, (0.,) * 1000, "copy"), threat()).value
        negative = cal.evaluate(view((0.,) * 1000, (0.,) * 1000, "population"), threat()).value
        result = control_check(positive, negative, .9, .1)
        self.assertTrue(result["positive_detected"])
        self.assertTrue(result["negative_compatible"])
        self.assertEqual(result["status"], "inconclusive")
        self.assertFalse(control_check(negative, negative, .9, .1)["positive_detected"])
        self.assertFalse(control_check(positive, negative | {"recipe_sha256": h("other")}, .9, .1)["matched"])
        self.assertFalse(control_check(positive, negative | {"nonmembers": 1001}, .9, .1)["matched"])

    def test_population_predictability_is_not_attribute_disclosure(self):
        auxiliary = rows(((float(x), 2. * x) for x in range(10)), "aux", "auxiliary_train")
        released = rows(auxiliary.rows, "syn", "released_rows")
        queries = rows(((float(x), 2. * x) for x in range(10, 20)), "queries", "attack_evaluation")
        result = attribute_inference(released, auxiliary, queries, 1, DistanceRecipe(("numeric",) * 2, (1., 1.)))
        self.assertEqual(result["incremental_mse_improvement"], 0.)
        self.assertEqual(result["artifact_mse"], result["auxiliary_mse"])
        self.assertEqual(result["reference_rows"], 10)
        self.assertIsNone(result["uncertainty"])
        self.assertFalse(result["release_authorized"])
        predictable = (True,) * 90 + (False,) * 10
        self.assertEqual(attribute_increment(predictable, predictable, h("c"), h("c"), h("a"), h("a")), 0.)

    def test_attribute_baseline_refuses_unequal_counts_units_and_hidden_types(self):
        syn = rows(((0., 0.), (1., 1.)), "syn", "released_rows")
        aux = rows(syn.rows, "aux", "auxiliary_train")
        query = rows(((2., 2.),), "query", "attack_evaluation")
        recipe = DistanceRecipe(("numeric",) * 2, (1., 1.))
        for changed, hidden in ((rows(((0., 0.),), "short", "auxiliary_train"), 1), (aux, True)):
            with self.assertRaises(PrivacyError):
                attribute_inference(syn, changed, query, hidden, recipe)
        with self.assertRaises(PrivacyError):
            attribute_inference(syn, aux, replace(query, units=(aux.units[0],)), 1, recipe)
        with self.assertRaises(PrivacyError):
            attribute_increment((True,), (False,), h("large"), h("small"), h("a"), h("a"))

    def test_type_aware_features_use_declared_units_and_matched_counts(self):
        query = rows(((0., 7.),), "query", "attack_development")
        syn = rows(((0., 9.),), "syn", "released_rows")
        aux = rows(((2., 7.),), "aux", "auxiliary_train")
        recipe = DistanceRecipe(("numeric", "nominal"), (2., 1.))
        result = membership_features(query, syn, aux, recipe, 1, 1)
        self.assertAlmostEqual(result.rows[0][1], 0.)
        self.assertEqual(result.role, query.role)
        with self.assertRaisesRegex(PrivacyError, "unmatched_reference_counts"):
            membership_features(query, syn, aux, recipe, 1, 2)
        with self.assertRaises(PrivacyError):
            membership_features(query, syn, aux, recipe, True, True)
        with self.assertRaisesRegex(PrivacyError, "sample_budget_mismatch"):
            membership_features(query, syn, aux, recipe, 2, 2)

    def test_distance_budget_refuses_work_instead_of_truncating(self):
        query = rows(((0.,), (1.,)), "query", "attack_evaluation")
        syn = rows(((0.,), (1.,)), "syn", "released_rows")
        aux = rows(((2.,), (3.,)), "aux", "auxiliary_train")
        with self.assertRaisesRegex(PrivacyError, "distance_budget_exceeded"):
            membership_features(query, syn, aux, DistanceRecipe(("numeric",), (1.,), max_pairs=3), 2, 2)

    def test_holdout_nndr_reports_copies_and_zero_denominator_without_privacy_pass(self):
        syn = rows(((0.,), (1.,)), "syn", "released_rows")
        fit = rows(((0.,), (1.,)), "fit", "attack_training")
        held = rows(((1.,), (3.,)), "held", "attack_evaluation")
        result = holdout_nndr(syn, fit, held, DistanceRecipe(("numeric",), (1.,)))
        self.assertEqual((result["exact_fit_matches"], result["zero_holdout_distances"], result["identified_ratios"]), (2, 1, 1))
        self.assertEqual(result["median_ratio"], 0.)
        self.assertEqual(result["status"], "inconclusive")
        with self.assertRaisesRegex(PrivacyError, "attack_observation_overlap"):
            holdout_nndr(syn, fit, replace(fit, role="attack_evaluation"), DistanceRecipe(("numeric",), (1.,)))


if __name__ == "__main__":
    unittest.main()
