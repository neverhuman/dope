"""Synthetic complete-ledger fixtures; no source access or DP certification."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import unittest

from jsonschema import Draft202012Validator

from research.transfer.privacy.accountant import account
from research.transfer.privacy.attacks import PrivacyError


ROOT = Path(__file__).resolve().parents[3]
ROW = {"unit": "row", "relation": "add_remove", "max_rows_per_unit": 1}
STAGES = ("domain_discovery", "binning", "winsorization_clipping_bounds", "transform_fitting",
          "structure_selection", "parameter_fitting", "candidate_selection", "source_derived_lengths",
          "source_derived_statuses", "source_derived_metrics", "post_summary_private_reads")


def fixture(method="zcdp"):
    cost = {"zcdp": {"rho": .005}, "rdp": {"rdp": [.01, .02, .04]},
            "pld": {"epsilon": .05, "delta": 1e-8}}[method]
    composition = {"accountant": method}
    if method == "rdp":
        composition["rdp_orders"] = [2., 4., 8.]
    return {"format": "dope-dp-profile", "version": 1, "adjacency": deepcopy(ROW),
            "bounds": {"provenance": "public_prespecified", "clipping_norm": "l2",
                       "clipping_bound": 1., "sensitivity_bound": 2., "evidence_sha256": "1" * 64},
            "noise_mechanism": {"name": "gaussian", "scale": 10., "implementation_sha256": "2" * 64},
            "composition": composition, "budget": {"epsilon": 3., "delta": 1e-6},
            "implementation_assumptions": {"prng": "cryptographically_secure_required",
                "prng_implementation_sha256": "3" * 64, "floating_point": "audited_discretization_required",
                "discretization_sha256": "4" * 64},
            "release_history_sha256": "5" * 64, "independent_cryptographic_audit": "unverified_design",
            "steps": [{"stage": stage, "access": "accounted_dp", "adjacency": deepcopy(ROW),
                       "evidence_sha256": "6" * 64, "cost": deepcopy(cost)} for stage in sorted(STAGES)]}


def run(profile, **kwargs):
    return account(profile, expected_adjacency=ROW, **kwargs)


def fixture_audit(profile):
    # Recognizes only this synthetic profile and the exact code/schema bytes.
    digest = hashlib.sha256(json.dumps(profile, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    source = hashlib.sha256((ROOT / "research/transfer/privacy/accountant.py").read_bytes()).hexdigest()
    schema = hashlib.sha256((ROOT / "schemas/dp-profile-v1.schema.json").read_bytes()).hexdigest()
    return lambda c: (set(c) == {"profile_sha256", "accountant_sha256", "schema_sha256", "epsilon", "delta"}
                      and c["profile_sha256"] == digest and c["accountant_sha256"] == source
                      and c["schema_sha256"] == schema and math.isfinite(c["epsilon"]) and c["delta"] < 1)


class AccountantTests(unittest.TestCase):
    def test_zcdp_composition_matches_analytic_fixture_with_upward_margin(self):
        profile = fixture()
        result = run(profile)
        expected = .055 + 2 * math.sqrt(.055 * math.log(1e6))
        self.assertGreaterEqual(result["epsilon"], expected)
        self.assertAlmostEqual(result["epsilon"], expected)
        self.assertEqual(result["delta"], 1e-6)
        self.assertTrue(result["composition_passed"])
        self.assertFalse(result["formal_dp_verified"])

    def test_rdp_adds_at_common_orders_then_converts(self):
        profile = fixture("rdp")
        result = run(profile)
        expected = min(.11 + math.log(1e6), .22 + math.log(1e6) / 3, .44 + math.log(1e6) / 7)
        self.assertGreaterEqual(result["epsilon"], expected)
        self.assertAlmostEqual(result["epsilon"], expected)
        self.assertTrue(result["composition_passed"])

    def test_pld_envelopes_charge_both_epsilon_and_delta(self):
        profile = fixture("pld")
        result = run(profile)
        self.assertAlmostEqual(result["epsilon"], .55)
        self.assertAlmostEqual(result["delta"], 11e-8)
        self.assertTrue(result["composition_passed"])
        for budget in ({"epsilon": .3, "delta": 1e-6}, {"epsilon": 3., "delta": 6e-8}):
            result = run(profile | {"budget": budget})
            self.assertEqual(result["formal_dp_status"], "not_dp")
            self.assertIn("dp_budget_exceeded", result["reasons"])

    def test_all_repeated_releases_and_rejected_candidates_consume_budget(self):
        for method in ("zcdp", "rdp", "pld"):
            profile = fixture(method)
            once = run(profile)
            profile["steps"] *= 2
            twice = run(profile)
            self.assertGreater(twice["epsilon"], once["epsilon"])
            profile["budget"]["epsilon"] = once["epsilon"]
            self.assertFalse(run(profile)["composition_passed"])

    def test_public_prespecified_choices_have_no_private_budget_cost(self):
        profile = fixture()
        for step in profile["steps"]:
            step.update(access="public_prespecified", cost=None)
        result = run(profile)
        self.assertEqual((result["epsilon"], result["delta"]), (0., 0.))
        self.assertTrue(result["composition_passed"])
        self.assertFalse(result["formal_dp_verified"])

    def test_every_unaccounted_private_read_overrides_even_fixture_audit(self):
        # Min/max, quantile edges, clipping extrema, transform fitting,
        # structure search, parameter fitting, selection and observable channels.
        for stage in sorted(STAGES):
            with self.subTest(stage=stage):
                profile = fixture()
                profile["independent_cryptographic_audit"] = "independently_audited"
                step = next(s for s in profile["steps"] if s["stage"] == stage)
                step.update(access="unaccounted_private", cost=None, evidence_sha256=None)
                def forbidden_callback(_):
                    self.fail("private-read refusal must precede proof audit")
                result = run(profile, independent_audit_passed=True, audit_verifier=forbidden_callback)
                self.assertEqual(result["formal_dp_status"], "not_dp")
                self.assertIn("unaccounted_private_read", result["reasons"])
                self.assertFalse(result["formal_dp_verified"])

    def test_claimed_private_mechanism_without_a_cost_is_detected(self):
        for stage in sorted(STAGES):
            profile = fixture()
            next(s for s in profile["steps"] if s["stage"] == stage)["cost"] = None
            result = run(profile)
            self.assertEqual(result["formal_dp_status"], "not_dp")
            self.assertIn("unaccounted_private_read", result["reasons"])

    def test_entity_group_relation_and_contribution_mismatches_refuse(self):
        for adjacency in ({"unit": "entity", "relation": "add_remove", "max_rows_per_unit": 4},
                          {"unit": "user_group", "relation": "add_remove", "max_rows_per_unit": 8},
                          ROW | {"relation": "replace_one"}):
            profile = fixture()
            result = account(profile, expected_adjacency=adjacency)
            self.assertIn("adjacency_mismatch", result["reasons"])
            profile["adjacency"] = deepcopy(adjacency)
            for step in profile["steps"]:
                step["adjacency"] = deepcopy(adjacency)
            self.assertTrue(account(profile, expected_adjacency=adjacency)["composition_passed"])
            self.assertNotIn("adjacency_mismatch", account(profile, expected_adjacency=adjacency)["reasons"])
            profile["steps"][0]["adjacency"] = deepcopy(ROW)
            self.assertEqual(account(profile, expected_adjacency=adjacency)["formal_dp_status"], "not_dp")
        profile = fixture()
        entity = {"unit": "entity", "relation": "add_remove", "max_rows_per_unit": 4}
        profile["adjacency"] = entity
        self.assertIn("adjacency_mismatch", account(profile, expected_adjacency=entity | {"max_rows_per_unit": 5})["reasons"])

    def test_missing_stage_and_privately_chosen_bounds_refuse(self):
        profile = fixture()
        profile["steps"][0] = deepcopy(profile["steps"][1])
        self.assertIn("incomplete_private_read_scope", run(profile)["reasons"])
        profile = fixture()
        profile["bounds"]["provenance"] = "accounted_dp"
        self.assertNotIn("unaccounted_private_bounds", run(profile)["reasons"])
        step = next(s for s in profile["steps"] if s["stage"] == "winsorization_clipping_bounds")
        step.update(access="public_prespecified", cost=None)
        self.assertIn("unaccounted_private_bounds", run(profile)["reasons"])

    def test_audit_flag_status_and_trusted_signoff_are_all_required(self):
        profile = fixture()
        self.assertEqual(run(profile)["formal_dp_status"], "unverified_design")
        self.assertFalse(run(profile, independent_audit_passed=True, audit_verifier=fixture_audit(profile))["formal_dp_verified"])
        profile["independent_cryptographic_audit"] = "independently_audited"
        self.assertFalse(run(profile, audit_verifier=fixture_audit(profile))["formal_dp_verified"])
        self.assertFalse(run(profile, independent_audit_passed=True)["formal_dp_verified"])
        for decision in (False, None, 1, "true"):
            self.assertFalse(run(profile, independent_audit_passed=True,
                                 audit_verifier=lambda _: decision)["formal_dp_verified"])
        result = run(profile, independent_audit_passed=True, audit_verifier=fixture_audit(profile))
        self.assertTrue(result["formal_dp_verified"])
        self.assertTrue(result["independent_audit_passed"])
        self.assertEqual(result["formal_dp_status"], "verified")
        self.assertFalse(result["release_authorized"])

    def test_empirical_screening_can_never_satisfy_formal_dp(self):
        profile = fixture()
        profile["independent_cryptographic_audit"] = "independently_audited"
        result = run(profile, empirical_screened_only=True, independent_audit_passed=True,
                     audit_verifier=fixture_audit(profile))
        self.assertEqual(result["formal_dp_status"], "empirical_screened_only")
        self.assertTrue(result["composition_passed"])
        self.assertFalse(result["formal_dp_verified"])

    def test_audit_binds_history_steps_bounds_assumptions_and_budget(self):
        profile = fixture()
        profile["independent_cryptographic_audit"] = "independently_audited"
        verifier = fixture_audit(profile)
        for key in ("release_history_sha256", "bounds", "implementation_assumptions", "budget", "steps"):
            changed = deepcopy(profile)
            if key == "release_history_sha256":
                changed[key] = "9" * 64
            elif key == "steps":
                changed[key][0]["cost"]["rho"] *= 2
            elif key == "implementation_assumptions":
                changed[key]["discretization_sha256"] = "9" * 64
            else:
                changed[key]["epsilon" if key == "budget" else "clipping_bound"] *= 2
            self.assertFalse(run(changed, independent_audit_passed=True, audit_verifier=verifier)["formal_dp_verified"])

    def test_cost_mismatch_zero_vectors_and_zero_conversion_delta_refuse(self):
        for method in ("rdp", "zcdp"):
            profile = fixture(method)
            profile["budget"]["delta"] = 0.
            self.assertIn("dp_conversion_delta_required", run(profile)["reasons"])
        for cost in ({"rho": .01}, {"rdp": [.1]}, {"rdp": [0., 0., 0.]}):
            profile = fixture("rdp")
            profile["steps"][0]["cost"] = cost
            self.assertIn("invalid_dp_cost", run(profile)["reasons"])
            self.assertEqual(run(profile)["formal_dp_status"], "not_dp")

    def test_errors_and_reports_never_echo_source_canaries(self):
        for secret in ("PRIVATE_HEADER_97531", "/private/source.csv", "-987654.25"):
            profile = fixture()
            profile[secret] = secret
            with self.assertRaisesRegex(PrivacyError, "^invalid_dp_profile$") as error:
                run(profile)
            self.assertNotIn(secret, str(error.exception))
            self.assertIsNone(error.exception.__cause__)
            def broken(_):
                raise ValueError(secret)
            profile = fixture()
            profile["bounds"]["clipping_bound"] = 987654.25
            profile["independent_cryptographic_audit"] = "independently_audited"
            result = run(profile, independent_audit_passed=True, audit_verifier=broken)
            for canary in (secret, "987654.25"):
                self.assertNotIn(canary, json.dumps(result))
            self.assertFalse(result["formal_dp_verified"])

    def test_invalid_nonfinite_types_and_flags_refuse_with_fixed_codes(self):
        for bad in (True, float("nan"), float("inf"), -1., "PRIVATE_HEADER_97531"):
            profile = fixture()
            profile["budget"]["epsilon"] = bad
            with self.assertRaisesRegex(PrivacyError, "^invalid_dp_profile$"):
                run(profile)
        for kwargs in ({"independent_audit_passed": 1}, {"empirical_screened_only": "true"}):
            with self.assertRaisesRegex(PrivacyError, "^invalid_dp_assessment$"):
                run(fixture(), **kwargs)
        for bad in (None, ROW | {"max_rows_per_unit": True}, ROW | {"private_path": "/private/source.csv"}):
            with self.assertRaisesRegex(PrivacyError, "^invalid_dp_adjacency$"):
                account(fixture(), expected_adjacency=bad)

    def test_deterministic_result_and_no_mutation_of_inputs(self):
        profile = fixture()
        before = deepcopy(profile)
        first = run(profile)
        self.assertEqual(first, run(profile))
        self.assertEqual(profile, before)
        first["formal_dp_verified"] = True
        self.assertFalse(run(profile)["formal_dp_verified"])


class ProfileSchemaTests(unittest.TestCase):
    def test_draft_schema_is_closed_at_every_object_and_fixtures_validate(self):
        schema = json.loads((ROOT / "schemas/dp-profile-v1.schema.json").read_text())
        Draft202012Validator.check_schema(schema)
        validator = Draft202012Validator(schema)
        def closed(node):
            if isinstance(node, dict):
                if node.get("type") == "object":
                    self.assertIs(node.get("additionalProperties"), False)
                for child in node.values():
                    closed(child)
            elif isinstance(node, list):
                for child in node:
                    closed(child)
        closed(schema)
        for method in ("rdp", "zcdp", "pld"):
            profile = fixture(method)
            validator.validate(profile)
            for key in profile:
                changed = deepcopy(profile)
                del changed[key]
                self.assertFalse(validator.is_valid(changed), key)
            for field in ("bounds", "noise_mechanism", "composition", "budget", "implementation_assumptions", "adjacency"):
                changed = deepcopy(profile)
                changed[field]["private_header"] = "PRIVATE_HEADER_97531"
                self.assertFalse(validator.is_valid(changed), field)
        for target in ("cost", "adjacency"):
            profile = fixture()
            profile["steps"][0][target]["extra"] = 1
            self.assertFalse(validator.is_valid(profile))


if __name__ == "__main__":
    unittest.main()
