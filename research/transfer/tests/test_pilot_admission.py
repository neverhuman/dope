"""Development pilot-plan reader, version 2. No pilot is executed."""

from __future__ import annotations

import hashlib
import hmac
import unittest
from unittest import mock

from research.transfer import pilot_admission
from research.transfer.evidence import canonical_json, digest
from research.transfer.lineage import (
    ExposureState, ExposureStatus, Permission, Registry, Relation, RightsRecord, Role,
    SplitManifest, SourceFamily, Suite,
)
from research.transfer.pilot_admission import (
    ATTEMPT_STATUSES, FORMAT, GROUP_REASONS, MAX_JOBS, STAGES, SUMMARY_FORMAT, TOKEN_SCHEME, VERSION,
    PilotError, build_plan, reconcile_attempts,
)


SHA_A = "ab" * 32
SHA_B = "cd" * 32
SENTINEL = "99999.5"
# Synthetic fixture key: public, for tests and the cross-language fixture only. It is not a custody key.
FIXTURE_KEY = bytes(range(32))
GOLDEN = {
    "group_two_views": "809c8f1f6cefc62e3f98957c95a59554d49a0fa1a0836636d94c24c4c5b3d690",
    "group_one_view": "e2d2b25ed15f24dff10a559cbd26062b8cbcd967788898199666a6c74daab297",
    "registry_content_commitment": "02f8b75a0e62c6c6b4e33d90728da0f501e28e2ae6e7481b153d74eaffb7f843",
    "owner_manifest_commitment": "f87d394d6817d21b85785c1bc0452f7552ac678dd92e5012e5e50aefdff3458d",
    "plan_digest": "d4c3a0b488210b65bafa66474a468530419a7883be07bc95cc0bb678c9c42d5b",
    "summary_digest": "ac8d8c10b0bebed6a65e98d6dedbb8b3fb90cc2f1d4ec88837ed055424c0d9cb",
}


def split(partition: str = "part-alpha") -> SplitManifest:
    return SplitManifest(
        split_id=partition,
        kind="entity_time",
        rationale="Entity and time grain. Audit labels stay external.",
        meta_partition=partition,
        fit_view="producer",
        selection_view="producer",
        audit_view="external",
        labels_in_repository=False,
    )


def family(**overrides: object) -> SourceFamily:
    payload: dict[str, object] = {
        "family_id": "dev-lineage-alpha",
        "parent_lineage_id": "dev-lineage-alpha",
        "relation": Relation.ORIGINAL,
        "suite": Suite.DEVELOPMENT,
        "roles": (Role.FIT, Role.SELECTION),
        "rights": RightsRecord(Permission.GRANTED, Permission.GRANTED, Permission.GRANTED),
        "exposure": ExposureStatus(ExposureState.CLEAN, ExposureState.CLEAN),
        "split": split(),
        "alias_of": None,
    }
    payload.update(overrides)
    return SourceFamily(**payload)  # type: ignore[arg-type]


def budget(**overrides: object) -> dict:
    stages = {
        stage: {"attempts": 1 if stage in {"fit", "sample", "learner"} else 0, "seconds_per_attempt": 1 if stage in {"fit", "sample", "learner"} else 0}
        for stage in STAGES
    }
    document = {"stages": stages, "total_seconds": 20}
    document.update(overrides)
    return document


def owners() -> dict:
    return {
        "format": "dope-development-pilot-owner-v1",
        "version": 1,
        "sources": {
            "dev-lineage-alpha": {"owner": "custodian-alpha", "source_identity_sha256": SHA_A},
            "dev-lineage-beta": {"owner": "custodian-beta", "source_identity_sha256": SHA_B},
        },
    }


def registry() -> Registry:
    original = family()
    alias = family(
        family_id="dev-lineage-alpha-alias",
        relation=Relation.ALIAS,
        alias_of="dev-lineage-alpha",
    )
    other = family(
        family_id="dev-lineage-beta",
        parent_lineage_id="dev-lineage-beta",
        split=split("part-beta"),
    )
    return Registry([original, alias, other], final_label_custody="external")


def plan_of(**overrides: object) -> dict:
    arguments = {"seed": 0, "token_key": FIXTURE_KEY}
    arguments.update(overrides)
    return build_plan(registry(), owners(), budget(), **arguments)  # type: ignore[arg-type]


def fixture_attempts(plan: dict) -> list:
    """Three observations of the six planned jobs; the other three stay not_attempted."""
    alpha = next(group["group"] for group in plan["source_groups"] if group["views"] == 2)
    beta = next(group["group"] for group in plan["source_groups"] if group["views"] == 1)
    return [
        {"group": alpha, "stage": "fit", "attempt": 1, "status": "success", "seconds": 1},
        {"group": alpha, "stage": "sample", "attempt": 1, "status": "failed", "seconds": 2},
        {"group": beta, "stage": "fit", "attempt": 1, "status": "timeout", "seconds": 3},
    ]


class PilotAdmissionTests(unittest.TestCase):
    def test_aliases_stay_inside_one_planned_source(self):
        plan = plan_of()
        self.assertEqual(plan, plan_of())
        self.assertEqual(plan["format"], FORMAT)
        self.assertEqual(plan["version"], VERSION)
        self.assertEqual(plan["group_token_scheme"], TOKEN_SCHEME)
        self.assertEqual(plan["scope"], "development_only")
        self.assertFalse(plan["execution_authorized"])
        self.assertFalse(plan["scientific_admission"])
        self.assertFalse(plan["confirmation_allowed"])
        self.assertFalse(plan["signed_admission"])
        self.assertEqual(plan["cost_basis"], "estimate_not_observation")
        self.assertIsNone(plan["observed_seconds"])
        self.assertEqual(plan["planned_independent_sources"], 2)
        self.assertEqual(plan["unavailable_sources"], 0)
        alpha = next(group for group in plan["source_groups"] if group["views"] == 2)
        beta = next(group for group in plan["source_groups"] if group["views"] == 1)
        self.assertEqual(alpha["availability"], "planned")
        self.assertEqual(alpha["reasons"], [])
        expected = hmac.new(FIXTURE_KEY, canonical_json(["development-source-group-v2", SHA_A]), hashlib.sha256)
        self.assertEqual(alpha["group"], expected.hexdigest())
        self.assertEqual(alpha["group"], GOLDEN["group_two_views"])
        self.assertEqual(beta["group"], GOLDEN["group_one_view"])
        self.assertEqual(alpha["partition"], pilot_admission.PARTITIONS[int(digest([0, alpha["group"]]), 16) % 3])
        self.assertEqual(plan["registry_content_commitment"], GOLDEN["registry_content_commitment"])
        self.assertEqual(plan["owner_manifest_commitment"], GOLDEN["owner_manifest_commitment"])
        self.assertEqual(digest(plan), GOLDEN["plan_digest"])
        self.assertEqual(plan["planned_attempts"], 6)
        self.assertEqual(plan["estimated_seconds"], 6)
        encoded = repr(plan)
        for private in ("custodian-alpha", "dev-lineage-alpha", SENTINEL, "family_id", SHA_A, SHA_B,
                        FIXTURE_KEY.hex()):
            self.assertNotIn(private, encoded)

    def test_public_tokens_resist_a_dictionary_of_public_ids(self):
        names = [f"synthetic-public-id-{index:03d}" for index in range(64)]
        secret = names[17]
        manifest = owners()
        manifest["sources"]["dev-lineage-alpha"]["source_identity_sha256"] = hashlib.sha256(secret.encode()).hexdigest()
        plan = build_plan(registry(), manifest, budget(), seed=0, token_key=FIXTURE_KEY)
        published = {group["group"] for group in plan["source_groups"]}

        def recovered(recipe):
            return [name for name in names if recipe(hashlib.sha256(name.encode()).hexdigest()) in published]

        self.assertEqual(recovered(lambda sha: digest(["development-source-group-v1", sha])), [])
        self.assertEqual(recovered(lambda sha: digest(["development-source-group-v2", sha])), [])
        wrong = bytes(reversed(FIXTURE_KEY))
        self.assertEqual(
            recovered(lambda sha: hmac.new(wrong, canonical_json(["development-source-group-v2", sha]),
                                           hashlib.sha256).hexdigest()),
            [],
        )
        # Only the custody key links a token to a source: the key is the whole boundary.
        self.assertEqual(
            recovered(lambda sha: hmac.new(FIXTURE_KEY, canonical_json(["development-source-group-v2", sha]),
                                           hashlib.sha256).hexdigest()),
            [secret],
        )
        self.assertNotEqual(plan["owner_manifest_commitment"], digest(manifest))
        other = build_plan(registry(), manifest, budget(), seed=0, token_key=wrong)
        self.assertNotEqual(other["owner_manifest_commitment"], plan["owner_manifest_commitment"])
        self.assertNotEqual(other["registry_content_commitment"], plan["registry_content_commitment"])

    def test_token_keys_are_private_and_refused_statically(self):
        for bad in (None, "k" * 64, FIXTURE_KEY[:31], bytes(32), bytes([1, 2]) * 16, bytearray(FIXTURE_KEY)):
            with self.assertRaises(PilotError) as refused:
                plan_of(token_key=bad)
            self.assertEqual(str(refused.exception), "pilot_token_key_invalid")
        long_key = bytes(range(64))
        plan = plan_of(token_key=long_key)
        self.assertNotIn(long_key.hex(), repr(plan))
        self.assertNotIn(long_key.hex()[:32], repr(plan))

    def test_unknown_denied_missing_and_conflicting_sources_stay_unavailable(self):
        unknown = family(
            family_id="dev-lineage-gamma",
            parent_lineage_id="dev-lineage-gamma",
            rights=RightsRecord(Permission.UNKNOWN, Permission.GRANTED, Permission.GRANTED),
            split=split("part-gamma"),
        )
        denied = family(
            family_id="dev-lineage-delta",
            parent_lineage_id="dev-lineage-delta",
            rights=RightsRecord(Permission.DENIED, Permission.GRANTED, Permission.GRANTED),
            split=split("part-delta"),
        )
        exposed = family(
            family_id="dev-lineage-epsilon",
            parent_lineage_id="dev-lineage-epsilon",
            exposure=ExposureStatus(ExposureState.EXPOSED, ExposureState.CLEAN),
            split=split("part-epsilon"),
        )
        historical = family(
            family_id="dev-lineage-zeta",
            parent_lineage_id="dev-lineage-zeta",
            suite=Suite.HISTORICAL,
            split=split("part-zeta"),
        )
        audit = family(
            family_id="dev-lineage-eta",
            parent_lineage_id="dev-lineage-eta",
            roles=(Role.FIT, Role.AUDIT),
            split=split("part-eta"),
        )
        rows = Registry(
            [family(), unknown, denied, exposed, historical, audit],
            final_label_custody="external",
        )
        manifest = owners()
        manifest["sources"]["dev-lineage-gamma"] = {"owner": "custodian-gamma", "source_identity_sha256": "12" * 32}
        manifest["sources"]["dev-lineage-delta"] = {"owner": "custodian-delta", "source_identity_sha256": "34" * 32}
        manifest["sources"]["dev-lineage-epsilon"] = {"owner": "custodian-epsilon", "source_identity_sha256": "56" * 32}
        manifest["sources"]["dev-lineage-zeta"] = {"owner": SENTINEL, "source_identity_sha256": "78" * 32}
        manifest["sources"]["dev-lineage-eta"] = {"owner": "custodian-eta", "source_identity_sha256": "90" * 32}
        plan = build_plan(rows, manifest, budget(total_seconds=100), seed=1, token_key=FIXTURE_KEY)
        self.assertEqual(plan["planned_independent_sources"], 1)
        self.assertEqual(plan["unavailable_sources"], 5)
        reasons = {reason for group in plan["source_groups"] for reason in group["reasons"]}
        self.assertIn("rights_unknown", reasons)
        self.assertIn("rights_denied", reasons)
        self.assertIn("exposure_unavailable_or_contaminated", reasons)
        self.assertIn("historical_or_non_development_source", reasons)
        self.assertIn("producer_role_contamination", reasons)
        self.assertTrue(reasons <= set(GROUP_REASONS))
        planned = [group for group in plan["source_groups"] if group["availability"] == "planned"]
        self.assertEqual(len(planned), 1)
        self.assertEqual(planned[0]["views"], 1)
        self.assertNotIn(SENTINEL, repr(plan))
        conflicted = owners()
        conflicted["sources"]["dev-lineage-beta"]["source_identity_sha256"] = SHA_A
        conflict = build_plan(registry(), conflicted, budget(), seed=0, token_key=FIXTURE_KEY)
        self.assertEqual(conflict["planned_independent_sources"], 0)
        self.assertEqual(
            {tuple(group["reasons"]) for group in conflict["source_groups"]},
            {("source_identity_conflict",)},
        )
        missing = owners()
        del missing["sources"]["dev-lineage-beta"]
        absent = build_plan(registry(), missing, budget(), seed=0, token_key=FIXTURE_KEY)
        self.assertEqual(absent["planned_independent_sources"], 1)
        self.assertIn(
            "owner_or_identity_unavailable",
            next(group["reasons"] for group in absent["source_groups"] if group["group"] is None),
        )

    def test_bad_budget_and_redacted_errors_keep_static_messages(self):
        for seed in (True, -1, 2**53, 2**64 - 1):
            with self.assertRaises(PilotError) as refused:
                plan_of(seed=seed)
            self.assertEqual(str(refused.exception), "pilot_seed_invalid")
        self.assertEqual(plan_of(seed=2**53 - 1)["seed"], 2**53 - 1)
        with self.assertRaises(PilotError) as owner:
            build_plan(registry(), {"format": SENTINEL, "version": 1, "sources": {}}, budget(), seed=0,
                       token_key=FIXTURE_KEY)
        self.assertEqual(str(owner.exception), "pilot_owner_manifest_invalid")
        self.assertNotIn(SENTINEL, str(owner.exception))
        with self.assertRaises(PilotError) as stage:
            bad = budget()
            bad["stages"]["fit"]["attempts"] = 0
            build_plan(registry(), owners(), bad, seed=0, token_key=FIXTURE_KEY)
        self.assertEqual(str(stage.exception), "pilot_required_stage_missing")
        with self.assertRaises(PilotError) as cost:
            build_plan(registry(), owners(), budget(total_seconds=1), seed=0, token_key=FIXTURE_KEY)
        self.assertEqual(str(cost.exception), "pilot_time_budget_exceeded")
        self.assertNotIn("1", str(cost.exception))

    def test_reconcile_keeps_the_planned_denominator_and_non_success(self):
        plan = plan_of()
        published = digest(plan)
        summary = reconcile_attempts(plan, fixture_attempts(plan), expected_plan_digest=published)
        self.assertEqual(summary["format"], SUMMARY_FORMAT)
        self.assertEqual(summary["planned_attempts"], 6)
        self.assertEqual(summary["failure_denominator"], 6)
        self.assertEqual(summary["execution_error_numerator"], 2)
        self.assertEqual(summary["non_success_numerator"], 5)
        self.assertNotIn("failure_numerator", summary)
        self.assertEqual(summary["observed_attempts"], 3)
        self.assertEqual(summary["observed_seconds"], 6)
        self.assertEqual(summary["status_counts"],
                         {"success": 1, "failed": 1, "timeout": 1, "unavailable": 0, "not_attempted": 3})
        self.assertEqual(set(summary["status_counts"]), set(ATTEMPT_STATUSES))
        self.assertEqual(summary["planned_independent_sources"], 2)
        self.assertEqual(summary["unavailable_sources"], 0)
        self.assertFalse(summary["execution_authorized"])
        self.assertFalse(summary["signed_admission"])
        self.assertFalse(summary["scientific_admission"])
        self.assertFalse(summary["confirmation_allowed"])
        self.assertEqual(summary["plan_digest"], published)
        self.assertEqual(summary["budget_exceeded"], False)
        self.assertEqual(digest(summary), GOLDEN["summary_digest"])

    def test_missing_and_unavailable_attempts_never_read_as_success(self):
        plan = plan_of()
        published = digest(plan)
        jobs = plan["jobs"]
        one = [{"group": jobs[0]["group"], "stage": jobs[0]["stage"], "attempt": 1, "status": "success", "seconds": 1}]
        summary = reconcile_attempts(plan, one, expected_plan_digest=published)
        self.assertEqual((summary["non_success_numerator"], summary["failure_denominator"]), (5, 6))
        self.assertEqual(summary["execution_error_numerator"], 0)
        self.assertEqual(summary["status_counts"]["not_attempted"], 5)
        unavailable = [{"group": job["group"], "stage": job["stage"], "attempt": job["attempt"],
                        "status": "unavailable", "seconds": 0} for job in jobs]
        summary = reconcile_attempts(plan, unavailable, expected_plan_digest=published)
        self.assertEqual((summary["non_success_numerator"], summary["failure_denominator"]), (6, 6))
        self.assertEqual(summary["execution_error_numerator"], 0)
        nothing = reconcile_attempts(plan, [], expected_plan_digest=published)
        self.assertEqual(nothing["non_success_numerator"], 6)
        self.assertEqual(nothing["status_counts"]["not_attempted"], 6)

    def test_reconcile_binds_the_published_plan(self):
        plan = plan_of()
        published = digest(plan)
        over = [{"group": job["group"], "stage": job["stage"], "attempt": job["attempt"], "status": "success",
                 "seconds": 100} for job in plan["jobs"]]
        self.assertTrue(reconcile_attempts(plan, over, expected_plan_digest=published)["budget_exceeded"])
        forged = dict(plan, total_seconds_cap=10**9)
        with self.assertRaises(PilotError) as raised:
            reconcile_attempts(forged, over, expected_plan_digest=published)
        self.assertEqual(str(raised.exception), "pilot_plan_digest_mismatch")
        with self.assertRaises(PilotError) as other:
            reconcile_attempts(plan, over, expected_plan_digest="0" * 64)
        self.assertEqual(str(other.exception), "pilot_plan_digest_mismatch")
        for bad in (None, SENTINEL, published.upper(), published[:63]):
            with self.assertRaises(PilotError) as invalid:
                reconcile_attempts(plan, over, expected_plan_digest=bad)  # type: ignore[arg-type]
            self.assertEqual(str(invalid.exception), "pilot_plan_digest_invalid")
        malformed = []
        for field, value in (("attempt", "1"), ("attempt", True), ("group", [SENTINEL]), ("stage", SENTINEL),
                             ("estimated_seconds", float("nan"))):
            changed = dict(plan, jobs=[dict(plan["jobs"][0], **{field: value})] + plan["jobs"][1:])
            malformed.append(changed)
        malformed.append(dict(plan, jobs=plan["jobs"] + [dict(plan["jobs"][0], extra=1)]))
        malformed.append(dict(plan, version=1))
        malformed.append({key: value for key, value in plan.items() if key != "group_token_scheme"})
        malformed.append(dict(plan, jobs=plan["jobs"] * ((MAX_JOBS // 6) + 1)))
        for document in malformed:
            with self.assertRaises(PilotError) as refused:
                reconcile_attempts(document, [], expected_plan_digest=published)
            self.assertEqual(str(refused.exception), "pilot_plan_invalid")
            self.assertNotIn(SENTINEL, str(refused.exception))
        too_many = [dict(over[0])] * (MAX_JOBS + 1)
        with self.assertRaises(PilotError) as crowded:
            reconcile_attempts(plan, too_many, expected_plan_digest=published)
        self.assertEqual(str(crowded.exception), "pilot_attempts_invalid")
        for row in (dict(over[0], group=[SENTINEL]), dict(over[0], attempt=True), dict(over[0], group=SENTINEL),
                    dict(over[0], status=SENTINEL), dict(over[0], seconds=-1)):
            with self.assertRaises(PilotError) as rejected:
                reconcile_attempts(plan, [row], expected_plan_digest=published)
            self.assertEqual(str(rejected.exception), "pilot_attempt_invalid")
            self.assertNotIn(SENTINEL, str(rejected.exception))
        with self.assertRaises(PilotError) as repeated:
            reconcile_attempts(plan, [over[0], over[0]], expected_plan_digest=published)
        self.assertEqual(str(repeated.exception), "pilot_attempt_invalid")

    def test_budgets_are_refused_before_any_job_is_built(self):
        rows = []
        manifest = {"format": "dope-development-pilot-owner-v1", "version": 1, "sources": {}}
        for index in range(3):
            name = f"dev-lineage-{index}"
            rows.append(family(family_id=name, parent_lineage_id=name, split=split(f"part-{index}")))
            manifest["sources"][name] = {
                "owner": f"custodian-{index}",
                "source_identity_sha256": f"{index + 1:02x}" * 32,
            }
        stages = {stage: {"attempts": 128, "seconds_per_attempt": 1} for stage in STAGES}
        built = mock.Mock(side_effect=pilot_admission._job)
        with mock.patch.object(pilot_admission, "_job", built):
            with self.assertRaises(PilotError) as exceeded:
                build_plan(
                    Registry(rows, final_label_custody="external"),
                    manifest,
                    {"stages": stages, "total_seconds": 1_000_000_000},
                    seed=0,
                    token_key=FIXTURE_KEY,
                )
            self.assertEqual(str(exceeded.exception), "pilot_job_budget_exceeded")
            self.assertNotIn("custodian-0", str(exceeded.exception))
            self.assertEqual(built.call_count, 0)
            with self.assertRaises(PilotError) as slow:
                build_plan(registry(), owners(), budget(total_seconds=5), seed=0, token_key=FIXTURE_KEY)
            self.assertEqual(str(slow.exception), "pilot_time_budget_exceeded")
            self.assertEqual(built.call_count, 0)
            huge = budget()
            huge["stages"]["fit"]["seconds_per_attempt"] = 1.7e308
            huge["stages"]["sample"]["seconds_per_attempt"] = 1.7e308
            huge["total_seconds"] = 1.7e308
            with self.assertRaises(PilotError) as overflow:
                build_plan(registry(), owners(), huge, seed=0, token_key=FIXTURE_KEY)
            self.assertEqual(str(overflow.exception), "pilot_time_budget_exceeded")
            self.assertEqual(built.call_count, 0)
            plan = build_plan(registry(), owners(), budget(), seed=0, token_key=FIXTURE_KEY)
            self.assertEqual(built.call_count, plan["planned_attempts"])


if __name__ == "__main__":
    unittest.main()
