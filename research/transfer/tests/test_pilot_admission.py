"""Development pilot-plan reader. No pilot is executed."""

from __future__ import annotations

import unittest

from research.transfer.evidence import digest
from research.transfer.lineage import (
    ExposureState, ExposureStatus, Permission, Registry, Relation, RightsRecord, Role,
    SplitManifest, SourceFamily, Suite,
)
from research.transfer.pilot_admission import (
    ATTEMPT_STATUSES, FORMAT, GROUP_REASONS, STAGES, VERSION, PilotError, build_plan,
    reconcile_attempts,
)


SHA_A = "ab" * 32
SHA_B = "cd" * 32
SENTINEL = "99999.5"


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
        "version": VERSION,
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


class PilotAdmissionTests(unittest.TestCase):
    def test_aliases_stay_inside_one_planned_source(self):
        plan = build_plan(registry(), owners(), budget(), seed=0)
        again = build_plan(registry(), owners(), budget(), seed=0)
        self.assertEqual(plan, again)
        self.assertEqual(plan["format"], FORMAT)
        self.assertEqual(plan["version"], VERSION)
        self.assertEqual(plan["scope"], "development_only")
        self.assertFalse(plan["execution_authorized"])
        self.assertFalse(plan["scientific_admission"])
        self.assertFalse(plan["confirmation_allowed"])
        self.assertFalse(plan["signed_admission"])
        self.assertEqual(plan["cost_basis"], "estimate_not_observation")
        self.assertIsNone(plan["observed_seconds"])
        self.assertEqual(plan["planned_independent_sources"], 2)
        self.assertEqual(plan["unavailable_sources"], 0)
        self.assertEqual(len(plan["registry_content_digest"]), 64)
        self.assertEqual(len(plan["owner_manifest_digest"]), 64)
        alpha = next(group for group in plan["source_groups"] if group["views"] == 2)
        beta = next(group for group in plan["source_groups"] if group["views"] == 1)
        self.assertEqual(alpha["availability"], "planned")
        self.assertEqual(alpha["reasons"], [])
        self.assertEqual(alpha["group"], digest(["development-source-group-v1", SHA_A]))
        self.assertEqual(beta["group"], digest(["development-source-group-v1", SHA_B]))
        self.assertEqual(alpha["group"], "7b5092828923b4f278a64719b538b221c1c3c64732b7c6a723d1ca99fea16c7f")
        self.assertEqual(beta["group"], "2e66eecaebb24122c3262852d7c99ec6ff77fcea842600e519c18bc7bf262f1e")
        self.assertEqual(alpha["partition"], "development_selection")
        self.assertEqual(beta["partition"], "development_query")
        self.assertEqual(plan["registry_content_digest"], "6b8f90d3f03e322faacc5c09d9cf7133680a2d17f4f5242715d87a3c4a9e86aa")
        self.assertEqual(plan["owner_manifest_digest"], "7640e30d94723d00fda25e07ed87851d7f6a30a7b526e31595cef659fcd85636")
        self.assertEqual(digest(plan), "c2003e135e0b17349aca273c27f6268541b06635fe5ceb59121a66cf9facec49")
        self.assertEqual(plan["planned_attempts"], 6)
        self.assertEqual(plan["estimated_seconds"], 6)
        self.assertEqual(plan["owner_manifest_digest"], digest(owners()))
        encoded = repr(plan)
        self.assertNotIn("custodian-alpha", encoded)
        self.assertNotIn("dev-lineage-alpha", encoded)
        self.assertNotIn(SENTINEL, encoded)
        self.assertNotIn("family_id", encoded)

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
        plan = build_plan(rows, manifest, budget(total_seconds=100), seed=1)
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
        conflict = build_plan(registry(), conflicted, budget(), seed=0)
        self.assertEqual(conflict["planned_independent_sources"], 0)
        self.assertEqual(
            {tuple(group["reasons"]) for group in conflict["source_groups"]},
            {("source_identity_conflict",)},
        )
        missing = owners()
        del missing["sources"]["dev-lineage-beta"]
        absent = build_plan(registry(), missing, budget(), seed=0)
        self.assertEqual(absent["planned_independent_sources"], 1)
        self.assertIn(
            "owner_or_identity_unavailable",
            next(group["reasons"] for group in absent["source_groups"] if group["group"] is None),
        )

    def test_bad_budget_and_redacted_errors_keep_static_messages(self):
        with self.assertRaises(PilotError) as seed:
            build_plan(registry(), owners(), budget(), seed=True)  # type: ignore[arg-type]
        self.assertEqual(str(seed.exception), "pilot_seed_invalid")
        with self.assertRaises(PilotError) as owner:
            build_plan(registry(), {"format": SENTINEL, "version": VERSION, "sources": {}}, budget(), seed=0)
        self.assertEqual(str(owner.exception), "pilot_owner_manifest_invalid")
        self.assertNotIn(SENTINEL, str(owner.exception))
        with self.assertRaises(PilotError) as stage:
            bad = budget()
            bad["stages"]["fit"]["attempts"] = 0
            build_plan(registry(), owners(), bad, seed=0)
        self.assertEqual(str(stage.exception), "pilot_required_stage_missing")
        with self.assertRaises(PilotError) as cost:
            build_plan(registry(), owners(), budget(total_seconds=1), seed=0)
        self.assertEqual(str(cost.exception), "pilot_time_budget_exceeded")
        self.assertNotIn("1", str(cost.exception))

    def test_reconcile_retains_the_planned_denominator(self):
        plan = build_plan(registry(), owners(), budget(), seed=0)
        group = plan["source_groups"][0]["group"]
        attempts = [{
            "group": group,
            "stage": "fit",
            "attempt": 1,
            "status": "failed",
            "seconds": 4,
        }]
        summary = reconcile_attempts(plan, attempts)
        self.assertEqual(summary["planned_attempts"], 6)
        self.assertEqual(summary["failure_denominator"], 6)
        self.assertEqual(summary["failure_numerator"], 1)
        self.assertEqual(summary["observed_attempts"], 1)
        self.assertEqual(summary["observed_seconds"], 4)
        self.assertEqual(summary["status_counts"]["failed"], 1)
        self.assertEqual(summary["status_counts"]["not_attempted"], 5)
        self.assertEqual(set(summary["status_counts"]), set(ATTEMPT_STATUSES))
        self.assertFalse(summary["execution_authorized"])
        self.assertFalse(summary["signed_admission"])
        self.assertEqual(summary["plan_digest"], digest(plan))
        self.assertEqual(summary["budget_exceeded"], False)
        with self.assertRaises(PilotError) as unknown:
            reconcile_attempts(plan, [{
                "group": SENTINEL,
                "stage": "fit",
                "attempt": 1,
                "status": "failed",
                "seconds": 1,
            }])
        self.assertEqual(str(unknown.exception), "pilot_attempt_invalid")
        self.assertNotIn(SENTINEL, str(unknown.exception))

    def test_too_many_stage_attempts_are_refused_before_a_plan(self):
        rows = []
        manifest = {"format": "dope-development-pilot-owner-v1", "version": VERSION, "sources": {}}
        for index in range(3):
            name = f"dev-lineage-{index}"
            rows.append(family(family_id=name, parent_lineage_id=name, split=split(f"part-{index}")))
            manifest["sources"][name] = {
                "owner": f"custodian-{index}",
                "source_identity_sha256": f"{index + 1:02x}" * 32,
            }
        stages = {stage: {"attempts": 128, "seconds_per_attempt": 1} for stage in STAGES}
        with self.assertRaises(PilotError) as exceeded:
            build_plan(
                Registry(rows, final_label_custody="external"),
                manifest,
                {"stages": stages, "total_seconds": 1_000_000_000},
                seed=0,
            )
        self.assertEqual(str(exceeded.exception), "pilot_job_budget_exceeded")
        self.assertNotIn("custodian-0", str(exceeded.exception))
