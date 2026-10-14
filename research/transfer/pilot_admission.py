"""Development pilot-plan reader. Plans do not admit or execute a pilot.

The canonical registry is grouped by parent lineage. Aliases and other nested
views stay inside that parent and do not add a source. Unknown or refused
rights and exposure stay unavailable. Owner names and family ids are planning
inputs; the public plan carries opaque group tokens and content digests only.

This is `dope-development-source-pilot-v1` version 1. It is not an N20 freeze,
not statistics-v1, and not a signed admission. Digests use dope-finite-json-v1.
"""

from __future__ import annotations

import math

from .evidence import digest
from .lineage import ExposureState, LineageError, Permission, Registry, Role, Suite


FORMAT = "dope-development-source-pilot-v1"
SUMMARY_FORMAT = "dope-development-source-pilot-summary-v1"
VERSION = 1
STAGES = (
    "fit", "sample", "learner", "selection", "probe", "pfi", "geometry",
    "calibration", "real_reference", "ensemble", "cache_hit",
)
REQUIRED_STAGES = ("fit", "sample", "learner")
PARTITIONS = ("development_fit", "development_selection", "development_query")
GROUP_REASONS = (
    "owner_or_identity_unavailable",
    "source_identity_conflict",
    "lineage_permission_or_exposure_conflict",
    "historical_or_non_development_source",
    "producer_role_contamination",
    "rights_unknown",
    "rights_denied",
    "exposure_unavailable_or_contaminated",
    "lineage_partition_conflict",
)
ATTEMPT_STATUSES = ("success", "failed", "timeout", "unavailable", "not_attempted")
OBSERVED_STATUSES = ("success", "failed", "timeout", "unavailable")


class PilotError(ValueError):
    """Static diagnostic. Source values are never part of the message."""


def _sha(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _number(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _registry_binding(registry: Registry) -> dict:
    families = []
    for family in registry.families():
        families.append({
            "alias_of": family.alias_of,
            "exposure": {
                "model_contamination": family.exposure.model_contamination.value,
                "prior_design": family.exposure.prior_design.value,
            },
            "family_id": family.family_id,
            "meta_partition": family.split.meta_partition,
            "parent_lineage_id": family.parent_lineage_id,
            "relation": family.relation.value,
            "rights": {
                "acquisition": family.rights.acquisition.value,
                "redistribution": family.rights.redistribution.value,
                "training": family.rights.training.value,
            },
            "roles": sorted(role.value for role in family.roles),
            "suite": family.suite.value,
        })
    return {
        "families": families,
        "final_label_custody": registry.final_label_custody,
        "format": "dope-development-pilot-registry-binding-v1",
        "version": VERSION,
    }


def _require_budget(budget: dict) -> dict:
    if not isinstance(budget, dict) or set(budget) != {"stages", "total_seconds"}:
        raise PilotError("pilot_budget_invalid")
    stages = budget["stages"]
    if (
        not isinstance(stages, dict)
        or set(stages) != set(STAGES)
        or not _number(budget["total_seconds"])
        or budget["total_seconds"] <= 0
    ):
        raise PilotError("pilot_budget_invalid")
    for stage in STAGES:
        item = stages[stage]
        if (
            not isinstance(item, dict)
            or set(item) != {"attempts", "seconds_per_attempt"}
            or type(item["attempts"]) is not int
            or not 0 <= item["attempts"] <= 128
            or not _number(item["seconds_per_attempt"])
            or (item["attempts"] and item["seconds_per_attempt"] <= 0)
        ):
            raise PilotError("pilot_stage_budget_invalid")
    if any(stages[stage]["attempts"] == 0 for stage in REQUIRED_STAGES):
        raise PilotError("pilot_required_stage_missing")
    return stages


def build_plan(registry: Registry, owners: dict, budget: dict, *, seed: int) -> dict:
    """Group a loaded registry into one sanitized development plan.

    `execution_authorized`, `scientific_admission`, `confirmation_allowed`, and
    `signed_admission` are constants. A caller boolean cannot change them.
    """
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise PilotError("pilot_seed_invalid")
    if not isinstance(registry, Registry):
        raise PilotError("pilot_registry_invalid")
    if registry.final_label_custody != "external":
        raise PilotError("pilot_external_custody_required")
    if (
        not isinstance(owners, dict)
        or owners.get("format") != "dope-development-pilot-owner-v1"
        or owners.get("version") != VERSION
        or not isinstance(owners.get("sources"), dict)
    ):
        raise PilotError("pilot_owner_manifest_invalid")
    stages = _require_budget(budget)
    try:
        families = registry.families()
        binding = _registry_binding(registry)
    except (LineageError, AttributeError, TypeError):
        raise PilotError("pilot_registry_invalid") from None
    if not families or len(families) > 4096:
        raise PilotError("pilot_registry_size_invalid")
    grouped: dict[str, list] = {}
    for family in families:
        grouped.setdefault(family.parent_lineage_id, []).append(family)
    identities: dict[str, int] = {}
    for parent, row in owners["sources"].items():
        if parent in grouped and isinstance(row, dict) and _sha(row.get("source_identity_sha256")):
            sha = row["source_identity_sha256"]
            identities[sha] = identities.get(sha, 0) + 1
    groups = []
    jobs = []
    for parent, members in sorted(grouped.items()):
        reasons: set[str] = set()
        owner = owners["sources"].get(parent)
        owner_ok = (
            isinstance(owner, dict)
            and set(owner) == {"owner", "source_identity_sha256"}
            and type(owner["owner"]) is str
            and bool(owner["owner"].strip())
            and _sha(owner["source_identity_sha256"])
        )
        if not owner_ok:
            reasons.add("owner_or_identity_unavailable")
        elif identities[owner["source_identity_sha256"]] != 1:
            reasons.add("source_identity_conflict")
        rights = {family.rights for family in members}
        exposure = {family.exposure for family in members}
        if len(rights) != 1 or len(exposure) != 1:
            reasons.add("lineage_permission_or_exposure_conflict")
        if len({family.split.meta_partition for family in members}) != 1:
            reasons.add("lineage_partition_conflict")
        for family in members:
            if family.suite is not Suite.DEVELOPMENT:
                reasons.add("historical_or_non_development_source")
            if Role.AUDIT in family.roles or not any(
                role in family.roles for role in (Role.FIT, Role.SELECTION)
            ):
                reasons.add("producer_role_contamination")
            grants = (family.rights.acquisition, family.rights.redistribution, family.rights.training)
            if Permission.UNKNOWN in grants:
                reasons.add("rights_unknown")
            if Permission.DENIED in grants:
                reasons.add("rights_denied")
            if family.exposure.status() is not ExposureState.CLEAN:
                reasons.add("exposure_unavailable_or_contaminated")
        token = digest(["development-source-group-v1", owner["source_identity_sha256"]]) if owner_ok else None
        partition = None
        if not reasons:
            partition = PARTITIONS[int(digest([seed, token]), 16) % len(PARTITIONS)]
            for stage in STAGES:
                attempts = stages[stage]["attempts"]
                for attempt in range(attempts):
                    jobs.append({
                        "attempt": attempt + 1,
                        "estimated_seconds": stages[stage]["seconds_per_attempt"],
                        "group": token,
                        "stage": stage,
                    })
        groups.append({
            "availability": "planned" if not reasons else "unavailable",
            "group": token,
            "partition": partition,
            "reasons": sorted(reasons),
            "views": len(members),
        })
    if len(jobs) > 4096:
        raise PilotError("pilot_job_budget_exceeded")
    estimate = sum(job["estimated_seconds"] for job in jobs)
    if not math.isfinite(estimate) or estimate > budget["total_seconds"]:
        raise PilotError("pilot_time_budget_exceeded")
    plan = {
        "confirmation_allowed": False,
        "cost_basis": "estimate_not_observation",
        "estimated_seconds": estimate,
        "execution_authorized": False,
        "format": FORMAT,
        "jobs": jobs,
        "observed_seconds": None,
        "owner_manifest_digest": digest(owners),
        "planned_attempts": len(jobs),
        "planned_independent_sources": sum(not group["reasons"] for group in groups),
        "registry_content_digest": digest(binding),
        "scientific_admission": False,
        "scope": "development_only",
        "seed": seed,
        "signed_admission": False,
        "source_groups": groups,
        "total_seconds_cap": budget["total_seconds"],
        "unavailable_sources": sum(bool(group["reasons"]) for group in groups),
        "version": VERSION,
    }
    _require_public(plan)
    return plan


def reconcile_attempts(plan: dict, attempts: list) -> dict:
    """Keep the plan's attempt count as the denominator.

    Observed seconds are caller observations. They are not receipts and they
    do not raise the original cap or authorize execution.
    """
    if (
        not isinstance(plan, dict)
        or plan.get("format") != FORMAT
        or plan.get("version") != VERSION
        or plan.get("execution_authorized") is not False
        or plan.get("signed_admission") is not False
    ):
        raise PilotError("pilot_plan_invalid")
    jobs = plan.get("jobs")
    if not isinstance(jobs, list) or not isinstance(attempts, list):
        raise PilotError("pilot_attempts_invalid")
    planned = []
    for job in jobs:
        if not isinstance(job, dict):
            raise PilotError("pilot_attempts_invalid")
        key = (job.get("group"), job.get("stage"), job.get("attempt"))
        if key in planned:
            raise PilotError("pilot_attempts_invalid")
        planned.append(key)
    observed = {}
    for row in attempts:
        if not isinstance(row, dict) or set(row) != {"attempt", "group", "seconds", "stage", "status"}:
            raise PilotError("pilot_attempt_invalid")
        key = (row["group"], row["stage"], row["attempt"])
        if (
            type(row["attempt"]) is not int
            or key not in planned
            or key in observed
            or row["status"] not in OBSERVED_STATUSES
            or not _number(row["seconds"])
        ):
            raise PilotError("pilot_attempt_invalid")
        observed[key] = {
            "attempt": row["attempt"],
            "group": row["group"],
            "seconds": row["seconds"],
            "stage": row["stage"],
            "status": row["status"],
        }
    counts = {status: 0 for status in ATTEMPT_STATUSES}
    for row in observed.values():
        counts[row["status"]] += 1
    counts["not_attempted"] = len(planned) - len(observed)
    seconds = sum(row["seconds"] for row in observed.values())
    if not math.isfinite(seconds):
        raise PilotError("pilot_attempt_cost_invalid")
    summary = {
        "budget_exceeded": seconds > plan["total_seconds_cap"],
        "confirmation_allowed": False,
        "execution_authorized": False,
        "failure_denominator": len(planned),
        "failure_numerator": counts["failed"] + counts["timeout"],
        "failures": [observed[key] for key in sorted(observed) if observed[key]["status"] in {"failed", "timeout"}],
        "format": SUMMARY_FORMAT,
        "observed_attempts": len(observed),
        "observed_seconds": seconds,
        "original_seconds_cap": plan["total_seconds_cap"],
        "plan_digest": digest(plan),
        "planned_attempts": len(planned),
        "scientific_admission": False,
        "scope": "development_only",
        "signed_admission": False,
        "status_counts": counts,
        "version": VERSION,
    }
    _require_public(summary)
    return summary


def _require_public(document: dict) -> None:
    try:
        from .lineage import assert_public_provenance
        assert_public_provenance(document)
        digest(document)
    except (LineageError, ValueError, TypeError):
        raise PilotError("pilot_output_refused") from None
