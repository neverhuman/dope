"""Development pilot-plan reader. Plans do not admit or execute a pilot.

The canonical registry is grouped by parent lineage. Aliases and other nested
views stay inside that parent and do not add a source. Unknown or refused
rights and exposure stay unavailable. Owner names, family ids and source
identities are planning inputs; the public plan carries keyed group tokens,
keyed commitments, counts and static reasons only.

This is `dope-development-source-pilot-v2` version 2. It is not an N20 freeze,
not statistics-v1, not a signed admission and not a privacy result. Version 1
(n19-pilot-schema-v1) published unkeyed group tokens that a dictionary of
public source names could reverse; version 2 keys every token and commitment
derived from custody inputs with a custody-held secret.

Key boundary: `token_key` is a private input of at least 32 bytes. It is never
written to a plan, a summary or an error. The degenerate-key check refuses only
obviously weak keys; it cannot prove that a key is random or secret, and no
output of this module is a privacy guarantee. Digests use dope-finite-json-v1.
"""

from __future__ import annotations

import hashlib
import hmac
import math

from .evidence import EvidenceError, canonical_json, digest
from .lineage import ExposureState, LineageError, Permission, Registry, Role, Suite


FORMAT = "dope-development-source-pilot-v2"
SUMMARY_FORMAT = "dope-development-source-pilot-summary-v2"
VERSION = 2
OWNER_FORMAT = "dope-development-pilot-owner-v1"
OWNER_VERSION = 1
BINDING_FORMAT = "dope-development-pilot-registry-binding-v1"
BINDING_VERSION = 1
TOKEN_SCHEME = "hmac-sha256-custody-key-v1"
GROUP_DOMAIN = "development-source-group-v2"
OWNER_DOMAIN = "development-owner-manifest-v2"
REGISTRY_DOMAIN = "development-registry-binding-v2"
MIN_TOKEN_KEY_BYTES = 32
MIN_TOKEN_KEY_DISTINCT_BYTES = 16
MAX_SEED = 2**53 - 1
MAX_FAMILIES = 4096
MAX_JOBS = 4096
MAX_STAGE_ATTEMPTS = 128
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
PLAN_FIELDS = frozenset({
    "confirmation_allowed", "cost_basis", "estimated_seconds", "execution_authorized", "format",
    "group_token_scheme", "jobs", "observed_seconds", "owner_manifest_commitment", "planned_attempts",
    "planned_independent_sources", "registry_content_commitment", "scientific_admission", "scope", "seed",
    "signed_admission", "source_groups", "total_seconds_cap", "unavailable_sources", "version",
})
GROUP_FIELDS = frozenset({"availability", "group", "partition", "reasons", "views"})
JOB_FIELDS = frozenset({"attempt", "estimated_seconds", "group", "stage"})
ATTEMPT_FIELDS = frozenset({"attempt", "group", "seconds", "stage", "status"})


class PilotError(ValueError):
    """Static diagnostic. Source values and key material are never part of the message."""


def _sha(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _number(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _require_key(token_key: object) -> bytes:
    if (
        type(token_key) is not bytes
        or len(token_key) < MIN_TOKEN_KEY_BYTES
        or len(set(token_key)) < MIN_TOKEN_KEY_DISTINCT_BYTES
    ):
        raise PilotError("pilot_token_key_invalid")
    return token_key


def _keyed(token_key: bytes, domain: str, value: object) -> str:
    try:
        message = canonical_json([domain, value])
    except EvidenceError:
        raise PilotError("pilot_commitment_invalid") from None
    return hmac.new(token_key, message, hashlib.sha256).hexdigest()


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
        "format": BINDING_FORMAT,
        "version": BINDING_VERSION,
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
            or not 0 <= item["attempts"] <= MAX_STAGE_ATTEMPTS
            or not _number(item["seconds_per_attempt"])
            or (item["attempts"] and item["seconds_per_attempt"] <= 0)
        ):
            raise PilotError("pilot_stage_budget_invalid")
    if any(stages[stage]["attempts"] == 0 for stage in REQUIRED_STAGES):
        raise PilotError("pilot_required_stage_missing")
    return stages


def _job(token: str, stage: str, attempt: int, seconds: object) -> dict:
    return {"attempt": attempt, "estimated_seconds": seconds, "group": token, "stage": stage}


def build_plan(registry: Registry, owners: dict, budget: dict, *, seed: int, token_key: bytes) -> dict:
    """Group a loaded registry into one development plan with keyed public tokens.

    `execution_authorized`, `scientific_admission`, `confirmation_allowed`, and
    `signed_admission` are constants. A caller boolean cannot change them. The
    job count and the cost estimate are checked before any job is built.
    """
    if type(seed) is not int or not 0 <= seed <= MAX_SEED:
        raise PilotError("pilot_seed_invalid")
    key = _require_key(token_key)
    if not isinstance(registry, Registry):
        raise PilotError("pilot_registry_invalid")
    if registry.final_label_custody != "external":
        raise PilotError("pilot_external_custody_required")
    if (
        not isinstance(owners, dict)
        or owners.get("format") != OWNER_FORMAT
        or owners.get("version") != OWNER_VERSION
        or not isinstance(owners.get("sources"), dict)
    ):
        raise PilotError("pilot_owner_manifest_invalid")
    stages = _require_budget(budget)
    try:
        families = registry.families()
        binding = _registry_binding(registry)
    except (LineageError, AttributeError, TypeError):
        raise PilotError("pilot_registry_invalid") from None
    if not families or len(families) > MAX_FAMILIES:
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
        token = _keyed(key, GROUP_DOMAIN, owner["source_identity_sha256"]) if owner_ok else None
        partition = None
        if not reasons:
            partition = PARTITIONS[int(digest([seed, token]), 16) % len(PARTITIONS)]
        groups.append({
            "availability": "planned" if not reasons else "unavailable",
            "group": token,
            "partition": partition,
            "reasons": sorted(reasons),
            "views": len(members),
        })
    planned_tokens = [group["group"] for group in groups if not group["reasons"]]
    per_group = sum(stages[stage]["attempts"] for stage in STAGES)
    if len(planned_tokens) * per_group > MAX_JOBS:
        raise PilotError("pilot_job_budget_exceeded")
    estimate = 0
    for _token in planned_tokens:
        for stage in STAGES:
            for _attempt in range(stages[stage]["attempts"]):
                estimate += stages[stage]["seconds_per_attempt"]
    if not math.isfinite(estimate) or estimate > budget["total_seconds"]:
        raise PilotError("pilot_time_budget_exceeded")
    jobs = [
        _job(token, stage, attempt + 1, stages[stage]["seconds_per_attempt"])
        for token in planned_tokens
        for stage in STAGES
        for attempt in range(stages[stage]["attempts"])
    ]
    plan = {
        "confirmation_allowed": False,
        "cost_basis": "estimate_not_observation",
        "estimated_seconds": estimate,
        "execution_authorized": False,
        "format": FORMAT,
        "group_token_scheme": TOKEN_SCHEME,
        "jobs": jobs,
        "observed_seconds": None,
        "owner_manifest_commitment": _keyed(key, OWNER_DOMAIN, owners),
        "planned_attempts": len(jobs),
        "planned_independent_sources": len(planned_tokens),
        "registry_content_commitment": _keyed(key, REGISTRY_DOMAIN, binding),
        "scientific_admission": False,
        "scope": "development_only",
        "seed": seed,
        "signed_admission": False,
        "source_groups": groups,
        "total_seconds_cap": budget["total_seconds"],
        "unavailable_sources": sum(bool(group["reasons"]) for group in groups),
        "version": VERSION,
    }
    _require_public(plan, key)
    return plan


def _require_plan(plan: object) -> tuple[list, set]:
    """Shape and type checks for a caller-supplied plan. The digest binding follows."""
    if (
        not isinstance(plan, dict)
        or set(plan) != PLAN_FIELDS
        or plan["format"] != FORMAT
        or plan["version"] != VERSION
        or plan["group_token_scheme"] != TOKEN_SCHEME
        or plan["execution_authorized"] is not False
        or plan["scientific_admission"] is not False
        or plan["confirmation_allowed"] is not False
        or plan["signed_admission"] is not False
        or not _number(plan["total_seconds_cap"])
        or plan["total_seconds_cap"] <= 0
        or not isinstance(plan["jobs"], list)
        or len(plan["jobs"]) > MAX_JOBS
        or type(plan["planned_attempts"]) is not int
        or plan["planned_attempts"] != len(plan["jobs"])
        or not isinstance(plan["source_groups"], list)
        or len(plan["source_groups"]) > MAX_FAMILIES
        or type(plan["planned_independent_sources"]) is not int
        or type(plan["unavailable_sources"]) is not int
        or plan["planned_independent_sources"] + plan["unavailable_sources"] != len(plan["source_groups"])
    ):
        raise PilotError("pilot_plan_invalid")
    planned_groups = set()
    for group in plan["source_groups"]:
        if not isinstance(group, dict) or set(group) != GROUP_FIELDS:
            raise PilotError("pilot_plan_invalid")
        if group["availability"] == "planned":
            if not _sha(group["group"]):
                raise PilotError("pilot_plan_invalid")
            planned_groups.add(group["group"])
    keys = []
    for job in plan["jobs"]:
        if (
            not isinstance(job, dict)
            or set(job) != JOB_FIELDS
            or not _sha(job["group"])
            or job["group"] not in planned_groups
            or type(job["stage"]) is not str
            or job["stage"] not in STAGES
            or type(job["attempt"]) is not int
            or not 1 <= job["attempt"] <= MAX_STAGE_ATTEMPTS
            or not _number(job["estimated_seconds"])
        ):
            raise PilotError("pilot_plan_invalid")
        keys.append((job["group"], job["stage"], job["attempt"]))
    planned = set(keys)
    if len(planned) != len(keys):
        raise PilotError("pilot_plan_invalid")
    return keys, planned


def reconcile_attempts(plan: dict, attempts: list, *, expected_plan_digest: str) -> dict:
    """Reconcile caller observations against one exact published plan.

    The plan is a caller input; it is accepted only when its digest equals the
    published `expected_plan_digest`, so an edited cap or job list is refused.
    This binding is not a signed admission. The planned attempt count stays the
    denominator. Observed seconds are caller observations, not receipts; they do
    not raise the original cap or authorize execution.
    """
    if not _sha(expected_plan_digest):
        raise PilotError("pilot_plan_digest_invalid")
    keys, planned = _require_plan(plan)
    try:
        published = digest(plan)
    except EvidenceError:
        raise PilotError("pilot_plan_invalid") from None
    if not hmac.compare_digest(published, expected_plan_digest):
        raise PilotError("pilot_plan_digest_mismatch")
    if not isinstance(attempts, list) or len(attempts) > MAX_JOBS:
        raise PilotError("pilot_attempts_invalid")
    observed: dict[tuple, dict] = {}
    for row in attempts:
        if (
            not isinstance(row, dict)
            or set(row) != ATTEMPT_FIELDS
            or type(row["group"]) is not str
            or type(row["stage"]) is not str
            or type(row["attempt"]) is not int
            or type(row["status"]) is not str
            or row["status"] not in OBSERVED_STATUSES
            or not _number(row["seconds"])
        ):
            raise PilotError("pilot_attempt_invalid")
        key = (row["group"], row["stage"], row["attempt"])
        if key not in planned or key in observed:
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
    counts["not_attempted"] = len(keys) - len(observed)
    seconds = sum(row["seconds"] for row in observed.values())
    if not math.isfinite(seconds):
        raise PilotError("pilot_attempt_cost_invalid")
    summary = {
        "budget_exceeded": seconds > plan["total_seconds_cap"],
        "confirmation_allowed": False,
        "execution_authorized": False,
        "execution_error_numerator": counts["failed"] + counts["timeout"],
        "failure_denominator": len(keys),
        "failures": [observed[key] for key in sorted(observed) if observed[key]["status"] in {"failed", "timeout"}],
        "format": SUMMARY_FORMAT,
        "non_success_numerator": len(keys) - counts["success"],
        "observed_attempts": len(observed),
        "observed_seconds": seconds,
        "original_seconds_cap": plan["total_seconds_cap"],
        "plan_digest": published,
        "planned_attempts": len(keys),
        "planned_independent_sources": plan["planned_independent_sources"],
        "scientific_admission": False,
        "scope": "development_only",
        "signed_admission": False,
        "status_counts": counts,
        "unavailable_sources": plan["unavailable_sources"],
        "version": VERSION,
    }
    _require_public(summary, None)
    return summary


def _require_public(document: dict, token_key: bytes | None) -> None:
    try:
        from .lineage import assert_public_provenance
        assert_public_provenance(document)
        text = canonical_json(document)
    except (LineageError, EvidenceError, ValueError, TypeError):
        raise PilotError("pilot_output_refused") from None
    if token_key is not None and (token_key.hex().encode() in text or token_key in text):
        raise PilotError("pilot_output_refused")
