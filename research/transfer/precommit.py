"""Fail-closed integrity checks for the E04 confirmatory design packet.

This local checker validates the blocked packet and reports missing gates. It
does not open protected labels or provide a signature-verification authority.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from research.transfer.evidence import canonical_json

FORMAT = "dope-confirmatory-precommit-v1"
PROTOCOL = "research/transfer/protocols/confirmatory-precommit-v1.json"
HEX = re.compile(r"^[0-9a-f]{64}$")
PINNED_PATHS = (
    "schemas/campaign-v1.schema.json",
    "contracts/tdl-contract-v2-candidate.json",
    "research/transfer/protocols/statistics-v1.json",
    "research/transfer/metrics.py",
    "research/transfer/tdl/transfer.py",
    "research/transfer/tdl/drivers.py",
    "research/transfer/tdl/frontier.py",
    "research/transfer/evidence.py",
    "research/transfer/scheduler.py",
    "research/transfer/lineage.py",
    "research/transfer/precommit.py",
    "research/transfer/selection_regret.py",
    "research/transfer/adapters/base.py",
    "research/transfer/adapters/worker_abi.py",
    "research/transfer/tdl/common.py",
    "research/transfer/tdl/aggregate.py",
    "research/transfer/tdl/charge.py",
    "research/transfer/manifests/datasets.json",
    "research/transfer/registries/sources.json",
    "research/transfer/registries/models.json",
    "research/transfer/registries/generators.json",
    "research/transfer/registries/generator-panel.json",
    "research/transfer/registries/auditors.json",
)
BYTE_GRID = (512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144)
FIELDS = frozenset({
    "format", "version", "campaign_id", "admission_state", "sealed_access_allowed",
    "protocol_path", "protocol_sha256", "evidence_pins", "design", "source_roles",
    "model_roles", "prepared_maps", "candidate_budget_slots", "full_comparison_roster",
    "roster_complete", "thresholds", "sample_schedule", "privacy_outcomes",
    "budget_and_power", "retry_exclusion", "review_signature", "custodian_attestation",
    "admission_blockers", "binding_sha256",
})


class PrecommitError(ValueError):
    """Static refusal code; never includes protected values or paths."""


def _fail(code: str) -> None:
    raise PrecommitError(code)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_pins(root: Path) -> dict[str, str]:
    """Return hashes for the executable metric, protocol, cache, and closure surface."""
    root = Path(root).resolve()
    result = {}
    for name in (*PINNED_PATHS, PROTOCOL):
        path = root / name
        if path.is_symlink() or not path.is_file():
            _fail("precommit_pin_unavailable")
        result[name] = _sha256(path)
    return result


def validate_blocked_precommit(document: object, root: Path) -> tuple[str, ...]:
    """Verify the immutable design skeleton and return its explicit admission blockers."""
    if (type(document) is not dict or set(document) != FIELDS
            or document.get("format") != FORMAT or document.get("version") != 1):
        _fail("invalid_precommit_format")
    body = dict(document)
    binding = body.pop("binding_sha256", None)
    if type(binding) is not str or not HEX.fullmatch(binding):
        _fail("precommit_binding_missing")
    if hashlib.sha256(canonical_json(body)).hexdigest() != binding:
        _fail("precommit_binding_mismatch")
    if document.get("admission_state") != "blocked" or document.get("sealed_access_allowed") is not False:
        _fail("precommit_must_remain_blocked")
    if document.get("protocol_path") != PROTOCOL:
        _fail("precommit_protocol_path_invalid")
    if document.get("protocol_sha256") != expected_pins(root)[PROTOCOL]:
        _fail("precommit_protocol_pin_mismatch")
    if document.get("evidence_pins") != expected_pins(root):
        _fail("precommit_evidence_pin_mismatch")

    slots = document.get("candidate_budget_slots")
    if type(slots) is not list or len(slots) != len(BYTE_GRID):
        _fail("precommit_budget_roster_incomplete")
    for budget, slot in zip(BYTE_GRID, slots):
        if (type(slot) is not dict or slot.get("budget_bytes") != budget
                or slot.get("candidate_id") != f"development-selected-at-{budget}-bytes"
                or slot.get("artifact_sha256") is not None
                or slot.get("recipe_sha256") is not None
                or slot.get("eligible_for_minimum_tdl_reporting") is not False):
            _fail("precommit_candidate_slot_invalid")
    if document.get("full_comparison_roster") != [] or document.get("roster_complete") is not False:
        _fail("precommit_unresolved_roster_must_not_claim_complete")
    if document.get("review_signature") is not None or document.get("custodian_attestation") is not None:
        _fail("precommit_unverified_authority")
    blockers = document.get("admission_blockers")
    if type(blockers) is not list or not blockers or any(type(item) is not str or not item for item in blockers):
        _fail("precommit_blockers_missing")
    return tuple(blockers)


def sealed_access_allowed(document: object, root: Path) -> bool:
    """The E04 checker has no trusted signer verifier and therefore never authorizes access."""
    validate_blocked_precommit(document, root)
    return False


def load_precommit(path: Path, root: Path) -> tuple[dict, tuple[str, ...]]:
    try:
        document = json.loads(Path(path).read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        _fail("precommit_unreadable")
    return document, validate_blocked_precommit(document, root)
