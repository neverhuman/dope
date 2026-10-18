#!/usr/bin/env python3
"""Validate TDL panels and migrate legacy locks without granting admission."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STATES = ("specified", "source_verified", "weights_verified", "environment_verified",
          "parity_verified", "resource_verified", "qualified")
REQUIRED = {
    "source_verified": ("source_revision", "code_sha256", "code_license"),
    "weights_verified": ("checkpoint_sha256", "weight_license"),
    "environment_verified": ("dependency_closure_sha256", "preprocessing_sha256"),
    "parity_verified": ("native_parity_sha256",),
    "resource_verified": ("resource_evidence_sha256",),
    "qualified": ("license_review_sha256", "contamination_review_sha256", "last_verified"),
}


def foundation_model(entry: dict) -> bool:
    return entry.get("family") == "foundation_tabular" or bool(re.search(
        r"kumo|tabicl|mitra|tabpfn", " ".join(str(entry.get(key, ""))
                                            for key in ("id", "auditor", "display")), re.I))


def pretraining_disclosure(entry: dict) -> dict:
    """Unrun legacy stubs have unknown overlap; no checkpoint receives an implicit pass."""
    disclosure = entry.get("pretraining_corpus_disclosure")
    required = foundation_model(entry) or disclosure is not None
    if disclosure is None:
        if required and (entry.get("checkpoint_sha256") is not None or
                         entry.get("qualification", {}).get("state", "specified") != "specified"):
            raise ValueError("CR-10: checkpoint requires pretraining-corpus disclosure")
        return ({"state": "unknown", "caveat_required": True,
                 "caveat_text": "Pretraining overlap is unknown; source-disjoint benefit is unverified."}
                if required else {"state": "not_applicable", "caveat_required": False})
    if not isinstance(disclosure, dict) or disclosure.get("state") not in (
            "known_none", "overlapping", "unknown", "not_applicable"):
        raise ValueError("CR-10: invalid pretraining-corpus disclosure")
    if required and foundation_model(entry) and disclosure["state"] == "not_applicable":
        raise ValueError("CR-10: foundation checkpoint requires an overlap state")
    if type(disclosure.get("caveat_required")) is not bool:
        raise ValueError("CR-10: disclosure requires explicit caveat state")
    if disclosure["state"] in ("unknown", "overlapping") and (
            disclosure["caveat_required"] is not True or
            not isinstance(disclosure.get("caveat_text"), str) or not disclosure["caveat_text"].strip()):
        raise ValueError("CR-10: unknown or overlapping pretraining requires an explicit caveat")
    checkpoint = entry.get("checkpoint_sha256")
    if checkpoint is not None and disclosure.get("checkpoint_sha256") != checkpoint:
        raise ValueError("CR-10: disclosure must bind the exact checkpoint")
    return copy.deepcopy(disclosure)


def equivalence_passed(procedure: dict | None) -> bool:
    if not isinstance(procedure, dict):
        return False
    margin = procedure.get("equivalence_margin")
    return (procedure.get("claimed") is True and procedure.get("verified") is True and
            procedure.get("bound_side") == "both" and type(margin) in (int, float) and
            0 < margin <= 1.7976931348623157e308)


def receipt_digests(value) -> bool:
    return isinstance(value, list) and bool(value) and all(
        isinstance(pin, str) and re.fullmatch(r"[0-9a-fA-F]{64}", pin) for pin in value)


def validate_result_claims(data: dict) -> list[str]:
    """Check optional successor fields without reclassifying historical results."""
    failures = []
    for row in data.get("results", []):
        kind = row.get("claim_class")
        if kind is not None and kind not in ("measured", "historical", "specified", "illustrative"):
            failures.append("results: invalid claim class")
        historical = row.get("source", {}).get("path") == "docs/whitepaper/generated/v2-headline.json"
        if historical and kind not in (None, "historical"):
            failures.append("results: historical source cannot change claim class")
        if kind == "measured" and (data.get("protocol_frozen") is not True or
                                   not receipt_digests(row.get("receipt_sha256s"))):
            failures.append("results: measured claim requires frozen protocol and receipts")
        try:
            if foundation_model(row) and (not row.get("checkpoint_sha256") or
                                           "pretraining_corpus_disclosure" not in row):
                raise ValueError("CR-10: foundation result requires per-checkpoint disclosure")
            disclosure = pretraining_disclosure(row)
            if kind == "measured" and disclosure["state"] == "overlapping":
                failures.append("CR-10: overlapping pretraining cannot support a measured claim")
            if disclosure["state"] in ("unknown", "overlapping") and row.get("claim_allowed") is True:
                failures.append("CR-10: uncertain overlap cannot grant an unqualified claim")
        except ValueError as error:
            failures.append(str(error))
        procedure = row.get("equivalence_procedure", {})
        if not isinstance(procedure, dict) or (procedure.get("claimed") is True and (
                not equivalence_passed(procedure) or not receipt_digests(row.get("receipt_sha256s")))):
            failures.append("CR-13: equivalence requires verified two-sided bounds and a positive finite margin")
    return failures


def validate_claim_ledger(ledger: dict, root: Path = ROOT, *, check_sources: bool = False) -> list[str]:
    """Validate structure and safe semantics; diagnostics never quote supplied data."""
    from jsonschema import Draft202012Validator
    schema = json.loads((root / "schemas/claim-v1.schema.json").read_text())
    failures = [f"claim ledger: schema {error.validator} violation"
                for error in Draft202012Validator(schema).iter_errors(ledger)]
    if failures:
        return failures
    ids = [row["claim_id"] for row in ledger["claims"]]
    if len(ids) != len(set(ids)):
        failures.append("claim ledger: duplicate claim id")
    from tools.claims.check import (lint_cr04_redaction_and_unbacked_badges,
                                   lint_s1_r074_no_significant_difference_equivalence, overlap_language_failures)
    for row in ledger["claims"]:
        procedure = row["equivalence_procedure"]
        margin = procedure["equivalence_margin"]
        if margin is not None and (type(margin) not in (int, float) or not math.isfinite(margin)):
            failures.append("CR-13: equivalence margin must be finite")
        if procedure["claimed"] and not equivalence_passed(procedure):
            failures.append("CR-13: equivalence procedure has not passed")
        failures.extend(lint_s1_r074_no_significant_difference_equivalence(
            row["allowed_scope"], "claim ledger", procedure, row["receipt_sha256s"]))
        disclosure = row["pretraining_corpus_disclosure"]
        if disclosure["state"] in ("unknown", "overlapping"):
            failures.extend(overlap_language_failures(
                row["allowed_scope"] + ' ' + disclosure['caveat_text'], "claim ledger",
                {'kumo', 'tabicl', 'mitra', 'tabpfn'}))
        failures.extend(lint_cr04_redaction_and_unbacked_badges(json.dumps(row), "claim ledger"))
    if check_sources:
        from tools.claims.render import claim_ledger
        data = json.loads((root / "paper/generated/headline-results.json").read_text())
        status = json.loads((root / "evidence/status.json").read_text())
        if ledger != claim_ledger(data, status, root=root):
            failures.append("claim ledger: differs from evidence status or declared capability scopes")
    return failures


def specified() -> dict:
    return {"state": "specified", "admission_status": "unqualified", "history": [],
            "rejection_reason": "Qualification evidence and D-07 approval are pending."}


def validate_qualification(value: dict, entry: dict) -> None:
    if foundation_model(entry) or "pretraining_corpus_disclosure" in entry:
        pretraining_disclosure(entry)
    state = value["state"]
    if state not in STATES:
        raise ValueError("unknown qualification state")
    history = value["history"]
    expected = list(STATES[1:STATES.index(state) + 1])
    if [event["state"] for event in history] != expected:
        raise ValueError("qualification history skips or repeats a transition")
    for event in history:
        if not isinstance(event.get("receipt_sha256"), str) or len(event["receipt_sha256"]) != 64:
            raise ValueError("transition requires a receipt digest")
        try:
            bytes.fromhex(event["receipt_sha256"])
        except ValueError as error:
            raise ValueError("transition requires a hexadecimal digest") from error
    for reached in expected:
        if any(not entry.get(field) or any(token in str(entry[field]).lower() for token in ("unreviewed", "unresolved", "pending", "tbd"))
               for field in REQUIRED[reached]):
            raise ValueError(f"{reached} lacks required evidence")
    if value["admission_status"] != ("qualified" if state == "qualified" else "unqualified"):
        raise ValueError("admission disagrees with qualification state")


def advance(entry: dict, state: str, receipt_sha256: str) -> dict:
    result = copy.deepcopy(entry)
    q = result["qualification"]
    validate_qualification(q, result)
    if STATES.index(state) != STATES.index(q["state"]) + 1:
        raise ValueError("qualification transitions must advance exactly one state")
    q["state"] = state
    if "status" in result:
        result["status"] = state
    q["history"].append({"state": state, "receipt_sha256": receipt_sha256})
    q["admission_status"] = "qualified" if state == "qualified" else "unqualified"
    validate_qualification(q, result)
    return result


def validate(root: Path = ROOT) -> list[str]:
    from jsonschema import Draft202012Validator
    failures = []
    pairs = [("auditors.json", "auditor-registry-v1.schema.json"),
             ("generator-panel.json", "generator-registry-v1.schema.json")]
    for filename, schema_name in pairs:
        value = json.loads((root / "research/transfer/registries" / filename).read_text())
        schema = json.loads((root / "schemas" / schema_name).read_text())
        for error in Draft202012Validator(schema).iter_errors(value):
            failures.append(f"{filename}: schema {error.validator} violation")
        names = [row["id"] for row in value["entries"]]
        if len(names) != len(set(names)):
            failures.append(f"{filename}: duplicate entry id")
        for entry in value["entries"]:
            try:
                validate_qualification(entry["qualification"], entry)
                if entry["status"] != entry["qualification"]["state"]:
                    raise ValueError("registry status disagrees with typed state")
            except (KeyError, ValueError) as error:
                failures.append(f"{filename}: invalid qualification ({type(error).__name__})")
    path = root / "research/transfer/manifests/datasets.json"
    schema = json.loads((root / "schemas/dataset-manifest-v1.schema.json").read_text())
    failures.extend(f"datasets: schema {e.validator} violation"
                    for e in Draft202012Validator(schema).iter_errors(json.loads(path.read_text())))
    index = json.loads((root / "research/transfer/registries/lock-states.json").read_text())
    mapped = {item["path"]: item for item in index["entries"]}
    paths = {str(p.relative_to(root)) for p in (root / "research/transfer/locks").glob("*.json")}
    if set(mapped) != paths or len(mapped) != len(index["entries"]):
        failures.append("lock-state index must map every existing lock exactly once")
    for name, item in mapped.items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != item["sha256"]:
            failures.append(f"{name}: lock-state source hash drift")
        try:
            validate_qualification(item["qualification"], item)
        except (KeyError, ValueError) as error:
            failures.append(f"{name}: {error}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--migrate", action="store_true")
    args = parser.parse_args()
    if args.migrate:
        entries = []
        for path in sorted((ROOT / "research/transfer/locks").glob("*.json")):
            value = json.loads(path.read_text())
            entries.append({"path": str(path.relative_to(ROOT)),
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                            "legacy_format": value["format"], "qualification": specified(),
                            "model_version": value["format"],
                            "source_revision": value.get("source", {}).get("revision"),
                            "code_sha256": None, "checkpoint_sha256": None,
                            "code_license": "unreviewed", "weight_license": "unreviewed",
                            "dependency_closure_sha256": None,
                            "tasks": ["regression", "classification"], "mode": "legacy_reference",
                            "context_cap": None, "preprocessing_sha256": None,
                            "native_parity_sha256": None, "resource_evidence_sha256": None,
                            "license_review_sha256": None, "contamination_review_sha256": None,
                            "last_verified": None})
        index_path = ROOT / "research/transfer/registries/lock-states.json"
        if index_path.exists():
            previous = json.loads(index_path.read_text())
            by_path = {item["path"]: item for item in previous["entries"]}
            for index, entry in enumerate(entries):
                if entry["path"] in by_path:
                    if entry["sha256"] != by_path[entry["path"]]["sha256"]:
                        raise ValueError("changed legacy lock requires explicit qualification review")
                    entries[index] = {**entry, **by_path[entry["path"]]}
        index_path.write_text(json.dumps({"format": "dope-lock-state-index-v1", "entries": entries}, indent=2) + "\n")
    failures = validate()
    print("\n".join(failures) if failures else "Registry schemas and typed lock admission passed")
    return bool(failures)


if __name__ == "__main__":
    raise SystemExit(main())
