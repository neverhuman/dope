#!/usr/bin/env python3
"""Validate TDL panels and migrate legacy locks without granting admission."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
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


def specified() -> dict:
    return {"state": "specified", "admission_status": "unqualified", "history": [],
            "rejection_reason": "Qualification evidence and D-07 approval are pending."}


def validate_qualification(value: dict, entry: dict) -> None:
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
            failures.append(f"{filename}: {error.json_path}: {error.message}")
        names = [row["id"] for row in value["entries"]]
        if len(names) != len(set(names)):
            failures.append(f"{filename}: duplicate entry id")
        for entry in value["entries"]:
            try:
                validate_qualification(entry["qualification"], entry)
                if entry["status"] != entry["qualification"]["state"]:
                    raise ValueError("registry status disagrees with typed state")
            except (KeyError, ValueError) as error:
                failures.append(f"{filename}: {entry['id']}: {error}")
    path = root / "research/transfer/manifests/datasets.json"
    schema = json.loads((root / "schemas/dataset-manifest-v1.schema.json").read_text())
    failures.extend(f"datasets: {e.json_path}: {e.message}"
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
