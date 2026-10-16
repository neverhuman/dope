"""Deterministic reference container and exact byte scopes (T-034/T-036).

This fixture container is a conformance format, not a replacement DPK decoder.
No row count, scalar count, or denominator can reduce charged bytes.
"""
import hashlib
import re
import struct

from .common import MetricError, count, failure, opt_default, require

MAGIC = b"TDL1"


def _sections(values):
    require(type(values) is list, "invalid_sections")
    decoded, seen = [], set()
    for section in values:
        require(type(section) is dict, "invalid_section")
        name = section.get("name")
        require(type(name) is str and re.fullmatch(r"[a-z0-9][a-z0-9_.-]{0,63}", name), "invalid_section_name")
        require(name not in seen, "duplicate_section")
        seen.add(name)
        encoded = section.get("hex")
        require(type(encoded) is str and len(encoded) % 2 == 0 and
                re.fullmatch(r"[0-9a-f]*", encoded) is not None, "invalid_section_hex")
        decoded.append((name, bytes.fromhex(encoded)))
    return sorted(decoded)


def canonical_container(sections):
    sections = _sections(sections)
    parts = [MAGIC, struct.pack("<I", len(sections))]
    for name, content in sections:
        token = name.encode("ascii")
        parts.extend((struct.pack("<I", len(token)), token, struct.pack("<Q", len(content)), content))
    return b"".join(parts)


def evaluate(data):
    try:
        require(type(data) is dict)
        if data.get("sections") is None or data.get("closure_files") is None:
            return failure("missing_charge_manifest", "incomplete")
        sections = _sections(data["sections"])
        require(bool(sections), "empty_payload")
        closure = _sections(data["closure_files"])
        require(all(file.get("scope") in ("runtime", "shared_prior") and
                    file.get("dataset_specific") is False for file in data["closure_files"]),
                "undeclared_or_dataset_specific_closure")
        # A section is dataset-specific by definition; all fixed sampling code,
        # libraries and disclosed shared priors belong in the closure inventory.
        require(not ({n for n, _ in sections} & {n for n, _ in closure}), "duplicate_charge_entry")
        artifact = canonical_container(data["sections"])
        payload = len(artifact)
        runtime = sum(len(content) for _, content in closure)
        total = count(payload + runtime)
        users = count(opt_default(data, "amortization_count", 1), 1)
        return {"status": "complete", "reason": None, "payload_bytes": payload,
                "section_bytes": {name: len(content) for name, content in sections},
                "framing_bytes": payload - sum(len(content) for _, content in sections),
                "shared_bytes": runtime, "sampling_closure_bytes": total,
                "cold_bundle_bytes": total, "amortized_bytes": payload + runtime / users,
                "artifact_sha256": hashlib.sha256(artifact).hexdigest()}
    except MetricError as error:
        return failure(str(error))
