"""Private manifest admission and custody-key-authenticated opaque projections.

No source is acquired or opened here. Evidence fields are owner assertions,
not license verification, checkpoint qualification, or relief from D18 HOLD.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from datetime import datetime
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker


class LineageError(ValueError):
    """A lineage, rights, exposure, or public-provenance rule failed."""


SCHEMA_PATH = Path(__file__).resolve().parents[2] / "schemas/source-manifest-v1.schema.json"
_SCHEMA = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
_MANIFEST = Draft202012Validator(_SCHEMA, format_checker=FormatChecker())
_PUBLIC = Draft202012Validator(
    {"$ref": "#/$defs/public_projection", "$defs": _SCHEMA["$defs"]},
    format_checker=FormatChecker(),
)


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=True, allow_nan=False).encode("ascii")
    except (TypeError, ValueError, RecursionError):
        raise LineageError("source_manifest_invalid_json") from None


def _validate(value: object, validator: Draft202012Validator, reason: str) -> dict:
    # Snapshot before validation: a later caller mutation cannot change the binding.
    snapshot = json.loads(_canonical(value))
    if not validator.is_valid(snapshot):
        raise LineageError(reason)
    return snapshot


def _key(value: bytes) -> bytes:
    if type(value) is not bytes or len(value) < 32:
        raise LineageError("source_projection_key_invalid")
    return value


def _token(key: bytes, domain: str, value: object) -> str:
    return hmac.new(key, _canonical([domain, value]), hashlib.sha256).hexdigest()


def _timestamp(value: str) -> datetime:
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if result.utcoffset() is not None:
            return result
    except ValueError:
        pass
    raise LineageError("source_timestamp_invalid")


def _rights(source: dict, family) -> None:
    rights = source["rights"]
    if (rights["license"].strip().casefold() in {"unknown", "unlicensed", "pending", "none"}
            or rights["acquisition"] != "granted" or rights["training"] != "granted"
            or family.rights.acquisition.value != "granted"
            or family.rights.training.value != "granted"):
        raise LineageError("source_rights_unapproved")
    if rights["redistribution"] == "unknown":
        raise LineageError("source_redistribution_unapproved")
    if rights["redistribution"] == "permitted_redistribution" and (
        not rights["redistribution_approved"] or family.rights.redistribution.value != "granted"
    ):
        raise LineageError("source_redistribution_unapproved")
    if source["use"] == "redistribute" and (
        rights["redistribution"] != "permitted_redistribution"
        or not rights["redistribution_approved"]
        or family.rights.redistribution.value != "granted"
    ):
        raise LineageError("source_redistribution_unapproved")
    if rights["redistribution_approved"] and rights["redistribution"] != "permitted_redistribution":
        raise LineageError("source_redistribution_inconsistent")


def _features(source: dict) -> None:
    prediction = _timestamp(source["prediction_time"]["timestamp"])
    seen = set()
    for feature in source["features"]:
        column = feature["column"]
        if column in seen or column == source["target"]["column"]:
            raise LineageError("source_feature_leakage")
        seen.add(column)
        if _timestamp(feature["feature_timestamp"]) > prediction:
            raise LineageError("source_post_outcome_feature")
    checkpoints = [(item["checkpoint"], item["checkpoint_sha256"])
                   for item in source["pretraining"]]
    if len(set(checkpoints)) != len(checkpoints):
        raise LineageError("source_duplicate_checkpoint")


def admit_source_manifest(document: dict, registry, *, signing_key: bytes) -> dict:
    """Admit development metadata and return only its authenticated public view.

    `registry` supplies canonical alias/parent identities and independent rights
    grants. Keep the manifest and random custody key private. Unknown/overlapping
    pretraining is explicitly caveated, never clean or confirmation-qualified.
    """
    key = _key(signing_key)
    manifest = _validate(document, _MANIFEST, "source_manifest_schema_invalid")
    families = {family.family_id: family for family in registry.families()}
    ids, cohorts, public = set(), {}, []
    if registry.final_label_custody != "external":
        raise LineageError("source_custody_hold")
    for source in manifest["sources"]:
        family = families.get(source["family_id"])
        if (family is None or family.parent_lineage_id != source["source_family"]
                or family.suite.value not in {"development", "historical"}):
            raise LineageError("source_family_unregistered")
        if source["source_id"] in ids:
            raise LineageError("source_duplicate_id")
        ids.add(source["source_id"])
        # Canonical parents catch aliases/variants; content identity also catches
        # the same bytes misregistered under two otherwise independent families.
        for identity in (("family", family.parent_lineage_id), ("content", source["content_sha256"])):
            if cohorts.setdefault(identity, source["cohort"]) != source["cohort"]:
                raise LineageError("source_family_cohort_leakage")
        _rights(source, family)
        _features(source)
        contamination = [{
            "checkpoint_token": _token(key, "checkpoint", [item["checkpoint"], item["checkpoint_sha256"]]),
            "status": item["status"], "caveated": item["status"] != "known_none",
        } for item in source["pretraining"]]
        public.append({
            "source_token": _token(key, "source", source["source_id"]),
            "family_token": _token(key, "family", family.parent_lineage_id),
            "cohort": source["cohort"], "use": source["use"],
            "redistribution": source["rights"]["redistribution"],
            "admission": ("development_only_caveated" if any(item["caveated"] for item in contamination)
                          else "development_only"),
            "pretraining": contamination,
        })
    projection = {
        "format": "dope-source-projection-v1", "version": 1,
        "custody_hold": "D18_HOLD", "confirmation_allowed": False,
        "sealed_source_claims": False,
        "manifest_commitment": _token(key, "manifest", manifest),
        "sources": public, "signature_scheme": "hmac-sha256-custody-v1",
    }
    projection["signature"] = _token(key, "projection", projection)
    return verify_source_projection(projection, signing_key=key)


def verify_source_projection(document: dict, *, signing_key: bytes) -> dict:
    """Authenticate with the custody key; this is a MAC, not a public-key signature.

    A valid MAC binds metadata only. It cannot authorize sealed-source access.
    Strict allowlists prevent raw free text even in an authenticated envelope.
    """
    key = _key(signing_key)
    projection = _validate(document, _PUBLIC, "source_projection_schema_invalid")
    unsigned = {name: value for name, value in projection.items() if name != "signature"}
    if not hmac.compare_digest(projection["signature"], _token(key, "projection", unsigned)):
        raise LineageError("source_projection_signature_invalid")
    return projection
