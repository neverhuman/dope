"""Fail-closed calibration receipt. A development protocol is not N13b.

The decision never qualifies coverage. A caller flag is not a measurement,
and this module does not run a calibration campaign.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import MappingProxyType

from .evidence import digest


FORMAT = "dope-calibration-receipt"
VERSION = 1
_HEX = re.compile(r"^[0-9a-f]{64}$")
_PROTOCOL = Path(__file__).resolve().parent / "protocols" / "statistics-v1.json"
_FIELDS = (
    "format", "version", "protocol_id", "protocol_digest", "repetitions", "bootstrap_draws",
    "frozen_for_confirmation", "manifest_digest", "tolerances_frozen_before_outcomes",
    "mcse_reported", "wilson_reported", "binding_digest",
)
_ASSERTIONS = {"qualifies", "calibration", "n13b", "authenticates", "campaign", "release"}


def development_protocol() -> dict:
    return json.loads(_PROTOCOL.read_text())


def _decision(authenticates: bool, reason: str) -> MappingProxyType:
    return MappingProxyType({
        "format": "dope-calibration-decision",
        "version": 1,
        "authenticates": authenticates,
        "qualifies": False,
        "calibration": "unmeasured",
        "reason": reason,
        "campaign": "unmeasured",
        "n13b": False,
        "release": False,
    })


def _hex(value) -> bool:
    return type(value) is str and _HEX.fullmatch(value) is not None


def _count(value) -> bool:
    return type(value) is int and value >= 1


def verify_calibration_receipt(record) -> MappingProxyType:
    """Authenticate a version-1 receipt structure. Qualification stays false."""
    if not isinstance(record, dict):
        raise ValueError("invalid_calibration_receipt")
    version = record.get("version")
    if type(version) is int and version != VERSION:
        raise ValueError("unknown_calibration_receipt_version")
    if _ASSERTIONS & set(record):
        raise ValueError("self_asserted_qualification")
    if set(record) != set(_FIELDS) or record.get("format") != FORMAT or version != VERSION:
        raise ValueError("invalid_calibration_receipt")
    if type(record["protocol_id"]) is not str or not record["protocol_id"] or not _hex(record["protocol_digest"]):
        raise ValueError("invalid_calibration_receipt")
    if not _count(record["repetitions"]) or not _count(record["bootstrap_draws"]):
        raise ValueError("invalid_calibration_receipt")
    flags = (
        record["frozen_for_confirmation"], record["tolerances_frozen_before_outcomes"],
        record["mcse_reported"], record["wilson_reported"],
    )
    if any(type(flag) is not bool for flag in flags):
        raise ValueError("invalid_calibration_receipt")
    manifest = record["manifest_digest"]
    if manifest is not None and not _hex(manifest):
        raise ValueError("invalid_calibration_receipt")
    if not _hex(record["binding_digest"]):
        raise ValueError("invalid_calibration_receipt")
    body = {key: record[key] for key in _FIELDS if key != "binding_digest"}
    if record["binding_digest"] != digest(body):
        return _decision(False, "tampered_calibration_receipt")
    protocol = development_protocol()
    if record["protocol_id"] != protocol["protocol_id"]:
        return _decision(False, "unknown_protocol")
    if record["protocol_digest"] != digest(protocol):
        return _decision(False, "protocol_digest_mismatch")
    return _decision(True, "development_protocol_is_not_n13b")
