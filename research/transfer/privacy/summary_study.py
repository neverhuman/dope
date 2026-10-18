"""Pure postprocessing boundary for DP-summary comparison studies."""
from __future__ import annotations

import hashlib
import json
import math


class SummaryStudyError(ValueError):
    """Fixed-code refusal; never includes summary values or source details."""


def canonical_summary(summary: object, *, accounting: dict) -> bytes:
    """Serialize a bounded entity summary only after an independent review.

    This gate intentionally does not accept source rows, callbacks, or paths.
    It verifies comparison metadata, not the truth of the external proof.
    """
    if (type(accounting) is not dict
            or accounting.get("formal_dp_status") != "verified"
            or accounting.get("independent_review") != "passed"
            or accounting.get("unit") != "entity"
            or type(accounting.get("adjacency")) is not str
            or accounting["adjacency"] != "add_remove"
            or not isinstance(accounting.get("mechanism_sha256"), str)
            or len(accounting["mechanism_sha256"]) != 64
            or type(accounting.get("epsilon")) not in (int, float)
            or type(accounting.get("delta")) not in (int, float)
            or not math.isfinite(accounting["epsilon"])
            or not math.isfinite(accounting["delta"])
            or accounting["epsilon"] < 0 or not 0 <= accounting["delta"] < 1):
        raise SummaryStudyError("summary_accounting_not_admitted")
    try:
        encoded = json.dumps(summary, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, UnicodeError, RecursionError):
        raise SummaryStudyError("invalid_summary") from None
    if (type(summary) is not dict
            or set(summary) != {"format", "statistics"}
            or summary.get("format") != "bounded-entity-dp-summary-v1"
            or type(summary.get("statistics")) is not dict
            or not summary["statistics"]
            or len(summary["statistics"]) > 256
            or any(type(name) is not str or not name or len(name) > 64
                   or type(values) is not list or not 1 <= len(values) <= 4096
                   or any(type(value) not in (int, float) or not math.isfinite(value)
                          for value in values)
                   for name, values in summary["statistics"].items())
            or len(encoded) > 1_000_000):
        raise SummaryStudyError("invalid_summary")
    return encoded


def summary_digest(summary: object, *, accounting: dict) -> str:
    """Content identity for a reviewed summary; this is not a DP certificate."""
    return hashlib.sha256(canonical_summary(summary, accounting=accounting)).hexdigest()
