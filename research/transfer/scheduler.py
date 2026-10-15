"""Durable per-source execution plan.

One source id has one identity hash. One query id has one source.
An interrupted attempt keeps the queries it already charged.
This plan does not qualify a release or a privacy result.
"""

import os
import re
from pathlib import Path

from research.transfer.evidence import canonical_json, parse_json


FORMAT = "dope-transfer-execution-plan"
VERSION = 1
_HEX = re.compile(r"^[0-9a-f]{64}$")
_STATUSES = frozenset({"scheduled", "interrupted", "complete"})
_SOURCE_KEYS = frozenset({"source_id", "identity_sha256"})
_ATTEMPT_KEYS = frozenset({"attempt_id", "source_id", "query_id", "charged_queries", "status"})


class ExecutionError(ValueError):
    """Stable codes only. Messages do not carry source values."""


def _fail(code):
    raise ExecutionError(code)


def _text(value):
    return type(value) is str and bool(value)


def _source(item):
    if (
        type(item) is not dict
        or set(item) != _SOURCE_KEYS
        or not _text(item["source_id"])
        or type(item["identity_sha256"]) is not str
        or _HEX.fullmatch(item["identity_sha256"]) is None
    ):
        _fail("invalid_execution_plan")
    return {"source_id": item["source_id"], "identity_sha256": item["identity_sha256"]}


def _attempt(item):
    if (
        type(item) is not dict
        or set(item) != _ATTEMPT_KEYS
        or not _text(item["attempt_id"])
        or not _text(item["source_id"])
        or not _text(item["query_id"])
        or type(item["charged_queries"]) is not int
        or item["charged_queries"] < 0
        or item["status"] not in _STATUSES
    ):
        _fail("invalid_execution_plan")
    return {
        "attempt_id": item["attempt_id"],
        "source_id": item["source_id"],
        "query_id": item["query_id"],
        "charged_queries": item["charged_queries"],
        "status": item["status"],
    }


def _partition(sources, attempts):
    if type(sources) is not list or not sources or type(attempts) is not list:
        _fail("invalid_execution_plan")
    stored = []
    identities = {}
    for item in sources:
        source = _source(item)
        previous = identities.get(source["source_id"])
        if previous is None:
            identities[source["source_id"]] = source["identity_sha256"]
            stored.append(source)
        elif previous != source["identity_sha256"]:
            _fail("source_identity_mismatch")
    seen = set()
    query_sources = {}
    stored_attempts = []
    for item in attempts:
        attempt = _attempt(item)
        if attempt["attempt_id"] in seen:
            _fail("duplicate_attempt")
        if attempt["source_id"] not in identities:
            _fail("unknown_source")
        owner = query_sources.get(attempt["query_id"])
        if owner is not None and owner != attempt["source_id"]:
            _fail("query_charge_source_mismatch")
        seen.add(attempt["attempt_id"])
        query_sources[attempt["query_id"]] = attempt["source_id"]
        stored_attempts.append(attempt)
    return stored, stored_attempts


class DurableSchedule:
    """One on-disk plan. Reload sees the charge an interrupt left behind."""

    def __init__(self, path, sources):
        stored, attempts = _partition(sources, [])
        self.path = Path(path)
        self.sources = stored
        self.attempts = attempts
        self._write()

    def document(self):
        return {
            "format": FORMAT,
            "version": VERSION,
            "sources": self.sources,
            "attempts": self.attempts,
        }

    def _write(self):
        payload = canonical_json(self.document())
        self.path.parent.mkdir(parents=True, exist_ok=True)
        pending = self.path.with_name(self.path.name + ".writing")
        pending.write_bytes(payload)
        os.replace(pending, self.path)

    def charge(self, attempt_id, source_id, query_id, charged_queries):
        attempt = _attempt({
            "attempt_id": attempt_id,
            "source_id": source_id,
            "query_id": query_id,
            "charged_queries": charged_queries,
            "status": "scheduled",
        })
        if any(item["attempt_id"] == attempt["attempt_id"] for item in self.attempts):
            _fail("duplicate_attempt")
        if attempt["source_id"] not in {item["source_id"] for item in self.sources}:
            _fail("unknown_source")
        for item in self.attempts:
            if item["query_id"] == attempt["query_id"] and item["source_id"] != attempt["source_id"]:
                _fail("query_charge_source_mismatch")
        self.attempts.append(attempt)
        self._write()

    def interrupt(self, attempt_id):
        match = next((item for item in self.attempts if item["attempt_id"] == attempt_id), None)
        if match is None:
            _fail("unknown_attempt")
        match["status"] = "interrupted"
        self._write()
        return match["charged_queries"]

    @classmethod
    def load(cls, path):
        value = parse_json(Path(path).read_bytes())
        if (
            type(value) is not dict
            or set(value) != {"format", "version", "sources", "attempts"}
            or value["format"] != FORMAT
            or type(value["version"]) is not int
            or value["version"] != VERSION
        ):
            _fail("invalid_execution_plan")
        stored, attempts = _partition(value["sources"], value["attempts"])
        schedule = object.__new__(cls)
        schedule.path = Path(path)
        schedule.sources = stored
        schedule.attempts = attempts
        return schedule
