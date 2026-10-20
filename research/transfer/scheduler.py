"""Immutable campaign denominator and durable, append-only execution ledger.

Planning is outcome independent. A campaign is complete only when every frozen
cell has one accepted terminal result. This module records evidence; it does not
execute workers or grant scientific/release authority.
"""
from __future__ import annotations

import copy
import hashlib
import os
import re
from pathlib import Path
from itertools import product

from research.transfer.evidence import canonical_json, digest, parse_json

FORMAT = "dope-transfer-execution-plan"
CAMPAIGN_FORMAT = "dope-transfer-campaign"
VERSION = 1
_HEX = re.compile(r"^[0-9a-f]{64}$")
_TERMINAL = frozenset({"completed", "timeout", "oom", "unsupported", "license_denied",
                       "missing", "invalid", "cancelled", "unresolved"})
_BASE_DIMENSIONS = ("task", "method", "budget", "fitseed", "drawseed", "auditor",
                    "auditorseed", "endpoint")
_DIMENSIONS = (*_BASE_DIMENSIONS, "sample_size", "context_policy")
_GRID_BASE = [512] + [1024 * (2 ** power) for power in range(9)]


class ExecutionError(ValueError):
    """Stable codes only; errors never include source or cell values."""


def _fail(code: str) -> None:
    raise ExecutionError(code)


def byte_grid(native_rungs=(), dev_locked_rungs=()):
    """512 B, 1..256 KiB powers of two, native MiB/GiB and locked 4x rungs."""
    values = set(_GRID_BASE)
    for rung in native_rungs:
        if type(rung) is not int or rung <= 0:
            _fail("invalid_byte_rung")
        values.add(rung)
    for rung in dev_locked_rungs:
        if type(rung) is not int or rung <= 0:
            _fail("invalid_byte_rung")
        values.add(rung)
        values.add(rung * 4)
    for power in range(20, 31, 10):
        values.add(1 << power)
    return sorted(values)


def _axes(dimensions):
    if type(dimensions) is not dict or set(dimensions) != set(_BASE_DIMENSIONS):
        _fail("invalid_campaign_dimensions")
    axes = {}
    for name in _BASE_DIMENSIONS:
        values = dimensions[name]
        if type(values) is not list or not values:
            _fail("invalid_campaign_dimensions")
        encoded = [canonical_json(item) for item in values]
        if len(set(encoded)) != len(encoded):
            _fail("duplicate_campaign_dimension")
        axes[name] = values
    return axes


def build_roster(dimensions, primary_n, context_policies):
    """Freeze primary n and an independent 4n sensitivity cell for each axis tuple."""
    axes = _axes(dimensions)
    if type(primary_n) is not int or primary_n < 1:
        _fail("invalid_primary_sample_size")
    if type(context_policies) is not list or not context_policies:
        _fail("invalid_context_policies")
    cells = []
    for values in product(*(axes[name] for name in _BASE_DIMENSIONS)):
        base = dict(zip(_BASE_DIMENSIONS, values))
        for policy in context_policies:
            if type(policy) is not str or not policy:
                _fail("invalid_context_policies")
            for sensitivity, sample_size in (("primary", primary_n), ("sensitivity_4n", primary_n * 4)):
                cell = {**base, "context_policy": policy, "sample_size": sample_size,
                        "sample_policy": sensitivity}
                cell["cell_id"] = digest({"format": CAMPAIGN_FORMAT, "version": VERSION, "cell": cell})
                cells.append(cell)
    return cells


def campaign_cache_key(cell, policies):
    """Bind sample/context choices to scientific identity, preventing policy aliasing."""
    if type(cell) is not dict or type(policies) is not dict:
        _fail("invalid_cache_identity")
    return digest({"format": "dope-campaign-cache-key", "version": 1,
                   "cell": cell, "policies": policies})


def _atomic_write(path, document):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json(document)
    pending = path.with_name(path.name + ".writing")
    with open(pending, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(pending, path)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_create(path, document):
    """Create only a new ledger; preserve existing or uncertain evidence."""
    path = Path(path)
    payload = canonical_json(document)
    descriptor = parent = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        except FileExistsError:
            _fail("execution_ledger_exists")
        remaining = memoryview(payload)
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                _fail("execution_ledger_write_uncertain")
            remaining = remaining[written:]
        os.fsync(descriptor)
        parent = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        os.fsync(parent)
    except OSError:
        raise ExecutionError("execution_ledger_write_uncertain") from None
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if parent is not None:
            os.close(parent)


def _validate_attempt(attempt):
    required = {"attempt_id", "run_id", "cell_id", "parent_attempt_id", "original_failure",
                "retry_authorized", "status", "requested_rows", "produced_rows", "provided_rows",
                "consumed_rows", "wall_ms", "cpu_ms", "gpu_ms", "residency_ms",
                "peak_ram_bytes", "peak_vram_bytes", "disk_bytes", "cost_basis", "run_pins"}
    if type(attempt) is not dict or set(attempt) != required:
        _fail("invalid_attempt")
    for key in ("attempt_id", "run_id", "cell_id"):
        if type(attempt[key]) is not str or not attempt[key]:
            _fail("invalid_attempt")
    if attempt["parent_attempt_id"] is not None and (type(attempt["parent_attempt_id"]) is not str or not attempt["parent_attempt_id"]):
        _fail("invalid_attempt")
    if attempt["status"] not in _TERMINAL:
        _fail("invalid_terminal_status")
    if type(attempt["retry_authorized"]) is not bool:
        _fail("invalid_attempt")
    if attempt["original_failure"] is not None and attempt["original_failure"] not in _TERMINAL:
        _fail("invalid_attempt")
    numeric = ("requested_rows", "produced_rows", "provided_rows", "consumed_rows", "wall_ms",
               "cpu_ms", "gpu_ms", "residency_ms", "peak_ram_bytes", "peak_vram_bytes", "disk_bytes")
    if any(type(attempt[k]) is not int or attempt[k] < 0 for k in numeric):
        _fail("invalid_attempt")
    if type(attempt["cost_basis"]) is not dict or set(attempt["cost_basis"]) != {"wall", "cpu", "gpu", "residency", "disk"}:
        _fail("invalid_cost_basis")
    if any(type(v) is not str or not v for v in attempt["cost_basis"].values()):
        _fail("invalid_cost_basis")
    if len(set(attempt["cost_basis"].values())) != len(attempt["cost_basis"]):
        _fail("overlapping_cost_basis")
    pins = attempt["run_pins"]
    if (type(pins) is not dict or set(pins) != {"parent_hashes", "toolchain_sha256", "command_sha256", "budget_sha256"}
            or type(pins["parent_hashes"]) is not list or not pins["parent_hashes"]
            or any(type(pin) is not str or not _HEX.fullmatch(pin) for pin in pins["parent_hashes"])
            or any(type(pins[key]) is not str or not _HEX.fullmatch(pins[key])
                   for key in ("toolchain_sha256", "command_sha256", "budget_sha256"))):
        _fail("invalid_run_pins")
    return copy.deepcopy(attempt)


class CampaignSchedule:
    """Frozen campaign denominator with fsynced attempt transitions."""

    def __init__(self, path, dimensions, primary_n, context_policies, policies=None, resource_caps=None):
        self.path = Path(path)
        self.roster = build_roster(dimensions, primary_n, context_policies)
        self.policies = policies or {"primary_n": primary_n, "sensitivity_multiplier": 4,
                                     "primary_sample_policy": "actual_R", "interpolation": False,
                                     "context_policies": context_policies}
        self.resource_caps = resource_caps or {}
        self.attempts = []
        self._persistence_failed = False
        self._validate_caps(self.resource_caps)
        _atomic_create(self.path, self.document())

    @staticmethod
    def _validate_caps(caps):
        if type(caps) is not dict or any(type(v) is not int or v < 0 for v in caps.values()):
            _fail("invalid_resource_caps")

    def document(self):
        return {"format": CAMPAIGN_FORMAT, "version": VERSION, "roster": self.roster,
                "roster_sha256": hashlib.sha256(canonical_json(self.roster)).hexdigest(),
                "policies": self.policies, "resource_caps": self.resource_caps,
                "attempts": self.attempts}

    def _write(self):
        _atomic_write(self.path, self.document())

    def append_attempt(self, attempt):
        if getattr(self, "_persistence_failed", False):
            _fail("ledger_write_uncertain")
        attempt = _validate_attempt(attempt)
        if attempt["cell_id"] not in {cell["cell_id"] for cell in self.roster}:
            _fail("unknown_campaign_cell")
        if any(item["attempt_id"] == attempt["attempt_id"] for item in self.attempts):
            _fail("duplicate_attempt")
        if any(item["run_id"] == attempt["run_id"] for item in self.attempts):
            _fail("duplicate_run")
        previous = [item for item in self.attempts if item["cell_id"] == attempt["cell_id"]]
        if previous:
            prior = previous[-1]
            if attempt["parent_attempt_id"] != prior["attempt_id"] or not attempt["retry_authorized"]:
                _fail("retry_not_authorized")
            if prior["status"] == "completed":
                _fail("completed_cell_replay")
            if attempt["original_failure"] != (prior["original_failure"] or prior["status"]):
                _fail("original_failure_changed")
        elif attempt["parent_attempt_id"] is not None or attempt["retry_authorized"]:
            _fail("retry_parent_missing")
        loading = getattr(self, "_loading", False)
        if not loading:
            # A retained overrun is a terminal budget breach, including after reload.
            for recorded in self.attempts:
                self._enforce_caps(recorded)
        previous = self.attempts
        self.attempts = [*previous, attempt]
        if not loading:
            persisted = False
            try:
                self._write()
                persisted = True
            except OSError:
                raise ExecutionError("ledger_write_uncertain") from None
            finally:
                if not persisted:
                    # The replace/fsync outcome may be unknown. No implicit replay.
                    self.attempts = previous
                    self._persistence_failed = True
            # A measured terminal outcome must be saved before signalling its overrun.
            self._enforce_caps(attempt)

    def _caps_exceeded(self, attempt):
        fields = {"wall_ms": "wall_ms", "cpu_ms": "cpu_ms", "gpu_ms": "gpu_ms",
                  "residency_ms": "residency_ms", "peak_ram_bytes": "peak_ram_bytes",
                  "peak_vram_bytes": "peak_vram_bytes", "disk_bytes": "disk_bytes"}
        return any(cap in self.resource_caps and attempt[field] > self.resource_caps[cap]
                   for cap, field in fields.items())

    def _enforce_caps(self, attempt):
        if self._caps_exceeded(attempt):
            _fail("resource_cap_exceeded")

    def guard_live_resources(self, usage, stop_owned_job):
        """Stop the owned process immediately when a measured cap is crossed."""
        if type(usage) is not dict or not callable(stop_owned_job):
            _fail("invalid_resource_usage")
        self._validate_caps(usage)
        exceeded = any(key in self.resource_caps and value > self.resource_caps[key]
                       for key, value in usage.items())
        if exceeded:
            stop_owned_job()
            return False
        return True

    def row_counts(self, attempt_id):
        item = next((row for row in self.attempts if row["attempt_id"] == attempt_id), None)
        if item is None:
            _fail("unknown_attempt")
        return {key: item[key] for key in ("requested_rows", "produced_rows", "provided_rows", "consumed_rows")}

    def complete(self, accepted=None):
        if (getattr(self, "_persistence_failed", False)
                or any(self._caps_exceeded(item) for item in self.attempts)):
            return False
        latest = {}
        for item in self.attempts:
            latest[item["cell_id"]] = item
        if accepted is None:
            accepted = lambda item: item["status"] == "completed"
        return (len(latest) == len(self.roster) and all(
            item["status"] != "unresolved" and accepted(item) for item in latest.values()))

    @classmethod
    def load(cls, path):
        value = parse_json(Path(path).read_bytes())
        required = {"format", "version", "roster", "roster_sha256", "policies", "resource_caps", "attempts"}
        if type(value) is not dict or set(value) != required or value["format"] != CAMPAIGN_FORMAT or value["version"] != VERSION:
            _fail("invalid_campaign_ledger")
        if hashlib.sha256(canonical_json(value["roster"])).hexdigest() != value["roster_sha256"]:
            _fail("roster_changed")
        if type(value["roster"]) is not list or not value["roster"] or type(value["attempts"]) is not list:
            _fail("invalid_campaign_ledger")
        schedule = object.__new__(cls)
        schedule.path = Path(path)
        schedule.roster, schedule.policies = value["roster"], value["policies"]
        schedule.resource_caps, schedule.attempts = value["resource_caps"], []
        schedule._persistence_failed = False
        schedule._loading = True
        cls._validate_caps(schedule.resource_caps)
        # Replay validation against an in-memory ledger; no action is reissued.
        for attempt in value["attempts"]:
            schedule.append_attempt(attempt)
        schedule._loading = False
        return schedule


class DurableSchedule:
    """Compatibility query-charge plan: interrupted charges survive reload."""
    def __init__(self, path, sources):
        self.path = Path(path)
        self.sources, self.attempts = self._partition(sources, [])
        _atomic_create(self.path, self.document())

    @staticmethod
    def _partition(sources, attempts):
        if type(sources) is not list or not sources or type(attempts) is not list:
            _fail("invalid_execution_plan")
        identities, stored = {}, []
        for item in sources:
            if type(item) is not dict or set(item) != {"source_id", "identity_sha256"}:
                _fail("invalid_execution_plan")
            source, identity = item["source_id"], item["identity_sha256"]
            if type(source) is not str or not source or type(identity) is not str or not _HEX.fullmatch(identity):
                _fail("invalid_execution_plan")
            if source in identities and identities[source] != identity:
                _fail("source_identity_mismatch")
            if source not in identities:
                identities[source] = identity
                stored.append(dict(item))
        seen, query_sources, stored_attempts = set(), {}, []
        for item in attempts:
            if (type(item) is not dict or set(item) != {"attempt_id", "source_id", "query_id", "charged_queries", "status"}
                    or any(type(item[k]) is not str or not item[k] for k in ("attempt_id", "source_id", "query_id"))
                    or type(item["charged_queries"]) is not int or item["charged_queries"] < 0
                    or item["status"] not in {"scheduled", "interrupted", "complete"}):
                _fail("invalid_execution_plan")
            if item["attempt_id"] in seen or item["source_id"] not in identities:
                _fail("duplicate_attempt" if item["attempt_id"] in seen else "unknown_source")
            if item["query_id"] in query_sources and query_sources[item["query_id"]] != item["source_id"]:
                _fail("query_charge_source_mismatch")
            seen.add(item["attempt_id"])
            query_sources[item["query_id"]] = item["source_id"]
            stored_attempts.append(dict(item))
        return stored, stored_attempts

    def document(self):
        return {"format": "dope-transfer-execution-plan", "version": 1,
                "sources": self.sources, "attempts": self.attempts}

    def _write(self):
        _atomic_write(self.path, self.document())

    def charge(self, attempt_id, source_id, query_id, charged_queries):
        if any(item["attempt_id"] == attempt_id for item in self.attempts):
            _fail("duplicate_attempt")
        if source_id not in {item["source_id"] for item in self.sources}:
            _fail("unknown_source")
        if any(item["query_id"] == query_id and item["source_id"] != source_id for item in self.attempts):
            _fail("query_charge_source_mismatch")
        attempt = {"attempt_id": attempt_id, "source_id": source_id,
                              "query_id": query_id, "charged_queries": charged_queries,
                              "status": "scheduled"}
        self._partition(self.sources, [*self.attempts, attempt])
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
        schedule = object.__new__(cls)
        if type(value) is not dict or set(value) != {"format", "version", "sources", "attempts"} or value["format"] != FORMAT or value["version"] != VERSION:
            _fail("invalid_execution_plan")
        schedule.path = Path(path)
        schedule.sources, schedule.attempts = schedule._partition(value["sources"], value["attempts"])
        return schedule
