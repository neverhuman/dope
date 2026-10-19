"""Development stage accounting; no worker launch or scientific admission.

The immutable spec binds an existing scheduler roster, source and command pins.
This journal records stage reservations before work and preserves interrupted
reservations across reload. It is bookkeeping, not a query-access ledger or an
independent native-runtime receipt. Real pilot/custodian execution is separate.
"""
from __future__ import annotations

from dataclasses import dataclass
import copy
import fcntl
import os
from pathlib import Path
import re
import stat

from research.transfer.evidence import EvidenceError, canonical_json, digest, parse_json
from research.transfer.scheduler import CampaignSchedule

FORMAT = "dope-development-campaign-run-spec"
STAGES = ("prepare", "compile", "reload", "sample", "learners", "drivers", "receipt", "report")
MAX_SPEC_BYTES = 1024 * 1024
MAX_JOURNAL_BYTES = 64 * 1024 * 1024
JOURNAL = "development-stages.jsonl"
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_ZERO = "0" * 64


class CampaignRunnerError(ValueError):
    """Static refusal codes only; do not include source values or paths."""


def _require(ok, code):
    if not ok:
        raise CampaignRunnerError(code)


def _integer(value, minimum=0):
    return type(value) is int and minimum <= value <= 2**53 - 1


def _hash(value):
    return type(value) is str and _HEX.fullmatch(value) is not None


@dataclass(frozen=True)
class CampaignRunSpec:
    """Canonical bytes prevent caller mutation from changing an admitted plan."""
    _encoded: bytes

    @classmethod
    def from_document(cls, document):
        try:
            encoded = canonical_json(document)
            document = parse_json(encoded)
        except EvidenceError:
            raise CampaignRunnerError("invalid_run_spec") from None
        fields = {"format", "version", "phase", "source_commit", "source_tree",
                  "environment_sha256", "protocol_sha256", "roster_sha256", "cells", "budget", "max_attempts"}
        _require(len(encoded) <= MAX_SPEC_BYTES and type(document) is dict
                 and set(document) == fields, "invalid_run_spec")
        _require(document["format"] == FORMAT and type(document["version"]) is int
                 and document["version"] == 1 and document["phase"] == "development",
                 "development_spec_required")
        for key in ("source_commit", "source_tree"):
            _require(type(document[key]) is str and _COMMIT.fullmatch(document[key]), "invalid_source_pin")
        for key in ("environment_sha256", "protocol_sha256", "roster_sha256"):
            _require(_hash(document[key]), "invalid_runtime_pin")
        cells = document["cells"]
        _require(type(cells) is list and 0 < len(cells) <= 4096, "invalid_cell_roster")
        for cell in cells:
            _require(type(cell) is dict and set(cell) == {"cell_id", "input_sha256", "command_sha256"}
                     and all(_hash(v) for v in cell.values()), "invalid_cell_binding")
        _require(len({c["cell_id"] for c in cells}) == len(cells), "duplicate_cell")
        budget = document["budget"]
        _require(type(budget) is dict and set(budget) == {"wall_ms", "peak_ram_bytes"}
                 and all(_integer(v, 1) for v in budget.values()), "invalid_budget")
        _require(_integer(document["max_attempts"], 1) and document["max_attempts"] <= 16,
                 "invalid_attempt_budget")
        return cls(encoded)

    @classmethod
    def from_bytes(cls, value):
        _require(type(value) is bytes and len(value) <= MAX_SPEC_BYTES, "invalid_run_spec")
        try:
            return cls.from_document(parse_json(value))
        except EvidenceError:
            raise CampaignRunnerError("invalid_run_spec") from None

    def document(self):
        return parse_json(self._encoded)

    @property
    def sha256(self):
        return digest(self.document())

    def verify_schedule(self, schedule):
        """An execution bridge must match every cell to the existing scheduler."""
        _require(type(schedule) is CampaignSchedule, "campaign_schedule_required")
        observed, declared = schedule.document(), self.document()
        _require(observed["roster_sha256"] == declared["roster_sha256"]
                 and {c["cell_id"] for c in observed["roster"]}
                 == {c["cell_id"] for c in declared["cells"]}, "campaign_roster_mismatch")

    def public_summary(self):
        return {"format": FORMAT, "plan_sha256": self.sha256,
                "cells": len(self.document()["cells"]), "stages_per_cell": len(STAGES),
                "worker_executed": False, "scientific_qualification": "unmeasured",
                "release_authorization": "not_authorized"}


class StageJournal:
    """Exclusive, private, fsynced stage-intent journal under an existing root.

    Callers run no work here. A begin reservation must return before invoking
    an independently admitted stage executor. Reload never reissues work.
    """
    def __init__(self, directory, spec, *, create=False, read_only=False):
        _require(type(spec) is CampaignRunSpec, "invalid_run_spec")
        _require(not (create and read_only), "invalid_journal_mode")
        self.directory = Path(directory)
        self.spec = CampaignRunSpec.from_bytes(spec._encoded)
        self.create, self.read_only = create, read_only
        self.fd = None
        self.records, self.state, self.charged, self.over_budget = [], {}, 0, False

    def __enter__(self):
        root = None
        try:
            root = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            info = os.fstat(root)
            _require(info.st_uid == os.geteuid() and info.st_mode & 0o077 == 0,
                     "unsafe_evidence_directory")
            flags = os.O_RDONLY if self.read_only else os.O_RDWR
            if self.create:
                flags |= os.O_CREAT | os.O_EXCL
            self.fd = os.open(JOURNAL, flags | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600, dir_fd=root)
            info = os.fstat(self.fd)
            _require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1
                     and info.st_uid == os.geteuid() and info.st_mode & 0o077 == 0,
                     "unsafe_stage_journal")
            fcntl.flock(self.fd, (fcntl.LOCK_SH if self.read_only else fcntl.LOCK_EX) | fcntl.LOCK_NB)
            _require(info.st_size <= MAX_JOURNAL_BYTES, "journal_size_limit")
            if self.create:
                self._append("header", {})
                os.fsync(root)
            else:
                chunks, size = [], 0
                while True:
                    chunk = os.read(self.fd, min(65536, MAX_JOURNAL_BYTES + 1 - size))
                    if not chunk:
                        break
                    chunks.append(chunk)
                    size += len(chunk)
                    _require(size <= MAX_JOURNAL_BYTES, "journal_size_limit")
                data = b"".join(chunks)
                _require(bool(data) and data.endswith(b"\n"), "incomplete_stage_journal")
                for line in data.splitlines():
                    self._accept(parse_json(line))
            return self
        except (OSError, EvidenceError):
            self.__exit__(None, None, None)
            raise CampaignRunnerError("stage_journal_unavailable") from None
        except Exception:
            self.__exit__(None, None, None)
            raise
        finally:
            if root is not None:
                os.close(root)

    def __exit__(self, *_):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None

    def _accept(self, record, *, persist=False):
        _require(type(record) is dict and set(record) == {
            "type", "plan_sha256", "sequence", "previous_sha256", "data", "sha256"},
            "invalid_stage_record")
        body = {k: v for k, v in record.items() if k != "sha256"}
        previous = self.records[-1]["sha256"] if self.records else _ZERO
        _require(record["sha256"] == digest(body) and record["plan_sha256"] == self.spec.sha256
                 and type(record["sequence"]) is int and record["sequence"] == len(self.records)
                 and record["previous_sha256"] == previous, "stage_binding_mismatch")
        state, charged, over = copy.deepcopy(self.state), self.charged, self.over_budget
        value, document = record["data"], self.spec.document()
        if not self.records:
            _require(record["type"] == "header" and value == {}, "invalid_stage_header")
        else:
            _require(type(record["type"]) is str and record["type"] in {"begin", "finish"}
                     and type(value) is dict,
                     "invalid_stage_transition")
            common = {"cell_id", "stage", "attempt"}
            extra = {"reserved_wall_ms", "retry"} if record["type"] == "begin" else {
                "status", "observed_wall_ms", "peak_ram_bytes", "output_sha256"}
            _require(set(value) == common | extra and _hash(value["cell_id"])
                     and type(value["stage"]) is str and value["cell_id"] in {
                c["cell_id"] for c in document["cells"]} and value["stage"] in STAGES,
                "invalid_stage_transition")
            key = (value["cell_id"], value["stage"])
            item = state.setdefault(key, {"attempts": 0, "pending": False, "status": None})
            if record["type"] == "begin":
                n = STAGES.index(value["stage"])
                _require(not over and not item["pending"] and item["status"] != "completed",
                         "stage_not_ready")
                _require(n == 0 or state.get((key[0], STAGES[n - 1]), {}).get("status") == "completed",
                         "previous_stage_incomplete")
                _require(type(value["retry"]) is bool and value["retry"] == (item["attempts"] > 0),
                         "retry_disposition_required")
                _require(type(value["attempt"]) is int and value["attempt"] == item["attempts"] + 1
                         and value["attempt"] <= document["max_attempts"], "attempt_budget_exceeded")
                _require(_integer(value["reserved_wall_ms"], 1)
                         and charged + value["reserved_wall_ms"] <= document["budget"]["wall_ms"],
                         "wall_budget_exceeded")
                charged += value["reserved_wall_ms"]
                item.update(attempts=value["attempt"], pending=True, reservation=value["reserved_wall_ms"])
            else:
                _require(item["pending"] and type(value["attempt"]) is int
                         and value["attempt"] == item["attempts"], "no_pending_stage")
                _require(type(value["status"]) is str
                         and value["status"] in {"completed", "failed", "interrupted", "timeout", "over_budget"}
                         and all(v is None or _integer(v) for v in (
                             value["observed_wall_ms"], value["peak_ram_bytes"]))
                         and (value["output_sha256"] is None or _hash(value["output_sha256"])),
                         "invalid_stage_result")
                if value["status"] == "completed":
                    _require(value["observed_wall_ms"] is not None and value["peak_ram_bytes"] is not None
                             and _hash(value["output_sha256"]), "stage_observations_required")
                exceeded = ((value["observed_wall_ms"] or 0) > item["reservation"]
                            or (value["peak_ram_bytes"] or 0) > document["budget"]["peak_ram_bytes"])
                _require(not exceeded or value["status"] == "over_budget", "resource_overrun_not_recorded")
                over = over or value["status"] == "over_budget"
                item.update(pending=False, status=value["status"])
        if persist:
            _require(not self.read_only and self.fd is not None, "journal_read_only")
            encoded = canonical_json(record) + b"\n"
            _require(os.fstat(self.fd).st_size + len(encoded) <= MAX_JOURNAL_BYTES, "journal_size_limit")
            try:
                os.lseek(self.fd, 0, os.SEEK_END)
                view = memoryview(encoded)
                while view:
                    written = os.write(self.fd, view)
                    _require(written > 0, "journal_write_failed")
                    view = view[written:]
                os.fsync(self.fd)
            except (OSError, CampaignRunnerError):
                os.close(self.fd)
                self.fd = None
                raise CampaignRunnerError("journal_write_failed") from None
        self.records.append(record)
        self.state, self.charged, self.over_budget = state, charged, over

    def _append(self, kind, data):
        body = {"type": kind, "plan_sha256": self.spec.sha256, "sequence": len(self.records),
                "previous_sha256": self.records[-1]["sha256"] if self.records else _ZERO, "data": data}
        self._accept({**body, "sha256": digest(body)}, persist=True)

    def begin(self, cell_id, stage, reserved_wall_ms, *, retry=False):
        item = self.state.get((cell_id, stage), {"attempts": 0})
        self._append("begin", {"cell_id": cell_id, "stage": stage, "attempt": item["attempts"] + 1,
                               "reserved_wall_ms": reserved_wall_ms, "retry": retry})

    def finish(self, cell_id, stage, status, *, observed_wall_ms=None, peak_ram_bytes=None,
               output_sha256=None):
        item = self.state.get((cell_id, stage), {"attempts": 0})
        self._append("finish", {"cell_id": cell_id, "stage": stage, "attempt": item["attempts"],
                                "status": status, "observed_wall_ms": observed_wall_ms,
                                "peak_ram_bytes": peak_ram_bytes, "output_sha256": output_sha256})

    def public_summary(self):
        return {**self.spec.public_summary(), "completion_kind": "stage_bookkeeping_only",
                "charged_wall_ms": self.charged, "pending_stages": sum(i["pending"] for i in self.state.values()),
                "completed_stages": sum(i["status"] == "completed" for i in self.state.values()),
                "resource_overrun": self.over_budget}
