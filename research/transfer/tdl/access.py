"""Declared sampler reads and exact charges, with full-process trace support.

The audit context is a diagnostic, covering Python events in all threads.
Only run() can issue a passed trace: its fresh worker is traced by Linux strace,
including native opens and threads. This is verification, not an OS sandbox.
"""
from contextlib import contextmanager
import hashlib
from pathlib import Path
import os
import re
import sys
import threading

from . import charge
from .common import MetricError, require

_ACTIVE = None
_ACTIVE_LOCK = threading.RLock()


def _hook(event, args):
    guard = _ACTIVE
    if guard is None:
        return
    if event == "open":
        guard.open(*args)
    elif event in {"socket.__new__", "subprocess.Popen", "os.system", "os.fork", "os.posix_spawn", "ctypes.dlopen"}:
        guard.block("sampler_external_access")


sys.addaudithook(_hook)


class AccessTrace:
    def __init__(self, reads, writes=()):
        """reads/writes map a local path to an opaque charged inventory ID."""
        self.reads = self._inventory(reads)
        self.writes = self._inventory(writes)
        self.events = []
        self.denied = False
        self.reason = None
        self.accounting = None
        self._pins = {}
        self._complete = False
        self._native = False

    @staticmethod
    def _inventory(entries):
        result = {}
        for path, identifier in dict(entries).items():
            require(type(identifier) is str and re.fullmatch(r"[a-z0-9][a-z0-9_.-]{0,63}", identifier),
                    "invalid_inventory_id")
            resolved = Path(path).resolve()
            require(resolved not in result, "duplicate_inventory_path")
            result[resolved] = identifier
        require(len(set(result.values())) == len(result), "duplicate_inventory_id")
        return result

    @classmethod
    def from_manifest(cls, artifact, manifest, closure_paths, writes=()):
        """Bind every readable file to the same TDL1 bytes charged by Rust."""
        accounting = charge.evaluate(manifest)
        require(accounting["status"] == "complete", accounting["reason"])
        expected = charge.canonical_container(manifest["sections"])
        paths = {"payload": Path(artifact).resolve()}
        contents = {"payload": expected}
        require(set(closure_paths) == {entry["name"] for entry in manifest["closure_files"]},
                "closure_inventory_mismatch")
        for entry in manifest["closure_files"]:
            name = entry["name"]
            require(name not in paths, "duplicate_inventory_id")
            paths[name] = Path(closure_paths[name]).resolve()
            contents[name] = bytes.fromhex(entry["hex"])
        result = cls({path: name for name, path in paths.items()}, writes)
        require(len(result.reads) == len(paths), "duplicate_inventory_path")
        require(not set(result.reads) & set(result.writes), "mutable_sampling_inventory")
        try:
            for name, path in paths.items():
                require(path.is_file() and path.read_bytes() == contents[name], "charge_inventory_mismatch")
                result._pins[path] = hashlib.sha256(contents[name]).digest()
        except OSError:
            raise MetricError("charge_inventory_unavailable") from None
        result.accounting = accounting
        return result

    def block(self, reason):
        self.denied = True
        self.reason = reason
        self.events.append({"operation": "denied", "inventory_id": None})
        raise MetricError(reason)

    def open(self, path, mode, flags):
        if type(path) is int:
            self.block("sampler_untracked_descriptor")
        try:
            # Absolute normalization avoids filesystem reads inside the hook.
            # Inventory entries are resolved at admission; symlinks are checked
            # against their destination each time to catch a swapped sidecar.
            resolved = Path(os.fsdecode(path)).resolve()
        except (TypeError, ValueError, OSError, RuntimeError):
            self.block("sampler_untracked_access")
        write = bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
        read = not bool(flags & os.O_WRONLY)
        if read and resolved not in self.reads:
            self.block("sampler_uncharged_read")
        if write and resolved not in self.writes:
            self.block("sampler_undeclared_write")
        if read:
            self.events.append({"operation": "read", "inventory_id": self.reads[resolved]})
        if write:
            self.events.append({"operation": "write", "inventory_id": self.writes[resolved]})

    def receipt(self):
        status = "failed" if self.denied else ("passed" if self._complete else "incomplete")
        result = {"trace_kind": "linux_strace" if self._native else "cpython_audit_hook",
                  "status": status, "reason": self.reason if self.denied else
                  (None if self._complete else "native_syscalls_untraced"),
                  "events": list(self.events), "citable": False}
        if self._complete:
            result.update(coverage="all_threads_and_native_file_opens", byte_accounting=self.accounting,
                          declared_payload=["payload"],
                          declared_closure=sorted(name for name in self.reads.values() if name != "payload"))
        return result

    def run(self, command, *, timeout=90):
        """Return worker stdout only after its full trace and charged files pass."""
        from .syscall_trace import trace_command, verify_sampler_access_trace
        require(self.accounting is not None, "missing_charge_manifest")
        self.events, self.denied, self.reason, self._complete = [], False, None, False
        self._native = True
        try:
            self._verify_pins()
            stdout, events = trace_command(command, self.reads, self.writes, timeout=timeout)
            self.events = events
            verify_sampler_access_trace(["payload"],
                                        [name for name in self.reads.values() if name != "payload"],
                                        [event["inventory_id"] for event in events if event["operation"] == "read"])
            self._verify_pins()
            self._complete = True
            return stdout, self.receipt()
        except MetricError as error:
            self.block(str(error))

    def _verify_pins(self):
        try:
            require(all(path.is_file() and hashlib.sha256(path.read_bytes()).digest() == pin
                        for path, pin in self._pins.items()), "sampler_inventory_changed")
        except OSError:
            raise MetricError("sampler_inventory_changed") from None

    @contextmanager
    def activate(self):
        global _ACTIVE
        with _ACTIVE_LOCK:
            require(_ACTIVE is None, "nested_access_trace")
            _ACTIVE = self
        try:
            yield self
        finally:
            with _ACTIVE_LOCK:
                _ACTIVE = None
