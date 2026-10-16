"""Python sampler access trace for trusted offline adapters (T-035/T-063).

The CPython audit hook observes Python-level file opens in the current thread.
It is not an OS sandbox: native samplers require an independently traced runner.
Receipts expose inventory IDs and reason codes, never machine paths or values.
"""
from contextlib import contextmanager
from pathlib import Path
import os
import sys
import threading

from .common import MetricError, require

_LOCAL = threading.local()


def _hook(event, args):
    guard = getattr(_LOCAL, "guard", None)
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

    @staticmethod
    def _inventory(entries):
        result = {}
        for path, identifier in dict(entries).items():
            require(type(identifier) is str and identifier and "/" not in identifier and "\\" not in identifier,
                    "invalid_inventory_id")
            resolved = Path(path).resolve()
            require(resolved not in result, "duplicate_inventory_path")
            result[resolved] = identifier
        return result

    def block(self, reason):
        self.denied = True
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
        return {"trace_kind": "cpython_audit_hook", "status": "failed" if self.denied else "passed",
                "events": list(self.events), "citable": False}

    @contextmanager
    def activate(self):
        require(getattr(_LOCAL, "guard", None) is None, "nested_access_trace")
        _LOCAL.guard = self
        try:
            yield self
        finally:
            _LOCAL.guard = None
