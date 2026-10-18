"""Irreversible Linux fresh-worker boundary for recipient bundle inventory.

Rust owns sampling. This helper proves Python inventory needs no external
files or sockets; it does not turn Python inventory into sampling evidence.
Load this module in a dedicated source-free process before calling verify.
"""

from __future__ import annotations

import ctypes
import fcntl
import os
from pathlib import Path
import platform

from research.transfer.bundle import BundleError, BundleLimits, _closure, _manifest, verify_bundle
from research.transfer.evidence import EvidenceError, parse_json
# Preload the reader before Landlock denies access to Python source modules.
from research.transfer import bundle_io as _bundle_io  # noqa: F401
from research.transfer import raw_profile as _raw_profile  # noqa: F401


class _Ruleset(ctypes.Structure):
    _fields_ = [("handled_access_fs", ctypes.c_uint64)]


class _PathRule(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int)]


class _Filter(ctypes.Structure):
    _fields_ = [("code", ctypes.c_ushort), ("jt", ctypes.c_ubyte),
                ("jf", ctypes.c_ubyte), ("k", ctypes.c_uint32)]


class _Program(ctypes.Structure):
    _fields_ = [("length", ctypes.c_ushort), ("filter", ctypes.POINTER(_Filter))]


_LIBC = ctypes.CDLL(None, use_errno=True)
_LIBC.syscall.restype = ctypes.c_long


def _fail(code):
    raise BundleError(code) from None


def _pin_directory(root):
    path = Path(root)
    if ".." in path.parts:
        _fail("bundle_directory_unsafe")
    flags = os.O_PATH | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    directory = os.open("/" if path.is_absolute() else ".", flags)
    try:
        for part in path.parts:
            if part in ("/", "."):
                continue
            child = os.open(part, flags, dir_fd=directory)
            os.close(directory)
            directory = child
        return directory
    except BaseException:
        os.close(directory)
        raise


def _filter_reads():
    # x86_64 Linux, default deny. Landlock confines opens and directory reads;
    # sockets, exec, io_uring and process-memory/descriptor borrowing are absent.
    allowed = (
        0, 1, 2, 3, 5, 8, 9, 10, 11, 12, 13, 14, 15, 19, 24, 28, 32, 33,
        35, 39, 60, 72, 157, 158, 186, 202, 217, 228, 230, 231, 234,
        257, 262, 273, 292, 302, 318, 332,
    )
    instructions = [
        (0x20, 0, 0, 4), (0x15, 1, 0, 0xC000003E),
        (0x06, 0, 0, 0x80000000), (0x20, 0, 0, 0),
        (0x35, 0, 1, 0x40000000), (0x06, 0, 0, 0x80000000),
    ]
    # O_PATH opens bypass Landlock's read permission checks. No such handles
    # or writable file opens are needed after the directory has been pinned.
    for number, offset in ((2, 24), (257, 32)):
        instructions.extend(((0x15, 0, 3, number), (0x20, 0, 0, offset),
                             (0x45, 0, 1, os.O_PATH | os.O_ACCMODE),
                             (0x06, 0, 0, 0x00050001), (0x20, 0, 0, 0)))
    for number in allowed:
        instructions.extend(((0x15, 0, 1, number), (0x06, 0, 0, 0x7FFF0000)))
    instructions.append((0x06, 0, 0, 0x00050001))  # EPERM
    filters = (_Filter * len(instructions))(*(_Filter(*item) for item in instructions))
    program = _Program(len(filters), filters)
    if _LIBC.syscall(317, 1, 1, ctypes.byref(program)) != 0:
        _fail("bundle_isolation_refused")


def confine_bundle_directory(root):
    """Close inherited inputs, clear environment, restrict reads and sockets.

    Fail closed if the kernel cannot enforce Landlock ABI 3 and seccomp.
    This is irreversible and requires a dedicated single-threaded process
    that has never loaded source rows, fit partitions or private priors.
    """
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        _fail("sampling_isolation_requires_linux_x86_64")
    try:
        if len(os.listdir("/proc/self/task")) != 1:
            _fail("sampling_worker_not_fresh")
        for fd in (1, 2):
            if fcntl.fcntl(fd, fcntl.F_GETFL) & os.O_ACCMODE != os.O_WRONLY:
                _fail("sampling_result_stream_must_be_write_only")
        if _LIBC.syscall(436, 3, ctypes.c_uint(-1), 0) != 0:
            _fail("sampling_fd_closure_unavailable")
        try:
            os.close(0)
        except OSError:
            pass
        directory = _pin_directory(root)
        try:
            rules = _Ruleset((1 << 15) - 1)
            ruleset = _LIBC.syscall(444, ctypes.byref(rules), ctypes.sizeof(rules), 0)
            if ruleset < 0:
                _fail("bundle_isolation_refused")
            try:
                rule = _PathRule((1 << 2) | (1 << 3), directory)
                if (_LIBC.syscall(445, ruleset, 1, ctypes.byref(rule), 0) != 0
                        or _LIBC.prctl(38, 1, 0, 0, 0) != 0
                        or _LIBC.syscall(446, ruleset, 0) != 0):
                    _fail("bundle_isolation_refused")
            finally:
                os.close(ruleset)
        finally:
            os.close(directory)
        os.environ.clear()
        _filter_reads()
    except BundleError:
        raise
    except (OSError, ValueError):
        _fail("bundle_isolation_refused")


def verify_source_free_bundle(root, expected_manifest_sha256, limits=None):
    """Confine before inventory reload; return only verified retained bytes."""
    limits = limits or BundleLimits()
    confine_bundle_directory(root)
    digest, _, data = _bundle_io._read_bytes(root, "manifest.json", limits.max_manifest_bytes)
    if digest != expected_manifest_sha256:
        _fail("manifest_digest_mismatch")
    try:
        handler, _, _, components = _manifest(parse_json(data), limits)
    except EvidenceError:
        _fail("invalid_manifest")
    if handler not in {"native-inline-dpk", "native-raw-complete-v1"}:
        _fail("sampling_handler_refused")
    edges = {c["id"]: c["depends_on"] for c in components}
    reachable = set()
    for component in components:
        if component["role"] == "sampling_payload":
            reachable.update(_closure(component["id"], edges))
    if any(c["role"] in {"evidence_authorization", "materialized_rows"}
           and not c["required"] and c["id"] not in reachable for c in components):
        _fail("uncharged_dependency")
    return verify_bundle(root, expected_manifest_sha256, limits)
