"""Linux file-open verification for fresh, trusted sampler processes.

All tracees are followed, including native threads. Undeclared accesses reject
the result after execution; use the native isolated runner for OS enforcement.
Raw logs, worker stderr and local paths never enter the public receipt.
"""
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

from .common import MetricError, require

ROOT = Path(__file__).resolve().parents[3]
_NAME = re.compile(r"[a-z0-9][a-z0-9_.-]{0,63}")
_METADATA = {"access", "faccessat", "faccessat2", "stat", "lstat", "stat64", "lstat64",
             "newfstatat", "statx", "readlink", "readlinkat", "statfs", "statfs64", "getcwd"}
_PROCESS = {"exit", "exit_group", "wait4", "waitid", "getpid", "getppid", "gettid"}


def verify_sampler_access_trace(declared_sections, declared_closure, observed_reads):
    """Same inventory-ID admission and reason codes as Rust's verifier."""
    allowed = set(declared_sections) | set(declared_closure)
    require(bool(observed_reads), "empty_sampler_trace")
    for name in observed_reads:
        require(type(name) is str and _NAME.fullmatch(name) is not None and name in allowed,
                "undeclared_sampler_file_access")


def _path(encoded):
    require(re.fullmatch(r"(?:\\x[0-9a-fA-F]{2})+", encoded) is not None, "malformed_sampler_trace")
    return Path(os.fsdecode(bytes.fromhex(encoded.replace("\\x", "")))).resolve()


def _calls(text):
    pending = {}
    for line in text.splitlines():
        match = re.fullmatch(r"(\d+)\s+(.*)", line)
        require(match is not None, "malformed_sampler_trace")
        pid, body = match.groups()
        if body.startswith(("--- ", "+++ ")):
            continue
        if body.endswith("<unfinished ...>"):
            require(pid not in pending, "malformed_sampler_trace")
            pending[pid] = body.removesuffix("<unfinished ...>")
            continue
        resumed = re.fullmatch(r"<\.\.\. (\w+) resumed>(.*)", body)
        if resumed:
            require(pid in pending, "malformed_sampler_trace")
            original = pending.pop(pid)
            require(original.startswith(resumed[1] + "("), "malformed_sampler_trace")
            body = original + resumed[2]
        match = re.fullmatch(r"(\w+)\((.*)\)\s+=\s+(.*)", body)
        require(match is not None, "malformed_sampler_trace")
        yield match.groups()
    require(not pending, "incomplete_sampler_trace")


def _events(text, reads, writes):
    events, executed, exited = [], False, False
    directories = {parent for path in (*reads, *writes) for parent in path.parents}
    for name, args, result in _calls(text):
        if name in {"execve", "execveat"}:
            require(not executed and name == "execve", "sampler_external_access")
            require(result == "0", "sampler_failed")
            token = re.match(r'"([^"]*)"', args)
            require(token is not None, "malformed_sampler_trace")
            path = _path(token[1])
            require(path in reads, "undeclared_sampler_file_access")
            events.append({"operation": "read", "inventory_id": reads[path]})
            executed = True
        elif name in {"open", "openat", "openat2", "creat"}:
            if result.startswith("-1 "):
                continue  # Failed loader probes did not open or read a file.
            opened = re.fullmatch(r"\d+<([^>]+)>", result)
            require(opened is not None, "malformed_sampler_trace")
            path = _path(opened[1])  # Kernel-resolved fd path catches symlink/dirfd escapes.
            if "O_DIRECTORY" in args and path in directories:
                continue  # Runtime directory discovery is metadata, not file content.
            write = name == "creat" or any(flag in args for flag in ("O_WRONLY", "O_RDWR", "O_CREAT", "O_TRUNC", "O_APPEND"))
            read = name != "creat" and "O_WRONLY" not in args
            require(not read or path in reads, "undeclared_sampler_file_access")
            require(not write or path in writes, "sampler_undeclared_write")
            if read:
                events.append({"operation": "read", "inventory_id": reads[path]})
            if write:
                events.append({"operation": "write", "inventory_id": writes[path]})
        elif name in {"clone", "clone3"}:
            require("CLONE_THREAD" in args, "sampler_external_access")
        elif name in _METADATA:
            continue
        elif name in _PROCESS:
            exited |= name == "exit_group"
        else:
            # Network, fork, namespace mutation, descriptor stealing and opaque
            # asynchronous I/O are never silently accepted as file-open proof.
            raise MetricError("sampler_external_access")
    require(executed and exited, "incomplete_sampler_trace")
    return events


def trace_command(command, reads, writes, *, timeout):
    tracer = shutil.which("strace")
    require(sys.platform == "linux" and tracer is not None, "sampler_trace_unavailable")
    require(type(command) is list and command and all(type(s) is str for s in command), "invalid_sampler_command")
    require(Path(command[0]).is_absolute() and Path(command[0]).resolve() in reads,
            "undeclared_sampler_file_access")
    target = ROOT / "target"
    target.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target, prefix="sampler-trace-") as directory:
        log = Path(directory) / "syscalls"
        trace = "%file,%process,%network,io_uring_setup,io_uring_enter,io_uring_register,pidfd_getfd,process_vm_readv,process_vm_writev,ptrace"
        try:
            worker = subprocess.run([tracer, "-f", "-qq", "-yy", "-xx", "-s", "0", "--kill-on-exit",
                                     "-e", "trace=" + trace, "-o", str(log), "--", *command],
                                    cwd=ROOT, env={"LC_ALL": "C", "PATH": os.defpath,
                                                   "PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1"},
                                    stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    close_fds=True, timeout=timeout)
            require(log.is_file(), "sampler_trace_unavailable")
            events = _events(log.read_text(), reads, writes)
            require(worker.returncode == 0, "sampler_failed")
            return worker.stdout, events
        except subprocess.TimeoutExpired:
            raise MetricError("sampler_timeout") from None
        except OSError:
            raise MetricError("sampler_trace_unavailable") from None
