"""Private bounded snapshots for admitted development native stages."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import stat

from research.transfer.campaign_runner import CampaignRunnerError, _require
from research.transfer.evidence import digest

MAX_INPUT = 8 << 20
MAX_BINARY = 256 << 20


def read_regular(path, limit=MAX_INPUT):
    """No FIFO, final symlink, hardlink alias or oversized read."""
    fd = None
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        info = os.fstat(fd)
        _require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1
                 and info.st_size <= limit, "native_file_unavailable")
        chunks, total = [], 0
        while total <= limit:
            part = os.read(fd, min(65536, limit + 1 - total))
            if not part:
                break
            chunks.append(part)
            total += len(part)
        _require(total <= limit, "native_file_limit")
        return b"".join(chunks)
    except OSError:
        raise CampaignRunnerError("native_file_unavailable") from None
    finally:
        if fd is not None:
            os.close(fd)


def file_hash(path, limit=MAX_INPUT):
    return hashlib.sha256(read_regular(path, limit)).hexdigest()


def create_private(path, data, mode=0o600):
    fd = None
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, mode)
        view = memoryview(data)
        while view:
            count = os.write(fd, view)
            _require(count > 0, "native_write_failed")
            view = view[count:]
        os.fsync(fd)
        parent = os.open(Path(path).parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
    except OSError:
        raise CampaignRunnerError("native_output_unavailable") from None
    finally:
        if fd is not None:
            os.close(fd)


def snapshot(source, destination, expected, *, executable=False):
    data = read_regular(source, MAX_BINARY if executable else MAX_INPUT)
    _require(hashlib.sha256(data).hexdigest() == expected, "native_input_pin_mismatch")
    create_private(destination, data, 0o500 if executable else 0o400)
    return {"sha256": expected, "bytes": len(data)}


def private_root(path):
    path = Path(path).absolute()
    fd = None
    try:
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        info = os.fstat(fd)
        _require(info.st_uid == os.geteuid() and info.st_mode & 0o077 == 0,
                 "unsafe_evidence_directory")
        return path
    except OSError:
        raise CampaignRunnerError("unsafe_evidence_directory") from None
    finally:
        if fd is not None:
            os.close(fd)


def directory_pin(path):
    """Bind operator authorization to one existing private evidence directory."""
    path = private_root(path)
    info = path.stat()
    return digest({"path": str(path), "device": info.st_dev,
                   "inode": info.st_ino, "owner": info.st_uid})


def sync_artifact(path):
    """Persist native output before a successful stage completion is journaled."""
    fd = None
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        info = os.fstat(fd)
        _require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1
                 and info.st_uid == os.geteuid(), "unsafe_native_artifact")
        os.fsync(fd)
        parent = os.open(Path(path).parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
    except OSError:
        raise CampaignRunnerError("native_artifact_sync_failed") from None
    finally:
        if fd is not None:
            os.close(fd)
