"""Read one registered paper input: exactly the bytes whose length and digest are declared in advance.

A registry maps a relative path under one root to its byte length and sha256.
A read refuses, before any byte is decoded: a name absent from the registry; an
empty, absolute, or traversing name; a symlink at the leaf or at any directory
between the root and the leaf; a root reached through a symlink; anything that
is not a regular file; and a length or digest that differs. The body is read
once, bounded by its declared length, and the caller parses those same bytes.
Error messages name the rule, never file contents.
"""

from __future__ import annotations

import hashlib
import os
import stat
from pathlib import Path, PurePosixPath


class PinRefused(ValueError):
    """A registered-input refusal; the message is a fixed rule name."""


def check_name(relative: str) -> None:
    """Refuse an empty, absolute, backslashed, or traversing name before any filesystem call."""
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise PinRefused("input name is empty or not a plain relative path")
    parts = PurePosixPath(relative).parts
    if PurePosixPath(relative).is_absolute() or any(part in ("", ".", "..") for part in parts):
        raise PinRefused("input name is absolute or traverses")


def declared(relative: str, registry: dict) -> tuple[int, str]:
    """The declared length and digest of a syntactically safe, registered name."""
    check_name(relative)
    if relative not in registry:
        raise PinRefused("input name is not registered")
    length, digest = registry[relative]
    if type(length) is not int or length <= 0 or not isinstance(digest, str) or len(digest) != 64:
        raise PinRefused("registry entry is malformed")
    return length, digest


def present(root: Path, relative: str) -> bool:
    """Whether a safe name exists under ``root``; it follows no symlink and reads no bytes."""
    check_name(relative)
    try:
        os.lstat(Path(root) / relative)
    except FileNotFoundError:
        return False
    return True


def read_pinned(root: Path, relative: str, registry: dict) -> bytes:
    """Return the authenticated bytes of one registered file under ``root``."""
    length, digest = declared(relative, registry)
    base = Path(os.path.abspath(root))
    if os.path.realpath(base) != str(base):
        raise PinRefused("input root is reached through a symlink")
    current = base
    mode = None
    for part in PurePosixPath(relative).parts:
        current = current / part
        mode = os.lstat(current).st_mode
        if stat.S_ISLNK(mode):
            raise PinRefused("input path has a symlink")
    if mode is None or not stat.S_ISREG(mode):
        raise PinRefused("input is not a regular file")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(current, flags)
    try:
        if os.fstat(descriptor).st_size != length:
            raise PinRefused("input length differs from its declaration")
        chunks, remaining = [], length + 1
        while remaining > 0:
            chunk = os.read(descriptor, min(remaining, 1 << 20))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
    finally:
        os.close(descriptor)
    body = b"".join(chunks)
    if len(body) != length or hashlib.sha256(body).hexdigest() != digest:
        raise PinRefused("input digest differs from its declaration")
    return body
