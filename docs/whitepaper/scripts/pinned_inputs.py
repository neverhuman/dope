"""Read one registered paper input: exactly the bytes whose length and digest are declared in advance.

A Registry binds one declared root to the files admitted under it, each a
relative path with its byte length and sha256. Each paper script declares its
registries next to their roots; a read names a registry and a relative path,
and the path can never leave that registry's root. A read refuses, before any
byte is decoded: a name absent from the registry; an empty, absolute, or
traversing name; a symlink at any component from the filesystem root to the
leaf; a leaf that is not a regular file; and a length or digest that differs.
The walk holds each directory open and opens the next component relative to it
without following links, so no component can be swapped between the check and
the read. The presence probe takes the same walk, so a redirected ancestor is
refused rather than reported absent. The body is read once, bounded by its
declared length, and the caller parses those same bytes. Error messages name
the rule, never file contents.
"""

from __future__ import annotations

import errno
import hashlib
import os
import stat
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
DIRECTORY = getattr(os, "O_DIRECTORY", 0)


class PinRefused(ValueError):
    """A registered-input refusal; the message is a fixed rule name."""


@dataclass(frozen=True)
class Registry:
    """A declared root and the files admitted under it: relative path -> (bytes, sha256)."""

    root: Path
    files: dict = field(default_factory=dict)


def check_name(relative: str) -> None:
    """Refuse an empty, absolute, backslashed, or traversing name before any filesystem call."""
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise PinRefused("input name is empty or not a plain relative path")
    parts = PurePosixPath(relative).parts
    if PurePosixPath(relative).is_absolute() or any(part in ("", ".", "..") for part in parts):
        raise PinRefused("input name is absolute or traverses")


def declared(registry: Registry, relative: str) -> tuple[int, str]:
    """The declared length and digest of a syntactically safe, registered name."""
    check_name(relative)
    if relative not in registry.files:
        raise PinRefused("input name is not registered")
    length, digest = registry.files[relative]
    if type(length) is not int or length <= 0 or not isinstance(digest, str) or len(digest) != 64:
        raise PinRefused("registry entry is malformed")
    return length, digest


def _components(registry: Registry, relative: str) -> list[str]:
    base = PurePosixPath(os.path.abspath(registry.root))
    return [*base.parts[1:], *PurePosixPath(relative).parts]


def _open_component(part: str, flags: int, parent: int) -> int:
    """Open one component beneath a held directory without following a link."""
    try:
        return os.open(part, flags | NOFOLLOW, dir_fd=parent)
    except OSError as error:
        if error.errno == errno.ENOENT:
            raise FileNotFoundError(errno.ENOENT, "registered input is absent") from None
        if error.errno in (errno.ELOOP, errno.ENOTDIR, errno.EMLINK):
            raise PinRefused("input path has a symlink or a non-directory ancestor") from None
        raise


def _hold_parent(components: list[str]) -> int:
    """Open every directory above the leaf from "/", holding each and following no link; return the parent."""
    held = [os.open("/", os.O_RDONLY | DIRECTORY)]
    try:
        for part in components[:-1]:
            held.append(_open_component(part, os.O_RDONLY | DIRECTORY, held[-1]))
        return held.pop()
    finally:
        for descriptor in held:
            os.close(descriptor)


def present(registry: Registry, relative: str) -> bool:
    """Whether a safe name exists beneath real directories under the registry's root; it reads no bytes.

    A missing leaf or ancestor is absent. A symlinked or non-directory ancestor
    is refused, never reported absent. A leaf that is itself a symlink counts as
    present, so the read that follows refuses it.
    """
    check_name(relative)
    components = _components(registry, relative)
    try:
        parent = _hold_parent(components)
    except FileNotFoundError:
        return False
    try:
        os.stat(components[-1], dir_fd=parent, follow_symlinks=False)
    except FileNotFoundError:
        return False
    finally:
        os.close(parent)
    return True


def _open_beneath(components: list[str]) -> int:
    """Open the last component from "/" holding every directory, following no link; return its descriptor."""
    parent = _hold_parent(components)
    try:
        # A FIFO leaf must not block the open; fstat then refuses it as not regular.
        return _open_component(components[-1], os.O_RDONLY | os.O_NONBLOCK, parent)
    finally:
        os.close(parent)


def read_pinned(registry: Registry, relative: str) -> bytes:
    """Return the authenticated bytes of one registered file under the registry's root."""
    length, digest = declared(registry, relative)
    descriptor = _open_beneath(_components(registry, relative))
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise PinRefused("input is not a regular file")
        if info.st_size != length:
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
