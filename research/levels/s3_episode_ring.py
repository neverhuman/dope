"""Parallel episode factory: fresh random split, FIT-only features, never cached (plan v3, package C6).

Every training episode gets a fresh random train/test split, and every feature, metric and packet
statistic is computed from that split's FIT rows only. Nothing derived from rows can therefore be
cached per table. Only the stored tables, their declared schema and dup-group ids are reused.
The factory computes these features ahead of the trainer on a pool of CPU producer processes, so
training batches are never starved.

Work unit ``u = step * batch_size + slot``. Its payload is a pure function of ``(seed, step,
slot)``: the counter sampler's plan, ``build_slot``'s group-respecting random split and view, then
a stage (for example ``native_dse7``, the DSE7 packet fitted in Rust from the FIT rows). Producer
``k`` of ``P`` owns ring slots ``j`` with ``j % P == k`` and builds their units in increasing
order, so the bytes of every payload are the same for any number of producers.

Ring: one file ``/dev/shm/dope-ring.<leader pid>.<token>``, 0600, attachable by name from any
process (a gradient-farm worker, say):

    page 0         int64 words: magic, version, n_slots, capacity, stride, unit0, stop,
                   slots_per_step, n_producers; producer counters from word 64; JSON meta at 2048.
    slot j         at 4096 + j * stride: int64 header [state, next_unit, unit, nbytes, flags,
                   build_ns, producer, 0], then ``capacity`` payload bytes.

Hand-off: the producer of unit ``u`` waits until slot ``(u - unit0) % n_slots`` is EMPTY and
expects ``u``, writes the payload, then sets READY. The consumer waits until the slot is READY
with unit ``u``, copies the payload out, sets ``next_unit = u + n_slots``, then EMPTY. Each unit
must be consumed exactly once, in any order within the window of ``n_slots`` units; a window of
two or more steps never deadlocks a step-synchronous trainer. Ordering relies on x86-64 total
store order (payload stores are visible before the state store), so the ring refuses other
machines.

A failed episode (no view after ``max_tries`` re-plans, a stage fault, a payload over capacity)
is a READY payload carrying a static error code: counted, never skipped, never a source value.

Producers are spawned, so they re-import the leader's ``__main__``: run the trainer as a module
(``python -m ...``) or a script whose work sits under ``if __name__ == "__main__":``, never from
stdin. A producer that cannot start is reported as ``producer_died``.

For v3 training use ``v3_spec(arena, seed, batch_size)``: the counter sampler with
``V3_SAMPLER``, the raw-emit view and the native DSE7 stage.
"""
from __future__ import annotations

import json
import mmap
import os
from pathlib import Path
import pickle
import platform
import time
from typing import Any, Callable

import numpy as np

SHM_DIR = Path("/dev/shm")
PREFIX = "dope-ring"
MAGIC = 0x31474E5245504F44  # "DOPERNG1" little-endian
VERSION = 1
PAGE = 4096
SLOT_HEAD = 64  # bytes: 8 int64 words
EMPTY, READY = 0, 1
FLAG_ERROR = 1
H_MAGIC, H_VERSION, H_SLOTS, H_CAP, H_STRIDE, H_UNIT0, H_STOP, H_STEP, H_PRODUCERS = range(9)
H_COUNTERS = 64  # producer k: words 64 + 4k .. 64 + 4k + 3 = built, build_ns, stage_ns, wait_ns
MAX_PRODUCERS = 64
META_AT = 2048


class FactoryError(RuntimeError):
    """Static refusal of the factory. The message is a code, never a source value."""


class StageError(ValueError):
    """A stage could not process one episode. The message is a static code."""


class EpisodeFailed(ValueError):
    """The unit's payload is a counted failure; ``str(exc)`` is its static code."""


class RingDown(OSError):
    """The ring stopped, timed out or lost a producer: a failed step, never a counted episode.

    An ``OSError`` so a gradient-farm worker reports it to its leader (``COMMAND_FAULTS``) instead
    of counting it as an episode fault (``EPISODE_FAULTS``).
    """


def resolve(path: str) -> Callable[..., Any]:
    """``"package.module:attr"`` -> callable (how stages cross the process boundary)."""
    import importlib

    module, sep, attr = str(path).partition(":")
    if not sep or not module or not attr:
        raise FactoryError("import_path")
    obj: Any = importlib.import_module(module)
    for part in attr.split("."):
        obj = getattr(obj, part)
    if not callable(obj):
        raise FactoryError("import_path_not_callable")
    return obj


def _fault(stage: str, exc: BaseException) -> str:
    """Static code of a fault: stage and exception type, plus the message when it is a code."""
    msg = str(exc)
    code = msg if isinstance(exc, (StageError, FactoryError)) or type(exc).__name__ == "ViewError" else ""
    if code and not code.replace("_", "").isalnum():
        code = ""
    return f"{stage}:{type(exc).__name__}" + (f":{code}" if code else "")


# -- ring -----------------------------------------------------------------------------------
class EpisodeRing:
    """The shared ring: created by the leader, attached by name by producers and consumers."""

    def __init__(self, name: str, mm: mmap.mmap, owner: bool) -> None:
        self.name, self.mm, self.owner = name, mm, owner
        self.head = np.frombuffer(mm, dtype=np.int64, count=PAGE // 8)
        if int(self.head[H_MAGIC]) != MAGIC or int(self.head[H_VERSION]) != VERSION:
            raise FactoryError("ring_magic")
        self.n_slots, self.capacity = int(self.head[H_SLOTS]), int(self.head[H_CAP])
        self.stride, self.unit0 = int(self.head[H_STRIDE]), int(self.head[H_UNIT0])
        self.slots_per_step = int(self.head[H_STEP])
        self.n_producers = int(self.head[H_PRODUCERS])
        raw = bytes(mm[META_AT:PAGE]).split(b"\0", 1)[0]
        self.meta = json.loads(raw.decode()) if raw else {}
        self.wait_s = 0.0  # this process's time blocked in ``get``
        self.got = 0

    @staticmethod
    def _check_machine() -> None:
        if platform.machine() not in ("x86_64", "AMD64"):
            raise FactoryError("ring_needs_x86_64")

    @classmethod
    def create(cls, n_slots: int, capacity: int, unit0: int, slots_per_step: int, n_producers: int,
               meta: dict[str, Any] | None = None) -> "EpisodeRing":
        cls._check_machine()
        if n_slots < 1 or capacity < 1 or slots_per_step < 1 or not 1 <= n_producers <= MAX_PRODUCERS:
            raise FactoryError("ring_shape")
        if n_slots % n_producers:
            raise FactoryError("ring_slots_not_multiple_of_producers")
        stride = -(-(SLOT_HEAD + capacity) // PAGE) * PAGE
        size = PAGE + n_slots * stride
        room = os.statvfs(SHM_DIR)
        if room.f_bavail * room.f_frsize < size + (64 << 20):
            raise FactoryError("shm_no_space")
        blob = json.dumps(meta or {}, sort_keys=True).encode()
        if len(blob) >= PAGE - META_AT:
            raise FactoryError("ring_meta_too_large")
        name = f"{PREFIX}.{os.getpid()}.{os.urandom(4).hex()}"
        fd = os.open(SHM_DIR / name, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
        try:
            os.ftruncate(fd, size)  # sparse: a slot's pages exist once a payload is written
            mm = mmap.mmap(fd, size)
        finally:
            os.close(fd)
        head = np.frombuffer(mm, dtype=np.int64, count=PAGE // 8)
        head[[H_SLOTS, H_CAP, H_STRIDE, H_UNIT0, H_STOP, H_STEP, H_PRODUCERS]] = (
            n_slots, capacity, stride, unit0, 0, slots_per_step, n_producers)
        mm[META_AT:META_AT + len(blob)] = blob
        for j in range(n_slots):
            sh = np.frombuffer(mm, dtype=np.int64, count=8, offset=PAGE + j * stride)
            sh[1] = unit0 + j  # next_unit
        head[H_VERSION] = VERSION
        head[H_MAGIC] = MAGIC  # last: a reader never sees a half-written header
        del head, sh
        return cls(name, mm, owner=True)

    @classmethod
    def attach(cls, name: str) -> "EpisodeRing":
        cls._check_machine()
        if "/" in name or not name.startswith(PREFIX + "."):
            raise FactoryError("ring_name")
        fd = os.open(SHM_DIR / name, os.O_RDWR)
        try:
            mm = mmap.mmap(fd, os.fstat(fd).st_size)
        finally:
            os.close(fd)
        return cls(name, mm, owner=False)

    # slot access
    def _slot(self, unit: int) -> int:
        if unit < self.unit0:
            raise FactoryError("unit_before_start")
        return (unit - self.unit0) % self.n_slots

    def _hdr(self, j: int) -> np.ndarray:
        return np.frombuffer(self.mm, dtype=np.int64, count=8, offset=PAGE + j * self.stride)

    @property
    def stopped(self) -> bool:
        return bool(self.head[H_STOP])

    def stop(self, failed: bool = False) -> None:
        self.head[H_STOP] = 2 if failed else 1

    def put(self, unit: int, payload: bytes, flags: int, build_ns: int, producer: int,
            alive: Callable[[], bool] = lambda: True) -> bool:
        """Producer side. Blocks until the slot expects ``unit``; False when the ring stopped."""
        j = self._slot(unit)
        hdr = self._hdr(j)
        nap = 1e-4
        while not (int(hdr[0]) == EMPTY and int(hdr[1]) == unit):
            if self.stopped or not alive():
                return False
            time.sleep(nap)
            nap = min(nap * 2, 2e-3)
        if len(payload) > self.capacity:
            raise FactoryError("ring_capacity")
        at = PAGE + j * self.stride + SLOT_HEAD
        self.mm[at:at + len(payload)] = payload
        hdr[2], hdr[3], hdr[4], hdr[5], hdr[6] = unit, len(payload), flags, build_ns, producer
        hdr[0] = READY  # last (x86-64 TSO: payload and fields are visible first)
        return True

    def get_raw(self, unit: int, timeout: float | None = None,
                alive: Callable[[], bool] = lambda: True) -> tuple[bytes, int]:
        """Consumer side: (payload bytes, flags) of ``unit``; frees its slot."""
        j = self._slot(unit)
        hdr = self._hdr(j)
        t0 = time.perf_counter()
        nap, deadline = 5e-5, None if timeout is None else t0 + timeout
        while not (int(hdr[0]) == READY and int(hdr[2]) == unit):
            state = int(hdr[0])
            if (state == EMPTY and int(hdr[1]) > unit) or (state == READY and int(hdr[2]) > unit):
                raise FactoryError("unit_already_consumed")
            if self.stopped:
                raise FactoryError("producer_failed" if int(self.head[H_STOP]) == 2 else "ring_stopped")
            if not alive():
                raise FactoryError("producer_died")
            if deadline is not None and time.perf_counter() > deadline:
                raise FactoryError("ring_timeout")
            time.sleep(nap)
            nap = min(nap * 2, 1e-3)
        at = PAGE + j * self.stride + SLOT_HEAD
        data, flags = bytes(self.mm[at:at + int(hdr[3])]), int(hdr[4])
        hdr[1] = unit + self.n_slots
        hdr[0] = EMPTY
        self.wait_s += time.perf_counter() - t0
        self.got += 1
        return data, flags

    def get(self, step: int, slot: int, timeout: float | None = None,
            alive: Callable[[], bool] = lambda: True) -> dict[str, Any]:
        """The payload of ``(step, slot)``; raises ``EpisodeFailed(code)`` for a counted failure."""
        if not 0 <= slot < self.slots_per_step:
            raise FactoryError("slot_range")
        data, flags = self.get_raw(step * self.slots_per_step + slot, timeout, alive)
        payload = pickle.loads(data)
        if flags & FLAG_ERROR:
            raise EpisodeFailed(payload["error"])
        return payload

    def counters(self) -> list[dict[str, float]]:
        """Per producer: episodes built and seconds spent building, in the stage and waiting."""
        out = []
        for k in range(self.n_producers):
            w = self.head[H_COUNTERS + 4 * k:H_COUNTERS + 4 * k + 4]
            out.append({"built": int(w[0]), "build_s": w[1] / 1e9, "stage_s": w[2] / 1e9,
                        "wait_s": w[3] / 1e9})
        return out

    def close(self) -> None:
        self.head = None  # drop the exported buffer before unmapping
        try:
            self.mm.close()
        except BufferError:  # a caller still holds a view; the mapping dies with the process
            pass
        if self.owner:
            try:
                os.unlink(SHM_DIR / self.name)
            except FileNotFoundError:
                pass


def sweep_stale() -> list[str]:
    """Remove ring files whose leader process is gone. Returns the names removed."""
    gone = []
    for path in SHM_DIR.glob(PREFIX + ".*"):
        try:
            pid = int(path.name.split(".")[1])
        except (IndexError, ValueError):
            continue
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            path.unlink(missing_ok=True)
            gone.append(path.name)
        except PermissionError:
            pass
    return gone
