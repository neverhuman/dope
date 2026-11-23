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
"""
from __future__ import annotations

import ctypes
import json
import mmap
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import platform
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Sequence

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
THREAD_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
              "RAYON_NUM_THREADS")


class FactoryError(RuntimeError):
    """Static refusal of the factory. The message is a code, never a source value."""


class StageError(ValueError):
    """A stage could not process one episode. The message is a static code."""


class EpisodeFailed(ValueError):
    """The unit's payload is a counted failure; ``str(exc)`` is its static code."""


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
        if room.f_bavail * room.f_frsize < size + (256 << 20):
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

    def stop(self) -> None:
        self.head[H_STOP] = 1

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
                raise FactoryError("ring_stopped")
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


# -- producers ------------------------------------------------------------------------------
@dataclass(frozen=True)
class FactorySpec:
    """Everything a producer needs; plain data that pickles into a spawned process.

    ``corpus`` and ``sampler`` are ``"module:callable"`` paths called once per producer:
    ``corpus(*corpus_args)`` and ``sampler(corpus, *sampler_args, **sampler_kwargs)``. For a flat
    arena: ``arena_corpus`` (memory-maps it) and ``arena_sampler`` with the ``SamplerConfig``.
    """

    seed: int
    corpus: str
    sampler: str
    corpus_args: tuple[Any, ...] = ()
    sampler_args: tuple[Any, ...] = ()
    sampler_kwargs: dict[str, Any] = field(default_factory=dict)
    view: Any = None  # ViewConfig; None = the builder's default
    stage: str = ""  # "module:callable" (episode, **stage_kwargs) -> payload; "" = the view itself
    stage_kwargs: dict[str, Any] = field(default_factory=dict)
    max_tries: int = 64
    niceness: int = 0


def _producer(ring_name: str, k: int, spec: FactorySpec, cpu: int | None, end_unit: int | None,
              leader: int) -> None:
    if cpu is not None:
        os.sched_setaffinity(0, {cpu})
    if spec.niceness:
        os.nice(spec.niceness)
    from .s3_view_builder import DEFAULT_VIEW, build_slot

    ring = EpisodeRing.attach(ring_name)
    corpus = resolve(spec.corpus)(*spec.corpus_args)
    sampler = resolve(spec.sampler)(corpus, *spec.sampler_args, **spec.sampler_kwargs)
    stage = resolve(spec.stage) if spec.stage else None
    view = spec.view if spec.view is not None else DEFAULT_VIEW
    per_step, n_prod = ring.slots_per_step, ring.n_producers
    counters = ring.head[H_COUNTERS + 4 * k:H_COUNTERS + 4 * k + 4]

    def alive() -> bool:
        try:
            os.kill(leader, 0)
            return True
        except ProcessLookupError:
            return False

    plan_step, plan = -1, []
    unit = ring.unit0 + k
    clock = time.perf_counter_ns
    try:
        while not ring.stopped and (end_unit is None or unit < end_unit):
            step, slot = divmod(unit, per_step)
            t0 = clock()
            flags, stage_ns, where = 0, 0, "plan"
            try:
                if step != plan_step:
                    plan_step, plan = step, sampler.plan_step(spec.seed, step)
                    if len(plan) != per_step:
                        raise FactoryError("plan_size")
                where = "build"
                episode, used, reasons = build_slot(corpus, sampler, spec.seed, plan[slot], view,
                                                    max_tries=spec.max_tries)
                t1 = clock()
                where = "stage"
                out = stage(episode, **spec.stage_kwargs) if stage is not None else episode
                stage_ns = clock() - t1
                payload = pickle.dumps({"step": step, "slot": slot, "attempt": int(used.attempt),
                                        "retry_reasons": reasons, "episode": out}, protocol=5)
                if len(payload) > ring.capacity:
                    raise FactoryError("ring_capacity")
            except (ValueError, LookupError, ArithmeticError, RuntimeError) as exc:
                flags = FLAG_ERROR
                payload = pickle.dumps({"step": step, "slot": slot, "error": _fault(where, exc)}, protocol=5)
            build_ns = clock() - t0
            t2 = clock()
            if not ring.put(unit, payload, flags, build_ns, k, alive):
                break
            counters[0] += 1
            counters[1] += build_ns - stage_ns
            counters[2] += stage_ns
            counters[3] += clock() - t2
            unit += n_prod
    finally:
        del counters
        ring.close()


@contextmanager
def _single_thread_env() -> Iterator[None]:
    """Spawned producers inherit one-thread BLAS / OpenMP / rayon settings."""
    old = {key: os.environ.get(key) for key in THREAD_ENV}
    os.environ.update({key: "1" for key in THREAD_ENV})
    try:
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class EpisodeFactory:
    """Leader: owns the ring and the producer processes; reads payloads in any order.

    ``ahead_steps`` sets the window (``ahead_steps * batch_size`` slots, rounded up to a multiple
    of ``n_producers``). ``capacity`` is the largest pickled payload in bytes.
    """

    def __init__(self, spec: FactorySpec, batch_size: int, n_producers: int, capacity: int,
                 start_step: int = 0, ahead_steps: int = 2, cpus: Sequence[int] | None = None,
                 end_step: int | None = None, meta: dict[str, Any] | None = None) -> None:
        if ahead_steps < 1 or n_producers < 1:
            raise FactoryError("factory_shape")
        sweep_stale()
        window = ahead_steps * batch_size
        window = -(-window // n_producers) * n_producers
        info = {"seed": spec.seed, "stage": spec.stage, "batch_size": batch_size, **(meta or {})}
        self.ring = EpisodeRing.create(window, capacity, start_step * batch_size, batch_size,
                                       n_producers, info)
        inherited = sorted(os.sched_getaffinity(0))
        if cpus is not None and not set(cpus) <= set(inherited):
            self.ring.close()
            raise FactoryError("cpus_outside_inherited_mask")
        pool = list(cpus) if cpus is not None else inherited
        end_unit = None if end_step is None else end_step * batch_size
        ctx = mp.get_context("spawn")
        self.procs = []
        try:
            with _single_thread_env():
                for k in range(n_producers):
                    cpu = pool[k % len(pool)] if cpus is not None else None
                    proc = ctx.Process(target=_producer, name=f"dope-episode-{k}", daemon=True,
                                       args=(self.ring.name, k, spec, cpu, end_unit, os.getpid()))
                    proc.start()
                    self.procs.append(proc)
        except BaseException:
            self.close()
            raise
        self.t_open = time.perf_counter()

    @property
    def name(self) -> str:
        return self.ring.name

    def _alive(self) -> bool:
        """Some producer still runs and none has failed (a ready payload is read regardless)."""
        codes = [p.exitcode for p in self.procs]
        return None in codes and all(c in (None, 0) for c in codes)

    def get(self, step: int, slot: int, timeout: float | None = None) -> dict[str, Any]:
        return self.ring.get(step, slot, timeout, self._alive)

    def batch(self, step: int, timeout: float | None = None) -> list[dict[str, Any] | EpisodeFailed]:
        """Every slot of ``step`` in slot order; a failed unit appears as its ``EpisodeFailed``."""
        out: list[dict[str, Any] | EpisodeFailed] = []
        for slot in range(self.ring.slots_per_step):
            try:
                out.append(self.get(step, slot, timeout))
            except EpisodeFailed as exc:
                out.append(exc)
        return out

    def stats(self) -> dict[str, Any]:
        """Throughput and the consumer's data-wait share since the factory opened."""
        wall = max(time.perf_counter() - self.t_open, 1e-9)
        per = self.ring.counters()
        built = sum(c["built"] for c in per)
        busy = sum(c["build_s"] + c["stage_s"] for c in per)
        return {"producers": len(per), "built": built, "episodes_per_s": built / wall,
                "ms_per_episode": 1e3 * busy / built if built else None,
                "stage_share": sum(c["stage_s"] for c in per) / busy if busy else None,
                "consumer_wait_s": self.ring.wait_s, "consumer_wait_share": self.ring.wait_s / wall,
                "per_producer": per}

    def close(self) -> None:
        ring = getattr(self, "ring", None)
        if ring is None:
            return
        ring.stop()
        for proc in getattr(self, "procs", []):
            proc.join(timeout=10)
            if proc.exitcode is None:
                proc.terminate()
                proc.join(timeout=5)
        ring.close()
        self.ring = None

    def __enter__(self) -> "EpisodeFactory":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def arena_corpus(arena: str) -> Any:
    from .s3_view_corpus import ArenaCorpus

    return ArenaCorpus(arena)


def arena_sampler(corpus: Any, cfg: Any) -> Any:
    """The counter sampler of a ``SamplerConfig`` that names its arena (``corpus`` is unused)."""
    from .s3_counter_sampler import sampler_for

    return sampler_for(cfg)


# -- native DSE7 stage ----------------------------------------------------------------------
_TASK_ID = {"regression": 0, "binary": 1, "multiclass": 2}
_CODES = {1: "null_pointer", 2: "shape", 3: "fit", 4: "capacity", 5: "panic", 6: "unpack", 7: "target"}
_BLOCKS = ("col_feats", "q_nl", "glob", "knots", "loadings", "w_base", "b_base", "misc", "col_mask",
           "is_const", "col_index", "head")
ABI_VERSION = 1
_LIB: Any = None  # the loaded library handle (code, not data: nothing row-derived is kept)


def ffi_path() -> Path:
    """``DOPE_FFI_LIB`` or the workspace build (release before debug)."""
    env = os.environ.get("DOPE_FFI_LIB")
    if env:
        return Path(env)
    target = Path(__file__).resolve().parents[2] / "engine" / "target"
    for profile in ("release", "debug"):
        candidate = target / profile / "libdope_ffi.so"
        if candidate.exists():
            return candidate
    raise FactoryError("ffi_library_missing")


def load_ffi(path: str | Path | None = None) -> Any:
    global _LIB
    if _LIB is None:
        lib = ctypes.CDLL(str(path or ffi_path()))
        lib.dope_ffi_abi_version.restype = ctypes.c_uint32
        if lib.dope_ffi_abi_version() != ABI_VERSION:
            raise FactoryError("ffi_abi_version")
        lib.dope_ffi_dse7_len.restype = ctypes.c_uint64
        lib.dope_ffi_dse7_len.argtypes = [ctypes.c_uint32, ctypes.c_uint32]
        lib.dope_ffi_dse7_offset.restype = ctypes.c_uint64
        lib.dope_ffi_dse7_offset.argtypes = [ctypes.c_uint32]
        f32p, u8p = ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_uint8)
        lib.dope_ffi_dse7_episode.restype = ctypes.c_int32
        lib.dope_ffi_dse7_episode.argtypes = [
            f32p, ctypes.c_uint32, ctypes.c_uint32, f32p, f32p, ctypes.c_uint32, f32p, ctypes.c_uint8,
            ctypes.c_uint32, f32p, ctypes.c_uint64, u8p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32)]
        lib.offsets = {name: int(lib.dope_ffi_dse7_offset(i)) for i, name in enumerate(_BLOCKS)}
        _LIB = lib
    return _LIB


def _ptr(a: np.ndarray, kind: Any) -> Any:
    return a.ctypes.data_as(ctypes.POINTER(kind))


def dse7_arrays(x_fit: np.ndarray, y_fit: np.ndarray, x_qry: np.ndarray, y_qry: np.ndarray,
                task: str, budget: int = 2048) -> tuple[bytes, np.ndarray]:
    """(DSE7 packet bytes, flat float32 tensors) from FIT rows; query rows are only mapped."""
    lib = load_ffi()
    if task not in _TASK_ID:
        raise StageError("task_unsupported")
    xf = np.ascontiguousarray(x_fit, dtype=np.float32)
    xq = np.ascontiguousarray(x_qry, dtype=np.float32)
    yf = np.ascontiguousarray(y_fit, dtype=np.float32)
    yq = np.ascontiguousarray(y_qry, dtype=np.float32)
    n_fit, p = xf.shape
    n_qry = xq.shape[0]
    if xq.shape[1] != p or yf.shape != (n_fit,) or yq.shape != (n_qry,):
        raise StageError("shape")
    out = np.empty(int(lib.dope_ffi_dse7_len(n_fit, n_qry)), dtype=np.float32)
    cap = max(budget, 64)
    pkt = np.empty(cap, dtype=np.uint8)
    pkt_len = ctypes.c_uint32(0)
    code = lib.dope_ffi_dse7_episode(
        _ptr(xf, ctypes.c_float), n_fit, p, _ptr(yf, ctypes.c_float), _ptr(xq, ctypes.c_float), n_qry,
        _ptr(yq, ctypes.c_float), _TASK_ID[task], budget, _ptr(out, ctypes.c_float), out.size,
        _ptr(pkt, ctypes.c_uint8), cap, ctypes.byref(pkt_len))
    if code != 0:
        raise StageError("ffi_" + _CODES.get(int(code), "unknown"))
    return pkt[:pkt_len.value].tobytes(), out


def native_dse7(episode: dict[str, Any], budget: int = 2048) -> dict[str, Any]:
    """Stage: the DSE7 packet of this episode's FIT rows plus FIT and query rows in its coordinates.

    Wants a view built with ``ViewConfig(emit="raw")`` so the Rust scale card sees raw values, as
    serving does. Multiclass FIT classes beyond the packet's eight most frequent get code -1.
    """
    packet, floats = dse7_arrays(episode["X_fit"], episode["y_fit"], episode["X_qry"], episode["y_qry"],
                                 episode["task"], budget)
    keep = ("dataset_hash", "family_stem", "source_kind", "task", "n_classes", "split_sig", "view_sig",
            "kind", "table_id", "step", "slot", "attempt", "target_col", "aug_stats")
    out = {key: episode[key] for key in keep}
    out.update(packet=packet, floats=floats, n_fit=int(episode["y_fit"].shape[0]),
               n_qry=int(episode["y_qry"].shape[0]), col_source=np.asarray(episode["col_source"]),
               distractor_mask=np.asarray(episode["distractor_mask"]))
    return out


def split_native(payload: dict[str, Any]) -> dict[str, np.ndarray]:
    """Named float32 views of a ``native_dse7`` payload (no copy)."""
    offsets = load_ffi().offsets
    f, n_fit, n_qry = payload["floats"], payload["n_fit"], payload["n_qry"]
    cols = offsets["is_const"] - offsets["col_mask"]  # MAX_COLS
    out = {}
    for name, nxt in zip(_BLOCKS[:-1], _BLOCKS[1:]):
        at, size = offsets[name], offsets[nxt] - offsets[name]
        out[name] = f[at:at + size].reshape(cols, -1) if size % cols == 0 and size > cols else f[at:at + size]
    out["w_base"] = out["w_base"].reshape(-1, cols)  # [class][column]
    at = offsets["head"]
    for name, n in (("fit", n_fit), ("qry", n_qry)):
        out["x_" + name] = f[at:at + n * cols].reshape(n, cols)
        at += n * cols
        out["y_" + name] = f[at:at + n]
        at += n
    if at != f.size:
        raise FactoryError("native_layout")
    return out


def native_capacity(max_fit: int, max_qry: int, budget: int = 2048) -> int:
    """Bytes of the largest pickled ``native_dse7`` payload, with headroom for metadata."""
    head = 48 * 42 + 48 * 4 + 16 + 48 * 17 + 48 * 12 + 8 * 48 + 8 + 8 + 3 * 48
    return 4 * (head + (max_fit + max_qry) * 49) + budget + 16 * (max_fit + max_qry) + (256 << 10)


# -- bench ----------------------------------------------------------------------------------
def bench(arena: str, producers: Sequence[int], steps: int, batch_size: int, seed: int = 7,
          budget: int = 2048, cpus: Sequence[int] | None = None, native: bool = True) -> list[dict[str, Any]]:
    """Episodes/s and ms per episode on a real arena, for each producer count (fixed plan)."""
    from .s3_counter_sampler import SamplerConfig
    from .s3_view_builder import ViewConfig

    cfg = SamplerConfig(arena=arena, batch_size=batch_size, multilabel_view_prob=0.0 if native else 0.12)
    spec = FactorySpec(seed=seed, corpus=f"{__name__}:arena_corpus", corpus_args=(arena,),
                       sampler=f"{__name__}:arena_sampler", sampler_args=(cfg,),
                       view=ViewConfig(emit="raw") if native else None,
                       stage=f"{__name__}:native_dse7" if native else "", stage_kwargs={"budget": budget} if native else {})
    cap = native_capacity(max(h for _, _, _, h in cfg.fit_law), cfg.qry_hi, budget) if native else 32 << 20
    rows = []
    for n_prod in producers:
        with EpisodeFactory(spec, batch_size, n_prod, cap, ahead_steps=2, cpus=cpus, end_step=steps) as fac:
            first = fac.batch(0, timeout=600)  # start-up (spawn, imports, index) is not throughput
            t0, base = time.perf_counter(), fac.stats()["built"]
            failed = sum(isinstance(e, EpisodeFailed) for e in first)
            bytes_ = [len(e["episode"]["packet"]) for e in first if not isinstance(e, EpisodeFailed) and native]
            for step in range(1, steps):
                items = fac.batch(step, timeout=600)
                failed += sum(isinstance(e, EpisodeFailed) for e in items)
                bytes_ += [len(e["episode"]["packet"]) for e in items if not isinstance(e, EpisodeFailed) and native]
            wall = time.perf_counter() - t0
            st = fac.stats()
        n = (steps - 1) * batch_size
        rows.append({"producers": n_prod, "episodes": n, "wall_s": round(wall, 3),
                     "episodes_per_s": round(n / wall, 2) if wall > 0 else None,
                     "ms_per_episode": round(st["ms_per_episode"], 2), "stage_share": round(st["stage_share"] or 0, 3),
                     "failed": failed, "packet_bytes_median": int(np.median(bytes_)) if bytes_ else None,
                     "packet_bytes_max": max(bytes_) if bytes_ else None, "built_after_start": st["built"] - base})
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="python -m research.levels.s3_episode_factory")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bench", help="throughput of the factory on a flat arena")
    b.add_argument("--arena", required=True)
    b.add_argument("--producers", default="1,2")
    b.add_argument("--steps", type=int, default=4)
    b.add_argument("--batch-size", type=int, default=64)
    b.add_argument("--budget", type=int, default=2048)
    b.add_argument("--cpus", default="", help="comma list; default: the inherited affinity mask")
    b.add_argument("--view-only", action="store_true", help="no native stage (the scaled view itself)")
    args = ap.parse_args(argv)
    cpus = [int(c) for c in args.cpus.split(",") if c] or None
    rows = bench(args.arena, [int(p) for p in args.producers.split(",")], args.steps, args.batch_size,
                 budget=args.budget, cpus=cpus, native=not args.view_only)
    for row in rows:
        print(json.dumps(row, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
