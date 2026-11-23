"""Shared-memory layout of the S3 gradient farm (plan M3, stage G1).

Two float32 segments live in ``/dev/shm`` while a farm is open, both named
``dope-farm.<leader pid>.<token>.<kind>``:

``params``  one vector of ``n_shared`` floats. The first ``dim`` entries are the trainable
            parameters in ``named_parameters()`` order (the gradient dimension D); frozen
            parameters and float32 buffers follow. Every process re-points its model's tensors
            at slices of this vector, so an in-place update on the leader is what every worker
            reads at its next step. Nothing is copied or broadcast.
``slab``    ``n_workers`` rows of ``stride`` floats. Row ``w`` belongs to worker ``w``:
            ``[0, D)`` is the SUM of its valid per-episode gradients, ``[D, D + S)`` its scalar
            block (counts, loss sum, squared norms, phase times, diagnostic sums). ``stride`` is
            ``D + S`` rounded up to 4 KiB so rows never share a page or a cache line.

The segments are sparse files: a page exists once somebody writes it, so a slab row is first
touched by the worker that owns it. Cost: 4 * (n_shared + n_workers * stride) bytes of tmpfs,
i.e. 64 MiB per worker at 16M parameters.

Leak rules. The leader unlinks both names on close, on an exception and on SIGTERM. A worker
that loses its leader (SIGKILL, OOM) unlinks them itself before exiting. ``sweep_stale`` removes
segments whose leader pid is gone; a farm calls it on start.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import json
import mmap
import os
from pathlib import Path
import pickle
from typing import Any, Callable, Sequence

import numpy as np
import torch

from .s3_sampler_index import key64

SHM_DIR = Path("/dev/shm")
PREFIX = "dope-farm"
ROW_ALIGN = 1024  # floats; rows start on 4 KiB boundaries
REDUCE_CHUNK = 32768  # floats per reduce block: accumulator + one incoming row block stay in L2
# Scalar block of a slab row. Append only: the order is part of the layout.
THREAD_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
# glibc malloc in a worker: serve tensors up to 32 MiB from the heap and never give the heap back.
# By default every large tensor is mmap'ed and unmapped again, i.e. one page fault per 4 KiB per
# episode: 28,000 faults and 124 ms of system time per episode at a 16M-float gradient (measured).
MALLOC_ENV = {"MALLOC_MMAP_THRESHOLD_": str(32 << 20), "MALLOC_TRIM_THRESHOLD_": str(16 << 30),
              "MALLOC_TOP_PAD_": str(64 << 20)}
SCALARS = ("n_planned", "n_valid", "n_fail", "n_retry", "n_nonfinite", "loss_sum", "sq_norm", "ep_sq_norm",
           "diag_dropped", "t_plan", "t_build", "t_forward", "t_backward", "t_reduce", "t_total", "t_cpu")


class FarmError(RuntimeError):
    """Farm refusal or failure. The message is a code or a process fact, never a data value."""


class _Retry:
    __slots__ = ()

    def __repr__(self) -> str:
        return "RETRY"


RETRY = _Retry()  # episode_fn returns this when a view cannot be built; counted, never an error


def resolve(path: str) -> Callable[..., Any]:
    """``"package.module:attr"`` -> callable. The only way code crosses the process boundary."""
    module, sep, attr = str(path).partition(":")
    if not sep or not module or not attr:
        raise FarmError("import_path")
    obj: Any = importlib.import_module(module)
    for part in attr.split("."):
        obj = getattr(obj, part)
    if not callable(obj):
        raise FarmError("import_path_not_callable")
    return obj


def episode_seed(seed: int, step: int, slot: int) -> int:
    """Seed of the RNG streams of one episode: a function of (seed, step, slot) only."""
    return key64("farm-episode", seed, step, slot) & 0x7FFF_FFFF_FFFF_FFFF


def fault_code(stage: str, exc: BaseException) -> str:
    """Short reason code of a failed episode: stage, exception type, innermost source line."""
    tb, where = exc.__traceback__, ""
    while tb is not None:
        where = f"@{os.path.basename(tb.tb_frame.f_code.co_filename)}:{tb.tb_lineno}"
        tb = tb.tb_next
    return f"{stage}:{type(exc).__name__}{where}"


# Exceptions that fail one episode (counted by `fault_code`) instead of the step: data-dependent
# faults. Torch raises RuntimeError and its subclasses; bad shapes or values raise ValueError,
# LookupError or ArithmeticError. Anything else, such as a TypeError from a coding mistake, is not
# an episode fault and reaches the leader as a worker error.
EPISODE_FAULTS: tuple[type[BaseException], ...] = (RuntimeError, ValueError, LookupError, ArithmeticError)

# Exceptions a worker reports to the leader as ("err", code) and survives: episode and farm faults
# (FarmError is a RuntimeError), I/O, a module or name that does not resolve, and the type and
# pickling errors of a bad command or answer. Any other exception ends the worker; the leader sees
# its socket close and raises FarmError with the worker's exit status.
COMMAND_FAULTS: tuple[type[BaseException], ...] = EPISODE_FAULTS + (
    OSError, ImportError, TypeError, AttributeError, pickle.PicklingError)


# -- CPUs -----------------------------------------------------------------------------------
def cpu_topology(cpu: int) -> tuple[str, tuple[int, int]]:
    """(L3 cache domain, physical core) of a logical CPU; unknown topology is one domain, own core."""
    base = Path(f"/sys/devices/system/cpu/cpu{cpu}")
    try:
        core = (int((base / "topology/physical_package_id").read_text()), int((base / "topology/core_id").read_text()))
    except (OSError, ValueError):
        core = (0, cpu)
    try:
        return (base / "cache/index3/shared_cpu_list").read_text().strip(), core
    except OSError:
        return "", core


def pick_cpus(n_workers: int, explicit: Sequence[int] | None = None) -> list[int]:
    """One logical CPU per worker, taken from the inherited affinity mask and never outside it.

    Default order: spread over L3 cache domains first, then over the physical cores of a domain,
    SMT siblings last. Few workers therefore get a cache domain each, and two workers share a
    core only when there are more workers than cores. More workers than CPUs wraps round.
    """
    inherited = sorted(os.sched_getaffinity(0))
    if explicit is not None:
        cpus = [int(c) for c in explicit]
        if not cpus or not set(cpus) <= set(inherited):
            raise FarmError("cpus_outside_inherited_mask")
    else:
        found: dict[str, dict[tuple[int, int], list[int]]] = {}
        for cpu in inherited:
            domain, core = cpu_topology(cpu)
            found.setdefault(domain, {}).setdefault(core, []).append(cpu)
        domains = sorted((sorted(cores.values()) for cores in found.values()), key=lambda cores: cores[0][0])
        deepest, widest = max(len(c) for d in domains for c in d), max(len(d) for d in domains)
        cpus = [d[j][k] for k in range(deepest) for j in range(widest) for d in domains
                if j < len(d) and k < len(d[j])]
    return [cpus[w % len(cpus)] for w in range(n_workers)]


# -- segments -------------------------------------------------------------------------------
def new_prefix() -> str:
    return f"{PREFIX}.{os.getpid()}.{os.urandom(4).hex()}"


def _map(fd: int, n_floats: int) -> np.ndarray:
    return np.frombuffer(mmap.mmap(fd, n_floats * 4), dtype=np.float32)


def require_shm_room(n_bytes: int) -> None:
    """Refuse up front when tmpfs cannot hold a farm (a full tmpfs would SIGBUS a worker later)."""
    room = os.statvfs(SHM_DIR)
    if room.f_bavail * room.f_frsize < n_bytes + (64 << 20):
        raise FarmError("shm_no_space")


def create_segment(name: str, n_floats: int) -> np.ndarray:
    """New zero segment (sparse; no page is touched here)."""
    fd = os.open(SHM_DIR / name, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
    try:
        os.ftruncate(fd, n_floats * 4)
        return _map(fd, n_floats)
    finally:
        os.close(fd)


def attach_segment(name: str, n_floats: int) -> np.ndarray:
    fd = os.open(SHM_DIR / name, os.O_RDWR)
    try:
        if os.fstat(fd).st_size != n_floats * 4:
            raise FarmError("segment_size")
        return _map(fd, n_floats)
    finally:
        os.close(fd)


def unlink_segment(name: str) -> None:
    try:
        os.unlink(SHM_DIR / name)
    except FileNotFoundError:
        pass


def segments(prefix: str = PREFIX) -> list[str]:
    """Names in /dev/shm that start with ``prefix`` (a farm's own prefix, or every farm)."""
    return sorted(p.name for p in SHM_DIR.glob(prefix + "*"))


def sweep_stale() -> list[str]:
    """Unlink farm segments whose leader process no longer exists. Returns the removed names."""
    removed = []
    for name in segments(PREFIX + "."):
        try:
            os.kill(int(name.split(".")[1]), 0)
        except ProcessLookupError:
            try:
                unlink_segment(name)
                removed.append(name)
            except OSError:
                pass
        except (ValueError, IndexError, PermissionError):
            pass
    return removed


# -- model <-> flat vector ------------------------------------------------------------------
@dataclass(frozen=True)
class Layout:
    """Where each shared tensor of a model sits in the ``params`` vector."""

    entries: tuple[tuple[str, str, int, int, tuple[int, ...]], ...]  # name, kind, offset, numel, shape
    dim: int  # D: trainable entries, the first block
    n_shared: int  # D + frozen parameters + float32 buffers
    digest: str  # names, kinds, shapes: must be equal in every process
    private: str  # digest of the tensors that stay private (non-float32 buffers): must be equal too


def model_layout(model: torch.nn.Module) -> Layout:
    """Trainable float32 parameters first (named_parameters order), then frozen ones and buffers."""
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm) and module.track_running_stats:
            raise FarmError("batchnorm_running_stats")  # a forward pass must not write shared state
    train, static = [], []
    for name, p in model.named_parameters():
        if p.dtype != torch.float32 or p.device.type != "cpu":
            raise FarmError("parameter_not_cpu_float32")
        (train if p.requires_grad else static).append((name, "param" if p.requires_grad else "frozen", p))
    private = hashlib.blake2b(digest_size=16, person=b"dope-farm-priv")
    for name, b in model.named_buffers():
        if b.dtype == torch.float32:
            static.append((name, "buffer", b))
        else:
            private.update(f"{name}:{b.dtype}:{tuple(b.shape)}".encode())
            private.update(np.ascontiguousarray(b.detach().cpu().numpy()).tobytes())
    entries, offset, dim = [], 0, 0
    for name, kind, t in train + static:
        entries.append((name, kind, offset, t.numel(), tuple(t.shape)))
        offset += t.numel()
        dim += t.numel() if kind == "param" else 0
    if dim < 1:
        raise FarmError("no_trainable_parameters")
    shape = json.dumps([(n, k, list(s)) for n, k, _, _, s in entries]).encode()
    return Layout(tuple(entries), dim, offset, hashlib.blake2b(shape, digest_size=16).hexdigest(), private.hexdigest())


def bind_model(model: torch.nn.Module, layout: Layout, flat: torch.Tensor, *,
               copy_in: bool) -> list[torch.nn.Parameter]:
    """Re-point every shared tensor of ``model`` at its slice of ``flat``.

    ``copy_in`` first writes the model's current values into ``flat`` (leader); a worker binds
    without copying and thereby adopts the leader's values. Returns the trainable parameters in
    layout order.
    """
    params, trainable = dict(model.named_parameters()), []
    with torch.no_grad():
        for name, kind, off, n, shape in layout.entries:
            view = flat[off:off + n].view(shape)
            if kind == "buffer":
                owner, _, leaf = name.rpartition(".")
                module = model.get_submodule(owner)
                if copy_in:
                    view.copy_(module._buffers[leaf])
                module._buffers[leaf] = view
            else:
                if copy_in:
                    view.copy_(params[name])
                params[name].data = view
                if kind == "param":
                    trainable.append(params[name])
    return trainable


def checksum_model(model: torch.nn.Module, layout: Layout) -> str:
    """Digest of the shared tensors as THIS process's model holds them (not of the segment).

    A model whose tensors were re-bound to private memory therefore disagrees with the others.
    """
    tensors = {**dict(model.named_buffers()), **dict(model.named_parameters())}
    h = hashlib.blake2b(digest_size=16, person=b"dope-farm-sum")
    for name, _, _, _, _ in layout.entries:
        h.update(memoryview(np.ascontiguousarray(tensors[name].detach().numpy()).reshape(-1)))
    return h.hexdigest()


# -- slab -----------------------------------------------------------------------------------
@dataclass(frozen=True)
class Slab:
    n_workers: int
    dim: int
    scalars: tuple[str, ...]

    @property
    def n_scalars(self) -> int:
        return len(self.scalars)

    @property
    def stride(self) -> int:
        return -(-(self.dim + self.n_scalars) // ROW_ALIGN) * ROW_ALIGN

    @property
    def n_floats(self) -> int:
        return self.n_workers * self.stride


def reduce_rows(rows: np.ndarray, n: int, out: np.ndarray, divisor: float | None = None,
                chunk: int = REDUCE_CHUNK) -> float:
    """``out[:n] = rows[:, :n].sum(0) / divisor`` in ONE streaming pass; returns sum(out ** 2).

    Blocked by columns so the accumulator block stays in cache while each row block streams
    through once. Rows are added in row order, element by element, so the result does not depend
    on ``chunk`` and is bit-identical for a fixed number of rows. The squared norm doubles as the
    non-finite test of the reduced vector: it is finite iff every entry is finite (an entry
    beyond 1.8e19 overflows it, and such a gradient is treated as non-finite too).

    Multi-node seam: call with ``divisor=None`` over the first D + S floats of each node's slab,
    all-reduce (SUM) the resulting vector once across nodes, then divide by the global valid count
    that travelled in the same vector.
    """
    n_rows, total = rows.shape[0], 0.0
    scale = None if divisor is None else np.float32(divisor)
    tmp = np.empty(min(chunk, max(n, 1)), dtype=np.float32)
    with np.errstate(over="ignore", invalid="ignore"):
        for lo in range(0, n, chunk):
            hi = min(lo + chunk, n)
            acc = out[lo:hi]
            np.copyto(acc, rows[0, lo:hi])
            for k in range(1, n_rows):
                np.add(acc, rows[k, lo:hi], out=acc)
            if scale is not None:
                np.divide(acc, scale, out=acc)
            sq = np.multiply(acc, acc, out=tmp[: hi - lo])
            total += float(sq.sum(dtype=np.float64))
    return total
