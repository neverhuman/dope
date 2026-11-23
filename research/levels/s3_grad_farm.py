"""Gradient farm: W single-thread gradient workers over shared-memory parameters (plan M3, stage G1).

Exact large-batch SGD on CPUs. One optimizer step is one global plan (a pure function of
``(seed, step)``), dealt to W worker processes that each own one logical CPU. Every worker builds
its episodes, runs the loss and writes the SUM of its valid per-episode gradients into its row of
a shared slab. The leader then performs ONE reduction, divides by the GLOBAL number of valid
episodes and decides apply or skip. Parameters are never copied: the leader's ``nn.Module`` and
every worker's module alias one float32 vector in ``/dev/shm``.

    spec = FarmSpec(model="pkg.mod:build", model_kwargs={...},
                    episode_fn="pkg.mod:episode", episode_init="pkg.mod:open", episode_kwargs={...},
                    loss_fn="pkg.mod:loss", plan_kwargs={"arena": "...", "batch_size": 64},
                    seed=7, n_workers=16)
    with Farm(spec) as farm:
        model = farm.parameters()                  # leader module; tensors alias the shared vector
        optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
        for step in range(start, stop):
            res = farm.step(step, skip_rule=spike_rule(history))
            if res.decision == "apply":            # model.<param>.grad already is res.grad
                optim.step()                       # in place: every worker sees it at step + 1
                history.append(res.grad_norm)

Callables in a worker (import paths; kwargs must be JSON):

* ``model(**model_kwargs) -> nn.Module`` - float32 CPU parameters; no BatchNorm running statistics.
* ``plan_fn(seed=seed, **plan_kwargs) -> (step -> list of specs)``; a spec has ``slot`` and ``cells``
  (and optionally ``n_episodes``). Default: the counter sampler, ``plan_kwargs`` = ``SamplerConfig``.
* ``episode_init(episode_kwargs) -> state`` once per worker (omitted: state is the kwargs dict).
* ``episode_fn(spec, state) -> episode | RETRY``.
* ``loss_fn(model, episode, generator, **loss_kwargs) -> scalar tensor | (scalar tensor, {name: float})``.

Determinism. The gradient of step ``s`` is a function of (parameters, seed, s): the plan and the
shard are pure, each episode's RNG is seeded from (seed, s, slot), and sums run in a fixed order.
It is bit-identical for a fixed worker count across runs and restarts (same runtime), and equal
across worker counts up to float32 summation order. The farm holds no training state, so resume
is: restore the parameters into ``farm.parameters()`` in place, call ``step(s)``.

The farm never clips and never steps the optimizer. Multi-node (gloo) seam: see ``reduce_rows``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from multiprocessing.connection import Connection, wait as wait_ready
import os
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Iterable, Mapping, Sequence
import weakref

import numpy as np
import torch

from .s3_farm_shm import (
    MALLOC_ENV, RETRY, SCALARS, THREAD_ENV, FarmError, Slab, bind_model, checksum_model, create_segment,
    model_layout, new_prefix, pick_cpus, reduce_rows, require_shm_room, resolve, sweep_stale, unlink_segment,
)

__all__ = ["RETRY", "Farm", "FarmError", "FarmSpec", "StepResult", "counter_plan", "spike_rule"]

PHASES = ("plan", "build", "forward", "backward", "reduce")
# reduce="auto": from this many slab floats per step on, the workers reduce (one column block each)
# instead of the leader. The leader's pass grows with W, a worker's share stays D floats. On a shared host
# at D = 16M the worker reduce won three runs of four (W = 4: 38-59 ms against 73-84 ms) and lost the
# one with the busiest CPUs, where it waits for workers that are not scheduled. See farm-contract.md.
PARALLEL_REDUCE_FLOATS = 32_000_000
_BOOT = ("import json,sys;sys.path[:0]=json.loads(sys.argv[1]);"
         "from {package}.s3_farm_worker import main;main(int(sys.argv[2]))")
_guards = 0


@dataclass(frozen=True)
class FarmSpec:
    """What a farm runs. Plain data: it crosses the process boundary as JSON, nothing is pickled."""

    model: str
    episode_fn: str
    loss_fn: str
    model_kwargs: Mapping[str, Any] = field(default_factory=dict)
    episode_kwargs: Mapping[str, Any] = field(default_factory=dict)
    episode_init: str = ""
    loss_kwargs: Mapping[str, Any] = field(default_factory=dict)
    plan_fn: str = f"{__name__}:counter_plan"
    plan_kwargs: Mapping[str, Any] = field(default_factory=dict)
    seed: int = 0
    n_workers: int = 1
    cpus: tuple[int, ...] | None = None  # explicit CPU per worker; must lie inside the inherited mask
    diag_keys: tuple[str, ...] = ()  # diagnostics of loss_fn that get a slab slot (others are counted, dropped)
    train_mode: bool = True  # workers call model.train(train_mode) once
    reduce: str = "auto"  # auto | leader | workers; same bits either way
    worker_env: Mapping[str, str] = field(default_factory=dict)  # extra environment of the worker processes
    step_timeout_s: float = 600.0
    start_timeout_s: float = 300.0

    def to_json(self) -> dict[str, Any]:
        try:
            cfg = json.loads(json.dumps(asdict(self)))
        except (TypeError, ValueError):
            raise FarmError("spec_not_json") from None
        paths = [cfg[k] for k in ("model", "episode_fn", "loss_fn", "plan_fn")] + [cfg["episode_init"] or "a:b"]
        if (type(self.seed) is not int or self.seed < 0 or type(self.n_workers) is not int or self.n_workers < 1
                or self.reduce not in ("auto", "leader", "workers") or any(":" not in p for p in paths)
                or min(self.step_timeout_s, self.start_timeout_s) <= 0
                or not all(isinstance(v, str) for v in cfg["worker_env"].values())):
            raise FarmError("spec_invalid")
        return cfg


@dataclass
class StepResult:
    step: int
    grad: torch.Tensor  # (D,) float32: SUM of valid episode gradients / n_valid. Farm-owned; valid until the next step
    grad_norm: float  # L2 norm of ``grad``; non-finite iff ``grad`` holds a non-finite entry
    n_planned: int
    n_valid: int
    n_fail: int
    n_retry: int
    n_nonfinite: int
    loss_mean: float  # over valid episodes
    sq_norm_sum: float  # sum over workers of ||that worker's gradient SUM||^2
    ep_sq_norm_sum: float  # sum over valid episodes of ||episode gradient||^2
    diagnostics: dict[str, float]  # summed over valid episodes; "dropped" counts undeclared / non-finite values
    # phase (plan, build, forward, backward, reduce) -> {"total": seconds summed over workers, "workers": [...]};
    # "worker_wall" / "worker_cpu" -> seconds per worker (cpu < wall: the CPU was shared);
    # "leader" -> wait (signal to last answer), reduce, total
    timings: dict[str, Any]
    workers: dict[str, list[float]]  # scalar-block column -> value per worker (n_valid, sq_norm, loss_sum, ...)
    fail_reasons: dict[str, int] = field(default_factory=dict)  # reason code -> episodes, this step
    decision: str = ""  # "apply" | "skip:no_valid" | "skip:nonfinite" | "skip:<caller reason>"


def counter_plan(seed: int, **config: Any) -> Callable[[int], list[Any]]:
    """Default ``plan_fn``: the counter sampler of this lane. ``config`` are SamplerConfig fields."""
    from .s3_counter_sampler import SamplerConfig, sampler_for

    def frozen(value: Any) -> Any:  # JSON lists back to the tuples the (hashable) config holds
        return tuple(frozen(v) for v in value) if isinstance(value, list) else value

    sampler = sampler_for(SamplerConfig(**{key: frozen(value) for key, value in config.items()}))
    return lambda step: sampler.plan_step(seed, step)


def spike_rule(history: Sequence[float], factor: float = 10.0, floor: float = 20.0,
               min_history: int = 16) -> Callable[[StepResult], str | None]:
    """Caller-side skip rule of the plan: skip when ``||g|| > max(factor * median(history), floor)``.

    ``history`` (for example the last 64 applied norms) belongs to the caller and to its checkpoint.
    """
    def rule(result: StepResult) -> str | None:
        if len(history) < min_history:
            return None
        return "spike" if result.grad_norm > max(factor * statistics.median(history), floor) else None
    return rule


def _on_sigterm(signum: int, frame: Any) -> None:
    raise SystemExit(128 + signum)


def _guard_sigterm(delta: int) -> None:
    """While a farm is open and SIGTERM is at its default, turn it into SystemExit so ``with`` unwinds."""
    global _guards
    main = threading.current_thread() is threading.main_thread()
    if main and delta > 0 and _guards == 0 and signal.getsignal(signal.SIGTERM) is signal.SIG_DFL:
        signal.signal(signal.SIGTERM, _on_sigterm)
    _guards += delta
    if main and _guards == 0 and signal.getsignal(signal.SIGTERM) is _on_sigterm:
        signal.signal(signal.SIGTERM, signal.SIG_DFL)


class _Resources:
    """Everything that must not outlive a farm. ``release`` is idempotent and finaliser-safe."""

    def __init__(self) -> None:
        self.pid, self.closed = os.getpid(), False
        self.names: list[str] = []
        self.procs: list[subprocess.Popen[bytes]] = []
        self.conns: list[Connection] = []
        _guard_sigterm(+1)

    def release(self, grace: float = 5.0) -> None:
        if self.closed or os.getpid() != self.pid:  # a forked copy owns nothing
            return
        self.closed = True
        try:
            for conn in self.conns:
                try:
                    conn.send_bytes(b'["exit"]')
                except (OSError, ValueError):
                    pass
            for action, patience in (("poll", grace), ("terminate", 2.0), ("kill", 10.0)):
                deadline = time.monotonic() + patience
                for proc in self.procs:
                    if proc.poll() is None:
                        getattr(proc, action)()
                for proc in self.procs:
                    try:
                        proc.wait(timeout=max(0.0, deadline - time.monotonic()))
                    except subprocess.TimeoutExpired:
                        pass
        finally:
            for conn in self.conns:
                try:
                    conn.close()
                except OSError:
                    pass
            for name in self.names:
                try:
                    unlink_segment(name)
                except OSError:
                    pass
            _guard_sigterm(-1)


class Farm:
    """Leader side. Use as a context manager; every method is called from one thread."""

    def __init__(self, spec: FarmSpec) -> None:
        self.spec, cfg = spec, spec.to_json()  # the JSON round trip is exactly what the workers receive
        self._res = _Resources()
        self._finalizer = weakref.finalize(self, self._res.release)
        try:
            self._start(cfg)
        except BaseException:
            self._res.release(grace=0.5)
            raise

    def _start(self, cfg: dict[str, Any]) -> None:
        res, n = self._res, int(cfg["n_workers"])
        self.cpus = pick_cpus(n, cfg["cpus"])
        self._model = resolve(cfg["model"])(**cfg["model_kwargs"])
        self._layout = layout = model_layout(self._model)
        self._slab = slab = Slab(n, layout.dim, SCALARS + tuple("diag:" + key for key in cfg["diag_keys"]))
        self._parallel = cfg["reduce"] == "workers" or (
            cfg["reduce"] == "auto" and n >= 2 and n * layout.dim >= PARALLEL_REDUCE_FLOATS)
        sweep_stale()
        sizes = {"params": layout.n_shared, "slab": slab.n_floats, "grad": layout.dim}
        require_shm_room(4 * sum(sizes.values()))
        self.prefix = new_prefix()
        names = {kind: f"{self.prefix}.{kind}" for kind in sizes}
        arrays = {}
        for kind, size in sizes.items():
            res.names.append(names[kind])
            arrays[kind] = create_segment(names[kind], size)
        self._flat = torch.from_numpy(arrays["params"])
        self._params = bind_model(self._model, layout, self._flat, copy_in=True)
        self._rows = arrays["slab"].reshape(n, slab.stride)
        self._grad_np = arrays["grad"]
        self._grad = torch.from_numpy(self._grad_np)
        self._grad_views = [self._grad[off:off + size].view(shape) for _, kind, off, size, shape in layout.entries
                            if kind == "param"]

        env = {**MALLOC_ENV, **os.environ, "PYTHONHASHSEED": "0", **cfg["worker_env"]}
        env.update(dict.fromkeys(THREAD_ENV, "1"), CUDA_VISIBLE_DEVICES="")  # not negotiable
        boot = [sys.executable, *(["-B"] if sys.dont_write_bytecode else []), "-c",
                _BOOT.format(package=__package__), json.dumps([p for p in sys.path if p])]
        hello = {"n_workers": n, "leader_pid": os.getpid(), "segments": names, "layout_digest": layout.digest,
                 "scalars": list(slab.scalars), "spec": cfg}
        for w in range(n):
            ours, theirs = socket.socketpair()
            try:  # fresh interpreter (fork + exec); it inherits its socket end and nothing else
                proc = subprocess.Popen(boot + [str(theirs.fileno())], pass_fds=(theirs.fileno(),), env=env,
                                        stdin=subprocess.DEVNULL)
            except BaseException:
                ours.close()
                raise
            finally:
                theirs.close()
            res.procs.append(proc)
            res.conns.append(Connection(ours.detach()))
            res.conns[w].send_bytes(json.dumps({**hello, "worker": w, "cpu": self.cpus[w]}).encode())
        self.workers = self._gather(range(n), float(cfg["start_timeout_s"]), "start")
        if any(info["private"] != layout.private for info in self.workers):
            self._fail("private_state_mismatch: the model factory builds different non-float32 buffers per process")

    # -- plumbing ---------------------------------------------------------------------------
    def _fail(self, message: str) -> None:
        self._res.release(grace=0.5)
        raise FarmError(message)

    def _exit_status(self, w: int) -> str:
        try:
            code = self._res.procs[w].wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            return "closed its channel"
        return f"exited with code {code}" if code >= 0 else f"was killed by signal {-code}"

    def _gather(self, workers: Iterable[int], timeout: float, what: str, fatal: bool = True) -> list[Any]:
        """One answer per worker, or a FarmError naming the dead / silent / failed worker.

        A dead or silent worker always closes the farm; a worker that answers with an error closes
        it unless ``fatal`` is false (every answer has been read by then, so the channels stay in step).
        """
        res, order = self._res, list(workers)
        pending = {res.conns[w]: w for w in order}
        out: dict[int, Any] = {}
        errors: dict[int, str] = {}
        deadline = time.monotonic() + timeout
        while pending:
            ready = wait_ready(list(pending), timeout=max(0.0, deadline - time.monotonic()))
            if not ready:
                self._fail(f"timeout after {timeout:g} s: workers {sorted(pending.values())} silent during {what}")
            for conn in ready:
                w = pending.pop(conn)
                try:
                    status, payload = conn.recv()
                except (EOFError, OSError):
                    self._fail(f"worker {w} (pid {res.procs[w].pid}) {self._exit_status(w)} during {what}")
                except Exception as exc:
                    self._fail(f"worker {w} sent an unreadable answer during {what}: {type(exc).__name__}")
                (out if status == "ok" else errors)[w] = payload
        if errors:
            message = f"workers {sorted(errors)} failed during {what}: {errors[min(errors)]}"
            if fatal:
                self._fail(message)
            raise FarmError(message)
        return [out[w] for w in order]

    def _call(self, message: list[Any], what: str, workers: Sequence[int] | None = None,
              fatal: bool = True) -> list[Any]:
        if self._res.closed:
            raise FarmError("farm_closed")
        data, order = json.dumps(message).encode(), list(range(self.n_workers) if workers is None else workers)
        for w in order:
            try:
                self._res.conns[w].send_bytes(data)
            except (OSError, ValueError):
                self._fail(f"worker {w} (pid {self._res.procs[w].pid}) {self._exit_status(w)} before {what}")
        return self._gather(order, self.spec.step_timeout_s, what, fatal)

    # -- interface --------------------------------------------------------------------------
    def parameters(self) -> torch.nn.Module:
        """The leader's module. Its float32 tensors are views into the shared vector: update them
        IN PLACE (optimizer step, ``load_state_dict``); re-binding one (``p.data = ...``, ``.to()``)
        detaches the leader, which ``param_checksum`` reports."""
        return self._model

    @property
    def flat(self) -> torch.Tensor:
        """The trainable parameters as one shared float32 vector of length ``dim`` (layout order)."""
        return self._flat[: self._layout.dim]

    @property
    def dim(self) -> int:
        return self._layout.dim

    @property
    def n_workers(self) -> int:
        return self._slab.n_workers

    @property
    def pids(self) -> list[int]:
        return [proc.pid for proc in self._res.procs]

    @property
    def closed(self) -> bool:
        return self._res.closed

    @property
    def shm_bytes(self) -> dict[str, int]:
        """tmpfs bytes per segment once every page is touched (the slab is n_workers rows of ~4 * dim)."""
        return {"params": 4 * self._layout.n_shared, "slab": 4 * self._slab.n_floats, "grad": 4 * self._layout.dim}

    def step(self, step_index: int, skip_rule: Callable[[StepResult], str | None] | None = None) -> StepResult:
        """Gradient of global step ``step_index`` at the current shared parameters.

        Skip iff no episode was valid, the reduced vector holds a non-finite entry, or
        ``skip_rule(result)`` returns a reason. On "apply" the leader module's ``.grad`` fields are
        views of ``result.grad``; on a skip they are ``None``, so a stray ``optimizer.step()`` is inert.
        """
        if type(step_index) is not int or step_index < 0:
            raise FarmError("step_index")
        t0 = time.perf_counter()
        self._call(["step", step_index], f"step {step_index}")
        t1 = time.perf_counter()
        d, names = self.dim, self._slab.scalars
        per = self._rows[:, d:d + len(names)].astype(np.float64)  # (W, S) scalar blocks
        tot = dict(zip(names, per.sum(axis=0).tolist()))
        count = {key: int(round(tot["n_" + key])) for key in ("planned", "valid", "fail", "retry", "nonfinite")}
        n_valid = count["valid"]
        if n_valid == 0:
            self._grad_np.fill(0.0)
            sq = 0.0
        elif self._parallel:  # each worker reduces one column block of the slab: same bits, W-fold bandwidth
            sq = math.fsum(self._call(["reduce", n_valid], f"reduce of step {step_index}"))
        else:
            sq = reduce_rows(self._rows, d, self._grad_np, n_valid)
        t2 = time.perf_counter()
        workers = {name: per[:, i].tolist() for i, name in enumerate(names)}
        timings: dict[str, Any] = {ph: {"total": tot["t_" + ph], "workers": workers["t_" + ph]} for ph in PHASES}
        diagnostics = {name[5:]: tot[name] for name in names if name.startswith("diag:")}
        diagnostics["dropped"] = tot["diag_dropped"]
        result = StepResult(
            step=step_index, grad=self._grad, grad_norm=math.sqrt(sq), n_planned=count["planned"], n_valid=n_valid,
            n_fail=count["fail"], n_retry=count["retry"], n_nonfinite=count["nonfinite"],
            loss_mean=tot["loss_sum"] / n_valid if n_valid else math.nan, sq_norm_sum=tot["sq_norm"],
            ep_sq_norm_sum=tot["ep_sq_norm"], diagnostics=diagnostics, timings=timings, workers=workers)
        if count["fail"]:
            for reasons in self._call(["collect", "farm.step_fail_reasons"], "fail reasons"):
                for code, k in reasons.items():
                    result.fail_reasons[code] = result.fail_reasons.get(code, 0) + k
        if n_valid == 0:
            result.decision = "skip:no_valid"
        elif not math.isfinite(sq):
            result.decision = "skip:nonfinite"
        else:
            reason = skip_rule(result) if skip_rule is not None else None
            result.decision = f"skip:{reason}" if reason else "apply"
        for p, view in zip(self._params, self._grad_views):
            p.grad = view if result.decision == "apply" else None
        timings["worker_wall"], timings["worker_cpu"] = workers["t_total"], workers["t_cpu"]
        timings["leader"] = {"wait": t1 - t0, "reduce": t2 - t1, "total": time.perf_counter() - t0}
        return result

    def param_checksum(self) -> str:
        """Digest of the shared tensors as the leader's model and every worker's model hold them.

        Raises when any process reads different parameters (a worker on an out-of-date private copy, a
        leader tensor re-bound away from the shared vector). Costs one pass over the vector per
        process; meant for every few hundred steps.
        """
        tensors, base = {**dict(self._model.named_buffers()), **dict(self._model.named_parameters())}, self._flat
        if any(tensors[name].data_ptr() != base.data_ptr() + 4 * off for name, _, off, _, _ in self._layout.entries):
            raise FarmError("param_checksum mismatch: a leader tensor no longer aliases the shared vector")
        mine = checksum_model(self._model, self._layout)
        bad = [w for w, theirs in enumerate(self._call(["checksum"], "param_checksum")) if theirs != mine]
        if bad:
            raise FarmError(f"param_checksum mismatch: workers {bad} do not see the leader's parameters")
        return mine

    def collect(self, name: str) -> list[Any]:
        """One small picklable object per worker, in worker order.

        ``name`` is looked up in the worker's episode state: ``state.farm_collect(name)`` when the
        state has that method, else ``state[name]`` for a mapping. Built in: ``farm.info``,
        ``farm.fail_reasons`` (cumulative reason code -> episodes). An unknown name raises and
        leaves the farm open.
        """
        return self._call(["collect", str(name)], f"collect {name}", fatal=False)

    def _debug_detach(self, worker: int) -> None:
        """Test hook: one worker switches to a private copy of the parameters (no longer updated)."""
        self._call(["debug_detach"], "debug_detach", [worker])

    def close(self) -> None:
        """Stop the workers and unlink the segments. Idempotent. The leader's module stays usable:
        its tensors keep the (now anonymous) parameter mapping alive."""
        self._res.release()
        self._rows = self._grad_np = None  # type: ignore[assignment]

    def __enter__(self) -> "Farm":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
