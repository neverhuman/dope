"""Worker process of the S3 gradient farm (plan M3, stage G1).

Started by the leader as a fresh interpreter (fork + exec, never a fork of a process that holds
torch) with ``OMP_NUM_THREADS=1``, ``MKL_NUM_THREADS=1`` and ``CUDA_VISIBLE_DEVICES=""`` already
in its environment, so they are set before torch is imported. The worker pins itself to ONE
logical CPU of the mask it inherited, runs torch with one intra-op and one inter-op thread, builds
its own model from the factory import path and re-points the model's tensors at the shared
parameter vector. Everything the leader sends is JSON; the worker answers with small pickles.

One step, on the command ``["step", s]``:

1. ``specs = shard(plan(s), worker, n_workers)`` - planned HERE, nothing is shipped.
2. For each spec in slot order: ``episode_fn(spec, state)``; seed the global torch RNG and a
   ``torch.Generator`` with ``episode_seed(seed, s, spec.slot)``; ``loss_fn(model, episode,
   generator)``; ``torch.autograd.grad`` into FRESH tensors; squared norm; add into the slab row.
3. Write the scalar block; answer.

Why ``autograd.grad`` and not ``.backward()``: a per-episode gradient that lives in its own
tensors can be inspected and thrown away before it touches the accumulator, so one NaN episode
costs that episode and nothing else. The price over accumulating into ``.grad`` is one read pass
over D floats per episode (the squared norm, which the gradient-noise estimate wants anyway);
the add into the row replaces the add autograd would do into ``.grad``.

Counting. ``RETRY`` from ``episode_fn`` -> ``n_retry``; an exception in ``episode_fn``, ``loss_fn``
or the backward pass -> ``n_fail`` with a reason code (stage, exception type, source line; never
the message); a non-finite loss or gradient -> ``n_nonfinite``. None of them reaches the row.
A spec may carry ``n_episodes`` (default 1) when it stands for a micro-batch whose loss is a sum.
"""
from __future__ import annotations

import atexit
from collections import Counter
import ctypes
import json
import math
from multiprocessing.connection import Connection
import os
import signal
import sys
import time
from typing import Any, Mapping

import numpy as np
import torch

from .s3_counter_sampler import shard
from .s3_farm_shm import (
    RETRY, SCALARS, THREAD_ENV, FarmError, Slab, attach_segment, bind_model, checksum_model, episode_seed,
    COMMAND_FAULTS, EPISODE_FAULTS, fault_code, model_layout, reduce_rows, resolve, unlink_segment,
)

_PR_SET_PDEATHSIG = 1


class _Worker:
    def __init__(self, conn: Connection, hello: dict[str, Any]) -> None:
        self.conn, self.w, self.n_workers = conn, int(hello["worker"]), int(hello["n_workers"])
        self.leader_pid, self.names = int(hello["leader_pid"]), dict(hello["segments"])
        signal.signal(signal.SIGINT, signal.SIG_IGN)  # the leader coordinates shutdown
        signal.signal(signal.SIGTERM, self._on_term)
        ctypes.CDLL(None, use_errno=True).prctl(_PR_SET_PDEATHSIG, int(signal.SIGTERM), 0, 0, 0)
        if os.getppid() != self.leader_pid:
            self._abandon()
        cpu = int(hello["cpu"])
        if cpu not in os.sched_getaffinity(0):
            raise FarmError("cpu_outside_inherited_mask")
        os.sched_setaffinity(0, {cpu})  # a subset of the inherited mask: never wider
        if any(os.environ.get(k) != "1" for k in THREAD_ENV) or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
            raise FarmError("worker_environment")
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)

        spec = hello["spec"]
        self.seed = int(spec["seed"])
        self.model = resolve(spec["model"])(**spec["model_kwargs"])
        self.layout = layout = model_layout(self.model)
        if layout.digest != hello["layout_digest"]:
            raise FarmError("layout_mismatch")
        self.slab = slab = Slab(self.n_workers, layout.dim, tuple(hello["scalars"]))
        self.flat = torch.from_numpy(attach_segment(self.names["params"], layout.n_shared))
        self.rows = attach_segment(self.names["slab"], slab.n_floats).reshape(self.n_workers, slab.stride)
        self.grad = attach_segment(self.names["grad"], layout.dim)
        self.params = bind_model(self.model, layout, self.flat, copy_in=False)
        self.row_np = self.rows[self.w, : layout.dim]
        self.row = torch.from_numpy(self.row_np)
        self.row_views = [self.row[off:off + n].view(shape) for _, kind, off, n, shape in layout.entries
                          if kind == "param"]
        self.scal = self.rows[self.w, layout.dim: layout.dim + slab.n_scalars]
        self.diag_ix = {name[5:]: i for i, name in enumerate(slab.scalars) if name.startswith("diag:")}
        self.model.train(bool(spec["train_mode"]))
        self.episode_fn, self.loss_fn = resolve(spec["episode_fn"]), resolve(spec["loss_fn"])
        self.loss_kwargs = spec["loss_kwargs"]
        init = spec["episode_init"]
        self.state = resolve(init)(spec["episode_kwargs"]) if init else spec["episode_kwargs"]
        self.plan = resolve(spec["plan_fn"])(seed=self.seed, **spec["plan_kwargs"])
        self.gen = torch.Generator()
        self.fail_total: Counter[str] = Counter()
        self.fail_step: Counter[str] = Counter()

    # -- lifetime ---------------------------------------------------------------------------
    def _abandon(self) -> None:
        """The leader is gone without saying goodbye: remove the segment names and leave."""
        for name in self.names.values():
            try:
                unlink_segment(name)
            except OSError:
                pass
        os._exit(0)

    def _on_term(self, signum: int, frame: Any) -> None:
        if os.getppid() != self.leader_pid:
            self._abandon()
        os._exit(128 + signum)

    def info(self) -> dict[str, Any]:
        return {"worker": self.w, "pid": os.getpid(), "cpu": sorted(os.sched_getaffinity(0)),
                "torch_threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
                "env": {k: os.environ.get(k) for k in (*THREAD_ENV, "CUDA_VISIBLE_DEVICES")},
                "private": self.layout.private, "torch": torch.__version__, "numpy": np.__version__}

    # -- one step ---------------------------------------------------------------------------
    def run_step(self, step: int) -> None:
        clock, cpu0 = time.perf_counter, time.process_time()
        t0 = clock()
        specs = shard(self.plan(step), self.w, self.n_workers)
        t_plan = clock() - t0
        self.row.zero_()
        self.fail_step.clear()
        n = dict.fromkeys(("planned", "valid", "fail", "retry", "nonfinite", "diag_dropped"), 0)
        loss_sum = ep_sq = t_build = t_forward = t_backward = 0.0
        t_reduce = clock() - t0 - t_plan
        block = np.zeros(self.slab.n_scalars, dtype=np.float64)  # diagnostic sums, then the base scalars
        for spec in specs:
            weight = int(getattr(spec, "n_episodes", 1))
            n["planned"] += weight
            ta = clock()
            stage = "build"
            try:
                episode = self.episode_fn(spec, self.state)
                tb = clock()
                t_build += tb - ta
                if episode is RETRY:
                    n["retry"] += weight
                    continue
                stage = "loss"
                seed = episode_seed(self.seed, step, int(spec.slot))
                torch.manual_seed(seed)  # dropout and other global-RNG draws replay too
                self.gen.manual_seed(seed)
                out = self.loss_fn(self.model, episode, self.gen, **self.loss_kwargs)
                loss, diag = out if isinstance(out, tuple) else (out, None)
                value = float(loss)
                tc = clock()
                t_forward += tc - tb
                if not math.isfinite(value):
                    n["nonfinite"] += weight
                    continue
                stage = "grad"
                grads = torch.autograd.grad(loss, self.params, allow_unused=True)
                td = clock()
                t_backward += td - tc
            except EPISODE_FAULTS as exc:  # a data-dependent fault fails this episode, counted by type
                n["fail"] += weight
                self.fail_step[fault_code(stage, exc)] += weight
                continue
            sq = 0.0
            for g in grads:
                if g is not None:
                    flat = g.numpy().reshape(-1)
                    sq += float(np.dot(flat, flat))  # numpy's own BLAS; torch.dot can be a scalar loop
            if math.isfinite(sq):  # finite squared norm <=> every entry finite (and < 1.8e19)
                for view, g in zip(self.row_views, grads):
                    if g is not None:
                        view.add_(g)
                n["valid"] += weight
                loss_sum += value
                ep_sq += sq
                for key, val in (diag.items() if isinstance(diag, dict) else ()):
                    ix = self.diag_ix.get(key)
                    try:
                        val = float(val) if ix is not None else math.nan
                    except (TypeError, ValueError, OverflowError):
                        val = math.nan
                    if math.isfinite(val):
                        block[ix] += val
                    else:  # undeclared key, not a number, or not finite
                        n["diag_dropped"] += 1
            else:
                n["nonfinite"] += weight
            t_reduce += clock() - td
        t1 = clock()
        row_sq = float(np.dot(self.row_np, self.row_np)) if n["valid"] else 0.0
        self.fail_total.update(self.fail_step)
        t_end = clock()
        base = {"n_planned": n["planned"], "n_valid": n["valid"], "n_fail": n["fail"], "n_retry": n["retry"],
                "n_nonfinite": n["nonfinite"], "loss_sum": loss_sum, "sq_norm": row_sq, "ep_sq_norm": ep_sq,
                "diag_dropped": n["diag_dropped"], "t_plan": t_plan, "t_build": t_build, "t_forward": t_forward,
                "t_backward": t_backward, "t_reduce": t_reduce + t_end - t1, "t_total": t_end - t0,
                "t_cpu": time.process_time() - cpu0}
        for i, name in enumerate(SCALARS):
            block[i] = base[name]
        with np.errstate(over="ignore"):
            self.scal[:] = block

    def reduce_block(self, divisor: float) -> float:
        """Column block ``w`` of the global reduction: every row, this worker's share of D."""
        d = self.layout.dim
        lo, hi = d * self.w // self.n_workers, d * (self.w + 1) // self.n_workers
        return reduce_rows(self.rows[:, lo:hi], hi - lo, self.grad[lo:hi], divisor)

    # -- commands ---------------------------------------------------------------------------
    def collect(self, name: str) -> Any:
        if name == "farm.info":
            return self.info()
        if name == "farm.fail_reasons":
            return dict(self.fail_total)
        if name == "farm.step_fail_reasons":
            return dict(self.fail_step)
        hook = getattr(self.state, "farm_collect", None)
        if callable(hook):
            return hook(name)
        if isinstance(self.state, Mapping) and name in self.state:
            return self.state[name]
        raise FarmError("collect_unknown_name")

    def dispatch(self, cmd: str, args: list[Any]) -> Any:
        if cmd == "step":
            return self.run_step(int(args[0]))
        if cmd == "reduce":
            return self.reduce_block(float(args[0]))
        if cmd == "checksum":
            return checksum_model(self.model, self.layout)
        if cmd == "collect":
            return self.collect(str(args[0]))
        if cmd == "debug_detach":  # test hook: read from a private copy that is no longer updated
            for p in self.params:
                p.data = p.data.clone()
            return None
        raise FarmError("unknown_command")

    def serve(self) -> None:
        while True:
            try:
                msg = json.loads(self.conn.recv_bytes())
            except (EOFError, OSError):
                self._abandon()
            if msg[0] == "exit":
                return
            try:
                reply = ("ok", self.dispatch(msg[0], msg[1:]))
            except COMMAND_FAULTS as exc:
                reply = ("err", _describe(exc))
            try:
                self.conn.send(reply)
            except (BrokenPipeError, ConnectionError):
                self._abandon()
            except COMMAND_FAULTS as exc:  # the answer could not be pickled
                self.conn.send(("err", _describe(exc)))


def _describe(exc: BaseException) -> str:
    code = fault_code("worker", exc)
    return f"{code}:{exc}" if isinstance(exc, FarmError) else code


def main(fd: int) -> None:
    """Entry point of the worker interpreter; ``fd`` is its end of the leader's socket pair."""
    conn = Connection(fd)
    try:
        worker = _Worker(conn, json.loads(conn.recv_bytes()))
        conn.send(("ok", worker.info()))
    except COMMAND_FAULTS as exc:
        try:
            conn.send(("err", _describe(exc)))
        except OSError:
            pass
        sys.exit(1)
    worker.serve()
    atexit._run_exitfuncs()  # user clean-up runs; the slow teardown of the interpreter and torch does not
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
