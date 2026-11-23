"""Parallel FIT-only training producers and native episode adapters.

See s3_episode_ring for the shared-memory layout and worker handoff.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import pickle
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator, Sequence

import numpy as np

from .s3_episode_ring import (
    EMPTY, FLAG_ERROR, H_CAP, H_COUNTERS, H_MAGIC, H_PRODUCERS, H_SLOTS, H_STEP,
    H_STOP, H_STRIDE, H_UNIT0, H_VERSION, MAGIC, MAX_PRODUCERS, META_AT, PAGE,
    PREFIX, READY, SHM_DIR, SLOT_HEAD, VERSION, EpisodeFailed, EpisodeRing,
    FactoryError, RingDown, StageError, _fault, resolve, sweep_stale,
)
# Native DSE7/V8 stages live in s3_episode_native; re-exported for stage paths and callers.
from .s3_episode_native import (  # noqa: F401
    ABI_VERSION,
    META_FIELDS,
    V8_MAX_COLS,
    _BLOCKS,
    _CODES,
    _TASK_ID,
    _V8_BLOCKS,
    _V8_SHAPES,
    _ptr,
    column_meta,
    dse7_arrays,
    ffi_path,
    load_ffi,
    native_capacity,
    native_dse7,
    native_v8,
    split_native,
    split_v8,
    v8_arrays,
    v8_capacity,
)

THREAD_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
              "RAYON_NUM_THREADS")


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
                    new_plan = sampler.plan_step(spec.seed, step)
                    if len(new_plan) != per_step:  # every slot of the step fails, none builds from it
                        raise FactoryError("plan_size")
                    plan_step, plan = step, new_plan
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
        # A consumer in another process (a farm worker) cannot see the producers: a producer that
        # dies stops the ring with the failure flag, so every waiting consumer raises.
        self._done = threading.Event()
        self._watch = threading.Thread(target=self._watchdog, name="dope-episode-watchdog", daemon=True)
        self._watch.start()

    def _watchdog(self) -> None:
        while not self._done.wait(0.25):
            if any(p.exitcode not in (None, 0) for p in self.procs):
                ring = self.ring
                if ring is not None:
                    ring.stop(failed=True)
                return

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
        done = getattr(self, "_done", None)
        if done is not None:
            done.set()
            self._watch.join(timeout=5)
        if not ring.stopped:
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


# -- gradient-farm adapter ------------------------------------------------------------------
def farm_open(kwargs: dict[str, Any]) -> dict[str, Any]:
    """``episode_init`` of a gradient farm: attach the factory's ring once per farm worker.

    ``episode_kwargs = {"ring": factory.name, "timeout": seconds}`` (timeout default 600 s).
    """
    return {"ring": EpisodeRing.attach(str(kwargs["ring"])), "timeout": float(kwargs.get("timeout", 600.0))}


def farm_episode(spec: Any, state: dict[str, Any]) -> dict[str, Any]:
    """``episode_fn`` of a gradient farm: the factory's episode for ``(spec.step, spec.slot)``.

    A failed unit raises ``EpisodeFailed`` (counted by the farm). A ring failure raises
    ``RingDown`` (the step fails). The farm's ``plan_fn`` must be the factory's sampler and seed.
    """
    try:
        payload = state["ring"].get(int(spec.step), int(spec.slot), state["timeout"])
    except FactoryError as exc:
        raise RingDown(str(exc)) from None
    return payload["episode"]


# -- the v3 training preset (plan C3) -------------------------------------------------------
# Native stages have no multilabel task and 48 packet columns; query sets get at least 256 rows
# where the table allows. Measured on s3-v250 (seed 7, 10 steps x 64 through the factory):
# 75 % real views in every step, 0 failed, 2 of 640 slots retried and none fell back to a prior,
# n_fit >= 256 in 59.7 %, |Q| >= 256 in 62.3 %, packets 805 B median.
V3_SAMPLER: dict[str, Any] = {"qry_lo": 256, "multilabel_view_prob": 0.0, "max_view_cols": 48,
                              "max_total_cols": 48}


def v3_spec(arena: str, seed: int, batch_size: int = 64, budget: int = 2048, **sampler: Any) -> FactorySpec:
    """The v3 factory spec over a flat arena; ``sampler`` overrides ``V3_SAMPLER`` fields."""
    from .s3_counter_sampler import SamplerConfig
    from .s3_view_builder import ViewConfig

    cfg = SamplerConfig(arena=arena, batch_size=batch_size, **{**V3_SAMPLER, **sampler})
    return FactorySpec(seed=seed, corpus=f"{__name__}:arena_corpus", corpus_args=(arena,),
                       sampler=f"{__name__}:arena_sampler", sampler_args=(cfg,), view=ViewConfig(emit="raw"),
                       stage=f"{__name__}:native_dse7", stage_kwargs={"budget": budget})


def arena_corpus(arena: str) -> Any:
    from .s3_view_corpus import ArenaCorpus

    return ArenaCorpus(arena)


def arena_sampler(corpus: Any, cfg: Any) -> Any:
    """The counter sampler of a ``SamplerConfig`` that names its arena (``corpus`` is unused)."""
    from .s3_counter_sampler import sampler_for

    return sampler_for(cfg)


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
