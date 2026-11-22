"""Honest data-stream counters for the batched S3 training core (plan M3, stage C1).

The former ``unique_splits_seen`` was ``steps * 10 * world_size``: a counter that would read
the same if every rank produced identical data. These counters measure instead:

* exact bitmaps of the distinct tables and families that reached a built episode;
* HyperLogLog sketches (p = 14, 64-bit hash, 16 KiB each) of the distinct views
  (table, column set, fit rows, query rows, augmentation seed) and distinct splits
  (table, fit rows, query rows), hashed from content, never from a step or rank number;
* episodes planned / built / failed / committed, and repeated tables / families per global
  step, which must stay zero;
* an exposure ledger: committed visits per table and per family.

Every part merges (workers, ranks) and serialises to plain numpy arrays. Two ranks that build
the same views add nothing to the distinct counts, which makes the numbers falsifiable.
"""
from __future__ import annotations

import hashlib
import math
import struct
import time
from typing import Any, Iterable, Sequence

import numpy as np

from .s3_counter_sampler import TASKS, CounterSampler, EpisodeSpec, SamplerError, micro_batches

_ALPHA_INF = 0.5 / math.log(2.0)
_KINDS = ("real", "target_swap", "scm_prior")
_SCALARS = ("planned", "built", "failed", "committed", "steps_planned", "repeated_tables", "repeated_families")


def hash64(values: Any) -> np.ndarray:
    """splitmix64 output at each integer index: a 64-bit hash for sketching plain ids."""
    x = (np.asarray(values).astype(np.uint64) + np.uint64(1)) * np.uint64(0x9E3779B97F4A7C15)
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


def _sigma(x: float) -> float:
    if x == 1.0:
        return math.inf
    y, z = 1.0, x
    while True:
        x *= x
        before = z
        z += x * y
        y += y
        if z == before:
            return z


def _tau(x: float) -> float:
    if x in (0.0, 1.0):
        return 0.0
    y, z = 1.0, 1.0 - x
    while True:
        x = math.sqrt(x)
        before = z
        y *= 0.5
        z -= (1.0 - x) ** 2 * y
        if z == before:
            return z / 3.0


class HyperLogLog:
    """Distinct-count sketch over 64-bit hashes with Ertl's improved estimator (no bias table)."""

    def __init__(self, p: int = 14, registers: np.ndarray | None = None) -> None:
        if type(p) is not int or not 11 <= p <= 18:  # 64 - p bits must stay exact in a float64
            raise SamplerError("hll_precision")
        self.p, self.q = p, 64 - p
        self.reg = np.zeros(1 << p, dtype=np.uint8) if registers is None else np.array(registers, dtype=np.uint8)
        if self.reg.shape != (1 << p,) or int(self.reg.max(initial=0)) > self.q + 1:
            raise SamplerError("hll_registers")

    def add(self, h: int) -> None:
        h &= (1 << 64) - 1
        rank = self.q + 1 - (h & ((1 << self.q) - 1)).bit_length()
        if rank > self.reg[h >> self.q]:
            self.reg[h >> self.q] = rank

    def add_many(self, hashes: Any) -> None:
        h = np.asarray(hashes, dtype=np.uint64).ravel()
        low = (h & np.uint64((1 << self.q) - 1)).astype(np.float64)  # exact: fewer than 2**53
        rank = (self.q + 1 - np.frexp(low)[1]).astype(np.uint8)  # frexp exponent = bit length
        np.maximum.at(self.reg, (h >> np.uint64(self.q)).astype(np.intp), rank)

    def merge(self, other: "HyperLogLog") -> None:
        if other.p != self.p:
            raise SamplerError("hll_precision")
        np.maximum(self.reg, other.reg, out=self.reg)

    def estimate(self) -> float:
        m = float(self.reg.size)
        hist = np.bincount(self.reg, minlength=self.q + 2).astype(np.float64)
        z = m * _tau(1.0 - hist[self.q + 1] / m)
        for k in range(self.q, 0, -1):
            z = 0.5 * (z + hist[k])
        z += m * _sigma(hist[0] / m)
        return _ALPHA_INF * m * m / z


def _rows(a: Any) -> bytes:
    arr = np.sort(np.asarray(a, dtype="<i8").ravel())
    return struct.pack("<q", arr.size) + arr.tobytes()


def split_hash(spec: EpisodeSpec, fit_rows: Any, qry_rows: Any) -> int:
    """64-bit content hash of (table, fit rows, query rows); a prior table is keyed by its seed."""
    h = hashlib.blake2b(digest_size=8, person=b"dope-s3-split/v1")
    h.update(struct.pack("<qQ", spec.table_id, spec.view_seed if spec.table_id < 0 else 0))
    h.update(_rows(fit_rows) + _rows(qry_rows))
    return int.from_bytes(h.digest(), "little")


def view_hash(spec: EpisodeSpec, cols: Any, fit_rows: Any, qry_rows: Any, target_col: int = -1) -> int:
    """64-bit content hash of (table, target, column set, fit rows, query rows, augmentation seed)."""
    h = hashlib.blake2b(digest_size=8, person=b"dope-s3-view/v1")
    h.update(struct.pack("<qqqQ", _KINDS.index(spec.kind), spec.table_id, target_col, spec.view_seed))
    h.update(_rows(cols) + _rows(fit_rows) + _rows(qry_rows))
    return int.from_bytes(h.digest(), "little")


def step_repeats(specs: Iterable[EpisodeSpec]) -> tuple[int, int]:
    """(repeated tables, repeated families) inside one global step; both must be zero."""
    tables, families, n_tables, n_families = set(), set(), 0, 0
    for spec in specs:
        n_families += 1
        families.add(spec.family_id)
        if spec.table_id >= 0:
            n_tables += 1
            tables.add(spec.table_id)
    return n_tables - len(tables), n_families - len(families)


class ExposureLedger:
    """Committed visits per table and family. A step commits once, in increasing order."""

    def __init__(self, n_tables: int, n_families: int) -> None:
        self.table_visits = np.zeros(n_tables, dtype=np.int64)
        self.family_visits = np.zeros(n_families, dtype=np.int64)  # real + target-swap visits
        self.swap_visits = np.zeros(n_families, dtype=np.int64)  # the target-swap part
        self.prior_visits, self.steps, self.last_step = 0, 0, -1

    def commit(self, step: int, specs: Sequence[EpisodeSpec]) -> None:
        if type(step) is not int or step <= self.last_step:
            raise SamplerError("ledger_step_order")
        if any(spec.step != step for spec in specs):
            raise SamplerError("ledger_step_mismatch")
        seen = [spec for spec in specs if spec.table_id >= 0]
        np.add.at(self.table_visits, [spec.table_id for spec in seen], 1)
        np.add.at(self.family_visits, [spec.family_ix for spec in seen], 1)
        np.add.at(self.swap_visits, [spec.family_ix for spec in seen if spec.kind == "target_swap"], 1)
        self.prior_visits += len(specs) - len(seen)
        self.steps, self.last_step = self.steps + 1, step

    def merge(self, other: "ExposureLedger") -> None:
        """Combine ledgers of ranks that committed disjoint shards of the same steps."""
        self.table_visits += other.table_visits
        self.family_visits += other.family_visits
        self.swap_visits += other.swap_visits
        self.prior_visits += other.prior_visits
        self.steps, self.last_step = max(self.steps, other.steps), max(self.last_step, other.last_step)

    def histogram(self, which: str = "table") -> dict[int, int]:
        """visits -> number of tables (or families) committed that many times."""
        values, counts = np.unique(self.table_visits if which == "table" else self.family_visits, return_counts=True)
        return dict(zip(values.tolist(), counts.tolist()))

    def exposure(self, family_home: np.ndarray) -> dict[str, float]:
        """Mean committed real-slot visits per family by home task, and the ratios to regression."""
        real = self.family_visits - self.swap_visits
        out: dict[str, float] = {}
        for sid, task in enumerate(TASKS):
            mine = family_home == sid
            out[f"{task}_visits_per_family"] = float(real[mine].mean()) if mine.any() else 0.0
            out[f"{task}_all_visits_per_family"] = float(self.family_visits[mine].mean()) if mine.any() else 0.0
        base, base_all = out["regression_visits_per_family"], out["regression_all_visits_per_family"]
        for task in TASKS[1:]:
            out[f"{task}_to_regression"] = out[f"{task}_visits_per_family"] / base if base else math.inf
            out[f"{task}_to_regression_all"] = out[f"{task}_all_visits_per_family"] / base_all if base_all else math.inf
        return out


class HonestCounters:
    """Distinct tables, families, views and splits, episode counts and the exposure ledger."""

    def __init__(self, n_tables: int, n_families: int, p: int = 14) -> None:
        self.tables = np.zeros(n_tables, dtype=bool)
        self.families = np.zeros(n_families, dtype=bool)
        self.views, self.splits = HyperLogLog(p), HyperLogLog(p)
        self.ledger = ExposureLedger(n_tables, n_families)
        self.count = dict.fromkeys(_SCALARS, 0)

    def observe_plan(self, specs: Sequence[EpisodeSpec]) -> tuple[int, int]:
        """Called once per global step with the whole plan (the union over workers)."""
        repeats = step_repeats(specs)
        self.count["planned"] += len(specs)
        self.count["steps_planned"] += 1
        self.count["repeated_tables"] += repeats[0]
        self.count["repeated_families"] += repeats[1]
        return repeats

    def observe_built(self, spec: EpisodeSpec, cols: Any, fit_rows: Any, qry_rows: Any, target_col: int = -1) -> None:
        """Called by whoever built the episode, with the rows and columns it actually used."""
        self.count["built"] += 1
        if spec.table_id >= 0:
            self.tables[spec.table_id] = True
            self.families[spec.family_ix] = True
        self.views.add(view_hash(spec, cols, fit_rows, qry_rows, target_col))
        self.splits.add(split_hash(spec, fit_rows, qry_rows))

    def observe_failed(self, spec: EpisodeSpec) -> None:
        self.count["failed"] += 1

    def commit(self, step: int, specs: Sequence[EpisodeSpec]) -> None:
        """Called after the optimizer step that consumed ``specs``."""
        self.ledger.commit(step, specs)
        self.count["committed"] += len(specs)

    def merge(self, other: "HonestCounters") -> None:
        """Distinct sets take the union (idempotent); episode counts and visits add."""
        self.tables |= other.tables
        self.families |= other.families
        self.views.merge(other.views)
        self.splits.merge(other.splits)
        self.ledger.merge(other.ledger)
        for name in _SCALARS:
            self.count[name] += other.count[name]

    def summary(self) -> dict[str, float]:
        return {
            **self.count,
            "distinct_tables": int(self.tables.sum()),
            "distinct_families": int(self.families.sum()),
            "distinct_views": round(self.views.estimate(), 1),
            "distinct_splits": round(self.splits.estimate(), 1),
            "committed_tables": int((self.ledger.table_visits > 0).sum()),
            "committed_families": int((self.ledger.family_visits > 0).sum()),
            "committed_steps": self.ledger.steps,
        }

    def to_arrays(self) -> dict[str, np.ndarray]:
        """Plain arrays for a checkpoint; ``from_arrays`` restores an equal object."""
        led = self.ledger
        return {
            "shape": np.array([self.tables.size, self.families.size, self.views.p], dtype=np.int64),
            "scalars": np.array([self.count[n] for n in _SCALARS] + [led.prior_visits, led.steps, led.last_step], dtype=np.int64),
            "tables": np.packbits(self.tables),
            "families": np.packbits(self.families),
            "views": self.views.reg.copy(),
            "splits": self.splits.reg.copy(),
            "table_visits": led.table_visits.copy(),
            "family_visits": led.family_visits.copy(),
            "swap_visits": led.swap_visits.copy(),
        }

    @classmethod
    def from_arrays(cls, arrays: dict[str, np.ndarray]) -> "HonestCounters":
        n_tables, n_families, p = (int(v) for v in np.asarray(arrays["shape"]))
        out = cls(n_tables, n_families, p)
        scalars = np.asarray(arrays["scalars"], dtype=np.int64).tolist()
        if len(scalars) != len(_SCALARS) + 3:
            raise SamplerError("counters_arrays")
        out.count = dict(zip(_SCALARS, scalars))
        led = out.ledger
        led.prior_visits, led.steps, led.last_step = scalars[len(_SCALARS):]
        out.tables = np.unpackbits(np.asarray(arrays["tables"], dtype=np.uint8), count=n_tables).astype(bool)
        out.families = np.unpackbits(np.asarray(arrays["families"], dtype=np.uint8), count=n_families).astype(bool)
        out.views, out.splits = HyperLogLog(p, arrays["views"]), HyperLogLog(p, arrays["splits"])
        for name, size in (("table_visits", n_tables), ("family_visits", n_families), ("swap_visits", n_families)):
            values = np.array(arrays[name], dtype=np.int64)
            if values.shape != (size,):
                raise SamplerError("counters_arrays")
            setattr(led, name, values)
        return out


def measure_plan(sampler: CounterSampler, seed: int = 0, steps: int = 2000) -> dict[str, Any]:
    """Plan ``steps`` global steps and measure what the plan alone determines (no rows are read)."""
    ix, cfg = sampler.index, sampler.cfg
    counters = HonestCounters(ix.n_tables, ix.n_families)
    n_fit, n_qry, able, kinds, tasks = [], [], [], dict.fromkeys(_KINDS, 0), {}
    plan_seconds, voided, batches = 0.0, 0, 0
    for step in range(steps):
        started = time.perf_counter()
        specs = sampler.plan_step(seed, step)
        plan_seconds += time.perf_counter() - started
        counters.observe_plan(specs)
        counters.commit(step, specs)
        batches += len(micro_batches(specs))
        for spec in specs:
            kinds[spec.kind] += 1
            tasks[spec.task] = tasks.get(spec.task, 0) + 1
            voided += spec.stratum != "scm_prior" and spec.kind == "scm_prior"
            n_fit.append(spec.n_fit)
            n_qry.append(spec.n_qry)
            able.append(spec.table_id < 0 or int(spec.split_ratio * int(ix.n_rows[spec.table_id])) >= 256)
    fit, can = np.asarray(n_fit), np.asarray(able)
    total = float(fit.size)
    return {
        "steps": steps, "batch_size": cfg.batch_size, "seed": seed,
        "quota": {s.name: s.quota for s in sampler.strata} | {"scm_prior": len(sampler.prior_tasks)},
        "families": {s.name: s.size for s in sampler.strata},
        "eligible_tables": int(ix.eligible.size), "eligible_families": ix.n_families,
        "tables_touched": int((counters.ledger.table_visits > 0).sum()),
        "families_touched": int((counters.ledger.family_visits > 0).sum()),
        "repeated_tables": counters.count["repeated_tables"],
        "repeated_families": counters.count["repeated_families"],
        "voided_slots": int(voided),
        "exposure": {k: round(v, 4) for k, v in counters.ledger.exposure(sampler.family_home).items()},
        "max_table_visits": int(counters.ledger.table_visits.max()),
        "n_fit_share_ge_256": round(float((fit >= 256).mean()), 4),
        "n_fit_share_ge_256_where_table_allows": round(float((fit[can] >= 256).mean()), 4),
        "n_fit_share_ge_384": round(float((fit >= 384).mean()), 4),
        "n_fit_median": float(np.median(fit)), "n_fit_mean": round(float(fit.mean()), 1),
        "n_qry_median": float(np.median(n_qry)),
        "kind_share": {k: round(v / total, 4) for k, v in kinds.items()},
        "task_share": {k: round(v / total, 4) for k, v in sorted(tasks.items())},
        "micro_batches_per_step": round(batches / steps, 2),
        "plan_microseconds_per_episode": round(plan_seconds / total * 1e6, 2),
    }
