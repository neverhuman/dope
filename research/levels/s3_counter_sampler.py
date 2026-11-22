"""Counter-based episode sampler for the batched S3 training core (plan M3, stage C1).

A global optimizer step is planned as a pure function of ``(seed, optimizer_step)`` and the
corpus index. There is no RNG object, cursor, clock, rank or worker count in the plan, so a
restart replays the identical data stream and every process computes the same answer.

Strata: real regression / binary / multiclass tables, target-swap views of real tables and
synthetic prior tables. A family belongs to exactly one real stratum (its majority task).
Stratum ``k`` owns ``Q_k`` slots per step and walks a hashed permutation of its ``F_k``
families in epochs: slot ``q`` of step ``t`` reads stream position ``u = t * Q_k + q``,
epoch ``u // F_k``, index ``u % F_k``. Every family is therefore visited exactly once per
epoch, and the table inside a family rotates by epoch. Two local, replayable repairs keep a
step free of repeats:

* epoch boundary - the tail of an epoch that shares a step with the next epoch never holds a
  family of that epoch's head (the head depends on its own epoch only, so there is no chain);
* target swap - the swap stratum walks all swap-capable families, and a position whose family
  is used by a real slot of the same step trades places with a later position of its epoch.

Shapes are decided here (rows, kept columns, synthetic column counts, output width) so equal
bucket keys never need row padding. Values are decided by the view builder from ``view_seed``.
Only index metadata is read: no feature, no target, no query row reaches the sampler.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
import hashlib
import math
from typing import Iterable, Sequence

import numpy as np

from .s3_dynamic_augment import MAX_CLASSES, MAX_MULTILABEL
from .s3_sampler_index import (
    GENERIC_STEMS, GOLD, STRATA, TASKS, CorpusIndex, SamplerError, hash_perm, key64, mix64, unit,
)

__all__ = [
    "GENERIC_STEMS", "STRATA", "TASKS", "CorpusIndex", "CounterSampler", "EpisodeSpec", "SamplerConfig",
    "SamplerError", "draw_sizes", "micro_batches", "plan_digest", "plan_step", "sampler_for", "shard",
    "snap_down", "snap_nearest", "view_columns",
]
# Fixed positions in the per-view uniform stream; append only, never renumber.
_U_RATIO, _U_MIX, _U_FIT, _U_QRY, _U_TASK, _U_CLASSES, _U_COLS, _U_AUG, _U_NOISE, _U_DERIVED, _U_LABELS = range(11)
_MEMO_CAP = 256


def snap_down(rows: int) -> int:
    """Largest bucket-grid size <= rows (multiples of 8 below 32, of 32 to 512, of 128 above)."""
    if rows >= 512:
        return 512 + ((rows - 512) // 128) * 128
    return (rows // 32) * 32 if rows >= 32 else (rows // 8) * 8


def snap_nearest(rows: float) -> int:
    if rows >= 512:
        return 512 + int((rows - 512) / 128 + 0.5) * 128
    return int(rows / 32 + 0.5) * 32 if rows >= 32 else max(8, int(rows / 8 + 0.5) * 8)


@dataclass(frozen=True)
class SamplerConfig:
    """Plain data; ``arena`` / ``families`` let ``plan_step`` resolve the corpus by path."""

    batch_size: int = 64
    quotas: tuple[float, ...] = (0.45, 0.24, 0.09, 0.12, 0.10)  # STRATA order
    exposure_cap: float = 2.0  # per-family visit rate of any stratum vs regression
    fit_law: tuple[tuple[float, str, int, int], ...] = (
        (0.40, "uniform", 256, 512), (0.40, "log", 32, 2048), (0.20, "uniform", 2048, 4096))
    qry_lo: int = 128
    qry_hi: int = 1024
    min_fit_rows: int = 16
    min_qry_rows: int = 8
    split_ratio_lo: float = 0.45
    split_ratio_hi: float = 0.78
    max_view_cols: int = 64
    max_total_cols: int = 80
    max_noise_cols: int = 6
    max_derived_cols: int = 4
    aug_prob: float = 0.90
    multilabel_view_prob: float = 0.12
    max_classes: int = MAX_CLASSES
    swap_min_cols: int = 4
    swap_max_classes: int = 8
    swap_task_weights: tuple[float, float, float] = (0.2, 0.3, 0.5)  # TASKS order
    prior_task_weights: tuple[float, float, float] = (0.2, 0.3, 0.5)
    prior_min_cols: int = 4
    prior_max_cols: int = 35
    max_attempts: int = 4
    arena: str = ""
    families: str = ""


@dataclass(frozen=True)
class EpisodeSpec:
    """One planned episode: identity, shapes and the seed of its values."""

    step: int
    slot: int
    attempt: int
    kind: str  # real | target_swap | scm_prior
    stratum: str  # the slot's stratum; a voided or exhausted slot keeps it with kind scm_prior
    table_id: int  # arena table id, -1 for a synthetic prior table
    family_ix: int  # dense family index of the CorpusIndex, -1 for a synthetic prior table
    family_id: str
    task: str  # real: table task or multilabel; target_swap: requested target kind
    n_classes: int  # 0 regression, 2 binary, K multiclass (target_swap: upper bound), L multilabel
    n_fit: int
    n_qry: int
    p_view: int  # real columns to keep
    n_noise: int
    n_derived: int
    split_ratio: float
    view_seed: int

    @property
    def p_total(self) -> int:
        return self.p_view + self.n_noise + self.n_derived

    @property
    def c_out(self) -> int:
        if self.task in ("regression", "binary"):
            return 1
        return 8 if self.task == "multilabel" or self.n_classes <= 8 else 32

    @property
    def bucket_key(self) -> tuple[int, int, int, int]:
        return (self.c_out, -(-self.p_total // 8) * 8, self.n_fit, self.n_qry)

    @property
    def cells(self) -> int:
        """Padded cells of fit + query + synthetic rows (n_syn = 2 * n_fit)."""
        return (3 * self.n_fit + self.n_qry) * self.bucket_key[1]


@dataclass
class _Stratum:
    name: str
    sid: int
    fams: np.ndarray  # family indices walked by this stratum, ascending
    ptr: np.ndarray
    tables: np.ndarray
    quota: int = 0
    slot0: int = 0

    @property
    def size(self) -> int:
        return int(self.fams.size)

    @property
    def cross(self) -> bool:  # must also avoid the families of the real slots of its step
        return self.name == "target_swap"


class CounterSampler:
    """Index + config + memoised epoch orders. Memoisation never changes an answer."""

    def __init__(self, index: CorpusIndex, cfg: SamplerConfig) -> None:
        if (cfg.batch_size < 1 or len(cfg.quotas) != len(STRATA) or min(cfg.quotas) < 0.0 or sum(cfg.quotas) <= 0.0
                or cfg.min_fit_rows % 8 or cfg.min_qry_rows % 8
                or min(cfg.min_fit_rows, cfg.min_qry_rows) < 8 or not 0.0 < cfg.split_ratio_lo <= cfg.split_ratio_hi < 1.0
                or index.min_rows < cfg.min_fit_rows + cfg.min_qry_rows or index.max_classes > cfg.max_classes
                or cfg.max_total_cols < cfg.max_view_cols or cfg.max_attempts < 1):
            raise SamplerError("sampler_config")
        self.index, self.cfg = index, cfg
        counts = np.zeros((index.n_families, len(TASKS)), dtype=np.int64)
        np.add.at(counts, (index.table_family[index.eligible], index.task[index.eligible]), 1)
        # Home stratum = majority task of the family; a tie goes to the rarer task.
        self.family_home = np.argmax(counts * 4 + np.arange(len(TASKS)), axis=1).astype(np.int8)
        ptr, tables = index.members()
        self.strata = [_Stratum(name, sid, np.flatnonzero(self.family_home == sid), ptr, tables)
                       for sid, name in enumerate(TASKS)]
        sw_ptr, sw_tables = index.members(index.n_cols >= cfg.swap_min_cols)
        self.strata.append(_Stratum("target_swap", 3, np.flatnonzero(np.diff(sw_ptr) > 0), sw_ptr, sw_tables))
        self.prior_tasks = self._allocate()
        self._memo: dict[tuple[int, int, int, int], np.ndarray] = {}

    def _allocate(self) -> list[str | None]:
        """Integer quotas: largest remainder, feasibility clamp, exposure cap; overflow -> priors."""
        cfg = self.cfg
        raw = np.asarray(cfg.quotas, dtype=np.float64) / float(sum(cfg.quotas)) * cfg.batch_size
        want = np.floor(raw + 1e-9).astype(np.int64)
        for k in sorted(range(len(STRATA)), key=lambda k: (want[k] - raw[k], k))[: cfg.batch_size - int(want.sum())]:
            want[k] += 1
        for s in self.strata:  # F >= 2Q - 1 makes the epoch-boundary repair always succeed
            s.quota = min(int(want[s.sid]), s.size // 2 if s.size >= 2 else s.size)
        reg, cap = self.strata[0], Fraction(cfg.exposure_cap).limit_denominator(1000)
        if reg.quota:  # the cap is relative to regression; without regression slots there is no reference
            for s in self.strata[1:]:
                s.quota = min(s.quota, (cap.numerator * reg.quota * s.size) // (cap.denominator * reg.size))
        hints: list[str | None] = []
        slot = 0
        for s in self.strata:
            s.slot0, slot = slot, slot + s.quota
            hints += [s.name if s.name in TASKS else None] * (int(want[s.sid]) - s.quota)
        self.prior_slot0 = slot
        return hints + [None] * (cfg.batch_size - slot - len(hints))

    # -- epoch orders -------------------------------------------------------------------
    def _keep(self, key: tuple[int, int, int, int], order: np.ndarray) -> np.ndarray:
        if len(self._memo) >= _MEMO_CAP:
            self._memo.pop(next(iter(self._memo)))
        self._memo[key] = order
        return order

    def _real_stream(self, seed: int, step0: int, step1: int) -> np.ndarray:
        """``step * n_families + family`` for every real slot of steps [step0, step1]."""
        parts = [np.zeros(0, dtype=np.int64)]
        for s in self.strata:
            if s.quota and not s.cross:
                fams = self.stream(seed, s.name, step0 * s.quota, (step1 + 1) * s.quota)
                parts.append((step0 + np.arange(fams.size) // s.quota) * self.index.n_families + fams)
        return np.concatenate(parts)

    def _arranged(self, seed: int, s: _Stratum, epoch: int) -> np.ndarray:
        """Hashed permutation whose head (shared step with the previous epoch) is conflict-free."""
        key = (seed, s.sid, epoch, 0)
        if key in self._memo:
            return self._memo[key]
        order = s.fams[hash_perm(key64("perm", seed, s.sid, epoch), s.size)]
        head = (-(epoch * s.size)) % s.quota
        if head and s.cross:
            step = (epoch * s.size) // s.quota
            taken = set((self._real_stream(seed, step, step) - step * self.index.n_families).tolist())
            j = head
            for i in range(head):
                if int(order[i]) in taken:
                    while j < s.size and int(order[j]) in taken:
                        j += 1
                    if j < s.size:
                        order[i], order[j] = order[j], order[i]
                        j += 1
                    else:
                        order[i] = -1  # voided: the slot becomes a synthetic prior table
        return self._keep(key, order)

    def _order(self, seed: int, s: _Stratum, epoch: int) -> np.ndarray:
        """Final family order of one epoch; -1 marks a voided position (degenerate corpora only)."""
        key = (seed, s.sid, epoch, 1)
        if key in self._memo:
            return self._memo[key]
        size, quota, nf = s.size, s.quota, self.index.n_families
        order = self._arranged(seed, s, epoch).copy()
        head, tail = (-(epoch * size)) % quota, ((epoch + 1) * size) % quota
        nxt = set(self._arranged(seed, s, epoch + 1)[: quota - tail].tolist()) if tail else set()
        lo, bad, taken = size - tail, np.zeros(size, dtype=bool), frozenset()
        if s.cross:
            keys = self._real_stream(seed, (epoch * size) // quota, ((epoch + 1) * size - 1) // quota)
            taken = frozenset(keys.tolist())
            bad = np.isin((epoch * size + np.arange(size)) // quota * nf + order, keys)
        if tail:
            bad[lo:] |= np.isin(order[lo:], list(nxt))
        bad[:head] = False
        bad &= order >= 0

        def clash(pos: int, fam: int) -> bool:
            return (pos >= lo and fam in nxt) or ((epoch * size + pos) // quota * nf + fam) in taken

        span = size - head
        for i in np.flatnonzero(bad).tolist():
            fam_i = int(order[i])
            if not clash(i, fam_i):
                continue  # already repaired as the partner of an earlier position
            for d in range(1, span):
                j = head + (i - head + d) % span
                fam_j = int(order[j])
                if fam_j >= 0 and not clash(i, fam_j) and not clash(j, fam_i):
                    order[i], order[j] = fam_j, fam_i
                    break
            else:
                order[i] = -1
        return self._keep(key, order)

    def stream(self, seed: int, stratum: str, u0: int, u1: int) -> np.ndarray:
        """Families at stream positions [u0, u1) of a stratum (position = step * quota + q)."""
        s = self.strata[STRATA.index(stratum)]
        parts = [np.zeros(0, dtype=np.int64)]
        for epoch in range(u0 // s.size, (u1 - 1) // s.size + 1 if u1 > u0 else 0):
            base = epoch * s.size
            parts.append(self._order(seed, s, epoch)[max(u0 - base, 0): min(u1 - base, s.size)])
        return np.concatenate(parts)

    # -- specs --------------------------------------------------------------------------
    def _spec(self, seed: int, step: int, slot: int, attempt: int, stratum: str, table: int,
              hint: str | None) -> EpisodeSpec:
        cfg, ix = self.cfg, self.index
        vs = key64("view", seed, step, slot, attempt)
        ratio = round(cfg.split_ratio_lo + (cfg.split_ratio_hi - cfg.split_ratio_lo) * unit(vs, _U_RATIO), 6)
        if table < 0:
            kind, fam, fam_id = "scm_prior", -1, f"scm:{vs:016x}"
            task = hint or _pick(cfg.prior_task_weights, unit(vs, _U_TASK))
            n_classes = (0, 2, 3 + int(unit(vs, _U_CLASSES) * 8))[TASKS.index(task)]
            p_view = cfg.prior_min_cols + int(unit(vs, _U_COLS) * (cfg.prior_max_cols - cfg.prior_min_cols + 1))
            fit_supply = qry_supply = 1 << 30
        else:
            fam = int(ix.table_family[table])
            fam_id, rows, cols = ix.family_ids[fam], int(ix.n_rows[table]), int(ix.n_cols[table])
            if stratum == "target_swap":
                kind, cols = "target_swap", cols - 1
                task = _pick(cfg.swap_task_weights, unit(vs, _U_TASK))
                n_classes = (0, 2, cfg.swap_max_classes)[TASKS.index(task)]
            else:
                kind, task, n_classes = "real", TASKS[int(ix.task[table])], int(ix.n_classes[table])
            p_view = min(cols, cfg.max_view_cols)
            fit_rows = min(max(int(ratio * rows), cfg.min_fit_rows), rows - cfg.min_qry_rows)
            shrink = max(attempt - 1, 0)  # retry 1 redraws at full supply, later retries ask for half each
            fit_supply = max(cfg.min_fit_rows, fit_rows >> shrink)
            qry_supply = max(cfg.min_qry_rows, (rows - fit_rows) >> shrink)
        n_fit, n_qry = draw_sizes(cfg, unit(vs, _U_MIX), unit(vs, _U_FIT), unit(vs, _U_QRY), fit_supply, qry_supply)
        room, n_noise, n_derived = cfg.max_total_cols - p_view, 0, 0
        if room >= 1 and unit(vs, _U_AUG) < cfg.aug_prob:
            n_noise = 1 + int(unit(vs, _U_NOISE) * min(cfg.max_noise_cols, room))
            if room > n_noise and p_view >= 2:
                n_derived = int(unit(vs, _U_DERIVED) * (min(cfg.max_derived_cols, room - n_noise) + 1))
        if (task == "regression" and kind != "target_swap" and p_view + n_noise + n_derived >= 4
                and unit(vs, _U_LABELS) < cfg.multilabel_view_prob):
            most = min(MAX_MULTILABEL, max(3, p_view + n_derived))
            task, n_classes = "multilabel", 2 + int(unit(vs, _U_CLASSES) * (most - 1))
        if table < 0:
            ratio = round(n_fit / (n_fit + n_qry), 6)  # a prior table is synthesised at n_fit + n_qry rows
        return EpisodeSpec(step, slot, attempt, kind, stratum, table, fam, fam_id, task, n_classes,
                           n_fit, n_qry, p_view, n_noise, n_derived, ratio, vs)

    def plan_step(self, seed: int, step: int) -> list[EpisodeSpec]:
        if type(seed) is not int or type(step) is not int or seed < 0 or step < 0:
            raise SamplerError("plan_arguments")
        specs: list[EpisodeSpec] = []
        for s in self.strata:
            for q in range(s.quota):
                epoch, i = divmod(step * s.quota + q, s.size)
                fam = int(self._order(seed, s, epoch)[i])
                table = -1
                if fam >= 0:  # the table inside a family rotates by epoch
                    lo, n = int(s.ptr[fam]), int(s.ptr[fam + 1] - s.ptr[fam])
                    table = int(s.tables[lo + (epoch + mix64(seed * GOLD + s.sid * 0x10001 + fam)) % n])
                specs.append(self._spec(seed, step, s.slot0 + q, 0, s.name, table, None))
        for j, hint in enumerate(self.prior_tasks):
            specs.append(self._spec(seed, step, self.prior_slot0 + j, 0, "scm_prior", -1, hint))
        return specs

    def retry(self, seed: int, spec: EpisodeSpec) -> EpisodeSpec:
        """Next attempt for a slot whose view could not be built; labels are never fabricated.

        Same table and a new seed; from the second retry on, half the row supply per attempt. When
        the attempts are used up the slot becomes a synthetic prior table of the same task, which
        can always be built.
        """
        attempt = spec.attempt + 1
        if spec.kind == "scm_prior" or attempt >= self.cfg.max_attempts:
            hint = "regression" if spec.task == "multilabel" else spec.task
            return self._spec(seed, spec.step, spec.slot, attempt, spec.stratum, -1, hint)
        return self._spec(seed, spec.step, spec.slot, attempt, spec.stratum, spec.table_id, None)


def _pick(weights: Sequence[float], u: float) -> str:
    acc, total = 0.0, float(sum(weights))
    for task, weight in zip(TASKS, weights):
        acc += weight / total
        if u < acc:
            return task
    return TASKS[-1]


def draw_sizes(cfg: SamplerConfig, u_mix: float, u_fit: float, u_qry: float,
               fit_supply: int, qry_supply: int) -> tuple[int, int]:
    """Eval-matched size law, capped by the rows each half can supply, on the bucket grid."""
    acc = 0.0
    for weight, kind, lo, hi in cfg.fit_law:
        acc += weight
        if u_mix < acc:
            break
    raw = math.exp(math.log(lo) + u_fit * (math.log(hi) - math.log(lo))) if kind == "log" else lo + u_fit * (hi - lo)
    n_qry = snap_nearest(cfg.qry_lo + u_qry * (cfg.qry_hi - cfg.qry_lo))
    return min(snap_nearest(raw), snap_down(fit_supply)), min(n_qry, snap_down(qry_supply))


@lru_cache(maxsize=4)
def sampler_for(cfg: SamplerConfig) -> CounterSampler:
    """Sampler of a config that names its arena; cached per config value (an arena is written once)."""
    if not cfg.arena:
        raise SamplerError("sampler_config_arena")
    index = CorpusIndex.load(cfg.arena, cfg.families or None, min_rows=cfg.min_fit_rows + cfg.min_qry_rows,
                             max_classes=cfg.max_classes)
    return CounterSampler(index, cfg)


def plan_step(seed: int, optimizer_step: int, cfg: SamplerConfig | CounterSampler) -> list[EpisodeSpec]:
    """The global step plan: ``cfg.batch_size`` specs in slot order. Pure in its arguments."""
    sampler = cfg if isinstance(cfg, CounterSampler) else sampler_for(cfg)
    return sampler.plan_step(seed, optimizer_step)


def shard(specs: Sequence[EpisodeSpec], worker: int, n_workers: int) -> list[EpisodeSpec]:
    """Worker's share of a step: longest-processing-time deal on padded cells, slot order.

    The partition depends only on the plan, so the union over workers is the plan for any
    ``n_workers`` and the loads differ by at most one episode.
    """
    if type(worker) is not int or type(n_workers) is not int or not 0 <= worker < n_workers:
        raise SamplerError("shard_worker")
    loads, mine = [0] * n_workers, []
    for spec in sorted(specs, key=lambda s: (-s.cells, s.slot)):
        w = loads.index(min(loads))
        loads[w] += spec.cells
        if w == worker:
            mine.append(spec)
    return sorted(mine, key=lambda s: s.slot)


def micro_batches(specs: Iterable[EpisodeSpec], cell_budget: int = 524288,
                  row_slack: float = 0.0) -> list[list[EpisodeSpec]]:
    """Ordered micro-batches, each of one output width C and one padded column count ceil8(P).

    With ``row_slack == 0`` a micro-batch is one exact bucket (C, ceil8(P), n_fit, n_qry) and needs
    no row padding. A positive slack also merges neighbouring buckets while the padded cells stay
    within ``1 + row_slack`` of the true cells. Either way B * padded cells <= cell_budget, except
    that a single over-budget episode still runs alone.
    """
    out: list[list[EpisodeSpec]] = []
    true = qry = 0
    for spec in sorted(specs, key=lambda s: (s.bucket_key, s.slot)):  # n_fit ascends inside (C, P)
        if out and out[-1][0].bucket_key[:2] == spec.bucket_key[:2]:
            rows = 3 * spec.n_fit + max(qry, spec.n_qry)
            padded = (len(out[-1]) + 1) * rows * spec.bucket_key[1]
            if padded <= cell_budget and padded <= (true + spec.cells) * (1.0 + row_slack):
                out[-1].append(spec)
                true, qry = true + spec.cells, max(qry, spec.n_qry)
                continue
        out.append([spec])
        true, qry = spec.cells, spec.n_qry
    return out


def view_columns(view_seed: int, n_candidates: int, p_view: int) -> np.ndarray:
    """The seeded column subset of a view: ``p_view`` of ``n_candidates`` positions, ascending."""
    if p_view >= n_candidates:
        return np.arange(n_candidates, dtype=np.int64)
    return np.sort(hash_perm(key64("cols", view_seed), n_candidates)[:p_view]).astype(np.int64)


def plan_digest(specs: Iterable[EpisodeSpec]) -> str:
    """Stable hex digest of a list of specs, for replay checks and checkpoint identity."""
    h = hashlib.blake2b(digest_size=16, person=b"dope-s3-plan/v1")
    for s in specs:
        h.update(repr((s.step, s.slot, s.attempt, s.kind, s.stratum, s.table_id, s.family_ix, s.family_id, s.task,
                       s.n_classes, s.n_fit, s.n_qry, s.p_view, s.n_noise, s.n_derived, s.split_ratio,
                       s.view_seed)).encode("utf-8"))
    return h.hexdigest()
