"""Row selection of an S3 view (plan M3, stage C2): Rule A-1 on the view the model sees.

Duplicate groups are recomputed on the final real-feature view, after the column subset and
the target are fixed: every row is hashed over its selected columns with values rounded to
1e-5 (and, when the caller asks for the whole-row key, its target). Rows of one group share
one random key (the key of the group's first row), so a group never straddles the split and
a large group is no likelier to land in FIT than a single row. The corpus-level ``dup`` ids
cover all stored columns and cannot give this guarantee once columns are dropped or a feature
becomes the target.

Each half is then subsampled in a hashed order. For a classification task the FIT half keeps
the class priors of its pool: proportional allocation by largest remainder, so every class
count is the floor or the ceiling of its quota, with the remainder seats going first to
classes that would otherwise vanish. Nothing here draws from a numpy RNG: all choices are the
order of splitmix64 keys, identical in every process and numpy version. Query rows are never
read except for their own hash and, for classification, to drop rows whose class is absent
from the FIT half (they cannot be scored and are never relabelled).
"""
from __future__ import annotations

import hashlib
import struct
from typing import Any, NamedTuple

import numpy as np

from .s3_sampler_index import GOLD, M64

__all__ = ["RowPlan", "class_allocation", "group_split", "plan_rows", "row_hashes", "row_keys", "signatures"]

_MIX_A, _MIX_B = np.uint64(0xBF58476D1CE4E5B9), np.uint64(0x94D049BB133111EB)
_S30, _S27, _S31 = np.uint64(30), np.uint64(27), np.uint64(31)
_KINDS = ("real", "target_swap", "scm_prior")
_ROUND = np.float32(1e5)
_BASE = np.arange(1, 8193, dtype=np.uint64) * np.uint64(GOLD)  # grows on demand; a cache, never state


def row_keys(key: int, n: int) -> np.ndarray:
    """``n`` distinct 64-bit keys; their sort order is ``s3_sampler_index.hash_perm(key, n)``."""
    global _BASE
    if n > _BASE.size:
        _BASE = np.arange(1, max(n, 2 * _BASE.size) + 1, dtype=np.uint64) * np.uint64(GOLD)
    x = _BASE[:n] + np.uint64(key & M64)
    x ^= x >> _S30
    x *= _MIX_A
    x ^= x >> _S27
    x *= _MIX_B
    x ^= x >> _S31
    return x


_WEIGHTS = row_keys(0x5EEDC0DE5EEDC0DE, 1024) | np.uint64(1)  # odd multipliers of the row hash


def row_hashes(view: np.ndarray, target: np.ndarray | None = None) -> np.ndarray:
    """64-bit hash of every row of ``view`` (n, p) float32, with its float target when given.

    Values are rounded to 1e-5; -0.0 and every NaN are canonical. Equal rows always hash equal.
    A collision can only merge two groups, which keeps the split valid, so no stronger mixing
    is needed: the rounded float32 codes are read as 64-bit words and combined with odd weights.
    """
    n, p = view.shape
    used = p + (target is not None)
    words = (used + 1) // 2
    if words > _WEIGHTS.size:
        raise ValueError("view_too_wide")
    r = np.empty((n, 2 * words), dtype=np.float32)
    if 2 * words != used:
        r[:, used] = 0.0
    with np.errstate(over="ignore", invalid="ignore"):  # a huge value rounds to inf, still a valid code
        np.multiply(view, _ROUND, out=r[:, :p])
        if target is not None:
            np.multiply(target, _ROUND, out=r[:, p])
    np.rint(r, out=r)
    r += np.float32(0.0)
    flat = r.ravel()
    if np.isnan(np.dot(flat, flat)):
        r[np.isnan(r)] = np.float32(np.nan)
    return np.einsum("ij,j->i", r.view(np.uint64), _WEIGHTS[:words])


def group_split(h: np.ndarray, key: int, n_fit: int, n_qry: int, ratio: float,
                valid: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, int] | None:
    """Group-respecting random split into (FIT pool, QUERY pool, duplicate rows); ``None`` when no cut fits.

    The cut starts at ``int(ratio * rows)`` and moves to the nearest group boundary that leaves
    ``n_fit`` rows before and ``n_qry`` after (a tie moves forward), so a group that straddles
    the nominal cut goes to the side holding most of it and its FIT share stays near the ratio.
    """
    n = h.size
    keys = row_keys(key, n)
    hs = np.sort(h)
    dup = int(np.count_nonzero(hs[1:] == hs[:-1]))
    if dup:
        by_hash = np.argsort(h, kind="stable")  # inside a group: ascending row index
        hs = h[by_hash]
        first = np.empty(n, dtype=bool)
        first[0] = True
        np.not_equal(hs[1:], hs[:-1], out=first[1:])
        group_key = np.empty(n, dtype=np.uint64)  # key of the group's first row
        group_key[by_hash] = keys[by_hash[first]][np.cumsum(first) - 1]
        order = np.argsort(group_key, kind="stable")
    else:
        order = np.argsort(keys)  # distinct keys: one answer under any sort algorithm
    if valid is not None:
        order = order[valid[order]]
    m = order.size
    if n_fit < 1 or n_qry < 1 or m < n_fit + n_qry:
        return None
    cut = min(max(int(ratio * m), n_fit), m - n_qry)
    if dup:
        gs = group_key[order]
        boundary = gs[n_fit:m - n_qry + 1] != gs[n_fit - 1:m - n_qry]  # boundary[i]: a cut at n_fit + i splits no group
        at = np.flatnonzero(boundary)
        if not at.size:
            return None
        cut = n_fit + int(at[np.argmin(np.abs(at - (cut - n_fit)) * 2 + (at < cut - n_fit))])  # nearest; a tie goes forward
    return order[:cut], order[cut:], dup


def class_allocation(counts: np.ndarray, n: int) -> np.ndarray:
    """Rows per class for a prior-preserving subsample of ``n`` rows: largest remainder.

    Exact integer arithmetic. Every result is the floor or the ceiling of ``n * count / total``,
    so each class prior moves by less than ``1 / n``. Remainder seats go first to present classes
    whose floor is zero (largest quota first), then by largest remainder; ties to the lower class.
    """
    counts = np.asarray(counts, dtype=np.int64)
    total = int(counts.sum())
    if not 0 <= n <= total:
        raise ValueError("allocation_size")
    if n == total:
        return counts.copy()
    scaled = counts * n
    alloc = scaled // total
    seats = n - int(alloc.sum())
    if seats:
        priority = (scaled - alloc * total) + total * ((alloc == 0) & (counts > 0))
        alloc[np.argsort(-priority, kind="stable")[:seats]] += 1
    return alloc


class RowPlan(NamedTuple):
    fit: np.ndarray  # source row indices of the FIT half, hashed order
    qry: np.ndarray  # source row indices of the QUERY half, hashed order
    fit_pool: int  # rows on the FIT side of the split before subsampling
    qry_pool: int
    dup_rows: int  # rows of the view that repeat an earlier row (rows minus groups)
    pool_counts: np.ndarray | None  # class counts of the FIT pool (classification)
    fit_counts: np.ndarray | None  # class counts of the FIT half
    qry_dropped: int  # QUERY pool rows of classes absent from the FIT half


def plan_rows(h: np.ndarray, view_seed: int, n_fit: int, n_qry: int, ratio: float,
              labels: np.ndarray | None = None, valid: np.ndarray | None = None) -> RowPlan | str:
    """Split and subsample; a string is the reason the view cannot be built (caller retries).

    ``labels`` (int class id per row) switches on the prior-preserving FIT subsample and the
    two-class requirement. ``valid`` masks rows whose target is unusable.
    """
    pools = group_split(h, view_seed ^ 0xA5A5A5A5A5A5A5A5, n_fit, n_qry, ratio, valid)
    if pools is None:
        return "split_groups"
    fit, qry, dup = pools
    if dup:  # group order is not a row-level shuffle: subsample in an independent hashed order
        sub = row_keys(view_seed ^ 0x3C3C3C3C3C3C3C3C, h.size)
        fit, qry = fit[np.argsort(sub[fit])], qry[np.argsort(sub[qry])]
    if labels is None:
        return RowPlan(fit[:n_fit], qry[:n_qry], fit.size, qry.size, dup, None, None, 0)
    cls = labels[fit]
    counts = np.bincount(cls)
    alloc = class_allocation(counts, n_fit)
    if int((alloc > 0).sum()) < 2:
        return "one_class_fit"
    by_class = np.argsort(cls.astype(np.int16) if counts.size <= 32767 else cls, kind="stable")
    cs = cls[by_class]
    rank = np.arange(fit.size) - (np.cumsum(counts) - counts)[cs]
    keep = np.zeros(fit.size, dtype=bool)
    keep[by_class[rank < alloc[cs]]] = True
    qcls = labels[qry]
    seen = np.zeros(max(counts.size, int(qcls.max()) + 1), dtype=bool)
    seen[:counts.size] = alloc > 0
    scored = seen[qcls]
    dropped = int(qry.size - scored.sum())
    if dropped:
        qry = qry[scored]
    if qry.size < n_qry:
        return "query_rows"
    return RowPlan(fit[keep], qry[:n_qry], fit.size, qry.size + dropped, dup, counts, alloc, dropped)


def _rows(a: Any) -> bytes:
    arr = np.array(a, dtype="<i8").ravel()
    arr.sort()
    return struct.pack("<q", arr.size) + arr.tobytes()


def signatures(spec: Any, cols: Any, fit_rows: Any, qry_rows: Any, target_col: int = -1) -> tuple[int, int]:
    """(view_sig, split_sig): equal to ``s3_honest_counters.view_hash`` / ``split_hash``, rows sorted once."""
    rows = _rows(fit_rows) + _rows(qry_rows)
    head = struct.pack("<qqqQ", _KINDS.index(spec.kind), spec.table_id, target_col, spec.view_seed)
    view = hashlib.blake2b(head + _rows(cols) + rows, digest_size=8, person=b"dope-s3-view/v1").digest()
    head = struct.pack("<qQ", spec.table_id, spec.view_seed if spec.table_id < 0 else 0)
    split = hashlib.blake2b(head + rows, digest_size=8, person=b"dope-s3-split/v1").digest()
    return int.from_bytes(view, "little"), int.from_bytes(split, "little")
