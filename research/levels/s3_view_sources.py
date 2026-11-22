"""Sources of an S3 view other than a stored table as it stands (plan M3, stage C2).

``swap_target`` turns a non-constant feature of a real table into the target, with the
semantics of ``s3_fast_pipeline._apply_target_swap``: two distinct values make a binary task,
3 to 12 a multiclass task, more a regression task, and the swapped column leaves the
features. The sampler asks for a target kind; a column of that kind is preferred, another kind
is accepted only when its output width fits the planned bucket, and nothing is invented when
no column qualifies (the caller retries the slot).

``scm_prior_table`` is the synthetic prior of ``s3_fast_pipeline.synthesize_scm_prior_table``
(1-3 regimes of low-rank Gaussian factors, nonlinear parent edges on 45 % of the columns, 25 %
of the columns quantised to 2-8 bins, a planted nonlinear target) with rows, columns, task and
class count fixed by the spec instead of drawn, and with the per-column loop replaced by one
pass per depth level of the causal graph. It is a pure function of the view seed.
"""
from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np

from .s3_sampler_index import TASKS, hash_perm, key64

__all__ = ["SwapTarget", "scm_prior_table", "swap_target"]


class SwapTarget(NamedTuple):
    column: int  # source column that became the target
    task: str  # the task the column actually supports
    n_classes: int  # 0 regression, 2 binary, K multiclass
    y: np.ndarray  # float32 (n,): value, or class id
    labels: np.ndarray | None  # int64 class ids for a classification target
    valid: np.ndarray | None  # rows with a finite target; None when every row has one


def swap_target(x: np.ndarray, spec: Any, max_distinct: int = 12, probe_cols: int = 16, probe_rows: int = 256,
                max_tries: int = 4) -> SwapTarget | str:
    """Pick the swapped target column; a string is the reason no column qualifies.

    Up to ``probe_cols`` columns in hashed order are classified on an evenly spaced sample of
    ``probe_rows`` rows (one small sort), then at most ``max_tries`` of them are verified on
    the full column. Only the verified kind is ever reported.
    """
    n, p = x.shape
    if p < 2 or spec.task not in TASKS:
        return "no_swap_column"
    order = hash_perm(key64("swap", spec.view_seed), p)[:probe_cols]
    probe = np.sort(x[:: max(1, n // probe_rows)][:probe_rows][:, order], axis=0)
    fin = np.isfinite(probe)
    distinct = (fin[1:] & (probe[1:] != probe[:-1])).sum(axis=0) + fin[0]
    want = TASKS.index(spec.task)
    cap = min(max_distinct, max(int(spec.n_classes), 2)) if want == 2 else 0  # multiclass must fit the planned width
    kind = np.where(distinct <= 1, -1, np.where(distinct == 2, 1, np.where(distinct <= max_distinct, 2, 0)))
    allowed = (kind == 0) | (kind == 1) | ((kind == 2) & (distinct <= cap))
    primary = allowed & (kind == want)
    for i in np.concatenate([np.flatnonzero(primary), np.flatnonzero(allowed & ~primary)])[:max_tries].tolist():
        col = x[:, int(order[i])]
        finite = np.isfinite(col)
        clean = bool(finite.all())
        valid = None if clean else finite
        if kind[i] == 0:  # more than max_distinct values in the sample: regression for certain
            return SwapTarget(int(order[i]), "regression", 0, col, None, valid)
        levels = np.unique(col if clean else col[finite])
        if levels.size < 2 or cap < levels.size <= max_distinct and levels.size > 2:
            continue  # constant after all, or more classes than the planned output width
        if levels.size > max_distinct:
            return SwapTarget(int(order[i]), "regression", 0, col, None, valid)
        labels = np.searchsorted(levels, col).astype(np.int64)
        if not clean:
            labels[~finite] = 0
        task = "binary" if levels.size == 2 else "multiclass"
        return SwapTarget(int(order[i]), task, int(levels.size), labels.astype(np.float32), labels, valid)
    return "no_swap_column"


def scm_prior_table(view_seed: int, n: int, p: int, task: str, n_classes: int,
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Synthetic prior table: (X (n, p) float32, y (n,) float32, int64 class ids or None)."""
    if n < 2 or p < 2 or task not in TASKS:
        raise ValueError("scm_prior_shape")
    f32 = np.float32
    rng = np.random.Generator(np.random.PCG64(key64("scm", view_seed)))
    n_regimes = int(rng.integers(1, 4))
    r_dim = int(rng.integers(2, min(8, p + 1)))
    sizes = rng.multinomial(n, np.full(n_regimes, 1.0 / n_regimes))
    latent = rng.standard_normal((n, r_dim), dtype=f32)
    loading = rng.standard_normal((n_regimes, r_dim, p), dtype=f32) * f32(0.7)
    shift = rng.standard_normal((n_regimes, p), dtype=f32) * f32(0.8)
    x = rng.standard_normal((n, p), dtype=f32) * f32(0.45)
    start = 0
    for r in range(n_regimes):  # at most three regimes; rows of a regime are one block
        stop = start + int(sizes[r])
        x[start:stop] += latent[start:stop] @ loading[r] + shift[r]
        start = stop

    # Causal graph: column j may take one nonlinear edge from earlier columns and may be quantised.
    u = rng.random((6, p))
    j = np.arange(p)
    edge = (u[0] < 0.45) & (j > 0)
    pa, pb = (u[1] * j).astype(np.intp), (u[3] * j).astype(np.intp)
    kind = (u[2] * 4).astype(np.intp)
    quantised = (u[4] < 0.25) & (j > 0)
    bins = 2 + (u[5] * 7).astype(np.intp)
    depth = np.zeros(p, dtype=np.intp)
    for _ in range(p):  # rounds = longest parent chain, not columns
        deeper = np.where(edge, 1 + np.maximum(depth[pa], np.where(kind == 3, depth[pb], 0)), 0)
        if (deeper == depth).all():
            break
        depth = deeper
    ranks = np.arange(n)[:, None]
    for level in range(int(depth.max()) + 1):  # parents are final (and quantised) before a child reads them
        cols = np.flatnonzero(depth == level)
        if level:
            for k in range(4):  # the four edge laws, each evaluated only where it applies
                ck = cols[kind[cols] == k]
                if not ck.size:
                    continue
                a = x[:, pa[ck]]
                if k == 0:
                    x[:, ck] += f32(0.6) * np.tanh(a)
                elif k == 1:
                    x[:, ck] += f32(0.35) * (a * a - f32(1.0))
                elif k == 2:
                    x[:, ck] += f32(0.5) * np.maximum(a, f32(0.0))
                else:
                    x[:, ck] += f32(0.4) * a * x[:, pb[ck]]
        qc = cols[quantised[cols]]
        if qc.size:  # equal-frequency codes: the rank form of digitize(x, quantile(x, k / bins))
            rank = np.empty((n, qc.size), dtype=np.intp)
            np.put_along_axis(rank, np.argsort(x[:, qc], axis=0), ranks, axis=0)
            x[:, qc] = np.minimum(rank * bins[qc] // max(n - 1, 1), bins[qc] - 1)

    k_drv = int(rng.integers(2, min(8, p + 1)))
    d = x[:, rng.permutation(p)[:k_drv]].astype(np.float64)
    labels = None
    if task == "regression":
        sig = d @ rng.standard_normal(k_drv) + 0.5 * np.sin(1.4 * d[:, 0]) + 0.45 * d[:, 0] * d[:, -1]
        if k_drv >= 3:
            sig += 0.4 * (d[:, 1] > 0.0) * d[:, 2]
        y = sig + rng.standard_normal(n) * (float(rng.uniform(0.15, 0.55)) * max(float(sig.std()), 1e-4))
    elif task == "binary":
        lg = d @ (1.2 * rng.standard_normal(k_drv)) + 0.7 * d[:, 0] * d[:, -1] - 0.5 * np.abs(d[:, 0])
        lg = (lg - np.median(lg)) / max(float(lg.std()), 1e-4) * float(rng.uniform(1.2, 2.8))
        labels = (rng.random(n) < 1.0 / (1.0 + np.exp(-np.clip(lg, -12.0, 12.0)))).astype(np.int64)
        y = labels
    else:
        kc = max(int(n_classes), 3)
        logits = d @ (1.1 * rng.standard_normal((k_drv, kc)))
        ca, cb = np.arange(kc) % k_drv, (np.arange(kc) + 1) % k_drv
        logits += 0.55 * d[:, ca] * d[:, cb] + 0.4 * np.tanh(d[:, ca])
        logits = (logits - logits.mean(axis=0)) / max(float(logits.std()), 1e-4) * float(rng.uniform(1.6, 3.2))
        probs = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs /= probs.sum(axis=1, keepdims=True)
        labels = np.minimum((rng.random((n, 1)) > np.cumsum(probs, axis=1)).sum(axis=1), kc - 1).astype(np.int64)
        y = labels
    if not np.isfinite(x).all():
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x, y.astype(np.float32), labels
